from datetime import date, timedelta
from functools import lru_cache
from typing import Optional, Any
import base64
import os
import secrets
import urllib.parse

import pandas as pd
import requests
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, HTMLResponse
from pybaseball import (
    chadwick_register,
    playerid_lookup,
    statcast_batter,
    statcast_batter_exitvelo_barrels,
    statcast_pitcher,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

YAHOO_CLIENT_ID = os.getenv("YAHOO_CLIENT_ID", "").strip()
YAHOO_CLIENT_SECRET = os.getenv("YAHOO_CLIENT_SECRET", "").strip()
YAHOO_REDIRECT_URI = os.getenv("YAHOO_REDIRECT_URI", "http://127.0.0.1:8000/auth/yahoo/callback").strip()
FBS_FRONTEND_URL = os.getenv("FBS_FRONTEND_URL", "").strip()
FBS_LIGHT_MODE = os.getenv("FBS_LIGHT_MODE", "1" if os.getenv("RENDER") else "0").strip() == "1"

# Local beta storage (single-process). For production, move to DB/session store.
_yahoo_auth_state: Optional[str] = None
_yahoo_token: dict[str, Any] = {}

# --- LOAD SAVANT DATA ---
CURRENT_SEASON = date.today().year
LAST_SEASON = CURRENT_SEASON - 1
MLB_SEASON = CURRENT_SEASON
SAVANT_SEASON = CURRENT_SEASON

print(f"Loading Statcast data for {CURRENT_SEASON}...")
if FBS_LIGHT_MODE:
    print("FBS_LIGHT_MODE enabled: skipping startup Savant preload to reduce RAM.")
    df_savant = pd.DataFrame()
else:
    try:
        df_savant = statcast_batter_exitvelo_barrels(CURRENT_SEASON)
        # Preseason/early-season responses may be near-empty; fallback to last season.
        if df_savant.empty or len(df_savant) < 50:
            raise ValueError(f"Savant {CURRENT_SEASON} dataset too small ({len(df_savant)} rows)")
        print(f"Statcast data loaded for {CURRENT_SEASON}.")
    except Exception as e:
        print(f"Statcast current-season load error: {e}")
        try:
            SAVANT_SEASON = LAST_SEASON
            df_savant = statcast_batter_exitvelo_barrels(SAVANT_SEASON)
            print(f"Statcast fallback loaded for {SAVANT_SEASON}.")
        except Exception as e2:
            print(f"Statcast fallback error: {e2}")
            df_savant = pd.DataFrame()


@lru_cache(maxsize=2)
def _savant_batter_season_df(season: int):
    if FBS_LIGHT_MODE:
        return pd.DataFrame()
    try:
        df = statcast_batter_exitvelo_barrels(season)
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    return pd.DataFrame()


@lru_cache(maxsize=256)
def _get_player_details_by_id(mlb_id: int):
    url = f"https://statsapi.mlb.com/api/v1/people/{mlb_id}"
    params = {"hydrate": "currentTeam,primaryPosition"}
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    people = response.json().get("people", [])
    return people[0] if people else None


def _lookup_candidates_from_query(nombre: str):
    parts = [p for p in nombre.split() if p]
    if not parts:
        return pd.DataFrame()

    try:
        frames = []
        if len(parts) == 1:
            # Most users type just a last name (e.g., "Judge").
            frames.append(playerid_lookup(parts[0], ""))
        else:
            first = parts[0]
            last = " ".join(parts[1:])
            frames.append(playerid_lookup(last, first))
            # Critical fallback: allow nickname mismatch in first name (Peter -> Pete).
            frames.append(playerid_lookup(last, ""))
            # Fallback in case user enters "Last First".
            frames.append(playerid_lookup(first, last))

        if not frames:
            return pd.DataFrame()

        valid = [f for f in frames if isinstance(f, pd.DataFrame) and not f.empty]
        if not valid:
            return pd.DataFrame()
        out = pd.concat(valid, ignore_index=True)
        if "key_mlbam" in out.columns:
            out = out.drop_duplicates(subset=["key_mlbam"])
        return out
    except Exception:
        return pd.DataFrame()


def _normalize_name_text(text: str):
    return (
        (text or "")
        .lower()
        .replace("-", " ")
        .replace(".", " ")
        .replace("'", "")
    )


@lru_cache(maxsize=1)
def _name_index():
    df = chadwick_register()
    df = df[df["key_mlbam"].notna()].copy()
    df["key_mlbam"] = pd.to_numeric(df["key_mlbam"], errors="coerce")
    df = df[df["key_mlbam"] > 0].copy()
    df["name_first"] = df["name_first"].fillna("")
    df["name_last"] = df["name_last"].fillna("")
    df["full_name"] = (df["name_first"] + " " + df["name_last"]).str.strip()
    df["full_name_norm"] = df["full_name"].map(_normalize_name_text)
    df["name_first_norm"] = df["name_first"].map(_normalize_name_text)
    df["name_last_norm"] = df["name_last"].map(_normalize_name_text)
    df["mlb_played_last"] = pd.to_numeric(df["mlb_played_last"], errors="coerce").fillna(0)
    return df


def _autocomplete_candidates(query: str, limit: int = 6):
    q = _normalize_name_text(query).strip()
    if len(q) < 2:
        return []

    idx = _name_index()
    parts = [p for p in q.split() if p]
    if not parts:
        return []

    # Stricter autocomplete: score close matches and avoid broad noise.
    cand = idx.copy()
    cand["auto_score"] = 0.0
    exact_name_mask = None

    full_starts = cand["full_name_norm"].str.startswith(q)
    cand.loc[full_starts, "auto_score"] += 120

    if len(parts) >= 2:
        first = parts[0]
        last = " ".join(parts[1:])
        exact_full = (cand["name_first_norm"] == first) & (cand["name_last_norm"] == last)
        last_plus_first_prefix = cand["name_last_norm"].str.startswith(last) & cand["name_first_norm"].str.startswith(first[:2])

        cand.loc[exact_full, "auto_score"] += 200
        cand.loc[last_plus_first_prefix, "auto_score"] += 130
        exact_name_mask = exact_full
    else:
        token = parts[0]
        exact_last = cand["name_last_norm"] == token
        exact_first = cand["name_first_norm"] == token
        last_prefix = cand["name_last_norm"].str.startswith(token)
        first_prefix = cand["name_first_norm"].str.startswith(token)

        cand.loc[exact_last, "auto_score"] += 180
        cand.loc[exact_first, "auto_score"] += 120
        cand.loc[last_prefix, "auto_score"] += 90
        cand.loc[first_prefix, "auto_score"] += 65

    cand = cand[cand["auto_score"] > 0].copy()
    if exact_name_mask is not None and bool(exact_name_mask.any()):
        cand = cand[exact_name_mask.loc[cand.index]].copy()
    if cand.empty:
        return []

    # Prefer recent MLB players.
    cand["auto_score"] = cand["auto_score"] + ((cand["mlb_played_last"] - 1900) / 10.0)
    cand = cand.sort_values(by=["auto_score", "mlb_played_last"], ascending=False).drop_duplicates(subset=["key_mlbam"])
    out = []
    for _, row in cand.head(limit * 3).iterrows():
        mlb_id = int(row["key_mlbam"])
        if mlb_id <= 0:
            continue
        player = _get_player_details_by_id(mlb_id)
        if not player:
            continue
        out.append(
            {
                "id": mlb_id,
                "nombre": player.get("fullName", row["full_name"]),
                "equipo": player.get("currentTeam", {}).get("name", "N/A"),
                "posicion": player.get("primaryPosition", {}).get("abbreviation", "N/A"),
                "_active": bool(player.get("active")),
                "_team_id": player.get("currentTeam", {}).get("id"),
                "_score": float(row.get("auto_score", 0)),
            }
        )
    if not out:
        return []

    out = sorted(
        out,
        key=lambda x: (
            x.get("_active", False),
            bool(x.get("_team_id")),
            x.get("_score", 0),
        ),
        reverse=True,
    )

    cleaned = []
    for item in out:
        cleaned.append(
            {
                "id": item["id"],
                "nombre": item["nombre"],
                "equipo": item["equipo"],
                "posicion": item["posicion"],
            }
        )
        if len(cleaned) >= limit:
            break
    return cleaned


def _rank_candidates(df: pd.DataFrame, nombre: str):
    if df.empty:
        return pd.DataFrame()

    df = df[df["key_mlbam"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    parts = [p for p in nombre.split() if p]
    df["name_first_l"] = df["name_first"].fillna("").str.lower()
    df["name_last_l"] = df["name_last"].fillna("").str.lower()
    df["mlb_played_last"] = pd.to_numeric(df["mlb_played_last"], errors="coerce").fillna(0)
    df["name_score"] = 0.0

    if len(parts) == 1:
        token = parts[0]
        df.loc[df["name_last_l"] == token, "name_score"] += 100
        df.loc[df["name_first_l"] == token, "name_score"] += 60
        df.loc[df["name_last_l"].str.startswith(token), "name_score"] += 30
        df.loc[df["name_first_l"].str.startswith(token), "name_score"] += 20
    else:
        first = parts[0]
        last = " ".join(parts[1:])
        df.loc[
            (df["name_first_l"] == first) & (df["name_last_l"] == last),
            "name_score",
        ] += 250
        df.loc[
            (df["name_first_l"] == last) & (df["name_last_l"] == first),
            "name_score",
        ] += 180
        df.loc[df["name_last_l"] == last, "name_score"] += 80
        df.loc[df["name_first_l"] == first, "name_score"] += 50

    df["savant_attempts"] = -1.0
    if not df_savant.empty and "attempts" in df_savant.columns:
        savant_subset = df_savant[["player_id", "attempts"]].copy()
        savant_subset["player_id"] = pd.to_numeric(savant_subset["player_id"], errors="coerce")
        savant_subset = savant_subset.dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])
        attempts_map = savant_subset.set_index("player_id")["attempts"]
        df_ids = pd.to_numeric(df["key_mlbam"], errors="coerce")
        df["savant_attempts"] = df_ids.map(attempts_map).fillna(-1)

    # Final score: strong name match first, then actual Statcast sample, then recency.
    df["total_score"] = (
        df["name_score"]
        + (df["savant_attempts"] / 5.0)
        + ((df["mlb_played_last"] - 1900) / 10.0)
    )
    return df.sort_values(by="total_score", ascending=False)


def _parse_query_hints(raw_query: str):
    pos_aliases = {
        "rp": "RP",
        "reliever": "RP",
        "closer": "RP",
        "sp": "SP",
        "starter": "SP",
        "c": "C",
        "catcher": "C",
        "1b": "1B",
        "2b": "2B",
        "3b": "3B",
        "ss": "SS",
        "lf": "LF",
        "cf": "CF",
        "rf": "RF",
        "of": "OF",
        "dh": "DH",
    }

    parts = [p for p in raw_query.split() if p]
    cleaned = []
    prefer_pos = None
    for token in parts:
        t = token.lower()
        if t in pos_aliases and prefer_pos is None:
            prefer_pos = pos_aliases[t]
        else:
            cleaned.append(token)
    nombre = " ".join(cleaned).strip().lower()
    return nombre, prefer_pos


def _passes_hints(player: dict, prefer_pos: Optional[str], prefer_team: Optional[str]):
    if prefer_pos:
        pos = (player.get("primaryPosition", {}).get("abbreviation") or "").upper()
        desired = prefer_pos.upper()
        pos_matches = pos == desired
        if desired in {"RP", "SP"} and pos == "P":
            pos_matches = True
        if desired == "OF" and pos in {"LF", "CF", "RF"}:
            pos_matches = True
        if not pos_matches:
            return False
    if prefer_team:
        team = (player.get("currentTeam", {}).get("name") or "").lower()
        if prefer_team.lower() not in team:
            return False
    return True


def _candidate_option(player: dict):
    mlb_id = player.get("id")
    return {
        "id": mlb_id,
        "nombre": player.get("fullName", "N/A"),
        "equipo": player.get("currentTeam", {}).get("name", "N/A"),
        "posicion": player.get("primaryPosition", {}).get("abbreviation", "N/A"),
        "foto_url": (
            "https://img.mlbstatic.com/mlb-photos/image/upload/"
            "d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/"
            f"people/{mlb_id}/headshot/67/current"
        ),
    }


def buscar_jugador_mlb(query: str, prefer_pos: Optional[str] = None, prefer_team: Optional[str] = None):
    nombre, inferred_pos = _parse_query_hints(urllib.parse.unquote(query).strip())
    prefer_pos = prefer_pos or inferred_pos
    if not nombre:
        return None

    try:
        candidates = _lookup_candidates_from_query(nombre)
        ranked = _rank_candidates(candidates, nombre)
        if ranked.empty:
            return None

        options = []
        for _, row in ranked.head(12).iterrows():
            player = _get_player_details_by_id(int(row["key_mlbam"]))
            if not player:
                continue
            if _passes_hints(player, prefer_pos, prefer_team):
                options.append(player)

        if not options:
            return None

        # Disambiguate exact same-name hits (e.g., Will Smith C vs Will Smith RP).
        top_name = (options[0].get("fullName") or "").lower()
        same_name = [p for p in options if (p.get("fullName") or "").lower() == top_name]
        if len(same_name) > 1 and not prefer_pos and not prefer_team:
            return {"ambiguous": True, "options": [_candidate_option(p) for p in same_name[:6]]}

        return {"ambiguous": False, "player": options[0]}
    except Exception:
        return None


def _to_percent(val):
    if val is None or pd.isna(val):
        return "N/A"
    try:
        return f"{float(val):.1f}%"
    except Exception:
        return "N/A"


def _to_mph(val):
    if val is None or pd.isna(val):
        return "N/A"
    try:
        return f"{float(val):.1f} mph"
    except Exception:
        return "N/A"


def _to_rate(val):
    if val is None or pd.isna(val):
        return "N/A"
    try:
        n = float(val)
        if n <= 1:
            return f"{n * 100:.1f}%"
        return f"{n:.1f}%"
    except Exception:
        return "N/A"


def _to_decimal(val, digits=3):
    if val is None or val == "" or pd.isna(val):
        return "N/A"
    try:
        num = float(val)
        return f"{num:.{digits}f}"
    except Exception:
        return "N/A"


def _to_num(val, digits=1):
    if val is None or val == "" or pd.isna(val):
        return "N/A"
    try:
        num = float(val)
        return f"{num:.{digits}f}"
    except Exception:
        return "N/A"


@lru_cache(maxsize=512)
def _get_stat_line(
    mlb_id: int,
    group: str,
    stats_type: str,
    season: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    game_type: Optional[str] = None,
):
    url = f"https://statsapi.mlb.com/api/v1/people/{mlb_id}/stats"
    params = {"stats": stats_type, "group": group}
    if season is not None:
        params["season"] = season
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    if game_type:
        params["gameType"] = game_type
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    stats = response.json().get("stats", [])
    if not stats:
        return {}
    splits = stats[0].get("splits", [])
    if not splits:
        return {}
    return splits[0].get("stat", {}) or {}


@lru_cache(maxsize=256)
def _leaderboard_player_ranks(group: str, category: str, season: int, limit: int = 200):
    url = "https://statsapi.mlb.com/api/v1/stats/leaders"
    params = {
        "leaderCategories": category,
        "season": season,
        "leaderGameTypes": "R",
        "statGroup": group,
        "limit": limit,
    }
    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    data = response.json()
    league_leaders = data.get("leagueLeaders", [])
    if not league_leaders:
        return {}

    leaders = league_leaders[0].get("leaders", []) or []
    ranks = {}
    for leader in leaders:
        person = leader.get("person") or {}
        player_id = person.get("id")
        rank = leader.get("rank")
        if player_id is None or rank is None:
            continue
        try:
            ranks[int(player_id)] = int(rank)
        except Exception:
            continue
    return ranks


def _leader_tier(rank: int):
    if rank == 1:
        return "gold"
    if 2 <= rank <= 10:
        return "silver"
    if 11 <= rank <= 30:
        return "bronze"
    return None


def _yahoo_configured():
    return bool(YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET and YAHOO_REDIRECT_URI)


def _yahoo_token_expired():
    expires_at = _yahoo_token.get("expires_at")
    if not expires_at:
        return True
    try:
        return float(expires_at) <= float(pd.Timestamp.utcnow().timestamp()) + 20
    except Exception:
        return True


def _yahoo_exchange_token(grant_type: str, **kwargs):
    auth_b64 = base64.b64encode(f"{YAHOO_CLIENT_ID}:{YAHOO_CLIENT_SECRET}".encode("utf-8")).decode("utf-8")
    headers = {
        "Authorization": f"Basic {auth_b64}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"grant_type": grant_type, **kwargs}
    resp = requests.post("https://api.login.yahoo.com/oauth2/get_token", headers=headers, data=data, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _yahoo_store_token(token_payload: dict):
    access_token = token_payload.get("access_token")
    refresh_token = token_payload.get("refresh_token") or _yahoo_token.get("refresh_token")
    expires_in = token_payload.get("expires_in", 0)
    now_ts = float(pd.Timestamp.utcnow().timestamp())
    _yahoo_token.clear()
    _yahoo_token.update(
        {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "expires_in": expires_in,
            "expires_at": now_ts + float(expires_in or 0),
            "token_type": token_payload.get("token_type", "bearer"),
            "scope": token_payload.get("scope", ""),
        }
    )


def _yahoo_refresh_if_needed():
    if not _yahoo_token:
        return False
    if not _yahoo_token_expired():
        return True
    refresh_token = _yahoo_token.get("refresh_token")
    if not refresh_token:
        _yahoo_token.clear()
        return False
    try:
        new_payload = _yahoo_exchange_token("refresh_token", refresh_token=refresh_token, redirect_uri=YAHOO_REDIRECT_URI)
        _yahoo_store_token(new_payload)
        return True
    except Exception:
        _yahoo_token.clear()
        return False


def _yahoo_api_get_json(url: str):
    if not _yahoo_refresh_if_needed():
        raise RuntimeError("Yahoo token missing or expired")
    headers = {"Authorization": f"Bearer {_yahoo_token.get('access_token', '')}"}
    resp = requests.get(url, headers=headers, timeout=20)
    if resp.status_code == 401 and _yahoo_refresh_if_needed():
        headers = {"Authorization": f"Bearer {_yahoo_token.get('access_token', '')}"}
        resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _extract_leagues(node):
    out = []
    if isinstance(node, dict):
        if "league_key" in node and "name" in node:
            out.append(
                {
                    "league_key": node.get("league_key"),
                    "name": node.get("name"),
                    "url": node.get("url"),
                    "season": node.get("season"),
                }
            )
        for v in node.values():
            out.extend(_extract_leagues(v))
    elif isinstance(node, list):
        for item in node:
            out.extend(_extract_leagues(item))
    return out


def _deep_find_first(node, key: str):
    if isinstance(node, dict):
        if key in node:
            return node.get(key)
        for v in node.values():
            got = _deep_find_first(v, key)
            if got is not None:
                return got
    elif isinstance(node, list):
        for item in node:
            got = _deep_find_first(item, key)
            if got is not None:
                return got
    return None


def _extract_team_rows(node):
    rows = []
    if isinstance(node, dict):
        if "team_key" in node and "name" in node:
            rows.append(
                {
                    "team_key": node.get("team_key"),
                    "name": node.get("name"),
                    "url": node.get("url"),
                    "rank": node.get("team_standings", {}).get("rank") if isinstance(node.get("team_standings"), dict) else None,
                    "points": node.get("team_standings", {}).get("points", {}).get("total")
                    if isinstance(node.get("team_standings"), dict)
                    else None,
                }
            )
        for v in node.values():
            rows.extend(_extract_team_rows(v))
    elif isinstance(node, list):
        for item in node:
            rows.extend(_extract_team_rows(item))
    return rows


def _extract_stat_categories(node):
    out = []
    if isinstance(node, dict):
        has_id = "stat_id" in node
        has_name = "name" in node
        if has_id or has_name:
            out.append(
                {
                    "stat_id": node.get("stat_id"),
                    "name": node.get("name"),
                    "sort_order": node.get("sort_order"),
                    "display_name": node.get("display_name"),
                }
            )
        for v in node.values():
            out.extend(_extract_stat_categories(v))
    elif isinstance(node, list):
        for item in node:
            out.extend(_extract_stat_categories(item))
    return out


@lru_cache(maxsize=512)
def _leaderboard_rows(
    group: str,
    category: str,
    season: Optional[int],
    start_date: Optional[str],
    end_date: Optional[str],
    game_type: str = "R",
    limit: int = 10,
):
    url = "https://statsapi.mlb.com/api/v1/stats/leaders"
    params = {
        "leaderCategories": category,
        "leaderGameTypes": game_type,
        "statGroup": group,
        "limit": limit,
    }
    if season is not None:
        params["season"] = season
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date

    response = requests.get(url, params=params, timeout=12)
    response.raise_for_status()
    data = response.json()
    league_leaders = data.get("leagueLeaders", [])
    if not league_leaders:
        return []

    leaders = league_leaders[0].get("leaders", []) or []
    out = []
    for leader in leaders:
        person = leader.get("person") or {}
        team = leader.get("team") or {}
        out.append(
            {
                "rank": leader.get("rank"),
                "value": leader.get("value"),
                "id": person.get("id"),
                "name": person.get("fullName") or "N/A",
                "team": team.get("name") or "N/A",
            }
        )
    return out


def _waiver_candidates(
    group: str,
    metric_map: dict,
    season: Optional[int],
    start_date: Optional[str],
    end_date: Optional[str],
    game_type: str,
    limit: int = 140,
    min_rank: int = 15,
    max_rank: int = 120,
    top_n: int = 14,
):
    bucket = {}
    for label, cfg in metric_map.items():
        category = cfg["category"]
        weight = float(cfg.get("weight", 1.0))
        try:
            rows = _leaderboard_rows(
                group=group,
                category=category,
                season=season,
                start_date=start_date,
                end_date=end_date,
                game_type=game_type,
                limit=limit,
            )
        except Exception:
            rows = []

        for row in rows:
            try:
                player_id = int(row.get("id"))
                rank = int(row.get("rank"))
            except Exception:
                continue
            if rank < min_rank or rank > max_rank:
                continue
            item = bucket.setdefault(
                player_id,
                {
                    "id": player_id,
                    "name": row.get("name", "N/A"),
                    "team": row.get("team", "N/A"),
                    "score": 0.0,
                    "metrics": {},
                },
            )
            item["score"] += (limit - rank + 1) * weight
            item["metrics"][label] = {"rank": rank, "value": row.get("value", "N/A")}

    items = []
    for _, item in bucket.items():
        if len(item["metrics"]) < 2:
            continue
        metric_preview = sorted(
            item["metrics"].items(),
            key=lambda kv: kv[1]["rank"],
        )[:3]
        strengths = [f"{k} #{v['rank']} ({v['value']})" for k, v in metric_preview]
        items.append(
            {
                "id": item["id"],
                "name": item["name"],
                "team": item["team"],
                "score": round(item["score"], 1),
                "strengths": strengths,
            }
        )

    items.sort(key=lambda x: x["score"], reverse=True)

    out = []
    for cand in items[: top_n * 2]:
        try:
            player = _get_player_details_by_id(int(cand["id"]))
        except Exception:
            player = None
        if player:
            pos = (player.get("primaryPosition", {}) or {}).get("abbreviation", "N/A")
            team_name = (player.get("currentTeam", {}) or {}).get("name", cand["team"])
            cand["team"] = team_name or cand["team"]
            cand["posicion"] = pos or "N/A"
        else:
            cand["posicion"] = "N/A"
        out.append(cand)
        if len(out) >= top_n:
            break
    return out


def _build_leader_highlights(mlb_id: int, hitting_season: Optional[int], pitching_season: Optional[int]):
    hitting_map = {
        "AVG": "battingAverage",
        "OBP": "onBasePercentage",
        "SLG": "sluggingPercentage",
        "OPS": "ops",
        "HR": "homeRuns",
        "RBI": "runsBattedIn",
        "SB": "stolenBases",
    }
    pitching_map = {
        "ERA": "era",
        "WHIP": "whip",
        "IP": "inningsPitched",
        "K": "strikeouts",
        "BB": "walks",
        "SV": "saves",
        "H": "hits",
        "K/9": "strikeoutsPer9Inn",
        "BB/9": "walksPer9Inn",
    }

    result = {"hitting": {}, "pitching": {}}

    if hitting_season:
        for stat_key, category in hitting_map.items():
            try:
                rank = _leaderboard_player_ranks("hitting", category, hitting_season).get(mlb_id)
            except Exception:
                rank = None
            if rank is None:
                continue
            tier = _leader_tier(rank)
            if tier:
                result["hitting"][stat_key] = {"rank": rank, "tier": tier}

    if pitching_season:
        for stat_key, category in pitching_map.items():
            try:
                rank = _leaderboard_player_ranks("pitching", category, pitching_season).get(mlb_id)
            except Exception:
                rank = None
            if rank is None:
                continue
            tier = _leader_tier(rank)
            if tier:
                result["pitching"][stat_key] = {"rank": rank, "tier": tier}

    return result


def _resolve_period_request(
    mode: Optional[str],
    season: Optional[int],
    start_date: Optional[str],
    end_date: Optional[str],
):
    today = date.today()
    range_anchor = today if today.year == CURRENT_SEASON else date(CURRENT_SEASON, 9, 30)
    m = (mode or "season").lower()

    if m == "career":
        return {
            "mode": "career",
            "stats_type": "career",
            "season": None,
            "start_date": None,
            "end_date": None,
            "label": "Career",
            "leader_season": None,
            "game_type": "R",
        }

    if m == "today":
        d = range_anchor.isoformat()
        return {
            "mode": "today",
            "stats_type": "byDateRange",
            "season": None,
            "start_date": d,
            "end_date": d,
            "label": f"Today View ({d})",
            "leader_season": None,
            "game_type": "R",
        }

    if m in {"last7", "last14", "last30"}:
        days = {"last7": 7, "last14": 14, "last30": 30}[m]
        end = range_anchor
        start = range_anchor - timedelta(days=days - 1)
        return {
            "mode": m,
            "stats_type": "byDateRange",
            "season": None,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "label": f"Last {days} Days",
            "leader_season": None,
            "game_type": "R",
        }

    if m == "custom":
        if not start_date or not end_date:
            return {
                "mode": "custom",
                "stats_type": "byDateRange",
                "season": None,
                "start_date": None,
                "end_date": None,
                "label": "Custom Range",
                "leader_season": None,
                "error": "Custom mode requires start_date and end_date",
                "game_type": "R",
            }
        return {
            "mode": "custom",
            "stats_type": "byDateRange",
            "season": None,
            "start_date": start_date,
            "end_date": end_date,
            "label": f"{start_date} to {end_date}",
            "leader_season": None,
            "game_type": "R",
        }

    if m in {"spring2026", "spring", "springtraining"}:
        spring_start = date(2026, 2, 20).isoformat()
        spring_end = date(2026, 3, 31).isoformat()
        return {
            "mode": "spring2026",
            "stats_type": "byDateRange",
            "season": 2026,
            "start_date": spring_start,
            "end_date": spring_end,
            "label": "Spring Training 2026",
            "leader_season": None,
            "game_type": "S",
        }

    if m in {"lastseason", "previousseason", "prevseason"}:
        selected_season = LAST_SEASON
        return {
            "mode": "lastseason",
            "stats_type": "season",
            "season": selected_season,
            "start_date": None,
            "end_date": None,
            "label": f"Last Season ({selected_season})",
            "leader_season": selected_season,
            "game_type": "R",
        }

    # "season" remains an alias to current season for backward compatibility.
    selected_season = season or CURRENT_SEASON
    mode_name = "currentseason" if m in {"season", "currentseason"} else m
    return {
        "mode": mode_name,
        "stats_type": "season",
        "season": selected_season,
        "start_date": None,
        "end_date": None,
        "label": f"Current Season ({selected_season})",
        "leader_season": selected_season,
        "game_type": "R",
    }


def _hitting_stats(mlb_id: int, period_req: dict):
    game_type = period_req.get("game_type") if period_req.get("stats_type") in {"season", "byDateRange"} else None
    try:
        stat = _get_stat_line(
            mlb_id=mlb_id,
            group="hitting",
            stats_type=period_req["stats_type"],
            season=period_req.get("season"),
            start_date=period_req.get("start_date"),
            end_date=period_req.get("end_date"),
            game_type=game_type,
        )
    except Exception:
        stat = {}
    if not stat:
        return {}

    pa = float(stat.get("plateAppearances", 0) or 0)
    so = float(stat.get("strikeOuts", 0) or 0)
    bb = float(stat.get("baseOnBalls", 0) or 0)
    bb_pct = (bb / pa * 100.0) if pa > 0 else None
    k_pct = (so / pa * 100.0) if pa > 0 else None

    return {
        "PA": str(stat.get("plateAppearances", "N/A")),
        "AB": str(stat.get("atBats", "N/A")),
        "H": str(stat.get("hits", "N/A")),
        "R": str(stat.get("runs", "N/A")),
        "HR": str(stat.get("homeRuns", "N/A")),
        "RBI": str(stat.get("rbi", "N/A")),
        "BB": str(stat.get("baseOnBalls", "N/A")),
        "K": str(stat.get("strikeOuts", "N/A")),
        "SB": str(stat.get("stolenBases", "N/A")),
        "AVG": _to_decimal(stat.get("avg"), 3),
        "OBP": _to_decimal(stat.get("obp"), 3),
        "SLG": _to_decimal(stat.get("slg"), 3),
        "OPS": _to_decimal(stat.get("ops"), 3),
        "BB%": _to_percent(bb_pct),
        "K%": _to_percent(k_pct),
        "SLAM": str(stat.get("grandSlams", "N/A")),
    }


def _pitching_stats(mlb_id: int, period_req: dict):
    game_type = period_req.get("game_type") if period_req.get("stats_type") in {"season", "byDateRange"} else None
    try:
        stat = _get_stat_line(
            mlb_id=mlb_id,
            group="pitching",
            stats_type=period_req["stats_type"],
            season=period_req.get("season"),
            start_date=period_req.get("start_date"),
            end_date=period_req.get("end_date"),
            game_type=game_type,
        )
    except Exception:
        stat = {}
    if not stat:
        return {}

    ip_text = str(stat.get("inningsPitched", "0") or "0")
    try:
        ip_num = float(ip_text)
    except Exception:
        ip_num = 0.0

    so = float(stat.get("strikeOuts", 0) or 0)
    bb = float(stat.get("baseOnBalls", 0) or 0)
    batters_faced = float(stat.get("battersFaced", 0) or 0)
    k9 = (so * 9.0 / ip_num) if ip_num > 0 else None
    bb9 = (bb * 9.0 / ip_num) if ip_num > 0 else None
    k_minus_bb_pct = ((so - bb) / batters_faced * 100.0) if batters_faced > 0 else None

    return {
        "ERA": _to_num(stat.get("era"), 2),
        "WHIP": _to_num(stat.get("whip"), 2),
        "W": str(stat.get("wins", "N/A")),
        "IP": ip_text,
        "K": str(stat.get("strikeOuts", "N/A")),
        "BB": str(stat.get("baseOnBalls", "N/A")),
        "HLD": str(stat.get("holds", "N/A")),
        "SV": str(stat.get("saves", "N/A")),
        "SHO": str(stat.get("shutouts", "N/A")),
        "QS": str(stat.get("qualityStarts", "N/A")),
        "H": str(stat.get("hits", "N/A")),
        "K/9": _to_num(k9, 2),
        "BB/9": _to_num(bb9, 2),
        "K-BB%": _to_percent(k_minus_bb_pct),
    }


def _edad_desde_fecha(birth_date_str: str):
    if not birth_date_str:
        return "N/A"
    try:
        dob = date.fromisoformat(birth_date_str)
        today = date.today()
        return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
    except Exception:
        return "N/A"


def _ip_text_to_float(ip_text: str):
    """
    MLB innings format can be X.0, X.1, X.2 where .1=.333 and .2=.667 innings.
    """
    if not ip_text:
        return 0.0
    s = str(ip_text).strip()
    try:
        if "." not in s:
            return float(s)
        whole, frac = s.split(".", 1)
        w = float(whole or 0)
        if frac == "0":
            return w
        if frac == "1":
            return w + (1.0 / 3.0)
        if frac == "2":
            return w + (2.0 / 3.0)
        return float(s)
    except Exception:
        return 0.0


def _is_meaningful_pitching_profile(primary_pos: str, pitching_stats: dict):
    pitcher_positions = {"P", "SP", "RP", "CP", "TWP"}
    if primary_pos in pitcher_positions:
        return True
    ip_num = _ip_text_to_float((pitching_stats or {}).get("IP", "0"))
    # Position-player mop-up innings should not classify as pitching profile.
    return ip_num >= 8.0


@lru_cache(maxsize=256)
def _batter_statcast_window(mlb_id: int, start_date: str, end_date: str):
    try:
        df = statcast_batter(start_date, end_date, mlb_id)
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _period_dates_for_savant_hitting(period_req: dict):
    stats_type = period_req.get("stats_type")
    if stats_type == "byDateRange":
        s = period_req.get("start_date")
        e = period_req.get("end_date")
        if s and e:
            return s, e
        return None, None
    if stats_type == "season":
        season = period_req.get("season")
        if season:
            return f"{season}-03-01", f"{season}-11-30"
        return None, None
    return None, None


def _hitting_savant_from_season_row(row):
    brl_pa = row.get("brl_pa")
    if brl_pa is None or pd.isna(brl_pa):
        brl_pa = row.get("barrels_per_pa")
    return {
        "Avg Exit Velocity": _to_mph(row.get("avg_hit_speed")),
        "Max Exit Velocity": _to_mph(row.get("max_hit_speed")),
        "EV50": _to_mph(row.get("ev50")),
        "Barrel %": _to_percent(row.get("brl_percent")),
        "Hard Hit %": _to_percent(row.get("ev95percent")),
        "Barrels / PA": _to_rate(brl_pa),
        "Sweet Spot %": _to_percent(row.get("anglesweetspotpercent")),
        "Avg Launch Angle": _to_num(row.get("avg_hit_angle"), 1),
        "xBA": _to_decimal(row.get("xba"), 3),
        "xSLG": _to_decimal(row.get("xslg"), 3),
    }


def _hitting_savant_from_window(df: pd.DataFrame):
    if df.empty:
        return {}

    out = {}
    launch_speed = pd.to_numeric(df.get("launch_speed"), errors="coerce")
    launch_angle = pd.to_numeric(df.get("launch_angle"), errors="coerce")
    lsa = pd.to_numeric(df.get("launch_speed_angle"), errors="coerce")
    valid_bip = launch_speed.notna()
    bip_n = int(valid_bip.sum())
    if bip_n == 0:
        return {}

    ev = launch_speed[valid_bip]
    out["Avg Exit Velocity"] = _to_mph(ev.mean())
    out["Max Exit Velocity"] = _to_mph(ev.max())
    out["EV50"] = _to_mph(ev.quantile(0.5))
    out["Hard Hit %"] = _to_percent((ev >= 95).mean() * 100.0)

    barrel_n = int((lsa == 6).sum()) if lsa is not None else 0
    out["Barrel %"] = _to_percent((barrel_n / bip_n) * 100.0)

    pa_events = df.get("events")
    if pa_events is not None:
        pa_events = pa_events.fillna("").astype(str)
        pa_n = int((pa_events != "").sum())
        out["Barrels / PA"] = _to_percent((barrel_n / pa_n) * 100.0) if pa_n > 0 else "N/A"
    else:
        out["Barrels / PA"] = "N/A"

    valid_la = launch_angle.notna()
    if int(valid_la.sum()) > 0:
        la_vals = launch_angle[valid_la]
        sweet_spot = ((la_vals >= 8) & (la_vals <= 32)).mean() * 100.0
        out["Sweet Spot %"] = _to_percent(sweet_spot)
        out["Avg Launch Angle"] = _to_num(la_vals.mean(), 1)
    else:
        out["Sweet Spot %"] = "N/A"
        out["Avg Launch Angle"] = "N/A"

    est_ba_raw = df.get("estimated_ba_using_speedangle")
    est_slg_raw = df.get("estimated_slg_using_speedangle")
    est_ba = pd.to_numeric(est_ba_raw, errors="coerce") if est_ba_raw is not None else pd.Series(dtype="float64")
    est_slg = pd.to_numeric(est_slg_raw, errors="coerce") if est_slg_raw is not None else pd.Series(dtype="float64")
    out["xBA"] = _to_decimal(est_ba.mean(), 3) if hasattr(est_ba, "notna") and est_ba.notna().any() else "N/A"
    out["xSLG"] = _to_decimal(est_slg.mean(), 3) if hasattr(est_slg, "notna") and est_slg.notna().any() else "N/A"

    return out


def _build_hot_board(
    group: str,
    metric_map: dict,
    season: Optional[int],
    start_date: Optional[str],
    end_date: Optional[str],
    game_type: str,
    limit: int = 30,
    top_n: int = 10,
):
    bucket = {}
    for metric_label, cfg in metric_map.items():
        category = cfg["category"]
        weight = float(cfg.get("weight", 1.0))
        try:
            rows = _leaderboard_rows(
                group=group,
                category=category,
                season=season,
                start_date=start_date,
                end_date=end_date,
                game_type=game_type,
                limit=limit,
            )
        except Exception:
            rows = []

        for row in rows:
            try:
                player_id = int(row.get("id"))
                rank = int(row.get("rank"))
            except Exception:
                continue
            item = bucket.setdefault(
                player_id,
                {
                    "id": player_id,
                    "name": row.get("name", "N/A"),
                    "team": row.get("team", "N/A"),
                    "score": 0.0,
                    "metrics": {},
                },
            )
            item["score"] += (limit - rank + 1) * weight
            item["metrics"][metric_label] = {"rank": rank, "value": row.get("value", "N/A")}

    ranked = sorted(bucket.values(), key=lambda x: x["score"], reverse=True)
    out = []
    for idx, item in enumerate(ranked[:top_n], start=1):
        best = sorted(item["metrics"].items(), key=lambda kv: kv[1]["rank"])[:2]
        highlights = [f"{k} {v['value']}" for k, v in best]
        out.append(
            {
                "rank": idx,
                "id": item["id"],
                "name": item["name"],
                "team": item["team"],
                "value": " | ".join(highlights) if highlights else f"Score {round(item['score'], 1)}",
            }
        )
    return out


def _hitting_savant_stats(mlb_id: int, period_req: dict):
    if FBS_LIGHT_MODE:
        return {}, None, period_req.get("label", "Savant disabled (light mode)")
    stats_type = period_req.get("stats_type")
    # Career Savant aggregation is too expensive for live calls; use latest available season.
    if stats_type == "career":
        season_df = _savant_batter_season_df(SAVANT_SEASON)
        if season_df.empty:
            return {}, None, "Latest Available Savant"
        row = season_df[season_df.get("player_id") == mlb_id]
        if row.empty:
            return {}, SAVANT_SEASON, f"Season {SAVANT_SEASON}"
        return _hitting_savant_from_season_row(row.iloc[0]), SAVANT_SEASON, f"Season {SAVANT_SEASON}"

    if stats_type == "season":
        season = period_req.get("season") or SAVANT_SEASON
        season_df = _savant_batter_season_df(int(season))
        row = pd.DataFrame()
        if not season_df.empty:
            row = season_df[season_df.get("player_id") == mlb_id]

        # If selected season is thin/empty (e.g. new season), fallback to last season.
        used_season = int(season)
        if row.empty and used_season == CURRENT_SEASON:
            fallback_df = _savant_batter_season_df(LAST_SEASON)
            if not fallback_df.empty:
                fallback_row = fallback_df[fallback_df.get("player_id") == mlb_id]
                if not fallback_row.empty:
                    return _hitting_savant_from_season_row(fallback_row.iloc[0]), LAST_SEASON, f"Season {LAST_SEASON} (fallback)"

        if row.empty:
            return {}, used_season, f"Season {used_season}"
        return _hitting_savant_from_season_row(row.iloc[0]), used_season, f"Season {used_season}"

    start_date, end_date = _period_dates_for_savant_hitting(period_req)
    if not start_date or not end_date:
        return {}, None, period_req.get("label", "Range")

    df = _batter_statcast_window(mlb_id, start_date, end_date)
    return _hitting_savant_from_window(df), None, period_req.get("label", f"{start_date} to {end_date}")


@lru_cache(maxsize=256)
def _pitcher_statcast_window(mlb_id: int, start_date: str, end_date: str):
    try:
        df = statcast_pitcher(start_date, end_date, mlb_id)
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _period_dates_for_savant_pitching(period_req: dict):
    stats_type = period_req.get("stats_type")
    if stats_type == "byDateRange":
        s = period_req.get("start_date")
        e = period_req.get("end_date")
        if s and e:
            return s, e
        return None, None
    if stats_type == "season":
        season = period_req.get("season")
        if season:
            return f"{season}-03-01", f"{season}-11-30"
        return None, None
    return None, None


def _pitching_savant_stats(mlb_id: int, period_req: dict):
    if FBS_LIGHT_MODE:
        return {}
    start_date, end_date = _period_dates_for_savant_pitching(period_req)
    if not start_date or not end_date:
        return {}

    df = _pitcher_statcast_window(mlb_id, start_date, end_date)
    if df.empty:
        return {}

    out = {}
    launch_speed = pd.to_numeric(df.get("launch_speed"), errors="coerce")
    valid_bip = launch_speed.notna()
    bip_count = int(valid_bip.sum())
    if bip_count > 0:
        avg_ev_allowed = launch_speed[valid_bip].mean()
        hard_hit_allowed = (launch_speed[valid_bip] >= 95).mean() * 100.0
        out["Avg EV Allowed"] = _to_mph(avg_ev_allowed)
        out["Hard Hit % Allowed"] = _to_percent(hard_hit_allowed)

    desc = df.get("description")
    if desc is not None:
        desc = desc.fillna("").astype(str)
        swing_desc = {
            "swinging_strike",
            "swinging_strike_blocked",
            "foul",
            "foul_tip",
            "hit_into_play",
            "hit_into_play_no_out",
            "hit_into_play_score",
            "foul_bunt",
            "missed_bunt",
        }
        whiff_desc = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
        csw_desc = {"called_strike", "swinging_strike", "swinging_strike_blocked"}

        swings = desc.isin(swing_desc)
        whiffs = desc.isin(whiff_desc)
        csw = desc.isin(csw_desc)
        pitches = max(len(df), 1)

        swing_n = int(swings.sum())
        if swing_n > 0:
            out["Whiff %"] = _to_percent((int((swings & whiffs).sum()) / swing_n) * 100.0)
        out["CSW %"] = _to_percent((int(csw.sum()) / pitches) * 100.0)

        zone = pd.to_numeric(df.get("zone"), errors="coerce")
        if zone is not None:
            outside = zone > 9
            out_pitches = int(outside.sum())
            if out_pitches > 0:
                chase_swings = swings & outside
                out["Chase %"] = _to_percent((int(chase_swings.sum()) / out_pitches) * 100.0)

    events = df.get("events")
    if events is not None:
        events = events.fillna("").astype(str)
        pa_events = events[events != ""]
        pa_n = len(pa_events)
        if pa_n > 0:
            strikeouts = int(pa_events.isin({"strikeout", "strikeout_double_play"}).sum())
            walks = int(pa_events.isin({"walk", "intent_walk"}).sum())
            k_minus_bb = ((strikeouts - walks) / pa_n) * 100.0
            out["K-BB% (Savant)"] = _to_percent(k_minus_bb)

    return out


@app.get("/jugador/{nombre_busqueda}")
def api_get_jugador(
    nombre_busqueda: str,
    player_id: Optional[int] = None,
    prefer_pos: Optional[str] = None,
    prefer_team: Optional[str] = None,
    mode: Optional[str] = "currentseason",
    season: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    if player_id:
        jugador = _get_player_details_by_id(player_id)
        if not jugador:
            return {"error": "No encontrado"}
        search_result = {"ambiguous": False, "player": jugador}
    else:
        search_result = buscar_jugador_mlb(nombre_busqueda, prefer_pos=prefer_pos, prefer_team=prefer_team)
        if not search_result:
            return {"error": "No encontrado"}

    if search_result.get("ambiguous"):
        return {
            "disambiguation": True,
            "options": search_result.get("options", []),
            "query": nombre_busqueda,
        }

    jugador = search_result["player"]

    if not jugador:
        return {"error": "No encontrado"}

    mlb_id = jugador["id"]
    birth_date = jugador.get("birthDate")
    period_req = _resolve_period_request(mode, season, start_date, end_date)
    if period_req.get("error"):
        return {"error": period_req["error"]}

    savant_stats, savant_hitting_season, savant_hitting_label = _hitting_savant_stats(mlb_id, period_req)
    savant_pitching = _pitching_savant_stats(mlb_id, period_req)
    hitting_stats = _hitting_stats(mlb_id, period_req)
    pitching_stats = _pitching_stats(mlb_id, period_req)
    hitting_season = period_req.get("season")
    pitching_season = period_req.get("season")
    if period_req.get("stats_type") == "season" and period_req.get("leader_season"):
        leader_highlights = _build_leader_highlights(mlb_id, period_req["leader_season"], period_req["leader_season"])
    else:
        leader_highlights = {"hitting": {}, "pitching": {}}

    primary_pos = (jugador.get("primaryPosition", {}).get("abbreviation", "") or "").upper()
    has_meaningful_pitching = _is_meaningful_pitching_profile(primary_pos, pitching_stats)
    if not has_meaningful_pitching:
        pitching_stats = {}
        savant_pitching = {}

    is_two_way = primary_pos == "TWP" or (bool(hitting_stats) and bool(pitching_stats))

    return {
        "player_id": mlb_id,
        "nombre": jugador.get("fullName", "N/A"),
        "equipo": jugador.get("currentTeam", {}).get("name", "N/A"),
        "team_id": jugador.get("currentTeam", {}).get("id"),
        "team_logo_url": f"https://www.mlbstatic.com/team-logos/{jugador.get('currentTeam', {}).get('id')}.svg" if jugador.get("currentTeam", {}).get("id") else None,
        "team_logo_cap_dark_url": f"https://www.mlbstatic.com/team-logos/team-cap-on-dark/{jugador.get('currentTeam', {}).get('id')}.svg" if jugador.get("currentTeam", {}).get("id") else None,
        "team_logo_cap_light_url": f"https://www.mlbstatic.com/team-logos/team-cap-on-light/{jugador.get('currentTeam', {}).get('id')}.svg" if jugador.get("currentTeam", {}).get("id") else None,
        "posicion": jugador.get("primaryPosition", {}).get("abbreviation", "N/A"),
        "tipo_jugador": "Two-Way" if is_two_way else ("Pitcher" if primary_pos in {"P", "SP", "RP", "CP"} else "Hitter"),
        "numero": jugador.get("primaryNumber", "N/A"),
        "edad": _edad_desde_fecha(birth_date),
        "batea": jugador.get("batSide", {}).get("description", "N/A"),
        "lanza": jugador.get("pitchHand", {}).get("description", "N/A"),
        "altura": jugador.get("height", "N/A"),
        "peso_lb": jugador.get("weight", "N/A"),
        "nacimiento": birth_date or "N/A",
        "debut_mlb": jugador.get("mlbDebutDate", "N/A"),
        "foto_url": (
            "https://img.mlbstatic.com/mlb-photos/image/upload/"
            "d_people:generic:headshot:67:current.png/w_426,q_auto:best/v1/"
            f"people/{mlb_id}/headshot/67/current"
        ),
        "savant_hitting": savant_stats,
        "savant_pitching": savant_pitching,
        "hitting_stats": hitting_stats,
        "pitching_stats": pitching_stats,
        "hitting_season": hitting_season,
        "pitching_season": pitching_season,
        "stats_mode": period_req.get("mode"),
        "stats_period_label": period_req.get("label"),
        "leader_highlights": leader_highlights,
        "savant_season": savant_hitting_season,
        "savant_period_label": savant_hitting_label,
    }


@app.get("/autocomplete/{texto}")
def api_autocomplete(texto: str):
    return {"suggestions": _autocomplete_candidates(texto, limit=6)}


@app.get("/dashboard")
def api_dashboard(
    mode: Optional[str] = "currentseason",
    season: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    requested_mode = (mode or "").lower()
    if requested_mode in {"spring2026", "spring", "springtraining"}:
        period_req = _resolve_period_request("spring2026", 2026, None, None)
    else:
        # Dashboard is intentionally independent of selected player range.
        # Leaders stay on current season; "on fire" uses a rolling short window.
        period_req = _resolve_period_request("currentseason", CURRENT_SEASON, None, None)

    group_hitting = {
        "HR": "homeRuns",
        "OPS": "ops",
        "RBI": "runsBattedIn",
        "AVG": "battingAverage",
        "SB": "stolenBases",
    }
    group_pitching = {
        "ERA": "era",
        "WHIP": "whip",
        "K": "strikeouts",
        "SV": "saves",
        "K/9": "strikeoutsPer9Inn",
    }

    use_spring = period_req.get("mode") == "spring2026"
    game_type = period_req.get("game_type", "R")
    season_arg = period_req.get("season") if not use_spring else None
    start_arg = period_req.get("start_date") if use_spring else None
    end_arg = period_req.get("end_date") if use_spring else None

    hitting_boards = {}
    pitching_boards = {}

    for label, category in group_hitting.items():
        try:
            hitting_boards[label] = _leaderboard_rows("hitting", category, season_arg, start_arg, end_arg, game_type=game_type, limit=8)
        except Exception:
            hitting_boards[label] = []

    for label, category in group_pitching.items():
        try:
            pitching_boards[label] = _leaderboard_rows("pitching", category, season_arg, start_arg, end_arg, game_type=game_type, limit=8)
        except Exception:
            pitching_boards[label] = []

    # "On fire" snapshot = rolling 7-day trend board, split for hitters/pitchers.
    if use_spring:
        spring_start_d = date(2026, 2, 20)
        spring_end_d = date(2026, 3, 31)
        fire_end = min(date.today(), spring_end_d)
        fire_start = max(spring_start_d, fire_end - timedelta(days=6))
    else:
        fire_end = date.today()
        fire_start = fire_end - timedelta(days=6)
    fire_start_s = fire_start.isoformat()
    fire_end_s = fire_end.isoformat()
    on_fire_hitter_map = {
        "OPS": {"category": "ops", "weight": 2.6},
        "HR": {"category": "homeRuns", "weight": 2.0},
        "RBI": {"category": "runsBattedIn", "weight": 1.7},
        "AVG": {"category": "battingAverage", "weight": 1.4},
    }
    on_fire_pitcher_map = {
        "ERA": {"category": "era", "weight": 2.4},
        "WHIP": {"category": "whip", "weight": 2.2},
        "K": {"category": "strikeouts", "weight": 2.0},
        "K/9": {"category": "strikeoutsPer9Inn", "weight": 1.5},
    }
    try:
        on_fire_hitters = _build_hot_board(
            group="hitting",
            metric_map=on_fire_hitter_map,
            season=None,
            start_date=fire_start_s,
            end_date=fire_end_s,
            game_type=game_type,
            limit=30,
            top_n=10,
        )
    except Exception:
        on_fire_hitters = []
    try:
        on_fire_pitchers = _build_hot_board(
            group="pitching",
            metric_map=on_fire_pitcher_map,
            season=None,
            start_date=fire_start_s,
            end_date=fire_end_s,
            game_type=game_type,
            limit=30,
            top_n=10,
        )
    except Exception:
        on_fire_pitchers = []

    return {
        "stats_mode": period_req.get("mode"),
        "stats_period_label": period_req.get("label"),
        "hitting_leaders": hitting_boards,
        "pitching_leaders": pitching_boards,
        "on_fire": on_fire_hitters,
        "on_fire_hitters": on_fire_hitters,
        "on_fire_pitchers": on_fire_pitchers,
        "on_fire_label": f"Last 7 Days ({fire_start.isoformat()} to {fire_end.isoformat()})",
    }


@app.get("/waiver_scanner")
def api_waiver_scanner(mode: Optional[str] = "currentseason"):
    requested_mode = (mode or "").lower()
    if requested_mode in {"spring2026", "spring", "springtraining"}:
        period_req = _resolve_period_request("spring2026", 2026, None, None)
    else:
        period_req = _resolve_period_request("currentseason", CURRENT_SEASON, None, None)

    use_spring = period_req.get("mode") == "spring2026"
    game_type = period_req.get("game_type", "R")
    season_arg = period_req.get("season") if not use_spring else None
    start_arg = period_req.get("start_date") if use_spring else None
    end_arg = period_req.get("end_date") if use_spring else None

    # Yahoo Roto league tuned profile (from user-provided categories).
    # Scanner emphasizes positive categories and approximates a few "net" buckets.
    hitter_metric_map = {
        "R": {"category": "runs", "weight": 2.2},
        "HR": {"category": "homeRuns", "weight": 2.5},
        "RBI": {"category": "runsBattedIn", "weight": 2.5},
        "BB": {"category": "walks", "weight": 1.4},
        "AVG": {"category": "battingAverage", "weight": 2.0},
        "OBP": {"category": "onBasePercentage", "weight": 2.0},
        "SLG": {"category": "sluggingPercentage", "weight": 2.1},
        "NSB": {"category": "stolenBases", "weight": 1.5},   # approximation
        "SLAM": {"category": "grandSlams", "weight": 1.1},
    }
    pitcher_metric_map = {
        "SHO": {"category": "shutouts", "weight": 1.0},
        "K": {"category": "strikeouts", "weight": 2.2},
        "HLD": {"category": "holds", "weight": 1.8},
        "ERA": {"category": "era", "weight": 2.4},
        "WHIP": {"category": "whip", "weight": 2.4},
        "BB/9": {"category": "walksPer9Inn", "weight": 1.8},
        "QS": {"category": "qualityStarts", "weight": 2.1},
        "NSV": {"category": "saves", "weight": 1.8},         # approximation
        "NW": {"category": "wins", "weight": 1.7},           # approximation
    }

    hitters = _waiver_candidates(
        group="hitting",
        metric_map=hitter_metric_map,
        season=season_arg,
        start_date=start_arg,
        end_date=end_arg,
        game_type=game_type,
        top_n=12,
    )
    pitchers = _waiver_candidates(
        group="pitching",
        metric_map=pitcher_metric_map,
        season=season_arg,
        start_date=start_arg,
        end_date=end_arg,
        game_type=game_type,
        top_n=12,
    )

    return {
        "mode": period_req.get("mode"),
        "label": period_req.get("label"),
        "hitters": hitters,
        "pitchers": pitchers,
        "note": (
            "Yahoo Roto tuned. Approximations: NSB uses SB, NSV uses SV, NW uses W. "
            "Negative categories (bat K/GIDP/E, pitch HR/GIDP/TB) are not fully modeled yet."
        ),
    }


@app.get("/auth/yahoo/start")
def auth_yahoo_start():
    global _yahoo_auth_state
    if not _yahoo_configured():
        return {"error": "Yahoo OAuth not configured. Set YAHOO_CLIENT_ID, YAHOO_CLIENT_SECRET, YAHOO_REDIRECT_URI."}
    _yahoo_auth_state = secrets.token_urlsafe(24)
    params = {
        "client_id": YAHOO_CLIENT_ID,
        "redirect_uri": YAHOO_REDIRECT_URI,
        "response_type": "code",
        "language": "en-us",
        "scope": "fspt-r",
        "state": _yahoo_auth_state,
    }
    url = "https://api.login.yahoo.com/oauth2/request_auth?" + urllib.parse.urlencode(params)
    return RedirectResponse(url=url, status_code=302)


@app.get("/auth/yahoo/callback")
def auth_yahoo_callback(code: Optional[str] = None, state: Optional[str] = None):
    global _yahoo_auth_state
    if not _yahoo_configured():
        return {"error": "Yahoo OAuth not configured."}
    if not code:
        return {"error": "Missing Yahoo authorization code."}
    if not state or state != _yahoo_auth_state:
        return {"error": "Invalid OAuth state."}

    try:
        token_payload = _yahoo_exchange_token(
            "authorization_code",
            code=code,
            redirect_uri=YAHOO_REDIRECT_URI,
        )
        _yahoo_store_token(token_payload)
        _yahoo_auth_state = None
    except Exception as e:
        return {"error": f"Yahoo token exchange failed: {e}"}

    if FBS_FRONTEND_URL:
        return RedirectResponse(url=FBS_FRONTEND_URL, status_code=302)
    return HTMLResponse(
        "<html><body style='font-family:Arial;padding:20px;background:#0b1220;color:#e2e8f0;'>"
        "<h3>Yahoo connected</h3><p>You can return to Fantasy Baseball Scout and refresh.</p></body></html>"
    )


@app.get("/auth/yahoo/status")
def auth_yahoo_status():
    connected = _yahoo_refresh_if_needed()
    return {
        "configured": _yahoo_configured(),
        "connected": bool(connected and _yahoo_token.get("access_token")),
        "scope": _yahoo_token.get("scope", ""),
    }


@app.get("/auth/yahoo/logout")
def auth_yahoo_logout():
    global _yahoo_auth_state
    _yahoo_auth_state = None
    _yahoo_token.clear()
    return {"ok": True}


@app.get("/auth/yahoo/leagues")
def auth_yahoo_leagues():
    if not _yahoo_refresh_if_needed():
        return {"error": "Yahoo not connected"}
    try:
        # Yahoo Fantasy current user MLB leagues (JSON format).
        data = _yahoo_api_get_json(
            "https://fantasysports.yahooapis.com/fantasy/v2/users;use_login=1/games;game_keys=mlb/leagues?format=json"
        )
    except Exception as e:
        return {"error": f"Yahoo leagues fetch failed: {e}"}

    leagues = _extract_leagues(data)
    dedup = {}
    for lg in leagues:
        key = lg.get("league_key")
        if key and key not in dedup:
            dedup[key] = lg
    out = sorted(dedup.values(), key=lambda x: str(x.get("name", "")).lower())
    return {"leagues": out}


@app.get("/auth/yahoo/league_context")
def auth_yahoo_league_context(league_key: str):
    if not league_key:
        return {"error": "league_key is required"}
    if not _yahoo_refresh_if_needed():
        return {"error": "Yahoo not connected"}

    try:
        settings_data = _yahoo_api_get_json(
            f"https://fantasysports.yahooapis.com/fantasy/v2/league/{urllib.parse.quote(league_key)}/settings?format=json"
        )
        standings_data = _yahoo_api_get_json(
            f"https://fantasysports.yahooapis.com/fantasy/v2/league/{urllib.parse.quote(league_key)}/standings?format=json"
        )
        teams_data = _yahoo_api_get_json(
            f"https://fantasysports.yahooapis.com/fantasy/v2/league/{urllib.parse.quote(league_key)}/teams?format=json"
        )
    except Exception as e:
        return {"error": f"Yahoo league context fetch failed: {e}"}

    scoring_type = _deep_find_first(settings_data, "scoring_type")
    league_name = _deep_find_first(settings_data, "name") or _deep_find_first(teams_data, "name")
    max_teams = _deep_find_first(settings_data, "num_teams")
    categories = _extract_stat_categories(settings_data)
    teams_rows = _extract_team_rows(standings_data) or _extract_team_rows(teams_data)
    teams_rows = [t for t in teams_rows if t.get("team_key")]
    teams_rows = sorted(
        teams_rows,
        key=lambda x: (int(x["rank"]) if str(x.get("rank", "")).isdigit() else 9999, str(x.get("name", ""))),
    )

    return {
        "league_key": league_key,
        "league_name": league_name,
        "scoring_type": scoring_type,
        "max_teams": max_teams,
        "stat_categories": categories[:40],
        "teams": teams_rows[:20],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
