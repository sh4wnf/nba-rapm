# tbis file includes the main data pipeline including:
# fetching nba play-by-play, building stints, and preparing rapm matrices

from __future__ import annotations

import argparse
import ast
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from nba_api.stats.endpoints import playbyplayv3
from nba_api.stats.library.http import NBAStatsHTTP
from requests.adapters import HTTPAdapter
from scipy.sparse import csr_matrix
from urllib3.util.retry import Retry

from . import config
from .rapm_model import create_sparse_matrices as _create_sparse_matrices
from .utils import ensure_dirs, normalize_game_id, setup_logging


logger = setup_logging()


# scraping and raw pbp

NBA_API_BASE = "https://stats.nba.com/stats"
NBA_API_TIMEOUT = 60
MAX_RETRIES = 5
RETRY_DELAY_BASE = 2.0
RETRY_DELAY_MAX = 30.0
REQUEST_DELAY_MIN = 1.0
REQUEST_DELAY_MAX = 2.5
ERROR_DELAY_MIN = 5.0
ERROR_DELAY_MAX = 10.0

NBA_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}

MIN_FILE_SIZE = 100
MIN_FILE_CONTENT_LENGTH = 10


# creates a smarter requests session with retries + nba headers
def _create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(NBA_HEADERS)

    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


# finds the game id prefix for a season string
def _get_season_prefix(season: str) -> str:
    if season == "2023-24":
        return "00223"
    if season == "2024-25":
        return "00224"
    year = season.split("-")[0][-2:]
    return f"002{year}"


# pulls the full list of nba game ids for a season
def _fetch_game_list(
    session: requests.Session, season: str, season_type: str = "Regular Season"
) -> List[str]:
    logger.info("Fetching game list for season %s (%s)", season, season_type)

    endpoint = "leaguegamefinder"
    params = {
        "Season": season,
        "SeasonType": season_type,
        "LeagueID": "00",
    }

    url = f"{NBA_API_BASE}/{endpoint}"
    response = session.get(url, params=params, timeout=NBA_API_TIMEOUT)
    response.raise_for_status()

    data = response.json()

    if "resultSets" in data:
        result_sets = data["resultSets"]
    elif "resultSet" in data:
        result_sets = [data["resultSet"]] if isinstance(data["resultSet"], dict) else data["resultSet"]
    else:
        raise ValueError(f"Unexpected API response structure: {list(data.keys())}")

    games_data = None
    for rs in result_sets:
        if isinstance(rs, dict):
            name = rs.get("name", "")
            if "LeagueGameFinder" in name or "GameFinder" in name:
                games_data = rs
                break
    if not games_data and result_sets:
        games_data = result_sets[0] if isinstance(result_sets[0], dict) else None
    if not games_data:
        raise ValueError("Could not find game finder result set in response")

    headers = games_data.get("headers", [])
    rows = games_data.get("rowSet", [])
    if "GAME_ID" not in headers:
        raise ValueError(f"GAME_ID not found in response headers: {headers}")

    game_id_idx = headers.index("GAME_ID")
    game_ids = [str(row[game_id_idx]) for row in rows]

    season_prefix = _get_season_prefix(season)
    valid_ids = [
        gid
        for gid in game_ids
        if gid.startswith("002") and len(gid) == 10 and gid.startswith(season_prefix)
    ]

    unique_ids = sorted(set(valid_ids))
    logger.info(
        "Found %d regular-season game IDs (%d unique)",
        len(valid_ids),
        len(unique_ids),
    )
    return unique_ids


# grabs the v3 play-by-play for one game with some retry attempts
def _fetch_play_by_play(session: requests.Session, game_id: str) -> Optional[pd.DataFrame]:
    NBAStatsHTTP.timeout = NBA_API_TIMEOUT

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                delay = min(RETRY_DELAY_BASE * (2 ** (attempt - 2)), RETRY_DELAY_MAX)
                time.sleep(delay)

            pbp = playbyplayv3.PlayByPlayV3(
                game_id=game_id,
                timeout=NBA_API_TIMEOUT,
            )
            data_frames = pbp.get_data_frames()
            if not data_frames:
                if attempt < MAX_RETRIES:
                    continue
                return None

            df = data_frames[0]
            if df.empty:
                if attempt < MAX_RETRIES:
                    continue
                return pd.DataFrame(columns=["GAME_ID"])

            df["GAME_ID"] = normalize_game_id(game_id)
            return df
        except Exception as e:
            logger.warning("Error fetching game %s (attempt %d/%d): %s", game_id, attempt, MAX_RETRIES, e)
            if attempt == MAX_RETRIES:
                logger.error("Failed to fetch game %s after %d attempts", game_id, MAX_RETRIES)
                return None

    return None


# checking that a downloaded pbp csv actually looks like data
def _validate_file(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    try:
        if file_path.stat().st_size < MIN_FILE_SIZE:
            return False
        df = pd.read_csv(file_path, nrows=1)
        return "GAME_ID" in df.columns
    except Exception:
        return False


# scans the raw pbp folder to prevent re-downloading games that already have been downloaded
def _get_existing_game_ids() -> Set[str]:
    existing: Set[str] = set()
    for file_path in config.PBP_RAW_DIR.glob("*.csv"):
        if _validate_file(file_path):
            existing.add(file_path.stem)
    return existing


# making sure to keep only game ids that look like they belong to this season
def _filter_valid_game_ids(game_ids: List[str], season: str) -> List[str]:
    season_prefix = _get_season_prefix(season)
    valid_ids = [
        gid
        for gid in game_ids
        if isinstance(gid, str) and gid.startswith(season_prefix) and len(gid) == 10
    ]
    if len(valid_ids) != len(game_ids):
        logger.warning(
            "Filtered out %d invalid game IDs (expected %s prefix, 10 digits)",
            len(game_ids) - len(valid_ids),
            season_prefix,
        )
    return valid_ids


# downloads one game's play-by-play, normalizes, and writes a csv
def _fetch_and_save_game(
    session: requests.Session,
    game_id: str,
    checkpoint: dict,
) -> bool:
    dest = config.PBP_RAW_DIR / f"{game_id}.csv"

    if _validate_file(dest):
        logger.debug("Skipping existing game: %s", game_id)
        return True

    if dest.exists():
        try:
            dest.unlink()
        except Exception as e:
            logger.warning("Failed to remove invalid file %s: %s", dest.name, e)

    df = _fetch_play_by_play(session, game_id)
    if df is None or df.empty:
        logger.warning("No data available for game %s", game_id)
        return False

    df["GAME_ID"] = normalize_game_id(game_id)

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dest, index=False)
        checkpoint["last_game_id"] = game_id
        from .utils import save_checkpoint 

        save_checkpoint(checkpoint)
        logger.info("Saved PBP %s (%d rows)", game_id, len(df))
        return True
    except Exception as e:
        logger.error("Failed to save game %s: %s", game_id, e, exc_info=True)
        return False


# the main loop that downloads the games with some basic pacing
def _fetch_games(
    session: requests.Session,
    game_ids: List[str],
    checkpoint: dict,
    shuffle: bool = True,
) -> Tuple[List[str], List[str]]:
    if shuffle:
        game_ids = game_ids.copy()
        random.shuffle(game_ids)
        logger.info("Shuffled fetch order for robustness")

    successful: List[str] = []
    failed: List[str] = []

    for i, game_id in enumerate(game_ids, 1):
        logger.info("[%d/%d] Processing game %s", i, len(game_ids), game_id)
        if _fetch_and_save_game(session, game_id, checkpoint):
            successful.append(game_id)
            delay = REQUEST_DELAY_MIN + random.random() * (REQUEST_DELAY_MAX - REQUEST_DELAY_MIN)
            time.sleep(delay)
        else:
            failed.append(game_id)
            delay = ERROR_DELAY_MIN + random.random() * (ERROR_DELAY_MAX - ERROR_DELAY_MIN)
            time.sleep(delay)

    return successful, failed


# put together all the individual game csvs into one big play-by-play file
def _combine_game_files(season: str) -> None:
    season_prefix = _get_season_prefix(season)
    all_frames: List[pd.DataFrame] = []
    processed_games: Set[str] = set()

    logger.info("Combining game files for season %s (prefix %s)", season, season_prefix)

    game_files = [
        p
        for p in sorted(config.PBP_RAW_DIR.glob("*.csv"))
        if p.stem.startswith(season_prefix) and len(p.stem) == 10
    ]
    if not game_files:
        logger.warning("No game files found for season %s", season)
        return

    for p in game_files:
        game_id = p.stem
        if game_id in processed_games:
            continue
        try:
            df = pd.read_csv(p, dtype=str)
            if "GAME_ID" in df.columns:
                df["GAME_ID"] = normalize_game_id(game_id)
            before = len(df)
            df = df.drop_duplicates()
            if before != len(df):
                logger.debug("Removed %d duplicate rows from game %s", before - len(df), game_id)
            all_frames.append(df)
            processed_games.add(game_id)
        except Exception as e:
            logger.error("Error reading game file %s: %s", p, e, exc_info=True)

    if not all_frames:
        logger.error("No valid game files to combine")
        return

    combined = pd.concat(all_frames, ignore_index=True).drop_duplicates()
    combined.to_csv(config.PBP_COMBINED, index=False)
    unique_games = combined["GAME_ID"].nunique()
    logger.info(
        "Wrote combined PBP to %s (%d rows, %d unique games)",
        config.PBP_COMBINED,
        len(combined),
        unique_games,
    )


# writes out a text file listing any game_ids that failed to download
def _save_missing_game_ids(failed_ids: List[str]) -> None:
    if not failed_ids:
        return
    missing_path = config.DATA_DIR / "missing_game_ids.txt"
    try:
        missing_path.write_text("\n".join(sorted(set(failed_ids))))
        logger.warning("Wrote list of %d missing game IDs to %s", len(failed_ids), missing_path)
    except Exception as e:
        logger.error("Failed to write missing_game_ids.txt: %s", e, exc_info=True)


# ingestion driver that's used by the fetch_pbp_for_season helper
def _ingest_season(
    season: str = "2023-24",
    dry_run: bool = True,
    sample_size: Optional[int] = None,
) -> int:
    from .utils import load_checkpoint

    ensure_dirs()
    logger.info("=" * 60)
    logger.info("Starting ingestion for season %s (dry-run=%s)", season, dry_run)
    if sample_size:
        logger.info("Sample mode: fetching only %d games", sample_size)
    logger.info("=" * 60)

    session = _create_session()

    try:
        api_game_ids = _fetch_game_list(session, season, "Regular Season")
    except Exception as e:
        logger.error("Failed to fetch game list: %s", e, exc_info=True)
        return 1

    checkpoint = load_checkpoint()
    existing_ids = _get_existing_game_ids()
    valid_ids = _filter_valid_game_ids(api_game_ids, season)

    if sample_size:
        valid_ids = valid_ids[:sample_size]
        logger.info("Limited to sample of %d games", sample_size)

    to_fetch = [gid for gid in valid_ids if gid not in existing_ids]
    logger.info(
        "Game status: total=%d, existing=%d, to_fetch=%d",
        len(valid_ids),
        len(existing_ids),
        len(to_fetch),
    )

    if dry_run:
        logger.info("Dry-run only; re-run with dry_run=False to fetch games.")
        return 0

    if to_fetch:
        logger.info("Fetching %d missing games...", len(to_fetch))
        successful, failed = _fetch_games(session, to_fetch, checkpoint, shuffle=True)
        logger.info("Fetch complete: successful=%d, failed=%d", len(successful), len(failed))
        if failed:
            _save_missing_game_ids(failed)
    else:
        logger.info("All games already exist on disk")

    try:
        _combine_game_files(season)
    except Exception as e:
        logger.error("Failed to combine game files: %s", e, exc_info=True)
        return 1

    logger.info("Ingestion complete for season %s", season)
    return 0


# public entry point: fetch and combine all play-by-play for one season
def fetch_pbp_for_season(season: str = config.SEASON) -> None:
    ensure_dirs()
    exit_code = _ingest_season(season=season, dry_run=False, sample_size=None)
    if exit_code != 0:
        raise RuntimeError(f"Failed to fetch PBP data for season {season}")
    # Build an index of all games as in the original pbp_scraper helper.
    create_all_games_index(season)


# build a csv index with one row per raw game on the localdisk
def create_all_games_index(season: str = config.SEASON) -> Path:
    ensure_dirs()

    game_files = list(config.PBP_RAW_DIR.glob("*.csv"))
    if not game_files:
        logger.warning("No game files found in %s", config.PBP_RAW_DIR)
        return config.ALL_GAMES_CSV

    games: List[Dict] = []
    for game_file in game_files:
        game_id = game_file.stem
        try:
            df = pd.read_csv(game_file, nrows=1)
            games.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "file_path": str(game_file.relative_to(config.DATA_DIR)),
                    "file_size_bytes": game_file.stat().st_size,
                }
            )
        except Exception as e:
            logger.warning("Failed to read metadata for game %s: %s", game_id, e)
            games.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "file_path": str(game_file.relative_to(config.DATA_DIR)),
                    "file_size_bytes": game_file.stat().st_size,
                }
            )

    games_df = pd.DataFrame(games).sort_values("game_id")
    games_df.to_csv(config.ALL_GAMES_CSV, index=False)
    logger.info("Created all_games.csv index with %d games", len(games_df))
    return config.ALL_GAMES_CSV


# stint building

# event constants for the v3 play-by-play
EVENT_MADE_SHOT = 1
EVENT_FREE_THROW = 3
EVENT_OFFENSIVE_REBOUND = 4
EVENT_TURNOVER = 5
EVENT_SUBSTITUTION = 8

MIN_PLAYER_ID = 1
MAX_PLAYER_ID = 10_000_000

# only scan the first 80 events to try to find starters
MAX_STARTING_LINEUP_SCAN = 80  


# checking that values look like a real nba player id
def _is_valid_player_id(player_id) -> bool:
    try:
        pid = int(float(player_id))
        return MIN_PLAYER_ID <= pid < MAX_PLAYER_ID
    except (ValueError, TypeError):
        return False


# converts raw player_id values into clean ints
def _parse_player_id(player_id) -> Optional[int]:
    if player_id is None or player_id == "" or pd.isna(player_id):
        return None
    try:
        return int(float(player_id))
    except (ValueError, TypeError):
        return None


# guesses the starting lineups from early events + subs in the game
def _get_starting_lineup(pbp_df: pd.DataFrame) -> Tuple[List[int], List[int]]:
    teams = pbp_df[pbp_df["teamTricode"].notna()]["teamTricode"].unique()
    if len(teams) < 2:
        return [], []

    home_team = teams[0]
    away_team = teams[1]

    home_players: Dict[int, int] = {}
    away_players: Dict[int, int] = {}
    events_to_scan = pbp_df.head(min(MAX_STARTING_LINEUP_SCAN * 2, len(pbp_df)))

    first_sub_idx: Optional[int] = None

    for idx, row in events_to_scan.iterrows():
        action_type = row.get("actionType")
        if pd.notna(action_type) and "Substitution" in str(action_type):
            if first_sub_idx is None:
                first_sub_idx = idx
            if first_sub_idx is not None and idx > first_sub_idx + 10:
                break

        player_id = row.get("personId")
        team = row.get("teamTricode")

        if player_id and player_id != 0 and _is_valid_player_id(player_id):
            try:
                player_id = int(float(player_id))
            except (ValueError, TypeError):
                continue

            if team == home_team and player_id not in home_players:
                home_players[player_id] = idx
            elif team == away_team and player_id not in away_players:
                away_players[player_id] = idx

        if (
            len(home_players) >= 5
            and len(away_players) >= 5
            and first_sub_idx is not None
            and idx > first_sub_idx + 5
        ):
            break

    home_lineup = [pid for pid, _ in sorted(home_players.items(), key=lambda x: x[1])[:5]]
    away_lineup = [pid for pid, _ in sorted(away_players.items(), key=lambda x: x[1])[:5]]

    if len(home_lineup) < 5 or len(away_lineup) < 5:
        logger.warning(
            "Incomplete starting lineup: home=%d players, away=%d players",
            len(home_lineup),
            len(away_lineup),
        )

    return home_lineup, away_lineup


# helper for initializing counting stats for a new stint
def _init_stint_stats() -> Dict[str, int]:
    return {
        "team_pts": 0,
        "opp_pts": 0,
        "turnovers": 0,
        "off_reb": 0,
        "fta": 0,
        "fga": 0,
    }


# updates the current stint's counters based on one pbp row
def _update_stint_stats(row: pd.Series, stats: Dict[str, int]) -> None:
    action_type = row.get("actionType")
    if pd.isna(action_type):
        return
    action_type = str(action_type).strip()

    if "Turnover" in action_type:
        stats["turnovers"] += 1
    elif action_type == "Rebound":
        desc = str(row.get("description", "")).lower()
        if "off" in desc or "offensive" in desc:
            stats["off_reb"] += 1
    elif "Free Throw" in action_type:
        stats["fta"] += 1
    elif "Shot" in action_type or "Field Goal" in action_type:
        stats["fga"] += 1


# creates a dictionary mapping player names to player ids
def _build_name_to_id_mapping(game_df: pd.DataFrame) -> Dict[str, int]:
    name_to_id: Dict[str, int] = {}
    for _, row in game_df.iterrows():
        if pd.notna(row.get("personId")) and row.get("personId") != 0:
            try:
                player_id = int(float(row.get("personId")))
            except (ValueError, TypeError):
                continue
            player_name = str(row.get("playerName", "")).strip()
            if player_name and _is_valid_player_id(player_id):
                name_to_id[player_name] = player_id
                parts = player_name.split()
                if len(parts) > 1:
                    last = parts[-1]
                    if len(last) > 2:
                        name_to_id[last] = player_id
                    first_last = f"{parts[0]} {parts[-1]}"
                    name_to_id[first_last] = player_id
    return name_to_id


 # trying to figure out who checkec in
def _parse_sub_description(description: str, name_to_id: Dict[str, int]) -> Optional[int]:
    import re

    if pd.isna(description):
        return None
    desc = str(description).strip()

    patterns = [
        r"SUB:\s*([A-Za-z\s\.\-\']+)\s+FOR\s+([A-Za-z\s\.\-\']+)",
        r"([A-Za-z\s\.\-\']+)\s+FOR\s+([A-Za-z\s\.\-\']+)",
        r"SUB:\s*(\w+)\s+FOR\s+(\w+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, desc, re.IGNORECASE)
        if not match:
            continue
        player_in_name = match.group(1).strip()
        if player_in_name in name_to_id:
            return name_to_id[player_in_name]
        for name, pid in name_to_id.items():
            if player_in_name.lower() in name.lower() or name.lower() in player_in_name.lower():
                return pid
        last_word = player_in_name.split()[-1] if player_in_name.split() else player_in_name
        for name, pid in name_to_id.items():
            if last_word.lower() in name.lower():
                return pid
    return None


 # update lineups when a substitution event happens
def _update_lineup_for_sub(
    home_lineup: List[int],
    away_lineup: List[int],
    player_out: Optional[int],
    player_in: Optional[int],
    team: Optional[str] = None,
    home_team: Optional[str] = None,
) -> Tuple[List[int], List[int], bool]:

    # use the team info when available to decide which lineup to edit.
    if team and home_team:
        if team == home_team:
            if player_out and player_out in home_lineup:
                updated = [p for p in home_lineup if p != player_out]
                if player_in and player_in not in updated:
                    updated.append(player_in)
                if len(updated) > 5:
                    updated = updated[:5]
                elif len(updated) < 5 and player_in:
                    updated.append(player_in)
                return updated, away_lineup, True
            if player_in and len(home_lineup) < 5 and player_in not in home_lineup:
                updated = (home_lineup + [player_in])[:5]
                return updated, away_lineup, True
        else:
            if player_out and player_out in away_lineup:
                updated = [p for p in away_lineup if p != player_out]
                if player_in and player_in not in updated:
                    updated.append(player_in)
                if len(updated) > 5:
                    updated = updated[:5]
                elif len(updated) < 5 and player_in:
                    updated.append(player_in)
                return home_lineup, updated, True
            if player_in and len(away_lineup) < 5 and player_in not in away_lineup:
                updated = (away_lineup + [player_in])[:5]
                return home_lineup, updated, True

    # if there's no team info then search both lineups
    if player_out:
        if player_out in home_lineup:
            updated = [p for p in home_lineup if p != player_out]
            if player_in and player_in not in updated:
                updated.append(player_in)
            if len(updated) > 5:
                updated = updated[:5]
            elif len(updated) < 5 and player_in:
                updated.append(player_in)
            return updated, away_lineup, True
        if player_out in away_lineup:
            updated = [p for p in away_lineup if p != player_out]
            if player_in and player_in not in updated:
                updated.append(player_in)
            if len(updated) > 5:
                updated = updated[:5]
            elif len(updated) < 5 and player_in:
                updated.append(player_in)
            return home_lineup, updated, True

    # if neither then just add player_in to incomplete lineups
    if player_in:
        if len(home_lineup) < 5 and player_in not in home_lineup:
            updated = (home_lineup + [player_in])[:5]
            return updated, away_lineup, False
        if len(away_lineup) < 5 and player_in not in away_lineup:
            updated = (away_lineup + [player_in])[:5]
            return home_lineup, updated, False

    return home_lineup, away_lineup, False


# turn player list  into a clean int list
def _normalize_player_list(player_list) -> List[int]:
    if player_list is None:
        return []
    if isinstance(player_list, str):
        try:
            player_list = ast.literal_eval(player_list)
        except (ValueError, SyntaxError):
            return []
    result: List[int] = []
    for item in player_list:
        try:
            if pd.isna(item):
                continue
        except (TypeError, ValueError):
            pass
        try:
            pid = int(float(item))
        except (ValueError, TypeError):
            continue
        if _is_valid_player_id(pid):
            result.append(pid)
    return result


 # adds possessions and net_rating per stint using a standard formula
def _compute_possessions_and_rating(stints_df: pd.DataFrame) -> pd.DataFrame:
    df = stints_df.copy()
    # quick possessions estimate based on fga, oreb, to, and fta
    raw_possessions = (
        df.get("fga", 0)
        - df.get("off_reb", 0)
        + df.get("turnovers", 0)
        + 0.44 * df.get("fta", 0)
    )
    df["possessions"] = (0.5 * raw_possessions).clip(lower=0.1)
    df["net_rating"] = (
        (df["team_pts"] - df["opp_pts"]) / df["possessions"].clip(lower=0.1)
    ) * 100
    return df


 # converts one game's play-by play into a list of stints
def build_stints_from_game(
    game_id: str,
    game_df: pd.DataFrame,
) -> Tuple[List[Dict], int, int, int]:
    game_df = game_df.reset_index(drop=True)
    if game_df.empty:
        return [], 0, 0, 0

    teams = game_df[game_df["teamTricode"].notna()]["teamTricode"].unique()
    if len(teams) < 2:
        return [], 0, 0, 0

    home_team = teams[0]
    away_team = teams[1]

    name_to_id = _build_name_to_id_mapping(game_df)
    home_lineup, away_lineup = _get_starting_lineup(game_df)
    home_lineup = [p for p in home_lineup if _is_valid_player_id(p)]
    away_lineup = [p for p in away_lineup if _is_valid_player_id(p)]

    stints: List[Dict] = []
    current_home = home_lineup.copy()
    current_away = away_lineup.copy()
    stint_start_idx = 0
    stint_stats = _init_stint_stats()

    latest_home_score = 0
    latest_away_score = 0
    stint_start_home_score = 0
    stint_start_away_score = 0

    total_subs = 0
    matched_subs = 0
    unmatched_subs = 0
    lineup_errors = 0

    for event_idx, row in game_df.iterrows():
        action_type = row.get("actionType")
        action_type_str = str(action_type).strip() if pd.notna(action_type) else ""

        if "scoreHome" in row.index and pd.notna(row.get("scoreHome")):
            try:
                latest_home_score = int(float(row.get("scoreHome", 0)))
            except (ValueError, TypeError):
                pass
        if "scoreAway" in row.index and pd.notna(row.get("scoreAway")):
            try:
                latest_away_score = int(float(row.get("scoreAway", 0)))
            except (ValueError, TypeError):
                pass

        if "Substitution" in action_type_str:
            total_subs += 1
            stint_stats["team_pts"] = max(latest_home_score - stint_start_home_score, 0)
            stint_stats["opp_pts"] = max(latest_away_score - stint_start_away_score, 0)

            stints.append(
                {
                    "GAME_ID": game_id,
                    "start_idx": stint_start_idx,
                    "end_idx": event_idx,
                    "home_players": current_home.copy(),
                    "away_players": current_away.copy(),
                    **stint_stats,
                }
            )

            player_out = _parse_player_id(row.get("personId"))
            description = str(row.get("description", ""))
            team = row.get("teamTricode")
            player_in = _parse_sub_description(description, name_to_id)

            updated_home, updated_away, matched = _update_lineup_for_sub(
                current_home,
                current_away,
                player_out,
                player_in,
                team,
                home_team,
            )

            if matched:
                matched_subs += 1
                current_home, current_away = updated_home, updated_away
            else:
                unmatched_subs += 1
                if len(updated_home) == 5 and len(updated_away) == 5:
                    current_home, current_away = updated_home, updated_away

            if len(current_home) != 5 or len(current_away) != 5:
                lineup_errors += 1
                if len(current_home) < 5:
                    recent = game_df.iloc[max(0, event_idx - 20) : event_idx + 1]
                    for _, r in recent.iterrows():
                        if r.get("teamTricode") == home_team:
                            cand = _parse_player_id(r.get("personId"))
                            if cand and cand not in current_home and _is_valid_player_id(cand):
                                current_home.append(cand)
                                if len(current_home) >= 5:
                                    current_home = current_home[:5]
                                    break
                if len(current_away) < 5:
                    recent = game_df.iloc[max(0, event_idx - 20) : event_idx + 1]
                    for _, r in recent.iterrows():
                        if r.get("teamTricode") == away_team:
                            cand = _parse_player_id(r.get("personId"))
                            if cand and cand not in current_away and _is_valid_player_id(cand):
                                current_away.append(cand)
                                if len(current_away) >= 5:
                                    current_away = current_away[:5]
                                    break

            stint_start_idx = event_idx + 1
            stint_start_home_score = latest_home_score
            stint_start_away_score = latest_away_score
            stint_stats = _init_stint_stats()
            latest_home_score = 0
            latest_away_score = 0
        else:
            _update_stint_stats(row, stint_stats)

    stint_stats["team_pts"] = max(latest_home_score - stint_start_home_score, 0)
    stint_stats["opp_pts"] = max(latest_away_score - stint_start_away_score, 0)

    stints.append(
        {
            "GAME_ID": game_id,
            "start_idx": stint_start_idx,
            "end_idx": len(game_df) - 1,
            "home_players": current_home.copy(),
            "away_players": current_away.copy(),
            **stint_stats,
        }
    )

    return stints, total_subs, matched_subs, unmatched_subs


# loops over all games in the pbp file and builds stints for each file
def build_stints_from_combined(pbp_df: pd.DataFrame) -> pd.DataFrame:
    ensure_dirs()
    pbp_df = pbp_df.copy().reset_index(drop=True)

    all_stints: List[Dict] = []
    total_subs = 0
    total_matched = 0
    total_unmatched = 0

    for game_id, game_df in pbp_df.groupby("GAME_ID"):
        stints, subs, matched, unmatched = build_stints_from_game(game_id, game_df)
        all_stints.extend(stints)
        total_subs += subs
        total_matched += matched
        total_unmatched += unmatched

        match_rate = (matched / subs * 100) if subs > 0 else 0.0
        logger.info(
            "Game %s: subs=%d matched=%d unmatched=%d (%.1f%% match rate)",
            game_id,
            subs,
            matched,
            unmatched,
            match_rate,
        )

        if subs > 0 and match_rate < 30:
            logger.warning(
                "Game %s has low substitution match rate (%.1f%%); lineups may be noisy",
                game_id,
                match_rate,
            )

    stints_df = pd.DataFrame(all_stints)
    if stints_df.empty:
        logger.warning("No stints created from play-by-play data")
        return stints_df

    unique_games = pbp_df["GAME_ID"].nunique()
    overall_match_rate = (total_matched / total_subs * 100) if total_subs > 0 else 0.0
    logger.info(
        "Stint summary: games=%d stints=%d subs=%d matched=%d unmatched=%d (%.1f%% match rate)",
        unique_games,
        len(stints_df),
        total_subs,
        total_matched,
        total_unmatched,
        overall_match_rate,
    )

    if not stints_df.empty:
        home_sizes = stints_df["home_players"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        away_sizes = stints_df["away_players"].apply(lambda x: len(x) if isinstance(x, list) else 0)
        incomplete_home = (home_sizes != 5).sum()
        incomplete_away = (away_sizes != 5).sum()
        total_incomplete = incomplete_home + incomplete_away
        if total_incomplete > 0:
            logger.warning(
                "Found %d stints with incomplete lineups (%.1f%% of total): %d home, %d away",
                total_incomplete,
                (total_incomplete / len(stints_df) * 100),
                incomplete_home,
                incomplete_away,
            )
        else:
            logger.info("All stints have complete 5‑player lineups.")

    stints_df = _compute_possessions_and_rating(stints_df)
    if "home_players" in stints_df.columns:
        stints_df["home_players"] = stints_df["home_players"].apply(_normalize_player_list)
    if "away_players" in stints_df.columns:
        stints_df["away_players"] = stints_df["away_players"].apply(_normalize_player_list)
    return stints_df


 # helper function that takes a combined pbp dataframe and returns stint rows
def build_stints_from_pbp(pbp_df: pd.DataFrame) -> pd.DataFrame:
    ensure_dirs()
    if "GAME_ID" in pbp_df.columns:
        pbp_df = pbp_df.copy()
        pbp_df["GAME_ID"] = pbp_df["GAME_ID"].apply(normalize_game_id).astype(str)
    stints_df = build_stints_from_combined(pbp_df)
    if not stints_df.empty and "GAME_ID" in stints_df.columns:
        stints_df = stints_df.copy()
        stints_df["stint_id"] = stints_df.groupby("GAME_ID").cumcount()
        if "period" not in stints_df.columns:
            stints_df["period"] = 1
    logger.info("Built %d stints from PBP data", len(stints_df))
    return stints_df


 # another helper functio nthat reads combined pbp from disk, builds stints, and then saves them
def build_season_stints() -> pd.DataFrame:
    ensure_dirs()
    if not config.PBP_COMBINED.exists():
        logger.error(
            "Combined PBP not found at %s — run fetch_pbp_for_season() first",
            config.PBP_COMBINED,
        )
        return pd.DataFrame()

    logger.info("Loading combined PBP from %s", config.PBP_COMBINED)
    pbp = pd.read_csv(config.PBP_COMBINED)
    stints_df = build_stints_from_pbp(pbp)

    if not stints_df.empty:
        config.PROCESSED_STINTS_DIR.mkdir(parents=True, exist_ok=True)
        stints_file = config.PROCESSED_STINTS_DIR / "stints.csv"
        stints_df.to_csv(stints_file, index=False)
        logger.info("Saved stints to %s (%d rows)", stints_file, len(stints_df))
    else:
        logger.warning("No stints created")

    return stints_df


""" sparse matrixs building """


 # turning stints into sparse design matrices + weights
def build_sparse_matrices(
    stints_df: pd.DataFrame,
) -> Tuple[csr_matrix, np.ndarray, csr_matrix, np.ndarray, np.ndarray]:
    ensure_dirs()
    logger.info("Building sparse matrices from %d stints", len(stints_df))

    (
        X_off,
        X_def,
        y_off,
        y_def,  
        player_to_col,
        all_players,
        player_to_representative,
        sample_weights,
    ) = _create_sparse_matrices(stints_df)

    logger.info(
        "Built sparse matrices: X_off %s, X_def %s, %d players",
        X_off.shape,
        X_def.shape,
        len(player_to_col),
    )

    # this keeps the existing on-disk format exactly the same.
    _save_matrices(X_off, y_off, X_def, y_def, player_to_col)
    return X_off, y_off, X_def, y_def, sample_weights


def _save_matrices(
    X_off: csr_matrix,
    y_off: np.ndarray,
    X_def: csr_matrix,
    y_def: np.ndarray,
    player_to_col: dict,
    output_dir: Path | None = None,
) -> None:
    from scipy.sparse import save_npz

    if output_dir is None:
        output_dir = config.MATRICES_DIR

    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_npz(output_dir / "X_off.npz", X_off)
    save_npz(output_dir / "X_def.npz", X_def)
    np.save(output_dir / "y_off.npy", y_off)
    np.save(output_dir / "y_def.npy", y_def)

    mapping_path = output_dir / "player_to_col.json"
    with mapping_path.open("w") as f:
        json.dump({str(k): int(v) for k, v in player_to_col.items()}, f, indent=2)

    logger.info("Saved sparse matrices and mapping to %s", output_dir)


__all__ = [
    "fetch_pbp_for_season",
    "create_all_games_index",
    "build_stints_from_pbp",
    "build_season_stints",
    "build_sparse_matrices",
]


