# this file includeslogging, checkpoints, directories, and a retry decorator
from pathlib import Path
import json
import logging
import time
from functools import wraps
from typing import Callable

import requests_cache

from . import config


# make sure the folders the pipeline is expecting actually exist
def ensure_dirs():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.PBP_RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_STINTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MATRICES_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# sets up a logger that writes both to file and to stdout
def setup_logging(log_file: Path = None, level=logging.INFO):
    ensure_dirs()
    if log_file is None:
        log_file = config.INGEST_LOG
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("rapm")


# read the last-saved checkpoint json
def load_checkpoint():
    if not config.CHECKPOINT.exists():
        return {}
    try:
        return json.loads(config.CHECKPOINT.read_text())
    except Exception:
        return {}


# dumps a small checkpoint dict to disk so can resume later
def save_checkpoint(obj: dict):
    config.CHECKPOINT.write_text(json.dumps(obj, indent=2))


# creates a cached http session so repeated nba_api calls hit sqlite instead
def requests_session(cache_name: str = "nba_cache", expire_after: int = 3600):
    cache_path = config.DATA_DIR / f"{cache_name}.sqlite"
    session = requests_cache.CachedSession(str(cache_path), expire_after=expire_after)
    return session


# retry decorator for flaky network calls
def retry(exceptions, tries=5, delay=1.0, backoff=2.0):
    def deco(f: Callable):
        @wraps(f)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    time.sleep(_delay)
                    _tries -= 1
                    _delay *= backoff
            # the final attempt
            return f(*args, **kwargs)
        return wrapper
    return deco


# normalize the game ids into a consistent '002yyyyggg' string
def normalize_game_id(game_id) -> str:
    if game_id is None:
        return None
    
    gid_str = str(game_id).strip()    
    if gid_str.startswith('002') and len(gid_str) == 10:
        return gid_str
    
    numeric_part = ''.join(c for c in gid_str if c.isdigit())
    
    if not numeric_part:
        return gid_str
    
    # if length is weird fix it
    if gid_str.startswith('002'):
        after_prefix = numeric_part[3:] if len(numeric_part) > 3 else numeric_part
        if len(after_prefix) == 7:
            return f'002{after_prefix}'
        elif len(numeric_part) == 10:
            return f'002{numeric_part[3:]}'
    
    #  10-digit id starting with 002
    if len(numeric_part) == 10 and numeric_part.startswith('002'):
        return numeric_part
    
    if len(numeric_part) == 10 and numeric_part.startswith('222'):
        return f'002{numeric_part[1:]}'
    
    # if we have 7-9 digits add the 002 prefix
    if 7 <= len(numeric_part) <= 9:
        padded = numeric_part.zfill(7)
        return f'002{padded}'
    
    # 8 digits where the season part the it's missing a zero
    if len(numeric_part) == 8:
        if numeric_part.startswith('22'):
            return f'002{numeric_part[1:]}'
        else:
            return f'002{numeric_part.zfill(7)}'
    
    # grab the last 7 digits and throw 002 in front
    if len(numeric_part) >= 7:
        return f'002{numeric_part[-7:]}'
    
    return gid_str


__all__ = [
    "ensure_dirs",
    "setup_logging",
    "load_checkpoint",
    "save_checkpoint",
    "requests_session",
    "retry",
    "normalize_game_id",
]
