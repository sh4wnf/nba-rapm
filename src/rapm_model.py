# this file builds sparse matrices, solves ridge regression, and adds bootstrap functionality

from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import ast

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, identity, hstack
from scipy.sparse.linalg import spsolve

from . import config
from .utils import setup_logging

logger = setup_logging()


""" the core helper functions """

# used to filter out obviously bad values
MIN_PLAYER_ID = 1
MAX_PLAYER_ID = 200_000_000


# range and type check for player ids
def is_valid_player_id(player_id) -> bool:
    try:
        pid = int(player_id)
        return MIN_PLAYER_ID <= pid < MAX_PLAYER_ID
    except (TypeError, ValueError):
        return False


# normalize the home players and away players  columns on the stints dataframe
def clean_player_lists(stints_df: pd.DataFrame) -> None:
    for col in ("home_players", "away_players"):
        if col not in stints_df.columns:
            logger.warning("Column %s not found in stints DataFrame", col)
            continue

        if stints_df[col].dtype == object:
            stints_df[col] = stints_df[col].apply(
                lambda s: ast.literal_eval(s) if isinstance(s, str) else s
            )

        stints_df[col] = stints_df[col].apply(
            lambda lst: [int(p) for p in (lst or []) if is_valid_player_id(p)]
        )


# making sure we have a clean not negative possessions column
def prepare_possessions(stints_df: pd.DataFrame) -> pd.DataFrame:
    df = stints_df.copy()

    if "possessions" in df.columns:
        df["possessions"] = df["possessions"].clip(lower=0.0)
    else:
        raw_possessions = (
            df.get("fga", 0)
            - df.get("off_reb", 0)
            + df.get("turnovers", 0)
            + 0.44 * df.get("fta", 0)
        )
        df["possessions"] = (0.5 * raw_possessions).fillna(0.0).clip(lower=0.0)

    return df


# dropping super tiny stints so the regression isn't dominated by noise
def filter_small_stints(
    stints_df: pd.DataFrame,
    min_possessions: float,
) -> pd.DataFrame:
    if min_possessions is None or min_possessions <= 0:
        return stints_df

    before = len(stints_df)
    filtered = stints_df[stints_df["possessions"] >= min_possessions].copy()
    after = len(filtered)

    if before != after:
        logger.info(
            "Filtered small stints: before=%d after=%d (min_possessions=%.1f)",
            before,
            after,
            min_possessions,
        )
    return filtered


# grabs the sorted list of all players that appear in any stint
def get_all_players(stints_df: pd.DataFrame) -> List[int]:
    home_players = set(p for lst in stints_df["home_players"] for p in lst)
    away_players = set(p for lst in stints_df["away_players"] for p in lst)
    all_players = sorted(home_players | away_players)
    logger.debug("Found %d unique players", len(all_players))
    return all_players


# builds a lookup that mapsplayer_id to which stint indices they played in
def build_player_to_stints_mapping(stints_df: pd.DataFrame) -> Dict[int, List[int]]:
    player_to_stints: Dict[int, List[int]] = defaultdict(list)

    for stint_idx, (home_players, away_players) in enumerate(
        zip(stints_df["home_players"], stints_df["away_players"])
    ):
        for player_id in home_players:
            player_to_stints[player_id].append(stint_idx)
        for player_id in away_players:
            player_to_stints[player_id].append(stint_idx)

    return dict(player_to_stints)


# to detect multicollinearity this group players who always appear in exactly the same set of stints
def detect_multicollinearity(
    player_to_stints: Dict[int, List[int]]
) -> Tuple[Dict[Tuple[int, ...], List[int]], Dict[int, int]]:
    pattern_map: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
    for player_id, stint_indices in player_to_stints.items():
        pattern_key = tuple(sorted(stint_indices))
        pattern_map[pattern_key].append(player_id)

    multicollinear_groups = [g for g in pattern_map.values() if len(g) > 1]
    if multicollinear_groups:
        logger.warning(
            "Found %d groups of players with identical appearance patterns "
            "(perfect multicollinearity)",
            len(multicollinear_groups),
        )
        for group in multicollinear_groups[:10]:
            logger.warning("  Example group (size=%d): %s", len(group), group[:8])

    player_to_representative: Dict[int, int] = {}
    for pattern_key, player_group in pattern_map.items():
        representative = player_group[0]
        for pid in player_group:
            player_to_representative[pid] = representative

    return dict(pattern_map), player_to_representative


# count how many appearnces each player has
def count_player_appearances(stints_df: pd.DataFrame) -> Counter:
    appearance_counts: Counter = Counter()
    for home_players, away_players in zip(
        stints_df["home_players"], stints_df["away_players"]
    ):
        appearance_counts.update(home_players)
        appearance_counts.update(away_players)
    return appearance_counts


# the ridge regression solver, includes optional sample weights per stint
def ridge_regression(
    X: csr_matrix,
    y: np.ndarray,
    lam: float,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    if sample_weights is not None:
        
        # turn weights into sqrt(w) so it can scale therows of x and  y directly
        w = np.sqrt(np.asarray(sample_weights, dtype=float)).reshape(-1, 1)
        # for sparse matrices, use elementwise multiply along rows
        X_w = X.multiply(w)
        y_w = y * w.ravel()
    else:
        X_w = X
        y_w = y
    
    # compute x^t x and x^t y
    XtX = X_w.T @ X_w
    XtY = X_w.T @ y_w
    
    # adding ridge penalty on the diagonal: (x^t x + λi)
    n_players = X.shape[1]
    A = XtX + lam * identity(n_players, format="csr")
    
    # solves the linear system for beta coefficients
    beta = spsolve(A, XtY)
    
    logger.debug("Solved ridge regression with lambda=%.1f", lam)
    
    return beta


# this function fits one joint model that learns offensive + defensive rapm at the same time
def fit_joint_orapm_drapm(
    X_off: csr_matrix,
    X_def: csr_matrix,
    y: np.ndarray,
    lam: float,
    sample_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # check if shapes line up
    n_stints, n_players = X_off.shape
    assert X_def.shape == X_off.shape
    assert y.shape[0] == n_stints

    # builds the joint design matrix:
    #   y = x_off * beta_off  -  x_def * beta_def
    # so x_joint = [x_off, -x_def]
    X_joint = hstack([X_off, -X_def], format="csr")

    # applying sample weights by scaling rows
    if sample_weights is not None:
        w = np.asarray(sample_weights).reshape(-1)
        assert w.shape[0] == n_stints
        w_sqrt = np.sqrt(w)

        # then multiply each row of x_joint by sqrt(weight)
        Xw = X_joint.multiply(w_sqrt[:, None])
        yw = y * w_sqrt
    else:
        Xw = X_joint
        yw = y

    # ridge regression - (x^t x + λi) β = x^t y
    n_params = 2 * n_players
    lamI = lam * np.eye(n_params)

    XtX = Xw.T @ Xw
    XtX = XtX.toarray() if hasattr(XtX, "toarray") else XtX
    XtX = XtX + lamI

    Xty = Xw.T @ yw
    Xty = np.asarray(Xty).reshape(-1)

    coeffs = np.linalg.solve(XtX, Xty)

    # splits the big coefficient vector back into offense and defense parts
    beta_off = coeffs[:n_players] # offensive rapm
    beta_def = coeffs[n_players:] # defensive rapm

    # total rapm per player is just adding offense + defense
    rapm_tot = beta_off + beta_def

    # center the total rapm so league-average is 0
    rapm_tot = rapm_tot - rapm_tot.mean()

    return beta_off, beta_def, rapm_tot


# this function builds the sparse design matrix where each row = stint and cols = players on offense
def build_offensive_design_matrix(
    stints_df: pd.DataFrame,
    player_to_col: Dict[int, int]
) -> csr_matrix:
    row_indices = []
    col_indices = []
    values = []
    
    for stint_idx, (home_players, away_players) in enumerate(
        zip(stints_df["home_players"], stints_df["away_players"])
    ):
        # home team players or the offense in stint show up as +1
        for player_id in home_players:
            col_idx = player_to_col.get(player_id)
            if col_idx is not None:
                row_indices.append(stint_idx)
                col_indices.append(col_idx)
                values.append(+1.0)
    
    # builds the sparse matrix in coo then converts it to csr that way it can faster math
    n_stints = len(stints_df)
    n_players = len(player_to_col)
    X = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_stints, n_players)
    ).tocsr()
    
    logger.info("Offensive design matrix shape: %s | non-zero elements: %d", X.shape, X.nnz)
    
    return X


# this function builds the sparse design matrix where each row = stint and cols = players on defense
def build_defensive_design_matrix(
    stints_df: pd.DataFrame,
    player_to_col: Dict[int, int]
) -> csr_matrix:
    row_indices = []
    col_indices = []
    values = []
    
    for stint_idx, (home_players, away_players) in enumerate(
        zip(stints_df["home_players"], stints_df["away_players"])
    ):
        # away team players or the defense in stint show up as +1
        for player_id in away_players:
            col_idx = player_to_col.get(player_id)
            if col_idx is not None:
                row_indices.append(stint_idx)
                col_indices.append(col_idx)
                values.append(+1.0)
    
    # builds the sparse matrix in coo then converts it to csr that way it can faster math
    n_stints = len(stints_df)
    n_players = len(player_to_col)
    X = coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_stints, n_players)
    ).tocsr()
    
    logger.info("Defensive design matrix shape: %s | non-zero elements: %d", X.shape, X.nnz)
    
    return X


# checks that the offense and defense matrices are not identical
def _check_off_def_difference(X_off: csr_matrix, X_def: csr_matrix, logger) -> int:
    if X_off.shape != X_def.shape:
        logger.warning(
            "X_off and X_def shapes differ: off=%s, def=%s",
            X_off.shape,
            X_def.shape,
        )
        return -1
    
    diff_nnz = (X_off != X_def).nnz
    logger.info("Off/Def matrix difference nnz = %d", diff_nnz)
    if diff_nnz == 0:
        logger.warning("X_off and X_def are IDENTICAL – this will force ORAPM == DRAPM.")
    return diff_nnz


# runs ridge regression on the offensive design matrix to get orapm
def compute_orapm(
    X_off: csr_matrix,
    y_off: np.ndarray,
    lam: float = config.DEFAULT_LAMBDA,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    logger.info("Computing ORAPM with lambda=%.1f", lam)
    beta = ridge_regression(X_off, y_off, lam, sample_weights=sample_weights)
    logger.info("ORAPM computation complete")
    return beta


# runs ridge regression on the defensive design matrix to find the defensive rapm
def compute_drapm(
    X_def: csr_matrix,
    y_def: np.ndarray,
    lam: float = config.DEFAULT_LAMBDA,
    sample_weights: np.ndarray | None = None,
) -> np.ndarray:
    logger.info("Computing DRAPM with lambda=%.1f", lam)
    beta = ridge_regression(X_def, y_def, lam, sample_weights=sample_weights)
    logger.info("DRAPM computation complete")
    return beta


# combine orapm + drapm into a single total metric
def combine_rapm(ora: np.ndarray, dra: np.ndarray) -> np.ndarray:
    if len(ora) != len(dra):
        raise ValueError(f"ORAPM and DRAPM vectors must have same length: {len(ora)} vs {len(dra)}")
    
    total = ora.copy()
    logger.info("Using ORAPM as total RAPM (DRAPM kept as a separate column only)")
    return total


# computes rapm by fitting the joint orapm + drapm model while also logging
def compute_rapm(
    X_off: csr_matrix,
    y: np.ndarray,
    X_def: csr_matrix,
    lam: float,
    sample_weights: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    logger.info("computing joint orapm + drapm with lambda=%.1f", lam)
    orapm, drapm, rapm = fit_joint_orapm_drapm(
        X_off=X_off,
        X_def=X_def,
        y=y,
        lam=lam,
        sample_weights=sample_weights,
    )
    logger.info("rapm computation complete (joint orapm + drapm)")
    return orapm, drapm, rapm


# this function goes from stint dataframe to design matrices and rapm target
def create_sparse_matrices(
    stints_df: pd.DataFrame,
    min_possessions: float = config.DEFAULT_MIN_POSSESSIONS,
    net_cap: float = config.DEFAULT_NET_CAP
) -> Tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, Dict[int, int], List[int], Dict[int, int]]:
    
    # cleans up player lists, possessions and drops tiny stints
    clean_player_lists(stints_df)
    stints_df = prepare_possessions(stints_df)
    stints_df = filter_small_stints(stints_df, min_possessions)
    
    if stints_df.empty:
        raise ValueError("No stints remaining after filtering")
    
    # gets all players and how often their together
    all_players = get_all_players(stints_df)
    player_to_stints = build_player_to_stints_mapping(stints_df)
    pattern_map, player_to_representative = detect_multicollinearity(player_to_stints)
    
    representatives = sorted(set(player_to_representative.values()))
    representative_to_col = {rep: idx for idx, rep in enumerate(representatives)}
    
    # this matchs every raw player_id to a column index through its representative
    all_players_to_col = {}
    for player_id, rep_id in player_to_representative.items():
        if rep_id in representative_to_col:
            all_players_to_col[player_id] = representative_to_col[rep_id]
    
    # builds the separate offensive and defensive design matrices
    X_off = build_offensive_design_matrix(stints_df, all_players_to_col)
    X_def = build_defensive_design_matrix(stints_df, all_players_to_col)
    diff_nnz = _check_off_def_difference(X_off, X_def, logger)
    assert diff_nnz > 0, "X_off and X_def are still identical; offensive/defensive models are not separated."
    
    player_to_col = representative_to_col
    
    # creates the scalar net_rating target for both orapm and drapm
    #while looking for precomputed net_rating if not derive it from points and possessions
    if "net_rating" in stints_df.columns:
        y = stints_df["net_rating"].astype(float).values
        poss = stints_df["possessions"].astype(float).values
    else:
        poss = stints_df["possessions"].astype(float).values
        if "team_pts" in stints_df.columns and "opp_pts" in stints_df.columns:
            pts_for = stints_df["team_pts"].astype(float).values
            pts_against = stints_df["opp_pts"].astype(float).values
        else:
            pts_for = stints_df["points_for"].astype(float).values
            pts_against = stints_df["points_against"].astype(float).values
        safe_poss = np.where(poss > 0, poss, 1.0)
        y = 100.0 * (pts_for - pts_against) / safe_poss
    
    # make sure target is float and clamps it to a reasonable nba like range
    y = y.astype(float)
    y = np.clip(y, -50.0, 50.0)
    
    # stores the rapm target back onto stints so that diagnostics can use it again
    stints_df["net_rating_rapm"] = y
    
    # use capped possessions as sample weights so longer stints matter more
    sample_weights = np.clip(poss.astype(float), 1.0, 20.0)
    
    # drops saturated stints that reach the ±50 cap as they're very noisy
    saturated = np.isclose(y, 50.0) | np.isclose(y, -50.0)
    keep_mask = ~saturated
    num_saturated = int(saturated.sum())
    
    if keep_mask.sum() == 0:
        raise ValueError("All stints are saturated at ±50; cannot fit RAPM.")
    
    # apply mask to stints, matrices, targets, and weights
    stints_df = stints_df.loc[keep_mask].reset_index(drop=True)
    X_off = X_off[keep_mask]
    X_def = X_def[keep_mask]
    y = y[keep_mask]
    sample_weights = sample_weights[keep_mask]
    
    logger.info("created sparse matrices: %d stints, %d players", len(stints_df), len(player_to_col))
    
    # log basic target stats to check scaling and filtering
    try:
        logger.info(
            "target stats (filtered stints used for rapm) — net_rating_rapm: "
            "mean=%.2f std=%.2f min=%.2f max=%.2f (after removing %d saturated stints)",
            float(y.mean()),
            float(y.std()),
            float(y.min()),
            float(y.max()),
            num_saturated,
        )
    except Exception:
        pass
    
    return (
        X_off,                  # offensive design matrix
        X_def,                  # defensive design matrix
        y,                      # RAPM target (net_rating_rapm)
        sample_weights,         
        player_to_col,          # Representative player mapping to column
        all_players,            # all of the raw player IDs that are seen in stints
        player_to_representative,  
        sample_weights,         
    )


def compute_full_rapm(
    stints_df: pd.DataFrame,
    lam: float = config.DEFAULT_LAMBDA,
    min_possessions: float = config.DEFAULT_MIN_POSSESSIONS,
    net_cap: float = config.DEFAULT_NET_CAP,
) -> pd.DataFrame:
    (
        X_off,
        X_def,
        y_rapm,
        sample_weights,
        player_to_col,
        all_players,
        player_to_representative,
        _,
    ) = create_sparse_matrices(stints_df, min_possessions, net_cap)

    ora_beta, dra_beta, total_beta = fit_joint_orapm_drapm(
        X_off=X_off,
        X_def=X_def,
        y=y_rapm,
        lam=lam,
        sample_weights=sample_weights,
    )

    appearance_counts = count_player_appearances(stints_df)

    results = []
    for player_id in all_players:
        try:
            player_id = int(float(player_id))
        except (ValueError, TypeError):
            logger.warning("Invalid player_id type: %s (type: %s)", player_id, type(player_id))
            continue

        representative = player_to_representative.get(player_id, player_id)
        col_idx = player_to_col.get(representative)
        if col_idx is None:
            logger.warning("Player %d has no column index (representative: %d)", player_id, representative)
            continue

        orapm_value = float(ora_beta[col_idx])
        drapm_value = float(dra_beta[col_idx])
        rapm_value = float(total_beta[col_idx])
        appearances = int(appearance_counts.get(player_id, 0))

        results.append(
            {
                "player_id": player_id,
                "orapm": orapm_value,
                "drapm": drapm_value,
                "rapm": rapm_value,
                "stint_appearances": appearances,
            }
        )

    rapm_df = pd.DataFrame(results)
    if not rapm_df.empty and "player_id" in rapm_df.columns:
        rapm_df["player_id"] = rapm_df["player_id"].astype(int)

    rapm_df = add_player_names(rapm_df, validate=True)
    rapm_df = rapm_df[["player_id", "player", "orapm", "drapm", "rapm", "stint_appearances"]]
    return rapm_df


# player metadata, bootstraping, and experiment logging

from typing import Optional, Set  
from pathlib import Path  
import unicodedata 
import ast 

_PLAYER_METADATA_CACHE: Optional[Dict[int, Dict]] = None
_PLAYER_ID_TO_NAME_CACHE: Optional[Dict[int, str]] = None


 # normalize names by turning them into ascii so matching becomes easier
def _normalize_name_for_match(name: str) -> str:
    if not isinstance(name, str):
        return ""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_bytes = normalized.encode("ascii", "ignore")
    return ascii_bytes.decode("ascii").lower()


# hits nba_api.stats once and keeps a cache of player info by id
def load_player_metadata(force_reload: bool = False) -> Dict[int, Dict]:
    global _PLAYER_METADATA_CACHE
    if _PLAYER_METADATA_CACHE is not None and not force_reload:
        return _PLAYER_METADATA_CACHE
    try:
        from nba_api.stats.static import players as static_players

        player_list = static_players.get_players()
        meta: Dict[int, Dict] = {}
        for p in player_list:
            pid = int(p["id"])
            meta[pid] = {
                "id": pid,
                "full_name": p.get("full_name", ""),
                "first_name": p.get("first_name", ""),
                "last_name": p.get("last_name", ""),
                "is_active": p.get("is_active", False),
            }
        _PLAYER_METADATA_CACHE = meta
        logger.info("Loaded player metadata for %d players", len(meta))
        return meta
    except Exception as e:
        logger.error("Failed to load player metadata: %s", e, exc_info=True)
        _PLAYER_METADATA_CACHE = {}
        return _PLAYER_METADATA_CACHE


# just returns player_id -> full_name
def get_player_id_to_name(force_reload: bool = False) -> Dict[int, str]:
    global _PLAYER_ID_TO_NAME_CACHE
    if _PLAYER_ID_TO_NAME_CACHE is not None and not force_reload:
        return _PLAYER_ID_TO_NAME_CACHE
    meta = load_player_metadata(force_reload=force_reload)
    _PLAYER_ID_TO_NAME_CACHE = {pid: info["full_name"] for pid, info in meta.items()}
    return _PLAYER_ID_TO_NAME_CACHE


# attach a playerss full name column to any dataframe with player_id
def add_player_names(rapm_df: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
    df = rapm_df.copy()
    if "player_id" in df.columns:
        df["player_id"] = df["player_id"].astype(int)

    id_to_name = get_player_id_to_name()
    df["player"] = df["player_id"].map(id_to_name)

    if validate:
        df["player_mapped"] = df["player"].notna()
        unmapped = df[~df["player_mapped"]]
        if not unmapped.empty:
            unmapped_ids = unmapped["player_id"].unique().tolist()
            logger.warning(
                "Found %d unmapped player IDs in RAPM results: %s",
                len(unmapped_ids),
                unmapped_ids[:20],
            )
            df.loc[~df["player_mapped"], "player"] = df.loc[
                ~df["player_mapped"], "player_id"
            ].astype(str)
        else:
            logger.info("All %d players in RAPM results have name mappings", len(df))

    df["player"] = df["player"].fillna(df["player_id"].astype(str))
    return df


# estimate the minutes for each player using stint possessions
def compute_player_minutes_from_stints(stints_df: pd.DataFrame) -> pd.DataFrame:
    df = stints_df.copy()
    for col in ("home_players", "away_players"):
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x)
                if isinstance(x, str)
                else (x if isinstance(x, list) else [])
            )

    player_stats: Dict[int, Dict[str, float]] = {}
    for _, row in df.iterrows():
        possessions = float(row.get("possessions", 0.0))
        minutes = (possessions / 100.0) * 48.0
        home_players = row.get("home_players", []) or []
        away_players = row.get("away_players", []) or []
        for pid in list(home_players) + list(away_players):
            pid = int(pid)
            stats = player_stats.setdefault(
                pid, {"stints": 0, "possessions": 0.0, "minutes": 0.0}
            )
            stats["stints"] += 1
            stats["possessions"] += possessions
            stats["minutes"] += minutes

    rows = []
    for pid, stats in player_stats.items():
        rows.append(
            {
                "player_id": pid,
                "stints_played": stats["stints"],
                "total_possessions": stats["possessions"],
                "estimated_minutes": stats["minutes"],
            }
        )
    minutes_df = pd.DataFrame(rows)
    if not minutes_df.empty:
        minutes_df = add_player_names(minutes_df, validate=False)
    return minutes_df.sort_values("stints_played", ascending=False)


# sample stints with replacement to help with bootstrapping
def bootstrap_resample(stints_df: pd.DataFrame) -> pd.DataFrame:
    n = len(stints_df)
    idx = np.random.choice(n, size=n, replace=True)
    return stints_df.iloc[idx].reset_index(drop=True)


# run rapm many times on resampled stints to get uncertainty estimates
def bootstrap_rapm_full(
    stints_df: pd.DataFrame,
    lam: float = config.DEFAULT_LAMBDA,
    num_bootstrap: int = config.DEFAULT_BOOTSTRAP_ITERATIONS,
    min_possessions: float = config.DEFAULT_MIN_POSSESSIONS,
    net_cap: float = config.DEFAULT_NET_CAP,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    if seed is not None:
        np.random.seed(seed)

    logger.info("starting bootstrap rapm with %d iterations", num_bootstrap)

    clean_player_lists(stints_df)
    stints_df = prepare_possessions(stints_df)
    all_players = get_all_players(stints_df)

    bootstrap_orapm: List[np.ndarray] = []
    bootstrap_drapm: List[np.ndarray] = []
    bootstrap_total: List[np.ndarray] = []

    # base matricesst to lock player ordering
    create_sparse_matrices(stints_df, min_possessions, net_cap)

    for b in range(num_bootstrap):
        if (b + 1) % 50 == 0:
            logger.info("Bootstrap iteration %d / %d", b + 1, num_bootstrap)
        try:
            resampled = bootstrap_resample(stints_df)
            (
                X_off,
                X_def,
                y_boot,
                sample_weights,
                player_to_col,
                _,
                player_to_rep,
                _,
            ) = create_sparse_matrices(resampled, min_possessions, net_cap)

            ora_beta, dra_beta, total_beta = fit_joint_orapm_drapm(
                X_off=X_off,
                X_def=X_def,
                y=y_boot,
                lam=lam,
                sample_weights=sample_weights,
            )

            rep_to_col = {rep: idx for idx, rep in enumerate(sorted(set(player_to_rep.values())))}

            ora_all = np.zeros(len(all_players))
            dra_all = np.zeros(len(all_players))
            rap_all = np.zeros(len(all_players))
            for idx, pid in enumerate(all_players):
                rep = player_to_rep.get(pid, pid)
                col_idx = rep_to_col.get(rep)
                if col_idx is not None and col_idx < len(ora_beta):
                    ora_all[idx] = ora_beta[col_idx]
                    dra_all[idx] = dra_beta[col_idx]
                    rap_all[idx] = total_beta[col_idx]

            bootstrap_orapm.append(ora_all)
            bootstrap_drapm.append(dra_all)
            bootstrap_total.append(rap_all)
        except Exception as e:
            logger.warning("Bootstrap iteration %d failed: %s", b + 1, e)
            if bootstrap_orapm:
                bootstrap_orapm.append(bootstrap_orapm[-1])
                bootstrap_drapm.append(bootstrap_drapm[-1])
                bootstrap_total.append(bootstrap_total[-1])
            else:
                bootstrap_orapm.append(np.zeros(len(all_players)))
                bootstrap_drapm.append(np.zeros(len(all_players)))
                bootstrap_total.append(np.zeros(len(all_players)))

    bootstrap_orapm = np.array(bootstrap_orapm)
    bootstrap_drapm = np.array(bootstrap_drapm)
    bootstrap_total = np.array(bootstrap_total)

    logger.info("bootstrap complete, now computing confidence intervals")

    orapm_mean = np.mean(bootstrap_orapm, axis=0)
    drapm_mean = np.mean(bootstrap_drapm, axis=0)
    rapm_mean = np.mean(bootstrap_total, axis=0)

    rapm_low = np.percentile(bootstrap_total, 2.5, axis=0)
    rapm_high = np.percentile(bootstrap_total, 97.5, axis=0)

    appearances = count_player_appearances(stints_df)
    rows = []
    for idx, pid in enumerate(all_players):
        rows.append(
            {
                "player_id": pid,
                "orapm": float(orapm_mean[idx]),
                "drapm": float(drapm_mean[idx]),
                "rapm": float(rapm_mean[idx]),
                "ci_low": float(rapm_low[idx]),
                "ci_high": float(rapm_high[idx]),
                "stint_appearances": int(appearances.get(pid, 0)),
            }
        )
    rapm_df_bs = pd.DataFrame(rows)
    rapm_df_bs = add_player_names(rapm_df_bs, validate=True)
    rapm_df_bs = rapm_df_bs[
        [
            "player_id",
            "player",
            "orapm",
            "drapm",
            "rapm",
            "ci_low",
            "ci_high",
            "stint_appearances",
        ]
    ]

    bootstrap_samples = {
        "orapm": bootstrap_orapm,
        "drapm": bootstrap_drapm,
        "rapm": bootstrap_total,
    }
    logger.info("bootstrap rapm complete for %d players", len(rapm_df_bs))
    return rapm_df_bs, bootstrap_samples


# for getting a summary of some values
def compute_summary_stats(values: list) -> Dict:
    if not values:
        return {}
    arr = np.array(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "count": int(len(arr)),
    }


# this function creates params and results so that we can log for experiments
def create_experiment_record(
    lambda_reg: float,
    num_stints: int,
    num_players: int,
    runtime_seconds: float,
    orapm_stats: Optional[Dict] = None,
    drapm_stats: Optional[Dict] = None,
    rapm_stats: Optional[Dict] = None,
    bootstrap_iterations: Optional[int] = None,
    season: Optional[str] = None,
    train_test_split: Optional[float] = None,
    **kwargs,
) -> tuple[Dict, Dict]:
    parameters = {
        "lambda": lambda_reg,
        "season": season or config.SEASON,
        "bootstrap_iterations": bootstrap_iterations,
        "train_test_split": train_test_split,
        **{k: v for k, v in kwargs.items() if k.startswith("param_")},
    }
    results = {
        "num_stints": num_stints,
        "num_players": num_players,
        "runtime_seconds": runtime_seconds,
        "runtime_minutes": runtime_seconds / 60.0,
    }
    if orapm_stats:
        results.update({f"orapm_{k}": v for k, v in orapm_stats.items()})
    if drapm_stats:
        results.update({f"drapm_{k}": v for k, v in drapm_stats.items()})
    if rapm_stats:
        results.update({f"rapm_{k}": v for k, v in rapm_stats.items()})
    results.update({k: v for k, v in kwargs.items() if not k.startswith("param_")})
    return parameters, results


# adds an experiment record to a json or csv log on on local disk
def log_experiment(
    parameters: Dict,
    results: Dict,
    log_path: Optional[Path] = None,
    fmt: str = "json",
) -> Path:
    config.EXPERIMENT_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_path = config.EXPERIMENT_LOGS_DIR / f"log.{fmt}"

    import json as _json

    record = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "parameters": parameters,
        "results": results,
    }

    if fmt.lower() == "json":
        if log_path.exists():
            try:
                existing = _json.loads(log_path.read_text())
                if not isinstance(existing, list):
                    existing = [existing]
            except Exception:
                existing = []
        else:
            existing = []
        existing.append(record)
        log_path.write_text(_json.dumps(existing, indent=2))
    elif fmt.lower() == "csv":
        flat = {
            "timestamp": record["timestamp"],
            **{f"param_{k}": v for k, v in parameters.items()},
            **{f"result_{k}": v for k, v in results.items()},
        }
        if log_path.exists():
            df = pd.read_csv(log_path)
            df = pd.concat([df, pd.DataFrame([flat])], ignore_index=True)
        else:
            df = pd.DataFrame([flat])
        df.to_csv(log_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    logger.info("Logged experiment to %s", log_path)
    return log_path


# summary of the best and worst rapm players from the logs
def print_rapm_summary(rapm_df: pd.DataFrame, min_stints: int = 0, top_n: int = 20) -> None:
    if rapm_df.empty:
        logger.warning("RAPM summary requested on empty DataFrame")
        return
    df = rapm_df.copy()
    if "stint_appearances" in df.columns and min_stints > 0:
        df = df[df["stint_appearances"] >= min_stints]
    if df.empty:
        logger.warning("No players meet min_stints=%d for RAPM summary", min_stints)
        return
    top = df.nlargest(top_n, "rapm")
    bottom = df.nsmallest(top_n, "rapm")
    logger.info(
        "Top %d RAPM players:\n%s",
        top_n,
        top[["player", "rapm", "stint_appearances"]].to_string(index=False),
    )
    logger.info(
        "Bottom %d RAPM players:\n%s",
        top_n,
        bottom[["player", "rapm", "stint_appearances"]].to_string(index=False),
    )


 # loader for the streamlit app to get clean rapm results
def load_rapm_results_with_stats(csv_path: Path | None = None) -> pd.DataFrame:
    if csv_path is None:
        primary_path = config.RAPM_OUTPUTS_CSV
    else:
        primary_path = Path(csv_path)

    demo_path = config.RESULTS_DIR / "rapm_demo.csv"

    if primary_path.exists():
        df = pd.read_csv(primary_path)
    elif demo_path.exists():
        logger.warning(
            "Primary RAPM results file not found at %s. Falling back to demo data at %s.",
            primary_path,
            demo_path,
        )
        df = pd.read_csv(demo_path)
    else:
        raise FileNotFoundError(
            f"Missing RAPM data file. Expected '{primary_path}' or demo fallback '{demo_path}'."
        )

    numeric_cols = [
        "orapm",
        "drapm",
        "rapm",
        "estimated_minutes",
        "stint_appearances",
        "ci_low",
        "ci_high",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

