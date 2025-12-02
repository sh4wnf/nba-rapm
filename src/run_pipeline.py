#  script for running the full rapm pipeline end-to-end offline

import argparse
import sys
import time

import numpy as np
import pandas as pd

from . import config
from .data_pipeline import (
    fetch_pbp_for_season,
    build_season_stints,
    build_sparse_matrices,
)
from .rapm_model import (
    bootstrap_rapm_full,
    compute_full_rapm,
    compute_player_minutes_from_stints,
    compute_summary_stats,
    create_experiment_record,
    log_experiment,
    print_rapm_summary,
)
from .utils import ensure_dirs, setup_logging

logger = setup_logging()


# save final rapm dataframe to csv
def save_results(
    rapm_df: pd.DataFrame,
    output_path=None,
    min_stints: int = 0,
    min_minutes: float = None,
):
    from pathlib import Path as _Path

    if output_path is None:
        output_path = config.RAPM_OUTPUTS_CSV
    else:
        output_path = _Path(output_path)
    
    ensure_dirs()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # apply filters before saving
    df_filtered = rapm_df.copy()
    original_count = len(df_filtered)
    
    if min_stints > 0 and "stint_appearances" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["stint_appearances"] >= min_stints]
        logger.info("Filtered to %d players with >= %d stints (from %d)", len(df_filtered), min_stints, original_count)
    
    if min_minutes is not None and "estimated_minutes" in df_filtered.columns:
        before = len(df_filtered)
        df_filtered = df_filtered[df_filtered["estimated_minutes"] >= min_minutes]
        logger.info("Filtered to %d players with >= %.1f minutes (from %d)", len(df_filtered), min_minutes, before)
    
    # sorts by total rapm
    if "rapm" in df_filtered.columns or "Total RAPM" in df_filtered.columns:
        sort_col = "Total RAPM" if "Total RAPM" in df_filtered.columns else "rapm"
        df_filtered = df_filtered.sort_values(sort_col, ascending=False)
    
    # saves to csv
    df_filtered.to_csv(output_path, index=False)
    logger.info("Saved RAPM results to %s (%d players)", output_path, len(df_filtered))
    
    return output_path


# main function: puts together ingestion, stints, rapm, and logging
def main(
    season: str = "2024-25",
    lambda_reg: float = None,
    num_bootstrap: int = None,
    skip_ingest: bool = False,
    skip_bootstrap: bool = False,
    min_stints: int = 0,
    min_minutes: float = None,
    run_diagnostics: bool = True,
) -> int:
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("RAPM Pipeline - Starting")
    logger.info("Season: %s", season)
    logger.info("=" * 60)
    
    ensure_dirs()
    
    if lambda_reg is None:
        lambda_reg = config.DEFAULT_LAMBDA
    if num_bootstrap is None:
        num_bootstrap = config.DEFAULT_BOOTSTRAP_ITERATIONS
    min_possessions = config.DEFAULT_MIN_POSSESSIONS
    net_cap = config.DEFAULT_NET_CAP
    
    # step 1: fetch pbp data (optional)
    if not skip_ingest:
        logger.info("")
        logger.info("Step 1: Fetch PBP Data")
        logger.info("-" * 60)
        try:
            fetch_pbp_for_season(season)
        except Exception as e:
            logger.error("Failed to fetch PBP data: %s", e, exc_info=True)
            return 1
    else:
        logger.info("Skipping PBP ingestion")
    
    # step 2: build season stints from the combined pbp csv
    logger.info("")
    logger.info("Step 2: Build Season Stints")
    logger.info("-" * 60)
    try:
        stints = build_season_stints()
        if stints.empty:
            logger.error("No stints created")
            return 1
        num_stints = len(stints)
        logger.info("Built %d stints", num_stints)
    except Exception as e:
        logger.error("Failed to build stints: %s", e, exc_info=True)
        return 1
    
    # step 3: build sparse matrices and save them to disk
    logger.info("")
    logger.info("Step 3: Build Sparse Matrices")
    logger.info("-" * 60)
    try:
        X_off, y_off, X_def, y_def, sample_weights = build_sparse_matrices(stints)
        num_players = X_off.shape[1]
        logger.info(
            "Built sparse matrices: %d stints, %d players (saved under data/matrices/)",
            X_off.shape[0],
            num_players,
        )
    except Exception as e:
        logger.error("Failed to build sparse matrices: %s", e, exc_info=True)
        return 1
    
    # step 4: compute rapm from stints (joint orapm / drapm)
    logger.info("")
    logger.info("Step 4: Compute RAPM")
    logger.info("-" * 60)
    try:
        rapm_df_main = compute_full_rapm(
            stints_df=stints,
            lam=lambda_reg,
            min_possessions=min_possessions,
            net_cap=net_cap,
        )
        logger.info("Computed RAPM for %d players", len(rapm_df_main))
    except Exception as e:
        logger.error("Failed to compute RAPM: %s", e, exc_info=True)
        return 1
    
    # step 5: bootstrap rapm (optional as well, adds uncertainty bands)
    if not skip_bootstrap:
        logger.info("")
        logger.info("step 5: bootstrap rapm (confidence intervals)")
        logger.info("-" * 60)
        logger.info("this may take a while (%d iterations)...", num_bootstrap)
        try:
            rapm_df_bs, bootstrap_samples = bootstrap_rapm_full(
                stints,
                lam=lambda_reg,
                num_bootstrap=num_bootstrap,
            )

            if "ci_low" in rapm_df_bs.columns and "ci_high" in rapm_df_bs.columns:
                ci_low = rapm_df_bs["ci_low"].values 
                ci_high = rapm_df_bs["ci_high"].values 
                rapm_df = rapm_df_bs.copy()
            else:
                rapm_df = None

            logger.info("Bootstrap complete")
        except Exception as e:
            logger.error("Bootstrap failed: %s", e, exc_info=True)
            logger.warning("Continuing without confidence intervals...")
            rapm_df = None
    else:
        logger.info("Skipping bootstrap")
        rapm_df = None
    
    # if bootstrap was skipped use point estimates only
    if rapm_df is None:
        logger.info("Using point estimates only (no bootstrap CIs).")
        rapm_df = rapm_df_main.copy()
        rapm_df["ci_low"] = np.nan
        rapm_df["ci_high"] = np.nan
    
    # step 6: small diagnostics
    if run_diagnostics:
        logger.info("")
        logger.info("Step 6: Diagnostics")
        logger.info("-" * 60)
        try:
            print_rapm_summary(rapm_df, min_stints=min_stints, top_n=20)
        except Exception as e:
            logger.warning("Diagnostics failed: %s", e, exc_info=True)
            logger.warning("Continuing without diagnostics...")
    
    # step 7: attach minutes and save results to csv
    logger.info("")
    logger.info("Step 7: Save Results")
    logger.info("-" * 60)
    try:
        # attach the estimated minutes if we have stint data
        if not stints.empty:
            try:
                minutes_df = compute_player_minutes_from_stints(stints)
                rapm_df = rapm_df.merge(
                    minutes_df[["player_id", "estimated_minutes", "stints_played"]],
                    on="player_id",
                    how="left",
                )
                if "stints_played" in rapm_df.columns:
                    rapm_df["stint_appearances"] = rapm_df["stint_appearances"].fillna(
                        rapm_df["stints_played"]
                    )
            except Exception as e:
                logger.warning("Could not add estimated minutes: %s", e)
        
        output_path = save_results(rapm_df, min_stints=min_stints, min_minutes=min_minutes)
        logger.info("Results saved")
    except Exception as e:
        logger.error("Failed to save results: %s", e, exc_info=True)
        return 1
    
    # step 8: logs experiment metadata and summary stats
    logger.info("")
    logger.info("Step 8: Log Experiment")
    logger.info("-" * 60)
    try:
        runtime = time.time() - start_time
        
        # computes summary statistics for orapm, drapm, total rapm
        orapm_stats = compute_summary_stats(rapm_df['orapm'].dropna().tolist())
        drapm_stats = compute_summary_stats(rapm_df['drapm'].dropna().tolist())
        rapm_stats = compute_summary_stats(rapm_df['rapm'].dropna().tolist())
        
        # creates experiment record to send to the logger
        parameters, results = create_experiment_record(
            lambda_reg=lambda_reg,
            num_stints=num_stints,
            num_players=num_players,
            runtime_seconds=runtime,
            orapm_stats=orapm_stats,
            drapm_stats=drapm_stats,
            rapm_stats=rapm_stats,
            bootstrap_iterations=num_bootstrap if not skip_bootstrap else None,
            season=season,
        )
        
        log_experiment(parameters, results)
        logger.info("Experiment logged")
    except Exception as e:
        logger.error("Failed to log experiment: %s", e, exc_info=True)
        logger.warning("Continuing without experiment log...")
    
    # the pipeline is complete
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline Complete!")
    logger.info("=" * 60)
    logger.info("Output files:")
    logger.info("  RAPM Results: %s", config.RAPM_OUTPUTS_CSV)
    logger.info("  Experiment Log: %s", config.EXPERIMENT_LOGS_DIR / "log.json")
    logger.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the complete RAPM pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--season",
        type=str,
        default="2024-25",
        help='Season string (e.g., "2024-25")'
    )
    parser.add_argument(
        "--lambda",
        type=float,
        default=None,
        dest="lambda_reg",
        help=f"Regularization parameter (default: {config.DEFAULT_LAMBDA})"
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=None,
        dest="num_bootstrap",
        help=f"Number of bootstrap iterations (default: {config.DEFAULT_BOOTSTRAP_ITERATIONS})"
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip data ingestion step"
    )
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip bootstrapping step"
    )
    parser.add_argument(
        "--min-stints",
        type=int,
        default=0,
        help="Minimum stints required for inclusion in final output (default: 0)"
    )
    parser.add_argument(
        "--min-minutes",
        type=float,
        default=None,
        help="Minimum estimated minutes required for inclusion (default: None)"
    )
    parser.add_argument(
        "--skip-diagnostics",
        action="store_true",
        help="Skip diagnostic output"
    )
    
    args = parser.parse_args()
    
    exit_code = main(
        season=args.season,
        lambda_reg=args.lambda_reg,
        num_bootstrap=args.num_bootstrap,
        skip_ingest=args.skip_ingest,
        skip_bootstrap=args.skip_bootstrap,
        min_stints=args.min_stints,
        min_minutes=args.min_minutes,
        run_diagnostics=not args.skip_diagnostics,
    )
    
    sys.exit(exit_code)

