"""Validate ranked output and summarize metrics available from that output."""

import numpy as np
import pandas as pd

from src.ranking_engine.config import OBJECTIVES

REQUIRED_COLUMNS = ("rank", "movieId", "title", "score", "sources", *OBJECTIVES)
NUMERIC_COLUMNS = ("rank", "movieId", "score", *OBJECTIVES)


def prepare_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean, rank-ordered copy of a Module 4 output frame."""
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    results = frame.loc[:, REQUIRED_COLUMNS].copy()
    if results.empty:
        return results
    for column in NUMERIC_COLUMNS:
        results[column] = pd.to_numeric(results[column], errors="coerce")
    if not np.isfinite(results[list(NUMERIC_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Rank, movie ID, score, and objective columns must be finite numbers.")
    if (results["rank"] < 1).any() or (results["rank"] % 1 != 0).any():
        raise ValueError("Ranks must be positive whole numbers.")
    if (results["movieId"] % 1 != 0).any():
        raise ValueError("Movie IDs must be whole numbers.")
    if results["rank"].duplicated().any() or results["movieId"].duplicated().any():
        raise ValueError("Ranks and movie IDs must be unique.")
    if results["title"].isna().any() or results["title"].astype(str).str.strip().eq("").any():
        raise ValueError("Every recommendation needs a title.")
    results["rank"] = results["rank"].astype(int)
    results["movieId"] = results["movieId"].astype(int)
    results["sources"] = results["sources"].fillna("").astype(str)
    return results.sort_values("rank").reset_index(drop=True)


def output_kpis(results: pd.DataFrame) -> dict[str, float | int]:
    """These describe the displayed list, not held-out recommendation accuracy."""
    if results.empty:
        return {"titles": 0, "mean_score": 0.0, "mean_relevance": 0.0, "mean_novelty": 0.0}
    return {
        "titles": len(results),
        "mean_score": float(results["score"].mean()),
        "mean_relevance": float(results["relevance"].mean()),
        "mean_novelty": float(results["novelty"].mean()),
    }
