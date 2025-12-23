import pytest


def test_imports():
    import numpy, pandas, scipy, sklearn, matplotlib, seaborn, statsmodels, pyarrow, polars
    import requests, httpx, pydantic, dotenv, tqdm, rich, loguru, sqlalchemy, psycopg2, typer

    # Verify core packages report versions
    assert numpy.__version__
    assert pandas.__version__

    # xgboost and lightgbm require OpenMP (libomp) on macOS. Skip if missing.
    try:
        import xgboost  # noqa: F401
    except Exception as e:
        if "libomp" in str(e).lower():
            pytest.skip("libomp missing; install via Homebrew: brew install libomp")
        else:
            raise

    try:
        import lightgbm  # noqa: F401
    except Exception as e:
        if "libomp" in str(e).lower():
            pytest.skip("libomp missing; install via Homebrew: brew install libomp")
        else:
            raise
