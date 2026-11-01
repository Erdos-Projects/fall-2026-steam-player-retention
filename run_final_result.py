# run_final_results.py

"""
Reproduce the final results for the Steam retention project.

This script executes the main project notebooks in order:

1. Data preparation / feature engineering
2. Baseline modeling
3. Modeling experiments
4. Final evaluation and presentation figures

Run from the project root with:

    python run_final_results.py
"""

from pathlib import Path
import subprocess
import sys


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

NOTEBOOK_DIR = PROJECT_DIR / "notebooks"


# ============================================================
# NOTEBOOK EXECUTION ORDER
# ============================================================

NOTEBOOKS = [
    "steam_data_setup.ipynb",
    "modeling_baselines.ipynb",
    "modeling_experiments.ipynb",
    "final_results.ipynb",
]


# ============================================================
# EXECUTE NOTEBOOK
# ============================================================

def run_notebook(notebook_name):
    """
    Execute a notebook in place using the same Python
    environment that launched this script.
    """

    notebook_path = NOTEBOOK_DIR / notebook_name

    if not notebook_path.exists():
        raise FileNotFoundError(
            f"Notebook not found: {notebook_path}"
        )

    print("\n" + "=" * 70)
    print(f"Running: {notebook_name}")
    print("=" * 70)

    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--inplace",
        "--ExecutePreprocessor.timeout=600",
        str(notebook_path),
    ]

    subprocess.run(
        command,
        cwd=PROJECT_DIR,
        check=True,
    )

    print(f"Finished: {notebook_name}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEAM RETENTION PROJECT")
    print("FINAL RESULTS REPRODUCTION")
    print("=" * 70)

    print("\nPython:")
    print(sys.executable)

    print("\nProject directory:")
    print(PROJECT_DIR)

    for notebook in NOTEBOOKS:
        run_notebook(notebook)

    print("\n" + "=" * 70)
    print("REPRODUCTION COMPLETE")
    print("=" * 70)

    print("\nFinal outputs should be available in:")
    print(PROJECT_DIR / "results" / "final")

    print("\nModel artifact:")
    print(PROJECT_DIR / "artifacts" / "final_model.joblib")


if __name__ == "__main__":
    main()