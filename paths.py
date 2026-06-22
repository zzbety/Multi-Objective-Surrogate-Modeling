"""Shared path constants for the project."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
SURROGATE_DIR = OUTPUT_DIR / "surrogate"
PARETO_DIR = OUTPUT_DIR / "pareto"
FIGURES_DIR = OUTPUT_DIR / "figures"

NASBENCH_CSV = DATA_DIR / "nasbench201_cifar10_all.csv"
