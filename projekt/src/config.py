from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

@dataclass(frozen=True)
class Paths:
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    reports_dir: Path = PROJECT_ROOT / "reports"
    models_dir: Path = PROJECT_ROOT / "models"

    loans_csv: Path = raw_dir / "loans.csv"
    customers_csv: Path = raw_dir / "customers.csv"

paths = Paths()
