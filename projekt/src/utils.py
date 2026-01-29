from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def describe_basic(df: pd.DataFrame) -> str:
    lines = []
    lines.append(f"Shape: {df.shape}\n")

    lines.append("Class balance (paid_on_time):")
    lines.append(df["paid_on_time"].value_counts(normalize=True).to_string())
    lines.append("")

    lines.append("Missing values (top 10):")
    lines.append(df.isna().sum().sort_values(ascending=False).head(10).to_string())
    lines.append("")

    num_desc = df.select_dtypes(include="number").describe().round(2)
    lines.append("Numeric describe:")
    lines.append(num_desc.to_string())

    return "\n".join(lines)


def plot_target_hist(y: pd.Series, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    y.value_counts().sort_index().plot(kind="bar")
    plt.title("Target distribution")
    plt.xlabel("class")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
