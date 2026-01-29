from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import paths
from .data import load_raw_data, basic_cleaning, merge_sources, make_train_test
from .train import train_logreg, train_random_forest, save_model
from .evaluate import evaluate_classifier, compare_models
from .utils import describe_basic, plot_target_hist
from .interpret import get_feature_importance_rf


TARGET_COL = "paid_on_time"

def main() -> None:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)

    loans, customers = load_raw_data()
    loans = basic_cleaning(loans)
    customers = basic_cleaning(customers)

    df = merge_sources(loans, customers)

    # Save basic EDA text
    report_txt = paths.reports_dir / "basic_report.txt"
    report_txt.write_text(describe_basic(df), encoding="utf-8")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL}' not found. "
            f"Available columns: {list(df.columns)[:20]}..."
        )

    # Plot target distribution
    plot_target_hist(df[TARGET_COL], paths.reports_dir / "target_distribution.png")

    X_train, X_test, y_train, y_test = make_train_test(df, target_col=TARGET_COL)

    models = [
        train_logreg(X_train, y_train),
        train_random_forest(X_train, y_train),
    ]

    results = []
    for m in models:
        metrics = evaluate_classifier(m.pipeline, X_test, y_test)
        metrics["model"] = m.name
        results.append(metrics)
        save_model(m)


    table = compare_models(results)
    table_path = paths.reports_dir / "metrics.csv"
    table.to_csv(table_path, index=False)



    # ---- Feature importance for Random Forest ----
    rf_model = next(m for m in models if m.name == "rf")
    importance_df = get_feature_importance_rf(rf_model.pipeline, top_n=25)

    importance_path = paths.reports_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)

    print("\nTop cechy wpływające na decyzję modelu (Random Forest):")
    print(importance_df.to_string(index=False))
    print(f"\nSaved: {importance_path}")




    print("Done")
    print(table.to_string(index=False))
    print(f"\nSaved: {report_txt}")
    print(f"Saved: {paths.reports_dir / 'target_distribution.png'}")
    print(f"Saved: {table_path}")
    print(f"Models in: {paths.models_dir}")

if __name__ == "__main__":
    main()
