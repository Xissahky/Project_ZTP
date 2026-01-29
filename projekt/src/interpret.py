import pandas as pd
import numpy as np


def get_feature_importance_rf(pipeline, top_n: int = 15) -> pd.DataFrame:
    """
    Zwraca najważniejsze cechy dla modelu Random Forest.
    """
    model = pipeline.named_steps["model"]
    pre = pipeline.named_steps["preprocess"]

    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not support feature_importances_")

    feature_names = pre.get_feature_names_out()
    importances = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    df = df.sort_values("importance", ascending=False).head(top_n)
    df["importance"] = df["importance"].round(4)

    return df
