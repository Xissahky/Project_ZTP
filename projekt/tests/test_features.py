import pandas as pd
from src.features import infer_feature_types, build_preprocess_pipeline

def test_infer_feature_types():
    X = pd.DataFrame({
        "age": [20, 30],
        "income": [1000.0, 1200.0],
        "city": ["A", "B"],
    })
    num, cat = infer_feature_types(X)
    assert "age" in num and "income" in num
    assert "city" in cat

def test_preprocess_pipeline_fit_transform():
    X = pd.DataFrame({
        "age": [20, None, 40],
        "income": [1000.0, 1200.0, None],
        "city": ["A", "B", "A"],
    })
    pre = build_preprocess_pipeline(X)
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == X.shape[0]
