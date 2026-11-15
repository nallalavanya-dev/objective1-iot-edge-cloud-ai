"""
Objective 1: IoT multi-node end-to-end completion-time prediction and task placement.
Dataset: objective1_iot_multinode_training_dataset.csv (long format, one row per task-node candidate)
Decision dataset: objective1_task_decision_dataset.csv (one row per task, six node alternatives)
Synthetic benchmark only. Replace/supplement with measured NS-3/hardware traces for research claims.
"""
from pathlib import Path
import time, joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, balanced_accuracy_score, classification_report
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, VotingRegressor, StackingRegressor, RandomForestClassifier, ExtraTreesClassifier

HERE = Path(__file__).resolve().parent
LONG_FILE = HERE / "objective1_iot_multinode_training_dataset.csv"
DECISION_FILE = HERE / "objective1_task_decision_dataset.csv"
OUT = HERE / "trained_models"
OUT.mkdir(exist_ok=True)

TARGET = "completion_time_ms"
CAT = ["iot_application", "task_type", "task_priority", "node_id", "location", "node_type"]
NUM = ["task_input_kb", "cpu_cycles_giga", "return_data_kb", "deadline_ms",
       "bandwidth_mbps", "rtt_ms", "queue_jobs", "cpu_capacity_ghz",
       "cpu_utilization_pct"]
FEATURES = CAT + NUM

def make_preprocessor():
    return ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUM),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CAT)
    ])

def main():
    df = pd.read_csv(LONG_FILE)
    train = df[df.dataset_split == "train"].copy()
    test = df[df.dataset_split == "test"].copy()
    Xtr, ytr = train[FEATURES], train[TARGET]
    Xte, yte = test[FEATURES], test[TARGET]

    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge(alpha=1.0)),
        ("Decision Tree", DecisionTreeRegressor(max_depth=20, min_samples_leaf=3, random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42)),
        ("Extra Trees", ExtraTreesRegressor(n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42)),
        ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=.05, random_state=42)),
        ("Hist Gradient Boosting", HistGradientBoostingRegressor(max_iter=120, learning_rate=.07, max_leaf_nodes=31, random_state=42)),
        ("Voting Ensemble", VotingRegressor([
            ("rf", RandomForestRegressor(n_estimators=60, min_samples_leaf=2, n_jobs=-1, random_state=42)),
            ("et", ExtraTreesRegressor(n_estimators=60, min_samples_leaf=2, n_jobs=-1, random_state=43)),
            ("gb", GradientBoostingRegressor(n_estimators=70, max_depth=3, learning_rate=.05, random_state=42))
        ])),
        ("Stacking Ensemble", StackingRegressor([
            ("rf", RandomForestRegressor(n_estimators=45, min_samples_leaf=2, n_jobs=-1, random_state=42)),
            ("et", ExtraTreesRegressor(n_estimators=45, min_samples_leaf=2, n_jobs=-1, random_state=43)),
            ("gb", GradientBoostingRegressor(n_estimators=55, max_depth=3, learning_rate=.05, random_state=42))
        ], final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=-1))
    ]

    score_rows, fitted = [], {}
    for name, estimator in models:
        pipe = Pipeline([("preprocess", make_preprocessor()), ("model", estimator)])
        start = time.perf_counter()
        pipe.fit(Xtr, ytr)
        train_sec = time.perf_counter() - start
        start = time.perf_counter()
        pred = pipe.predict(Xte)
        inference_ms = (time.perf_counter() - start) * 1000 / len(Xte)
        score_rows.append({
            "Algorithm": name,
            "MAE_ms": mean_absolute_error(yte, pred),
            "RMSE_ms": mean_squared_error(yte, pred) ** .5,
            "R2": r2_score(yte, pred),
            "Training_seconds": train_sec,
            "Inference_ms_per_row": inference_ms
        })
        fitted[name] = pipe
        print(f"{name}: MAE={score_rows[-1]['MAE_ms']:.3f} ms | RMSE={score_rows[-1]['RMSE_ms']:.3f} ms | R2={score_rows[-1]['R2']:.5f}")

    scores = pd.DataFrame(score_rows).sort_values(["RMSE_ms", "MAE_ms"])
    scores.to_csv(OUT / "regression_model_comparison.csv", index=False)
    best_name = scores.iloc[0]["Algorithm"]
    joblib.dump(fitted[best_name], OUT / "best_completion_time_regressor.joblib")
    (OUT / "best_regressor_name.txt").write_text(str(best_name), encoding="utf-8")
    print("\\nBest regression model by held-out RMSE:", best_name)

    # Train a task-level classifier for the recommended execution node.
    ddf = pd.read_csv(DECISION_FILE)
    dtrain = ddf[ddf.dataset_split == "train"].copy()
    dtest = ddf[ddf.dataset_split == "test"].copy()
    drop_cols = ["task_id", "task_name", "recommended_node", "recommended_node_completion_ms",
                 "decision_status", "dataset_split"]
    dcats = ["iot_application", "task_type", "task_priority"]
    dnums = [c for c in ddf.columns if c not in drop_cols + dcats]
    dprep = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), dnums),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), dcats)
    ])
    clf = Pipeline([("preprocess", dprep),
                    ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                     class_weight="balanced_subsample",
                                                     n_jobs=-1, random_state=42))])
    clf.fit(dtrain[dcats + dnums], dtrain["recommended_node"])
    dpred = clf.predict(dtest[dcats + dnums])
    print("\\nTask placement classifier:")
    print("Accuracy:", round(accuracy_score(dtest["recommended_node"], dpred), 4))
    print("Macro F1:", round(f1_score(dtest["recommended_node"], dpred, average="macro"), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(dtest["recommended_node"], dpred), 4))
    print(classification_report(dtest["recommended_node"], dpred, zero_division=0))
    joblib.dump(clf, OUT / "task_execution_node_classifier.joblib")
    print("\\nSaved model files and regression_model_comparison.csv in:", OUT)

if __name__ == "__main__":
    main()
