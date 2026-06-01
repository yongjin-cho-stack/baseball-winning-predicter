import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

MODEL_PATH = "models/rf_model.pkl"
FEATURE_COLS_PATH = "models/feature_cols.txt"


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("home_roll_", "away_roll_"))]


def train(data_path: str = "data/processed/mlb_features.csv"):
    df = pd.read_csv(data_path)
    feature_cols = get_feature_cols(df)

    X = df[feature_cols]
    y = df["home_win"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    # 피처 중요도 상위 10개 출력
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\n=== 피처 중요도 Top 10 ===")
    print(importances.nlargest(10).to_string())

    joblib.dump(model, MODEL_PATH)
    with open(FEATURE_COLS_PATH, "w") as f:
        f.write("\n".join(feature_cols))
    print(f"\n모델 저장 → {MODEL_PATH}")


if __name__ == "__main__":
    train()
