import argparse
import pandas as pd
import joblib

MODEL_PATH = "models/rf_model.pkl"
FEATURES = ["home", "win_pct", "run_diff", "R", "RA"]


def predict_game(home_stats: dict, away_stats: dict) -> str:
    """Predict winner given home and away team stat dicts."""
    model = joblib.load(MODEL_PATH)

    home_input = pd.DataFrame([{**home_stats, "home": 1}])[FEATURES]
    away_input = pd.DataFrame([{**away_stats, "home": 0}])[FEATURES]

    home_prob = model.predict_proba(home_input)[0][1]
    away_prob = model.predict_proba(away_input)[0][1]

    winner = "Home team" if home_prob >= away_prob else "Away team"
    print(f"Home win probability : {home_prob:.2%}")
    print(f"Away win probability : {away_prob:.2%}")
    print(f"Predicted winner     : {winner}")
    return winner


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict MLB game winner")
    parser.add_argument("--date", type=str, help="Game date (YYYY-MM-DD)")
    args = parser.parse_args()

    # Example usage with placeholder stats — replace with real fetched data
    example_home = {"win_pct": 0.56, "run_diff": 45, "R": 5.1, "RA": 4.2}
    example_away = {"win_pct": 0.48, "run_diff": -10, "R": 4.5, "RA": 4.8}

    print(f"Predicting game for date: {args.date}")
    predict_game(example_home, example_away)
