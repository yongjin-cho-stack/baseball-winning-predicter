import pandas as pd
import joblib
import statsapi
from datetime import datetime, timedelta

MODEL_PATH = "models/rf_model.pkl"
FEATURE_COLS_PATH = "models/feature_cols.txt"
ROLLING_WINDOW = 15

BAT_COLS = [
    "bat_hits", "bat_homeRuns", "bat_baseOnBalls",
    "bat_strikeOuts", "bat_rbi", "bat_stolenBases",
    "bat_obp", "bat_slg", "bat_ops",
    "bat_runs", "bat_doubles", "bat_triples", "bat_leftOnBase",
]
PIT_COLS = [
    "pit_era", "pit_strikeOuts", "pit_baseOnBalls",
    "pit_hits", "pit_earnedRuns",
    "pit_numberOfPitches", "pit_strikes", "pit_inningsPitched",
    "pit_homeRuns", "pit_runs",
]
STAT_COLS = BAT_COLS + PIT_COLS


def fetch_recent_stats(team_id: int, before_date: str) -> dict | None:
    """Fetch rolling average stats for a team from recent N games before given date."""
    end = datetime.strptime(before_date, "%Y-%m-%d") - timedelta(days=1)
    start = end - timedelta(days=60)

    games = statsapi.schedule(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        team=team_id,
        sportId=1,
    )
    completed = [g for g in games if g["game_type"] == "R" and g["status"] == "Final"]
    recent = completed[-ROLLING_WINDOW:]

    if len(recent) < 3:
        print(f"  team_id={team_id}: 최근 경기 데이터 부족 ({len(recent)}경기)")
        return None

    stats_list = []
    for g in recent:
        try:
            box = statsapi.boxscore_data(g["game_id"])
            side = "home" if g["home_id"] == team_id else "away"
            batting = box[side]["teamStats"]["batting"]
            pitching = box[side]["teamStats"]["pitching"]
            row = {}
            for col in BAT_COLS:
                key = col.replace("bat_", "")
                row[col] = float(batting.get(key, 0) or 0)
            for col in PIT_COLS:
                key = col.replace("pit_", "")
                row[col] = float(pitching.get(key, 0) or 0)
            stats_list.append(row)
        except Exception:
            continue

    if not stats_list:
        return None

    avg = pd.DataFrame(stats_list).mean().to_dict()
    return avg


def predict_date(date: str):
    """Predict winners for all MLB games on a given date."""
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLS_PATH) as f:
        feature_cols = [line.strip() for line in f]

    games = statsapi.schedule(date=date, sportId=1)
    games = [g for g in games if g["game_type"] == "R"]

    if not games:
        print(f"{date}에 예정된 정규시즌 경기가 없습니다.")
        return

    print(f"\n=== {date} 경기 예측 ({len(games)}경기) ===\n")

    for g in games:
        home = g["home_name"]
        away = g["away_name"]
        home_id = g["home_id"]
        away_id = g["away_id"]

        print(f"{away} @ {home}")

        home_stats = fetch_recent_stats(home_id, date)
        away_stats = fetch_recent_stats(away_id, date)

        if home_stats is None or away_stats is None:
            print("  → 데이터 부족으로 예측 불가\n")
            continue

        row = {}
        for col in STAT_COLS:
            row[f"home_roll_{col}"] = home_stats.get(col, 0)
            row[f"away_roll_{col}"] = away_stats.get(col, 0)

        X = pd.DataFrame([row])[feature_cols]
        prob = model.predict_proba(X)[0]
        home_prob, away_prob = prob[1], prob[0]

        winner = home if home_prob >= away_prob else away
        print(f"  홈팀 승리 확률: {home_prob:.1%}")
        print(f"  원정팀 승리 확률: {away_prob:.1%}")
        print(f"  예측 승리팀: {winner}\n")


if __name__ == "__main__":
    date = input("예측할 날짜를 입력하세요 (YYYY-MM-DD): ").strip()
    predict_date(date)
