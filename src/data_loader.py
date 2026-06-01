import time
import pandas as pd
import statsapi
from datetime import datetime, timedelta


def date_range(start: str, end: str):
    """Yield each date string between start and end (YYYY-MM-DD)."""
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while current <= end_dt:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def fetch_starter_stats(pitchers: list) -> dict:
    """Extract starting pitcher stats from boxscore pitcher list (index 1)."""
    if len(pitchers) < 2:
        return {}
    starter = pitchers[1]
    return {
        "starter_era":  starter.get("era", None),
        "starter_ip":   starter.get("ip", None),
        "starter_k":    starter.get("k", None),
        "starter_bb":   starter.get("bb", None),
        "starter_h":    starter.get("h", None),
        "starter_er":   starter.get("er", None),
        "starter_name": starter.get("name", ""),
    }


def fetch_boxscore(game_id: int) -> dict | None:
    """Return batting/pitching team stats + starting pitcher stats."""
    try:
        box = statsapi.boxscore_data(game_id)
        row = {"game_id": game_id}

        for side in ("home", "away"):
            batting = box[side]["teamStats"]["batting"]
            pitching = box[side]["teamStats"]["pitching"]
            for key, val in batting.items():
                row[f"{side}_bat_{key}"] = val
            for key, val in pitching.items():
                row[f"{side}_pit_{key}"] = val

            # 선발 투수 스탯
            pitcher_key = "homePitchers" if side == "home" else "awayPitchers"
            starter = fetch_starter_stats(box[pitcher_key])
            for key, val in starter.items():
                row[f"{side}_{key}"] = val

        return row
    except Exception as e:
        print(f"  boxscore error game_id={game_id}: {e}")
        return None


def collect(start_date: str, end_date: str, delay: float = 0.1) -> pd.DataFrame:
    """
    Collect game results, team stats, and starting pitcher stats
    for all regular season games between start_date and end_date.
    """
    records = []

    for date in date_range(start_date, end_date):
        games = statsapi.schedule(date=date, sportId=1)
        daily = [g for g in games if g["game_type"] == "R" and g["status"] == "Final"]

        if not daily:
            continue

        print(f"{date}: {len(daily)} games")

        for g in daily:
            box = fetch_boxscore(g["game_id"])
            if box is None:
                continue

            box.update({
                "game_date":  g["game_date"],
                "home_team":  g["home_name"],
                "away_team":  g["away_name"],
                "home_score": g["home_score"],
                "away_score": g["away_score"],
                "winning_team": g.get("winning_team", ""),
                "home_win":   int(g["home_score"] > g["away_score"]),
            })
            records.append(box)
            time.sleep(delay)

    return pd.DataFrame(records)


if __name__ == "__main__":
    # 2023 시즌 재수집 (선발 투수 스탯 포함) 후 2025와 합치기
    print("\n=== 2023 시즌 수집 중 (선발 투수 스탯 포함) ===")
    df_2023 = collect("2023-03-30", "2023-10-01")

    existing = pd.read_csv("data/raw/mlb_games.csv")
    df_2025 = existing[existing["game_date"] >= "2025-01-01"]

    result = pd.concat([df_2023, df_2025], ignore_index=True)
    result.to_csv("data/raw/mlb_games.csv", index=False)
    print(f"\n완료: 총 {len(result)}경기 저장 → data/raw/mlb_games.csv")
    print(f"  2023: {len(df_2023)}경기 + 2025: {len(df_2025)}경기")
