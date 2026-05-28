import pandas as pd
from pybaseball import schedule_and_record


def load_season_data(start_year: int, end_year: int) -> pd.DataFrame:
    """Load season records for all MLB teams across given year range."""
    frames = []
    teams = [
        "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE",
        "COL", "DET", "HOU", "KCR", "LAA", "LAD", "MIA", "MIL",
        "MIN", "NYM", "NYY", "OAK", "PHI", "PIT", "SDP", "SEA",
        "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
    ]
    for year in range(start_year, end_year + 1):
        for team in teams:
            try:
                df = schedule_and_record(year, team)
                df["team"] = team
                df["year"] = year
                frames.append(df)
            except Exception as e:
                print(f"Skipping {team} {year}: {e}")
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = load_season_data(2021, 2023)
    df.to_csv("data/raw/mlb_seasons.csv", index=False)
    print(f"Saved {len(df)} rows to data/raw/mlb_seasons.csv")
