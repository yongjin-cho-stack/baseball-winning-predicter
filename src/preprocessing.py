import pandas as pd

ROLLING_WINDOW = 15  # 최근 N경기 평균

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


def _team_view(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Extract one side (home/away) as a team-centric view."""
    cols = {f"{side}_{c}": c for c in STAT_COLS}
    cols.update({"game_date": "game_date", f"{side}_team": "team", "home_win": "home_win"})
    view = df[list(cols.keys())].rename(columns=cols).copy()
    view["is_home"] = int(side == "home")
    view["won"] = view["home_win"] if side == "home" else 1 - view["home_win"]
    return view[["game_date", "team", "is_home", "won"] + STAT_COLS]


def compute_rolling(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling mean and cumulative win rate for each team."""
    df = df.sort_values("game_date").copy()
    for col in STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 롤링 평균
    rolled = (
        df.groupby("team")[STAT_COLS]
        .transform(lambda s: s.shift(1).rolling(ROLLING_WINDOW, min_periods=5).mean())
    )
    rolled.columns = [f"roll_{c}" for c in STAT_COLS]

    # 시즌 누적 승률 (해당 경기 이전까지)
    df["season"] = df["game_date"].dt.year
    cumwins = df.groupby(["team", "season"])["won"].transform(
        lambda s: s.shift(1).expanding().sum()
    )
    cumgames = df.groupby(["team", "season"])["won"].transform(
        lambda s: s.shift(1).expanding().count()
    )
    df["season_win_pct"] = (cumwins / cumgames).fillna(0.5)

    return pd.concat([df, rolled], axis=1).dropna(subset=[f"roll_{c}" for c in STAT_COLS])


def build_game_features(raw_path: str) -> pd.DataFrame:
    """
    Build one row per game with rolling stats, season win rate,
    and home indicator for both teams.
    Target: home_win (1 = home team wins).
    """
    df = pd.read_csv(raw_path, parse_dates=["game_date"])
    df = df.sort_values("game_date").reset_index(drop=True)

    home_view = _team_view(df, "home")
    away_view = _team_view(df, "away")

    all_games = pd.concat([home_view, away_view], ignore_index=True)
    all_rolled = compute_rolling(all_games)

    home_rolled = all_rolled[all_rolled["is_home"] == 1].copy()
    away_rolled = all_rolled[all_rolled["is_home"] == 0].copy()

    roll_cols = [f"roll_{c}" for c in STAT_COLS]
    home_rolled = home_rolled.rename(columns={c: f"home_{c}" for c in roll_cols})
    away_rolled = away_rolled.rename(columns={c: f"away_{c}" for c in roll_cols})
    home_rolled = home_rolled.rename(columns={"season_win_pct": "home_season_win_pct"})
    away_rolled = away_rolled.rename(columns={"season_win_pct": "away_season_win_pct"})

    home_rolled["key"] = home_rolled["game_date"].astype(str) + "_" + home_rolled["team"]
    away_rolled["key"] = away_rolled["game_date"].astype(str) + "_" + away_rolled["team"]

    merged = df[["game_date", "home_team", "away_team", "home_win"]].copy()
    merged["home_key"] = merged["game_date"].astype(str) + "_" + merged["home_team"]
    merged["away_key"] = merged["game_date"].astype(str) + "_" + merged["away_team"]

    home_feat_cols = [f"home_roll_{c}" for c in STAT_COLS] + ["home_season_win_pct"]
    away_feat_cols = [f"away_roll_{c}" for c in STAT_COLS] + ["away_season_win_pct"]

    home_feat = home_rolled.set_index("key")[home_feat_cols]
    away_feat = away_rolled.set_index("key")[away_feat_cols]

    merged = merged.join(home_feat, on="home_key").join(away_feat, on="away_key")
    merged = merged.drop(columns=["home_key", "away_key"]).dropna()

    # 홈 어드밴티지 피처 추가
    merged["home_advantage"] = 1

    return merged


if __name__ == "__main__":
    features = build_game_features("data/raw/mlb_games.csv")
    features.to_csv("data/processed/mlb_features.csv", index=False)
    print(f"완료: {len(features)}경기 → data/processed/mlb_features.csv")
    print(features.head())
