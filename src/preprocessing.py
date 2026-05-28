import pandas as pd


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw schedule data and engineer features."""
    df = df.copy()

    df = df[df["W/L"].notna()]
    df["home"] = (~df["Unnamed: 4"].notna()).astype(int)  # '@' marks away games
    df["win"] = df["W/L"].str.startswith("W").astype(int)

    df["R"] = pd.to_numeric(df["R"], errors="coerce")
    df["RA"] = pd.to_numeric(df["RA"], errors="coerce")
    df["W"] = pd.to_numeric(df["W"], errors="coerce")
    df["L"] = pd.to_numeric(df["L"], errors="coerce")

    df["win_pct"] = df["W"] / (df["W"] + df["L"])
    df["run_diff"] = df["R"] - df["RA"]

    df = df.dropna(subset=["win_pct", "run_diff"])
    return df[["team", "year", "Date", "home", "win", "win_pct", "run_diff", "R", "RA"]]


if __name__ == "__main__":
    raw = pd.read_csv("data/raw/mlb_seasons.csv")
    processed = preprocess(raw)
    processed.to_csv("data/processed/mlb_features.csv", index=False)
    print(f"Processed {len(processed)} rows saved to data/processed/mlb_features.csv")
