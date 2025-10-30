import pandas as pd
from pathlib import Path

from src.encoder.encoder import PestDataEncoder
from src.utils import sanitize_filename

CLEAN_CSV = Path("data/pest_survey_cleaned.csv")  # adjust if needed

def main():
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(f"{CLEAN_CSV} not found. Please clean data first.")

    df = pd.read_csv(CLEAN_CSV)
    encoder = PestDataEncoder()

    for idx, row in df.iterrows():
        tehsil = row.get("TEHSILS", f"row_{idx}")
        timeframe = row.get("TIMEFRAME", "unknown_period")
        print(f"\n>>> Processing {tehsil} | {timeframe}")
        encoder.analyze(row.to_dict(), save_output=True)

if __name__ == "__main__":
    main()
