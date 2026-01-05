from pathlib import Path
import pandas as pd

from src.data.load_data import load_raw_sms_data


PROCESSED_DATA_DIR = Path("data/processed")
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "clean_sms.csv"

VALID_LABELS = {"spam", "ham"}


def clean_sms_data() -> pd.DataFrame:
    """
    Clean and validate the SMS Spam dataset.

    Returns:
        Cleaned pandas DataFrame
    """
    df = load_raw_sms_data()
    original_size = len(df)

    # Strip whitespace
    df["label"] = df["label"].astype(str).str.strip()
    df["text"] = df["text"].astype(str).str.strip()

    # Validate labels
    df = df[df["label"].isin(VALID_LABELS)]

    # Remove empty messages
    df = df[df["text"] != ""]

    # Remove exact duplicates
    df = df.drop_duplicates()

    cleaned_size = len(df)

    print("=== CLEANING SUMMARY ===")
    print(f"Original rows: {original_size}")
    print(f"Cleaned rows:  {cleaned_size}")
    print(f"Rows removed:  {original_size - cleaned_size}")
    print("\nLabel distribution after cleaning:")
    print(df["label"].value_counts())

    return df


def save_clean_data(df: pd.DataFrame) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"\nClean data saved to: {CLEAN_DATA_PATH}")


if __name__ == "__main__":
    df_clean = clean_sms_data()
    save_clean_data(df_clean)
