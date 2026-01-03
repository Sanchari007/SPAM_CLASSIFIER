from pathlib import Path
import pandas as pd


RAW_DATA_PATH = Path("data/raw/SMSSpamCollection")


def load_raw_sms_data() -> pd.DataFrame:
    """
    Load the raw SMS Spam Collection dataset.

    Returns:
        pd.DataFrame with columns:
        - label: spam or ham
        - text: message content
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {RAW_DATA_PATH}. "
            "Make sure SMSSpamCollection is placed in data/raw/"
        )

    df = pd.read_csv(
        RAW_DATA_PATH,
        sep="\t",
        header=None,
        names=["label", "text"],
        encoding="utf-8"
    )

    return df


if __name__ == "__main__":
    df = load_raw_sms_data()
    print(df.head())
    print("\nDataset shape:", df.shape)
    print("\nLabel distribution:")
    print(df["label"].value_counts())
