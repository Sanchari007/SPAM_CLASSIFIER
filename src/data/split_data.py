from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.seed import set_seed


PROCESSED_DATA_DIR = Path("data/processed")
CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "clean_sms.csv"

TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_PATH = PROCESSED_DATA_DIR / "val.csv"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42


def split_data():
    set_seed(RANDOM_SEED)

    df = pd.read_csv(CLEAN_DATA_PATH)

    X_train, X_val = train_test_split(
        df,
        test_size=1 - TRAIN_RATIO,
        stratify=df["label"],
        random_state=RANDOM_SEED,
    )

    print("=== SPLIT SUMMARY ===")
    print(f"Train size: {len(X_train)}")
    print(f"Validation size: {len(X_val)}")

    print("\nTrain label distribution:")
    print(X_train["label"].value_counts(normalize=True))

    print("\nValidation label distribution:")
    print(X_val["label"].value_counts(normalize=True))

    X_train.to_csv(TRAIN_PATH, index=False)
    X_val.to_csv(VAL_PATH, index=False)

    print("\nSaved files:")
    print(TRAIN_PATH)
    print(VAL_PATH)


if __name__ == "__main__":
    split_data()
