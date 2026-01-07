from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer

from src.utils.seed import set_seed


PROCESSED_DATA_DIR = Path("data/processed")
TOKENIZED_DATA_DIR = Path("data/tokenized")

TRAIN_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_PATH = PROCESSED_DATA_DIR / "val.csv"

MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH = 128
RANDOM_SEED = 42


def tokenize_dataset():
    set_seed(RANDOM_SEED)

    TOKENIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    # Load CSVs into Hugging Face datasets
    train_df = Dataset.from_csv(str(TRAIN_PATH))
    val_df = Dataset.from_csv(str(VAL_PATH))

    # Label mapping
    label2id = {"ham": 0, "spam": 1}

    def encode(batch):
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )
        tokens["labels"] = [label2id[label] for label in batch["label"]]
        return tokens

    # Tokenize
    train_tokenized = train_df.map(encode, batched=True)
    val_tokenized = val_df.map(encode, batched=True)

    # Remove unused columns
    train_tokenized = train_tokenized.remove_columns(["text", "label"])
    val_tokenized = val_tokenized.remove_columns(["text", "label"])

    # Save to disk
    train_tokenized.save_to_disk(TOKENIZED_DATA_DIR / "train")
    val_tokenized.save_to_disk(TOKENIZED_DATA_DIR / "val")

    print("=== TOKENIZATION COMPLETE ===")
    print(f"Train samples: {len(train_tokenized)}")
    print(f"Validation samples: {len(val_tokenized)}")
    print(f"Saved to: {TOKENIZED_DATA_DIR}")

if __name__ == "__main__":
    tokenize_dataset()

