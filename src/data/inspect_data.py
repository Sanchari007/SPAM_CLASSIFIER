from src.data.load_data import load_raw_sms_data


def inspect_raw_data():
    df = load_raw_sms_data()

    print("=== BASIC INFO ===")
    print("Shape:", df.shape)
    print("\nColumn types:")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== LABEL DISTRIBUTION ===")
    print(df["label"].value_counts())
    print("\nLabel proportions:")
    print(df["label"].value_counts(normalize=True))

    print("\n=== SAMPLE MESSAGES ===")
    print("\nHAM examples:")
    print(df[df["label"] == "ham"]["text"].head(3))

    print("\nSPAM examples:")
    print(df[df["label"] == "spam"]["text"].head(3))


if __name__ == "__main__":
    inspect_raw_data()
