import pandas as pd
from src.config import COLUMNS, DATA_URL, RAW_DATA_PATH, PROCESSED_DATA_PATH


def download_dataset(output_path=RAW_DATA_PATH):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_URL, header=None, names=COLUMNS, na_values="?")
    df.to_csv(output_path, index=False)
    return df


def clean_dataset(input_path=RAW_DATA_PATH, output_path=PROCESSED_DATA_PATH):
    df = pd.read_csv(input_path)
    df["target_binary"] = (df["target"] > 0).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    data = download_dataset()
    clean = clean_dataset()
    print("Downloaded raw dataset:", data.shape)
    print("Saved cleaned dataset:", clean.shape)
    print(clean.head())
