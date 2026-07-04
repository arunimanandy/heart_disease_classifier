import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from src.config import PROCESSED_DATA_PATH, REPORT_SCREENSHOT_DIR
from src.data_ingestion import clean_dataset, download_dataset


def main():
    if not PROCESSED_DATA_PATH.exists():
        download_dataset()
        clean_dataset()

    REPORT_SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROCESSED_DATA_PATH)

    print(df.info())
    print(df.describe())
    print("Missing values:\n", df.isna().sum())
    print("Class balance:\n", df["target_binary"].value_counts())

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="target_binary")
    plt.title("Heart Disease Class Balance")
    plt.xlabel("Target: 0 = No Disease, 1 = Disease")
    plt.ylabel("Count")
    plt.savefig(REPORT_SCREENSHOT_DIR / "class_balance.png", bbox_inches="tight")
    plt.close()

    df.drop(columns=["target_binary"]).hist(figsize=(14, 10))
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig(REPORT_SCREENSHOT_DIR / "feature_histograms.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(REPORT_SCREENSHOT_DIR / "correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    missing = df.isna().sum().reset_index()
    missing.columns = ["feature", "missing_count"]
    missing.to_csv(REPORT_SCREENSHOT_DIR / "missing_values_summary.csv", index=False)


if __name__ == "__main__":
    main()
