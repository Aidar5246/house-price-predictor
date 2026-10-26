from pathlib import Path
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib
import pandas as pd
import seaborn as sns
from house_price_predictor.data import TARGET_COLUMN

matplotlib.use("Agg")


def inspect_data(df: pd.DataFrame) -> None:
    """Print a summary of the dataset."""
    print("\n--- Dataset shape ---")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n--- Data types ---")
    print(df.dtypes)

    print("\n--- Missing values ---")
    print(df.isna().sum())

    print("\n--- Duplicate rows ---")
    print(f"Duplicates: {df.duplicated().sum()}")

    print("\n--- Descriptive statistics ---")
    print(df.describe().round(2))


def create_visualizations(df: pd.DataFrame, output_dir: Path) -> None:
    """Save basic exploratory data analysis plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.histplot(data=df, x=TARGET_COLUMN, bins=40)
    plt.title("Distribution of House Values")
    plt.xlabel("Median house value (hundreds of thousands of USD)")
    plt.ylabel("Number of observations")
    plt.tight_layout()
    plt.savefig(output_dir / "target_distribution.png", dpi=150)
    plt.close()

    correlation = df.corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=150)
    plt.close()
