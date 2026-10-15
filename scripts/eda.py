"""Exploratory data analysis: prints summaries and writes figures to reports/figures."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from src import config  # noqa: E402
from src.data_processing import build_clean_dataset, download_data, load_raw  # noqa: E402

sns.set_theme(style="whitegrid")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=130)
    plt.close(fig)


def main():
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    download_data()
    raw = load_raw()
    missing = raw.isna().sum()
    print("missing values in raw data:")
    print(missing[missing > 0])
    df = build_clean_dataset()
    print(df.describe().T.round(2))

    # class balance
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.countplot(x=config.TARGET, data=df, ax=ax, hue=config.TARGET, legend=False)
    ax.set_xticks([0, 1], ["No disease", "Disease"])
    ax.set_title("Class balance")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    save(fig, "class_balance.png")

    # histograms of numeric features
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), config.NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=config.TARGET, kde=True, ax=ax, bins=20)
        ax.set_title(f"Distribution of {col}")
    axes.ravel()[-1].axis("off")
    save(fig, "histograms.png")

    # correlation heatmap
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, annot_kws={"size": 7})
    ax.set_title("Feature correlation matrix")
    save(fig, "correlation_heatmap.png")

    # boxplots by target
    fig, axes = plt.subplots(1, 5, figsize=(15, 4))
    for ax, col in zip(axes, config.NUMERIC_FEATURES):
        sns.boxplot(data=df, x=config.TARGET, y=col, ax=ax, hue=config.TARGET, legend=False)
    save(fig, "boxplots_by_target.png")

    # categorical features vs target
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), config.CATEGORICAL_FEATURES):
        sns.countplot(data=df, x=col, hue=config.TARGET, ax=ax)
        ax.set_title(f"{col} vs target")
    save(fig, "categorical_vs_target.png")
    print("figures saved to", config.FIGURES_DIR)


if __name__ == "__main__":
    main()
