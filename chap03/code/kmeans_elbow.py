from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sales data.csv"
FIG_DIR = ROOT / "slides" / "figures"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(DATA_PATH)
    categorical_features = ["Channel", "Region"]

    for column in categorical_features:
        dummies = pd.get_dummies(data[column], prefix=column)
        data = pd.concat([data, dummies], axis=1)
        data.drop(column, axis=1, inplace=True)

    scaler = MinMaxScaler()
    transformed = scaler.fit_transform(data)

    inertias = []
    cluster_range = range(1, 15)
    for k in cluster_range:
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        model.fit(transformed)
        inertias.append(model.inertia_)

    plt.figure(figsize=(7, 4))
    plt.plot(cluster_range, inertias, "o-", color="#1f77b4", linewidth=2)
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("K-means Elbow Curve")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kmeans_elbow.png", dpi=200)


if __name__ == "__main__":
    main()
