from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "credit card.csv"
FIG_DIR = ROOT / "slides" / "figures"


def prepare_projection() -> pd.DataFrame:
    data = pd.read_csv(DATA_PATH)
    data = data.drop("CUST_ID", axis=1)
    data = data.ffill()

    scaled = StandardScaler().fit_transform(data)
    normalized = normalize(scaled)
    projected = PCA(n_components=2, random_state=42).fit_transform(normalized)
    return pd.DataFrame(projected, columns=["P1", "P2"])


def save_plot(points: pd.DataFrame, eps: float, min_samples: int, filename: str) -> None:
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)

    plt.figure(figsize=(6.5, 6))
    scatter = plt.scatter(
        points["P1"],
        points["P2"],
        c=labels,
        cmap="tab10",
        s=14,
        alpha=0.8,
    )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"DBSCAN on PCA Projection (min_samples={min_samples})")
    plt.grid(alpha=0.2)
    handles, legend_labels = scatter.legend_elements()
    plt.legend(handles, legend_labels, title="Cluster", loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / filename, dpi=200)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    points = prepare_projection()
    save_plot(points, eps=0.0375, min_samples=3, filename="pca_dbscan_min3.png")
    save_plot(points, eps=0.0375, min_samples=50, filename="pca_dbscan_min50.png")
    save_plot(points, eps=0.0375, min_samples=100, filename="pca_dbscan_min100.png")


if __name__ == "__main__":
    main()
