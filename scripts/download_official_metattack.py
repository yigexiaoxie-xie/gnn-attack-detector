"""Download four official DeepRobust/Pro-GNN Metattack graphs for one profile.

The files are sparse adjacency matrices. This script ignores broken system
proxy settings, validates every cached/downloaded file, and never replaces a
valid cached graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import urllib.request

import numpy as np
import scipy.sparse as sparse


RATES = ("0.05", "0.1", "0.15", "0.2")
BASE_URL = "https://raw.githubusercontent.com/ChandlerBang/Pro-GNN/master/meta"
SUPPORTED_DATASETS = frozenset({"cora", "citeseer", "polblogs", "pubmed"})


def validate_adjacency(path: Path) -> tuple[int, int]:
    adjacency = sparse.load_npz(path).tocsr()
    adjacency.eliminate_zeros()
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Metattack adjacency is not square: {path}")
    if adjacency.shape[0] <= 0 or adjacency.nnz <= 0:
        raise ValueError(f"Metattack adjacency is empty: {path}")
    if not np.isfinite(adjacency.data).all():
        raise ValueError(f"Metattack adjacency contains non-finite values: {path}")
    difference = (adjacency - adjacency.T).tocsr()
    difference.eliminate_zeros()
    if difference.nnz:
        raise ValueError(f"Metattack adjacency is not symmetric: {path}")
    return int(adjacency.shape[0]), int(adjacency.nnz // 2)


def _download(url: str, destination: Path, timeout: int) -> None:
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "gnn-attack-detector/1.0"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            temporary.write_bytes(response.read())
        validate_adjacency(temporary)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def prepare_official_metattack(
    destination: Path,
    dataset: str,
    timeout: int = 60,
) -> None:
    name = str(dataset).lower()
    if name not in SUPPORTED_DATASETS:
        raise ValueError(
            f"official Metattack files are unavailable for dataset: {dataset}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    for rate in RATES:
        filename = f"{name}_meta_adj_{rate}.npz"
        target = destination / filename
        if target.exists():
            nodes, edges = validate_adjacency(target)
            print(f"[REUSED] {filename} nodes={nodes} undirected_edges={edges}")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"[DOWNLOADING] {url}")
        try:
            _download(url, target, timeout)
        except Exception as exc:
            raise RuntimeError(
                "Could not download the official Metattack graph. Download "
                f"{url} on another computer and copy it to {target}."
            ) from exc
        nodes, edges = validate_adjacency(target)
        print(f"[DOWNLOADED] {filename} nodes={nodes} undirected_edges={edges}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("artifacts/raw/deeprobust/meta"),
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--timeout", type=int, default=60)
    arguments = parser.parse_args()
    dataset = arguments.dataset
    if dataset is None:
        from gnn_attack_detector.config import load_config

        config = load_config(Path("configs/base.yaml"))
        profile = config["profiles"][config["active_profile"]]
        dataset = str(profile["name"]).lower()
    prepare_official_metattack(arguments.destination, dataset, arguments.timeout)
    print(f"Official {dataset} Metattack graphs are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
