"""Portable complete-graph file input and output helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import pickle
import zipfile
import math
from types import SimpleNamespace

import numpy as np
import torch


SUPPORTED_GRAPH_SUFFIXES = frozenset({".pt", ".npz"})
NPZ_KEYS = frozenset({"x", "edge_index", "y", "attribute_mode"})
SCIPY_CSR_KEYS = frozenset({"data", "indices", "indptr", "format", "shape"})
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_NODES = 200_000
MAX_DIRECTED_EDGES = 5_000_000


def _validate_graph_tensors(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor | None,
) -> None:
    if x.dtype != torch.float32 or x.ndim != 2:
        raise ValueError("x must be a two-dimensional float32 array")
    if not 0 < int(x.shape[0]) <= MAX_NODES:
        raise ValueError(f"graph must have between 1 and {MAX_NODES} nodes")
    if not bool(torch.isfinite(x).all()):
        raise ValueError("x must contain only finite values")
    if (
        edge_index.dtype != torch.long
        or edge_index.ndim != 2
        or edge_index.shape[0] != 2
    ):
        raise ValueError(
            "edge_index must be an int64 array with shape [2, num_edges]"
        )
    if edge_index.shape[1] > MAX_DIRECTED_EDGES:
        raise ValueError("graph exceeds the directed edge limit")
    if y is not None and (y.dtype != torch.long or y.ndim != 1 or y.numel() != x.shape[0]):
        raise ValueError(
            "y must be a one-dimensional int64 array with one label per node"
        )
    if edge_index.numel() and (
        int(edge_index.min()) < 0 or int(edge_index.max()) >= int(x.shape[0])
    ):
        raise ValueError("edge_index contains a node outside the graph")
    directed = [tuple(map(int, pair)) for pair in edge_index.t().tolist()]
    if any(left == right for left, right in directed):
        raise ValueError("self-loops are not accepted")
    if len(set(directed)) != len(directed):
        raise ValueError("duplicate directed edges are not accepted")
    directed_set = set(directed)
    if any((right, left) not in directed_set for left, right in directed_set):
        raise ValueError("the undirected edge list must contain both directions")


def graph_to_npz_bytes(graph: Any) -> bytes:
    """Serialize one complete graph into the documented safe NPZ contract."""

    x = graph.x.detach().cpu().to(torch.float32).contiguous()
    edge_index = graph.edge_index.detach().cpu().to(torch.long).contiguous()
    y = getattr(graph, "y", None)
    y = y.detach().cpu().to(torch.long).contiguous() if y is not None else None
    _validate_graph_tensors(x, edge_index, y)
    buffer = BytesIO()
    arrays = dict(x=x.numpy(), edge_index=edge_index.numpy())
    if y is not None:
        arrays["y"] = y.numpy()
    mode = getattr(graph, "attribute_mode", "auto")
    if mode not in {"auto", "none", "content"}:
        raise ValueError("invalid attribute_mode")
    if mode != "auto":
        arrays["attribute_mode"] = np.array(mode)
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _load_scipy_csr_npz(path: Path, clean_graph: Any) -> Any:
    if clean_graph is None:
        raise ValueError("CSR-only graph lacks attributes; export a complete x/edge_index NPZ from the source project")
    try:
        import scipy.sparse as sp

        adjacency = sp.load_npz(path).tocsr(copy=True)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid SciPy CSR adjacency: {exc}") from exc
    expected_nodes = int(clean_graph.num_nodes)
    if adjacency.shape != (expected_nodes, expected_nodes):
        raise ValueError(
            f"adjacency has shape {adjacency.shape}; trusted graph has "
            f"{expected_nodes} nodes"
        )
    adjacency.sum_duplicates()
    adjacency.eliminate_zeros()
    if adjacency.diagonal().any():
        raise ValueError("SciPy CSR adjacency must not contain self-loops")
    if (adjacency != adjacency.T).nnz:
        raise ValueError("SciPy CSR adjacency must be symmetric")
    if adjacency.nnz and not np.all(adjacency.data == 1):
        raise ValueError("SciPy CSR adjacency must be unweighted and binary")
    coordinates = adjacency.tocoo()
    order = np.lexsort((coordinates.col, coordinates.row))
    edge_array = np.vstack(
        (coordinates.row[order], coordinates.col[order])
    ).astype(np.int64, copy=False)
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise RuntimeError("torch-geometric is required to load graph files") from exc
    graph = Data(
        x=clean_graph.x.detach().cpu().to(torch.float32).clone(),
        edge_index=torch.from_numpy(edge_array).long().contiguous(),
        y=clean_graph.y.detach().cpu().to(torch.long).clone(),
    )
    _validate_graph_tensors(graph.x, graph.edge_index, graph.y)
    return graph


def _check_npz_limits(path: Path) -> None:
    """Read ZIP/NPY headers, never materialize unbounded compressed arrays."""
    if not zipfile.is_zipfile(path):
        raise ValueError("NPZ must be a ZIP archive of NPY arrays")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if sum(item.file_size for item in members) > MAX_EXPANDED_BYTES:
            raise ValueError("expanded NPZ graph exceeds the memory limit")
        if len({item.filename for item in members}) != len(members):
            raise ValueError("duplicate NPZ entries are not accepted")
        for item in members:
            if not item.filename.endswith('.npy'):
                raise ValueError("NPZ must contain only NPY arrays")
            with archive.open(item) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError("unsupported NPY header version; export standard numeric NPZ")
                if dtype.hasobject:
                    raise ValueError("object/pickle arrays are not accepted")
                if any(dimension < 0 for dimension in shape):
                    raise ValueError("invalid declared NPY shape")
                byte_count = math.prod(shape) * dtype.itemsize
                if byte_count > MAX_EXPANDED_BYTES or byte_count > item.file_size - stream.tell():
                    raise ValueError("declared NPY array exceeds available bytes or expanded memory limit")
                if item.filename == 'x.npy' and len(shape) == 2 and shape[0] > MAX_NODES:
                    raise ValueError("declared node count exceeds graph limit")
                if item.filename == 'edge_index.npy' and len(shape) == 2 and shape[1] > MAX_DIRECTED_EDGES:
                    raise ValueError("declared edge count exceeds graph limit")


def _load_npz(path: Path, clean_graph: Any) -> Any:
    _check_npz_limits(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = frozenset(archive.files)
            if keys == SCIPY_CSR_KEYS:
                return _load_scipy_csr_npz(path, clean_graph)
            if not {"x", "edge_index"} <= keys or keys - NPZ_KEYS:
                missing = sorted({"x", "edge_index"} - keys)
                unexpected = sorted(keys - NPZ_KEYS)
                raise ValueError(
                    "NPZ keys are invalid; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            x_array = archive["x"]
            edge_array = archive["edge_index"]
            y_array = archive["y"] if "y" in keys else None
            attribute_mode = str(archive["attribute_mode"].item()) if "attribute_mode" in keys else "auto"
            if attribute_mode not in {"auto", "none", "content"}:
                raise ValueError("invalid attribute_mode")
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid NPZ graph: {path}: {exc}") from exc
    if (
        x_array.dtype != np.float32
        or edge_array.dtype != np.int64
        or (y_array is not None and y_array.dtype != np.int64)
    ):
        raise ValueError(
            "NPZ dtypes must be x=float32, edge_index=int64, y=int64"
        )
    try:
        from torch_geometric.data import Data
    except ImportError as exc:
        raise RuntimeError("torch-geometric is required to load graph files") from exc
    graph = Data(
        x=torch.from_numpy(x_array.copy()),
        edge_index=torch.from_numpy(edge_array.copy()),
        y=torch.from_numpy(y_array.copy()) if y_array is not None else None,
        attribute_mode=attribute_mode,
    )
    _validate_graph_tensors(graph.x, graph.edge_index, graph.y)
    return graph


class _GraphUnpickler(pickle.Unpickler):
    """Narrow legacy PyG compatibility; uploaded pickle must not execute code."""
    def find_class(self, module, name):
        allowed = {
            ("collections", "OrderedDict"),
            ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor"),
            ("torch_geometric.data.data", "Data"),
            ("torch_geometric.data.data", "DataEdgeAttr"),
            ("torch_geometric.data.data", "DataTensorAttr"),
            ("torch_geometric.data.storage", "GlobalStorage"),
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f"unsupported graph pickle type: {module}.{name}; export tensor-only PT or NPZ")
        return super().find_class(module, name)


def load_pt_payload(path):
    # Refuse raw-pickle/tar legacy formats before any deserialization. TorchScript
    # archives are executable models, not graph data, and are not accepted either.
    if not zipfile.is_zipfile(path):
        raise ValueError("PT must use the torch.save ZIP format; export older files to NPZ in a trusted environment")
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if any(item.filename.endswith("/constants.pkl") for item in members):
            raise ValueError("TorchScript models are not accepted as graph inputs")
        if sum(item.file_size for item in members) > MAX_EXPANDED_BYTES:
            raise ValueError("expanded PT graph exceeds 512 MB")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        restricted = SimpleNamespace(__name__="restricted_graph_pickle", Unpickler=_GraphUnpickler,
                                     load=lambda stream, **kwargs: _GraphUnpickler(stream, **kwargs).load())
        return torch.load(path, map_location="cpu", pickle_module=restricted, weights_only=False)


def _load_pt(path: Path, clean_graph: Any) -> Any:
    payload = load_pt_payload(path)
    if isinstance(payload, dict) and payload.get("format_version") in (2, 3):
        from gnn_attack_detector.integrity import graph_fingerprint
        if clean_graph is None:
            raise ValueError("legacy compact graph requires its source attributes; export complete NPZ first")
        if payload.get("clean_fingerprint") != graph_fingerprint(clean_graph):
            raise ValueError("compact graph source fingerprint does not match")
        graph = clean_graph.clone().cpu()
        graph.edge_index = payload["edge_index"]
        if payload.get("x") is not None:
            graph.x = payload["x"]
    elif isinstance(payload, dict) and "edge_index" in payload and ("x" in payload or "num_nodes" in payload):
        try:
            from torch_geometric.data import Data
        except ImportError as exc:
            raise RuntimeError(
                "torch-geometric is required to load graph dictionaries"
            ) from exc
        graph = Data(
            x=payload.get("x", torch.empty((int(payload.get("num_nodes", 0)), 0))),
            edge_index=payload["edge_index"],
            y=payload.get("y"),
            attribute_mode=payload.get("attribute_mode", "auto"),
        )
    elif all(
        hasattr(payload, name) for name in ("x", "edge_index", "y", "num_nodes")
    ):
        graph = payload.cpu()
    else:
        raise RuntimeError(
            "input .pt must contain one PyG graph or a supported graph dictionary"
        )
    _validate_graph_tensors(graph.x, graph.edge_index, graph.y)
    if int(graph.num_nodes) != int(graph.x.shape[0]):
        raise ValueError("num_nodes must match attribute rows")
    return graph


def load_graph_file(path: str | Path, *, clean_graph: Any = None) -> Any:
    """Load one validated `.pt` or safe `.npz` complete graph."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"graph file not found: {source}")
    suffix = source.suffix.lower()
    if suffix not in SUPPORTED_GRAPH_SUFFIXES:
        raise ValueError("graph file must use .pt or .npz")
    if suffix == ".npz":
        return _load_npz(source, clean_graph)
    return _load_pt(source, clean_graph)
