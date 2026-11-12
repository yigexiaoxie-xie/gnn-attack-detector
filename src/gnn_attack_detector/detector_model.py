"""One jointly trained GCN encoder and graph-level binary head across datasets."""
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
import copy

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from gnn_attack_detector.detector_features import NODE_FEATURE_NAMES, PREPROCESSING_VERSION, PreparedGraph
from gnn_attack_detector.detector_dataset import validate_split_integrity
from gnn_attack_detector.detector_metrics import calibrate_group_threshold
from gnn_attack_detector.device import run_with_cpu_fallback


class SharedGraphDetector(torch.nn.Module):
    def __init__(self, hidden_channels=32, dropout=.2):
        super().__init__()
        self.hidden_channels, self.dropout = int(hidden_channels), float(dropout)
        self.conv1 = GCNConv(12, self.hidden_channels)
        self.conv2 = GCNConv(self.hidden_channels, self.hidden_channels)
        self.head = torch.nn.Sequential(torch.nn.Linear(3 * self.hidden_channels + 4, self.hidden_channels),
                                        torch.nn.ReLU(), torch.nn.Dropout(self.dropout), torch.nn.Linear(self.hidden_channels, 1))
        self.register_buffer('node_mean', torch.zeros(12))
        self.register_buffer('node_std', torch.ones(12))
        self.register_buffer('global_mean', torch.zeros(4))
        self.register_buffer('global_std', torch.ones(4))

    def forward(self, graph: PreparedGraph):
        x = (graph.x - self.node_mean) / self.node_std
        x = F.relu(self.conv1(x, graph.edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, graph.edge_index))
        pooled = torch.cat((x.mean(0), x.max(0).values, torch.quantile(x, .95, dim=0),
                            (graph.global_features - self.global_mean) / self.global_std))
        return self.head(pooled).squeeze(-1)


@dataclass(frozen=True)
class GraphDetectorBundle:
    model: SharedGraphDetector
    seed: int
    threshold: float
    trained_profiles: tuple[str, ...]
    attribute_modes: dict[str, str]
    fitted_fingerprints: tuple[str, ...]
    fitted_source_groups: tuple[str, ...]

    @property
    def feature_names(self):
        return NODE_FEATURE_NAMES


@dataclass(frozen=True)
class GraphDetectorTrainingResult:
    bundle: GraphDetectorBundle
    best_epoch: int
    best_validation_loss: float
    epochs_ran: int
    threshold_metrics: dict


def balanced_sample_weights(labels, profiles=None, attacks=None):
    truth = labels.detach().cpu().long().tolist()
    if set(truth) != {0, 1}:
        raise ValueError('balanced graph loss requires clean and attacked rows')
    profiles = profiles or ('all',) * len(truth)
    attacks = attacks or tuple('attacked' if value else 'clean' for value in truth)
    if len(profiles) != len(truth) or len(attacks) != len(truth):
        raise ValueError('balanced graph loss metadata must align')
    profile_names = tuple(str(value) for value in profiles)
    attack_names = tuple(str(value) for value in attacks)
    class_groups = Counter(zip(profile_names, truth))
    if any((profile, label) not in class_groups for profile in set(profile_names) for label in (0, 1)):
        raise ValueError('each selected profile must have both clean and attacked graphs')
    attack_groups = Counter(
        (profile, attack)
        for profile, attack, label in zip(profile_names, attack_names, truth)
        if label == 1
    )
    attacks_per_profile = {
        profile: len({attack for group_profile, attack in attack_groups if group_profile == profile})
        for profile in set(profile_names)
    }
    total = len(truth)
    profile_count = len(set(profile_names))
    weights = []
    for profile, attack, label in zip(profile_names, attack_names, truth):
        if label == 0:
            denominator = profile_count * 2 * class_groups[(profile, 0)]
        else:
            denominator = profile_count * 2 * attacks_per_profile[profile] * attack_groups[(profile, attack)]
        weights.append(total / denominator)
    return torch.tensor(weights, dtype=torch.float32)


def _fit_normalization(model, dataset, indices):
    # Equal graph weight, so large PubMed graphs do not dominate scaling.
    means, seconds, globals_ = [], [], []
    for i in indices:
        graph = dataset.graph(i)
        means.append(graph.x.mean(0))
        seconds.append(graph.x.square().mean(0))
        globals_.append(graph.global_features)
    mean = torch.stack(means).mean(0)
    std = (torch.stack(seconds).mean(0) - mean.square()).clamp_min(0).sqrt()
    global_matrix = torch.stack(globals_)
    model.node_mean.copy_(mean)
    model.node_std.copy_(torch.where(std > 1e-5, std, torch.ones_like(std)))
    model.global_mean.copy_(global_matrix.mean(0))
    std = global_matrix.std(0, unbiased=False)
    model.global_std.copy_(torch.where(std > 1e-5, std, torch.ones_like(std)))


def train_graph_detector_model(dataset, *, hidden_channels=32, dropout=.2, epochs=100,
                               patience=15, learning_rate=.001, weight_decay=.0005,
                               seed=42, device='cpu', target_group_recall=.8,
                               minimum_clean_specificity=.8,
                               target_consecutive_epochs=3,
                               minimum_validation_group_size=1):
    torch.manual_seed(int(seed))
    validate_split_integrity(dataset)
    modes = {}
    for profile, mode in zip(dataset.dataset_ids, dataset.attribute_modes):
        if mode not in {'auto', 'none', 'content'} or (profile in modes and modes[profile] != mode):
            raise ValueError('inconsistent attribute policy in prepared dataset; rebuild inputs')
        modes[profile] = mode
    train = [i for i, p in enumerate(dataset.partitions) if p == 'train']
    val = [i for i, p in enumerate(dataset.partitions) if p == 'val']
    if not train or not val or epochs < 1 or patience < 1:
        raise ValueError('nonempty train/val and positive epochs/patience are required')
    if target_consecutive_epochs < 1 or minimum_validation_group_size < 1:
        raise ValueError('group stopping counts must be positive')
    weights = balanced_sample_weights(
        dataset.y[train],
        tuple(dataset.dataset_ids[i] for i in train),
        tuple(dataset.attack_names[i] for i in train),
    )
    val_weights = balanced_sample_weights(
        dataset.y[val],
        tuple(dataset.dataset_ids[i] for i in val),
        tuple(dataset.attack_names[i] for i in val),
    )
    model = SharedGraphDetector(hidden_channels, dropout)
    _fit_normalization(model, dataset, train)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state, best_loss, best_epoch = None, float('inf'), 0
    best_calibration, best_rank = None, None
    target_streak = 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, epochs + 1):
        model.train()
        # Accumulate one graph's backward pass at a time: memory does not grow
        # with corpus size, and weighted full-epoch gradients are order independent.
        optimizer.zero_grad(set_to_none=True)
        for position, j in enumerate(torch.randperm(len(train), generator=generator).tolist(), 1):
            i = train[j]
            logit = model(dataset.graph(i).to(device))
            loss = F.binary_cross_entropy_with_logits(logit, dataset.y[i].float().to(device))
            (loss * float(weights[j]) / len(train)).backward()
            if position % 25 == 0:
                print(f'[DETECTOR] epoch={epoch} graphs={position}/{len(train)}', flush=True)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = torch.stack([model(dataset.graph(i).to(device)).cpu() for i in val])
            val_loss = float(F.binary_cross_entropy_with_logits(val_logits, dataset.y[val].float(), weight=val_weights))
        if not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError('nonfinite detector validation loss')
        calibration = calibrate_group_threshold(
            val_logits.sigmoid(),
            dataset.y[val],
            tuple(dataset.dataset_ids[i] for i in val),
            tuple(dataset.attack_names[i] for i in val),
            minimum_group_recall=float(target_group_recall),
            minimum_clean_specificity=float(minimum_clean_specificity),
            minimum_group_size=int(minimum_validation_group_size),
        )
        rank = (
            int(calibration['clean_target_met']),
            float(calibration['worst_group_recall']),
            int(calibration['groups_at_target']),
            float(calibration['macro_group_recall']),
            float(calibration['clean_specificity']),
            -val_loss,
        )
        if best_rank is None or rank > best_rank:
            best_rank, best_loss, best_epoch = rank, val_loss, epoch
            best_calibration = calibration
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        target_streak = target_streak + 1 if calibration['target_met'] else 0
        print(
            f"[DETECTOR] epoch={epoch} validation_loss={val_loss:.6f} "
            f"threshold={calibration['threshold']:.6f} "
            f"worst_group_recall={calibration['worst_group_recall']:.4f} "
            f"groups_at_target={calibration['groups_at_target']}/{calibration['total_groups']} "
            f"clean_specificity={calibration['clean_specificity']:.4f} "
            f"target_streak={target_streak}/{target_consecutive_epochs}",
            flush=True,
        )
        if target_streak >= target_consecutive_epochs:
            break
    if best_state is None or best_calibration is None:
        raise RuntimeError('detector training produced no validation checkpoint')
    model.load_state_dict(best_state)
    fitted = [i for i in train + val if not (dataset.source_groups[i].startswith('benchmark:')
              and int(dataset.y[i]) == 0 and not bool(dataset.independent_clean[i]))]
    bundle = GraphDetectorBundle(model.cpu(), seed, float(best_calibration['threshold']),
                                 tuple(sorted(set(dataset.dataset_ids[i] for i in train))), modes,
                                 tuple(sorted({dataset.input_fingerprints[i] for i in fitted if dataset.input_fingerprints[i]})),
                                 tuple(sorted({dataset.source_groups[i] for i in fitted if dataset.source_groups[i]})))
    threshold_metrics = dict(best_calibration)
    threshold_metrics.pop('metrics', None)
    return GraphDetectorTrainingResult(bundle, best_epoch, best_loss, epoch, threshold_metrics)


def predict_graph_probabilities(bundle, graphs, *, device='auto'):
    def predict(selected_device):
        model = copy.deepcopy(bundle.model).to(selected_device).eval()
        values = []
        with torch.no_grad():
            for i in range(len(graphs)):
                graph = graphs.graph(i) if hasattr(graphs, 'graph') else graphs[i]
                values.append(model(graph.to(selected_device)).sigmoid().cpu())
        return torch.stack(values) if values else torch.empty(0)
    return run_with_cpu_fallback(predict, requested=str(device), label='detector inference')


def save_graph_detector_bundle(path, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = result.bundle
    temporary = path.with_suffix('.tmp')
    torch.save({'format_version': 4, 'architecture': 'shared_gcn_reference_free',
                'detector_unit': 'complete_graph', 'feature_names': NODE_FEATURE_NAMES,
                'hidden_channels': bundle.model.hidden_channels, 'dropout': bundle.model.dropout,
                'seed': bundle.seed, 'threshold': bundle.threshold, 'trained_profiles': bundle.trained_profiles,
                'preprocessing_version': PREPROCESSING_VERSION, 'attribute_modes': bundle.attribute_modes,
                'fitted_fingerprints': bundle.fitted_fingerprints, 'fitted_source_groups': bundle.fitted_source_groups,
                'state_dict': {k: v.detach().cpu() for k, v in bundle.model.state_dict().items()},
                'best_epoch': result.best_epoch, 'best_validation_loss': result.best_validation_loss,
                'epochs_ran': result.epochs_ran, 'threshold_metrics': result.threshold_metrics}, temporary)
    temporary.replace(path)


def load_graph_detector_bundle(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('format_version') != 4 or payload.get('architecture') != 'shared_gcn_reference_free':
        raise RuntimeError('obsolete detector checkpoint; run rebuild-detector to create a current active model')
    if tuple(payload['feature_names']) != NODE_FEATURE_NAMES or not 0 <= float(payload['threshold']) <= 1:
        raise ValueError('invalid detector schema or threshold')
    if payload.get('preprocessing_version') != PREPROCESSING_VERSION:
        raise ValueError('unsupported detector preprocessing version; rebuild detector')
    model = SharedGraphDetector(payload['hidden_channels'], payload['dropout'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    return GraphDetectorBundle(model, int(payload['seed']), float(payload['threshold']), tuple(payload['trained_profiles']),
                               dict(payload['attribute_modes']), tuple(payload['fitted_fingerprints']), tuple(payload['fitted_source_groups']))
