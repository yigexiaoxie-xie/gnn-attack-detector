"""Integration coverage for the reference-free replacement, not the retired MLP."""
from pathlib import Path
from unittest.mock import patch
import pickle
import io
import zipfile
import numpy as np

import pytest
import torch
from torch_geometric.data import Data

from gnn_attack_detector.config import load_config
from gnn_attack_detector.corpus import save_compact_graph
from gnn_attack_detector.integrity import graph_fingerprint, hash_file
from gnn_attack_detector.detector_dataset import dataset_from_records, save_graph_detector_dataset
from gnn_attack_detector.detector_features import prepare_graph_input, save_prepared_graph
from gnn_attack_detector.detector_model import (
    balanced_sample_weights,
    load_graph_detector_bundle,
    predict_graph_probabilities,
    train_graph_detector_model,
)
from gnn_attack_detector.graph_io import load_graph_file, graph_to_npz_bytes
from gnn_attack_detector.pipeline import (
    _write_jsonl, _available_profile_datasets, _build_profile_dataset, _profile_feature_path,
    rebuild_detector, activate_detector, detect_graph, _official_for_rows,
    _load_detection_input,
)
from gnn_attack_detector.profiles import load_profiles
from gnn_attack_detector.platform_service import load_platform_evaluation

ROOT = Path(__file__).resolve().parents[1]


def graph():
    return Data(x=torch.tensor([[1., 0., .2], [0., 1., .1], [1., 0., .8], [0., 1., .7]]),
                edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]), y=torch.tensor([0, 1, 0, 1]))


def test_loss_weights_balance_datasets_classes_and_attacks():
    labels = torch.tensor([0, 0, 1, 1, 1, 1, 1, 0, 1, 1])
    profiles = ("a", "a", "a", "a", "a", "a", "a", "b", "b", "b")
    attacks = ("clean", "clean", "x", "x", "x", "x", "y", "clean", "x", "x")

    weights = balanced_sample_weights(labels, profiles, attacks)

    def total(indices):
        return float(weights[torch.tensor(indices)].sum())

    assert total(range(7)) == pytest.approx(5.0)
    assert total((7, 8, 9)) == pytest.approx(5.0)
    assert total((0, 1)) == pytest.approx(2.5)
    assert total((2, 3, 4, 5)) == pytest.approx(1.25)
    assert total((6,)) == pytest.approx(1.25)
    assert total((7,)) == pytest.approx(2.5)
    assert total((8, 9)) == pytest.approx(2.5)


def test_training_stops_after_consecutive_group_targets(tmp_path):
    prepared_path = tmp_path / 'prepared.pt'
    save_prepared_graph(prepared_path, prepare_graph_input(graph()))
    records = [
        dict(dataset_id='a', label=0, sample_id='train-clean', partition='train', attack_name='clean'),
        dict(dataset_id='a', label=1, sample_id='train-attack', partition='train', attack_name='hard'),
        dict(dataset_id='a', label=0, sample_id='val-clean', partition='val', attack_name='clean'),
        dict(dataset_id='a', label=1, sample_id='val-attack', partition='val', attack_name='hard'),
    ]
    dataset = dataset_from_records(records, [prepared_path] * len(records))

    result = train_graph_detector_model(
        dataset,
        hidden_channels=4,
        dropout=0,
        epochs=5,
        patience=2,
        learning_rate=0.001,
        weight_decay=0,
        seed=42,
        device='cpu',
        target_group_recall=0,
        minimum_clean_specificity=0,
        target_consecutive_epochs=2,
        minimum_validation_group_size=1,
    )

    assert result.epochs_ran == 2
    assert result.threshold_metrics['target_met'] is True
    assert result.threshold_metrics['worst_group_recall'] >= 0


def test_training_runs_to_epoch_limit_when_group_target_is_impossible(tmp_path):
    prepared_path = tmp_path / 'prepared.pt'
    save_prepared_graph(prepared_path, prepare_graph_input(graph()))
    records = [
        dict(dataset_id='a', label=0, sample_id='train-clean', partition='train', attack_name='clean'),
        dict(dataset_id='a', label=1, sample_id='train-attack', partition='train', attack_name='hard'),
        dict(dataset_id='a', label=0, sample_id='val-clean', partition='val', attack_name='clean'),
        dict(dataset_id='a', label=1, sample_id='val-attack', partition='val', attack_name='hard'),
    ]
    dataset = dataset_from_records(records, [prepared_path] * len(records))

    result = train_graph_detector_model(
        dataset,
        hidden_channels=4,
        dropout=0,
        epochs=2,
        patience=1,
        learning_rate=0.001,
        weight_decay=0,
        seed=42,
        device='cpu',
        target_group_recall=1,
        minimum_clean_specificity=1,
        target_consecutive_epochs=2,
        minimum_validation_group_size=1,
    )

    assert result.epochs_ran == 2
    assert result.threshold_metrics['target_met'] is False


def test_atomic_report_failure_preserves_existing(tmp_path):
    path = tmp_path / 'predictions.jsonl'
    path.write_text('previous\n')
    with patch.object(Path, 'replace', side_effect=OSError('replace failed')):
        with pytest.raises(OSError):
            _write_jsonl(path, [{'sample_id': 'new'}])
    assert path.read_text() == 'previous\n'


def test_foreign_manifest_rejected_before_loading():
    config = load_config(ROOT / 'configs/base.yaml')
    with pytest.raises(RuntimeError, match='another dataset profile'):
        _build_profile_dataset(ROOT, config, {}, load_profiles(config['profiles'])['cora-lcc'], [{'dataset_id': 'polblogs-lcc'}])


def test_dataset_selection_and_foreign_id(tmp_path):
    config = load_config(ROOT / 'configs/base.yaml')
    cache = tmp_path / 'graph.pt'
    save_prepared_graph(cache, prepare_graph_input(graph()))
    for name in ('cora-lcc', 'citeseer-lcc', 'polblogs-lcc'):
        dataset = dataset_from_records([dict(dataset_id=name, label=1, sample_id=name, partition='train')], [cache])
        save_graph_detector_dataset(_profile_feature_path(tmp_path, config, name), dataset)
    selected = _available_profile_datasets(tmp_path, config, profile_ids=('cora-lcc', 'citeseer-lcc'))
    assert [d.dataset_ids[0] for d in selected] == ['cora-lcc', 'citeseer-lcc']
    assert selected[0].graph(0).x.shape == (4, 12)
    save_graph_detector_dataset(_profile_feature_path(tmp_path, config, 'cora-lcc'), dataset)
    with pytest.raises(RuntimeError, match='another dataset profile'):
        _available_profile_datasets(tmp_path, config, profile_ids=('cora-lcc',))


def test_unlabelled_legacy_pyg_and_compact_upload(tmp_path):
    g = graph()
    path = tmp_path / 'pyg.pt'
    torch.save(g, path)
    assert torch.equal(load_graph_file(path).x, g.x)
    compact = tmp_path / 'compact.pt'
    attacked = g.clone()
    attacked.x[0, 0] = .5
    save_compact_graph(compact, attacked, graph_fingerprint(g), {'dataset_id': 'cora-lcc'})
    assert torch.equal(load_graph_file(compact, clean_graph=g).x, attacked.x)
    with pytest.raises(ValueError, match='source attributes'):
        load_graph_file(compact)
    g.y = None
    npz = tmp_path / 'standalone.npz'
    npz.write_bytes(graph_to_npz_bytes(g))
    assert load_graph_file(npz).y is None
    g.attribute_mode = 'none'
    npz.write_bytes(graph_to_npz_bytes(g))
    assert load_graph_file(npz).attribute_mode == 'none'
    assert torch.count_nonzero(prepare_graph_input(load_graph_file(npz)).global_features) == 0


def test_detection_profile_supplies_attributes_for_csr_npz(tmp_path):
    import scipy.sparse as sp

    config = load_config(ROOT / 'configs/base.yaml')
    config['profiles']['cora-lcc']['expected_num_nodes'] = 4
    profile = load_profiles(config['profiles'])['cora-lcc']
    clean = graph()
    clean_path = tmp_path / profile.clean_reference
    clean_path.parent.mkdir(parents=True)
    torch.save(clean, clean_path)
    adjacency = sp.csr_matrix(
        (np.ones(clean.edge_index.shape[1], dtype=np.float32),
         (clean.edge_index[0].numpy(), clean.edge_index[1].numpy())),
        shape=(clean.num_nodes, clean.num_nodes),
    )
    upload = tmp_path / 'attacked-adjacency.npz'
    sp.save_npz(upload, adjacency)

    loaded, provenance = _load_detection_input(
        tmp_path, config, upload, profile_id='cora-lcc',
    )

    assert provenance == 'cora-lcc'
    assert torch.equal(loaded.x, clean.x)
    assert torch.equal(loaded.edge_index, clean.edge_index)


def test_detection_profile_rejects_wrong_node_count_and_preserves_complete_x(tmp_path):
    config = load_config(ROOT / 'configs/base.yaml')
    config['profiles']['cora-lcc']['expected_num_nodes'] = 4
    profile = load_profiles(config['profiles'])['cora-lcc']
    clean = graph()
    clean_path = tmp_path / profile.clean_reference
    clean_path.parent.mkdir(parents=True)
    torch.save(clean, clean_path)

    complete = graph()
    complete.x = complete.x + 7
    complete_path = tmp_path / 'complete.npz'
    complete_path.write_bytes(graph_to_npz_bytes(complete))
    loaded, _ = _load_detection_input(
        tmp_path, config, complete_path, profile_id='cora-lcc',
    )
    assert torch.equal(loaded.x, complete.x)

    wrong = Data(
        x=torch.ones(5, 3),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        y=None,
    )
    wrong_path = tmp_path / 'wrong-nodes.npz'
    wrong_path.write_bytes(graph_to_npz_bytes(wrong))
    with pytest.raises(ValueError, match='selected dataset has 4 nodes'):
        _load_detection_input(
            tmp_path, config, wrong_path, profile_id='cora-lcc',
        )


def test_detection_profile_rejects_corrupt_trusted_node_count(tmp_path):
    config = load_config(ROOT / 'configs/base.yaml')
    config['profiles']['cora-lcc']['expected_num_nodes'] = 5
    profile = load_profiles(config['profiles'])['cora-lcc']
    clean = graph()
    clean_path = tmp_path / profile.clean_reference
    clean_path.parent.mkdir(parents=True)
    torch.save(clean, clean_path)
    upload = tmp_path / 'complete.npz'
    upload.write_bytes(graph_to_npz_bytes(graph()))

    with pytest.raises(RuntimeError, match='trusted graph has 4 nodes; profile expects 5'):
        _load_detection_input(
            tmp_path, config, upload, profile_id='cora-lcc',
        )


class _Malicious:
    def __reduce__(self):
        return eval, ('1 + 1',)


def test_upload_rejects_arbitrary_pickle(tmp_path):
    path = tmp_path / 'unsafe.pt'
    torch.save(_Malicious(), path)
    with pytest.raises(pickle.UnpicklingError):
        load_graph_file(path)


def test_nonzip_pickle_is_rejected_before_deserialization(tmp_path):
    path = tmp_path / 'legacy.pt'
    path.write_bytes(pickle.dumps(_Malicious()))
    with pytest.raises(ValueError, match='ZIP'):
        load_graph_file(path)


def test_npz_declared_shape_is_checked_before_materializing(tmp_path):
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(header, {'descr': '<f4', 'fortran_order': False, 'shape': (2**30, 2)})
    path = tmp_path / 'oversized.npz'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('x.npy', header.getvalue())
    with patch('gnn_attack_detector.graph_io.np.load', side_effect=AssertionError('must check header before np.load')):
        with pytest.raises(ValueError, match='expanded|declared'):
            load_graph_file(path)


def test_npz_expansion_limit_is_checked_first(tmp_path, monkeypatch):
    import gnn_attack_detector.graph_io as module
    monkeypatch.setattr(module, 'MAX_EXPANDED_BYTES', 64, raising=False)
    path = tmp_path / 'compressed.npz'
    path.write_bytes(graph_to_npz_bytes(graph()))
    with patch.object(module.np, 'load', side_effect=AssertionError('must check size before np.load')):
        with pytest.raises(ValueError, match='expanded'):
            load_graph_file(path)


def test_same_graph_with_reordered_edges_and_changed_labels_cannot_cross_splits(tmp_path):
    config = load_config(ROOT / 'configs/base.yaml')
    create_corpus(tmp_path, config, 'cora-lcc')
    g = graph()
    first = tmp_path / 'train.pt'
    second = tmp_path / 'val.pt'
    torch.save(g, first)
    g.edge_index = g.edge_index.flip(1)
    g.y = g.y.flip(0)
    torch.save(g, second)
    records = [dict(dataset_id='cora-lcc', label=1, sample_id=p, partition=p,
                    attack_name='random_add', graph_path=f'{p}.pt', graph_hash=p) for p in ('train', 'val')]
    with pytest.raises(ValueError, match='across partitions'):
        _build_profile_dataset(tmp_path, config, {}, load_profiles(config['profiles'])['cora-lcc'], records)


def create_corpus(root, config, profile_id):
    g = graph()
    profile = load_profiles(config['profiles'])[profile_id]
    clean_path = root / profile.clean_reference
    clean_path.parent.mkdir(parents=True)
    torch.save(g, clean_path)
    records = []
    for i, partition in enumerate(('train', 'val', 'test')):
        attacked = g.clone()
        attacked.x[0, 0] = .2 + i * .05
        path = root / f'artifacts/attacks/{profile_id}-{partition}.pt'
        record = dict(dataset_id=profile_id, sample_id=f'{profile_id}-{partition}', partition=partition,
                      label=1, attack_name='random_add', strength_value=.05, strength_unit='edge_rate',
                      graph_hash=graph_fingerprint(attacked)['combined'], graph_path=path.relative_to(root).as_posix())
        save_compact_graph(path, attacked, graph_fingerprint(g), record)
        records.append(record)
    _write_jsonl(root / config['corpus']['sample_index_dir'] / f'{profile_id}.jsonl', records)
    return records


def test_negative_independence_requires_provenance_and_preserves_false(tmp_path):
    config = load_config(ROOT / 'configs/base.yaml')
    records = create_corpus(tmp_path, config, 'cora-lcc')
    config['detector']['nonattack_manifest'] = 'negatives.jsonl'
    negatives = []
    for i, partition in enumerate(('train', 'val', 'test')):
        g = graph()
        g.x[0, 0] = 1.1 + i * .2
        path = tmp_path / f'negative-{partition}.pt'
        torch.save(g, path)
        negatives.append(dict(dataset_id='cora-lcc', sample_id=f'n-{partition}', label=0, partition=partition,
                              graph_path=path.name, source_group=f'source-{partition}', independent_clean=False))
    _write_jsonl(tmp_path / 'negatives.jsonl', negatives)
    dataset = _build_profile_dataset(tmp_path, config, {}, load_profiles(config['profiles'])['cora-lcc'], records)
    assert dataset.independent_clean.sum() == 0
    negatives[2]['independent_clean'] = True
    _write_jsonl(tmp_path / 'negatives.jsonl', negatives)
    with pytest.raises(ValueError, match='independence_evidence'):
        _build_profile_dataset(tmp_path, config, {}, load_profiles(config['profiles'])['cora-lcc'], records)
    negatives[2]['independence_evidence'] = 'independent externally collected source'
    negatives[2]['source_group'] = negatives[0]['source_group']
    _write_jsonl(tmp_path / 'negatives.jsonl', negatives)
    with pytest.raises(ValueError, match='source group appears across partitions'):
        _build_profile_dataset(tmp_path, config, {}, load_profiles(config['profiles'])['cora-lcc'], records)


def test_official_metrics_exclude_repeated_controls():
    rows = [dict(label=0, prediction=1, independent_clean=False, source_group='benchmark:a'),
            dict(label=0, prediction=0, independent_clean=True, source_group='external-1'),
            dict(label=0, prediction=0, independent_clean=True, source_group='external-2'),
            dict(label=1, prediction=1)]
    formal = _official_for_rows(rows, 2)
    assert formal['clean_checks'] == 2
    assert formal['official_graph_accuracy'] == 1


def test_end_to_end_reuse_train_and_preserve_profiles(tmp_path):
    config = load_config(ROOT / 'configs/base.yaml')
    config['profiles']['cora-lcc']['expected_num_nodes'] = 4
    config['profiles']['citeseer-lcc']['expected_num_nodes'] = 4
    config['runtime']['device'] = 'cpu'
    config['detector']['model'].update(
        epochs=3,
        patience=2,
        dropout=0,
        target_group_recall=0,
        minimum_clean_specificity=0,
        target_consecutive_epochs=2,
        minimum_validation_group_size=1,
    )
    config['attacks']['structural_rates'] = [0.05]
    config['attacks']['targeted_target_counts'] = [5]
    config['corpus']['sample_policy'] = {
        'default': {'train': 1, 'val': 1, 'test': 1},
        'overrides': {},
    }
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for profile_id in ('cora-lcc', 'citeseer-lcc'):
            create_corpus(tmp_path, config, profile_id)
        run = tmp_path / 'run.yaml'
        run.write_text('run:\n  profiles: [cora-lcc, citeseer-lcc]\n  attacks: [random_add, fga]\n  partitions: [train, val, test]\n')
        with patch('gnn_attack_detector.pipeline.generate_graph_corpus', side_effect=AssertionError('must not generate attacks')):
            first = rebuild_detector(config, run, project_root=tmp_path)
        bundle = load_graph_detector_bundle(tmp_path / first['training']['model_path'])
        assert first['attack_availability'] == {
            'cora-lcc': {
                'available_attacks': ['random_add'],
                'unavailable_attacks': ['fga'],
                'unavailable_reasons': {'fga': 'no rows in selected partitions'},
            },
            'citeseer-lcc': {
                'available_attacks': ['random_add'],
                'unavailable_attacks': ['fga'],
                'unavailable_reasons': {'fga': 'no rows in selected partitions'},
            },
        }
        assert bundle.trained_profiles == ('citeseer-lcc', 'cora-lcc')
        assert bundle.model.head[0].in_features == 100
        assert bundle.attribute_modes == {'cora-lcc': 'auto', 'citeseer-lcc': 'auto'}
        assert first['evaluation']['official_complete_graph_metrics']['official_graph_metric_eligible'] is False
        assert (tmp_path / config['detector']['model_path']).exists()
        assert first['training']['activation_required'] is False
        assert first['training']['active_model_updated'] is True
        active_hash = hash_file(tmp_path / config['detector']['model_path'])
        candidate_score = predict_graph_probabilities(bundle, [prepare_graph_input(graph())], device='cpu')[0]
        g = graph()
        g.y = None
        upload = tmp_path / 'independent.npz'
        upload.write_bytes(graph_to_npz_bytes(g))
        # No source graph can be loaded during standalone detection.
        import gnn_attack_detector.pipeline as pipeline
        original_loader = pipeline.torch.load
        def checked_load(path, *args, **kwargs):
            assert Path(path).name != 'clean.pt'
            return original_loader(path, *args, **kwargs)
        with patch.object(pipeline.torch, 'load', side_effect=checked_load):
            result = detect_graph(config, upload, project_root=tmp_path)
        assert result['profile_id'] is None
        assert abs(result['graph_attack_probability'] - float(candidate_score)) < 1e-6
        config['detector']['attribute_modes']['cora-lcc'] = 'none'
        same = detect_graph(config, upload, project_root=tmp_path, profile_id='cora-lcc')
        assert same['attribute_mode'] == 'auto'  # Saved policy, not modified config.
        override = detect_graph(config, upload, project_root=tmp_path, profile_id='cora-lcc', attribute_mode='none')
        assert override['attribute_mode'] == 'none'
        del config['detector']['attribute_modes']['cora-lcc']
        run.write_text('run:\n  profiles: [cora-lcc]\n  attacks: [random_add]\n  partitions: [train, val, test]\n')
        second = rebuild_detector(config, run, project_root=tmp_path)
        assert second['training']['profiles'] == ['cora-lcc']
        assert hash_file(tmp_path / config['detector']['model_path']) != active_hash
        saved = load_platform_evaluation(tmp_path / config['detector']['report_path'], profile_id='citeseer-lcc', model_path=tmp_path / config['detector']['model_path'])
        assert saved['report']['detector_model_hash'] == first['evaluation']['detector_model_hash']
        assert saved['historical'] is True
        assert len(list((tmp_path / 'artifacts/reports/shared-gcn-history').glob('*/evaluation.json'))) >= 2
    finally:
        torch.set_num_threads(old_threads)
