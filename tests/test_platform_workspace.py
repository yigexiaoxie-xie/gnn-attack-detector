"""Regression checks for dataset-scoped reports and the platform input paths."""

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch_geometric.data import Data

from gnn_attack_detector.config import load_config
from gnn_attack_detector.detector_metrics import binary_metrics, profile_evaluation_summaries
from gnn_attack_detector.graph_attacks import CompleteGraphAttackOutcome
from gnn_attack_detector.integrity import graph_fingerprint, hash_file
from gnn_attack_detector.platform_generation import generate_attack_batch
from gnn_attack_detector.platform_service import (
    collect_platform_status, generate_and_detect_platform_graphs,
    load_platform_accuracy_summary, load_platform_evaluation,
)


ROOT = Path(__file__).resolve().parents[1]


def prediction_rows():
    rows = []
    for profile, count, correct in [("cora-lcc", 20, 15), ("citeseer-lcc", 8, 2)]:
        for index in range(count):
            rows.append({
                "profile_id": profile, "sample_id": str(index), "partition": "test",
                "label": 1, "prediction": int(index < correct),
                "attack_name": "random_add", "strength_value": 0.05, "strength_unit": "edge_rate",
            })
        rows.append({"profile_id": profile, "sample_id": "clean", "partition": "test", "label": 0, "prediction": 0})
    return rows


def report_fixture():
    rows = prediction_rows()
    summaries = profile_evaluation_summaries(rows)
    return {
        "format_version": 3,
        "test_metrics": binary_metrics([item["label"] for item in rows], [item["prediction"] for item in rows]),
        "metrics_by_profile": {key: item["test_metrics"] for key, item in summaries.items()},
        "profile_evaluations": summaries,
        "attacked_complete_graph_recall": {},
        "clean_complete_graph_check": {},
        "official_complete_graph_metrics": {},
    }


def status_fixture():
    defense_names = (
        "gcn", "gcn_jaccard", "gcn_svd", "rgcn", "median_gcn",
        "simpgcn", "prognn", "gnnguard", "airgnn", "sgc",
    )
    modes = {name: "input_fit" for name in defense_names}
    modes["prognn"] = "offline"
    return {
        "profiles": [
            {
                "id": name + "-lcc", "name": name.title(), "ready": True,
                "expected_num_nodes": 4,
                "defenses": {
                    defense: {"ready": True, "execution_mode": modes[defense]}
                    for defense in defense_names
                },
            }
            for name in ("cora", "citeseer", "polblogs", "pubmed")
        ],
        "ready_for_detection": True, "supported_attack_count": 21, "attack_graph_count": 48,
        "evaluation_report": "report.json", "prediction_report": "predictions.jsonl",
        "defenses": [
            {"id": name, "display_name": name.upper(), "execution_mode": modes[name]}
            for name in defense_names
        ],
    }


class PlatformWorkspaceTests(unittest.TestCase):
    def test_platform_status_exposes_all_defense_modes_per_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            shutil.copyfile(ROOT / "configs/base.yaml", root / "configs/base.yaml")
            config = load_config(root / "configs/base.yaml")
            cora = config["profiles"]["cora-lcc"]
            clean = root / cora["clean_reference"]
            probes = root / cora["probe_dir"]
            clean.parent.mkdir(parents=True)
            probes.mkdir(parents=True)
            clean.write_bytes(b"clean")
            (probes / "gcn.pt").write_bytes(b"gcn")
            status = collect_platform_status(root)

        self.assertEqual(len(status["defenses"]), 10)
        self.assertEqual(
            {item["id"]: item["execution_mode"] for item in status["defenses"]}["gcn_svd"],
            "input_fit",
        )
        cora_status = next(item for item in status["profiles"] if item["id"] == "cora-lcc")
        self.assertTrue(cora_status["defenses"]["gcn"]["ready"])
        self.assertTrue(cora_status["defenses"]["rgcn"]["ready"])
        self.assertTrue(cora_status["defenses"]["sgc"]["ready"])
        self.assertFalse(next(item for item in status["defenses"] if item["id"] == "gcn")["requires_preparation"])

    def test_platform_marks_report_current_when_model_hash_matches(self):
        report = report_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pt"
            model.write_bytes(b"detector-a")
            report["detector_model_hash"] = hash_file(model)
            report["evaluated_at_utc"] = "2026-09-07T12:00:00+00:00"
            path = root / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            loaded = load_platform_evaluation(path, model_path=model)

        self.assertFalse(loaded["historical"])

    def test_platform_marks_report_historical_when_model_hash_changes(self):
        report = report_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pt"
            model.write_bytes(b"detector-a")
            report["detector_model_hash"] = hash_file(model)
            report["evaluated_at_utc"] = "2026-09-07T12:00:00+00:00"
            path = root / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            model.write_bytes(b"detector-b")

            loaded = load_platform_evaluation(path, model_path=model)

        self.assertTrue(loaded["historical"])

    def test_platform_marks_legacy_report_without_hash_historical(self):
        report = report_fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.pt"
            model.write_bytes(b"current-detector")
            path = root / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            loaded = load_platform_evaluation(path, model_path=model)

        self.assertTrue(loaded["historical"])

    def test_report_does_not_pool_datasets_or_validation_rows(self):
        rows = prediction_rows()
        rows.append(dict(rows[0], partition="val", prediction=1))
        summaries = profile_evaluation_summaries(rows)
        cora = summaries["cora-lcc"]
        citeseer = summaries["citeseer-lcc"]
        self.assertEqual(cora["attacked_complete_graph_recall"]["by_attack"]["random_add"], {"total": 20, "detected": 15, "recall": .75})
        self.assertEqual(citeseer["attacked_complete_graph_recall"]["by_attack"]["random_add"]["total"], 8)
        self.assertAlmostEqual(cora["test_metrics"]["accuracy"], 16 / 21)
        self.assertAlmostEqual(citeseer["test_metrics"]["accuracy"], 3 / 9)

    def test_platform_accuracy_summary_averages_available_profiles_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "report.json"
            report = report_fixture()
            report_path.write_text(json.dumps(report), encoding="utf-8")
            summary = load_platform_accuracy_summary(
                report_path,
                profile_ids=("cora-lcc", "citeseer-lcc", "pubmed-lcc", "polblogs-lcc"),
            )
        self.assertEqual(summary["evaluated_profiles"], ["cora-lcc", "citeseer-lcc"])
        self.assertEqual(summary["missing_profiles"], ["pubmed-lcc", "polblogs-lcc"])
        self.assertAlmostEqual(summary["macro_accuracy"], ((16 / 21) + (3 / 9)) / 2)
        self.assertAlmostEqual(summary["weighted_accuracy"], 19 / 30)

    def test_old_report_uses_matching_predictions_without_changing_saved_files(self):
        report = report_fixture()
        del report["profile_evaluations"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            predictions = Path(directory) / "predictions.jsonl"
            original = json.dumps(report)
            path.write_text(original, encoding="utf-8")
            predictions.write_text("\n".join(map(json.dumps, prediction_rows())), encoding="utf-8")
            loaded = load_platform_evaluation(path, prediction_path=predictions)
            self.assertEqual(loaded["report"]["profile_evaluations"]["cora-lcc"]["attacked_complete_graph_recall"]["overall"]["total"], 20)
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            rows = prediction_rows()
            rows[0]["prediction"] = 0
            predictions.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            loaded = load_platform_evaluation(path, prediction_path=predictions)
            self.assertNotIn("profile_evaluations", loaded["report"])

    def test_zero_strength_uses_selected_clean_graph_and_real_prediction(self):
        generator = Mock(side_effect=AssertionError("clean detection must not generate attacks"))
        loader = Mock(side_effect=AssertionError("clean detection must not load attack context"))
        config = load_config(ROOT / "configs/base.yaml")
        with tempfile.TemporaryDirectory() as directory:
            for profile_id, profile in config["profiles"].items():
                detector = Mock(return_value={"status": "ok", "prediction": "attacked", "profile_id": profile_id})
                rows = generate_and_detect_platform_graphs(
                    ROOT / "configs/base.yaml", project_root=directory,
                    dataset_name=profile_id, attack_name="nettack_feature", strength=0, count=1,
                    generator=generator, context_loader=loader, detector=detector,
                )
                self.assertEqual(detector.call_args.args[1], Path(directory).resolve() / profile["clean_reference"])
                self.assertEqual(rows[0]["prediction_label"], "检测到攻击")
                self.assertFalse(rows[0]["correctly_detected"])
                self.assertEqual(rows[0]["actual_modifications"], 0)
            self.assertFalse((Path(directory) / "artifacts").exists())

    def test_live_generation_retries_corpus_duplicate_and_saves_new_graph(self):
        clean = Data(x=torch.ones(4, 2), y=torch.tensor([0, 1, 0, 1]), edge_index=torch.tensor([[0, 1], [1, 0]]))

        def outcome(pair):
            attacked = clean.clone()
            attacked.edge_index = torch.cat((clean.edge_index, torch.tensor([pair, pair[::-1]]).t()), dim=1)
            return CompleteGraphAttackOutcome(attacked, "random_add", (), (pair,), (), (), 1, 1)

        duplicate, fresh = outcome((0, 2)), outcome((0, 3))
        runner = Mock(side_effect=[duplicate, fresh])
        seeds = iter([101, 102])
        with tempfile.TemporaryDirectory() as directory:
            rows = generate_attack_batch(
                clean_graph=clean, surrogate=None, dataset_name="cora-lcc", attack_name="random_add",
                strength=.05, count=1, output_root=directory, structural_rates=(.05,),
                targeted_target_counts=(5,), targeted_local_budget=3, minimum_target_degree=5,
                max_attempts=2, seed_factory=lambda: next(seeds), attack_runner=runner,
                excluded_fingerprints={graph_fingerprint(duplicate.graph)["combined"]},
            )
            self.assertEqual(runner.call_count, 2)
            self.assertEqual(rows[0]["status"], "ok")
            self.assertEqual(rows[0]["graph_fingerprint"], graph_fingerprint(fresh.graph)["combined"])
            self.assertTrue(Path(rows[0]["pt_path"]).is_file())
            self.assertTrue(Path(rows[0]["npz_path"]).is_file())

    def test_live_service_excludes_all_partitions_of_selected_profile(self):
        config = load_config(ROOT / "configs/base.yaml")
        from gnn_attack_detector.profiles import load_profiles

        profile = load_profiles(config["profiles"])["polblogs-lcc"]
        generator = Mock(return_value=[])
        loader = Mock(return_value={"config": config, "profile": profile, "clean_graph": None, "surrogate": None})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            manifest = root / config["corpus"]["sample_index_dir"] / "polblogs-lcc.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text("\n".join(json.dumps({"partition": part, "graph_hash": part}) for part in ("train", "val", "test")), encoding="utf-8")
            before = manifest.read_bytes()
            generate_and_detect_platform_graphs(
                ROOT / "configs/base.yaml", project_root=root, dataset_name=profile.identifier,
                attack_name="random_add", strength=.05, count=1, generator=generator, context_loader=loader,
            )
            self.assertEqual(generator.call_args.kwargs["excluded_fingerprints"], {"train", "val", "test"})
            self.assertEqual(generator.call_args.kwargs["output_root"], root / "artifacts/platform/generated")
            self.assertEqual(manifest.read_bytes(), before)


class WorkspaceUiTests(unittest.TestCase):
    def setUp(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Streamlit is not installed")
        self.AppTest = AppTest
        self.status = status_fixture()

    def app(self):
        return self.AppTest.from_string("from gnn_attack_detector.web_app import render_app\nrender_app('.')", default_timeout=20)

    def test_defense_page_lists_all_configured_models(self):
        with patch(
            "gnn_attack_detector.web_app.collect_platform_status",
            return_value=self.status,
        ):
            app = self.app().run()
            self.assertIn("鲁棒防御模型", app.button(key="open-03").label)
            app.button(key="open-03").click().run()

        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key="platform-page").value, "鲁棒防御模型")
        self.assertEqual(len(app.selectbox(key="defense-model").options), 10)

    def test_upload_selects_dataset_and_has_no_attribute_type_control(self):
        import inspect
        from gnn_attack_detector.web_app import _upload_tab

        source = inspect.getsource(_upload_tab)
        self.assertIn('_select_dataset', source)
        self.assertIn('profile_id=profile_id', source)
        self.assertNotIn('节点属性类型', source)
        self.assertNotIn('attribute_mode=', source)

    def test_frontend_uses_graph_security_console_visual_system(self):
        import inspect
        from gnn_attack_detector import web_app

        source = inspect.getsource(web_app._style) + inspect.getsource(web_app._overview)
        self.assertIn('--gs-ink', source)
        self.assertIn('security-console', source)
        self.assertIn('status-rail', source)
        self.assertIn('数据集就绪度', source)

    def test_frontend_rebuilds_right_side_modules_and_accuracy_overview(self):
        import inspect
        from gnn_attack_detector import web_app

        source = inspect.getsource(web_app._style)
        source += inspect.getsource(web_app._show_detection_results)
        source += inspect.getsource(web_app._evaluation)
        source += inspect.getsource(web_app._defenses)
        self.assertIn('module-header', source)
        self.assertIn('result-strip', source)
        self.assertIn('accuracy-overview-chart', source)
        self.assertIn('ATTACK BREAKDOWN', source)
        self.assertIn('prefers-reduced-motion', source)
        self.assertNotIn('transform: translateY', source)

    def test_frontend_uses_pale_blue_apple_like_visual_layers(self):
        import inspect
        from gnn_attack_detector import web_app

        source = inspect.getsource(web_app._style) + inspect.getsource(web_app.render_app)
        self.assertIn('--gs-sky', source)
        self.assertIn('sidebar-brand-panel', source)
        self.assertIn('sidebar-metric', source)
        self.assertIn('background: linear-gradient', source)

    def test_frontend_uses_cosmic_indigo_data_panels(self):
        import inspect
        from gnn_attack_detector import web_app

        source = inspect.getsource(web_app._style) + inspect.getsource(web_app._overview)
        self.assertIn('--gs-plasma', source)
        self.assertIn('--gs-surface', source)
        self.assertIn('metric-data-plate', source)
        self.assertIn('panel-corner', source)

    def test_sidebar_uses_a_layered_blue_cyan_violet_console_palette(self):
        import inspect
        from gnn_attack_detector import web_app

        source = inspect.getsource(web_app._style)
        self.assertIn('--gs-violet', source)
        self.assertIn('--gs-sidebar', source)
        self.assertIn('sidebar-status-chip', source)
        self.assertIn('linear-gradient(180deg', source)
        self.assertNotIn('[data-testid="stSidebar"] {background: var(--gs-ink);', source)

    def test_evaluation_shows_model_identity_and_historical_warning(self):
        report = report_fixture()
        report["detector_model_hash"] = "0123456789abcdef"
        report["evaluated_at_utc"] = "2026-09-07T12:00:00+00:00"
        payload = {"available": True, "report": report, "historical": True}
        with (
            patch(
                "gnn_attack_detector.web_app.collect_platform_status",
                return_value=self.status,
            ),
            patch(
                "gnn_attack_detector.web_app.load_platform_evaluation",
                return_value=payload,
            ),
        ):
            app = self.app().run()
            app.radio(key="platform-page").set_value("准确率评测").run()

        self.assertTrue(any("历史模型" in item.value for item in app.warning))
        self.assertTrue(any("0123456789ab" in item.value for item in app.caption))
        self.assertTrue(
            any(
                "2026-09-07T12:00:00+00:00" in item.value
                for item in app.caption
            )
        )

    def test_dataset_switch_updates_metrics_table_and_empty_state(self):
        with patch("gnn_attack_detector.web_app.collect_platform_status", return_value=self.status), patch("gnn_attack_detector.web_app.load_platform_evaluation", return_value={"available": True, "report": report_fixture()}):
            app = self.app().run()
            app.radio(key="platform-page").set_value("准确率评测").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="evaluation-dataset").options, ["Cora", "Citeseer", "Polblogs", "Pubmed"])
            self.assertEqual(app.metric[0].value, "76.19%")
            self.assertEqual(app.dataframe[0].value.iloc[0]["测试图数量"], 20)
            self.assertNotIn("是否达标", app.dataframe[0].value.columns)
            app.selectbox(key="evaluation-dataset").set_value("citeseer-lcc").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.metric[0].value, "33.33%")
            self.assertEqual(app.dataframe[0].value.iloc[0]["测试图数量"], 8)
            app.selectbox(key="evaluation-dataset").set_value("pubmed-lcc").run()
            self.assertEqual(len(app.metric), 0)
            self.assertTrue(any("Pubmed 尚无测试结果" in item.value for item in app.info))

    def test_overview_navigation_zero_strength_and_result_persistence(self):
        result = [{"status": "ok", "prediction": "clean", "prediction_label": "未检测到攻击", "profile_id": "cora-lcc", "source_name": "Cora · 干净图", "graph_attack_probability": .2}]
        with patch("gnn_attack_detector.web_app.collect_platform_status", return_value=self.status), patch("gnn_attack_detector.web_app.generate_and_detect_platform_graphs", return_value=result) as generate:
            app = self.app().run()
            self.assertEqual(list(app.dataframe[0].value.columns), ["数据集", "节点数", "状态"])
            app.button(key="open-01").click().run()
            self.assertFalse(app.exception)
            app.selectbox(key="generate-strength-edges").set_value(0).run()
            self.assertTrue(app.number_input(key="generate-count-clean").disabled)
            app.button(key="generate-submit").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(generate.call_args.kwargs["strength"], 0)
            self.assertEqual(generate.call_args.kwargs["count"], 1)
            self.assertEqual(app.dataframe[0].value.iloc[0]["数据集"], "Cora")
            app.run()
            self.assertEqual(len(app.dataframe), 1)
            self.assertEqual(generate.call_count, 1)
            app.selectbox(key="generate-dataset").set_value("polblogs-lcc").run()
            self.assertEqual(len(app.dataframe), 0)


if __name__ == "__main__":
    unittest.main()
