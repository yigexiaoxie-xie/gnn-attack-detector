from pathlib import Path
import tempfile
import unittest

from gnn_attack_detector.config import load_config
from gnn_attack_detector.workflow import WorkflowOperations, run_selected_workflow


ROOT = Path(__file__).resolve().parents[1]


class SelectedWorkflowTests(unittest.TestCase):
    def test_evaluation_separates_selected_progress_from_formal_catalog_gate(self) -> None:
        source = (ROOT / "src/gnn_attack_detector/pipeline.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"selected_attacks"', source)
        self.assertIn('"catalog_attacks"', source)
        self.assertIn("minimum_attack_count", source)
        self.assertIn("eligible_attacks", source)

    def test_two_profiles_run_in_order_with_one_universal_training(self) -> None:
        events: list[str] = []
        seen_attacks: list[tuple[str, ...]] = []
        universal_call_kwargs: list[dict[str, object]] = []

        def profile_event(prefix, payload=None):
            def operation(config, **kwargs):
                profile = str(config["active_profile"])
                events.append(f"{prefix}:{profile}")
                seen_attacks.append(tuple(config["attacks"]["protocols"]))
                return dict(payload or {"status": "ok"})

            return operation

        def train(config, **kwargs):
            events.append("train:universal")
            seen_attacks.append(tuple(config["attacks"]["protocols"]))
            universal_call_kwargs.append(dict(kwargs))
            return {"status": "ok", "training_rows": 10}

        def evaluate(config, **kwargs):
            events.append("evaluate:universal")
            universal_call_kwargs.append(dict(kwargs))
            return {"status": "ok", "acceptance": {"passed": False}}

        operations = WorkflowOperations(
            prepare_profile=profile_event("prepare"),
            prepare_trainval=profile_event(
                "trainval",
                {"status": "ok", "generated_by_partition": {}, "reused_by_partition": {}},
            ),
            train_universal=train,
            prepare_test=profile_event(
                "test",
                {"status": "ok", "generated_test_graphs": 0, "reused_test_graphs": 1},
            ),
            evaluate_universal=evaluate,
        )
        text = """run:
  profiles: [cora-lcc, citeseer-lcc]
  attacks: [random_add, dice_mix]
  partitions: [train, val, test]
"""
        with tempfile.TemporaryDirectory() as folder:
            run_path = Path(folder) / "run.yaml"
            run_path.write_text(text, encoding="utf-8")
            result = run_selected_workflow(
                load_config(ROOT / "configs" / "base.yaml"),
                run_path,
                project_root=ROOT,
                operations=operations,
            )

        self.assertEqual(
            events,
            [
                "prepare:cora-lcc",
                "prepare:citeseer-lcc",
                "trainval:cora-lcc",
                "trainval:citeseer-lcc",
                "train:universal",
                "test:cora-lcc",
                "test:citeseer-lcc",
                "evaluate:universal",
            ],
        )
        self.assertTrue(all(value == ("random_add", "dice_mix") for value in seen_attacks))
        self.assertEqual(len(universal_call_kwargs), 2)
        self.assertTrue(
            all(kwargs["profile_ids"] == ("cora-lcc", "citeseer-lcc") for kwargs in universal_call_kwargs)
        )
        self.assertTrue(
            all(
                kwargs["project_root"] == ROOT.resolve()
                for kwargs in universal_call_kwargs
            )
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["selected_profiles"], ["cora-lcc", "citeseer-lcc"])

    def test_test_failure_does_not_evaluate_stale_profiles(self) -> None:
        events: list[str] = []

        def ok(config, **kwargs):
            events.append(f"prepare:{config['active_profile']}")
            return {"status": "ok"}

        def fail_test(config, **kwargs):
            events.append(f"test:{config['active_profile']}")
            raise RuntimeError("selected test corpus is incomplete")

        def forbidden(config, **kwargs):
            events.append("evaluation:forbidden")
            return {"status": "ok"}

        operations = WorkflowOperations(ok, ok, ok, fail_test, forbidden)
        text = """run:
  profiles: [cora-lcc]
  attacks: [nettack_feature]
  partitions: [test]
"""
        with tempfile.TemporaryDirectory() as folder:
            run_path = Path(folder) / "run.yaml"
            run_path.write_text(text, encoding="utf-8")
            result = run_selected_workflow(
                load_config(ROOT / "configs" / "base.yaml"),
                run_path,
                project_root=ROOT,
                operations=operations,
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["errors"][0]["phase"], "test")
        self.assertNotIn("evaluation:forbidden", events)
        self.assertNotIn("evaluation", result["results"])

    def test_trainval_failure_prevents_detector_freeze(self) -> None:
        events: list[str] = []

        def prepare(config, **kwargs):
            events.append(f"prepare:{config['active_profile']}")
            return {"status": "ok"}

        def fail(config, **kwargs):
            events.append(f"trainval:{config['active_profile']}")
            raise RuntimeError("broken corpus")

        def forbidden(config, **kwargs):
            events.append("forbidden")
            return {"status": "ok"}

        operations = WorkflowOperations(prepare, fail, forbidden, forbidden, forbidden)
        text = """run:
  profiles: [cora-lcc]
  attacks: [random_add]
  partitions: [train, val, test]
"""
        with tempfile.TemporaryDirectory() as folder:
            run_path = Path(folder) / "run.yaml"
            run_path.write_text(text, encoding="utf-8")
            result = run_selected_workflow(
                load_config(ROOT / "configs" / "base.yaml"),
                run_path,
                project_root=ROOT,
                operations=operations,
            )

        self.assertEqual(result["status"], "failed")
        self.assertNotIn("forbidden", events)
        self.assertEqual(result["errors"][0]["phase"], "train_validation")


if __name__ == "__main__":
    unittest.main()
