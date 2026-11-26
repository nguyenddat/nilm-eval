import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from training.data import REFITCSVDataset
from training.model.TwoStageNILMFormer.TwoStageNILMFormer import TwoStageOutput
from training.tests.test_twostage_nilmformer import make_model
from training.train_twostage_nilmformer import TwoStageLossConfig, detector_class_weights, temporal_validation_indices, train_twostage_nilmformer, two_stage_loss


class TwoStageStudyTests(unittest.TestCase):
    def test_per_device_bce_weights_keep_channel_alignment_after_mask(self):
        logits = torch.zeros(1, 2, 2, requires_grad=True)
        power = torch.ones_like(logits)
        output = TwoStageOutput(logits, logits.sigmoid(), power, power, power)
        on = torch.tensor([[[1.0, 0.0], [1.0, 1.0]]])
        valid = torch.tensor([[[True, True], [True, False]]])
        loss = two_stage_loss(output, power, on, valid, TwoStageLossConfig(regression_weight=0, auxiliary_on_weight=0, positive_weight=[2.0, 5.0]))
        self.assertAlmostEqual(float(loss["loss"].detach()), 8 * math.log(2) / 3, places=6)
        loss["loss"].backward()
        torch.testing.assert_close(logits.grad, torch.tensor([[[-1 / 3, 1 / 6], [-5 / 6, 0.0]]]))

    def test_temporal_validation_and_class_counts_exclude_future_labels(self):
        dataset = object.__new__(REFITCSVDataset)
        dataset.targets = ["WashingMachine", "Dishwasher", "Kettle", "Microwave"]
        dataset.length = 2
        dataset.shards = [{"house": 1, "windows": 10}]
        dataset.manifest = {"config": {"sampling_seconds": 60}, "time_splits": {"1": {"start": "2023-01-01", "train_end_exclusive": "2023-01-01 00:18"}}}
        starts = pd.date_range("2023-01-01", periods=10, freq="2min").to_numpy(dtype="datetime64[ns]").astype("int64")
        on = np.zeros((10, 4, 2), dtype="float32")
        on[0, 0, 0] = 1
        on[8:, 0] = 1
        on[:7, 3] = 1
        valid = np.ones_like(on, dtype=bool)
        valid[:, 2] = False
        dataset.cached = {0: {"start_time_ns": starts, "on": on, "valid": valid}}
        train, validation, rows = temporal_validation_indices(dataset)
        self.assertEqual(train[1], list(range(7)))
        self.assertEqual(validation[1], [8, 9])
        self.assertEqual(rows[0]["boundary_windows_dropped"], 1)
        weights, balance = detector_class_weights(dataset, train[1])
        self.assertEqual(balance[0]["on_count"], 1)
        self.assertEqual(balance[0]["off_count"], 13)
        self.assertEqual(weights[0], 7)
        self.assertFalse(balance[1]["has_positive_train_labels"])
        self.assertFalse(balance[2]["has_train_labels"])
        self.assertAlmostEqual(weights[3], 1 / 15)

    def test_early_stopping_restores_best_and_preserves_last_checkpoint(self):
        model = make_model()
        epoch = 0
        scores = [0.5, 0.4, 0.41, 0.42, 0.43]

        def fake_epoch(model, batches, device, config, optimizer=None, grad_clip=1, metrics_config=None, progress_callback=None):
            nonlocal epoch
            if optimizer is not None:
                epoch += 1
                with torch.no_grad():
                    model.Regressor.ProjStats2.bias.fill_(epoch)
            value = 1.0 if optimizer is not None else scores[epoch - 1]
            return {"loss": value, "regression": value, "detector": 0.0, "auxiliary_on": 0.0}

        with tempfile.TemporaryDirectory() as tmp, patch("training.train_twostage_nilmformer.run_two_stage_epoch", side_effect=fake_epoch):
            path = Path(tmp)
            history = train_twostage_nilmformer(model, [{}], [{}], epochs=10, patience=2, checkpoint_path=path / "best.pt", last_checkpoint_path=path / "last.pt")
            best = torch.load(path / "best.pt", weights_only=True)
            last = torch.load(path / "last.pt", weights_only=True)
            self.assertEqual(len(history), 4)
            self.assertTrue(history[-1]["stopped_early"])
            self.assertEqual(best["best_epoch"], 2)
            self.assertEqual(last["epoch"], 4)
            torch.testing.assert_close(model.Regressor.ProjStats2.bias, torch.full((2,), 2.0))
            torch.testing.assert_close(last["model"]["Regressor.ProjStats2.bias"], torch.full((2,), 4.0))


if __name__ == "__main__":
    unittest.main()
