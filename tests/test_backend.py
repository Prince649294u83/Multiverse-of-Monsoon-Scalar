"""Backend-only regression tests; no raw data, dashboard, or model checkpoint needed."""
from __future__ import annotations

import unittest

import torch

from model.features import advance_doy_sincos, internal_to_phys_torch, phys_to_internal_torch


class FeatureTests(unittest.TestCase):
    def test_precipitation_transform_round_trip(self) -> None:
        x = torch.zeros((2, 6, 32, 36), dtype=torch.float32)
        x[:, 1] = torch.linspace(0.0, 500.0, 32 * 36).reshape(32, 36)
        restored = internal_to_phys_torch(phys_to_internal_torch(x))
        self.assertTrue(torch.allclose(restored[:, 1], x[:, 1], atol=1e-4))

    def test_annual_phase_wraps(self) -> None:
        start = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        wrapped = advance_doy_sincos(start, 365)
        self.assertTrue(torch.allclose(wrapped, start, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
