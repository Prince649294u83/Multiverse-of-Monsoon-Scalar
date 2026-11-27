"""Regression tests for phase wraparound, physical clamping, causality gates, and model shapes.

Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import math
import unittest

import torch

from monsoon_r1 import contract as C
from model.architecture import ClimateDigitalTwin
from model.features import (
    advance_doy_sincos,
    clamp_physical_torch,
    doy_sincos_to_float,
    internal_to_phys_torch,
    lookup_climatology_torch,
    phys_to_internal_torch,
)


class TestFeaturesAndPhase(unittest.TestCase):
    def test_tp_invertibility(self) -> None:
        x = torch.zeros((2, 6, 32, 36), dtype=torch.float32)
        x[:, 0] = 295.0
        x[:, 1] = torch.tensor([[[0.0]], [[10.0]]])
        x[0, 1, 5, 5] = 185.5
        z = phys_to_internal_torch(x)
        x_rec = internal_to_phys_torch(z)
        self.assertTrue(torch.allclose(x, x_rec, atol=1e-4))

    def test_doy_phase_wraparound(self) -> None:
        d1 = torch.tensor([C.doy_sincos(1)], dtype=torch.float32)
        d365 = torch.tensor([C.doy_sincos(365)], dtype=torch.float32)
        self.assertAlmostEqual(float(doy_sincos_to_float(d1)[0]), 1.0, places=3)
        f365 = float(doy_sincos_to_float(d365)[0])
        self.assertTrue(f365 < 1e-3 or abs(f365 - 365.0) < 1e-3)

        # Advancing day 365 by 1 day equals day 1
        adv = advance_doy_sincos(d365, 1)
        self.assertAlmostEqual(float(doy_sincos_to_float(adv)[0]), 1.0, places=3)

        # Circular interpolation seamlessly wraps index 364 -> 0
        clim = torch.arange(365, dtype=torch.float32).view(365, 1, 1, 1).expand(365, 6, 32, 36)
        val = lookup_climatology_torch(clim, torch.tensor([364.5]))
        self.assertAlmostEqual(float(val[0, 0, 0, 0]), 182.0, places=3)

    def test_physical_clamping(self) -> None:
        x = torch.zeros((1, 6, 32, 36), dtype=torch.float32)
        x[0, 0, 0, 0] = 100.0
        x[0, 0, 0, 1] = 400.0
        x[0, 1, 0, 0] = -12.0
        x[0, 2, 0, 0] = float("nan")
        clamped = clamp_physical_torch(x)
        self.assertTrue(torch.isfinite(clamped).all())
        self.assertGreaterEqual(float(clamped[0, 0].min()), 233.15 - 1e-4)
        self.assertLessEqual(float(clamped[0, 0].max()), 333.15 + 1e-4)
        self.assertGreaterEqual(float(clamped[0, 1].min()), 0.0)


class TestCausalityAndModelContract(unittest.TestCase):
    def test_climate_digital_twin_steps_and_determinism(self) -> None:
        twin = ClimateDigitalTwin(
            arch="unet",
            internal_mean=torch.zeros(6),
            internal_std=torch.ones(6),
            anom_std=torch.ones(6),
            clim_internal=torch.zeros((365, 6, 32, 36)),
            static_coords=torch.zeros((4, 32, 36)),
        )
        twin.eval()
        init_s = torch.zeros((1, 3, 6, 32, 36), dtype=torch.float32)
        init_s[:, :, 0] = 295.0
        init_s[:, :, 4] = 101000.0
        doy = torch.tensor([C.doy_sincos(180)], dtype=torch.float32)

        with torch.inference_mode():
            out1 = twin(init_s, doy, steps=1, perturbation=None)
            out14 = twin(init_s, doy, steps=14, perturbation=None)
            out14_null = twin(init_s, doy, steps=14, perturbation=torch.zeros((1, 6, 32, 36)))
            out30 = twin(init_s, doy, steps=30, perturbation=None)

        self.assertEqual(tuple(out1.shape), (1, 1, 6, 32, 36))
        self.assertEqual(tuple(out14.shape), (1, 14, 6, 32, 36))
        self.assertEqual(tuple(out30.shape), (1, 30, 6, 32, 36))
        self.assertTrue(torch.isfinite(out30).all())
        self.assertTrue(torch.allclose(out14, out14_null, atol=1e-6))

    def test_perturbation_is_coupled_to_initial_condition(self) -> None:
        twin = ClimateDigitalTwin(
            arch="resnet", internal_mean=torch.zeros(6), internal_std=torch.ones(6),
            anom_std=torch.ones(6), clim_internal=torch.zeros((365, 6, 32, 36)),
            static_coords=torch.zeros((4, 32, 36)),
        ).eval()
        init_s = torch.zeros((1, 3, 6, 32, 36), dtype=torch.float32)
        init_s[:, :, 0] = 295.0; init_s[:, :, 4] = 101000.0
        pert = torch.zeros((1, 6, 32, 36)); pert[:, 0, 10:12, 10:12] = 2.0
        direct = init_s.clone(); direct[:, -1] += pert
        doy = torch.tensor([C.doy_sincos(180)], dtype=torch.float32)
        with torch.inference_mode():
            via_delta = twin(init_s, doy, steps=4, perturbation=pert)
            via_state = twin(direct, doy, steps=4, perturbation=None)
        self.assertTrue(torch.allclose(via_delta, via_state, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
