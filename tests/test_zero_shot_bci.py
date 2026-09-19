"""
Unit Tests for zero-shot-bci.
Verifies Gromov-Wasserstein manifold alignment, sub-15mW spike sparsification,
multi-modal biophysical liveness mesh, and end-to-end BCI benchmark execution.
"""

from __future__ import annotations
import math
import unittest

from zero_shot_bci.gromov_wasserstein import (
    GromovWassersteinAligner,
)
from zero_shot_bci.spike_sparsification import (
    EventDrivenSpikeSparsifier,
)
from zero_shot_bci.multimodal_liveness import (
    MultiModalLivenessMesh,
)
from zero_shot_bci.evaluator import (
    run_comprehensive_zero_shot_bci_benchmark,
)


class TestZeroShotBCI(unittest.TestCase):
    def test_gromov_wasserstein_transport_properties(self):
        """Verifies Gromov-Wasserstein transport matrix marginal constraints."""
        aligner = GromovWassersteinAligner(epsilon=0.05, max_gw_iter=10)
        # 3x3 distance matrices
        D_X = [
            [0.0, 1.0, 1.5],
            [1.0, 0.0, 0.8],
            [1.5, 0.8, 0.0],
        ]
        D_Y = [
            [0.0, 0.9, 1.4],
            [0.9, 0.0, 0.7],
            [1.4, 0.7, 0.0],
        ]
        T = aligner.solve_transport_matrix(D_X, D_Y)
        n = len(D_X)
        m = len(D_Y)

        # Row marginals sum to 1/n
        for i in range(n):
            row_sum = sum(T[i][j] for j in range(m))
            self.assertAlmostEqual(row_sum, 1.0 / n, places=3)

        # Column marginals sum to 1/m
        for j in range(m):
            col_sum = sum(T[i][j] for i in range(n))
            self.assertAlmostEqual(col_sum, 1.0 / m, places=3)

    def test_zero_shot_atlas_projection(self):
        """Verifies feature projection from patient space to canonical atlas coordinates."""
        aligner = GromovWassersteinAligner(epsilon=0.05, max_gw_iter=5)
        D_X = [[0.0, 1.0], [1.0, 0.0]]
        D_Y = [[0.0, 1.0], [1.0, 0.0]]
        T = aligner.solve_transport_matrix(D_X, D_Y)

        patient_feat = [2.0, 4.0]
        atlas_coords = aligner.project_features_to_atlas(patient_feat, T)

        self.assertEqual(len(atlas_coords), 2)
        # Average of projected features should be preserved
        self.assertAlmostEqual(sum(atlas_coords) / 2.0, sum(patient_feat) / 2.0, places=2)

    def test_spike_sparsification_bandwidth_reduction(self):
        """Verifies event-driven sparsification reduces data volume by >90% within <15mW."""
        sparsifier = EventDrivenSpikeSparsifier(channels=4, sample_rate_hz=30000)

        # 4 channels x 200 samples (mostly baseline noise < 10uV)
        raw_signals = []
        for ch in range(4):
            ch_data = [5.0] * 200
            # Inject 2 discrete spikes on channel 0 and 2
            if ch in (0, 2):
                ch_data[50] = 80.0
                ch_data[150] = 95.0
            raw_signals.append(ch_data)

        events, report = sparsifier.sparsify_window(raw_signals)

        self.assertGreater(report.bandwidth_reduction_pct, 90.0)
        self.assertLess(report.estimated_power_dissipation_mw, 15.0)
        self.assertTrue(report.fda_thermal_compliant)
        self.assertGreaterEqual(len(events), 2)

    def test_refractory_period_suppression(self):
        """Verifies spikes within 1ms refractory period are suppressed."""
        sparsifier = EventDrivenSpikeSparsifier(channels=1, refractory_samples=30)
        # Inject two large spikes spaced only 5 samples apart
        raw_channel = [[0.0] * 100]
        raw_channel[0][10] = 100.0
        raw_channel[0][15] = 100.0  # Within refractory window!
        raw_channel[0][60] = 100.0  # Outside refractory window

        events, _ = sparsifier.sparsify_window(raw_channel)
        # Should detect only sample 10 and sample 60 (sample 15 suppressed)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].sample_index, 10)
        self.assertEqual(events[1].sample_index, 60)

    def test_multimodal_liveness_genuine_vs_deepfake(self):
        """Verifies phase-locking value accepts coupled biological signals and rejects deepfakes."""
        liveness_mesh = MultiModalLivenessMesh(min_coherence_threshold=0.55)
        n = 20

        # 1. Genuine Biological Stream (Coherent phase coupling)
        base_phase = [math.sin(k / 3.0) for k in range(n)]
        erp_live = base_phase
        saccade_live = [p + 0.1 for p in base_phase]
        ptt_live = [p + 0.2 for p in base_phase]

        verdict_live = liveness_mesh.evaluate_tri_modal_stream(erp_live, saccade_live, ptt_live)
        self.assertTrue(verdict_live.is_live_human)
        self.assertGreater(verdict_live.liveness_coherence_score, 0.70)
        self.assertIn("Live Biological", verdict_live.rejection_cause)

        # 2. Synthetic Replay / Deepfake (Decoupled random phases)
        erp_fake = [1.57 * (k % 4) for k in range(n)]
        saccade_fake = [3.14 * ((-1) ** k) for k in range(n)]
        ptt_fake = [-1.0 * (k % 3) for k in range(n)]

        verdict_fake = liveness_mesh.evaluate_tri_modal_stream(erp_fake, saccade_fake, ptt_fake)
        self.assertFalse(verdict_fake.is_live_human)
        self.assertLess(verdict_fake.liveness_coherence_score, 0.50)
        self.assertIn("REJECTED", verdict_fake.rejection_cause)

    def test_end_to_end_zero_shot_bci_benchmark(self):
        """Verifies 300-cycle BCI foundation transport runner."""
        rep = run_comprehensive_zero_shot_bci_benchmark(total_cycles=300)
        self.assertEqual(rep.total_cycles, 300)
        self.assertGreater(rep.mean_bandwidth_reduction_pct, 90.0)
        self.assertLess(rep.mean_power_dissipation_mw, 15.0)
        self.assertEqual(rep.fda_thermal_compliance_pct, 100.0)
        self.assertGreater(rep.deepfake_spoof_detection_pct, 95.0)
        self.assertLess(rep.avg_cycle_latency_ms, 2.0)


if __name__ == "__main__":
    unittest.main()
