"""
Multi-Modal Continuous Bio-Physical Liveness Mesh.

Thwarts synthetic deepfakes, AI-generated EEG replays, and video injection attacks
by evaluating non-linear phase coherence across three coupled physiological channels:
1. Cortical Event-Related Potentials (ERP P300 / SSVEP)
2. Micro-Saccadic Involuntary Eye Jitter (50-100Hz)
3. Cardiovascular Pulse-Transit Time (PTT)
"""

from __future__ import annotations
import dataclasses
import math
import time
from typing import Sequence


@dataclasses.dataclass(frozen=True)
class LivenessVerdict:
    timestamp: float
    is_live_human: bool
    liveness_coherence_score: float  # [0.0, 1.0]
    erp_ocular_coherence: float
    ocular_cardio_coherence: float
    spoof_probability: float
    rejection_cause: str


class MultiModalLivenessMesh:
    """
    Evaluates physiological phase synchronization to detect synthetic biometric spoofing.
    """

    def __init__(self, min_coherence_threshold: float = 0.55):
        self.min_coherence_threshold = min_coherence_threshold

    def _compute_phase_locking_value(
        self,
        phase_series_a: Sequence[float],
        phase_series_b: Sequence[float],
    ) -> float:
        """
        Computes the Phase-Locking Value (PLV):
        PLV = (1 / N) * |sum_{k=1}^N exp(i * (phi_a[k] - phi_b[k]))|
        """
        n = min(len(phase_series_a), len(phase_series_b))
        if n == 0:
            return 0.0

        real_sum = 0.0
        imag_sum = 0.0

        for k in range(n):
            diff = phase_series_a[k] - phase_series_b[k]
            real_sum += math.cos(diff)
            imag_sum += math.sin(diff)

        plv = math.sqrt(real_sum * real_sum + imag_sum * imag_sum) / n
        return min(1.0, max(0.0, plv))

    def evaluate_tri_modal_stream(
        self,
        erp_phases: Sequence[float],
        saccade_phases: Sequence[float],
        ptt_phases: Sequence[float],
    ) -> LivenessVerdict:
        """
        Evaluates cross-modal phase coherence between cortical ERP,
        micro-saccadic eye tremor, and cardiovascular pulse transit time.
        """
        now = time.time()

        # 1. Cross-modal phase coherence PLV
        plv_erp_saccade = self._compute_phase_locking_value(erp_phases, saccade_phases)
        plv_saccade_ptt = self._compute_phase_locking_value(saccade_phases, ptt_phases)

        # Composite geometric mean coherence
        composite_coherence = math.sqrt(plv_erp_saccade * plv_saccade_ptt)

        is_live = composite_coherence >= self.min_coherence_threshold
        spoof_prob = 1.0 - composite_coherence

        if not is_live:
            if plv_erp_saccade < 0.3:
                cause = "REJECTED: Synthetic Cortical Replay (Decoupled ERP-Ocular Phase)"
            elif plv_saccade_ptt < 0.3:
                cause = "REJECTED: Deepfake Face/Video Injection (Decoupled Ocular-Cardio Phase)"
            else:
                cause = "REJECTED: Insufficient Multi-Modal Physiological Coherence"
        else:
            cause = "ACCEPTED: Live Biological Tri-Modal Synchronization Verified"

        return LivenessVerdict(
            timestamp=now,
            is_live_human=is_live,
            liveness_coherence_score=composite_coherence,
            erp_ocular_coherence=plv_erp_saccade,
            ocular_cardio_coherence=plv_saccade_ptt,
            spoof_probability=spoof_prob,
            rejection_cause=cause,
        )
