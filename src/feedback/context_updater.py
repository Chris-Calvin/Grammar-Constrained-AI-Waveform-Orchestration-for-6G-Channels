"""
context_updater.py - Closed-Loop Feedback Context Updater.

Implements feedback-driven context adaptation for the waveform orchestration
system, including transmission simulation, quality delta computation,
feedback tokenization, and exponential-decay context embedding.
"""

import math
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from dataclasses import dataclass

# =========================================================================
# Feedback token vocabulary
# =========================================================================
FEEDBACK_TOKENS = [
    "STRONGLY_POSITIVE",  # delta > 0.2
    "POSITIVE",           # 0.05 to 0.2
    "NEUTRAL",            # -0.05 to 0.05
    "NEGATIVE",           # -0.2 to -0.05
    "STRONGLY_NEGATIVE",  # < -0.2
]

WAVEFORM_NAMES = ["OFDM", "F_OFDM", "FBMC", "SC_FDMA", "OTFS", "NOMA"]

# SE targets per traffic type (fraction of theoretical max)
SE_TARGETS = {
    "eMBB": 0.8,
    "URLLC": 0.5,
    "mMTC": 0.3,
    "THz_broadband": 0.7,
}

# Typical PAPR values per waveform [dB]
PAPR_TYPICAL = {
    "OFDM": 10.5,
    "F_OFDM": 9.8,
    "FBMC": 9.2,
    "SC_FDMA": 6.5,
    "OTFS": 8.5,
    "NOMA": 11.0,
}

# PAPR budget per traffic class [dB]
PAPR_BUDGET = {
    "eMBB": 12.0,
    "URLLC": 10.0,
    "mMTC": 8.0,
    "THz_broadband": 11.0,
}

# BER targets per traffic type
BER_TARGETS = {
    "eMBB": 1e-3,
    "URLLC": 1e-5,
    "mMTC": 1e-2,
    "THz_broadband": 1e-4,
}


# =========================================================================
# Transmission Simulator
# =========================================================================
@dataclass
class TransmissionResult:
    """Results from a simulated transmission."""
    se_achieved: float       # bits/s/Hz
    ber_achieved: float      # dimensionless
    papr_actual: float       # dB
    throughput_mbps: float   # Mbps
    latency_ms: float        # ms


class TransmissionSimulator:
    """Simulates transmission outcomes using closed-form approximations.

    Computes theoretical BER, spectral efficiency, and PAPR for the
    selected waveform under given channel conditions.
    """

    def simulate(
        self, waveform_type: str, snr_db: float, doppler_hz: float,
        bandwidth_mhz: float, traffic_type: str,
        mol_absorption_dbkm: float = 0.0, distance_km: float = 0.1,
    ) -> TransmissionResult:
        """Simulate a single transmission.

        Parameters
        ----------
        waveform_type : str
        snr_db : float
        doppler_hz : float
        bandwidth_mhz : float
        traffic_type : str
        mol_absorption_dbkm : float
        distance_km : float

        Returns
        -------
        TransmissionResult
        """
        snr_lin = 10.0 ** (snr_db / 10.0)

        # Effective SNR: degrade by Doppler and absorption
        doppler_penalty = 1.0 / (1.0 + (doppler_hz / 500.0) ** 1.5)
        absorption_loss = mol_absorption_dbkm * distance_km  # dB
        effective_snr_db = snr_db - absorption_loss
        effective_snr_db = max(effective_snr_db, -10.0)
        effective_snr_lin = 10.0 ** (effective_snr_db / 10.0)

        # Waveform-specific SE factor
        se_factor = {
            "OFDM": 1.0, "F_OFDM": 1.05, "FBMC": 1.08,
            "SC_FDMA": 0.85, "OTFS": 1.1, "NOMA": 1.15,
        }.get(waveform_type, 1.0)

        # Doppler resilience per waveform
        doppler_resilience = {
            "OFDM": 0.7, "F_OFDM": 0.8, "FBMC": 0.75,
            "SC_FDMA": 0.9, "OTFS": 1.0, "NOMA": 0.65,
        }.get(waveform_type, 0.7)

        effective_doppler_penalty = doppler_penalty ** (1.0 / max(doppler_resilience, 0.1))

        # Spectral efficiency: Shannon bound × waveform factor × Doppler
        se_theoretical = math.log2(1.0 + effective_snr_lin) * se_factor
        se_achieved = se_theoretical * effective_doppler_penalty
        se_achieved = max(se_achieved, 0.01)

        # BER approximation (BPSK baseline, waveform adjusted)
        ber_factor = {
            "OFDM": 1.0, "F_OFDM": 0.9, "FBMC": 0.95,
            "SC_FDMA": 0.7, "OTFS": 0.6, "NOMA": 1.2,
        }.get(waveform_type, 1.0)

        from scipy.special import erfc
        ber_base = 0.5 * erfc(math.sqrt(max(effective_snr_lin, 0.0)))
        ber_achieved = ber_base * ber_factor * (1.0 / max(effective_doppler_penalty, 0.01))
        ber_achieved = min(max(ber_achieved, 1e-12), 0.5)

        # PAPR with small random variation (±1 dB)
        papr_base = PAPR_TYPICAL.get(waveform_type, 10.0)
        rng = np.random.default_rng()
        papr_actual = papr_base + rng.normal(0, 0.5)
        papr_actual = max(papr_actual, 3.0)

        # Throughput
        throughput_mbps = se_achieved * bandwidth_mhz

        # Latency approximation
        latency_base = {"eMBB": 4.0, "URLLC": 0.5, "mMTC": 10.0, "THz_broadband": 2.0}
        latency_ms = latency_base.get(traffic_type, 5.0) * (1.0 + 0.1 * rng.normal())
        latency_ms = max(latency_ms, 0.1)

        return TransmissionResult(
            se_achieved=se_achieved,
            ber_achieved=ber_achieved,
            papr_actual=papr_actual,
            throughput_mbps=throughput_mbps,
            latency_ms=latency_ms,
        )


# =========================================================================
# Closed-Loop Feedback Updater
# =========================================================================
class ClosedLoopFeedbackUpdater:
    """Feedback-driven context adaptation with exponential-decay buffer.

    Parameters
    ----------
    max_buffer_length : int
        Maximum number of feedback entries in the buffer.
    decay_rate : float
        Exponential decay rate for weighting buffer entries.
    embed_dim : int
        Dimension of the waveform+feedback embedding.
    """

    def __init__(
        self, max_buffer_length: int = 8, decay_rate: float = 0.3,
        embed_dim: int = 64,
    ):
        self.max_buffer_length = max_buffer_length
        self.decay_rate = decay_rate
        self.embed_dim = embed_dim
        self.context_dim = 16

        # Buffer of (waveform_token, feedback_token) pairs
        self.buffer: deque[tuple[str, str]] = deque(maxlen=max_buffer_length)

        # Learnable embeddings for waveform + feedback tokens
        self.waveform_embeddings = nn.Embedding(len(WAVEFORM_NAMES), embed_dim)
        self.feedback_embeddings = nn.Embedding(len(FEEDBACK_TOKENS), embed_dim)

        # Projection to context_dim
        self.context_projection = nn.Linear(embed_dim, self.context_dim)

        # Transmission simulator
        self.simulator = TransmissionSimulator()

    # ------------------------------------------------------------------
    # Quality delta
    # ------------------------------------------------------------------
    def compute_quality_delta(
        self, selected_config: str, channel_state: dict,
        transmission_result: TransmissionResult,
    ) -> float:
        """Compute composite quality delta from transmission result.

        Parameters
        ----------
        selected_config : str
            Waveform type name (e.g., "OFDM").
        channel_state : dict
            Channel state with keys: snr_db, traffic_type, etc.
        transmission_result : TransmissionResult

        Returns
        -------
        float — composite quality delta in [-1, 1] range.
        """
        traffic = channel_state.get("traffic_type", "eMBB")
        snr_db = channel_state.get("snr_db", 10.0)

        # SE delta
        snr_lin = 10.0 ** (snr_db / 10.0)
        se_max = math.log2(1.0 + snr_lin)
        se_target_frac = SE_TARGETS.get(traffic, 0.5)
        se_target = se_max * se_target_frac
        se_target = max(se_target, 0.01)
        delta_se = (transmission_result.se_achieved - se_target) / se_target
        delta_se = max(min(delta_se, 1.0), -1.0)

        # BER delta (positive = better than target)
        ber_target = BER_TARGETS.get(traffic, 1e-3)
        if transmission_result.ber_achieved < 1e-15:
            delta_ber = 1.0  # perfect
        elif ber_target < 1e-15:
            delta_ber = -1.0
        else:
            ratio = math.log10(ber_target) - math.log10(max(transmission_result.ber_achieved, 1e-15))
            delta_ber = max(min(ratio / 3.0, 1.0), -1.0)  # normalize

        # PAPR delta (positive = under budget)
        papr_budget = PAPR_BUDGET.get(traffic, 10.0)
        delta_papr = (papr_budget - transmission_result.papr_actual) / papr_budget
        delta_papr = max(min(delta_papr, 1.0), -1.0)

        # Composite
        composite = 0.4 * delta_se + 0.35 * delta_ber + 0.25 * delta_papr
        return float(max(min(composite, 1.0), -1.0))

    # ------------------------------------------------------------------
    # Feedback token encoding
    # ------------------------------------------------------------------
    @staticmethod
    def encode_feedback_token(quality_delta: float) -> str:
        """Map quality delta to feedback token.

        >0.2           → STRONGLY_POSITIVE
        0.05 to 0.2    → POSITIVE
        -0.05 to 0.05  → NEUTRAL
        -0.2 to -0.05  → NEGATIVE
        <-0.2          → STRONGLY_NEGATIVE
        """
        if quality_delta > 0.2:
            return "STRONGLY_POSITIVE"
        elif quality_delta > 0.05:
            return "POSITIVE"
        elif quality_delta >= -0.05:
            return "NEUTRAL"
        elif quality_delta >= -0.2:
            return "NEGATIVE"
        else:
            return "STRONGLY_NEGATIVE"

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------
    def update_buffer(self, waveform_token: str, feedback_token: str, token_ids=None) -> None:
        """Append a (waveform, feedback) pair to the buffer with decay."""
        import numpy as np
        from collections import deque
        if token_ids is not None:
            if hasattr(self, 'last_token_ids') and self.last_token_ids is not None:
                dist = np.linalg.norm(token_ids - self.last_token_ids)
                if dist > 2.0: # Significant change threshold
                    new_buf = deque(maxlen=self.max_buffer_length)
                    for entry in self.buffer:
                        w, f = entry[0], entry[1]
                        weight = entry[2] if len(entry) > 2 else 1.0
                        new_buf.append((w, f, weight * 0.5))
                    self.buffer = new_buf
            self.last_token_ids = token_ids.copy()
            
        self.buffer.append((waveform_token, feedback_token, 1.0))

    def is_stable(self) -> bool:
        # Bypass lock if most recent feedback is negative
        if len(self.buffer) > 0:
            last_fb = list(self.buffer)[-1][1]
            if last_fb in ["NEGATIVE", "STRONGLY_NEGATIVE"]:
                return False
                
        if len(self.buffer) < 3: return False
        recent = list(self.buffer)[-3:]
        for entry in recent:
            if entry[1] != "STRONGLY_POSITIVE":
                return False
        return True

    def clear_buffer(self) -> None:
        self.buffer.clear()

    # ------------------------------------------------------------------
    # Context embedding
    # ------------------------------------------------------------------
    def get_context_embedding(self) -> torch.Tensor:
        """Compute weighted mean of buffer embeddings with exponential decay.

        Returns
        -------
        Tensor, shape (context_dim,) = (16,)
        """
        if len(self.buffer) == 0:
            return torch.zeros(self.context_dim)

        buf_len = len(self.buffer)

        # Exponential decay weights: more recent entries have higher weight
        weights = []
        for i in range(buf_len):
            w = math.exp(-self.decay_rate * (buf_len - 1 - i))
            weights.append(w)
        w_sum = sum(weights)
        weights = [w / w_sum for w in weights]  # normalize to sum=1

        # Compute weighted embedding
        weighted_sum = torch.zeros(self.embed_dim)
        for i, entry in enumerate(self.buffer):
            wf = entry[0]
            fb = entry[1]
            wf_idx = WAVEFORM_NAMES.index(wf) if wf in WAVEFORM_NAMES else 0
            fb_idx = FEEDBACK_TOKENS.index(fb) if fb in FEEDBACK_TOKENS else 2

            with torch.no_grad():
                wf_emb = self.waveform_embeddings(torch.tensor(wf_idx))
                fb_emb = self.feedback_embeddings(torch.tensor(fb_idx))

            combined = wf_emb + fb_emb  # element-wise sum
            weighted_sum += weights[i] * combined

        # Project to context_dim
        with torch.no_grad():
            context = self.context_projection(weighted_sum)

        return context  # (16,)

    # ------------------------------------------------------------------
    # Augment token sequence
    # ------------------------------------------------------------------
    def augment_token_sequence(
        self, token_ids: torch.Tensor, context_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Prepend context embedding as an additional token.

        Parameters
        ----------
        token_ids : LongTensor, shape (B, 12) or (12,)
        context_embedding : Tensor, shape (16,) or (B, 16)

        Returns
        -------
        Tensor — augmented representation hint (context, token_ids concatenated as metadata).
        For simplicity, returns a dict-like tuple of both for downstream use.
        """
        # Return augmented tuple — the context is a auxiliary signal
        return token_ids, context_embedding

    # ------------------------------------------------------------------
    # Simulate a transmission
    # ------------------------------------------------------------------
    def simulate_transmission(
        self, selected_config: str, channel_state: dict,
    ) -> TransmissionResult:
        """Convenience wrapper around TransmissionSimulator."""
        return self.simulator.simulate(
            waveform_type=selected_config,
            snr_db=channel_state.get("snr_db", 10.0),
            doppler_hz=channel_state.get("doppler_hz", 10.0),
            bandwidth_mhz=channel_state.get("bandwidth_mhz", 100.0),
            traffic_type=channel_state.get("traffic_type", "eMBB"),
            mol_absorption_dbkm=channel_state.get("mol_absorption_dbkm", 0.0),
            distance_km=channel_state.get("distance_km", 0.1),
        )
