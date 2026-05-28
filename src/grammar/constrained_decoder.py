"""
constrained_decoder.py - Grammar-Constrained Beam Search Decoder.

Integrates the trained WaveformTransformerEncoder with WaveformGrammarValidator
to produce fully grammar-valid waveform configurations via beam search.

Architecture:
  Position 0 (WaveformType):  uses transformer's 6-class logits
  Positions 1-4 (Numerology, CPType, SCS, PAPRMode): uniform logits masked by grammar
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.transformer.model import WaveformTransformerEncoder
from src.grammar.waveform_grammar import (
    WaveformGrammarValidator, WaveformConfig, POS_VOCABS,
)

# Maps tokenizer Band domain labels → grammar band context labels
BAND_MAP = {
    "sub6":       "sub6",
    "mmWave_24":  "mmWave_28",
    "mmWave_28":  "mmWave_28",
    "mmWave_39":  "mmWave_28",
    "mmWave_60":  "mmWave_60",
    "THz_140":    "THz_140",
    "THz_220":    "THz_220",
    "THz_300":    "THz_300",
}

DOPPLER_LABELS = ["static", "pedestrian", "vehicular", "high_speed", "aeronautical"]
TRAFFIC_LABELS = ["eMBB", "URLLC", "mMTC", "THz_broadband"]
WAVEFORM_NAMES = ["OFDM", "F_OFDM", "FBMC", "SC_FDMA", "OTFS", "NOMA"]


class ConstrainedBeamSearchDecoder:
    """Grammar-constrained beam search integrating transformer + grammar validator.

    Parameters
    ----------
    model : WaveformTransformerEncoder
        Trained model (loaded weights).
    validator : WaveformGrammarValidator
        Grammar constraint checker.
    beam_width : int
        Number of beams to keep at each step.
    device : str
        'cpu' or 'cuda'.
    """

    def __init__(
        self,
        model: WaveformTransformerEncoder,
        validator: WaveformGrammarValidator,
        beam_width: int = 3,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.validator = validator
        self.beam_width = beam_width
        self.device = device

    # ------------------------------------------------------------------
    # Core decode
    # ------------------------------------------------------------------
    def decode(
        self, token_ids: torch.Tensor, channel_context: list[str],
    ) -> WaveformConfig:
        """Constrained beam search to produce a valid WaveformConfig.

        Parameters
        ----------
        token_ids : LongTensor, shape (12,) or (1, 12)
            Tokenized channel state.
        channel_context : list[str]
            Context tags like ["band:sub6", "doppler:static", "traffic:eMBB"].

        Returns
        -------
        WaveformConfig — the highest-scoring grammar-valid configuration.
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        # Get transformer logits for waveform type (position 0) — single call
        with torch.inference_mode():
            logits_np = self.model(token_ids.to(self.device)).squeeze(0).cpu().numpy()

        # Initialize beams: list of (partial_config, cumulative_log_prob)
        beams: list[tuple[list[str], float]] = [([], 0.0)]

        for pos in range(5):
            vocab = POS_VOCABS[pos]
            new_beams: list[tuple[list[str], float]] = []

            for partial, cum_logp in beams:
                # Get validity mask from grammar (cached)
                mask = self.validator.get_validity_mask(partial, pos, channel_context)

                # Build logits (numpy)
                if pos == 0:
                    raw = logits_np.copy()
                else:
                    raw = np.zeros(len(vocab), dtype=np.float32)

                # Mask invalid → -inf
                for k in range(len(vocab)):
                    if not mask[k]:
                        raw[k] = -1e9

                # Log-softmax in numpy
                raw_max = raw.max()
                log_sum_exp = raw_max + np.log(np.sum(np.exp(raw - raw_max)))
                log_probs = raw - log_sum_exp

                # Expand beams — only valid tokens
                for idx in range(len(vocab)):
                    if mask[idx]:
                        new_beams.append(
                            (partial + [vocab[idx]], cum_logp + float(log_probs[idx]))
                        )

            # Prune to beam_width
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[: self.beam_width]

        # Best beam → WaveformConfig
        best_partial, _ = beams[0]
        return WaveformConfig(*best_partial)

    def decode_from_logits(
        self, logits_np: np.ndarray, channel_context: list[str],
    ) -> WaveformConfig:
        """Beam search using pre-computed numpy logits (no model call).

        Parameters
        ----------
        logits_np : ndarray, shape (6,)
            Pre-computed waveform class logits.
        channel_context : list[str]

        Returns
        -------
        WaveformConfig
        """
        beams: list[tuple[list[str], float]] = [([], 0.0)]

        for pos in range(5):
            vocab = POS_VOCABS[pos]
            new_beams: list[tuple[list[str], float]] = []

            for partial, cum_logp in beams:
                mask = self.validator.get_validity_mask(partial, pos, channel_context)

                if pos == 0:
                    raw = logits_np.copy()
                else:
                    raw = np.zeros(len(vocab), dtype=np.float32)

                for k in range(len(vocab)):
                    if not mask[k]:
                        raw[k] = -1e9

                raw_max = raw.max()
                log_sum_exp = raw_max + np.log(np.sum(np.exp(raw - raw_max)))
                log_probs = raw - log_sum_exp

                for idx in range(len(vocab)):
                    if mask[idx]:
                        new_beams.append(
                            (partial + [vocab[idx]], cum_logp + float(log_probs[idx]))
                        )

            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[: self.beam_width]

        best_partial, _ = beams[0]
        return WaveformConfig(*best_partial)

    # ------------------------------------------------------------------
    # Decode with fallback
    # ------------------------------------------------------------------
    def decode_with_fallback(
        self, token_ids: torch.Tensor, channel_context: list[str],
    ) -> tuple[WaveformConfig, bool]:
        """Decode with fallback to enumeration if beam search fails.

        Returns
        -------
        (config, fallback_used)
        """
        try:
            config = self.decode(token_ids, channel_context)
            valid, _ = self.validator.is_valid_config(config, channel_context)
            if valid:
                return config, False
        except Exception:
            pass

        # Fallback: pick from valid configs
        valid_configs = self.validator.get_valid_configs_for_context(channel_context)
        if valid_configs:
            return valid_configs[0], True
        raise RuntimeError("No valid configurations exist for this context")

    # ------------------------------------------------------------------
    # Batch decode
    # ------------------------------------------------------------------
    def batch_decode(
        self,
        batch_token_ids: torch.Tensor,
        batch_channel_contexts: list[list[str]],
    ) -> list[WaveformConfig]:
        """Decode a batch of samples.

        Parameters
        ----------
        batch_token_ids : LongTensor, shape (B, 12)
        batch_channel_contexts : list of context lists, length B

        Returns
        -------
        list[WaveformConfig], length B
        """
        results = []
        for i in range(batch_token_ids.size(0)):
            config = self.decode(
                batch_token_ids[i], batch_channel_contexts[i],
            )
            results.append(config)
        return results

    # ------------------------------------------------------------------
    # Latency measurement
    # ------------------------------------------------------------------
    def get_decision_latency_ms(self, n_calls: int = 1000) -> float:
        """Measure average decode latency over n_calls.

        Uses a synthetic sample to benchmark.
        """
        from src.transformer.model import VOCAB_SIZE, SEQ_LEN
        torch.manual_seed(0)
        sample = torch.randint(0, VOCAB_SIZE, (SEQ_LEN,))
        ctx = ["band:sub6", "doppler:static", "traffic:eMBB"]

        # Warm up
        for _ in range(10):
            self.decode(sample, ctx)

        start = time.perf_counter()
        for _ in range(n_calls):
            self.decode(sample, ctx)
        elapsed = time.perf_counter() - start
        return (elapsed / n_calls) * 1000.0  # ms
