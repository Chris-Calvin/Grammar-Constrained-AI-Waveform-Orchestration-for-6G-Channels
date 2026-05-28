"""
tokenizer.py - Multi-Domain Channel State Tokenizer.

Tokenizes raw channel-state feature vectors into sequences of 12 discrete
token IDs, one per domain/parameter.  Each domain has its own sub-vocabulary
of categorical bins.
"""

import os
import pickle
import numpy as np

# =========================================================================
# Vocabulary definitions
# =========================================================================
# Each vocabulary maps a parameter domain to an ordered list of
# (token_name, predicate_function).  The *first* matching predicate wins,
# so order matters for contiguous ranges.

_VOCABS: dict[str, list[tuple[str, callable]]] = {}

# 0 — SNR [dB]
_VOCABS["snr"] = [
    ("very_low",  lambda v: v < 0),
    ("low",       lambda v: 0 <= v < 10),
    ("medium",    lambda v: 10 <= v < 20),
    ("high",      lambda v: 20 <= v < 30),
    ("very_high", lambda v: v >= 30),
]

# 1 — BER (derived)
_VOCABS["ber"] = [
    ("negligible", lambda v: v < 1e-6),
    ("low",        lambda v: 1e-6 <= v < 1e-4),
    ("moderate",   lambda v: 1e-4 <= v < 1e-2),
    ("high",       lambda v: 1e-2 <= v < 0.1),
    ("critical",   lambda v: v >= 0.1),
]

# 2 — Doppler [Hz]
_VOCABS["doppler"] = [
    ("static",       lambda v: v < 1),
    ("pedestrian",   lambda v: 1 <= v < 10),
    ("vehicular",    lambda v: 10 <= v < 200),
    ("high_speed",   lambda v: 200 <= v < 1000),
    ("aeronautical", lambda v: v >= 1000),
]

# 3 — Coherence bandwidth [MHz] (derived)
_VOCABS["coherence_bw"] = [
    ("very_narrow", lambda v: v < 1),
    ("narrow",      lambda v: 1 <= v < 5),
    ("medium",      lambda v: 5 <= v < 20),
    ("wide",        lambda v: 20 <= v < 100),
    ("very_wide",   lambda v: v >= 100),
]

# 4 — Interference [dBm]
_VOCABS["interference"] = [
    ("negligible", lambda v: v < -100),
    ("low",        lambda v: -100 <= v < -90),
    ("moderate",   lambda v: -90 <= v < -80),
    ("severe",     lambda v: -80 <= v < -70),
    ("critical",   lambda v: v >= -70),
]

# 5 — Mobility [km/h]
_VOCABS["mobility"] = [
    ("stationary", lambda v: v < 1),
    ("walking",    lambda v: 1 <= v < 10),
    ("urban",      lambda v: 10 <= v < 60),
    ("highway",    lambda v: 60 <= v < 200),
    ("aerial",     lambda v: v >= 200),
]

# 6 — Frequency band [GHz]
_VOCABS["band"] = [
    ("sub6",      lambda v: v < 6),
    ("mmWave_28", lambda v: 24 <= v < 30 if v >= 6 else False),
    ("mmWave_60", lambda v: 57 <= v < 66 if v >= 30 else False),
    ("THz_140",   lambda v: 130 <= v < 175 if v >= 66 else False),
    ("THz_220",   lambda v: 200 <= v < 300 if v >= 175 else False),
    ("THz_300",   lambda v: v >= 300),
]

# 7 — Molecular absorption [dB/km]
_VOCABS["absorption"] = [
    ("none",     lambda v: v < 1),
    ("low",      lambda v: 1 <= v < 10),
    ("moderate", lambda v: 10 <= v < 50),
    ("high",     lambda v: 50 <= v < 200),
    ("extreme",  lambda v: v >= 200),
]

# 8 — THz window (categorical integer from dataset)
_VOCABS["thz_window"] = [
    ("no_window",       lambda v: v == 0),
    ("window_1",        lambda v: v == 1),
    ("window_2",        lambda v: v == 2),
    ("window_3",        lambda v: v == 3),
    ("absorption_peak", lambda v: True),  # fallback
]

# 9 — Traffic type (categorical integer)
_VOCABS["traffic"] = [
    ("eMBB",           lambda v: v == 0),
    ("URLLC",          lambda v: v == 1),
    ("mMTC",           lambda v: v == 2),
    ("THz_broadband",  lambda v: True),   # fallback index 3
]

# 10 — QoS latency [ms]
_VOCABS["latency"] = [
    ("ultra_low", lambda v: v <= 1),
    ("low",       lambda v: v <= 4),
    ("medium",    lambda v: v <= 10),
    ("high",      lambda v: True),  # 100ms or anything else
]

# 11 — QoS reliability
_VOCABS["reliability"] = [
    ("ultra_high", lambda v: v <= 1e-5),
    ("high",       lambda v: v <= 1e-3),
    ("moderate",   lambda v: True),  # 1e-2 or anything else
]

# Ordered domain names (determines token position 0..11)
DOMAIN_ORDER = [
    "snr", "ber", "doppler", "coherence_bw", "interference", "mobility",
    "band", "absorption", "thz_window", "traffic", "latency", "reliability",
]

NUM_POSITIONS = len(DOMAIN_ORDER)  # 12

# =========================================================================
# Pre-compute global token ↔ ID mappings
# =========================================================================
_TOKEN_NAMES: list[str] = []          # global flat list
_DOMAIN_OFFSETS: dict[str, int] = {}  # domain → first global ID

_offset = 0
for domain in DOMAIN_ORDER:
    _DOMAIN_OFFSETS[domain] = _offset
    for name, _ in _VOCABS[domain]:
        _TOKEN_NAMES.append(f"{domain}:{name}")
    _offset += len(_VOCABS[domain])

TOTAL_VOCAB_SIZE = _offset  # sum of all domain vocab sizes

_TOKEN_NAME_TO_ID = {name: idx for idx, name in enumerate(_TOKEN_NAMES)}


class ChannelStateTokenizer:
    """Tokenize raw channel-state vectors into 12-token integer sequences.

    The tokenizer maps each of 12 channel-state domains to a discrete
    vocabulary bin, returning a list of global token IDs.

    Call ``fit(train_X)`` to learn normalisation statistics, then
    ``transform(X)`` to tokenise raw feature rows.
    """

    def __init__(self):
        self._fitted = False
        self._norm_stats: dict[str, tuple[float, float]] = {}

    # ------------------------------------------------------------------
    # Fit / Transform
    # ------------------------------------------------------------------
    def fit(self, raw_features: np.ndarray) -> "ChannelStateTokenizer":
        """Learn normalisation statistics from training data.

        Parameters
        ----------
        raw_features : np.ndarray, shape (N, 11)
            Raw feature matrix (as produced by DatasetGenerator).
        """
        self._norm_stats = {}
        for i in range(raw_features.shape[1]):
            col = raw_features[:, i]
            self._norm_stats[str(i)] = (float(col.mean()), float(col.std() + 1e-12))
        self._fitted = True
        return self

    def transform(self, raw_features: np.ndarray) -> np.ndarray:
        """Normalise and tokenise.

        Parameters
        ----------
        raw_features : np.ndarray, shape (N, 11)

        Returns
        -------
        np.ndarray, shape (N, 12)   — integer token IDs.
        """
        # Tokenisation uses the *raw* domain values (not normalised)
        # but we store normalisation stats for downstream use.
        N = raw_features.shape[0]
        token_ids = np.zeros((N, NUM_POSITIONS), dtype=np.int64)
        for i in range(N):
            token_ids[i] = self._tokenize_row(raw_features[i])
        return token_ids

    # ------------------------------------------------------------------
    # Core tokenize / detokenize
    # ------------------------------------------------------------------
    def tokenize(self, channel_state: dict | np.ndarray) -> list[int]:
        """Tokenize a single sample into 12 global token IDs.

        Parameters
        ----------
        channel_state : dict or 1-D array of 11 raw feature values.

        Returns
        -------
        list of 12 int token IDs.
        """
        if isinstance(channel_state, np.ndarray):
            return self._tokenize_row(channel_state).tolist()

        # Dict input — extract by key
        row = self._dict_to_row(channel_state)
        return self._tokenize_row(row).tolist()

    def detokenize(self, token_ids: list[int] | np.ndarray) -> list[str]:
        """Map token IDs back to human-readable names.

        Returns list of 12 strings like ``'snr:medium'``.
        """
        return [_TOKEN_NAMES[tid] for tid in token_ids]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @staticmethod
    def get_vocab_size() -> int:
        """Total number of unique tokens across all domains."""
        return TOTAL_VOCAB_SIZE

    @staticmethod
    def get_token_position_map() -> dict[int, str]:
        """Map position index (0 … 11) to domain name."""
        return {i: name for i, name in enumerate(DOMAIN_ORDER)}

    @staticmethod
    def get_domain_vocab(domain: str) -> list[str]:
        """Return ordered list of token names for a domain."""
        return [name for name, _ in _VOCABS[domain]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Pickle the fitted tokenizer."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "ChannelStateTokenizer":
        """Load a pickled tokenizer."""
        with open(path, "rb") as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _tokenize_row(self, row: np.ndarray) -> np.ndarray:
        """Tokenize a single 11-element raw feature row → 12 token IDs.

        Feature layout (from dataset_generator):
          0: snr_db, 1: doppler_hz, 2: bandwidth_mhz, 3: interference_dbm,
          4: mobility_kmh, 5: frequency_ghz, 6: traffic_type_idx,
          7: qos_latency_ms, 8: qos_reliability, 9: mol_absorption_dbkm,
          10: thz_window_id

        Derived features:
          BER ≈ 0.5 * erfc(sqrt(SNR_linear))   (BPSK baseline)
          Coherence BW ≈ 1 / (5 * RMS_delay_spread)   estimated from bandwidth
        """
        snr_db = float(row[0])
        doppler_hz = float(row[1])
        bw_mhz = float(row[2])
        interference_dbm = float(row[3])
        mobility_kmh = float(row[4])
        freq_ghz = float(row[5])
        traffic_idx = int(row[6])
        latency_ms = float(row[7])
        reliability = float(row[8])
        mol_abs = float(row[9])
        thz_win = int(row[10])

        # Derived: BER (BPSK)
        from scipy.special import erfc
        snr_lin = 10.0 ** (snr_db / 10.0)
        ber = 0.5 * erfc(np.sqrt(max(snr_lin, 0.0)))

        # Derived: Coherence bandwidth [MHz]
        # Estimate RMS delay spread from bandwidth (simple approximation)
        rms_delay_s = 1.0 / (bw_mhz * 1e6 + 1e-12)  # rough reciprocal
        coherence_bw_mhz = (1.0 / (5.0 * rms_delay_s)) / 1e6

        # Build domain value vector
        domain_values = {
            "snr": snr_db,
            "ber": ber,
            "doppler": doppler_hz,
            "coherence_bw": coherence_bw_mhz,
            "interference": interference_dbm,
            "mobility": mobility_kmh,
            "band": freq_ghz,
            "absorption": mol_abs,
            "thz_window": thz_win,
            "traffic": traffic_idx,
            "latency": latency_ms,
            "reliability": reliability,
        }

        tokens = np.zeros(NUM_POSITIONS, dtype=np.int64)
        for pos, domain in enumerate(DOMAIN_ORDER):
            val = domain_values[domain]
            offset = _DOMAIN_OFFSETS[domain]
            vocab = _VOCABS[domain]
            for local_id, (_, pred) in enumerate(vocab):
                if pred(val):
                    tokens[pos] = offset + local_id
                    break
        return tokens

    def _dict_to_row(self, d: dict) -> np.ndarray:
        """Convert a dict with named keys to a 11-element raw row."""
        return np.array([
            d.get("snr_db", 0),
            d.get("doppler_hz", 0),
            d.get("bandwidth_mhz", 100),
            d.get("interference_dbm", -100),
            d.get("mobility_kmh", 0),
            d.get("frequency_ghz", 3.5),
            d.get("traffic_type_idx", 0),
            d.get("qos_latency_ms", 10),
            d.get("qos_reliability", 1e-3),
            d.get("mol_absorption_dbkm", 0),
            d.get("thz_window_id", 0),
        ], dtype=np.float64)
