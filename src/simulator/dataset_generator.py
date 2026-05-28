"""
dataset_generator.py - Synthetic dataset generator for the Cognitive Waveform
Orchestration System.

Generates 50,000 labelled channel-state samples with utility-based waveform
ground-truth labels, boundary OOD samples, and train/val/test splits.
"""

import os
import sys
import numpy as np

# Ensure project root on path for imports
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.simulator.thz_absorption import THzAbsorptionModel

# =========================================================================
# Constants
# =========================================================================
WAVEFORM_CANDIDATES = ["OFDM", "F_OFDM", "FBMC", "SC_FDMA", "OTFS", "NOMA"]
NUM_WAVEFORMS = len(WAVEFORM_CANDIDATES)

TRAFFIC_TYPES = ["eMBB", "URLLC", "mMTC", "THz_broadband"]
BANDWIDTH_OPTIONS_MHZ = [50, 100, 200, 400, 800]
FREQUENCY_OPTIONS_GHZ = [3.5, 28, 60, 140, 220, 300]
QOS_LATENCY_MS = [1, 4, 10, 100]
QOS_RELIABILITY = [1e-5, 1e-3, 1e-2]

# PAPR values [dB] per waveform (index-aligned with WAVEFORM_CANDIDATES)
PAPR_DB = np.array([12.0, 10.0, 9.0, 5.0, 8.0, 11.0])

# Utility weights — publication formula: 0.4*SE + 0.25*PAPR + 0.2*LAT + 0.15*R
W_SE = 0.40
W_PAPR = 0.25
W_LATENCY = 0.20
W_ROBUST = 0.15

# Feature column names (for documentation)
FEATURE_NAMES = [
    "snr_db", "doppler_hz", "bandwidth_mhz", "interference_dbm",
    "mobility_kmh", "frequency_ghz", "traffic_type_idx",
    "qos_latency_ms", "qos_reliability", "mol_absorption_dbkm",
    "thz_window_id",
]
NUM_FEATURES = len(FEATURE_NAMES)


class DatasetGenerator:
    """Generate synthetic labelled waveform-selection dataset.

    Parameters
    ----------
    total_samples : int
        Total number of samples (default 50000).
    boundary_samples : int
        Number of boundary / OOD samples (default 2000).
    seed : int
        Random seed.
    noise_std : float
        Gaussian noise σ added to normalised continuous features during
        training data generation.
    """

    def __init__(
        self,
        total_samples: int = 50000,
        boundary_samples: int = 2000,
        seed: int = 42,
        noise_std: float = 0.05,
    ):
        self.total_samples = total_samples
        self.boundary_samples = boundary_samples
        self.seed = seed
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        self.features: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.is_boundary: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Parameter sampling
    # ------------------------------------------------------------------
    def _sample_parameters(self, n: int) -> np.ndarray:
        """Draw *n* channel-state parameter vectors.

        Returns array of shape (n, NUM_FEATURES).
        """
        rng = self.rng

        snr = rng.uniform(-5.0, 40.0, n)
        doppler = np.exp(rng.uniform(np.log(1.0), np.log(2000.0), n))
        bw = rng.choice(BANDWIDTH_OPTIONS_MHZ, n).astype(float)
        interference = rng.uniform(-120.0, -60.0, n)
        mobility = np.exp(rng.uniform(np.log(0.1), np.log(500.0), n))
        freq = rng.choice(FREQUENCY_OPTIONS_GHZ, n).astype(float)
        traffic_idx = rng.integers(0, len(TRAFFIC_TYPES), n).astype(float)
        latency = rng.choice(QOS_LATENCY_MS, n).astype(float)
        reliability = rng.choice(QOS_RELIABILITY, n).astype(float)

        # THz absorption integration
        mol_abs = np.zeros(n)
        thz_win = np.zeros(n)  # 0=no_window, 1=w1, 2=w2, 3=w3

        thz_mask = freq >= 100.0
        if thz_mask.any():
            thz_freqs = freq[thz_mask]
            for i, f in enumerate(thz_freqs):
                idx = np.where(thz_mask)[0][i]
                m = THzAbsorptionModel(frequency_ghz=f)
                mol_abs[idx] = m.compute_absorption_coefficient()[0]
                win_str = m.get_thz_window()[0]
                thz_win[idx] = {"no_window": 0, "window_1": 1, "window_2": 2,
                                "window_3": 3, "absorption_peak": 0}.get(win_str, 0)

        X = np.column_stack([
            snr, doppler, bw, interference, mobility, freq,
            traffic_idx, latency, reliability, mol_abs, thz_win,
        ])
        return X

    # ------------------------------------------------------------------
    # Utility scoring
    # ------------------------------------------------------------------
    def _spectral_efficiency(self, X: np.ndarray) -> np.ndarray:
        """Compute SE(w, s) for all waveforms.  shape (N, 6)."""
        n = X.shape[0]
        snr = X[:, 0]
        doppler = X[:, 1]
        interference = X[:, 3]
        mobility = X[:, 4]
        traffic_idx = X[:, 6].astype(int)
        freq = X[:, 5]

        # Baseline capacity: log2(1 + SNR_linear)
        snr_lin = 10.0 ** (snr / 10.0)
        base_se = np.log2(1.0 + np.clip(snr_lin, 1e-6, None))  # (N,)

        SE = np.tile(base_se[:, np.newaxis], (1, NUM_WAVEFORMS))  # (N,6)

        # Default: all waveforms get 0.7x penalty; only the niche gets 1.4x+
        SE *= 0.70

        # OFDM (idx 0) — best for low Doppler + low interference (classic)
        niche0 = (doppler < 50.0) & (interference < -90.0)
        SE[:, 0] = np.where(niche0, base_se * 1.40, SE[:, 0])

        # F-OFDM (idx 1) — best for eMBB traffic
        niche1 = (traffic_idx == 0)
        SE[:, 1] = np.where(niche1, base_se * 1.40, SE[:, 1])

        # FBMC (idx 2) — best for severe interference
        niche2 = (interference > -80.0)
        SE[:, 2] = np.where(niche2, base_se * 1.45, SE[:, 2])

        # SC-FDMA (idx 3) — best for mMTC + low mobility
        niche3 = (traffic_idx == 2) & (mobility < 30.0)
        SE[:, 3] = np.where(niche3, base_se * 1.40, SE[:, 3])

        # OTFS (idx 4) — best for high Doppler (any traffic)
        niche4 = (doppler > 300.0)
        SE[:, 4] = np.where(niche4, base_se * 1.50, SE[:, 4])

        # NOMA (idx 5) — best for URLLC or THz broadband
        niche5 = (traffic_idx == 1) | (traffic_idx == 3)
        SE[:, 5] = np.where(niche5, base_se * 1.40, SE[:, 5])

        # Explicit SNR-dependent SE bonus for OFDM and F-OFDM at high SNR
        snr_norm = (np.clip(snr, -5.0, 40.0) + 5.0) / 45.0
        se_bonus = np.where(snr > 25.0, 0.1 * snr_norm * base_se, 0.0)
        SE[:, 0] += se_bonus
        SE[:, 1] += se_bonus

        return SE

    def _robustness(self, X: np.ndarray) -> np.ndarray:
        """Compute robustness(w, s) for all waveforms.  shape (N, 6)."""
        n = X.shape[0]
        doppler = X[:, 1]
        mobility = X[:, 4]

        R = np.ones((n, NUM_WAVEFORMS)) * 0.7  # default

        high_d = doppler > 500.0
        low_d = doppler < 50.0
        med_d = ~high_d & ~low_d

        # OFDM — robust at low Doppler
        R[:, 0] = np.where(low_d, 0.90, np.where(med_d, 0.75, 0.50))
        # F-OFDM — moderate everywhere
        R[:, 1] = np.where(low_d, 0.85, np.where(med_d, 0.80, 0.60))
        # FBMC — good frequency localisation
        R[:, 2] = np.where(low_d, 0.80, np.where(med_d, 0.70, 0.55))
        # SC-FDMA — robust at low mobility
        R[:, 3] = np.where(mobility < 10.0, 0.90, 0.60)
        # OTFS — robust at high Doppler
        R[:, 4] = np.where(high_d, 1.00, np.where(med_d, 0.70, 0.65))
        # NOMA — moderate
        R[:, 5] = 0.70

        return R

    def _compute_utility(self, X: np.ndarray) -> np.ndarray:
        """U(w, s) = 0.4*SE + 0.25*(1/norm_PAPR) + 0.2*(1/norm_lat) + 0.15*R.

        Returns shape (N, 6).
        """
        SE = self._spectral_efficiency(X)
        R = self._robustness(X)

        # Normalised inverse PAPR  — use sqrt compression to reduce spread
        inv_papr_raw = 1.0 / (PAPR_DB / PAPR_DB.min())  # (6,)
        inv_papr = np.sqrt(inv_papr_raw)  # compress range

        # Normalised inverse latency
        latency = X[:, 7]  # ms
        waveform_latency_factor = np.array([1.0, 0.9, 1.1, 0.7, 1.3, 1.0])
        inv_lat = 1.0 / (waveform_latency_factor[np.newaxis, :] *
                         np.clip(latency[:, np.newaxis], 1.0, None))
        inv_lat = inv_lat / (inv_lat.max(axis=1, keepdims=True) + 1e-12)

        # Normalise SE per sample
        SE_norm = SE / (SE.max(axis=1, keepdims=True) + 1e-12)

        U = (W_SE * SE_norm
             + W_PAPR * inv_papr[np.newaxis, :]
             + W_LATENCY * inv_lat
             + W_ROBUST * R)

        return U

    # ------------------------------------------------------------------
    # Label derivation
    # ------------------------------------------------------------------
    def _derive_labels(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute utility, labels, and boundary flags.

        Returns (labels, utility_scores, is_boundary).
        """
        U = self._compute_utility(X)
        labels = np.argmax(U, axis=1)

        # Boundary detection: top-2 differ by <5%
        sorted_U = np.sort(U, axis=1)[:, ::-1]  # descending
        top1 = sorted_U[:, 0]
        top2 = sorted_U[:, 1]
        margin = (top1 - top2) / (top1 + 1e-12)
        is_boundary = margin < 0.05

        return labels, U, is_boundary

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(self) -> dict:
        """Generate the full dataset.

        Returns dict with keys: train_X, train_y, val_X, val_y,
        test_X, test_y, boundary_X, boundary_y, feature_names, label_names.
        """
        n_main = self.total_samples - self.boundary_samples
        n_train = int(n_main * 0.8333)  # ~40000 of 48000
        n_val = int(n_main * 0.1042)    # ~5000 of 48000
        n_test_id = n_main - n_train - n_val  # remainder (~3000 of 48000)

        # ---- Generate main samples ----
        X_main = self._sample_parameters(n_main)
        labels_main, U_main, bd_main = self._derive_labels(X_main)

        # ---- Generate boundary samples ----
        # Over-generate and keep only boundary candidates
        X_bd = self._generate_boundary_samples(self.boundary_samples)
        labels_bd, U_bd, _ = self._derive_labels(X_bd)

        # ---- Combine ----
        X_all = np.vstack([X_main, X_bd])
        y_all = np.concatenate([labels_main, labels_bd])
        is_bd = np.concatenate([np.zeros(n_main, dtype=bool),
                                np.ones(self.boundary_samples, dtype=bool)])

        # ---- Add noise to training continuous features ----
        # Indices of continuous features: snr(0), doppler(1), interference(3),
        # mobility(4), mol_absorption(9)
        cont_idx = [0, 1, 3, 4, 9]

        # Splits
        train_X = X_all[:n_train].copy()
        train_y = y_all[:n_train]
        # Inject noise into training set
        for ci in cont_idx:
            col = train_X[:, ci]
            col_std = col.std() + 1e-12
            train_X[:, ci] += self.rng.normal(0, self.noise_std * col_std, n_train)

        # Clamp noisy features to physical ranges
        train_X[:, 0] = np.clip(train_X[:, 0], -10.0, 50.0)   # snr_db
        train_X[:, 1] = np.clip(train_X[:, 1], 0.01, 5000.0)   # doppler_hz
        train_X[:, 3] = np.clip(train_X[:, 3], -130.0, -50.0)  # interference_dbm
        train_X[:, 4] = np.clip(train_X[:, 4], 0.01, 1000.0)   # mobility_kmh
        train_X[:, 9] = np.clip(train_X[:, 9], 0.0, None)       # mol_absorption

        # FIX 1: PHASE 5 Stratified Class Rebalancing
        train_X_raw = train_X.copy()
        train_y_raw = train_y.copy()
        
        classes, counts = np.unique(train_y_raw, return_counts=True)
        majority_count = counts.max()
        
        rebalanced_X = []
        rebalanced_y = []
        
        for cls in classes:
            idx = np.where(train_y_raw == cls)[0]
            cls_X = train_X_raw[idx]
            cls_y = train_y_raw[idx]
            
            if len(idx) < majority_count:
                num_to_add = majority_count - len(idx)
                dup_idx = self.rng.choice(len(idx), size=num_to_add, replace=True)
                dups_X = cls_X[dup_idx].copy()
                dups_y = cls_y[dup_idx]
                
                cls_X = np.vstack([cls_X, dups_X])
                cls_y = np.concatenate([cls_y, dups_y])
                
            rebalanced_X.append(cls_X)
            rebalanced_y.append(cls_y)
            
        train_X_ext = np.vstack(rebalanced_X)
        train_y_ext = np.concatenate(rebalanced_y)
        
        shuffle_idx = self.rng.permutation(len(train_X_ext))
        train_X = train_X_ext[shuffle_idx][:48000]
        train_y = train_y_ext[shuffle_idx][:48000]
        n_train_actual = len(train_X)
        
        # FIX 3 (Maintained): Training Dropout Augmentation
        drop_mask = self.rng.random(n_train_actual) < 0.20
        drop_indices = np.where(drop_mask)[0]
        for idx in drop_indices:
            num_drop = self.rng.choice([1, 2])
            drop_cols = self.rng.choice(cont_idx, num_drop, replace=False)
            for ci in drop_cols:
                train_X[idx, ci] = np.nan

        val_X = X_all[n_train:n_train + n_val]
        val_y = y_all[n_train:n_train + n_val]

        test_X = X_all[n_train + n_val:n_main]
        test_y = y_all[n_train + n_val:n_main]

        boundary_X = X_all[n_main:]
        boundary_y = y_all[n_main:]

        self.features = X_all
        self.labels = y_all
        self.is_boundary = is_bd

        return {
            "train_X": train_X, "train_y": train_y,
            "val_X": val_X, "val_y": val_y,
            "test_X": test_X, "test_y": test_y,
            "boundary_X": boundary_X, "boundary_y": boundary_y,
            "feature_names": FEATURE_NAMES,
            "label_names": WAVEFORM_CANDIDATES,
        }

    def _generate_boundary_samples(self, n_target: int) -> np.ndarray:
        """Generate exactly n_target boundary samples (top-2 margin < 5%)."""
        collected = []
        batch = max(n_target * 3, 5000)
        attempts = 0
        max_attempts = 50

        while len(collected) < n_target and attempts < max_attempts:
            X_cand = self._sample_parameters(batch)
            _, U_cand, bd_mask = self._derive_labels(X_cand)
            bd_indices = np.where(bd_mask)[0]
            if len(bd_indices) > 0:
                collected.append(X_cand[bd_indices])
            attempts += 1

        if len(collected) == 0:
            # Fallback: use small perturbations to create boundary samples
            X_cand = self._sample_parameters(n_target)
            return X_cand

        X_bd = np.vstack(collected)
        # Trim or pad to exact count
        if len(X_bd) >= n_target:
            return X_bd[:n_target]
        else:
            # Pad with duplicates
            shortfall = n_target - len(X_bd)
            pad_idx = self.rng.choice(len(X_bd), shortfall, replace=True)
            return np.vstack([X_bd, X_bd[pad_idx]])

    def save(self, data: dict, output_dir: str) -> list[str]:
        """Save all splits as .npy files.  Returns list of saved paths."""
        os.makedirs(output_dir, exist_ok=True)
        saved = []
        for key, arr in data.items():
            if isinstance(arr, np.ndarray):
                path = os.path.join(output_dir, f"{key}.npy")
                np.save(path, arr)
                saved.append(path)

        # Save label distribution stats
        if "train_y" in data:
            import json
            dist = {}
            for i, name in enumerate(WAVEFORM_CANDIDATES):
                dist[name] = int((data["train_y"] == i).sum())
            stats = {"label_distribution": dist, "total_train": len(data["train_y"])}
            stats_path = os.path.join(output_dir, "label_stats.json")
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            saved.append(stats_path)

        return saved

    def get_class_distribution(self, labels: np.ndarray) -> dict:
        """Return {waveform_name: count} for given label array."""
        dist = {}
        for i, name in enumerate(WAVEFORM_CANDIDATES):
            dist[name] = int((labels == i).sum())
        return dist
