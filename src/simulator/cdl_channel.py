"""
cdl_channel.py - 3GPP TR 38.901 Clustered Delay Line (CDL) Channel Models.

Implements CDL-A through CDL-E channel models with time-variant channel
impulse response generation, metric computation, and Doppler classification.

Reference: 3GPP TR 38.901 V17.0.0, Section 7.7.1
"""

import numpy as np
from scipy.special import erfc

# =========================================================================
# CDL Model Parameter Tables  (3GPP TR 38.901 Table 7.7.1-1 through 7.7.1-5)
# =========================================================================

# Each entry: (normalised_delay, power_dB, AoA_mean_deg, AoA_spread_deg)
# The normalised delays are multiplied by the desired-delay-spread to get
# actual cluster delays.

_CDL_A_TABLE = {
    "n_clusters": 23,
    "delay_spread_ns": 65.0,
    "aoa_spread_deg": 104.0,
    "k_factor_db": None,         # NLOS
    "cluster_powers_db": np.array([
        -13.4,  0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5,
        -7.5, -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3,
        -12.7, -16.2, -18.3, -18.9, -14.9, -18.6, -20.8,
    ]),
}

_CDL_B_TABLE = {
    "n_clusters": 22,
    "delay_spread_ns": 100.0,
    "aoa_spread_deg": 80.0,
    "k_factor_db": None,
    "cluster_powers_db": np.array([
         0.0, -2.2, -0.6, -0.6, -3.4, -3.6, -5.7, -7.0,
        -4.0, -8.0, -6.6, -10.0, -9.2, -11.0, -10.5, -13.0,
        -12.0, -14.0, -13.8, -17.2, -15.0, -18.7,
    ]),
}

_CDL_C_TABLE = {
    "n_clusters": 24,
    "delay_spread_ns": 300.0,
    "aoa_spread_deg": 60.0,
    "k_factor_db": None,
    "cluster_powers_db": np.array([
        -4.4, -1.2,  0.0, -2.0, -3.0, -5.2, -7.0, -8.0,
        -3.0, -6.6, -5.2, -9.0, -8.0, -11.0, -10.0, -12.0,
        -13.0, -14.0, -13.6, -16.0, -15.2, -17.6, -16.0, -20.0,
    ]),
}

_CDL_D_TABLE = {
    "n_clusters": 13,
    "delay_spread_ns": 15.0,
    "aoa_spread_deg": 25.0,
    "k_factor_db": 13.3,           # LOS
    "cluster_powers_db": np.array([
         0.0, -13.5, -18.8, -21.0, -22.8, -17.9, -20.1,
        -21.9, -22.9, -27.8, -23.6, -24.8, -30.0,
    ]),
}

_CDL_E_TABLE = {
    "n_clusters": 14,
    "delay_spread_ns": 20.0,
    "aoa_spread_deg": 30.0,
    "k_factor_db": 22.0,           # LOS
    "cluster_powers_db": np.array([
         0.0, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6,
        -20.8, -22.6, -22.3, -25.6, -20.2, -29.0, -31.0,
    ]),
}

_CDL_TABLES = {
    "A": _CDL_A_TABLE,
    "B": _CDL_B_TABLE,
    "C": _CDL_C_TABLE,
    "D": _CDL_D_TABLE,
    "E": _CDL_E_TABLE,
}

# Speed of light (m/s)
_C0 = 3.0e8


class CDLChannel:
    """3GPP TR 38.901 Clustered Delay Line channel model.

    Parameters
    ----------
    cdl_type : str
        One of 'A', 'B', 'C', 'D', 'E'.
    carrier_frequency_ghz : float
        Carrier frequency in GHz.
    terminal_velocity_kmh : float
        UE velocity in km/h.
    random_seed : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        cdl_type: str = "A",
        carrier_frequency_ghz: float = 3.5,
        terminal_velocity_kmh: float = 30.0,
        random_seed: int = 42,
    ):
        cdl_type = cdl_type.upper()
        if cdl_type not in _CDL_TABLES:
            raise ValueError(f"Unknown CDL type '{cdl_type}'. Must be A/B/C/D/E.")

        self.cdl_type = cdl_type
        self.carrier_freq_ghz = carrier_frequency_ghz
        self.carrier_freq_hz = carrier_frequency_ghz * 1e9
        self.velocity_kmh = terminal_velocity_kmh
        self.velocity_ms = terminal_velocity_kmh / 3.6   # m/s
        self.seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # CDL table look-up
        tbl = _CDL_TABLES[cdl_type]
        self.n_clusters = tbl["n_clusters"]
        self.delay_spread_ns = tbl["delay_spread_ns"]
        self.aoa_spread_deg = tbl["aoa_spread_deg"]
        self.k_factor_db = tbl["k_factor_db"]

        # Derived physical quantities
        self.wavelength = _C0 / self.carrier_freq_hz          # m
        self.max_doppler_hz = self.velocity_ms / self.wavelength  # Hz

        # Cluster powers (linear) from dB table
        cluster_powers_db = tbl["cluster_powers_db"]
        self._cluster_powers_lin = 10.0 ** (cluster_powers_db / 10.0)
        # Normalise so total power = 1
        self._cluster_powers_lin /= self._cluster_powers_lin.sum()

        # Number of sub-rays per cluster (3GPP uses 20)
        self.n_rays_per_cluster = 20

        # Pre-generate per-cluster delays (normalised * DS)
        self._cluster_delays_s = self._generate_cluster_delays()

        # Pre-generate per-cluster AoA (mean + spread)
        self._cluster_aoa_rad = self._generate_cluster_aoa()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _generate_cluster_delays(self) -> np.ndarray:
        """Sorted uniform delays scaled by delay spread."""
        delays = self.rng.uniform(0, 1, self.n_clusters)
        delays = np.sort(delays)
        delays -= delays[0]  # first cluster at zero delay
        return delays * self.delay_spread_ns * 1e-9  # seconds

    def _generate_cluster_aoa(self) -> np.ndarray:
        """Cluster mean AoA sampled within the angular spread."""
        spread_rad = np.deg2rad(self.aoa_spread_deg)
        return self.rng.uniform(-spread_rad / 2, spread_rad / 2, self.n_clusters)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_channel_response(
        self,
        n_time_samples: int = 256,
        sampling_interval_s: float = 1e-6,
    ) -> np.ndarray:
        """Compute H(t, τ) as a complex channel impulse response matrix.

        H(t, τ_n) = Σ_n Σ_m  a_{n,m} · δ(τ - τ_n) · exp(j 2π ν_{n,m} t)

        Parameters
        ----------
        n_time_samples : int
            Number of time-domain snapshots.
        sampling_interval_s : float
            Time between successive snapshots (seconds).

        Returns
        -------
        H : np.ndarray, shape (n_time_samples, n_clusters)
            Complex channel impulse response.
        """
        t = np.arange(n_time_samples) * sampling_interval_s  # (T,)

        H = np.zeros((n_time_samples, self.n_clusters), dtype=np.complex128)

        for n in range(self.n_clusters):
            # Per-ray AoA within cluster angular spread
            cluster_aoa_mean = self._cluster_aoa_rad[n]
            ray_offsets = self.rng.standard_normal(self.n_rays_per_cluster) * np.deg2rad(5.0)
            ray_aoa = cluster_aoa_mean + ray_offsets  # (M,)

            # Doppler shift per ray: ν_{n,m} = (v / λ) cos(θ_{n,m})
            doppler_nm = (self.velocity_ms / self.wavelength) * np.cos(ray_aoa)  # (M,)

            # Complex Gaussian gain per ray scaled by sqrt(cluster power / M)
            sigma = np.sqrt(self._cluster_powers_lin[n] / (2.0 * self.n_rays_per_cluster))
            a_nm = (
                self.rng.standard_normal(self.n_rays_per_cluster) * sigma
                + 1j * self.rng.standard_normal(self.n_rays_per_cluster) * sigma
            )  # (M,)

            # LOS component for CDL-D / CDL-E (first cluster only)
            if self.k_factor_db is not None and n == 0:
                k_lin = 10.0 ** (self.k_factor_db / 10.0)
                los_amplitude = np.sqrt(k_lin / (k_lin + 1.0))
                # Scale NLOS part
                a_nm *= np.sqrt(1.0 / (k_lin + 1.0))
                # Add deterministic LOS ray at boresight (AoA = 0)
                doppler_los = self.velocity_ms / self.wavelength
                phase_los = 2.0 * np.pi * doppler_los * t
                H[:, n] += los_amplitude * np.exp(1j * phase_los)

            # Sum over rays: exp(j 2π ν_{n,m} t) weighted by a_{n,m}
            # shape: (T, M) via outer product
            phase_matrix = 2.0 * np.pi * np.outer(t, doppler_nm)  # (T, M)
            H[:, n] += (a_nm[np.newaxis, :] * np.exp(1j * phase_matrix)).sum(axis=1)

        return H

    def compute_channel_metrics(
        self,
        transmit_power_dbm: float = 23.0,
        path_loss_db: float = 80.0,
        bandwidth_hz: float = 100e6,
        noise_figure_db: float = 7.0,
    ) -> dict:
        """Compute key channel quality metrics.

        Returns
        -------
        dict with keys:
            snr_db, rms_delay_spread_ns, coherence_bw_mhz,
            max_doppler_hz, coherence_time_ms, estimated_ber_bpsk.
        """
        # Thermal noise power (dBm)
        thermal_noise_dbm = -174.0 + 10.0 * np.log10(bandwidth_hz)
        noise_power_dbm = thermal_noise_dbm + noise_figure_db

        # SNR in dB
        snr_db = transmit_power_dbm - path_loss_db - noise_power_dbm

        # RMS delay spread
        delays = self._cluster_delays_s
        powers = self._cluster_powers_lin
        mean_delay = np.sum(powers * delays) / np.sum(powers)
        rms_delay_s = np.sqrt(
            np.sum(powers * (delays - mean_delay) ** 2) / np.sum(powers)
        )
        rms_delay_ns = rms_delay_s * 1e9

        # Coherence bandwidth (MHz)  =  1 / (5 * σ_τ)
        if rms_delay_s > 0:
            coherence_bw_hz = 1.0 / (5.0 * rms_delay_s)
        else:
            coherence_bw_hz = bandwidth_hz  # degenerate single-tap
        coherence_bw_mhz = coherence_bw_hz / 1e6

        # Max Doppler frequency
        max_doppler = self.max_doppler_hz

        # Coherence time (ms) = 0.423 / f_d
        if max_doppler > 0:
            coherence_time_s = 0.423 / max_doppler
        else:
            coherence_time_s = np.inf
        coherence_time_ms = coherence_time_s * 1e3

        # Estimated BER for BPSK: 0.5 * erfc(sqrt(SNR_linear))
        snr_linear = 10.0 ** (snr_db / 10.0)
        estimated_ber = 0.5 * erfc(np.sqrt(max(snr_linear, 0.0)))

        return {
            "snr_db": float(snr_db),
            "rms_delay_spread_ns": float(rms_delay_ns),
            "coherence_bw_mhz": float(coherence_bw_mhz),
            "max_doppler_hz": float(max_doppler),
            "coherence_time_ms": float(coherence_time_ms),
            "estimated_ber_bpsk": float(estimated_ber),
        }

    def get_doppler_class(self) -> str:
        """Map max Doppler frequency to mobility class.

        Returns
        -------
        str
            One of 'static', 'pedestrian', 'vehicular',
            'high_speed', 'aeronautical'.
        """
        fd = self.max_doppler_hz
        if fd < 1.0:
            return "static"
        elif fd < 10.0:
            return "pedestrian"
        elif fd < 200.0:
            return "vehicular"
        elif fd < 1000.0:
            return "high_speed"
        else:
            return "aeronautical"

    def __repr__(self) -> str:
        return (
            f"CDLChannel(type={self.cdl_type}, "
            f"fc={self.carrier_freq_ghz:.1f}GHz, "
            f"v={self.velocity_kmh:.0f}km/h, "
            f"clusters={self.n_clusters}, "
            f"DS={self.delay_spread_ns:.0f}ns, "
            f"f_d={self.max_doppler_hz:.1f}Hz)"
        )
