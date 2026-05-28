"""
thz_absorption.py - ITU-R P.676-12 Molecular Absorption Model for THz Bands.

Implements the Van Vleck-Weisskopf line-shape model for oxygen and water-vapor
absorption lines, THz-window identification, path-loss computation (FSPL +
molecular + rain), and absorption-class mapping.

References
----------
- ITU-R P.676-12 (2019): Attenuation by atmospheric gases
- ITU-R P.838-3  (2005): Rain-specific attenuation
"""

import numpy as np

# =========================================================================
# Physical constants
# =========================================================================
_C0 = 2.99792458e8           # speed of light  [m/s]
_K_B = 1.380649e-23          # Boltzmann const  [J/K]

# =========================================================================
# Spectral-line catalogue (dominant lines only)
# =========================================================================
# Water-vapor lines : (centre_freq_GHz, line_strength_coeff, temp_exponent)
# Strengths are *relative* coefficients normalised so the model gives the
# right order of magnitude vs ITU-R published curves at reference conditions.
_H2O_LINES = [
    # f_i (GHz),  S_coeff,  n_exp
    (  22.235,    0.0022,   2.143),
    ( 183.310,   26.000,    0.668),
    ( 325.153,    0.900,    1.540),
    ( 380.197,   11.000,    1.000),
    ( 448.001,    3.500,    0.770),
    ( 556.936,   52.000,    0.500),
    ( 752.033,    5.500,    0.586),
]

# Oxygen lines : (centre_freq_GHz, S_coeff, n_exp)
_O2_LINES = [
    # 60-GHz complex (represented as cluster of 5 effective lines)
    (  50.474,   3.600,    0.890),
    (  52.542,   3.150,    0.910),
    (  56.968,   5.100,    0.870),
    (  63.568,   4.800,    0.890),
    (  66.410,   3.000,    0.920),
    # Single strong line
    ( 118.750,   4.500,    0.630),
]


class THzAbsorptionModel:
    """ITU-R P.676-12 molecular absorption model.

    Parameters
    ----------
    frequency_ghz : float or array-like
        Operating frequency in GHz.
    temperature_kelvin : float
        Atmospheric temperature [K].
    pressure_hpa : float
        Atmospheric pressure [hPa].
    water_vapor_density_gm3 : float
        Water-vapor density [g/m³].
    """

    def __init__(
        self,
        frequency_ghz: float = 300.0,
        temperature_kelvin: float = 293.0,
        pressure_hpa: float = 1013.25,
        water_vapor_density_gm3: float = 7.5,
    ):
        self.frequency_ghz = np.atleast_1d(np.asarray(frequency_ghz, dtype=np.float64))
        self.temperature = temperature_kelvin
        self.pressure = pressure_hpa
        self.rho_w = water_vapor_density_gm3

        # Reference temperature ratio  (ITU-R P.676 notation: theta)
        self._theta = 300.0 / self.temperature

        # Pre-compute the absorption coefficient for stored frequency
        self._alpha_mol = self.compute_absorption_coefficient()

    # ------------------------------------------------------------------
    # Van Vleck-Weisskopf line shape
    # ------------------------------------------------------------------
    @staticmethod
    def _vvw_line_shape(f: np.ndarray, f_i: float, delta_f: float) -> np.ndarray:
        """Van Vleck-Weisskopf line shape F_i(f).

        F_i(f) = (f/f_i)^2 * [ δf / ((f - f_i)^2 + δf^2)
                              + δf / ((f + f_i)^2 + δf^2) ]
        """
        ratio_sq = (f / f_i) ** 2
        term1 = delta_f / ((f - f_i) ** 2 + delta_f ** 2)
        term2 = delta_f / ((f + f_i) ** 2 + delta_f ** 2)
        return ratio_sq * (term1 + term2)

    # ------------------------------------------------------------------
    # Absorption coefficient  [dB/km]
    # ------------------------------------------------------------------
    def compute_absorption_coefficient(
        self, frequency_ghz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute specific attenuation α_mol(f) in dB/km.

        Uses the ITU-R P.676 line-by-line summation for O₂ and H₂O.
        """
        f = self.frequency_ghz if frequency_ghz is None else np.atleast_1d(
            np.asarray(frequency_ghz, dtype=np.float64)
        )

        theta = self._theta
        p = self.pressure          # hPa
        rho = self.rho_w           # g/m³

        # Pressure-broadened half-width (GHz) — simplified P.676 Eq. 6a/b
        # δf ≈ a_w * (p * θ^0.8 + 1.1 * ρ * θ)   for H₂O
        # δf ≈ a_o * (p * θ^0.8)                   for O₂
        base_width_h2o = 2.784e-3 * (p * theta ** 0.8 + 1.1 * rho * theta)  # GHz
        base_width_o2 = 1.6e-3 * (p * theta ** 0.8)                          # GHz

        # Ensure minimum linewidth to avoid singularities
        base_width_h2o = max(base_width_h2o, 0.05)
        base_width_o2 = max(base_width_o2, 0.05)

        gamma = np.zeros_like(f)

        # --- Water-vapor contribution ---
        for f_i, s_coeff, n_exp in _H2O_LINES:
            S_i = s_coeff * rho * theta ** n_exp
            delta_f = base_width_h2o * (1.0 + 0.0005 * f_i)
            gamma += S_i * self._vvw_line_shape(f, f_i, delta_f)

        # --- Oxygen contribution ---
        for f_i, s_coeff, n_exp in _O2_LINES:
            # O₂ density proportional to pressure
            S_i = s_coeff * (p / 1013.25) * theta ** n_exp
            delta_f = base_width_o2 * (1.0 + 0.0003 * f_i)
            gamma += S_i * self._vvw_line_shape(f, f_i, delta_f)

        # Continuum absorption (dry + wet) — small broadband term
        gamma += 1.1e-6 * f ** 2 * (p / 1013.25) * theta ** 2  # dry
        gamma += 3.6e-6 * f ** 2 * rho * theta ** 3              # wet

        # Ensure non-negative
        gamma = np.maximum(gamma, 0.0)

        return gamma  # dB/km

    # ------------------------------------------------------------------
    # Path loss  [dB]
    # ------------------------------------------------------------------
    def compute_path_loss(
        self,
        distance_m: float | np.ndarray,
        rain_rate_mmh: float = 0.0,
    ) -> np.ndarray:
        """Total path loss PL(f, d) in dB.

        PL = FSPL + α_mol · d/1000 + α_rain · d/1000

        Parameters
        ----------
        distance_m : float or array-like
            Propagation distance in metres.
        rain_rate_mmh : float
            Rain rate in mm/h (0 = clear sky).

        Returns
        -------
        np.ndarray  shape (len(frequency), len(distance))
        """
        f = self.frequency_ghz  # (F,)
        d = np.atleast_1d(np.asarray(distance_m, dtype=np.float64))  # (D,)

        # Free-space path loss  [dB]
        # FSPL = 20 log10(4π f d / c)   with f in Hz, d in m
        f_hz = f * 1e9  # (F,)
        # Using broadcasting: (F,1) and (1,D) → (F,D)
        fspl = 20.0 * np.log10(
            4.0 * np.pi * f_hz[:, np.newaxis] * d[np.newaxis, :] / _C0
        )

        # Molecular absorption  [dB]
        alpha_mol = self._alpha_mol  # (F,)
        mol_loss = alpha_mol[:, np.newaxis] * (d[np.newaxis, :] / 1000.0)

        # Rain attenuation — ITU-R P.838 simplified
        alpha_rain = self._rain_attenuation(rain_rate_mmh)  # (F,)
        rain_loss = alpha_rain[:, np.newaxis] * (d[np.newaxis, :] / 1000.0)

        pl = fspl + mol_loss + rain_loss
        return pl  # (F, D)

    def _rain_attenuation(self, rain_rate: float) -> np.ndarray:
        """ITU-R P.838-3 simplified specific rain attenuation [dB/km]."""
        if rain_rate <= 0:
            return np.zeros_like(self.frequency_ghz)
        f = self.frequency_ghz
        # Simplified power-law: γ_R = k · R^α  (horizontal polarisation)
        # k and α are frequency-dependent; use regression approximation
        log_f = np.log10(np.clip(f, 1.0, 1000.0))
        k = 10.0 ** (-3.0 + 1.4 * log_f - 0.08 * log_f ** 2)
        alpha_coeff = 0.67 + 0.34 * np.exp(-0.1 * f)
        return k * rain_rate ** alpha_coeff

    # ------------------------------------------------------------------
    # THz window identification
    # ------------------------------------------------------------------
    def get_thz_window(self) -> np.ndarray:
        """Return window identifier string for each frequency.

        Returns
        -------
        np.ndarray of str
        """
        f = self.frequency_ghz
        alpha = self._alpha_mol
        results = np.full(f.shape, "no_window", dtype="U20")

        # Check proximity to known absorption lines
        all_lines = [l[0] for l in _H2O_LINES] + [l[0] for l in _O2_LINES]
        for i, freq in enumerate(f):
            near_line = any(abs(freq - fl) < 5.0 for fl in all_lines)
            if near_line and alpha[i] > 10.0:
                results[i] = "absorption_peak"
            elif freq < 100.0:
                results[i] = "no_window"
            elif 125.0 <= freq <= 175.0:
                results[i] = "window_1"
            elif 200.0 <= freq <= 300.0:
                results[i] = "window_2"
            elif 350.0 <= freq <= 450.0:
                results[i] = "window_3"
            elif freq >= 100.0:
                results[i] = "no_window"

        return results

    # ------------------------------------------------------------------
    # Absorption class
    # ------------------------------------------------------------------
    def get_absorption_class(self) -> np.ndarray:
        """Map α_mol to absorption severity class.

        Classes: none (<1), low (1-10), moderate (10-50),
                 high (50-200), extreme (>200)  [dB/km].
        """
        alpha = self._alpha_mol
        result = np.full(alpha.shape, "none", dtype="U10")
        result[alpha >= 1.0] = "low"
        result[alpha >= 10.0] = "moderate"
        result[alpha >= 50.0] = "high"
        result[alpha >= 200.0] = "extreme"
        return result

    def __repr__(self) -> str:
        fmin, fmax = self.frequency_ghz.min(), self.frequency_ghz.max()
        return (
            f"THzAbsorptionModel(f={fmin:.1f}-{fmax:.1f}GHz, "
            f"T={self.temperature}K, P={self.pressure}hPa, "
            f"ρ_w={self.rho_w}g/m³)"
        )
