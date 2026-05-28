"""
waveform_grammar.py - Context-Free Grammar for 3GPP NR Waveforms.

Defines the grammar for waveform configurations and implements a validator
that enforces 8 hard constraints based on channel state context.
"""

import functools
from dataclasses import dataclass
from typing import Tuple, List
from lark import Lark, Tree

# =========================================================================
# Grammar Definition
# =========================================================================
# WaveformConfig: WaveformType Numerology CPType SCS PAPRMode
GRAMMAR_STRING = """
    ?start: waveform_config
    
    waveform_config: waveform_type " " numerology " " cp_type " " scs " " papr_mode
    
    waveform_type: "OFDM" | "F_OFDM" | "FBMC" | "SC_FDMA" | "OTFS" | "NOMA"
    numerology: "mu0" | "mu1" | "mu2" | "mu3" | "mu4"
    cp_type: "normal" | "extended"
    scs: "scs_15" | "scs_30" | "scs_60" | "scs_120" | "scs_240"
    papr_mode: "standard" | "reduced"
"""

# Vocabularies per position for mask generation
POS_VOCABS = [
    ["OFDM", "F_OFDM", "FBMC", "SC_FDMA", "OTFS", "NOMA"],  # 0: WaveformType
    ["mu0", "mu1", "mu2", "mu3", "mu4"],                    # 1: Numerology
    ["normal", "extended"],                                 # 2: CPType
    ["scs_15", "scs_30", "scs_60", "scs_120", "scs_240"],   # 3: SCS
    ["standard", "reduced"],                                # 4: PAPRMode
]

# Total vocab size for the decoder
GRAMMAR_VOCAB_SIZE = sum(len(v) for v in POS_VOCABS)


@dataclass(frozen=True)
class WaveformConfig:
    waveform_type: str
    numerology: str
    cp_type: str
    scs: str
    papr_mode: str
    
    def __str__(self):
        return f"{self.waveform_type} {self.numerology} {self.cp_type} {self.scs} {self.papr_mode}"
    
    def as_tuple(self) -> Tuple[str, str, str, str, str]:
        return (self.waveform_type, self.numerology, self.cp_type, self.scs, self.papr_mode)


class WaveformGrammarValidator:
    """Enforces 3GPP constraints on waveform configurations given channel context."""
    
    def __init__(self):
        self.parser = Lark(GRAMMAR_STRING, parser='lalr')
        
    def parse(self, config_str: str) -> WaveformConfig | None:
        """Parse string to WaveformConfig object if syntactically valid."""
        try:
            tree = self.parser.parse(config_str)
            children = tree.children
            return WaveformConfig(
                waveform_type=children[0].children[0].value,
                numerology=children[1].children[0].value,
                cp_type=children[2].children[0].value,
                scs=children[3].children[0].value,
                papr_mode=children[4].children[0].value,
            )
        except Exception:
            return None

    def is_valid_config(
        self, config: dict | WaveformConfig, channel_tokens: list[str]
    ) -> Tuple[bool, str]:
        """Validate configuration against 8 hard constraints.
        
        Constraints:
        C1: extended CP only with mu2
        C2: THz bands (>=100GHz) only permit mu3 or mu4
        C3: SC_FDMA restricted to mMTC traffic
        C4: OTFS only valid when Doppler is vehicular/high_speed/aeronautical
        C5: NOMA requires mMTC traffic
        C6: reduced PAPR only with SC_FDMA or F_OFDM
        C7: sub6 band only permits mu0/mu1/mu2
        C8: mmWave bands only permit mu2/mu3
        """
        if isinstance(config, WaveformConfig):
            w, mu, cp, scs, papr = config.as_tuple()
        else:
            w, mu, cp, scs, papr = (
                config.get("waveform_type"), config.get("numerology"), 
                config.get("cp_type"), config.get("scs"), config.get("papr_mode")
            )
            
        # Extract context
        band = ""
        doppler = ""
        traffic = ""
        for t in channel_tokens:
            if t.startswith("band:"): band = t.split(":")[1]
            elif t.startswith("doppler:"): doppler = t.split(":")[1]
            elif t.startswith("traffic:"): traffic = t.split(":")[1]

        # C1: extended CP only with mu2
        if cp == "extended" and mu != "mu2":
            return False, "C1: extended CP only permitted with mu2"
            
        # C2: THz bands (>=100GHz) only permit mu3 or mu4
        if band.startswith("THz_"):
            if mu not in ["mu3", "mu4"]:
                return False, f"C2: THz band ({band}) requires mu3 or mu4"
                
        # C3: SC_FDMA restricted to mMTC
        if w == "SC_FDMA" and traffic != "mMTC":
            # Note: prompt says "mMTC or uplink traffic", but traffic vocab doesn't have "uplink"
            # It only has: eMBB, URLLC, mMTC, THz_broadband. So we restrict to mMTC.
            return False, "C3: SC_FDMA restricted to mMTC traffic"
            
        # C4: OTFS only valid with specific Doppler classes
        if w == "OTFS":
            valid_doppler = ["vehicular", "high_speed", "aeronautical"]
            if doppler not in valid_doppler:
                return False, f"C4: OTFS invalid for Doppler class {doppler}"
                
        # C5: NOMA requires mMTC (multi-user)
        if w == "NOMA" and traffic != "mMTC":
            return False, "C5: NOMA requires mMTC traffic"
            
        # C6: reduced PAPR only with SC_FDMA or F_OFDM
        if papr == "reduced":
            if w not in ["SC_FDMA", "F_OFDM"]:
                return False, f"C6: reduced PAPR invalid with {w}"
                
        # C7: sub6 band only permits mu0/mu1/mu2
        if band == "sub6":
            if mu not in ["mu0", "mu1", "mu2"]:
                return False, "C7: sub6 band only permits mu0, mu1, or mu2"
                
        # C8: mmWave bands only permit mu2/mu3
        if band.startswith("mmWave_"):
            if mu not in ["mu2", "mu3"]:
                return False, "C8: mmWave band requires mu2 or mu3"
                
        return True, "Valid"

    def get_valid_configs_for_context(self, channel_tokens: list[str]) -> List[WaveformConfig]:
        """Brute-force enumerate all valid configs for a given context.
        
        Total search space per context: 6 * 5 * 2 * 5 * 2 = 600 configs.
        """
        valid_configs = []
        for w in POS_VOCABS[0]:
            for mu in POS_VOCABS[1]:
                for cp in POS_VOCABS[2]:
                    for scs in POS_VOCABS[3]:
                        for papr in POS_VOCABS[4]:
                            config = WaveformConfig(w, mu, cp, scs, papr)
                            valid, _ = self.is_valid_config(config, channel_tokens)
                            if valid:
                                valid_configs.append(config)
        return valid_configs

    def get_validity_mask(self, partial_config: list[str], position: int, channel_tokens: list[str]) -> list[bool]:
        """Compute validity mask for the vocabulary at the given decoding position.
        
        Parameters
        ----------
        partial_config : list[str]
            Tokens chosen so far (length == position).
        position : int
            Decoding step (0 to 4).
        channel_tokens : list[str]
            Context sequence.
            
        Returns
        -------
        list[bool]
            Length equals len(POS_VOCABS[position]). True if token is valid.
        """
        # Use cached module-level function with hashable args
        return _get_validity_mask_cached(
            tuple(partial_config), position, tuple(channel_tokens)
        )

    def enumerate_all_valid_configs(self) -> int:
        """Count valid configs across ALL possible channel contexts.
        
        We iterate over the relevant determining context variables:
        Band (6) x Doppler (5) x Traffic (4) = 120 unique context conditions.
        (Other channel tokens do not affect validity constraints).
        """
        bands = ["sub6", "mmWave_28", "mmWave_60", "THz_140", "THz_220", "THz_300"]
        dopplers = ["static", "pedestrian", "vehicular", "high_speed", "aeronautical"]
        traffics = ["eMBB", "URLLC", "mMTC", "THz_broadband"]
        
        total_valid = 0
        for b in bands:
            for d in dopplers:
                for t in traffics:
                    ctx = [f"band:{b}", f"doppler:{d}", f"traffic:{t}"]
                    total_valid += len(self.get_valid_configs_for_context(ctx))
                    
        return total_valid


# =========================================================================
# Module-level cached validity mask computation
# =========================================================================
@functools.lru_cache(maxsize=4096)
def _get_validity_mask_cached(
    partial_config: tuple, position: int, channel_tokens: tuple,
) -> list[bool]:
    """Cached validity mask — avoids re-computing brute-force completions."""
    partial_list = list(partial_config)
    ctx_list = list(channel_tokens)
    assert len(partial_list) == position

    vocab = POS_VOCABS[position]
    mask = [False] * len(vocab)

    validator = WaveformGrammarValidator.__new__(WaveformGrammarValidator)

    for i, token in enumerate(vocab):
        proposed = partial_list + [token]

        def find_completion(curr_idx, current_tuple):
            if curr_idx == 5:
                c = WaveformConfig(*current_tuple)
                v, _ = validator.is_valid_config(c, ctx_list)
                return v
            for nxt in POS_VOCABS[curr_idx]:
                if find_completion(curr_idx + 1, current_tuple + [nxt]):
                    return True
            return False

        mask[i] = find_completion(position + 1, proposed)

    return mask

