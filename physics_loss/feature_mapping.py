"""Feature mapping for physics-based evaluation.

Maps physics state variable names to column indices in flight data arrays,
handling the variable column orderings across different dataset sources.
"""

import dataclasses
from typing import List, Optional

import numpy as np


@dataclasses.dataclass
class PhysicsFeatureMap:
    """Maps physics state variable names to column indices in the data array.

    The physics model uses a 5- or 6-element state vector:
        [roll(deg), pitch(deg), altitude(ft), heading(deg), airspeed(kts), fuel(gal)]
    Fuel is optional (some datasets lack it).
    """

    roll_idx: int
    pitch_idx: int
    altitude_idx: int
    heading_idx: int
    airspeed_idx: int
    fuel_idx: Optional[int] = None

    def state_indices(self) -> List[int]:
        """Ordered list of column indices for [roll, pitch, alt, hdg, airspeed, fuel]."""
        indices = [
            self.roll_idx,
            self.pitch_idx,
            self.altitude_idx,
            self.heading_idx,
            self.airspeed_idx,
        ]
        if self.fuel_idx is not None:
            indices.append(self.fuel_idx)
        return indices

    @property
    def num_states(self) -> int:
        return 6 if self.fuel_idx is not None else 5

    @property
    def has_fuel(self) -> bool:
        return self.fuel_idx is not None

    def extract_state(self, data: np.ndarray) -> np.ndarray:
        """Extract physics state columns from flight data.

        Args:
            data: Shape (seq_len, feat_dim) or (feat_dim,).

        Returns:
            Shape (seq_len, num_states) or (num_states,).
        """
        indices = self.state_indices()
        if data.ndim == 1:
            return data[indices]
        return data[:, indices]

    def state_names(self) -> List[str]:
        names = ["Roll (deg)", "Pitch (deg)", "Altitude (ft)", "Heading (deg)", "Airspeed (kts)"]
        if self.fuel_idx is not None:
            names.append("Fuel (gal)")
        return names

    def state_short_names(self) -> List[str]:
        names = ["roll", "pitch", "altitude", "heading", "airspeed"]
        if self.fuel_idx is not None:
            names.append("fuel")
        return names


# Column name aliases for auto-detection
_FEATURE_ALIASES = {
    "roll": ["roll", "rollangle", "bank", "bankangle"],
    "pitch": ["pitch", "pitchangle"],
    "altitude": ["altmsl", "altitude", "alt", "altitudemsl"],
    "heading": ["hdg", "heading", "maghdg", "magneticheading"],
    "airspeed": ["ias", "indicatedairspeed", "airspeed"],
    "fuel": ["totalfuel", "fuel", "fuelqty", "fuelquantity"],
}


def build_feature_map_from_csv_header(columns: List[str]) -> PhysicsFeatureMap:
    """Auto-detect physics feature indices from CSV column names.

    Args:
        columns: List of column names (will be lowercased and stripped).

    Returns:
        PhysicsFeatureMap with detected indices.

    Raises:
        ValueError: If required features (roll, pitch, altitude, heading, airspeed)
            cannot be found.
    """
    clean_cols = [c.strip().lower().replace(" ", "").replace("_", "") for c in columns]

    found = {}
    for feature, aliases in _FEATURE_ALIASES.items():
        for alias in aliases:
            clean_alias = alias.replace("_", "")
            for i, col in enumerate(clean_cols):
                if col == clean_alias:
                    found[feature] = i
                    break
            if feature in found:
                break

    required = ["roll", "pitch", "altitude", "heading", "airspeed"]
    missing = [f for f in required if f not in found]
    if missing:
        raise ValueError(
            f"Could not find columns for: {missing}. "
            f"Available columns: {columns}"
        )

    return PhysicsFeatureMap(
        roll_idx=found["roll"],
        pitch_idx=found["pitch"],
        altitude_idx=found["altitude"],
        heading_idx=found["heading"],
        airspeed_idx=found["airspeed"],
        fuel_idx=found.get("fuel"),
    )


# Preset feature maps for known dataset formats
PRESET_FEATURE_MAPS = {
    # 44-column preprocessed dataset
    # Columns: e1egtdivergence(0), e1egt4(1), vplwas(2), ias(3), wndspd(4), e1egt1(5),
    #          amp1(6), altagl(7), hdg(8), e1egt2(9), e1egt3(10), normac(11), fqtyr(12),
    #          wnddr(13), trk(14), hplfd(15), e1rpm(16), fqtyl(17), hplwas(18), stallindex(19),
    #          aoasimple(20), e1fflow(21), densityratio(22), altmsllagdiff(23), tas(24),
    #          magvar(25), e1oilp(26), oat(27), altgps(28), gndspd(29), crs(30), volt1(31),
    #          roll(32), e1oilt(33), pitch(34), trueairspeed(ft/min)(35), latac(36),
    #          vspdg(37), vspdcalculated(38), altb(39), totalfuel(40), altmsl(41), vspd(42), baroa(43)
    "ngafid_44col": PhysicsFeatureMap(
        roll_idx=32,       # roll
        pitch_idx=34,      # pitch
        altitude_idx=41,   # altmsl
        heading_idx=8,     # hdg
        airspeed_idx=3,    # ias
        fuel_idx=40,       # totalfuel
    ),
    # 40-column INPUT_COLS ordering (from benchmarks/conv_mhsa/flight.py)
    "input_cols_40": PhysicsFeatureMap(
        roll_idx=33,
        pitch_idx=18,
        altitude_idx=6,
        heading_idx=38,
        airspeed_idx=12,
        fuel_idx=21,
    ),
    # Toy dataset (6 numeric columns: altmsl, ias, vspd, pitch, roll, hdg)
    "toy_ngafid": PhysicsFeatureMap(
        roll_idx=4,
        pitch_idx=3,
        altitude_idx=0,
        heading_idx=5,
        airspeed_idx=1,
        fuel_idx=None,
    ),
}
