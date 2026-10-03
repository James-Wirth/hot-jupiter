from __future__ import annotations

import numpy as np
import pandas as pd

from hj.state import StopCode


def outcome_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    required = {"e", "a", "stop_code"}
    if not required <= set(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    eccentricity = pd.to_numeric(df.e, errors="coerce")
    semimajor_axis = pd.to_numeric(df.a, errors="coerce")
    finite = np.isfinite(eccentricity) & np.isfinite(semimajor_axis)
    ionised = df.stop_code == StopCode.ION
    valid_code = df.stop_code.isin([code.value for code in StopCode])
    flags = pd.DataFrame(
        {
            "negative_eccentricity": eccentricity < 0,
            "nonfinite_orbit": ~finite,
            "nonpositive_bound_semimajor_axis": finite
            & ~ionised
            & (semimajor_axis <= 0),
            "unbound_eccentricity_in_bound_outcome": finite
            & ~ionised
            & (eccentricity >= 1),
            "bound_orbit_in_ionised_outcome": finite
            & ionised
            & (semimajor_axis > 0)
            & (eccentricity >= 0)
            & (eccentricity < 1),
            "invalid_stop_code": ~valid_code,
        },
        index=df.index,
    )
    flags["flagged_system"] = flags.any(axis=1)
    rows = []
    masks = {code.name: df.stop_code == code.value for code in StopCode}
    masks["UNCLASSIFIED"] = ~valid_code
    masks["ALL"] = pd.Series(True, index=df.index)
    for label, mask in masks.items():
        selected = flags.loc[mask]
        total = len(selected)
        rows.append(
            {
                "outcome": label,
                "total": total,
                **{name: int(selected[name].sum()) for name in flags},
                "flagged_fraction": selected.flagged_system.mean() if total else np.nan,
            }
        )
    return pd.DataFrame(rows).set_index("outcome")
