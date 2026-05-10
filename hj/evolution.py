from __future__ import annotations

import logging
import math

import numpy as np
from joblib import Parallel, delayed

from hj import core, sampling
from hj.clusters import Cluster, Plummer
from hj.state import STOP_UNSET, State, StopCode

__all__ = ["StopCode", "run_simulation", "sample_initial_conditions"]

logger = logging.getLogger(__name__)

_NBODY_SERIAL_CUTOFF: int = 32
_MAX_STEPS: int = 1_000_000


def sample_initial_conditions(
    n: int, cluster: Cluster, rng: np.random.Generator
) -> State:
    state = State.empty(n)
    state.lagrange[:] = cluster.sample_lagrange(n_samples=n, t=0.0, rng=rng)
    state.e_init[:] = sampling.sample_e_init(n, rng)
    state.a_init[:] = sampling.sample_a_init(n, rng)
    state.m1[:] = sampling.sample_m1(n, rng)
    state.m2[:] = sampling.sample_m2(n, rng)
    state.e[:] = state.e_init
    state.a[:] = state.a_init
    return state


def _nbody_one(args: tuple[float, ...]) -> tuple[float, float]:
    v_inf, b, lan, inc, aop, e, a, m1, m2, m3, mean_anom = args
    return core.nbody_encounter_de(v_inf, b, lan, inc, aop, e, a, m1, m2, m3, mean_anom)


def _batch_nbody(
    parallel: Parallel,
    idx: np.ndarray,
    state: State,
    enc_v: np.ndarray,
    enc_b: np.ndarray,
    enc_lan: np.ndarray,
    enc_inc: np.ndarray,
    enc_aop: np.ndarray,
    enc_m3: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:

    m = idx.size
    mean_anoms = rng.uniform(-math.pi, math.pi, size=m)
    args_list = [
        (
            float(enc_v[i]),
            float(enc_b[i]),
            float(enc_lan[i]),
            float(enc_inc[i]),
            float(enc_aop[i]),
            float(state.e[i]),
            float(state.a[i]),
            float(state.m1[i]),
            float(state.m2[i]),
            float(enc_m3[i]),
            float(mean_anoms[k]),
        )
        for k, i in enumerate(idx)
    ]

    if m < _NBODY_SERIAL_CUTOFF:
        results = [_nbody_one(args) for args in args_list]
    else:
        results = parallel(delayed(_nbody_one)(args) for args in args_list)

    de = np.fromiter((r[0] for r in results), dtype=np.float64, count=m)
    da = np.fromiter((r[1] for r in results), dtype=np.float64, count=m)
    return de, da


def run_simulation(
    state: State,
    cluster: Plummer,
    time_total: float,
    rng: np.random.Generator,
    hybrid_switch: bool = True,
    n_jobs: int = -1,
) -> None:

    n = len(state)
    R_td, R_hj, R_wj = core.critical_radii(state.m1, state.m2)
    plummer_static = core.plummer_kernel_params(cluster)

    needs_nbody = np.zeros(n, dtype=np.bool_)
    enc_v = np.empty(n, dtype=np.float64)
    enc_b = np.empty(n, dtype=np.float64)
    enc_lan = np.empty(n, dtype=np.float64)
    enc_inc = np.empty(n, dtype=np.float64)
    enc_aop = np.empty(n, dtype=np.float64)
    enc_m3 = np.empty(n, dtype=np.float64)

    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
        for _step_idx in range(_MAX_STEPS):
            if not (state.stop_code == STOP_UNSET).any():
                break

            needs_nbody.fill(False)

            u_wt = rng.random(n)
            u_b = rng.random(n)
            u_lan = rng.random(n)
            u_aop = rng.random(n)
            u_inc = rng.random(n)
            u_m3 = rng.random(n)
            n_xyz = rng.standard_normal((3, n))

            core.step(
                state.e,
                state.a,
                state.m1,
                state.m2,
                state.lagrange,
                state.t,
                state.stop_code,
                state.stop_time,
                plummer_static,
                R_td,
                R_hj,
                R_wj,
                time_total,
                hybrid_switch,
                u_wt,
                u_b,
                u_lan,
                u_aop,
                u_inc,
                u_m3,
                n_xyz[0],
                n_xyz[1],
                n_xyz[2],
                needs_nbody,
                enc_v,
                enc_b,
                enc_lan,
                enc_inc,
                enc_aop,
                enc_m3,
            )

            if hybrid_switch:
                idx = np.flatnonzero(needs_nbody)
                if idx.size:
                    de, da = _batch_nbody(
                        parallel,
                        idx,
                        state,
                        enc_v,
                        enc_b,
                        enc_lan,
                        enc_inc,
                        enc_aop,
                        enc_m3,
                        rng,
                    )
                    state.e[idx] += de
                    state.a[idx] += da
                    core.recheck_stop(
                        idx,
                        state.e,
                        state.a,
                        state.t,
                        state.stop_code,
                        state.stop_time,
                        R_td,
                        R_hj,
                        R_wj,
                        time_total,
                    )
        else:
            survivors = state.stop_code == STOP_UNSET
            if survivors.any():
                logger.warning(
                    "Reached _MAX_STEPS=%d with %d active systems; forcing NM.",
                    _MAX_STEPS,
                    int(survivors.sum()),
                )
                state.stop_code[survivors] = StopCode.NM
                state.stop_time[survivors] = state.t[survivors]
