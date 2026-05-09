from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["State", "STOP_CODE_UNSET"]


STOP_CODE_UNSET: np.int8 = np.int8(-1)


@dataclass(slots=True)
class State:
    e: np.ndarray
    a: np.ndarray
    e_init: np.ndarray
    a_init: np.ndarray
    m1: np.ndarray
    m2: np.ndarray
    lagrange: np.ndarray
    t: np.ndarray
    stop_code: np.ndarray
    stop_time: np.ndarray

    @classmethod
    def empty(cls, n: int) -> State:
        return cls(
            e=np.empty(n, np.float64),
            a=np.empty(n, np.float64),
            e_init=np.empty(n, np.float64),
            a_init=np.empty(n, np.float64),
            m1=np.empty(n, np.float64),
            m2=np.empty(n, np.float64),
            lagrange=np.empty(n, np.float64),
            t=np.zeros(n, np.float64),
            stop_code=np.full(n, STOP_CODE_UNSET, np.int8),
            stop_time=np.zeros(n, np.float64),
        )

    def __len__(self) -> int:
        return self.e.shape[0]

    def active_mask(self) -> np.ndarray:
        return self.stop_code == STOP_CODE_UNSET

    def any_active(self) -> bool:
        return bool((self.stop_code == STOP_CODE_UNSET).any())
