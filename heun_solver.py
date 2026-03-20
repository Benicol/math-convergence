from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def solve_heun(
    f: Callable[[float, float], float],
    u0: float,
    t_span: Tuple[float, float],
    h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Résout du/dt = f(t, u) par la méthode de Heun (Runge-Kutta explicite d'ordre 2).
    """
    t_values: np.ndarray = np.arange(t_span[0], t_span[1] + h, h)
    u_values: np.ndarray = np.zeros(len(t_values))
    u_values[0] = u0

    for i in range(len(t_values) - 1):
        t_n = float(t_values[i])
        u_n = float(u_values[i])

        pente_depart = f(t_n, u_n)

        u_predit = u_n + h * pente_depart

        pente_arrivee = f(t_n + h, u_predit)

        u_values[i + 1] = u_n + (h / 2) * (pente_depart + pente_arrivee)

    return t_values, u_values
