"""Outils numériques pour résoudre une équation différentielle par la méthode d'Euler explicite."""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np


def solve_euler(
    f: Callable[[float, float], float],
    u0: float,
    t_span: Tuple[float, float],
    h: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcule l'approximation numérique de la solution de du/dt = f(t, u) via Euler explicite.

    Args:
        f: Fonction donnant la pente instantanée f(t, u).
        u0: Condition initiale u(t0).
        t_span: Intervalle temporel (t0, tf) couvert par l'intégration.
        h: Pas de temps constant strictement positif.

    Returns:
        Tuple des abscisses et ordonnées numériques (t_values, u_values).
    """
    t_start, t_end = t_span
    if h <= 0:
        raise ValueError(
            "Le pas de temps h doit être strictement positif pour Euler explicite."
        )

    t_values: np.ndarray = np.arange(t_start, t_end + h, h)
    u_values: np.ndarray = np.zeros(len(t_values), dtype=float)
    u_values[0] = u0

    # On avance la solution en utilisant la pente locale fournie par f sur chaque sous-intervalle.
    for i, t_current in enumerate(t_values[:-1]):
        slope = f(float(t_current), float(u_values[i]))
        u_values[i + 1] = u_values[i] + h * slope

    return t_values, u_values
