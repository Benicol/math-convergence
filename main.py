from __future__ import annotations

from typing import Union

import numpy as np

import benchmark

ArrayLike = Union[float, np.ndarray]

# --- 1. Paramètres Physiques ---
# du/dt = -k * u^2(t)
K: float = 1.0  # Constante de décharge k
U0: float = 10.0  # Valeur initiale U0
T_LIMIT: tuple[float, float] = (0.0, 2.0)  # Intervalle de temps [0, T]

# --- 2. Définition des fonctions ---


# L'équation différentielle f(t, u)
def f_projet(t: float, u: float) -> float:
    # Sécurité anti-explosion pour les très gros pas h
    u_safe = np.clip(u, -1000, 1000)
    return -K * float(u_safe**2)


# La solution analytique calculée au début : u(t) = U0 / (1 + k*U0*t)
def u_exact_projet(t: ArrayLike) -> ArrayLike:
    return U0 / (1 + K * U0 * t)


if __name__ == "__main__":
    print("=== ÉTAPE 1 : COMPARAISON VISUELLE ET STABILITÉ ===")
    # h=0.4 (Explose), h=0.1 (Tombe à 0), h=0.02 (Stable)
    pas_visuels = [0.4, 0.1, 0.02]

    print("Génération des graphiques temporels...")
    benchmark.run_multi_comparison(
        f_projet,
        u_exact_projet,
        U0,
        T_LIMIT,
        pas_visuels,
        "Analyse temporelle : Instabilité vs Stabilité",
    )

    print("\n=== ÉTAPE 2 : PREUVE DE L'ORDRE DE CONVERGENCE (LOG-LOG) ===")
    print("Génération du graphique Log-Log...")
    benchmark.plot_loglog_convergence(f_projet, u_exact_projet, U0, T_LIMIT)

    print("\n=== ÉTAPE 3 : ANIMATION TEMPS RÉEL (Pas h=0.05) ===")
    # On choisit le pas 0.05 pour voir la précision
    benchmark.create_comparison_animation(f_projet, u_exact_projet, U0, T_LIMIT, 0.05)

    print("\nSimulation terminée.")
