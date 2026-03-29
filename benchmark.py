from __future__ import annotations

from typing import Callable, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from euler_solver import solve_euler
from heun_solver import solve_heun

ArrayLike = Union[np.ndarray, float]


def run_multi_comparison(
    f: Callable[[float, float], float],
    u_exact_func: Callable[[ArrayLike], ArrayLike],
    u0: float,
    t_span: Tuple[float, float],
    h_list: Sequence[float],
    title: str = "",
) -> None:
    """Compare Euler and Heun solutions for multiple time steps."""
    fig, axes = plt.subplots(1, len(h_list), figsize=(18, 6))
    fig.suptitle(title, fontsize=16)
    t_ref = np.linspace(t_span[0], t_span[1], 1000)
    u_ref = u_exact_func(t_ref)

    for i, h in enumerate(h_list):
        ax = axes[i]
        t_e, u_e = solve_euler(f, u0, t_span, h)
        t_h, u_h = solve_heun(f, u0, t_span, h)

        val_exacte = u_exact_func(t_span[1])
        err_e = abs(u_e[-1] - val_exacte)
        err_h = abs(u_h[-1] - val_exacte)

        gain = err_e / err_h if err_h > 1e-15 else float("inf")

        if err_e > 100:
            txt_e = "EXPLOSION"
            txt_gain = "Heun est le seul survivant"
        else:
            txt_e = f"{err_e:.5f}"
            txt_gain = f"Heun est {gain:.1f}x plus précis"

        txt_h = f"{err_h:.5f}"

        ax.plot(t_ref, u_ref, "b-", lw=4, alpha=0.2, label="VÉRITÉ")
        ax.plot(t_e, u_e, "r--o", markersize=4, label=f"EULER (Err: {txt_e})")
        ax.plot(t_h, u_h, "g-s", markersize=4, label=f"HEUN (Err: {txt_h})")

        ax.set_title(f"Pas h = {h}\n{txt_gain}")
        ax.set_ylim(-2, u0 + 1)
        ax.legend(fontsize=8)
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def plot_loglog_convergence(
    f: Callable[[float, float], float],
    u_exact_func: Callable[[ArrayLike], ArrayLike],
    u0: float,
    t_span: Tuple[float, float],
) -> None:
    """Trace les erreurs globales en fonction du pas h."""
    h_values: list[float] = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    errors_euler: list[float] = []
    errors_heun: list[float] = []

    t_final = t_span[1]
    val_exacte = u_exact_func(t_final)

    for h in h_values:
        _, u_e = solve_euler(f, u0, t_span, h)
        _, u_h = solve_heun(f, u0, t_span, h)

        errors_euler.append(abs(u_e[-1] - val_exacte))
        errors_heun.append(abs(u_h[-1] - val_exacte))

    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors_euler, "ro-", label="Euler (Ordre 1)")
    plt.loglog(h_values, errors_heun, "gs-", label="Heun (Ordre 2)")

    plt.title("Erreur globale finale en fonction du pas h (Log-Log)")
    plt.xlabel("Pas de temps h (log)")
    plt.ylabel("Erreur absolue |u_num - u_exact| (log)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.show()


def create_comparison_animation(
    f: Callable[[float, float], float],
    u_exact_func: Callable[[ArrayLike], ArrayLike],
    u0: float,
    t_span: Tuple[float, float],
    h: float,
) -> None:
    """Génère une animation temps réel pour comparer Euler et Heun."""
    fig, ax = plt.subplots(figsize=(10, 6))

    t_ref = np.linspace(t_span[0], t_span[1], 500)
    u_ref = u_exact_func(t_ref)

    t_e, u_e = solve_euler(f, u0, t_span, h)
    t_h, u_h = solve_heun(f, u0, t_span, h)

    ax.plot(t_ref, u_ref, "b-", lw=2, alpha=0.3, label="VÉRITÉ (Exacte)")
    (line_euler,) = ax.plot([], [], "r--o", markersize=4, label="EULER")
    (line_heun,) = ax.plot([], [], "g-s", markersize=4, label="HEUN")

    ax.set_xlim(t_span[0], t_span[1])
    ax.set_ylim(-1, u0 + 1)
    ax.set_title(f"Animation de la résolution (h = {h})")
    ax.legend()
    ax.grid(True)

    def update(frame: int) -> tuple[Line2D, Line2D]:
        line_euler.set_data(t_e[:frame], u_e[:frame])
        line_heun.set_data(t_h[:frame], u_h[:frame])
        return line_euler, line_heun

    ani: FuncAnimation = FuncAnimation(
        fig, update, frames=len(t_e) + 1, blit=True, interval=100, repeat=False
    )

    plt.show()
    ani.save("simularion.mp4")
