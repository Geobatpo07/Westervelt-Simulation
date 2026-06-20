"""Génère la figure des conditions initiales gaussiennes pour la frame
'Validation par raffinement'. Reproduit §5.1.3 du mémoire :
  u0 : impulsion gaussienne centrée
  u1 : gaussienne moins sa moyenne (pour assurer F̂^0(0) = 0).
"""
import numpy as np
import matplotlib.pyplot as plt
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

BEAMER_RC = {
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "lines.linewidth": 2.2, "figure.constrained_layout.use": True,
    "mathtext.fontset": "cm",
}

L = 0.2
A1, A2 = 1.2e8, 1e11
x0, x1 = 0.1, 0.1
sig0, sig1 = 0.015, 0.02

x = np.linspace(0, L, 1001)
u0 = A1 * np.exp(-((x - x0) ** 2) / (2 * sig0 ** 2))
gauss = A2 * np.exp(-((x - x1) ** 2) / (2 * sig1 ** 2))
gbar = np.trapezoid(gauss, x) / L
u1 = gauss - gbar

with plt.rc_context(BEAMER_RC):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))

    # u0 : profil en pression initiale
    ax1.plot(x, u0 / 1e6, color="#1f77b4")
    ax1.fill_between(x, 0, u0 / 1e6, color="#1f77b4", alpha=0.12)
    ax1.set_xlabel(r"$x$ [m]")
    ax1.set_ylabel(r"$u_0(x)$ [MPa]")
    ax1.set_title(r"Surpression initiale $u_0$")
    ax1.grid(True, alpha=0.3)
    ax1.annotate(rf"$A_1 = 120$ MPa", xy=(x0, A1 / 1e6),
                 xytext=(0, -22), textcoords="offset points",
                 ha="center", fontsize=12, color="#1f77b4")

    # u1 : vitesse initiale recentrée
    ax2.plot(x, u1 / 1e9, color="#d62728", label=r"$u_1(x)$")
    ax2.axhline(0.0, color="0.5", lw=1, ls=":")
    ax2.fill_between(x, 0, u1 / 1e9, where=(u1 > 0), color="#d62728", alpha=0.15)
    ax2.fill_between(x, 0, u1 / 1e9, where=(u1 <= 0), color="#2ca02c", alpha=0.15)
    ax2.set_xlabel(r"$x$ [m]")
    ax2.set_ylabel(r"$u_1(x)$ [GPa/s]")
    ax2.set_title(r"Vitesse initiale $u_1 = \mathrm{gaussienne} - \bar{g}$")
    ax2.grid(True, alpha=0.3)
    ax2.text(0.04, 0.95,
             r"$\int_0^L u_1\,\mathrm{d}x = 0$" + "\n"
             + r"$\Rightarrow \widehat{F^0}(0) = 0$",
             transform=ax2.transAxes, va="top", fontsize=12,
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.6"))

OUT = pathlib.Path(OUTPUTS_DIR / "figures/initial_conditions_gaussiennes.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
print(OUT)
