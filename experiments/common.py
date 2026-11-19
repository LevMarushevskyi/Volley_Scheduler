"""Shared helpers for experiment scripts: config, paths, plot style."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

from vsched.cost import load_weights  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# Validated categorical order (dataviz reference palette, light mode) + neutrals.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BASELINE = "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def config() -> dict:
    with open(ROOT / "configs" / "experiments.yaml") as f:
        return yaml.safe_load(f)


def weights(overrides: dict | None = None) -> dict[str, float]:
    w = load_weights(ROOT / config()["weights"])
    w.update({k: float(v) for k, v in (overrides or {}).items()})
    return w


def has_templates() -> bool:
    return (ROOT / "data" / "templates.json").exists()


def style(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def save(fig, name: str):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"wrote figures/{name}")


def round_cost(c: dict) -> dict:
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items()}
