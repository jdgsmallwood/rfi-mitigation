"""Regenerate the four revised-paper figures from the archived result tables."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import OUT as P

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "pdf.fonttype": 42,
    }
)


def save(fig, name):
    fig.savefig(P / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


s = pd.read_csv(P / "satellite_brightness.csv")
d = pd.read_csv(P / "satellite_brightness_timeseries.csv")
e = pd.read_csv(P / "elevation.csv")
reference = s.mean_target.iloc[0]
fig, axs = plt.subplots(
    1, 4, figsize=(7.1, 2.25), sharex=True, sharey=True, layout="constrained"
)
for j, ax in enumerate(axs):
    w = j + 1
    q = d[d.window == w]
    el = e[e.window == w]
    db = np.full(len(q), np.nan)
    positive = q.target.to_numpy() > 0
    db[positive] = 10 * np.log10(q.target.to_numpy()[positive] / reference)
    ax.plot(q.seconds_from_window_start, db, color="C0", lw=1)
    right = ax.twinx()
    right.plot(el.seconds_from_window_start, el.elevation_deg, color="C3", ls=":", lw=1)
    right.set_ylim(-5, 25)
    right.set_yticks([0, 10, 20])
    right.tick_params(axis="y", colors="C3", labelright=j == 3)
    if j == 3:
        right.set_ylabel("Satellite elevation (deg)", color="C3")
    ax.set(
        title=f"Window {w}",
        xlim=(0, 120),
        ylim=(-45, 5),
        xticks=[0, 60, 120],
        xlabel="Time in window (s)",
    )
    ax.grid(alpha=0.15)
    below = el[el.elevation_deg <= 0]
    if len(below):
        ax.axvspan(below.seconds_from_window_start.iloc[0], 120, color=".92", zorder=-1)
axs[0].set_ylabel("Raw track brightness\n(dB relative to Window 1 mean)")
save(fig, "satellite_track_brightness_elevation")
# Four-window roll-offs: best fixed rank versus a separate rank per 60 s half.
df = pd.read_csv(P / "channels.csv")
ss = pd.read_csv(P / "summary.csv")
adaptive = pd.read_csv(P / "adaptive_ranks.csv").set_index("window")
freq = 137.5
fig, axs = plt.subplots(
    2, 2, figsize=(7.1, 4.6), sharex=True, sharey=True, layout="constrained"
)
for w, ax in enumerate(axs.flat, 1):
    q = df[df.window == w]
    st = ss[ss.window == w].iloc[0]
    ax.plot(
        q.frequency_MHz,
        q.correlation_with_clean,
        "o-",
        color=".5",
        ms=2.5,
        lw=0.8,
        label="Unmitigated, Sun-shrunk",
    )
    ax.plot(
        q.frequency_MHz,
        np.polyval([st.slope, st.intercept], q.frequency_MHz),
        "--",
        color="black",
        lw=0.8,
        label="Trend excluding affected channel",
    )
    ax.scatter(
        [freq],
        [st.best_protected],
        color="C0",
        s=22,
        zorder=4,
        label="Best protected–clean (fixed rank)",
    )
    ax.scatter(
        [freq],
        [st.unprotected_rank2],
        marker="s",
        color="C2",
        s=18,
        zorder=4,
        label="Unprotected, rank 2",
    )
    result = adaptive.loc[w]
    ax.scatter(
        [freq], [result.combined_correlation], marker="*",
        s=85, zorder=5, facecolors="none", edgecolors="C4", linewidths=1.2,
        label="Best protected–clean (adaptive rank)",
    )
    ax.set(
        title=(f"Window {w}: expected {st.expected:.3f}\n"
               f"Fixed rank {int(st.best_rank)}; adaptive "
               f"{int(result.first_half_rank)} → {int(result.second_half_rank)}"),
        ylim=(-0.05, 1.1),
    )
    ax.grid(alpha=0.15)
for ax in axs[-1]:
    ax.set_xlabel("Frequency (MHz)")
for ax in axs[:, 0]:
    ax.set_ylabel("Correlation with reference channel")
fig.legend(*axs[0, 0].get_legend_handles_labels(),
           loc="outside lower center", ncol=2, frameon=False, fontsize=7)
save(fig, "channel_correlations_four_windows")
# Protected-only half-window comparison.
h = pd.read_csv(P / "correlations.csv").query("window == 3")
fig, ax = plt.subplots(figsize=(3.4, 2.6), layout="constrained")
for half, label, color in [(1, "First 60 s", "C0"), (2, "Second 60 s", "C1")]:
    q = h[h.half == half]
    ax.plot(q["rank"], q.protected_clean, "o-", ms=3, lw=1, label=label, color=color)
ax.set(
    xlabel="Number of nulled modes",
    ylabel="Correlation with clean channel",
    xticks=range(0, 11, 2),
    ylim=(0, 0.65),
)
ax.grid(alpha=0.2)
ax.legend(frameon=False)
save(fig, "window3_protected_halves")

df = pd.read_csv(P / "rank_sweeps.csv").sort_values(["window", "fixed_rank"])
s = pd.read_csv(P / "satellite_brightness.csv")
assert len(df) == 44
plt.rcParams.update({"font.family": "serif", "font.size": 10})
fig, axs = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)
curves = [
    ("protected_clean", "Protected–clean", "o", "C0", "-"),
    ("protected_template", "Protected–template", "o", "C1", "-"),
    ("unprotected_clean", "Unprotected–clean", "s", "C2", "--"),
    ("rotated_50_protected_template", "50° protected–template", "^", "C5", "-."),
]
for w, ax in enumerate(axs.flat, 1):
    d = df[df.window == w]
    for col, label, marker, color, ls in curves:
        ax.plot(
            d.fixed_rank,
            d[col],
            label=label,
            marker=marker,
            color=color,
            ls=ls,
            ms=3,
            lw=1.2,
        )
    rel = s.loc[s.window == w, "relative_to_window1_db"].iloc[0]
    ax.set_title(f"Window {w} · track brightness {rel:+.1f} dB")
    ax.set(ylim=(-1, 1), xticks=range(11))
    ax.axhline(0, color=".7", lw=0.7, ls=":")
    ax.grid(alpha=0.15)
for ax in axs[-1]:
    ax.set_xlabel("Number of nulled modes")
for ax in axs[:, 0]:
    ax.set_ylabel("Image Pearson correlation")
fig.legend(
    *axs[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False
)
fig.tight_layout(rect=(0, 0.09, 1, 1))
save(fig, "four_window_rank_sweep")
