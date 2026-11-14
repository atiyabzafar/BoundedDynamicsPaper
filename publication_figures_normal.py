"""Recreate publication figures from the normal-alpha simulation dataset."""

from ast import literal_eval
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_FILENAME = (
    "output/publication/simulation_results_full_withallspecies_N100_T2000_c1_r0.01-1_"
    "threshold0.0001_reps50_p0-1_pcount101_betascale_50.0_"
    "alphascale_1.0_normal.csv"
)
INSET_DATA_FILENAME = (
    "output/publication/simulation_results_full_withallspecies_N100_T2000_c1_r0.01-1_"
    "threshold0.0001_reps50_p0-0.1_pcount101_betascale_50.0_"
    "alphascale_1.0_normal.csv"
)
DATA_PATHS = (
    PROJECT_ROOT / DATA_FILENAME,
    PROJECT_ROOT / "output" / "publication" / DATA_FILENAME,
)
INSET_DATA_PATHS = (
    PROJECT_ROOT / INSET_DATA_FILENAME,
    PROJECT_ROOT / "output" / "publication" / INSET_DATA_FILENAME,
)
FIGURES_DIR = PROJECT_ROOT / "figures/normal"

NOTATION_MAP = {"GLV": "SRD", "Bounded": "RRD"}
COLOR_MAP = {"SRD": "#1f4e79", "RRD": "#d65f5f"}
MARKER_MAP = {"SRD": "s", "RRD": "o"}


def load_data() -> pd.DataFrame:
    """Load the normal-alpha simulation output."""
    return load_dataset(DATA_PATHS)


def load_inset_data() -> pd.DataFrame:
    """Load the low-connectance normal-alpha output for the inset."""
    return load_dataset(INSET_DATA_PATHS)


def load_dataset(paths: tuple[Path, ...]) -> pd.DataFrame:
    """Load a dataset from the first available configured path."""
    for path in paths:
        if path.exists():
            return pd.read_csv(path)
    locations = ", ".join(str(path) for path in paths)
    raise FileNotFoundError(f"Normal dataset not found. Checked: {locations}")


def aggregate_survivors(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean survivors and standard error by model and connectance."""
    return (
        df.groupby(["model", "p_link"])
        .agg(
            mean_survivors=("survivors", "mean"),
            sem_survivors=(
                "survivors",
                lambda values: values.sem(ddof=1) if len(values) > 1 else 0.0,
            ),
        )
        .reset_index()
    )


def plot_diversity_vs_connectance(df: pd.DataFrame) -> None:
    """Recreate the diversity/connectance figure with a low-connectance inset."""
    summary = aggregate_survivors(df)
    df_small=load_inset_data()
    df_small = df_small[df_small["p_link"].isin(
    df_small["p_link"].drop_duplicates().sort_values().iloc[::2]
    )]
    #sweep_small = aggregate_survivors(df_small)
    inset_summary = aggregate_survivors(df_small)
    fig, ax_main = plt.subplots(figsize=(7.09, 4.5))
    ax_inset = ax_main.inset_axes([0.25, 0.35, 0.50, 0.4])

    for model_name in summary["model"].unique():
        pub_name = NOTATION_MAP.get(model_name, model_name)
        group = summary[summary["model"] == model_name]
        style = {
            "marker": MARKER_MAP[pub_name],
            "color": COLOR_MAP[pub_name],
            "linestyle": "-",
            "capsize": 3,
            "lw": 2.0,
            "elinewidth": 2.0,
            "capthick": 2.0,
            #"markersize": 5,
            "label": pub_name,
        }
        ax_main.errorbar(
            group["p_link"],
            group["mean_survivors"],
            yerr=group["sem_survivors"],
            **{**style,"markersize":5},
        )
        inset_group = inset_summary[inset_summary["model"] == model_name]
        ax_inset.errorbar(
            inset_group["p_link"],
            inset_group["mean_survivors"],
            yerr=inset_group["sem_survivors"],
            **{**style,"lw":1, "capthick":1,"elinewidth":1,"capsize": 2, "markersize": 3, "label": None},
        )
    for spine in ax_main.spines.values():
        spine.set_linewidth(2)
    ax_main.set_xlabel(r"Connectance, $p$", fontsize=12)
    ax_main.set_ylabel(
        r"Ensemble-average diversity, $\langle S_{\text{surv}}(p) \rangle$", fontsize=12
    )
    ax_main.grid(alpha=0.3)
    ax_main.legend(loc="lower right", frameon=False, fontsize=12)
    ax_main.tick_params(axis="both", labelsize=12)
    for spine in ax_inset.spines.values():
        spine.set_linewidth(1.5)
    ax_inset.set_xlim(-0.005, 0.105)
    ax_inset.set_ylim(-2, 102)
    ax_inset.grid(alpha=0.3)
    ax_inset.tick_params(axis="both", which="major", labelsize=12)
    ax_inset.set_title(r"Zoom: $p \in [0, 0.1]$", fontsize=11)
    ax_main.indicate_inset_zoom(ax_inset, edgecolor="gray", alpha=0.4, linewidth=2.0)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "Diversity_vs_Connectance_Combined_normal.png",dpi=600 ,bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "Diversity_vs_Connectance_Combined_normal.pdf",dpi=200 ,bbox_inches="tight")
    plt.close(fig)


def plot_mean_and_std_vs_connectance(df: pd.DataFrame) -> None:
    """Recreate the mean and within-network growth-rate spread figure."""
    work = df.copy()
    summary_mean = (
        work.groupby(["model", "p_link"])
        .agg(
            avg_mean_r=("mean_r_survivors", "mean"),
            std_mean_r=("mean_r_survivors", "std"),
        )
        .reset_index()
    )

    work["r_list"] = work["r_list"].apply(literal_eval)
    work["survivors_mask"] = work["survivors_mask"].apply(literal_eval)
    work["std_r_survivors"] = work.apply(
        lambda row: np.std(
            np.asarray(row["r_list"])[np.asarray(row["survivors_mask"])],
            ddof=1,
        )
        if np.sum(row["survivors_mask"]) > 1
        else 0.0,
        axis=1,
    )
    summary_std = (
        work.groupby(["model", "p_link"])["std_r_survivors"]
        .mean()
        .reset_index(name="avg_std_r")
    )

    fig, (ax_mean, ax_std) = plt.subplots(
        2,
        1,
        figsize=(7.09, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 0.9]},
    )

    for model_name, group in summary_mean.groupby("model"):
        pub_name = NOTATION_MAP.get(model_name, model_name)
        ax_mean.errorbar(
            group["p_link"],
            group["avg_mean_r"],
            yerr=group["std_mean_r"],
            marker=MARKER_MAP[pub_name],
            color=COLOR_MAP[pub_name],
            linestyle="-",
            capsize=3,
            lw=2.0,
            elinewidth=1.0,
            capthick=2.0,
            markersize=4,
        )

    population_mean = (0.01 + 1.0) / 2.0
    population_std = (1.0 - 0.01) / np.sqrt(12)
    ax_mean.axhline(population_mean, color="black", linestyle="--", linewidth=2.0, alpha=0.6)
    ax_mean.set_ylabel(r"Mean growth rate, $\langle \bar{r} \rangle$", labelpad=8)
    ax_mean.grid(alpha=0.3)
    ax_mean.text(0.9, 0.95, "(A)", transform=ax_mean.transAxes, ha="left", va="top", fontweight="bold")

    for model_name, group in summary_std.groupby("model"):
        pub_name = NOTATION_MAP.get(model_name, model_name)
        ax_std.plot(
            group["p_link"],
            group["avg_std_r"],
            marker=MARKER_MAP[pub_name],
            color=COLOR_MAP[pub_name],
            linestyle="-",
            lw=2.0,
            markersize=4,
        )

    ax_std.axhline(population_std, color="black", linestyle="--", linewidth=2.0, alpha=0.6)
    ax_std.set_xlabel(r"Connectance, $p$")
    ax_std.set_ylabel(r"Within-network std., $\langle \sigma_r \rangle$", labelpad=8)
    ax_std.grid(alpha=0.3)
    ax_std.text(0.9, 0.85, "(B)", transform=ax_std.transAxes, ha="left", va="top", fontweight="bold")

    model_handles = [
        Line2D([], [], color=COLOR_MAP[name], marker=MARKER_MAP[name], lw=2, markersize=4, label=name)
        for name in ("SRD", "RRD")
    ]
    reference_handle = Line2D(
        [], [], color="black", linestyle="--", lw=2, label=r"Basal reference, $\mathcal{U}(0.01, 1)$"
    )
    ax_std.legend(
        handles=model_handles + [reference_handle],
        loc="lower right",
        frameon=False,
        fontsize=9,
        handlelength=2.0,
    )

    fig.subplots_adjust(left=0.25, right=0.98, bottom=0.13, top=0.97, hspace=0.25)
    fig.savefig(FIGURES_DIR / "Combined_Mean_and_STD_vs_p_normal.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_conditional_survival(df: pd.DataFrame) -> None:
    """Recreate conditional survival probability panels."""
    target_p_links = [0.01, 0.02, 0.03, 0.05]
    available_ps = df["p_link"].unique()
    closest_ps = [available_ps[np.argmin(np.abs(available_ps - target))] for target in target_p_links]
    filtered = df[df["p_link"].isin(closest_ps)].copy()
    filtered["r_list"] = filtered["r_list"].apply(literal_eval)
    filtered["survivors_mask"] = filtered["survivors_mask"].apply(literal_eval)

    fig, axes = plt.subplots(2, 2, figsize=(7.09, 6.5), sharex=True, sharey=True)
    axes = axes.flatten()
    bins = np.linspace(0, 1.0, 11)
    centers = (bins[:-1] + bins[1:]) / 2

    for index, (target_p, axis) in enumerate(zip(closest_ps, axes)):
        subset = filtered[filtered["p_link"] == target_p]
        for model in ("GLV", "Bounded"):
            model_data = subset[subset["model"] == model]
            all_r = np.concatenate(model_data["r_list"].values)
            all_survivors = np.concatenate(model_data["survivors_mask"].values)
            total, _ = np.histogram(all_r, bins=bins)
            survived, _ = np.histogram(all_r[all_survivors], bins=bins)
            probability = np.divide(
                survived,
                total,
                out=np.zeros_like(total, dtype=float),
                where=total != 0,
            )
            pub_name = NOTATION_MAP[model]
            axis.plot(
                centers,
                probability,
                marker=MARKER_MAP[pub_name],
                color=COLOR_MAP[pub_name],
                linestyle="-",
                linewidth=2.0,
                markersize=6,
                label=pub_name if index == 0 else "",
            )

        axis.set_title(f"({chr(65 + index)}) $p = {target_p:.3f}$", loc="left", fontsize=12)
        axis.grid(alpha=0.3)
        axis.set_xlim(0.0, 1.05)
        axis.set_ylim(-0.05, 1.05)

    fig.supxlabel(r"Intrinsic growth rate, $r$")
    fig.supylabel(r"Conditional survival probability, $P(\text{survive} \mid r)$")
    axes[0].legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "Conditional_Survival_Prob_2x2_normal.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    plot_diversity_vs_connectance(data)
    plot_mean_and_std_vs_connectance(data)
    plot_conditional_survival(data)
    print(f"Created normal-dataset figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
