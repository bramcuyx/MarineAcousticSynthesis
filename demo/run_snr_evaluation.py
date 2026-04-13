"""Run SNR improvement evaluation for all metadata files and save a CSV report."""

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from evaluation.snr import evaluate_snr_improvement
from uw_sim.audio_simulator import MetadataManager


def main() -> None:
    """Automatically evaluate SNR improvement for all metadata files in the output directory and saves a CSV report with the results. It also generates a plot of non-masked SNR improvement vs target SNR."""
    project_root = pathlib.Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "config.yaml").read_text())

    output_dir = pathlib.Path(config["paths"]["output"])
    datasets_dir = pathlib.Path(config["paths"].get("datasets", output_dir))
    datasets_dir.mkdir(parents=True, exist_ok=True)

    metadata_files = sorted(output_dir.glob("metadata_*.json"))
    if not metadata_files:
        print(f"No metadata files found in {output_dir}")
        return

    rows = []
    for metadata_file in metadata_files:
        print(f"Evaluating SNR improvement for {metadata_file.name}")
        metadata = MetadataManager()
        try:
            metadata.load_metadata(metadata_file)
            (
                snr_after,
                snr_before,
                snr_after_nonmasked,
                snr_before_nonmasked,
            ) = evaluate_snr_improvement(
                metadata,
                masked=True,
                masked_noise=True,
                mode="masked",
                denoised_folder=pathlib.Path(config["paths"]["denoised"]),
                filtered_folder=pathlib.Path(config["paths"]["filters"]),
            )

            row = {
                "metadata_file": str(metadata_file),
                "uuid": metadata.metadata.get("uuid"),
                "target_snr": metadata.metadata.get("snr"),
                "snr_before": float(snr_before),
                "snr_after": float(snr_after),
                "snr_improvement": float(snr_after - snr_before),
                "snr_before_nonmasked": float(snr_before_nonmasked),
                "snr_after_nonmasked": float(snr_after_nonmasked),
                "snr_improvement_nonmasked": float(
                    snr_after_nonmasked - snr_before_nonmasked
                ),
                "status": "ok",
                "error": "",
            }
        except Exception as exc:
            print(f"Failed for {metadata_file.name}: {exc}")
            continue

        rows.append(row)

    output_csv = datasets_dir / "snr_evaluation_results_dmasked2.csv"
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved {len(rows)} evaluation rows to {output_csv}")

    plot_df = results_df[results_df["status"] == "ok"].copy()
    if plot_df.empty:
        print("No successful evaluations found. Skipping plot generation.")
        return

    summary_df = (
        plot_df.groupby("target_snr", as_index=False)
        .agg(
            mean_improvement=("snr_improvement_nonmasked", "mean"),
            std_improvement=("snr_improvement_nonmasked", "std"),
            n=("snr_improvement_nonmasked", "size"),
        )
        .sort_values("target_snr")
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        summary_df["target_snr"],
        summary_df["mean_improvement"],
        marker="o",
        linewidth=2,
        label="Mean non-masked SNR improvement",
    )
    ax.fill_between(
        summary_df["target_snr"],
        summary_df["mean_improvement"] - summary_df["std_improvement"].fillna(0),
        summary_df["mean_improvement"] + summary_df["std_improvement"].fillna(0),
        alpha=0.2,
        label="+/- 1 std",
    )
    ax.set_xlabel("Target SNR (dB)")
    ax.set_ylabel("Non-masked SNR improvement (dB)")
    ax.set_title("Non-masked SNR improvement as a function of target SNR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_plot = datasets_dir / "snr_improvement_nonmasked_vs_target_snr.png"
    fig.savefig(output_plot, dpi=200)
    plt.close(fig)
    print(f"Saved plot to {output_plot}")


if __name__ == "__main__":
    # main()
    import uw_sim.util as ut

    ut.snr_to_dat(
        pathlib.Path(
            "/mnt/fscompute_shared/simulation_dataset/datasets/snr_evaluation_results_dmasked2.csv"
        ),
        pathlib.Path("snr_results_framed_dmasked.dat"),
    )
