"""Create bacpipe annotation CSV files from the generated dataset dataframe."""

import argparse
import pathlib

import yaml

from uw_sim.util import write_bacpipe_annotations


def main() -> None:
    """Load config and run bacpipe annotation export."""
    project_root = pathlib.Path(__file__).resolve().parents[1]
    config = yaml.safe_load((project_root / "config.yaml").read_text())
    dataset_cfg = config.get("dataset", {})

    default_buffer = int(dataset_cfg.get("bacpipe_buffer_length", 1))
    default_annot_name = dataset_cfg.get("bacpipe_annotations_name", "annotations.csv")

    parser = argparse.ArgumentParser(
        description="Write bacpipe annotations for original and denoised audio files."
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=None,
        help="Buffer in seconds applied around events (overrides config).",
    )
    parser.add_argument(
        "--annot-name",
        default=None,
        help="Annotation CSV filename for both output folders (overrides config).",
    )
    args = parser.parse_args()

    buffer = args.buffer if args.buffer is not None else default_buffer
    annot_name = args.annot_name if args.annot_name is not None else default_annot_name

    datasets_dir = pathlib.Path(config["paths"]["datasets"])
    output_path = pathlib.Path(config["paths"]["output"])
    dataframe_name = config["dataset"]["dataframe_name"]
    dataframe_path = datasets_dir / dataframe_name

    denoised_path = pathlib.Path(config["paths"]["denoised"])

    if not dataframe_path.exists():
        raise FileNotFoundError(
            f"Dataframe file not found: {dataframe_path}. "
            "Generate and save the dataset dataframe first."
        )

    output_df, output_denoised_df = write_bacpipe_annotations(
        dataframe_path=str(dataframe_path),
        denoised_path=denoised_path,
        output_path=output_path,
        buffer=buffer,
        annot_name=annot_name,
    )

    output_csv = output_path / annot_name
    denoised_csv = denoised_path / annot_name

    print(f"Wrote {len(output_df)} rows to {output_csv}")
    print(f"Wrote {len(output_denoised_df)} rows to {denoised_csv}")


if __name__ == "__main__":
    main()
