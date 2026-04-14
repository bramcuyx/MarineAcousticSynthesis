"""Generate a dataset of simulated underwater audio recordings using the `uw_sim.Dataset` class. It creates a specified number of audio files with varying signal-to-noise ratios (SNRs) and saves them to the output directory defined in the configuration file."""

import pathlib

import yaml

from uw_sim.audio_simulator import DataSet

project_root = pathlib.Path(__file__).resolve().parents[1]
config = yaml.safe_load((project_root / "config.yaml").read_text())
PATHS = config["paths"]
backgrounds = pathlib.Path(PATHS["background"])
events = pathlib.Path(PATHS["events"])
masks = pathlib.Path(PATHS["masks"])
output = pathlib.Path(PATHS["output"])
dataframe_path = pathlib.Path(PATHS["datasets"]) / config["dataset"]["dataframe_name"]


dataset = DataSet(
    background_folder=backgrounds,
    events_folder=events,
    mask_folder=masks,
    output_folder=output,
    snr_values=config["dataset"]["snr_values"],
    files_per_snr=config["dataset"]["num_files_per_snr"],
    file_length=config["dataset"]["duration"],
    sample_rate=config["dataset"]["samplerate"],
    events_per_file=config["dataset"]["num_events_per_file"],
)
dataset.generate()
dataset.generate_dataframe()
dataset.save_dataframe(dataframe_path)
