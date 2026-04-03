# UW-Sim

UW-Sim is a Python package for simulating underwater audio events and managing metadata. It provides tools for generating audio simulations with specified signal-to-noise ratios (SNR) and creating datasets for machine learning and noise reduction.

## Features

- Simulate underwater audio events with specified SNR
- Generate spectrograms of audio data
- Manage metadata for audio events
- Create datasets for machine learning

## Installation

To install the dependencies, use [Poetry](https://python-poetry.org/):

```sh
poetry lock
poetry install
```

## Configuration

The repository ships a template config in `config.template.yaml` that is safe to commit.
Runtime scripts read values from `config.yaml` in the project root.

### Setup

1. Copy the template:

```sh
cp config.template.yaml config.yaml
```

2. Edit `config.yaml` and set your local paths.

### Template fields

```yaml
paths:
    background: /path/to/backgrounds
    events: /path/to/events
    output: /path/to/outputs
    masks: /path/to/masks
    denoised: /path/to/denoised
    filters: /path/to/filters
    datasets: /path/to/datasets

dataset:
    samplerate: 48000
    duration: 10
    snr_values: [-25, -20, -15, -10, -5, 0, 5, 10, 15, 20]
    num_files_per_snr: 100
    num_events_per_file: [0, 1]
    dataframe_name: dataset.pkl
    bacpipe_buffer_length: 1
    bacpipe_annotations_name: annotations.csv
    denoise_processes: 12
    Xi: 0.20
    beta: 0.97
```

Field reference:

- `paths.background`: directory containing background audio files.
- `paths.events`: directory containing foreground/event audio files.
- `paths.output`: destination directory where generated outputs are written.
- `paths.masks`: directory containing mask files.
- `paths.denoised`: destination for denoised `.wav` files.
- `paths.filters`: destination for Wiener filter coefficient files (`.npy`).
- `paths.datasets`: destination for generated dataset artifacts (CSV, plots, pickle).
- `dataset.bacpipe_buffer_length`: context buffer in seconds around events for bacpipe annotations (typically `1`).
- `dataset.bacpipe_annotations_name`: bacpipe annotation filename (recommended: `annotations.csv`) written into both audio folders (`paths.output` and `paths.denoised`).
- `dataset.denoise_processes`: number of worker processes used by `demo/denoise_dataset.py`.

Important denoising assumption:

- The current denoising flow uses `method="silence"` and expects about the first 1 second of each recording to contain background noise only (no foreground event).
- If an event starts immediately at `t=0`, denoising quality can degrade because the initial noise estimate is contaminated.

## Usage
### AudioSimulator
The `AudioSimulator` class is used to simulate underwater audio files.

```python
    from uw_sim.audio_simulator import AudioSimulator

    simulator = AudioSimulator(
        background_folder="path/to/backgrounds",
        events_folder="path/to/events",
        mask_folder="path/to/masks",
        output_folder="path/to/output",
        sample_rate=48000,
        duration=10
    )
    output_audio, metadata_file = simulator.simulate_audio(snr=10, num_events=3)
    print(f"Output audio file: {output_audio}")
    print(f"Metadata file: {metadata_file}")
```

## Dataset

Dataset generation is implemented through `uw_sim.audio_simulator.DataSet` and the script `demo/generate_dataset.py`.
The script reads settings from `config.yaml`, generates simulated files, builds a dataframe from metadata, and saves it to:

- `paths.datasets / dataset.dataframe_name`

Run it with:

```sh
poetry run python demo/generate_dataset.py
```

Equivalent Python usage:

```python
    from uw_sim.audio_simulator import DataSet
    import pathlib

    dataset = DataSet(
        background_folder="path/to/backgrounds",
        events_folder="path/to/events",
        mask_folder="path/to/masks",
        output_folder="path/to/output",
        snr_values=[-25, -20, -15, -10, -5, 0, 5, 10, 15, 20],
        files_per_snr=100,
        file_length=10,
        sample_rate=48000,
        events_per_file=[0, 1],
    )
    dataset.generate()
    dataset.generate_dataframe()
    dataset.save_dataframe(pathlib.Path("path/to/datasets") / "dataset.pkl")
```

## Demo

You can find a demo notebook in the `demo` directory:
```
demo/demo.ipynb
```

## Full pipeline

After configuring `config.yaml`, run the end-to-end workflow with:

1. Generate simulated dataset:

```sh
poetry run python demo/generate_dataset.py
```

2. Denoise generated audio files:

```sh
poetry run python demo/denoise_dataset.py
```

Note: this denoising step assumes there is approximately 1 second of silence/background before the first event.

3. Evaluate SNR improvement:

```sh
poetry run python demo/run_snr_evaluation.py
```

4. Generate bacpipe annotations:

```sh
poetry run python demo/write_bacpipe_annotations.py
```

This writes annotation CSV files to:

- `paths.output / dataset.bacpipe_annotations_name`
- `paths.denoised / dataset.bacpipe_annotations_name`

Optional overrides:

```sh
poetry run python demo/write_bacpipe_annotations.py --buffer 1 --annot-name annotations.csv
```

Prerequisite: `paths.datasets / dataset.dataframe_name` must exist (created by `demo/generate_dataset.py`).
