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

## Configuration (`config.yaml`)

The project uses `config.yaml` to define input/output locations for dataset generation and simulation scripts.

Current structure:

```yaml
paths:
    background: /mnt/fscompute_shared/simulation_dataset/backgrounds
    events: /mnt/fscompute_shared/simulation_dataset/events
    output: /mnt/fscompute_shared/simulation_dataset/outputs
    masks: /mnt/fscompute_shared/simulation_dataset/masks
```

Field reference:

- `paths.background`: directory containing background audio files.
- `paths.events`: directory containing foreground/event audio files.
- `paths.output`: destination directory where generated outputs are written.
- `paths.masks`: directory containing mask files used during generation.

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

The `DataSet` class is used to create datasets for machine learning or for the evaluation of noise reduction algorithms.

```python
    from uw_sim.audio_simulator import DataSet

    dataset = DataSet(
        background_folder="path/to/backgrounds",
        events_folder="path/to/events",
        mask_folder="path/to/masks",
        output_folder="path/to/output",
        lowest_snr=0,
        highest_snr=20,
        snr_steps=5,
        files_per_snr=2,
        file_length=10,
        sample_rate=48000
    )
    dataset.generate()
    df = dataset.generate_dataframe()
    print(df)
```

## Demo

You can find a demo notebook in the `demo` directory:
```
demo/demo.ipynb
```
