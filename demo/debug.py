# %% Demo of the audio simulator and dataset classes
import pathlib

import soundfile as sf
import yaml

from uw_sim.audio_simulator import AudioSimulator, DataSet

config = yaml.safe_load(pathlib.Path("../config.yaml").read_text())
PATHS = config["paths"]
backgrounds = pathlib.Path(PATHS["background"])
events = pathlib.Path(PATHS["events"])
masks = pathlib.Path(PATHS["masks"])
output = pathlib.Path(PATHS["output"])

simulator = AudioSimulator(
    background_folder=backgrounds,
    events_folder=events,
    mask_folder=masks,
    output_folder=output,
    sample_rate=48000,
    duration=10,
)

output_audio, metadata_file = simulator.simulate_audio(snr=10, num_events=1)
print(f"Output audio file: {output_audio}")

print(f"Metadata file: {metadata_file}")
# %%
event_dataset = events.glob("*.wav")
for event in event_dataset:
    data, sr = sf.read(event)
    print(
        f"Event: {event.name}, Sample Rate: {sr}, Duration: {len(data)/sr:.2f} seconds"
    )
# %%
# %%
str(backgrounds)
# %%
