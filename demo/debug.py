"""Debug script for dataset generation."""
# %%
import pathlib

import noise_reduction.single_channel as sc
import numpy as np
import soundfile as sf
import yaml

from evaluation.snr import evaluate_snr_improvement
from uw_sim.audio_simulator import AudioSimulator, DataSet, MetadataManager
from uw_sim.denoise import process_and_save_denoised_audio

config = yaml.safe_load(pathlib.Path("../config.yaml").read_text())
PATHS = config["paths"]
backgrounds = pathlib.Path(PATHS["background"])
events = pathlib.Path(PATHS["events"])
masks = pathlib.Path(PATHS["masks"])
output = pathlib.Path(PATHS["output"])
denoised = pathlib.Path(PATHS["denoised"])
filters = pathlib.Path(PATHS["filters"])

# %%
simulator = AudioSimulator(
    background_folder=backgrounds,
    events_folder=events,
    mask_folder=masks,
    output_folder=output,
    sample_rate=48000,
    duration=10,
)
for i in range(10):
    try:
        output_audio, metadata_file = simulator.simulate_audio(snr=10, num_events=1)
        print(f"Output audio file: {output_audio}")
        print(f"Metadata file: {metadata_file}")
    except Exception as e:
        print(f"Error occurred while simulating audio: {e}")
# %%
for f in output.glob("*.wav"):
    print(f"Output audio file: {f}")
    print("Processing file...")
    process_and_save_denoised_audio(
        f,
        output_path=output,
    )

# %%
for f in output.glob("*.json"):
    print(f"Metadata file: {f}")
    mdm = MetadataManager()
    mdm.load_metadata(f)
    snr_results = evaluate_snr_improvement(mdm, verbose=True)
    print(f"SNR improvement results: {snr_results}")
# %%

mdm = MetadataManager()
metadata_file = pathlib.Path(
    "/mnt/fscompute_shared/simulation_dataset/outputs/metadata_fd9ae858.json"
)
audio_file = pathlib.Path(
    "/mnt/fscompute_shared/simulation_dataset/outputs/simulated_audio_fd9ae858.wav"
)
denoised_audio_file = pathlib.Path(
    "/mnt/fscompute_shared/simulation_dataset/denoised/simulated_audio_fd9ae858.wav"
)


mdm.load_metadata(metadata_file)
mask = mdm.metadata["mask"]
mask = np.array(mask, dtype=bool)
print(mask.shape)

import scipy.signal as signal

# %%
from evaluation.snr import compute_snr

recording, sr = sf.read(audio_file)
recording_denoised, _ = sf.read(denoised_audio_file)
recording = signal.stft(recording, fs=sr)[2]
recording_denoised = signal.stft(recording_denoised, fs=sr)[2]
snr_global = compute_snr(recording, mask, mode="frequency")
snr_global_denoised = compute_snr(recording_denoised, mask, mode="frequency")
snr_improvement = snr_global_denoised - snr_global
print(f"Global SNR (noisy): {snr_global} dB")
print(f"Global SNR (denoised): {snr_global_denoised} dB")
print(f"SNR improvement after denoising: {snr_improvement} dB")

# %%

test = pathlib.Path(
    "/mnt/fscompute_shared/simulation_dataset/outputs/simulated_audio_fd9ae858.wav"
)
sc.single_channel_denoising(
    test, method="silence", nfft=256, overlap=128, new_samplerate=48000
)
# %%
