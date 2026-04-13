"""Denoise a dataset of audio files. This script processes all .wav files in the output directory defined in the configuration file, applies a Wiener filter-based denoising algorithm, and saves the denoised audio files to a specified output directory. The script uses multiprocessing to speed up the processing of multiple audio files."""
import os
import pathlib
from multiprocessing import Pool

import yaml
from tqdm import tqdm

from uw_sim.denoise import process_and_save_denoised_audio

config = yaml.safe_load(pathlib.Path("config.yaml").read_text())
PATHS = config["paths"]
output = pathlib.Path(PATHS["output"])
denoised_output = pathlib.Path(PATHS["denoised"])
wiener_output = pathlib.Path(PATHS["filters"])
Xi = config.get("denoise_parameters", {}).get("Xi", 0.20)
beta = config.get("denoise_parameters", {}).get("beta", 0.97)

wav_files = sorted(output.glob("*.wav"))
num_processes = config.get("dataset", {}).get("denoise_processes") or (
    os.cpu_count() or 1
)
print(
    f"Found {len(wav_files)} .wav files in {output}. Using {num_processes} processes for denoising."
)


def process_file(wav_file):
    """Package the denoising process for a single file to be used with multiprocessing."""
    process_and_save_denoised_audio(
        wav_path=wav_file,
        output_path=denoised_output,
        wiener_path=wiener_output,
        Xi=Xi,
        beta=beta,
    )  # -14dB max noise reduction


if __name__ == "__main__":
    if not wav_files:
        print(f"No .wav files found in {output}")
    else:
        with Pool(processes=num_processes) as pool:
            for _ in tqdm(
                pool.imap_unordered(process_file, wav_files), total=len(wav_files)
            ):
                pass
