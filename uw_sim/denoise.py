"""Functions for processing and saving denoised audio."""

import pathlib

import numpy as np
import scipy.signal as signal
import soundfile as sf
from noise_reduction import single_channel as sc


def process_and_save_denoised_audio(
    wav_path: pathlib.Path,
    output_path: pathlib.Path,
    wiener_path: pathlib.Path,
    verbose: bool = False,
    method="silence",
    **kwargs,
):
    """
    Process a single channel wav file and save the denoised audio and the filter.

    Parameters
    ----------
    wav_path: str
        The path to the wav file
    output_path: str
        The path to save the denoised audio
    method: str
        The method to use for denoising. Options are "masked" and "silence"
    kwargs:
        Additional keyword arguments to pass to the single_channel_denoising function

    Returns
    -------
    None
    """
    output_path.mkdir(parents=True, exist_ok=True)
    wiener_path.mkdir(parents=True, exist_ok=True)

    denoising_result = sc.single_channel_denoising(
        wav_path,
        method=method,
        nfft=kwargs.get("nfft", 256),
        overlap=kwargs.get("overlap", 128),
        **kwargs,
    )
    signal_estimate = denoising_result[0]
    wiener = denoising_result[1]
    # Save the denoised audio

    wiener_file = wiener_path / (wav_path.stem + ".npy")
    output_file = output_path / (wav_path.stem + ".wav")
    # Inverse STFT to get the time domain signal
    _, denoised_audio = signal.istft(
        signal_estimate,
        fs=kwargs.get("new_samplerate", 48000),
        nperseg=kwargs.get("nfft", 256),
        noverlap=kwargs.get("overlap", 128),
    )
    sf.write(output_file, denoised_audio, kwargs.get("new_samplerate", 48000))
    if verbose:
        print(f"Denoised audio saved to {output_file}")
        print(f"Wiener filter saved to {wiener_file}")
    # Save the wiener filter
    np.save(wiener_file, wiener)
