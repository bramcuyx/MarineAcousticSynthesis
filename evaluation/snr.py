"""Module for computing signal-to-noise ratio (SNR) from spectrograms and event masks."""
import pathlib

import numpy as np
import scipy.signal as signal
import soundfile as sf
from noise_reduction.evaluation_metrics import SNR

from uw_sim.audio_simulator import MetadataManager


def compute_snr(recording: np.ndarray, mask: np.ndarray, mode="global") -> np.ndarray:
    """
    Compute signal-to-noise ratio (SNR) from a spectrogram and event mask.

    Parameters
    ----------
    recording : np.ndarray
        2D array with shape ``(num_freq_bins, num_time_bins)`` containing the
        recording in the spectrogram domain.
    mask : np.ndarray
        Boolean-like 2D array with the same shape as ``recording``.
        ``mask[f, t]`` is true where event energy is present and false for
        background-only bins.
    mode : str, default='global'
        SNR output mode.
        - ``'global'`` returns one weighted SNR value for the full recording.
        - ``'frequency'`` returns one SNR value per frequency bin.

    Returns
    -------
    np.ndarray | float
        If ``mode='global'``, returns a scalar SNR in dB.
        If ``mode='frequency'``, returns a 1D array with per-frequency SNR in dB.

    Raises
    ------
    AssertionError
        If ``recording`` and ``mask`` do not have the same shape.
    ValueError`
        If ``mode`` is not one of ``'global'`` or ``'frequency'``.
    """
    # Ensure that recording and mask have the same shape
    assert recording.shape == mask.shape, "Recording and mask must have the same shape."
    power = np.abs(recording) ** 2
    # Compute signal power (where mask is 1) and noise power (where mask is 0)
    recording_power = np.mean(power, axis=1, where=mask)
    recording_power = np.nan_to_num(recording_power, nan=0.0)
    # Mean over time bins
    noise_power = np.mean(power, axis=1, where=~mask)  # Mean over time bins
    signal_power = np.maximum(
        recording_power - noise_power, 0
    )  # Ensure non-negative signal power

    # Avoid division by zero by adding a small epsilon to noise power
    epsilon = 1e-10
    snr_f = (signal_power + epsilon) / (noise_power + epsilon)
    if mode == "global":
        weight = recording_power / np.sum(
            recording_power
        )  # Proportion of time bins with events for each frequency bin
        snr_weighted = (
            snr_f * weight
        )  # Weight the SNR by the proportion of time bins with events
        snr_global = 10 * np.log10(np.mean(snr_weighted))
        return snr_global
    elif mode == "frequency":
        return 10 * np.log10(snr_f)
    else:
        raise ValueError("Invalid mode. Choose 'global' or 'frequency'.")


def get_wiener_coefficients(metadata):
    """Read the file containing the Wiener filter coefficients."""
    audio_file = pathlib.Path(metadata["output_audio_file"])
    # audio is in folder output, wiener is in folder wiener with same filename but different extension
    parent_folder = audio_file.parent.parent
    wiener_file = parent_folder / "wiener" / audio_file.with_suffix(".npy")
    wiener_coefficients = np.load(wiener_file)
    return wiener_coefficients


def get_denoised_audio(metadata):
    """Read the file containing the denoised audio.

    Parameters
    ----------
    metadata (dict): Metadata dictionary containing information about the recording, events, and mask.

    Returns
    -------
    tuple: A tuple containing the denoised audio waveform and its sampling rate.
    """
    audio_file = pathlib.Path(metadata["output_audio_file"])
    # audio is in folder output, wiener is in folder wiener with same filename but different extension
    parent_folder = audio_file.parent.parent
    denoised_file = parent_folder / "denoised" / audio_file.name
    denoised_audio, sr = sf.read(denoised_file)
    return denoised_audio, sr


def evaluate_snr_improvement(
    metadatamanager: MetadataManager, NFFT: int = 256, overlap: int = 128
):
    """Evaluate SNR improvement for a given metadata entry.

    Parameters
    ----------
    metadata : dict
        Metadata dictionary containing information about the recording, events, and mask.
    wiener : pathlib.Path
        Path to the Wiener filter coefficients (denoised spectrogram).
    NFFT : int, default=256
        Number of FFT points for spectrogram computation.
    overlap : int, default=128
        Number of overlapping samples for spectrogram computation.

    Returns
    -------
    tuple
        A tuple containing the following SNR values:
        - SNR after denoising (float): The SNR of the denoised
            recording in dB.
        - SNR before denoising (float): The SNR of the original noisy
            recording in dB.
        - SNR after denoising (non-masked) (float): The SNR
            of the denoised recording in dB, computed only on the
            non-masked (background) bins.
        - SNR before denoising (non-masked) (float): The SNR
            of the original noisy recording in dB, computed only on the
            non-masked (background) bins.
    """
    metadata = metadatamanager.metadata
    mask = metadata["mask"]
    sr = metadata["sample_rate"]
    length = metadata["duration"]
    noise_post = np.zeros_like(mask, dtype=np.float32)
    signal_pre = np.zeros_like(mask, dtype=np.float32)
    signal_post = np.zeros_like(mask, dtype=np.float32)

    signal_est = get_denoised_audio(metadata)[0]
    signal_est_stft = signal.stft(signal_est, fs=sr, nperseg=NFFT, noverlap=overlap)[2]

    background_path = metadata["background_file"]
    background_audio, sr_bg = sf.read(background_path)
    if sr_bg != sr:
        background_audio = signal.resample(
            background_audio, int(len(background_audio) * sr / sr_bg)
        )
    noise_pre = signal.stft(background_audio, fs=sr, nperseg=NFFT, noverlap=overlap)[2]

    wiener_coef = get_wiener_coefficients(metadata)
    events = metadata["events"]
    for event in events:
        event_path = event["event_file"]
        scaling_factor = event["scaling_factor"]
        event_audio, sr_event = sf.read(event_path)
        # resample if needed
        if sr_event != sr:
            event_audio = signal.resample(
                event_audio, int(len(event_audio) * sr / sr_event)
            )

        event_audio_scaled = event_audio * scaling_factor

        start = event["start"] * sr // overlap  # need a frame
        event_stft = signal.stft(
            event_audio_scaled, fs=sr, nperseg=NFFT, noverlap=overlap
        )[2]
        event_stft_length = event_stft.shape[1]

        signal_pre[:, start : start + event_stft_length] += event_stft
        # Compute SNR before and after denoising, and calculate improvement
        # This is a placeholder for the actual SNR computation logic

    for j in range(mask.shape[1]):
        signal_post[:, j] = signal_pre[:, j] * wiener_coef
        noise_post[:, j] = noise_pre[:, j] * wiener_coef

    snr = SNR(signal_pre, noise_pre, signal_post, noise_post, mask)
    snr_after = snr[0]
    snr_before = snr[1]
    snr_after_nonmasked = snr[2]
    snr_before_nonmasked = snr[3]
    return snr_after, snr_before, snr_after_nonmasked, snr_before_nonmasked
