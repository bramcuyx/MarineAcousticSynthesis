"""Module for computing signal-to-noise ratio (SNR) from spectrograms and event masks."""
import numpy as np


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
