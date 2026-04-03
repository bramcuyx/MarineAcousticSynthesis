"""Module for computing signal-to-noise ratio (SNR) from spectrograms and event masks."""
import pathlib

import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf
import tqdm
import yaml
from noise_reduction.evaluation_metrics import SNR, SNR_framed
from numpy.testing import verbose

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
    npy_file = parent_folder / "filters" / audio_file.with_suffix(".npy").name
    wiener_coefficients = np.load(npy_file)
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
    metadatamanager: MetadataManager,
    NFFT: int = 256,
    overlap: int = 128,
    verbose: bool = False,
    mode="framed",
    masked=False,
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
    mode : str, default='framed', choices=['framed', 'masked']


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
    noise_post = np.zeros_like(mask, dtype=np.complex128)
    signal_pre = np.zeros_like(mask, dtype=np.complex128)
    signal_post = np.zeros_like(mask, dtype=np.complex128)

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

        start = int(event["start"] * sr // overlap)
        end = int(event["end"] * sr // overlap)
        start_frame = int((event["start"] + 1) * sr // overlap)
        end_frame = int((event["end"] - 1) * sr // overlap)
        event_stft = signal.stft(
            event_audio_scaled, fs=sr, nperseg=NFFT, noverlap=overlap
        )[2]
        event_stft_length = event_stft.shape[1]

        signal_pre[:, start : start + event_stft_length] += event_stft
        # Compute SNR before and after denoising, and calculate improvement
        # This is a placeholder for the actual SNR computation logic
    signal_pre = signal_pre * mask

    signal_post = signal_pre * wiener_coef
    noise_post = noise_pre * wiener_coef
    if verbose:
        # plot the signal spectrogram before denoising and plot aside from that the mask
        # plot the signal after denoising below

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        ax[0].set_title("Signal spectrogram before denoising")
        ax[0].imshow(
            10 * np.log10(np.abs(signal_est_stft)),
            origin="lower",
            aspect="auto",
            cmap="inferno",
        )
        ax[1].set_title("Mask")
        ax[1].imshow(mask, origin="lower", aspect="auto", cmap="gray")
        ax[2].set_title("Signal spectrogram after denoising")
        ax[2].imshow(
            10 * np.log10(np.abs(signal_post)),
            origin="lower",
            aspect="auto",
            cmap="inferno",
        )
        plt.show()

    if mode == "masked" and np.any(mask == 1.0):
        snr = SNR(noise_pre, signal_pre, noise_post, signal_post, mask == 1.0)
        snr_after = snr[0]
        snr_before = snr[1]
        snr_after_nonmasked = snr[2]
        snr_before_nonmasked = snr[3]
        return snr_after, snr_before, snr_after_nonmasked, snr_before_nonmasked

    elif mode == "framed" and np.any(mask == 1.0):
        snr = SNR_framed(
            noise_pre,
            signal_pre,
            noise_post,
            signal_post,
            start_frame,
            end_frame,
            masked=masked,
            mask=mask,
        )
        snr_after = snr[0]
        snr_before = snr[1]
        return snr_after, snr_before

    else:
        raise ValueError("Invalid mode. Choose 'masked' or 'framed'.")


if __name__ == "__main__":
    mode = "framed"
    masked = True
    config = (
        pathlib.Path(__project_root := pathlib.Path(__file__).resolve().parents[1])
        / "config.yaml"
    )
    config_data = yaml.safe_load(config.read_text())
    output_dir = pathlib.Path(config_data["paths"]["output"])
    results_output_dir = pathlib.Path(config_data["paths"]["datasets"])
    metadata_files = sorted(output_dir.glob("metadata_*.json"))
    results_list = []
    results = {}
    if not metadata_files:
        print(f"No metadata files found in {output_dir}")
    for metadata_file in tqdm.tqdm(metadata_files):
        results_dict = {}
        metadata_manager = MetadataManager()
        try:
            metadata_manager.load_metadata(metadata_file)
            metadata_dict = metadata_manager.metadata
            if len(metadata_dict["events"]) != 0:
                snr_results = evaluate_snr_improvement(
                    metadata_manager, mode=mode, masked=masked, verbose=False
                )
                if mode == "masked":
                    snr_after, snr_before, snr_after_nonmasked, snr_before_nonmasked = snr_results  # type: ignore
                    results = {
                        "target_snr": float(metadata_dict["snr"]),
                        "snr_after": snr_after,
                        "snr_before": snr_before,
                        "snr_improvement_masked": snr_after - snr_before,
                        "snr_after_nonmasked": snr_after_nonmasked,
                        "snr_before_nonmasked": snr_before_nonmasked,
                        "snr_improvement": snr_after_nonmasked - snr_before_nonmasked,
                    }
                elif mode == "framed":
                    snr_after, snr_before = snr_results  # type: ignore
                    results = {
                        "target_snr": float(metadata_dict["snr"]),
                        "snr_after": snr_after,
                        "snr_before": snr_before,
                        "snr_improvement": snr_after - snr_before,
                    }
                results_list.append(results)
        except Exception as exc:
            print(f"Failed to process {metadata_file.name}: {exc}")

    results_df = pd.DataFrame(results_list)
    # save the results to a csv file
    results_df.to_csv(results_output_dir / f"snr_results_{mode}.csv", index=False)
    print(f"Saved SNR results to {results_output_dir / f'snr_results_{mode}.csv'}")
    # plot the target SNR vs the SNR improvement
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    grouped = results_df.groupby("target_snr")["snr_improvement"].mean().reset_index()
    plt.plot(grouped["target_snr"], grouped["snr_improvement"], marker="o")
    plt.xlabel("Target SNR (dB)")
    plt.ylabel("Mean SNR Improvement (dB)")
    plt.title("Mean SNR Improvement vs Target SNR")
    plt.grid(True)
    plt.show()
    # save the plot
    plt.savefig(results_output_dir / f"snr_improvement_{mode}.png")
    print(
        f"Saved SNR improvement plot to {results_output_dir / f'snr_improvement_{mode}.png'}"
    )
