"""Script for interactively choosing events from the annotation table. This script loads the annotation table, applies a path transformation to the "Begin Path" column to create a new "Path" column, and then uses an interactive row filter to allow the user to visually inspect and select rows based on their spectrograms. The selected rows are then saved to a pickle file for later use."""
# %%
import datetime

import librosa as lb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from numpy import nan
from plotandfilter.plotandfilter import interactive_row_filter as irf
from scipy.signal import spectrogram


def plot_row_event(row):
    """
    Plot an event and some context around it in the spectrogram domain. The event is defined by the "Beg File Samp (samples)" and "End File Samp (samples)" columns in the input row, which are expected to be in samples at the original sampling rate of the audio file. The function reads the audio file specified in the "Path" column, resamples it to 48 kHz if necessary, and then extracts a segment of the audio around the event with some additional context before and after the event. It then computes and plots the spectrogram of this segment, highlighting the event region.

    Parameters
    ----------
    row : pandas.Series
        A row from a pandas DataFrame containing at least the following columns:
        - "Path": The file path to the audio file.
        - "Beg File Samp (samples)": The starting sample index of the event in the original
            sampling rate.
        - "End File Samp (samples)": The ending sample index of the event in the original
            sampling rate.
        - "Label": A label for the event, used in the plot title.

    Raises
    ------
    ValueError
        If the sampling rate of the audio file is too low for resampling to 48 k
        Hz.


    """
    data, fs = sf.read(row["Path"])  # read the audio file and resample to 48 khz
    data_resampled = lb.resample(data, orig_sr=fs, target_sr=48000)
    if fs < 48000:
        raise ValueError(f"Sampling rate of {fs} is too low for resampling to 48 kHz")

    # data, fs = sf.read(row['Path'])
    _, ax = plt.subplots(1, 2, figsize=(14, 5))

    start_sample = int(row["Beg File Samp (samples)"])
    end_sample = int(row["End File Samp (samples)"])

    start_sample_resampled = int(start_sample * 48000 / fs)
    end_sample_resampled = int(end_sample * 48000 / fs)

    event_length = end_sample_resampled - start_sample_resampled

    # need for some context around the event to be able to see the event in the spectrogram
    neg_extension = int(2.5 * 48000)  # 2.5 seconds before the event
    pos_extension = int(2.5 * 48000)  # 2.5 seconds after the event

    if start_sample_resampled - neg_extension < 0:
        neg_extension = start_sample_resampled
    if end_sample_resampled + pos_extension > len(data_resampled):
        pos_extension = len(data_resampled) - end_sample_resampled

    overlap = 512
    N_FFT = 1024
    lower_bound = start_sample_resampled - neg_extension
    upper_bound = end_sample_resampled + pos_extension
    __, __, Spect = spectrogram(
        data_resampled[lower_bound:upper_bound], fs, nperseg=N_FFT, noverlap=overlap
    )
    ax[0].set_title(f"Row Label: {row.Label}")
    ax[0].set_xlabel("Time (s)")
    ax[0].set_ylabel("Frequency (Hz)")
    # time_bins = Spect.shape[1]
    # freq_bins = Spect.shape[0]
    time_extent = upper_bound - lower_bound
    freq_extent = 48000 / 2

    ax[0].imshow(
        np.log10(np.abs(Spect)),
        origin="lower",
        aspect="auto",
        cmap="inferno",
        extent=[0, time_extent / 48000, 0, freq_extent],
    )
    # plot in ax[1] a zoomed in version of the event
    ax[1].set_title("Zoomed in")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("Frequency (Hz)")
    spect_lower_bound = int(neg_extension / overlap)
    spect_upper_bound = int((neg_extension + event_length) / overlap)
    ax[1].imshow(
        np.log10(np.abs(Spect[:, spect_lower_bound:spect_upper_bound])),
        origin="lower",
        aspect="auto",
        cmap="inferno",
        extent=[0, event_length / 48000, 0, freq_extent],
    )
    vline1 = neg_extension / 48000
    vline2 = (neg_extension + event_length) / 48000
    ax[0].axvline(
        vline1,
        color="red",
        alpha=0.5,
    )
    ax[0].axvline(vline2, color="red", alpha=0.5)
    plt.show()


def _moc_path(path):
    if not isinstance(path, str):
        return nan
    elif path.startswith("D:\data"):
        return path.replace("D:\data", "//mnt/fscompute_shared").replace("\\", "/")
    return path.replace(
        "\\\\qarchive\\data_sensors", "//mnt/qarchive_data_sensors"
    ).replace("\\", "/")


annotation_location = "/mnt/fs_shared/onderzoek/6. Marine Observation Center/Projects/SoundLib_VLIZ2024/sound_db/sound_bpns/selection_tables/all_annotations.csv"
df = pd.read_csv(annotation_location)
df["Path"] = df["Begin Path"].apply(_moc_path)
df = df.dropna(subset=["Path"])

print(f"Loaded {len(df)} rows from {annotation_location}")
df = df.sample(frac=1)  # randomise the order of the rows
filtered_df = irf(df, plot_row_event)

# %%
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = (
    f"/data/bram.cuyx/Gitlab/uw-sim/chosen_rows/filtered_annotations_{current_time}.pkl"
)

filtered_df.to_pickle(output_filename)
print(f"Filtered dataframe saved to {output_filename}")

# %%
import pathlib

folder = pathlib.Path("/data/bram.cuyx/Gitlab/uw-sim/chosen_rows/")
files = folder.glob("filtered_annotations*.pkl")
files = [file for file in files]
outputdf = pd.read_pickle(files[0])
for f in files[1:]:
    df = pd.read_pickle(f)
    outputdf = pd.concat([outputdf, df], ignore_index=True)

print(f"Loaded {len(outputdf)} rows out of {len(files)} files")
print(outputdf.head())
# %%
