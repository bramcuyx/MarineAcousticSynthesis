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


def moc_path(path):
    if not isinstance(path, str):
        return nan
    elif path.startswith("D:\data"):
        return path.replace("D:\data", "//mnt/fscompute_shared").replace("\\", "/")
    elif path.startswith("O:\\"):
        return path.replace("O:\\", "/mnt/qarchive_data_sensors/").replace("\\", "/")
    elif path.startswith("Q:\\"):
        return path.replace("Q:\\", "/mnt/qarchive_data_sensors/").replace("\\", "/")
    return path.replace(
        "\\\\qarchive\\data_sensors", "//mnt/qarchive_data_sensors"
    ).replace("\\", "/")


def plot_row_background(row):
    data, fs = sf.read(row["Path"])  # read the audio file and resample to 48 khz
    data_resampled = lb.resample(data, orig_sr=fs, target_sr=48000)
    if fs < 48000:
        raise ValueError(f"Sampling rate of {fs} is too low for resampling to 48 kHz")

    fig, ax = plt.subplots()

    start_time = int(row["Begin Time background (s)"])
    start_time_samples = int(start_time * 48000)
    end_time = int(row["End Time background (s)"])
    end_time_samples = int(end_time * 48000)

    __, __, Spect = spectrogram(
        data_resampled[start_time_samples:end_time_samples],
        fs,
        nperseg=256,
        noverlap=128,
    )
    ax.set_title(f"Row Label: {row.Path}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    time_bins = Spect.shape[1]
    freq_bins = Spect.shape[0]
    freq_extent = 48000 / 2

    ax.set_xticks(np.linspace(0, time_bins, num=11))
    ax.set_xticklabels(np.round(np.linspace(0, 10, num=11), 2))
    ax.set_yticks(np.append(np.linspace(0, freq_bins, num=11), freq_bins))
    ax.set_yticklabels(
        np.append(np.round(np.linspace(0, freq_extent, num=11), 2), freq_extent)
    )
    ax.imshow(np.log10(np.abs(Spect)), origin="lower", aspect="auto", cmap="gray")
    plt.show()


# %%
annotation_location = (
    "/mnt/fscompute_shared/sound_classification/total_selections_linux.pkl"
)
df = pd.read_pickle(annotation_location)
# find all unique paths in the dataframe
df["Path"] = df["Begin Path"].apply(moc_path)
unique_paths = df["Path"].unique()
print(f"Found {len(unique_paths)} unique paths in the dataframe")
# %%
# loop through the paths, sort by event start time and find a 10s segment
# without events
background_df = pd.DataFrame(
    columns=["Path", "Begin Time background (s)", "End Time background (s)"]
)
for p in unique_paths:
    df_selection = df[df["Path"] == p]
    df_selection = df_selection.sort_values(by="Begin Time (s)")
    df_selection = df_selection.reset_index(drop=True)
    print(f"Found {len(df_selection)} events in {p}")
    # find a 10s segment without events
    for i in range(len(df_selection)):
        if i == 0:
            if df_selection.loc[i, "Begin Time (s)"] > 10:
                print(f"Found 10s segment without events in {p}")
                df_selection.loc[i, "Begin Time background (s)"] = 0
                df_selection.loc[i, "End Time background (s)"] = 10
                background_df = pd.concat(
                    [
                        background_df,
                        df_selection.loc[
                            [i],
                            [
                                "Path",
                                "Begin Time background (s)",
                                "End Time background (s)",
                            ],
                        ],
                    ],
                    ignore_index=True,
                )

        else:
            if (
                df_selection.loc[i, "Begin Time (s)"]
                - 5
                - (df_selection.loc[i - 1, "End Time (s)"] + 5)
                > 10
            ):
                print(f"Found 10s segment without events in {p}")
                df_selection.loc[i, "Begin Time background (s)"] = (
                    df_selection.loc[i - 1, "End Time (s)"] + 5
                )
                df_selection.loc[i, "End Time background (s)"] = (
                    df_selection.loc[i - 1, "End Time (s)"] + 15
                )
                background_df = pd.concat(
                    [
                        background_df,
                        df_selection.loc[
                            [i],
                            [
                                "Path",
                                "Begin Time background (s)",
                                "End Time background (s)",
                            ],
                        ],
                    ],
                    ignore_index=True,
                )

print(f"Found {len(background_df)} 10s segments without events")
# %%
print(background_df)
# %%
df_out = background_df.sample(frac=1)
filtered_df = irf(df_out, plot_row_background)
# %%
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = (
    f"/data/bram.cuyx/Gitlab/uw-sim/chosen_rows/filtered_background_{current_time}.pkl"
)

filtered_df.to_pickle(output_filename)
print(f"Filtered dataframe saved to {output_filename}")
# %%
