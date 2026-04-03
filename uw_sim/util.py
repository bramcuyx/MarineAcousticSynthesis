"""Utility functions for the UW-Sim project."""

import pathlib

import pandas as pd


def write_bacpipe_annotations(
    dataframe_path: str,
    denoised_path: pathlib.Path,
    buffer: int = 1,
    annot_name: str = "denoised_annotations.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Write bacpipe annotations to a CSV file.

    Parameters
    ----------
    dataframe_path : str
        Path to the DataFrame to be written.
    buffer : int
        Buffer time in seconds to add around each event.
    denoised_path : str, optional
        Path to the denoised audio files.

    Returns
    -------
    pd.DataFrame
        The written DataFrame.
    pd.DataFrame
        A copy of the DataFrame with modified audiofilename paths for denoised audio.
    """
    output_df = pd.DataFrame(columns=["audiofilename", "start", "end", "label:event"])
    df = pd.read_pickle(dataframe_path)

    def _append_row(row_data: dict) -> None:
        nonlocal output_df
        # Wrap dict in a list to build a one-row DataFrame from scalar values.
        output_df = pd.concat(
            [output_df, pd.DataFrame([row_data])],
            ignore_index=True,
        )

    for row in df.itertuples():
        filename = row.audio_file
        if len(row.event_starts) == 0:
            _append_row(
                {
                    "audiofilename": filename,
                    "start": 0,
                    "end": row.duration,
                    "label:event": 0,
                }
            )
            continue
        if row.event_starts[0] > 1:
            _append_row(
                {
                    "audiofilename": filename,
                    "start": 0,
                    "end": row.event_starts[0] + buffer,
                    "label:event": 0,
                }
            )

        for i in range(len(row.event_starts)):
            start_time = row.event_starts[i] + buffer
            end_time = row.event_ends[i] - buffer
            _append_row(
                {
                    "audiofilename": filename,
                    "start": start_time,
                    "end": end_time,
                    "label:event": 1,
                }
            )
        if row.event_ends[-1] < (row.duration - 1 - buffer):
            _append_row(
                {
                    "audiofilename": filename,
                    "start": row.event_ends[-1] - buffer,
                    "end": row.duration,
                    "label:event": 0,
                }
            )
    # form denoise filepath, parentfolder is replaced from output to denoised, filename is the same
    output_denoised_df = output_df.copy()
    for i, row in output_df.iterrows():
        audio_path = pathlib.Path(row["audiofilename"])
        filename = audio_path.name
        file_path = denoised_path / filename
        output_denoised_df.at[i, "audiofilename"] = file_path

    dataframe_path_str = str(dataframe_path)
    output_csv_path = dataframe_path_str.replace(".pkl", ".csv")
    output_denoised_path = dataframe_path_str.replace(".pkl", f"_{annot_name}")

    output_df.to_csv(output_csv_path, index=False, sep=",")
    output_denoised_df.to_csv(output_denoised_path, index=False, sep=",")
    return output_df, output_denoised_df


def snr_to_dat(SNR_csv_path: pathlib.Path, output_dat_path: pathlib.Path) -> None:
    """
    Convert SNR results from a CSV file to a DAT file.

    Parameters
    ----------
    SNR_csv_path : pathlib.Path
        Path to the input CSV file containing SNR results.
    output_dat_path : pathlib.Path
        Path to the output DAT file to be created.
    """
    df = pd.read_csv(SNR_csv_path)
    grouped = (
        df.groupby("target_snr", as_index=False)
        .agg(
            mean_improvement=("snr_improvement", "mean"),
            std_improvement=("snr_improvement", "std"),
            count=("snr_improvement", "size"),
        )
        .sort_values("target_snr")
    )
    grouped.to_csv(output_dat_path, sep=" ", index=False, float_format="%.6f")
