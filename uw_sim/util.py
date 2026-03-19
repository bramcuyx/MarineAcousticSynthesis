"""Utility functions for the UW-Sim project."""

import pandas as pd


def write_bacpipe_annotations(
    dataframe_path: str, buffer: int = 1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Write bacpipe annotations to a CSV file.

    Parameters
    ----------
    dataframe_path : str
        Path to the DataFrame to be written.
    buffer : int
        Buffer time in seconds to add around each event.

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
        audio_path = str(row["audiofilename"])
        denoised_path = audio_path.replace("outputs", "denoised")
        output_denoised_df.at[i, "audiofilename"] = denoised_path

    dataframe_path_str = str(dataframe_path)
    output_csv_path = dataframe_path_str.replace(".pkl", ".csv")
    output_denoised_path = dataframe_path_str.replace(".pkl", "_denoised.csv")

    output_df.to_csv(output_csv_path, index=False, sep=",")
    output_denoised_df.to_csv(output_denoised_path, index=False, sep=",")
    return output_df, output_denoised_df
