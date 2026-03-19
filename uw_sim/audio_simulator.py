"""
Audio Simulator.

This module provides classes and functions to simulate audio files with different signal-to-noise ratios (SNR).
It includes classes to handle audio files, events, and metadata management.

Classes:
    AudioFile: Represents an audio file and provides methods to manipulate it.
    Event: Represents an audio event and provides methods to scale it to a specified SNR.
    MetadataManager: Manages metadata for audio events.
    AudioSimulator: Simulates audio files with different SNR values.
    DataSet: Generates a dataset of simulated audio files for a range of SNR values.

Usage:
    simulator = AudioSimulator(background_folder, events_folder, mask_folder, output_folder, sample_rate, duration)
    dataset = DataSet(background_folder, events_folder, mask_folder, output_folder, ...)

Status: In development

TODO: Fill a folder with the backgrounds
TODO: Fill a folder with the events

Author: Bram Cuyx

"""

import json
import os
import pathlib
import random
import time
import uuid

import numpy as np
import pandas as pd
import scipy.signal as signal
import soundfile as sf
from scipy.signal import spectrogram

random.seed(time.time())  # Initialize random seed based on current time


class AudioFile:
    """
    Audio file container with utility methods for waveform length handling.

    Attributes
    ----------
    file_path : pathlib.Path
        Path to the audio file on disk.
    sample_rate : int
        Expected sampling rate in Hz.
    data : np.ndarray
        Audio waveform samples.
    file_sample_rate : int
        Sampling rate read from the file in Hz.

    """

    def __init__(self, file_path: pathlib.Path, sample_rate: int):
        """
        Load an audio file and validate its sampling rate.

        Parameters
        ----------
        file_path : pathlib.Path
            Path to the audio file.
        sample_rate : int
            Expected sampling rate in Hz.

        Returns
        -------
        None
        """
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.data, self.file_sample_rate = sf.read(file_path)
        if self.file_sample_rate != sample_rate:
            if self.file_sample_rate % sample_rate == 0:
                # Downsample by integer factor
                number_of_samples = round(
                    len(self.data) * float(sample_rate) / self.file_sample_rate
                )
                self.data = signal.resample(
                    self.data, num=number_of_samples
                )  # takes care of anti-aliasing
            else:
                raise ValueError(
                    f"File sampling rate ({self.file_sample_rate} Hz) does not match the specified sampling rate ({sample_rate} Hz)."
                )

    def trim_or_pad(self, target_length: int):
        """
        Trim or pad the waveform to a target number of samples.

        Parameters
        ----------
        target_length : int
            Desired waveform length in samples.

        Returns
        -------
        np.ndarray
            Audio data with length equal to `target_length`.
        """
        if len(self.data) > target_length:
            start_idx = random.randint(0, len(self.data) - target_length)
            self.data = self.data[start_idx : start_idx + target_length]
        else:
            self.data = np.pad(
                self.data, (0, target_length - len(self.data)), mode="symmetric"
            )
        return self.data


class Event:
    """
    Event representation used for SNR-aware scaling and mixing.

    Attributes
    ----------
    audio_file : AudioFile
        Loaded event waveform wrapper.
    sample_rate : int
        Sampling rate in Hz.
    mask_folder : pathlib.Path
        Directory containing event mask `.npy` files.
    start_pos : int | None
        Start index of the event in the simulated background.
    end_pos : int | None
        End index of the event in the simulated background.
    scaled_data : np.ndarray | None
        Event waveform after SNR scaling.
    class_label : str
        Event class name extracted from the filename.
    mask : np.ndarray
        Binary mask aligned with the event spectrogram.

    """

    def __init__(
        self, file_path: pathlib.Path, sample_rate: int, mask_folder: pathlib.Path
    ):
        """
        Initialize an event wrapper and load its corresponding mask.

        Parameters
        ----------
        file_path : pathlib.Path
            Path to the event audio file.
        sample_rate : int
            Sampling rate in Hz.
        mask_folder : pathlib.Path
            Directory containing event mask `.npy` files.

        Returns
        -------
        None
        """
        self.audio_file = AudioFile(file_path, sample_rate)
        self.sample_rate = sample_rate
        self.mask_folder = mask_folder
        self.start_pos = None
        self.end_pos = None
        self.scaled_data = None
        self.scaling_factor = None
        self.class_label = os.path.basename(file_path).split("_")[0]
        self.mask = np.load(self._get_corresponding_mask())

    def _get_corresponding_mask(self):
        """
        Resolve the mask path corresponding to the current event file.

        Returns
        -------
        str
            Path to the matching mask file.

        Raises
        ------
        FileNotFoundError
            If no matching mask file exists in `self.mask_folder`.
        """
        event_name = os.path.splitext(os.path.basename(self.audio_file.file_path))[0]
        mask_file = os.path.join(self.mask_folder, f"{event_name}.npy")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(
                f"Mask file for {event_name} not found in {self.mask_folder}."
            )
        return mask_file

    ## get Noise and signal power in the spectrogram domain

    def _get_spectrogram(self, data: np.ndarray):
        """
        Compute a spectrogram representation of input audio.

        Parameters
        ----------
        data : np.ndarray
            Time-domain audio waveform.

        Returns
        -------
        np.ndarray
            Spectrogram values produced by `scipy.signal.spectrogram`.
        """
        __, __, Sxx = spectrogram(data, fs=self.sample_rate, nperseg=256, noverlap=128)
        return Sxx

    def _get_event_power(self):
        """
        Estimate event power in the spectrogram domain.

        Returns
        -------
        float
            Mean squared spectrogram energy where `self.mask` is true.
        """
        Sxx = self._get_spectrogram(self.audio_file.data)  # type: ignore

        return np.mean(Sxx**2, where=self.mask)

    def _get_noise_power(self, background_segment: np.ndarray):
        """
        Estimate background power in the spectrogram domain.

        Parameters
        ----------
        background_segment : np.ndarray
            Background audio segment aligned with the event duration.

        Returns
        -------
        float
            Mean squared spectrogram energy where `self.mask` is true.
        """
        noise_spectrogram = self._get_spectrogram(background_segment)
        return np.mean(noise_spectrogram**2, where=self.mask)

    def scale_to_snr(self, background_segment: np.ndarray, snr: float):
        """
        Scale event audio to a target SNR against a background segment.

        Parameters
        ----------
        background_segment : np.ndarray
            Background segment used as the noise reference.
        snr : float
            Target signal-to-noise ratio in dB.

        Returns
        -------
        None
            The scaled waveform is stored in `self.scaled_data`.
        """
        signal_power = self._get_event_power()
        noise_power = self._get_noise_power(background_segment)
        self.scaling_factor = np.sqrt(noise_power * (10 ** (snr / 10)) / signal_power)
        self.scaled_data = self.audio_file.data * self.scaling_factor


class MetadataManager:
    """
    Metadata accumulator for generated audio files and embedded events.

    Attributes
    ----------
    metadata : dict
        Dictionary containing global simulation fields and event annotations.

    """

    def __init__(self):
        """
        Initialize an empty metadata structure for one generated sample.

        Returns
        -------
        None
        """
        self.metadata = {
            "uuid": str(uuid.uuid4()),
            "snr": None,
            "sample_rate": None,
            "duration": None,
            "background_file": None,
            "events": [],
            "mask": None,
            "output_audio_file": None,
        }

    def add_event(self, event: Event, start: float, end: float):
        """
        Add one event annotation to the metadata.

        Parameters
        ----------
        event : Event
            Event object embedded in the simulated output.
        start : float
            Event start time in seconds.
        end : float
            Event end time in seconds.

        Returns
        -------
        None
        """
        self.metadata["events"].append(
            {
                "event_file": event.audio_file.file_path,
                "start": start,
                "end": end,
                "class": event.class_label,
                "scaling_factor": event.scaling_factor,
            }
        )

    def set_global_metadata(
        self,
        snr: float,
        sample_rate: int,
        duration: float,
        background_file: pathlib.Path,
    ):
        """
        Set file-level metadata fields for one simulated sample.

        Parameters
        ----------
        snr : float
            Target signal-to-noise ratio in dB.
        sample_rate : int
            Sampling rate in Hz.
        duration : float
            Output duration in seconds.
        background_file : pathlib.Path
            Path to the selected background file.

        Returns
        -------
        None
        """
        self.metadata.update(
            {
                "snr": snr,
                "sample_rate": sample_rate,
                "background_file": background_file,
                "duration": duration,
            }
        )

    def save_metadata(self, output_path: pathlib.Path | str):
        """
        Save metadata as JSON to disk.

        Parameters
        ----------
        output_path : pathlib.Path | str
            Destination path for the metadata file.

        Returns
        -------
        None
        """

        def _json_serializer(obj):
            if isinstance(obj, pathlib.Path):
                return str(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=4, default=_json_serializer)

    def load_metadata(self, input_path: pathlib.Path | str):
        """
        Load metadata from a JSON file.

        Parameters
        ----------
        input_path : pathlib.Path | str
            Path to the metadata JSON file.

        Returns
        -------
        None
            Loaded metadata is stored in `self.metadata`.
        """
        with open(input_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
            self.metadata["mask"] = np.array(self.metadata["mask"])

    def print_metadata(self):
        """
        Print the current metadata content in a readable format.

        Returns
        -------
        None
        """
        print(json.dumps(self.metadata, indent=4))


class AudioSimulator:
    """
    End-to-end simulator for creating mixed audio examples at a target SNR.

    Attributes
    ----------
    background_folder : pathlib.Path
        Directory with background `.wav` files.
    events_folder : pathlib.Path
        Directory with event `.wav` files.
    mask_folder : pathlib.Path
        Directory with event mask `.npy` files.
    output_folder : pathlib.Path
        Directory where generated outputs are written.
    sample_rate : int
        Sampling rate in Hz.
    duration : int
        Output duration in seconds.
    bg_length : int
        Output length in samples.
    unique_id : str
        UUID for the generated sample pair.
    """

    def __init__(
        self,
        background_folder: pathlib.Path,
        events_folder: pathlib.Path,
        mask_folder: pathlib.Path,
        output_folder: pathlib.Path,
        sample_rate: int = 48000,
        duration: int = 10,
        NFFT: int = 256,
        overlap: int = 128,
    ):
        """
        Initialize simulation settings and input/output directories.

        Parameters
        ----------
        background_folder : pathlib.Path
            Directory containing background `.wav` files.
        events_folder : pathlib.Path
            Directory containing event `.wav` files.
        mask_folder : pathlib.Path
            Directory containing event mask `.npy` files.
        output_folder : pathlib.Path
            Directory where generated outputs are written.
        sample_rate : int, default=48000
            Sampling rate in Hz.
        duration : int, default=10
            Output duration in seconds.

        Returns
        -------
        None
        """
        self.background_folder = background_folder
        self.events_folder = events_folder
        self.mask_folder = mask_folder
        self.output_folder = output_folder
        self.sample_rate = sample_rate
        self.duration = duration
        self.bg_length = int(sample_rate * duration)
        self.NFFT = NFFT
        self.overlap = overlap

    def _select_random_file(self, folder: pathlib.Path) -> pathlib.Path:
        """
        Select one random `.wav` file from a directory.

        Parameters
        ----------
        folder : pathlib.Path
            Directory containing candidate audio files.

        Returns
        -------
        pathlib.Path
            Path to the selected audio file.

        Raises
        ------
        FileNotFoundError
            If no `.wav` files are present in `folder`.
        """
        files = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")
        ]
        if not files:
            raise FileNotFoundError(f"No .wav files found in {folder}.")
        return pathlib.Path(random.choice(files))

    def mask_event(self, event: Event):
        """
        Add one event to the output audio in the stft domain and add the corresponding mask to the aggregate mask.

        Parameters
        ----------
        event : Event
            The event to add.
        start : float
            The start time of the event in seconds.
        end : float
            The end time of the event in seconds.

        Returns
        -------
        event_audio_masked: np.ndarray
            The masked event audio to be added to the output.

        """
        # add events in the stft domain multiply the event with the mask and add it to the output audio
        event_spectrogram = signal.stft(
            event.scaled_data,
            fs=event.sample_rate,
            nperseg=self.NFFT,
            noverlap=self.overlap,
        )[2]
        min_len = np.minimum(event_spectrogram.shape[1], event.mask.shape[1])
        event_spectrogram_masked = (
            event_spectrogram[:, :min_len] * event.mask[:, :min_len]
        )
        event_audio_masked = signal.istft(
            event_spectrogram_masked,
            fs=event.sample_rate,
            nperseg=self.NFFT,
            noverlap=self.overlap,
        )[1]
        return event_audio_masked

    def simulate_audio(self, snr: float, num_events: int):
        """
        Generate one simulated audio file and matching metadata.

        Parameters
        ----------
        snr : float
            Target signal-to-noise ratio in dB.
        num_events : int
            Number of events to embed in the background.

        Returns
        -------
        tuple[pathlib.Path, pathlib.Path]
            Paths to the generated audio file and metadata JSON file.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Select and process background
        background_file = self._select_random_file(self.background_folder)
        background_audio = AudioFile(background_file, self.sample_rate)
        background = background_audio.trim_or_pad(self.bg_length)

        # Prepare metadata
        metadata_manager = MetadataManager()
        metadata_manager.set_global_metadata(
            snr, self.sample_rate, self.duration, background_file
        )
        # Add the full unique_id to metadata
        # metadata_manager.metadata["uuid"] = self.unique_id

        # Generate events and embed them into the background
        output_audio = background.copy()
        aggregate_mask = np.zeros(
            (self.NFFT // 2 + 1, len(output_audio) // self.overlap + 1)
        )  # 129 is the number of frequency bins in the spectrogram, 128 is the hop length

        for _ in range(num_events):
            event_file = self._select_random_file(self.events_folder)
            event = Event(event_file, self.sample_rate, self.mask_folder)

            # Randomly place the event in the background
            start_pos = random.randint(0, self.bg_length - len(event.audio_file.data))
            end_pos = start_pos + len(event.audio_file.data)

            # Scale and mix event
            event.scale_to_snr(background[start_pos:end_pos], snr)
            # add events in the stft domain multiply the event with the mask and add it to the output audio
            masked_event_audio = self.mask_event(event)
            end_pos = start_pos + len(masked_event_audio)
            output_audio[start_pos:end_pos] += masked_event_audio  # type: ignore

            # Load and process corresponding mask
            mask_file = event._get_corresponding_mask()
            event_mask = np.load(mask_file)  # Allow for 2d masks

            # Translate start_pos in samples to start_pos in spectrogram bins
            start_pos_spec = start_pos // self.overlap
            end_pos_spec = (
                start_pos_spec + event_mask.shape[1]
            )  # Assuming mask time dimension matches event duration in spectrogram bins

            # Update the aggregate mask
            aggregate_mask[:, start_pos_spec:end_pos_spec] += event_mask
            aggregate_mask = np.clip(aggregate_mask, 0, 1)

            # Record event metadata
            metadata_manager.add_event(
                event, start_pos / self.sample_rate, end_pos / self.sample_rate
            )

        # Add the aggregated mask to metadata
        metadata_manager.metadata["mask"] = aggregate_mask

        # Shorten, remove dashes, and use it in filenames
        shortened_uuid = metadata_manager.metadata["uuid"].replace("-", "")[:8]
        output_audio_filename = f"simulated_audio_{shortened_uuid}.wav"
        metadata_filename = f"metadata_{shortened_uuid}.json"

        # Save output audio
        output_file = self.output_folder / output_audio_filename
        sf.write(output_file, output_audio, self.sample_rate)
        metadata_manager.metadata["output_audio_file"] = output_file

        # Save metadata
        metadata_file = self.output_folder / metadata_filename
        metadata_manager.save_metadata(metadata_file)

        return output_file, metadata_file


class DataSet:
    """
    Dataset generator for batched simulation across an SNR range.

    Attributes
    ----------
    background_folder : pathlib.Path
        Directory with background `.wav` files.
    events_folder : pathlib.Path
        Directory with event `.wav` files.
    mask_folder : pathlib.Path
        Directory with event mask `.npy` files.
    output_folder : pathlib.Path
        Directory where generated outputs are written.
    lowest_snr : float
        Lowest SNR value in dB.
    highest_snr : float
        Highest SNR value in dB.
    snr_steps : float
        Step size between SNR values.
    files_per_snr : int
        Number of files generated for each SNR value.
    file_length : int
        Duration of each generated file in seconds.
    sample_rate : int
        Sampling rate in Hz.
    generated_files : list[tuple[pathlib.Path, pathlib.Path]]
        Generated `(audio_file, metadata_file)` pairs.
    """

    def __init__(
        self,
        background_folder: pathlib.Path,
        events_folder: pathlib.Path,
        mask_folder: pathlib.Path,
        output_folder: pathlib.Path,
        snr_values: list[float],
        files_per_snr: int,
        file_length: int,
        sample_rate: int,
        events_per_file: list[int],
    ):
        """
        Initialize dataset generation settings.

        Parameters
        ----------
        background_folder : pathlib.Path
            Directory containing background `.wav` files.
        events_folder : pathlib.Path
            Directory containing event `.wav` files.
        mask_folder : pathlib.Path
            Directory containing event mask `.npy` files.
        output_folder : pathlib.Path
            Directory where generated outputs are written.
        lowest_snr : float
            Minimum SNR value in dB.
        highest_snr : float
            Maximum SNR value in dB.
        snr_steps : float
            Step size between SNR values in dB.
        files_per_snr : int
            Number of files generated for each SNR value.
        file_length : int
            Duration of each generated file in seconds.
        sample_rate : int, default=48000
            Sampling rate in Hz.

        Returns
        -------
        None

        Throws
        ------
        ValueError
            If `lowest_snr` is greater than `highest_snr` or if `snr_steps` is not positive.
        """
        self.background_folder = background_folder
        self.events_folder = events_folder
        self.mask_folder = mask_folder
        self.output_folder = output_folder
        self.snr_values = snr_values
        self.files_per_snr = files_per_snr
        self.file_length = file_length
        self.sample_rate = sample_rate
        self.generated_files = []
        self.events_per_file = events_per_file
        self.dataframe: pd.DataFrame | None = None

    def generate(self):
        """
        Generate simulated files for each SNR value in the configured range.

        Returns
        -------
        None
            Generated file pairs are appended to `self.generated_files`.
        """
        for n_events in self.events_per_file:
            for snr in self.snr_values:
                for _ in range(self.files_per_snr):
                    simulator = AudioSimulator(
                        background_folder=self.background_folder,
                        events_folder=self.events_folder,
                        mask_folder=self.mask_folder,
                        output_folder=self.output_folder,
                        sample_rate=self.sample_rate,
                        duration=self.file_length,
                    )
                    try:
                        audio_file, metadata_file = simulator.simulate_audio(
                            snr=snr, num_events=n_events
                        )
                        self.generated_files.append((audio_file, metadata_file))
                    except Exception as e:
                        print(
                            f"Error generating file for SNR {snr} dB with {n_events} events: {e}"
                        )

    def generate_dataframe(self):
        """Build a tabular representation of all generated samples."""
        rows = []
        for audio_file, metadata_file in self.generated_files:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            snr = metadata.get("snr")
            sample_rate = metadata.get("sample_rate")
            duration = metadata.get("duration")
            background_file = metadata.get("background_file")
            events = metadata.get("events", [])
            unique_id = metadata.get("uuid", "")

            event_files = [event.get("event_file") for event in events]
            event_starts = [event.get("start") for event in events]
            event_ends = [event.get("end") for event in events]
            event_classes = [event.get("class") for event in events]

            rows.append(
                {
                    "audio_file": audio_file,
                    "metadata_file": metadata_file,
                    "snr": snr,
                    "sample_rate": sample_rate,
                    "duration": duration,
                    "background_file": background_file,
                    "event_files": event_files,
                    "event_starts": event_starts,
                    "event_ends": event_ends,
                    "event_classes": event_classes,
                    "uuid": unique_id,
                }
            )
        dataframe = pd.DataFrame(rows)
        self.dataframe = dataframe

    def save_dataframe(self, output_path: pathlib.Path | str):
        """Save the generated dataframe to disk as a pickle file."""
        if self.dataframe is None:
            raise ValueError(
                "No dataframe to save. Run `generate_dataframe()` before saving."
            )
        self.dataframe.to_pickle(output_path)
