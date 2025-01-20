"""
Audio Simulator

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
TODO: Calculate the Masks spectrogram shaped with 256 fft bins and 128 hop length

Author: Bram Cuyx
    
"""
import os
import random
import numpy as np
import soundfile as sf
import json
import uuid
import pandas as pd
from scipy.signal import spectrogram


class AudioFile:
    """
    Represents an audio file and provides methods to manipulate it.
    
    Attributes:
        file_path (str): The path to the audio file.
        sample_rate (int): The sampling rate of the audio file.
        data (np.array): The audio data.
        file_sample_rate (int): The sampling rate of the audio file.
        
    Methods:
        trim_or_pad(target_length): Trim or pad the audio data to the target length.
        
    Usage:
        audio_file = AudioFile(file_path, sample_rate)
        audio_file.trim_or_pad(target_length)
        
    """
    def __init__(self, file_path, sample_rate):
        self.file_path = file_path
        self.sample_rate = sample_rate
        self.data, self.file_sample_rate = sf.read(file_path)
        if self.file_sample_rate != sample_rate:
            raise ValueError(f"File sampling rate ({self.file_sample_rate} Hz) does not match the specified sampling rate ({sample_rate} Hz).")

    def trim_or_pad(self, target_length):
        """Trim or pad the audio data to the target length."""
        if len(self.data) > target_length:
            start_idx = random.randint(0, len(self.data) - target_length)
            self.data = self.data[start_idx:start_idx + target_length]
        else:
            # TODO: needs to be changed to pad with the audi itself
            self.data = np.pad(self.data, (0, target_length - len(self.data)), mode='symmetric')
        return self.data


class Event:
    """
    Represents an audio event and provides methods to scale it to a specified SNR.
    
    Attributes:
        audio_file (AudioFile): The audio file of the event.
        start_pos (int): The start position of the event in the background.
        end_pos (int): The end position of the event in the background.
        scaled_data (np.array): The scaled audio data of the event.
        class_label (str): The class label of the event.
        mask (np.array): The mask of the event.
        
    Methods:
        scale_to_snr(background_segment, snr): Scale the event audio to achieve the specified SNR when mixed with background.
        
    Usage:
        event = Event(file_path, sample_rate)
        event.scale_to_snr(background_segment, snr)
        
    """
    
    def __init__(self, file_path, sample_rate):
        self.audio_file = AudioFile(file_path, sample_rate)
        self.sample_rate = sample_rate
        self.start_pos = None
        self.end_pos = None
        self.scaled_data = None
        self.class_label = os.path.basename(file_path).split('_')[0]
        self.mask = None

    def _get_corresponding_mask(self):
        """
        Get the corresponding mask file for the given event file.

        Args:
            event_file (str): The path to the event file.

        Returns:
            str: The path to the mask file.
        """
        event_name = os.path.splitext(os.path.basename(self.audio_file.file_path))[0]
        mask_file = os.path.join(self.mask_folder, f"{event_name}.npy")
        if not os.path.exists(mask_file):
            raise FileNotFoundError(f"Mask file for {event_name} not found in {self.mask_folder}.")
        return mask_file
    ## get Noise and signal power in the spectrogram domain
    
    def _get_spectrogram(self, data):
        """
        Get the spectrogram of the audio data.
        
        Args:
            background_segment (np.array): The background segment.

        Returns:
            np.array: The noise spectrogram.
        """
        __, __, Sxx = spectrogram(data, fs=self.sample_rate, nperseg=256, noverlap=128) 
        return Sxx
    

    def _get_event_power(self):
        """
        Get the power of the event audio in the spectrogram domain, where mask is 1.

        Returns:
            float: The power of the event audio.
        """
        Sxx = self._get_spectrogram(self.audio_file.data)
                
        return np.mean(Sxx ** 2, where=self.mask)
 
    def _get_noise_power(self, background_segment):
        """
        Get the power of the noise in the spectrogram domain, where mask is 1.

        Args:
            background_segment (np.array): The background segment.

        Returns:
            float: The power of the noise.
        """
        noise_spectrogram = self._get_spectrogram(background_segment)
        return np.mean(noise_spectrogram ** 2, where=self.mask)
    
    def scale_to_snr(self, background_segment, snr):
        """Scale the event audio to achieve the specified SNR when mixed with background."""
        # TODO: scale based on the mask as well 
        self.mask = self._get_corresponding_mask()
        
        
        
        signal_power = self._get_event_power()
        noise_power = self._get_noise_power(background_segment)
        scaling_factor = np.sqrt(noise_power / (10 ** (snr / 10)) / signal_power)
        self.scaled_data = self.audio_file.data * scaling_factor


class MetadataManager:
    """
    Manages metadata for audio events.
    
    Attributes:
        metadata (dict): The metadata dictionary.
        
    Methods:
        add_event(event, start, end): Add an event to the metadata.
        set_global_metadata(snr, sample_rate, duration, background_file): Set global metadata.
        save_metadata(output_path): Save metadata to a JSON file.
        
    Usage:  
        metadata_manager = MetadataManager()
        metadata_manager.add_event(event, start, end)
        metadata_manager.set_global_metadata(snr, sample_rate, duration, background_file)
        metadata_manager.save_metadata(output_path)
        
    """
    
    def __init__(self):
        self.metadata = {
            "uuid": None,
            "snr": None,
            "sample_rate": None,
            "duration": None,
            "background_file": None,
            "events": []
        }

    def add_event(self, event, start, end):
        self.metadata["events"].append({
            "event_file": event.audio_file.file_path,
            "start": start,
            "end": end,
            "class": event.class_label
        })

    def set_global_metadata(self, snr, sample_rate, duration, background_file):
        self.metadata.update({
            "snr": snr,
            "sample_rate": sample_rate,
            "background_file": background_file,
            "duration": duration
        })

    def save_metadata(self, output_path):
        with open(output_path, "w") as f:
            json.dump(self.metadata, f, indent=4)


class AudioSimulator:
    """
    Simulates audio files with different signal-to-noise ratios (SNR).
    
    Attributes:
        background_folder (str): The path to the folder containing background audio files.
        events_folder (str): The path to the folder containing event audio files.
        mask_folder (str): The path to the folder containing binary mask files.
        output_folder (str): The path to the output folder.
        sample_rate (int): The sampling rate of the audio files.
        duration (int): The duration of the audio files in seconds.
        bg_length (int): The length of the background audio in samples.
        unique_id (str): A unique identifier for the simulation.
        
    Methods:
        _select_random_file(folder): Select a random file from the specified folder.
        simulate_audio(snr, num_events): Simulate audio with the specified SNR and number of events.
        
    Usage:
        simulator = AudioSimulator(background_folder, events_folder, mask_folder, output_folder, sample_rate, duration)
        simulator.simulate_audio(snr, num_events)
    """
    def __init__(self, background_folder, events_folder, mask_folder, output_folder, sample_rate=48000, duration=10):
        self.background_folder = background_folder
        self.events_folder = events_folder
        self.mask_folder = mask_folder
        self.output_folder = output_folder
        self.sample_rate = sample_rate
        self.duration = duration
        self.bg_length = int(sample_rate * duration)
        self.unique_id = str(uuid.uuid4())

    def _select_random_file(self, folder):
        """
        Select a random file from the specified folder.
        
        Args:
            folder (str): The path to the folder containing audio files.
            
        Returns:
            str: The path to the selected file.
        """
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.wav')]
        if not files:
            raise FileNotFoundError(f"No .wav files found in {folder}.")
        return random.choice(files)


    def simulate_audio(self, snr, num_events):
        """
        Simulate audio with the specified SNR and number of events.

        Args:
            snr (int): The signal-to-noise ratio.
            num_events (int): The number of events to simulate.
            
        Returns:
            str: The path to the output audio file.
            str: The path to the metadata file.
        """
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Select and process background
        background_file = self._select_random_file(self.background_folder)
        background_audio = AudioFile(background_file, self.sample_rate)
        background = background_audio.trim_or_pad(self.bg_length)

        # Prepare metadata
        metadata_manager = MetadataManager()
        metadata_manager.set_global_metadata(snr, self.sample_rate, self.duration, background_file)
        # Add the full unique_id to metadata
        metadata_manager.metadata["uuid"] = self.unique_id

        # Generate events and embed them into the background
        output_audio = background.copy()
        aggregate_mask = np.zeros((129, len(output_audio) // 128 + 1)) # 129 is the number of frequency bins in the spectrogram, 128 is the hop length

        for _ in range(num_events):
            event_file = self._select_random_file(self.events_folder)
            event = Event(event_file, self.sample_rate)

            # Trim or pad event
            event.audio_file.trim_or_pad(self.bg_length)

            # Randomly place the event in the background
            start_pos = random.randint(0, self.bg_length - len(event.audio_file.data))
            end_pos = start_pos + len(event.audio_file.data)

            # Scale and mix event
            event.scale_to_snr(background[start_pos:end_pos], snr)
            output_audio[start_pos:end_pos] += event.scaled_data

            # Load and process corresponding mask
            mask_file = self._get_corresponding_mask(event_file)
            event_mask = np.load(mask_file) # Allow for 2d masks
            
            # Translate start_pos in samples to start_pos in spectrogram bins
            start_pos_spec = start_pos // 128
            end_pos_spec = end_pos // 128
            
            # Update the aggregate mask
            aggregate_mask[:, start_pos_spec:end_pos_spec] += event_mask
            aggregate_mask = np.clip(aggregate_mask, 0, 1)
        
            # Record event metadata
            metadata_manager.add_event(event, start_pos / self.sample_rate, end_pos / self.sample_rate)

        # Add the aggregated mask to metadata
        metadata_manager.metadata["mask"] = aggregate_mask

        # Shorten, remove dashes, and use it in filenames
        shortened_uuid = self.unique_id.replace('-', '')[:8]
        output_audio_filename = f"simulated_audio_{shortened_uuid}.wav"
        metadata_filename = f"metadata_{shortened_uuid}.json"

        # Save output audio
        output_file = os.path.join(self.output_folder, output_audio_filename)
        sf.write(output_file, output_audio, self.sample_rate)

        # Save metadata
        metadata_file = os.path.join(self.output_folder, metadata_filename)
        metadata_manager.save_metadata(metadata_file)

        return output_file, metadata_file


class DataSet:
    """
    Generates a dataset of simulated audio files with different signal-to-noise ratios (SNR).
    
    Attributes:
        background_folder (str): The path to the folder containing background audio files.
        events_folder (str): The path to the folder containing event audio files.
        mask_folder (str): The path to the folder containing binary mask files.
        output_folder (str): The path to the output folder.
        lowest_snr (int): The lowest SNR value.
        highest_snr (int): The highest SNR value.
        snr_steps (int): The SNR step size.
        files_per_snr (int): The number of files to generate per SNR value.
        file_length (int): The length of the audio files in seconds.
        sample_rate (int): The sampling rate of the audio files.
        generated_files (list): A list to store the paths of generated audio and metadata files.
        
    Methods:
        generate(): Generate the dataset.
        generate_dataframe(): Generate a pandas DataFrame from the generated files.
        
    Usage:
        dataset = DataSet(background_folder, events_folder, mask_folder, output_folder, ...)
        dataset.generate()
        df = dataset.generate_dataframe()
    """
    def __init__(self, background_folder, events_folder, mask_folder, output_folder,
                 lowest_snr, highest_snr, snr_steps, files_per_snr, file_length,
                 sample_rate=48000):
        self.background_folder = background_folder
        self.events_folder = events_folder
        self.mask_folder = mask_folder
        self.output_folder = output_folder
        self.lowest_snr = lowest_snr
        self.highest_snr = highest_snr
        self.snr_steps = snr_steps
        self.files_per_snr = files_per_snr
        self.file_length = file_length
        self.sample_rate = sample_rate
        self.generated_files = []

    def generate(self):
        snr_values = np.arange(self.lowest_snr, self.highest_snr + self.snr_steps, self.snr_steps)
        for snr in snr_values:
            for _ in range(self.files_per_snr):
                simulator = AudioSimulator(
                    background_folder=self.background_folder,
                    events_folder=self.events_folder,
                    mask_folder=self.mask_folder,
                    output_folder=self.output_folder,
                    sample_rate=self.sample_rate,
                    duration=self.file_length
                )
                audio_file, metadata_file = simulator.simulate_audio(
                    snr=snr,
                    num_events=random.randint(1, 5)
                )
                self.generated_files.append((audio_file, metadata_file))

    def generate_dataframe(self):
        rows = []
        for audio_file, metadata_file in self.generated_files:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

            snr = metadata.get("snr")
            sample_rate = metadata.get("sample_rate")
            duration = metadata.get("duration")
            background_file = metadata.get("background_file")
            events = metadata.get("events", [])
            mask = metadata.get("mask", [])
            unique_id = metadata.get("uuid", "")

            event_files = [event.get("event_file") for event in events]
            event_starts = [event.get("start") for event in events]
            event_ends = [event.get("end") for event in events]
            event_classes = [event.get("class") for event in events]

            rows.append({
                "audio_file": audio_file,
                "snr": snr,
                "sample_rate": sample_rate,
                "duration": duration,
                "background_file": background_file,
                "event_files": event_files,
                "event_starts": event_starts,
                "event_ends": event_ends,
                "event_classes": event_classes,
                "mask": mask,
                "uuid": unique_id
            })

        dataframe = pd.DataFrame(rows)
        return dataframe
