# Quantize a QuartzNet ONNX model for Vietnamese ASR using VIVOS calibration data (static int8 quantization)

import glob
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import vai_q_onnx
from onnxruntime.quantization import QuantType
from vai_q_onnx import quantize_static


class AudioCalibrationDataReader:
    """
    Calibration data reader for audio files using QuartzNet-style preprocessing.
    """

    def __init__(
        self,
        dataset_path,
        input_name="mel_features",
        max_samples=100,
        sample_rate=16000,
        n_mels=64,
        max_duration=16.7,
    ):
        """
        Args:
            dataset_path: Path to VIVOS dataset (should contain train/dev/test folders).
            input_name: Name of the ONNX input node.
            max_samples: Maximum number of samples to use for calibration.
            sample_rate: Audio sample rate (16 kHz for QuartzNet).
            n_mels: Number of mel features (64 for QuartzNet).
            max_duration: Maximum audio duration in seconds (not enforced here, kept for compatibility).
        """
        self.dataset_path = dataset_path
        self.input_name = input_name
        self.max_samples = max_samples
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_duration = max_duration

        print("Audio processing config:")
        print(f"  Sample rate: {self.sample_rate}")
        print(f"  N mels: {self.n_mels}")
        print(f"  Max duration: {self.max_duration}s")

        # Find all audio files
        self.audio_files = self._find_audio_files()

        # Limit number of files for calibration
        if len(self.audio_files) > max_samples:
            random.seed(42)  # For reproducible sampling
            self.audio_files = random.sample(self.audio_files, max_samples)

        print(f"Found {len(self.audio_files)} audio files for calibration")
        if self.audio_files:
            print(f"First file: {self.audio_files[0]}")

        self.current_index = 0
        self.processed_count = 0

    def _find_audio_files(self):
        """Find all .wav files in VIVOS dataset."""
        audio_files = []

        # Expected VIVOS structure: {train,dev,test}/*/waves/*.wav
        for split in ["train", "dev", "test"]:
            pattern = os.path.join(self.dataset_path, split, "**", "*.wav")
            files = glob.glob(pattern, recursive=True)
            audio_files.extend(files)
            print(f"Found {len(files)} files in {split} split")

        if not audio_files:
            # Fallback: any .wav under dataset_path
            pattern = os.path.join(self.dataset_path, "**", "*.wav")
            audio_files = glob.glob(pattern, recursive=True)
            print(f"Fallback search found {len(audio_files)} wav files")

        return sorted(audio_files)

    def preprocess_audio(self, audio_path, sample_rate=16000):
        """
        Preprocess audio file with the same method as the ASR model:
        - mono conversion
        - resample to 16k
        - MelSpectrogram (n_fft=512, win=20ms, hop=10ms, n_mels=64)
        - log amplitude and per-feature normalization
        Returns numpy array shaped [1, n_mels, time].
        """
        try:
            waveform, orig_sr = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if needed
            if orig_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(orig_sr, sample_rate)
                waveform = resampler(waveform)

            mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=512,
                win_length=int(sample_rate * 0.02),
                hop_length=int(sample_rate * 0.01),
                n_mels=self.n_mels,
                window_fn=torch.hann_window,
                normalized=False,
            )

            mel = mel_spectrogram(waveform)                       # [1, n_mels, time]
            mel = torch.log(mel + 1e-6)                           # log-amplitude
            mel = (mel - mel.mean(dim=2, keepdim=True)) / (mel.std(dim=2, keepdim=True) + 1e-6)

            return mel.numpy()

        except Exception as e:
            print(f"\nError processing {audio_path}: {e}")
            return None

    def get_next(self):
        """
        Get next calibration sample in the format expected by quantize_static:
        returns a dict {input_name: np.ndarray} or None when finished.
        """
        if self.current_index >= len(self.audio_files):
            print(f"\nFinished processing all {self.processed_count} audio files")
            return None

        audio_path = self.audio_files[self.current_index]

        # Progress indicator
        print(
            f"\rProcessing audio {self.processed_count + 1}/{len(self.audio_files)}: "
            f"{os.path.basename(audio_path)}",
            end="",
            flush=True,
        )

        mel_features = self.preprocess_audio(audio_path, self.sample_rate)

        self.current_index += 1
        self.processed_count += 1

        if mel_features is None:
            # Skip this file and try the next
            return self.get_next()

        # Ensure float32
        mel_features = mel_features.astype(np.float32)

        return {self.input_name: mel_features}


def quantize_quartznet_model(model_path, output_path, dataset_path, max_samples=100):
    """
    Quantize a QuartzNet ONNX model using VIVOS dataset for static calibration.
    """
    print("=== QuartzNet Model Quantization ===")

    # Validate inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    # Create calibration data reader
    print("Creating calibration data reader...")
    calibration_data_reader = AudioCalibrationDataReader(
        dataset_path=dataset_path,
        input_name="mel_features",
        max_samples=max_samples,
        sample_rate=16000,
        n_mels=64,
        max_duration=16.7,
    )

    if len(calibration_data_reader.audio_files) == 0:
        raise ValueError("No audio files found in dataset")

    print(f"\nStarting quantization with {len(calibration_data_reader.audio_files)} samples...")

    # Quantize model
    quantize_static(
        model_input=model_path,
        model_output=output_path,
        calibration_data_reader=calibration_data_reader,
        quant_format=vai_q_onnx.QuantFormat.QOperator,
        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,
        input_nodes=[],
        output_nodes=[],
        per_channel=False,
        reduce_range=False,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "AddQDQPairToWeight": True,
        },
    )

    print(f"\nQuantization completed. Output saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    try:
        quantize_quartznet_model(
            model_path="quartznet_vietnamese_mel_fixed.onnx",
            output_path="quan_quartz.onnx",
            dataset_path="/home/andyloou/viet-asr/vivos",
            max_samples=1000,
        )
    except Exception as e:
        print(f"Quantization failed: {e}")
        import traceback
        traceback.print_exc()

"""
Command line usage example:

python quan.py \
  --model ./quartznet_vietnamese_mel_fixed.onnx \
  --output ./quan_quartz.onnx \
  --dataset /home/andyloou/viet-asr/vivos \
  --samples 100

Or programmatically:

quantize_quartznet_model(
    model_path="quartznet_vietnamese_mel_fixed.onnx",
    output_path="quan_quartz.onnx",
    dataset_path="/home/andyloou/viet-asr/vivos",
    max_samples=100
)
"""
