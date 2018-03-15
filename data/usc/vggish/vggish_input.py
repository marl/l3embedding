# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Compute input examples for VGGish from audio waveform."""

import numpy as np
import resampy
from scipy.io import wavfile

from . import mel_features


def waveform_to_examples(data, sample_rate, target_sample_rate=16000,
                         log_offset=0.01, stft_win_len_sec=0.025,
                         stft_hop_len_sec=0.010, num_mel_bins=64,
                         mel_min_hz=125, mel_max_hz=7500, frame_win_sec=0.96,
                         frame_hop_sec=0.96, **params):
  """Converts audio waveform into an array of examples for VGGish.

  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.

  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is stft_hop_len_sec.
  """

  # Convert to mono.
  if len(data.shape) > 1:
    data = np.mean(data, axis=1)
  # Resample to the rate assumed by VGGish.
  if sample_rate != target_sample_rate:
    data = resampy.resample(data, sample_rate, target_sample_rate)

  # Compute log mel spectrogram features.
  log_mel = mel_features.log_mel_spectrogram(
      data,
      audio_sample_rate=target_sample_rate,
      log_offset=log_offset,
      window_length_secs=stft_win_len_sec,
      hop_length_secs=stft_hop_len_sec,
      num_mel_bins=num_mel_bins,
      lower_edge_hertz=mel_min_hz,
      upper_edge_hertz=mel_max_hz)

  # Frame features into examples.
  features_sample_rate = 1.0 / stft_hop_len_sec
  example_window_length = int(round(
      frame_win_sec * features_sample_rate))
  example_hop_length = int(round(
      frame_hop_sec * features_sample_rate))
  log_mel_examples = mel_features.frame(
      log_mel,
      window_length=example_window_length,
      hop_length=example_hop_length)

  return log_mel_examples


def wavfile_to_examples(wav_file, **params):
  """Convenience wrapper around waveform_to_examples() for a common WAV format.

  Args:
    wav_file: String path to a file, or a file-like object. The file
    is assumed to contain WAV audio data with signed 16-bit PCM samples.

  Returns:
    See waveform_to_examples.
  """
  sr, wav_data = wavfile.read(wav_file)
  assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
  samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
  return waveform_to_examples(samples, sr, **params)
