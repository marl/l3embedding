import argparse
import h5py
import numpy as np
import os
import random
import warnings
import joblib
import sys
from data.avc.sample import get_max_abs_sample_value, write_to_h5
from data.utils import read_csv_as_dicts
import soundfile as sf
from IPython.display import Audio
import matplotlib.pyplot as plt


def sample_one_second(audio_data, sampling_frequency, start, augment=False):
    """Return one second audio samples randomly

    Args:
        audio_data: audio data to sample from
        sampling_frequency: audio sample rate
        augment: if True, perturb the data in some fashion

    Returns:
        One second samples, start time, and augmentation parameters

    """
    sampling_frequency = int(sampling_frequency)
    orig_audio_data = audio_data
    audio_data = audio_data[start:start+sampling_frequency]

    if audio_data.shape[0] != sampling_frequency:
        # Pad audio that isn't one second
        warnings.warn('Got audio that is less than one second', UserWarning)
        audio_data = np.pad(audio_data,
                                ((0, sampling_frequency - audio_data.shape[0]),),
                                mode='constant')

    if augment:
        orig_dtype = audio_data.dtype
        audio_data = audio_data.astype(float)
        # Make sure we don't clip
        if np.abs(audio_data).max():
            max_gain = min(0.1, get_max_abs_sample_value(orig_dtype)/np.abs(audio_data).max() - 1)
        else:
            warnings.warn('Got audio sample with all zeros', UserWarning)
            max_gain = 0.1
        gain = 1 + random.uniform(-0.1, max_gain)
        assert 0.9 <= gain <= 1.1
        audio_data *= gain

        audio_data = audio_data.astype(orig_dtype)
        audio_aug_params = {'gain': gain}
    else:
        audio_aug_params = {}

    return audio_data, audio_aug_params


def process_subset(subset_batch_dir, subset_path, n_jobs=1, verbose=0):
    fname_to_path = {os.path.basename(x['audio_filepath']): x['audio_filepath'] for x in read_csv_as_dicts(subset_path)}
    joblib.Parallel(n_jobs=n_jobs, verbose=verbose)(
        process_batch(os.path.join(subset_batch_dir, fname), fname_to_path) \
        for fname in os.listdir(subset_batch_dir)[:50]) # <---- TESTING
    

def process_batch(batch_path, fname_to_path):
    with h5py.File(batch_path, 'r+') as blob:
        audio_files = [x.decode('utf8') for x in blob['audio_file']]
        audio_start_sample_indices = [int(x) for x in blob['audio_start_sample_idx']]

        audio = []
        audio_gain = []
        for fname, start_idx in zip(audio_files, audio_start_sample_indices):
            audio_path = fname_to_path[fname]
            audio_data, sampling_frequency = sf.read(audio_path,
                                                     dtype='int16',
                                                     always_2d=True)
            audio_data = audio_data.mean(axis=-1).astype('int16')
            audio_data, aug_params = sample_one_second(audio_data, 48000, start_idx, augment=True)
            audio.append(audio_data)
            gain = aug_params['gain']

            if not (0.9 <= gain <= 1.1):
                err_msg = "File {} in batch {} has invalid audio gain {}"
                raise ValueError(err_msg.format(audio_path, batch_path, gain))

            audio_gain.append(gain)

        blob['audio'][:,:,:] = np.ascontiguousarray(np.vstack(audio)[:,None,:])
        blob['audio_gain'][:] = np.array(audio_gain)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recompute batch audio')
    parser.add_argument('batch_dir', type=str, help='Directory where batch files are')
    parser.add_argument('subset_path', type=str, help='Path to directory csv file')
    parser.add_argument('--n-jobs', type=int, default=1, help='Number of parallel jobs to run')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level for joblib')
    args = parser.parse_args()
    process_subset(args.batch_dir, args.subset_path, n_jobs=args.n_jobs, verbose=args.verbose)
