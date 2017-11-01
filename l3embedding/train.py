import glob
import random

import pescador
import skvideo.io
import soundfile as sf
from tqdm import tqdm


def get_file_list(data_dir):
    """Return audio and video file list.

    Args:
        data_dir: input directory that contains audio/ and video/

    Returns:
        audio_files: list of audio files
        video_files: list of video files

    """

    audio_files = glob.glob('{}/audio/*'.format(data_dir))
    video_files = glob.glob('{}/video/*'.format(data_dir))
    return audio_files, video_files


def video_to_audio(video_file):
    """Return corresponding audio_file.

    Args:
        video_file: video_file

    Returns:
        audio_file

    """

    *path, _, name = video_file.split('/')
    name = name.split('.')[0] + '.flac'
    return '/'.join(path + ['audio', name])


def sample_one_second(audio_data, sampling_frequency, start, label):
    """Return one second audio samples randomly if start is not specified,
       otherwise, return one second audio samples including start (seconds).

    Args:
        audio_file: audio_file to sample from
        start: starting time to fetch one second samples

    Returns:
        One second samples

    """
    if label:
        start = min(0, int(start * sampling_frequency) - random.randint(0, sampling_frequency))
    else:
        start = random.randrange(len(audio_data) - sampling_frequency)
    return audio_data[start:start+sampling_frequency], start / sampling_frequency


def sample_one_frame(video_data, fps=30):
    """Return one frame randomly and time (seconds).

    Args:
        video_data: video data to sample from
        fps: frame per second

    Returns:
        One frame sampled randomly and time in seconds

    """

    num_frames = video_data.shape[0]
    frame = random.randrange(num_frames - fps)
    return video_data[frame, :, :, :], frame / fps


def sampler(video_file, audio_files):
    """Sample one frame from video_file, with 50% chance sample one second from corresponding audio_file,
       50% chance sample one second from another audio_file in the list of audio_files.

    Args:
        video_file: video_file to sample from
        audio_files: candidate audio_files to sample from

    Returns:
        A generator that yields dictionary of video sample, audio sample,
        and label (0: not from corresponding files, 1: from corresponding files)

    """

    video_data = skvideo.io.vread(video_file)
    audio_file = video_to_audio(video_file)

    if random.random() < 0.5:
        audio_file = random.choice([af for af in audio_files if af != audio_file])
        label = 0
    else:
        label = 1

    audio_data, sampling_frequency = sf.read(audio_file)

    while True:
        sample_video_data, video_start = sample_one_frame(video_data)
        sample_audio_data, audio_start = sample_one_second(audio_data, sampling_frequency, video_start, label)

        yield {
            'video': sample_video_data,
            'audio': sample_audio_data[:,0],
            'label': label,
            'audio_file': audio_file,
            'video_file': video_file,
            'audio_start': audio_start,
            'video_start': video_start
        }


def data_generator(data_dir, k=32, batch_size=64, random_state=20171021):
    """Sample video and audio from data_dir, returns a streamer that yield samples infinitely.

    Args:
        data_dir: directory to sample video and audio from
        k: number of concurrent open streamer
        batch_size: batch size

    Returns:
        A generator that yield infinite video and audio samples from data_dir

    """

    random.seed(random_state)

    audio_files, video_files = get_file_list(data_dir)
    seeds = []
    for video_file in tqdm(random.sample(video_files, k)):
        seeds.append(pescador.Streamer(sampler, video_file, audio_files))

    mux = pescador.Mux(seeds, k)
    if batch_size == 1:
        return mux
    else:
        return pescador.BufferedStreamer(mux, batch_size)


def train(csv_file, batch_size=64, rate=16, seed=20171011):

    train_gen = data_generator(
        csv_file,
        batch_size=batch_size,
        lam=rate,
        revive=True,
        random_state=seed)
