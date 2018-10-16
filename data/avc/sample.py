import glob
import math
import logging
import os
import random
import warnings

import h5py
import numpy as np
import pescador
import scipy.misc
import skimage
import skimage.color
from skvideo.io import FFmpegReader, ffprobe
import soundfile as sf
from tqdm import tqdm
from data.utils import read_csv_as_dicts, flatten_dict
from log import LogTimer

LOGGER = logging.getLogger('sampling')
LOGGER.setLevel(logging.ERROR)


def adjust_saturation(rgb_img, factor):
    """
    Adjust the saturation of an RGB image

    Args:
        rgb_img: RGB image data array
        factor: Multiplicative scaling factor to be applied to saturation

    Returns:
        adjusted_img: RGB image with adjusted saturation
    """
    hsv_img = skimage.color.rgb2hsv(rgb_img)
    imin, imax = skimage.dtype_limits(hsv_img)
    hsv_img[:,:,1] = np.clip(hsv_img[:,:,1] * factor, imin, imax)
    return skimage.color.hsv2rgb(hsv_img)


def adjust_brightness(rgb_img, delta):
    """
    Adjust the brightness of an RGB image

    Args:
        rgb_img: RGB image data array
        delta: Additive (normalized) gain factor applied to each pixel

    Returns:
        adjusted_img: RGB image with adjusted saturation
    """
    imin, imax = skimage.dtype_limits(rgb_img)
    # Convert delta into the range of the image data
    delta = rgb_img.dtype.type((imax - imin) * delta)

    return np.clip(rgb_img + delta, imin, imax)


def horiz_flip(rgb_img):
    """
    Horizontally flip the given image

    Args:
        rgb_img: RGB image data array

    Returns:
        flipped_img: Horizontally flipped image
    """
    return rgb_img[:,::-1,:]


def get_filename(path):
    """Return the filename of a path

    Args: path: path to file

    Returns:
        filename: name of file (without extension)
    """
    return os.path.splitext(os.path.basename(path))[0]


def get_max_abs_sample_value(dtype):
    if np.issubdtype(dtype, np.unsignedinteger):
        return np.iinfo(dtype).max
    elif np.issubdtype(dtype, np.signedinteger):
        return -np.iinfo(dtype).min
    elif np.issubdtype(dtype, np.floating):
        return 1.0


def load_metadata(metadata_path):
    metadata = {}
    for path in glob.glob(metadata_path):
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if idx in (0, 1):
                    continue
                elif idx == 2:
                    fields = [field.strip() for field in line.lstrip('# ').rstrip().split(', ')]
                else:
                    row = [val.strip() for val in line.strip().split(', ')]
                    ytid = row[0]

                    entry = {field: val
                            for field, val in zip(fields[1:], row[1:])}

                    entry['positive_labels'] = entry['positive_labels'].strip('"').split(',')
                    entry['start_seconds'] = float(entry['start_seconds'])
                    entry['end_seconds'] = float(entry['end_seconds'])

                    metadata[ytid] = entry

    return metadata


def sample_one_second(audio_data, sampling_frequency, augment=False):
    """Return one second audio samples randomly

    Args:
        audio_data: audio data to sample from
        sampling_frequency: audio sample rate
        augment: if True, perturb the data in some fashion

    Returns:
        One second samples, start time, and augmentation parameters

    """
    sampling_frequency = int(sampling_frequency)
    if len(audio_data) > sampling_frequency:
        start = random.randrange(len(audio_data) - sampling_frequency)
    else:
        start = 0

    with LogTimer(LOGGER, 'Slicing audio'):
        audio_data = audio_data[start:start+sampling_frequency]

    if audio_data.shape[0] != sampling_frequency:
        # Pad audio that isn't one second
        warnings.warn('Got audio that is less than one second', UserWarning)
        with LogTimer(LOGGER, 'Slicing audio'):
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
            # TODO: Handle audio with all zeros
            warnings.warn('Got audio sample with all zeros', UserWarning)
            max_gain = 0.1
        gain = 1 + random.uniform(-0.1, max_gain)
        assert 0.9 <= gain <= 1.1
        with LogTimer(LOGGER, 'Applying gain to audio'):
            audio_data *= gain

        audio_data = audio_data.astype(orig_dtype)
        audio_aug_params = {'gain': gain}
    else:
        audio_aug_params = {}

    return audio_data, start, audio_aug_params


def sample_cropped_frame(frame_data):
    """
    Randomly crop a video frame, using the method from Look, Listen and Learn


    Args:
        frame_data: video frame data array

    Returns:
        scaled_frame_data: scaled and cropped frame data
        bbox: bounding box for the cropped image
    """
    nx, ny, nc = frame_data.shape
    start_x, start_y = random.randrange(nx - 224), random.randrange(ny - 224)
    end_x, end_y = start_x + 224, start_y + 224

    bbox = {
        'start_x': start_x,
        'start_y': start_y,
    }

    with LogTimer(LOGGER, 'Cropping frame'):
        frame_data = frame_data[start_x:end_x, start_y:end_y, :]

    return frame_data, bbox


def sample_one_frame(video_data, start=None, fps=30, augment=False):
    """Return one frame randomly and time (seconds).

    Args:
        video_data: video data to sample from

    Keyword Args:
        start: start frame of a one second window from which to sample
        fps: frame per second
        augment: if True, perturb the data in some fashion

    Returns:
        One frame sampled randomly, start time in seconds, and augmentation parameters

    """

    num_frames = len(video_data)
    if start is not None:
        start_frame = int(start * fps)
        # Sample frame from a one second window, or until the end of the video
        # if the video is less than a second for some reason
        # Audio should always be sampled one second from the end of the audio,
        # so video frames we're sampling from should also be a second. If it's
        # not, then our video is probably less than a second
        duration = min(fps, num_frames - start_frame)
        if duration != fps:
            warnings.warn('Got video that is less than one second', UserWarning)

        if duration > 0:
            frame = start_frame + random.randrange(duration)
        else:
            warnings.warn('Got video with only a single frame', UserWarning)
            # For robustness, use the last frame if the start_frame goes past
            # the end of video frame
            frame = min(start_frame, num_frames - 1)
    else:
        frame = random.randrange(num_frames)

    frame_data = video_data[frame]
    frame_data, bbox = sample_cropped_frame(frame_data)

    frame_data = skimage.img_as_float(frame_data)

    video_aug_params = {'bounding_box': bbox}

    if augment:
        # Randomly horizontally flip the image
        horizontal_flip = False
        if random.random() < 0.5:
            with LogTimer(LOGGER, 'Flipping frame'):
                frame_data = horiz_flip(frame_data)
            horizontal_flip = True

        # Ranges taken from https://github.com/tensorflow/models/blob/master/research/slim/preprocessing/inception_preprocessing.py

        # Randomize the order of saturation jitter and brightness jitter
        if random.random() < 0.5:
            # Add saturation jitter
            saturation_factor = np.float32(random.random() + 0.5)
            with LogTimer(LOGGER, 'Adjusting saturation'):
                frame_data = adjust_saturation(frame_data, saturation_factor)

            # Add brightness jitter
            max_delta = 32. / 255.
            brightness_delta = np.float32((2*random.random() - 1) * max_delta)
            with LogTimer(LOGGER, 'Adjusting brightness'):
                frame_data = adjust_brightness(frame_data, brightness_delta)
        else:
            # Add brightness jitter
            max_delta = 32. / 255.
            brightness_delta = np.float32((2*random.random() - 1) * max_delta)
            with LogTimer(LOGGER, 'Adjusting brightness'):
                frame_data = adjust_brightness(frame_data, brightness_delta)

            # Add saturation jitter
            saturation_factor = np.float32(random.random() + 0.5)
            with LogTimer(LOGGER, 'Adjusting saturation'):
                frame_data = adjust_saturation(frame_data, saturation_factor)

        video_aug_params.update({
            'horizontal_flip': horizontal_flip,
            'saturation_factor': saturation_factor,
            'brightness_delta': brightness_delta
        })

    frame_data = skimage.img_as_ubyte(frame_data)

    return frame_data, frame, video_aug_params


def read_video(video_path):
    """
    Read a video file as a numpy array

    Resizes frames so that the minimum side is 256 pixels

    Args:
        video_path: Path to video file

    Returns:
        video: Numpy data array

    """
    vinfo = ffprobe(video_path)['video']
    width = int(vinfo['@width'])
    height = int(vinfo['@height'])

    scaling = 256.0 / min(width, height)
    new_width = math.ceil(scaling * width)
    new_height = math.ceil(scaling * height)

    # Resize frames
    reader = FFmpegReader(video_path,
                          outputdict={'-s': "{}x{}".format(new_width,
                                                           new_height) })

    frames = []
    for frame in reader.nextFrame():
        frames.append(frame)
    reader.close()
    return frames


def generate_sample(audio_file_1, audio_data_1, audio_file_2, audio_data_2,
                    video_file_1, video_data_1, video_file_2, video_data_2,
                    audio_sampling_frequency, augment=False, include_metadata=False):
    """
    Generate a sample from the given audio and video files

    The video from which the audio come from is chosen with a fair coin, as is
    the video frame. Thus, there is a 50% chance of producing a positive or
    negative example.

    Args:
        audio_file_1: audio filename
        audio_data_1: audio data array
        audio_file_2: audio filename
        audio_data_2: audio data array
        video_file_1: video filename
        video_data_1: video data array
        video_file_2: video filename
        video_data_2: video data array
        audio_sampling_frequency: audio sample rate

    Keyword Args
        augment: If True, perform data augmention

    Returns:
        sample: sample dictionary
    """
    video_choice = random.random() < 0.5
    audio_choice = random.random() < 0.5

    if audio_choice:
        audio_file = audio_file_1
        audio_data = audio_data_1
    else:
        audio_file = audio_file_2
        audio_data = audio_data_2

    if video_choice:
        video_file = video_file_1
        video_data = video_data_1
    else:
        video_file = video_file_2
        video_data = video_data_2

    label = int(video_choice != audio_choice)

    sample_audio_data, audio_start, audio_aug_params \
        = sample_one_second(audio_data, audio_sampling_frequency, augment=augment)

    sample_video_data, video_start, video_aug_params \
        = sample_one_frame(video_data, start=audio_start, augment=augment)

    sample_audio_data = sample_audio_data.reshape((1, sample_audio_data.shape[0]))

    sample = {
        'video': np.ascontiguousarray(sample_video_data),
        'audio': np.ascontiguousarray(sample_audio_data),
        'label': np.ascontiguousarray(np.array([label, 1 - label])),
    }

    if include_metadata:
        sample['audio_file'] = os.path.basename(audio_file).encode('utf-8')
        sample['video_file'] = os.path.basename(video_file).encode('utf-8')
        sample['audio_start_sample_idx'] = audio_start
        sample['video_start_frame_idx'] = video_start
        sample.update(flatten_dict(audio_aug_params, 'audio'))
        sample.update(flatten_dict(video_aug_params, 'video'))

    return sample


def sampler(video_1, video_2, rate=32, augment=False, precompute=False, include_metadata=False):
    """Sample one frame from video_file, with 50% chance sample one second from corresponding audio_file,
       50% chance sample one second from another audio_file in the list of audio_files.

    Args:
        video_1: dict for candidate video to sample from
        video_2: dict for candidate video to sample from

    Keyword Args:
        rate: Poisson rate parameter. Used for precomputing samples
        augment: If True, perform data augmention
        precompute: If True, precompute samples during initialization so that
                    memory can be discarded

    Returns:
        A generator that yields dictionary of video sample, audio sample,
        and label (0: not from corresponding files, 1: from corresponding files)

    """
    video_file_1 = video_1['video_filepath']
    video_file_2 = video_2['video_filepath']
    audio_file_1 = video_1['audio_filepath']
    audio_file_2 = video_2['audio_filepath']

    debug_msg = 'Initializing streamer with videos "{}" and "{}"'
    LOGGER.debug(debug_msg.format(video_file_1, video_file_2))

    # Hack: choose a number of samples such that we with high probability, we
    #       won't run out of samples, but is also less than the entire length of
    #       the video so we don't have to resize all of the frames
    num_samples = int(scipy.stats.poisson.ppf(0.999, rate))


    try:
        with LogTimer(LOGGER, 'Reading video'):
            video_data_1 = read_video(video_file_1)
    except Exception as e:
        warn_msg = 'Could not open video file {} - {}: {}; Skipping...'
        warn_msg = warn_msg.format(video_file_1, type(e), e)
        LOGGER.warning(warn_msg)
        warnings.warn(warn_msg)
        raise StopIteration()

    try:
        with LogTimer(LOGGER, 'Reading video'):
            video_data_2 = read_video(video_file_2)
    except Exception as e:
        warn_msg = 'Could not open video file {} - {}: {}; Skipping...'
        warn_msg = warn_msg.format(video_file_2, type(e), e)
        LOGGER.warning(warn_msg)
        warnings.warn(warn_msg)
        raise StopIteration()

    try:
        with LogTimer(LOGGER, 'Reading audio'):
            audio_data_1, sampling_frequency = sf.read(audio_file_1,
                                                       dtype='int16',
                                                       always_2d=True)
            audio_data_1 = audio_data_1.mean(axis=-1).astype('int16')

    except Exception as e:
        warn_msg = 'Could not open audio file {} - {}: {}; Skipping...'
        warn_msg = warn_msg.format(audio_file_1, type(e), e)
        LOGGER.warning(warn_msg)
        warnings.warn(warn_msg)
        raise StopIteration()

    try:
        with LogTimer(LOGGER, 'Reading audio'):
            audio_data_2, sampling_frequency = sf.read(audio_file_2,
                                                       dtype='int16',
                                                       always_2d=True)
            audio_data_2 = audio_data_2.mean(axis=-1).astype('int16')
    except Exception as e:
        warn_msg = 'Could not open audio file {} - {}: {}; Skipping...'
        warn_msg = warn_msg.format(audio_file_2, type(e), e)
        LOGGER.warning(warn_msg)
        warnings.warn(warn_msg)
        raise StopIteration()

    if precompute:
        samples = []
        for _ in range(num_samples):
            sample = generate_sample(
                audio_file_1, audio_data_1, audio_file_2, audio_data_2,
                video_file_1, video_data_1, video_file_2, video_data_2,
                sampling_frequency, augment=augment, include_metadata=include_metadata)

            samples.append(sample)

        # Clear the data from memory
        video_data_1 = None
        video_data_2 = None
        audio_data_1 = None
        audio_data_2 = None
        video_data = None
        audio_data = None
        del video_data_1
        del video_data_2
        del audio_data_1
        del audio_data_2
        del video_data
        del audio_data

        while samples:
            # Yield the sample, and remove from the list to free up some memory
            yield samples.pop()
    else:
        while True:
            yield generate_sample(
                audio_file_1, audio_data_1, audio_file_2, audio_data_2,
                video_file_1, video_data_1, video_file_2, video_data_2,
                sampling_frequency, augment=augment, include_metadata=include_metadata)

    raise StopIteration()



def data_generator(subset_path, k=32, batch_size=64, random_state=20171021,
                   precompute=False, num_distractors=1, augment=False, rate=32,
                   max_videos=None, include_metadata=False, cycle=True):
    """Sample video and audio from data_dir, returns a streamer that yield samples infinitely.

    Args:
        subset_path: path to subset file
        k: number of concurrent open streamer
        batch_size: batch size
        random_state: Value used to initialize state of RNG
        num_distractors: Number of pairs to generate a stream for each video

    Returns:
        A generator that yield infinite video and audio samples from data_dir

    """

    random.seed(random_state)
    np.random.seed(random_state)


    LOGGER.info("Loading subset list")
    file_list = read_csv_as_dicts(subset_path)

    LOGGER.info("Creating streamers...")
    if max_videos is not None and max_videos < len(file_list):
        LOGGER.info("Using a subset of {} videos".format(max_videos))
        random.shuffle(file_list)
        file_list = file_list[:max_videos]

    seeds = []
    for video_1 in tqdm(file_list):
        for _ in range(num_distractors):
            video_2 = video_1
            # Make sure we sample a different file
            while video_2 == video_1:
                video_2 = random.choice(file_list)

            streamer = pescador.Streamer(sampler, video_1, video_2,
                                         rate=rate, augment=augment,
                                         precompute=precompute,
                                         include_metadata=include_metadata)
            seeds.append(streamer)

    # Randomly shuffle the seeds
    random.shuffle(seeds)

    mux = pescador.Mux(seeds, k, rate=rate, random_state=random_state)
    if cycle:
        mux = mux.cycle()

    if batch_size == 1:
        return mux
    else:
        return pescador.maps.buffer_stream(mux, batch_size)


def write_to_h5(path, batch):
    with h5py.File(path, 'w') as f:
        for key in batch.keys():
            f.create_dataset(key, data=batch[key], compression='gzip')


def sample_and_save(index, subset_path, num_batches, output_dir,
                    num_streamers=32, batch_size=64, random_state=20171021,
                    precompute=False, num_distractors=1, augment=False, rate=32,
                    max_videos=None, include_metadata=False):
    data_gen = data_generator(
        subset_path,
        batch_size=batch_size,
        random_state=random_state + index,
        k=num_streamers,
        augment=augment,
        num_distractors=num_distractors,
        max_videos=max_videos,
        precompute=precompute,
        rate=rate,
        include_metadata=include_metadata)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for sub_index, batch in enumerate(data_gen):
        batch_path = os.path.join(output_dir, '{}_{}_{}.h5'.format(random_state + index, index, sub_index))
        write_to_h5(batch_path, batch)

        if sub_index == (num_batches - 1):
            break
