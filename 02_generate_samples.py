import argparse
import logging
import multiprocessing_logging
import math
from functools import partial
from log import init_console_logger
from data.sample import sample_and_save
from data.utils import map_iterate_in_parallel


LOGGER = logging.getLogger('sampling')
LOGGER.setLevel(logging.DEBUG)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pre-sample videos and audios for L3 model.')
    parser.add_argument('-bs',
                        '--batch-size',
                        dest='batch_size',
                        action='store',
                        type=int,
                        default=64,
                        help='Number of examples per training batch')

    parser.add_argument('-ns',
                        '--num-streamers',
                        dest='num_streamers',
                        action='store',
                        type=int,
                        default=64,
                        help='Number of training pescador streamers that can be open concurrently')

    parser.add_argument('-mr',
                        '--mux-rate',
                        dest='mux_rate',
                        action='store',
                        type=float,
                        default=2.0,
                        help='Poisson distribution parameter for determining number of training samples to take from a streamer')

    parser.add_argument('-a',
                        '--augment',
                        dest='augment',
                        action='store_true',
                        default=False,
                        help='If True, performs data augmentation on audio and images')

    parser.add_argument('-pc',
                        '--precompute',
                        dest='precompute',
                        action='store_true',
                        default=False,
                        help='If True, streamer precompute samples')

    parser.add_argument('-nd',
                        '--num-distractors',
                        dest='num_distractors',
                        action='store',
                        type=int,
                        default=1,
                        help='Number of distractors for generating examples')

    parser.add_argument('-im',
                        '--include-metadata',
                        dest='include_metadata',
                        action='store_true',
                        help='If True, includes additional metadata in h5 files')

    parser.add_argument('-mv',
                        '--max-videos',
                        dest='max_videos',
                        action='store',
                        type=int,
                        help='Maximum number of videos to use for generating examples. If not specified, all videos will be used')

    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('-n',
                        '--num-workers',
                        dest='num_workers',
                        action='store',
                        type=int,
                        default=4,
                        help='Number of multiprocessing workers used to download videos')

    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help='Logs verbose info')


    parser.add_argument('subset_path',
                        action='store',
                        type=str,
                        help='Path to subset file')

    parser.add_argument('num_samples',
                        action='store',
                        type=int,
                        help='(Minimum) number of samples to generate')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')

    args = parser.parse_args()

    init_console_logger(LOGGER, verbose=args.verbose)
    multiprocessing_logging.install_mp_handler()

    # Just round up for now
    num_workers = args.num_workers
    batch_size = args.batch_size
    batches_per_worker = int(math.ceil(args.num_samples / (num_workers * batch_size)))

    worker_func = partial(sample_and_save,
        subset_path=args.subset_path,
        num_batches=batches_per_worker,
        output_dir=args.output_dir,
        num_streamers=args.num_streamers,
        batch_size=batch_size,
        random_state=args.random_state,
        precompute=args.precompute,
        num_distractors=args.num_distractors,
        augment=args.augment,
        rate=args.mux_rate,
        max_videos=args.max_videos,
        include_metadata=args.include_metadata)

    map_iterate_in_parallel(range(num_workers), worker_func,
                            processes=num_workers)

    LOGGER.info('Done!')
