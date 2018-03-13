import argparse
import logging
import os
import json
from l3embedding.model import load_embedding
from data.usc.us8k import generate_us8k_folds, generate_us8k_fold_data
from log import init_console_logger

LOGGER = logging.getLogger('cls-data-generation')
LOGGER.setLevel(logging.DEBUG)


def parse_arguments():
    """
    Parse arguments from the command line


    Returns:
        args:  Argument dictionary
               (Type: dict[str, *])
    """
    parser = argparse.ArgumentParser(description='Train an urban sound classification model')

    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('-v',
                        '--verbose',
                        dest='verbose',
                        action='store_true',
                        default=False,
                        help='If True, print detailed messages')

    parser.add_argument('-f',
                        '--features',
                        dest='features',
                        action='store',
                        type=str,
                        default='l3_frames_uniform',
                        help='Type of features to be used in training')

    parser.add_argument('-lf',
                        '--label-format',
                        dest='label_format',
                        action='store',
                        type=str,
                        default='int',
                        help='Type of format used for encoding outputs')

    parser.add_argument('-lmp',
                        '--l3embedding-model-path',
                        dest='l3embedding_model_path',
                        action='store',
                        type=str,
                        help='Path to L3 embedding model weights file')

    parser.add_argument('-lmt',
                        '--l3embedding-model-type',
                        dest='l3embedding_model_type',
                        action='store',
                        type=str,
                        default='cnn_L3_orig',
                        help='Type of L3 embedding model')

    parser.add_argument('-lpt',
                        '--l3embedding-pooling-type',
                        dest='l3embedding_pooling_type',
                        action='store',
                        type=str,
                        default='original',
                        help='Type of pooling used to downsample last conv layer of L3 embedding model')

    parser.add_argument('-hs',
                        '--hop-size',
                        dest='hop_size',
                        action='store',
                        type=float,
                        default=0.1,
                        help='Hop size in seconds')

    parser.add_argument('-nrs',
                        '--num-random-samples',
                        dest='num_random_samples',
                        action='store',
                        type=int,
                        help='Number of random samples for randomized sampling methods')

    parser.add_argument('--gpus',
                        dest='gpus',
                        type=int,
                        default=0,
                        help='Number of gpus used for running the embedding model.')

    parser.add_argument('--fold',
                        dest='fold',
                        type=int,
                        help='Fold number to generate. If unused, generate all folds')

    parser.add_argument('metadata_path',
                        action='store',
                        type=str,
                        help='Path to UrbanSound8K metadata file')

    parser.add_argument('model_id',
                        action='store',
                        type=str,
                        help='Model ID used to generate features (or experiment ID)')

    parser.add_argument('dataset_name',
                        action='store',
                        type=str,
                        choices=['us8k'],
                        help='Name of dataset')

    parser.add_argument('data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where training set files are stored')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where output data files will be stored')

    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_arguments()

    init_console_logger(LOGGER, verbose=args['verbose'])
    LOGGER.debug('Initialized logging.')

    # Unpack CL args
    model_type = args['l3embedding_model_type']
    pooling_type = args['l3embedding_pooling_type']
    metadata_path = args['metadata_path']
    data_dir = args['data_dir']
    features = args['features']
    hop_size = args['hop_size']
    random_state = args['random_state']
    num_random_samples = args['num_random_samples']
    label_format = args['label_format']
    model_path = args['l3embedding_model_path']
    num_gpus = args['gpus']
    output_dir = args['output_dir']
    model_id = args['model_id']
    dataset_name = args['dataset_name']
    fold_num = args['fold']

    if model_path:
        LOGGER.info('Loading embedding model...')
        l3embedding_model = load_embedding(model_path,
                                           model_type,
                                           'audio', pooling_type,
                                           tgt_num_gpus=num_gpus)
    else:
        l3embedding_model = None


    dataset_output_dir = os.path.join(output_dir, model_id,
                                      model_type, pooling_type, features)

    if not os.path.isdir(dataset_output_dir):
        os.makedirs(dataset_output_dir)

    # Write configurations to a file for reproducibility/posterity
    config_path = os.path.join(dataset_output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(args, f)
    LOGGER.info('Saved configuration to {}'.format(config_path))

    if dataset_name == 'us8k':

        if fold_num is not None:
            # Generate a single fold if a fold was specified
            generate_us8k_fold_data(metadata_path, data_dir, fold_num-1, dataset_output_dir,
                l3embedding_model=l3embedding_model,
                features=features, label_format=label_format, random_state=random_state,
                hop_size=hop_size, num_random_samples=num_random_samples)

        else:
            # Otherwise, generate all the folds
            generate_us8k_folds(metadata_path, data_dir, dataset_output_dir,
                l3embedding_model=l3embedding_model,
                features=features, label_format=label_format, random_state=random_state,
                hop_size=hop_size, num_random_samples=num_random_samples)
    else:
        LOGGER.error('Invalid dataset name: {}'.format(dataset_name))

    LOGGER.info('Done!')
