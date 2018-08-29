import argparse
import logging
import os
from csv import DictWriter

from data.avc.subsets import get_subset_split
from log import init_console_logger

LOGGER = logging.getLogger('data')
LOGGER.setLevel(logging.DEBUG)


def write_subset_file(path, subset_list):
    with open(path, 'w') as f:
        field_names = list(subset_list[0].keys())
        writer = DictWriter(f, field_names)
        writer.writeheader()

        for item in subset_list:
            item = dict(item)
            item['labels'] = ';'.join(item['labels'])
            writer.writerow(item)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Creates CSVs containing a train-valid-test split for the given dataset')

    parser.add_argument('-vr',
                        '--valid-ratio',
                        dest='valid_ratio',
                        action='store',
                        type=float,
                        default=0.1,
                        help='Ratio of dataset used for validation set')

    parser.add_argument('-tr',
                        '--test-ratio',
                        dest='test_ratio',
                        action='store',
                        type=float,
                        default=0.1,
                        help='Ratio of dataset used for test set')

    parser.add_argument('-rs',
                        '--random-seed',
                        dest='random_seed',
                        action='store',
                        type=int,
                        default=12345678,
                        help='Random seed used for generating split')

    parser.add_argument('-o',
                        '--ontology-path',
                        dest='ontology_path',
                        action='store',
                        type=str,
                        default=os.path.join(os.path.dirname(__file__), 'resources/ontology.json'),
                        help='Path to AudioSet ontology')

    parser.add_argument('-mp',
                        '--metadata-path',
                        dest='metadata_path',
                        action='store',
                        type=str,
                        help='Path to metadata csv file(s). Accepts a glob string.')

    parser.add_argument('-fp',
                        '--filter-path',
                        dest='filter_path',
                        action='store',
                        type=str,
                        help='Path to filter csv file(s).')


    parser.add_argument('-r',
                        '--random-state',
                        dest='random_state',
                        action='store',
                        type=int,
                        default=20171021,
                        help='Random seed used to set the RNG state')

    parser.add_argument('data_dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored')

    parser.add_argument('output_dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')

    parser.add_argument('filename_prefix',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')

    return parser.parse_args()


if __name__ == '__main__':
    init_console_logger(LOGGER, verbose=True)

    args = parse_arguments()

    train_list, valid_list, test_list \
        = get_subset_split(args.data_dir,
                           valid_ratio=args.valid_ratio,
                           test_ratio=args.test_ratio,
                           random_state=args.random_state,
                           metadata_path=args.metadata_path,
                           filter_path=args.filter_path,
                           ontology_path=args.ontology_path)

    output_dir = args.output_dir
    filename_prefix = args.filename_prefix
    train_subset_path = os.path.join(output_dir, filename_prefix + '_train.csv')
    valid_subset_path = os.path.join(output_dir, filename_prefix + '_valid.csv')
    test_subset_path = os.path.join(output_dir, filename_prefix + '_test.csv')

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    write_subset_file(train_subset_path, train_list)
    write_subset_file(valid_subset_path, valid_list)
    write_subset_file(test_subset_path, test_list)
