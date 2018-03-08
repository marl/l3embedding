import csv
import os
import glob
import logging
import random
from audioset.ontology import ASOntology
from collections import OrderedDict
from data.utils import read_csv_as_dicts

LOGGER = logging.getLogger('data')
LOGGER.setLevel(logging.DEBUG)

def get_filename(path):
    """Return the filename of a path

    Args: path: path to file

    Returns:
        filename: name of file (without extension)
    """
    return os.path.splitext(os.path.basename(path))[0]


def get_ytid_from_filename(filename):
    first_us_idx = filename.rindex('_')
    second_us_idx = filename.rindex('_', 0, first_us_idx)
    return filename[:second_us_idx]


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


def get_file_list(data_dir, metadata_path=None, filter_path=None, ontology_path=None):
    """Return audio and video file list.

    Args:
        data_dir: input directory that contains audio/ and video/

    Keyword Args:
        metadata_path: Path to audioset metadata file
        filter_path: Path to filter specification file
        ontology_path: Path to AudioSet ontology file

    Returns:
        audio_files: list of audio files
        video_files: list of video files

    """
    data_dir_contents = set(os.listdir(data_dir))
    LOGGER.info('Getting list of files...')
    if 'audio' in data_dir_contents and 'video' in data_dir_contents:
        audio_files = glob.glob('{}/audio/*'.format(data_dir))
        video_files = glob.glob('{}/video/*'.format(data_dir))
    else:
        audio_files = glob.glob('{}/**/audio/*'.format(data_dir))
        video_files = glob.glob('{}/**/video/*'.format(data_dir))

    # Make sure that audio files and video files correspond to each other
    audio_filenames = set([get_filename(path) for path in audio_files])
    video_filenames = set([get_filename(path) for path in video_files])

    valid_filenames = audio_filenames & video_filenames

    # Get map from filename to full audio and video paths
    audio_paths = {get_filename(path): path for path in audio_files
                                       if get_filename(path) in valid_filenames}
    video_paths = {get_filename(path): path for path in video_files
                                       if get_filename(path) in valid_filenames}

    if metadata_path:
        LOGGER.info('Loading metadata...')
        if not ontology_path:
            err_msg = 'Must provide ontology path to filter'
            LOGGER.error(err_msg)
            raise ValueError(err_msg)

        metadata = load_metadata(metadata_path)
        ontology = ASOntology(ontology_path)

    file_list = []
    for filename in valid_filenames:
        ytid = get_ytid_from_filename(filename)
        video_metadata = metadata[ytid]

        audio_path = audio_paths[filename]
        video_path = video_paths[filename]

        item = OrderedDict()
        item['ytid'] = ytid
        item['audio_filepath'] = audio_path
        item['video_filepath'] = video_path

        # Add the labels if the metadata is provided
        if metadata_path:
            item['labels'] = [ontology.get_node(label_id).name.lower()
                              for label_id in video_metadata['positive_labels']]

        file_list.append(item)


    if metadata_path and filter_path:
        LOGGER.info('Filtering examples...')
        filters = read_csv_as_dicts(filter_path)

        filtered_file_list = []

        for item in file_list:
            ytid = item['ytid']
            label_list = item['labels']

            accept = None
            has_accept_filter = False
            for _filter in filters:
                filter_type = _filter['filter_type']
                filter_accept = _filter['accept_reject'].lower() == 'accept'
                string = _filter['string']
                include_children = _filter['include_children'].lower() == 'true'

                if filter_accept:
                    has_accept_filter = True

                if filter_type == 'ytid':
                    match = ytid == string

                elif filter_type == 'label':
                    match = string.lower() in label_list

                    if include_children and not match:
                        # If this filter includes children classes,
                        # check each label to see if it is a descendent of the
                        # filter label
                        filter_node = ontology.get_node_by_name(string)
                        for label in label_list:
                            label_node = ontology.get_node_by_name(string)
                            if filter_node.is_child(label_node):
                                match = True
                                break

                if filter_accept:
                    # If "accept" has not been set, and there is a match with an accept filter, set "accept" to True
                    # -> If "accept" has already been set (which is True if another accept filter has already approved the
                    #    file or if a reject filter has rejected the file already, though this )
                    if accept is None and match:
                        accept = True
                elif not filter_accept and match:
                    # If a reject filter gets a match, do not accept file and break from the loop
                    accept = False
                    break

            if accept is None:
                accept = not has_accept_filter

            if accept:
                #LOGGER.debug('Using video: "{}"'.format(filename))
                filtered_file_list.append(item)


        file_list = filtered_file_list

    LOGGER.info('Total videos used: {}'.format(len(file_list)))
    return file_list


def get_subset_split(data_dir, valid_ratio=0.1, test_ratio=0.1, random_state=12345678,
                     metadata_path=None, filter_path=None, ontology_path=None):
    # Set random seed for reproducability
    random.seed(random_state)

    file_list = get_file_list(data_dir, metadata_path=metadata_path,
                              filter_path=filter_path, ontology_path=ontology_path)

    # Shuffle file list
    random.shuffle(file_list)

    # Figure out number of files for each subset
    num_files = len(file_list)
    num_valid = int(num_files * valid_ratio)
    num_test = int(num_files * test_ratio)

    # Get subset lists
    valid_list = file_list[:num_valid]
    test_list = file_list[num_valid:num_valid+num_test]
    train_list = file_list[num_valid+num_test:]

    return train_list, valid_list, test_list
