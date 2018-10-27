import argparse
import os
import pickle as pk
import numpy as np
import gzip
import io
from data.usc.folds import get_valid_fold_idx, load_feature_file, get_fold, get_train_validation_nonaugmented_split
from data.usc.features import expand_framewise_labels
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def np_savez(path, **data):
    with gzip.open(path, mode='wb') as f:
        buf = io.BytesIO()
        np.savez(buf, **data)
        f.write(buf.getbuffer())


def get_single_pass_train_folds_generator(feature_dir, test_fold_idx, num_folds, train_filenames_to_idxs,
        valid=True, batch_size=512):
    X_batch = []
    curr_batch_size = 0

    nonaugmented_files = set([os.path.basename(x)
                              for x in train_filenames_to_idxs.keys()])

    valid_fold_idx = get_valid_fold_idx(test_fold_idx, num_folds)

    # Get files
    for fold_idx in range(num_folds):
        if fold_idx == test_fold_idx or (valid and fold_idx == valid_fold_idx):
            continue

        fold_dir = os.path.join(feature_dir, 'fold{}'.format(fold_idx + 1))
        filenames = os.listdir(fold_dir)

        for feature_filename in filenames:
            feature_filepath = os.path.join(fold_dir, feature_filename)

            # Limit indices to just the non-valid set frames
            file_X, file_y = load_feature_file(feature_filepath)
            if feature_filename in nonaugmented_files:
                train_idxs = np.array(train_filenames_to_idxs[feature_filename])
                file_X = file_X[train_idxs]

            while file_X.shape[0] > 0:
                num_examples = file_X.shape[0]
                end_idx = min(num_examples, batch_size - curr_batch_size)

                X_batch.append(file_X[:end_idx])
                curr_batch_size += end_idx

                file_X = file_X[end_idx:]

                if curr_batch_size == batch_size:
                    X_batch = np.vstack(X_batch)

                    yield X_batch
                    X_batch = []
                    curr_batch_size = 0


def save_fold_standardizer(feature_dir, test_fold_idx,
        output_dir, train_filenames_to_idxs, valid=True, batch_size=512):
    data_gen = get_single_pass_train_folds_generator(feature_dir, test_fold_idx,
        10, train_filenames_to_idxs, valid=valid, batch_size=batch_size)

    std_scaler = StandardScaler()
    for X_batch in data_gen:
        std_scaler.partial_fit(X_batch)

    output_dir = os.path.join(output_dir, 'standardizers')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, 'fold{}'.format(test_fold_idx+1))

    with open(output_path, 'wb') as f:
        pk.dump(std_scaler, f)


def generate_shuffled_fold(feature_dir, fold_idx, output_dir, batch_size=512):
    print("Loading fold...")
    train_filenames_to_idxs, valid_data = get_train_validation_nonaugmented_split(
        feature_dir, fold_idx)
    """
    fold_data = get_fold(feature_dir, fold_idx, augment=True,
        train_filenames_to_idxs=train_filenames_to_idxs)
    output_fold_dir = os.path.join(output_dir, "train", 'fold{}'.format(fold_idx+1))
    if not os.path.exists(output_fold_dir):
        os.makedirs(output_fold_dir)

    valid_data_dir = os.path.join(output_dir, "valid")
    if not os.path.exists(valid_data_dir):
        os.makedirs(valid_data_dir)
    valid_fold_data_path = os.path.join(valid_data_dir, "fold{}.npz.gz".format(fold_idx+1))
    np_savez(valid_fold_data_path, **valid_data)

    del valid_data

    print("Preparing data...")
    expand_framewise_labels(fold_data)
    X = fold_data.pop('features')
    y = fold_data.pop('labels')
    fold_size = y.shape[0]
    shuffle_idxs = np.random.permutation(fold_size)
    X = X[shuffle_idxs]
    y = y[shuffle_idxs]

    print("Generating batches...")
    idx_list = np.arange(0, fold_size, batch_size)
    num_batches = len(idx_list)
    for batch_idx, start_idx in enumerate(idx_list):
        print("\tBatch {}/{}".format(batch_idx+1, num_batches))
        end_idx = min(start_idx + batch_size, fold_size)
        batch_path = os.path.join(output_fold_dir, '{}.npz.gz'.format(batch_idx))

        if os.path.exists(batch_path):
            continue
        X_batch = X[start_idx:end_idx]
        y_batch = y[start_idx:end_idx]
        np_savez(batch_path, X=X_batch, y=y_batch)
    """

    return train_filenames_to_idxs


def parse_arguments():
    parser = argparse.ArgumentParser(description='Make augmented training batches')
    parser.add_argument('feature_dir', help='Path to feature directory')
    parser.add_argument('output_dir', help='Path to output directory')
    parser.add_argument('fold', type=int, help='Fold to generate')
    parser.add_argument('--batch-size', type=int, default=512, help='Size of batches to save')
    args = parser.parse_args()
    return vars(args)


def main(feature_dir, output_dir, fold, batch_size=512):
    train_filenames_to_idxs = generate_shuffled_fold(feature_dir, fold-1, output_dir, batch_size=batch_size)
    save_fold_standardizer(feature_dir, fold-1, output_dir, train_filenames_to_idxs, batch_size=batch_size)


if __name__ == '__main__':
    main(**parse_arguments())
