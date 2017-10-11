
import pescador
import pandas as pd
from tqdm import tqdm


def sampler(filename, file_list):
    '''
    Sample audio/video fromthe filename, with 50% change of using pairs from
    the same file and 50% chance of mixing audio/video from another file
    chosen at random from the file_list.

    Parameters
    ----------
    filename
    file_list

    Returns
    -------

    '''


def data_generator(csv_file, batch_size=64):
    '''
    - Load up CSV file
    - Iterate over all training files
    - Create a streamer per traning file

    Parameters
    ----------
    csv_file

    Returns
    -------

    '''

    seeds = []

    for track in tqdm(tracks):
        fname = os.path.join(working,
                             os.path.extsep.join([str(track), 'h5']))
        seeds.append(pescador.Streamer(sampler, fname, file_list))

    # Send it all to a mux
    mux = pescador.Mux(seeds, k, **kwargs)

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