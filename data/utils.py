import csv
from multiprocessing import Pool


def read_csv_as_dicts(path):
    items = []

    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(row)

    return items

def map_iterate_in_parallel(iterable, function, processes=8):
    pool = Pool(processes=processes)
    output = pool.map(function, iterable)
    return list(output)


def flatten_dict(dct, parent_key=None):
    new_dct = type(dct)()
    for k,v in dct.items():
        if parent_key is not None:
            k = '{}_{}'.format(parent_key, k)

        if isinstance(v, dict):
            new_dct.update(flatten_dict(v, parent_key=k))
        else:
            new_dct[k] = v

    return new_dct

