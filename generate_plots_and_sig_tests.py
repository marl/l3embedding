from gsheets import get_credentials, get_row, append_row, update_experiment, request_with_retry
from googleapiclient import discovery
from collections import Counter
import pickle as pk
import os
import numpy as np
import scipy.stats
import shutil
import json
import time
import pandas
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from pprint import pprint

CLASSIFIER_FIELD_NAMES = [
    'username',
    'model_id',
    'model_dir',
    'git_commit',
    'features_dir',
    'fold_num',
    'parameter_search',
    'parameter_search_valid_fold',
    'parameter_search_valid_ratio',
    'parameter_search_train_with_valid',
    'model_type',
    'feature_mode',
    'train_batch_size',
    'non_overlap',
    'random_state',
    'num_epochs',
    'learning_rate',
    'weight_decay',
    'C',
    'tol',
    'max_iterations',
    'train_loss',
    'valid_loss',
    'train_acc',
    'valid_acc',
    'train_avg_class_acc',
    'valid_avg_class_acc',
    'train_class_acc',
    'valid_class_acc',
    'test_acc',
    'test_avg_class_acc',
    'test_class_acc',
]

def append_row(service, spreadsheet_id, param_dict, sheet_name):
    # The A1 notation of a range to search for a logical table of data.
    # Values will be appended after the last row of the table.
    range_ = '{}!A1:A{}'.format(sheet_name, len(FIELD_NAMES))
    # How the input data should be interpreted.
    value_input_option = 'USER_ENTERED'
    # How the input data should be inserted.
    insert_data_option = 'INSERT_ROWS'

    value_range_body = {
        "range": range_,
        "majorDimension": 'ROWS',
        "values": [[str(param_dict[field_name]) for field_name in FIELD_NAMES ]]
    }

    request = service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=range_,
        valueInputOption=value_input_option,
        insertDataOption=insert_data_option,
        body=value_range_body)
    response = request_with_retry(request)


value_render_option = 'UNFORMATTED_VALUE'
date_time_render_option = 'FORMATTED_STRING'
GOOGLE_DEV_APP_NAME = 'l3embeddingexperiments'
GSHEET_ID = '1eyFv_jUWJeEuG1VKQNfAmhQUREMRDgEMxfVDffRaHDo'

credentials = get_credentials(GOOGLE_DEV_APP_NAME)
service = discovery.build('sheets', 'v4', credentials=credentials)

def get_entries(target_dataset, sheet_name, first_row=3, l3_only=True, l3_model=None):
    range_ = '{}!A{}:AF'.format(sheet_name, first_row)
    request = service.spreadsheets().values().get(spreadsheetId=GSHEET_ID, range=range_, valueRenderOption=value_render_option, dateTimeRenderOption=date_time_render_option)
    response = request.execute()

    unique = Counter()

    if target_dataset == 'us8k':
        limit = 2
    elif target_dataset == 'esc50':
        limit = 5
    else:
        limit = 1

    data2 = {}
    data = []
    for datum in response['values']:
        if not datum:
            continue
        model_id = datum[1]
        if target_dataset not in model_id:
            continue
        if datum[21] == '-':
            continue
        if l3_only and 'L3' not in model_id:
            continue

        entry = {k: v for k,v in zip(CLASSIFIER_FIELD_NAMES, datum)}

        if 'L3' in model_id:
            dataset, _, embedding_length, audioset_subset, embedding_model_type, \
                    _, _, _, _ = model_id.split('/')

            if embedding_model_type == 'cnn_L3_orig':
                embedding_model_type = '1_orig'
                continue
            elif embedding_model_type == 'cnn_L3_kapredbinputbn':
                embedding_model_type = '2_norm'
            elif embedding_model_type == 'cnn_L3_melspec1':
                embedding_model_type = '3_mel1'
            elif embedding_model_type == 'cnn_L3_melspec2':
                embedding_model_type = '4_mel2'
        else:
            dataset, embedding_model_type, _, _, _, _ = model_id.split('/')
            audioset_subset = 'na'
            embedding_length = 'na'


        fold_num = datum[5]
        ident = (fold_num, dataset, embedding_length, audioset_subset, embedding_model_type)


        if not l3_only and 'L3' in model_id and l3_model and ident[2:] != l3_model:
            continue

        entry['dataset'] = dataset
        entry['embedding_length'] = embedding_length
        entry['audioset_subset'] = audioset_subset
        entry['embedding_model_type'] = embedding_model_type

        if unique[ident] < limit:
            unique[ident] += 1
            data.append(entry)
            if ident not in data2:
                data2[ident] = []
            data2[ident].append(entry['test_acc'])

    for ident, count in unique.items():
        if target_dataset == 'us8k':
            if count != 2:
                print('Missing trials for {}: {}'.format(ident, count))
        elif target_dataset == 'esc50':
            if count != 5:
                print('Missing trials for {}: {}'.format(ident, count))
        else:
            if count != 1:
                print('Missing trials for {}: {}'.format(ident, count))


    return data, data2

pretty_print_map = {
    'us8k': 'US8K',
    'esc50': 'ESC50',
    'original': '6K',
    'short': '512',
    '2_norm': 'Linear',
    '3_mel1': 'Mel-128',
    '4_mel2': 'Mel-256',
    'augment': 'Augmented',
    'music': 'Music',
    'environmental': 'Environmental'
}

def get_dataset(ident):
    if 'us8k' in ident:
        dataset = 'US8K'
    elif 'esc50' in ident:
        dataset = 'ESC50'
    elif 'dcase2013' in ident:
        dataset = 'DCASE2013'
    else:
        dataset = None
    return dataset


def get_embedding_size(ident):
    if 'original' in ident:
        emb_size = '6K'
    elif 'short' in ident:
        emb_size = '512'
    else:
        emb_size = None

    return emb_size

def get_subset(ident):
    if 'music' in ident:
        subset = 'Music'
    elif 'environmental' in ident:
        subset = 'Environmental'
    else:
        subset = None
    return subset

def get_tfrepr(ident):
    if '2_norm' in ident:
        tfrepr = 'Linear'
    elif '3_mel1' in ident:
        tfrepr = 'Mel-128'
    elif '4_mel2' in ident:
        tfrepr = 'Mel-256'
    elif 'vggish' in ident:
        tfrepr = "VGGish"
    elif 'soundnet' in ident:
        tfrepr = 'SoundNet'
    else:
        tfrepr = None

    return tfrepr


def get_print_parts(ident1, ident2):
    intersection = set(ident1[1:]) & set(ident2[1:])


    intersection_tuple = []
    int_emb_size = get_embedding_size(intersection)
    int_subset = get_subset(intersection)
    int_tfrepr = get_tfrepr(intersection)
    if int_emb_size:
        intersection_tuple.append(int_emb_size)
    if int_subset:
        intersection_tuple.append(int_subset)
    if int_tfrepr:
        intersection_tuple.append(int_tfrepr)

    model1_list = []
    model1 = set(ident1) - set(ident2)
    mod1_emb_size = get_embedding_size(model1)
    mod1_subset = get_subset(model1)
    mod1_tfrepr = get_tfrepr(model1)
    mod1_augmented = 'Augmented' if 'augmented' in model1 else None
    if mod1_emb_size:
        model1_list.append(mod1_emb_size)
    if mod1_subset:
        model1_list.append(mod1_subset)
    if mod1_tfrepr:
        model1_list.append(mod1_tfrepr)
    if mod1_augmented:
        model1_list.append(mod1_augmented)

    model2_list = []
    model2 = set(ident2) - set(ident1)
    mod2_emb_size = get_embedding_size(model2)
    mod2_subset = get_subset(model2)
    mod2_tfrepr = get_tfrepr(model2)
    mod2_augmented = 'Augmented' if 'augmented' in model2 else None
    if mod2_emb_size:
        model2_list.append(mod2_emb_size)
    if mod2_subset:
        model2_list.append(mod2_subset)
    if mod2_tfrepr:
        model2_list.append(mod2_tfrepr)
    if mod2_augmented:
        model2_list.append(mod2_augmented)


    return tuple(intersection_tuple), model1_list, model2_list


def compute_stat_test(data2, desc, var=None, dcase=False):
    keys = data2.keys()

    models = set()
    folds = set()
    a = {}

    if var == 'dataset':
        ident_idx = 1
    elif var == 'embedding_length':
        ident_idx = 2
    elif var == 'audioset_subset':
        ident_idx = 3
    elif var == 'embedding_model_type':
        ident_idx = 4

    if var:
        print(var)

    for k in keys:
        if var:
            ident = (k[ident_idx],)
        else:
            ident = k[1:]
        fold = k[0]
        models.add(ident)
        folds.add(fold)

        if ident not in a:
            a[ident] = {}

        if fold not in a[ident]:
            a[ident][fold] = []

        a[ident][fold] += data2[k]

    folds = list(sorted(list(folds)))
    models = list(set(models))

    num_folds = len(folds)

    if var:
        dataset = get_dataset(k[1:])
    else:
        dataset = get_dataset(models[0])

    print("{} - {}".format(dataset, desc))
    measurements = set()

    for k1, k2 in itertools.product(models, models):
        if k1 == k2:
            continue

        pair = tuple(sorted((k1, k2)))
        if pair in measurements:
            continue

        k1, k2 = pair
        k1_acc = []
        k2_acc = []
        if not dcase:
            for fold in range(1, num_folds+1):
                k1_acc += a[k1][fold]
                k2_acc += a[k2][fold]
        else:
            k1_acc += a[k1][2]
            k2_acc += a[k2][2]
        T, p = scipy.stats.wilcoxon(k1_acc, k2_acc)

        if p < 0.05:
            int_tuple, mod1_list, mod2_list = get_print_parts(*pair)

            k1_mean = np.mean(k1_acc)
            k2_mean = np.mean(k2_acc)
            diff = k1_mean - k2_mean

            if diff < 0:
                tmp = mod2_list
                mod2_list = mod1_list
                mod1_list = tmp

                tmp = k2_mean
                k2_mean = k1_mean
                k1_mean = tmp

            print("({}): [{}] better than [{}] by {} - {} = {} ~~ T={}, p={}".format(
                " + ".join(int_tuple),
                " + ".join(mod1_list),
                " + ".join(mod2_list),
                k1_mean, k2_mean, np.abs(diff), T, p
            ))
        measurements.add(pair)
    print("\n\n")

        #measurements[pair] = {'T': T, 'p': p}
    #return measurements

us8k_data, us8k_data2 = get_entries('us8k', 'classifier', first_row=3)
esc50_data, esc50_data2 = get_entries('esc50', 'ESC50', first_row=3)
dcase2013_data, dcase2013_data2 = get_entries('dcase2013', 'DCASE', first_row=3)

compute_stat_test(us8k_data2, "L3", 'embedding_model_type')
compute_stat_test(us8k_data2, "L3", 'embedding_length')
compute_stat_test(us8k_data2, "L3", 'audioset_subset')
compute_stat_test(esc50_data2, "L3", 'embedding_model_type')
compute_stat_test(esc50_data2, "L3", 'embedding_length')
compute_stat_test(esc50_data2, "L3", 'audioset_subset')
#compute_stat_test(dcase2013_data2, "L3")

for var in ['embedding_model_type', 'embedding_length', 'audioset_subset']:
    font = {'size' : 18}
    matplotlib.rc('font', **font)

    fig, axarr = plt.subplots(3, figsize=(7, 10), sharex=True)

    tick_labels = []
    """
    for model_name in ["L", "M128", "M256"]:
        for width in ["6K", "512"]:
            for subset in ["Env", "Mus"]:
                tick_labels.append("{}/{}/{}".format(model_name, width, subset))
    """
    if var == 'embedding_model_type':
        tick_labels = ["Linear", "Mel-128", "Mel-256"]
    elif var == 'embedding_length':
        tick_labels = ["6144", "512"]
    else:
        tick_labels = ["Env.", "Music"]

    for data, ax, dataset_name, tick_interval in zip([us8k_data, esc50_data, dcase2013_data], axarr, ['UrbanSound8K', 'ESC-50', 'DCASE 2013'], [0.03, 0.02, 0.01]):
        df = pandas.DataFrame(data)
        ax = df.boxplot(column='test_acc', by=var, figsize=(7,9), showmeans=True, ax=ax)

        ax.set_xticklabels(tick_labels, ha='right')
        ax.xaxis.set_tick_params(rotation=45)
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, tick_interval))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
        ax.set_title(dataset_name)
        ax.set_xlabel('')
        ax.set_ylabel('Classification accuracy')
        fig.suptitle("")

    # Set colors
        pal = sns.color_palette("Set2", n_colors=10)
        colors = []
        for c in pal[:8]:
            colors.append(c)
            #colors.append(c)
    #colors.extend(pal[-2:])
        children = ax.get_children()
        if var == 'embedding_model_type':
            num_boxes = 3
        else:
            num_boxes = 2
        for idx in range(num_boxes):
            box = children[idx*8]
            xs = box.get_xdata()
            ys = box.get_ydata()
            ax.fill_between((xs[0], xs[1]), ys[0], ys[2], color=colors[idx] + (0.5,))


    # remove legend()
    #ax.legend().set_visible(False)

    fig.tight_layout()
    fig.savefig('us8k_test_boxplot_{}.png'.format(var))










l3_model = ('original', 'music', '3_mel1')
us8k_data, us8k_data2 = get_entries('us8k', 'classifier', first_row=3, l3_only=False, l3_model=l3_model)
esc50_data, esc50_data2 = get_entries('esc50', 'ESC50', first_row=3, l3_only=False, l3_model=l3_model)
dcase2013_data, dcase2013_data2 = get_entries('dcase2013', 'DCASE', first_row=3, l3_only=False, l3_model=l3_model)

compute_stat_test(us8k_data2, "All Embeddings")
compute_stat_test(esc50_data2, "All Embeddings")
compute_stat_test(dcase2013_data2, "All Embeddings", dcase=True)


"""
font = {'size' : 8}
matplotlib.rc('font', **font)

fig, axarr = plt.subplots(1, 3, figsize=(6, 5), sharey=True)

for data, ax, dataset_name, tick_interval in zip([us8k_data, esc50_data, dcase2013_data], axarr, ['UrbanSound8K', 'ESC-50', 'DCASE 2013'], [0.05, 0.05, 0.01]):
    df = pandas.DataFrame(data)
    ax = df.boxplot(column='test_acc', by=['embedding_model_type', 'embedding_length', 'audioset_subset'], showmeans=True, ax=ax)
    ax.set_xticklabels(["L3-M256/6K/Mus", "SoundNet", "VGGish"], ha='right')
    ax.xaxis.set_tick_params(rotation=45)
    start, end = ax.get_ylim()
    ax.yaxis.set_ticks(np.arange(start, end, tick_interval))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    ax.set_title(dataset_name)
    ax.set_xlabel('')
    fig.suptitle("")


# Set colors
    colors = sns.color_palette(None, n_colors=3)
#colors.extend(pal[-2:])
    children = ax.get_children()
    for idx in range(3):
        box = children[idx*8]
        xs = box.get_xdata()
        ys = box.get_ydata()
        ax.fill_between((xs[0], xs[1]), ys[0], ys[2], color=colors[idx] + (0.5,))

axarr[0].set_ylabel('Classification accuracy')

# remove legend()
#ax.legend().set_visible(False)

fig.tight_layout()
fig.savefig('us8k_test_boxplot_2.png')
"""















font = {'size' : 8}
matplotlib.rc('font', **font)

#fig, axarr = plt.subplots(3, figsize=(3, 6), sharex=True)
fig, axarr = plt.subplots(1, 3, figsize=(6, 4))

for data, ax, dataset_name, tick_interval, (start, end) in zip([us8k_data, esc50_data, dcase2013_data], axarr, ['UrbanSound8K', 'ESC-50', 'DCASE 2013'], [0.05, 0.05, 0.02], [(0.55, 0.9), (0.40, 0.85), (0.80, 1.)]):
    df = pandas.DataFrame(data)
    ax = df.boxplot(column='test_acc', by=['embedding_model_type', 'embedding_length', 'audioset_subset'], showmeans=True, ax=ax)
    ax.set_xticklabels(["L3-M128/6K/Mus", "SoundNet", "VGGish"], ha='right')
    ax.xaxis.set_tick_params(rotation=45)
    #tick_interval = 0.05
    #start, end = 0.4, 1.0
    ax.set_ylim(start, end)
    ax.yaxis.set_ticks(np.arange(start, end, tick_interval))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    ax.set_title(dataset_name)
    ax.set_xlabel('')
    #ax.set_ylabel('Classification accuracy')


# Set colors
    colors = sns.color_palette(None, n_colors=3)
#colors.extend(pal[-2:])
    children = ax.get_children()
    for idx in range(3):
        box = children[idx*8]
        xs = box.get_xdata()
        ys = box.get_ydata()
        ax.fill_between((xs[0], xs[1]), ys[0], ys[2], color=colors[idx] + (0.5,))

axarr[0].set_ylabel('Classification accuracy')
fig.suptitle("")

# remove legend()
#ax.legend().set_visible(False)

fig.tight_layout()
fig.savefig('us8k_test_boxplot_2.png')













l3_model = ('original', 'music', '4_mel2')
us8k_data, us8k_data2, = get_entries('us8k', 'classifier', first_row=3, l3_only=False, l3_model=l3_model)
us8k_aug_data, us8k_aug_data2, = get_entries('us8k', 'classifier_augmented', first_row=3, l3_only=False, l3_model=l3_model)

[x.update({'augmented': "false"}) for x in us8k_data]
[x.update({'augmented': "true"}) for x in us8k_aug_data]

us8k_data2 = {k:v for k,v in us8k_data2.items() if k[1:] == ('us8k', 'original', 'music', '4_mel2')}
us8k_data = [x for x in us8k_data if 'l3' in x['model_id']]
us8k_data += us8k_aug_data
us8k_data2.update({k + ('augment',): v for k, v in us8k_aug_data2.items()})

compute_stat_test(us8k_data2, "Augmented Trials")


font = {'size' : 8}
matplotlib.rc('font', **font)

fig = plt.figure(figsize=(4, 1.5))

df = pandas.DataFrame(us8k_data)
ax = df.boxplot(column='test_acc', by=['augmented'], showmeans=True, ax=fig.gca())
ax.set_xticklabels(["L3-M256/6K/Mus", "L3-M256/6K/Mus-Aug"])
start, end = ax.get_ylim()
ax.yaxis.set_ticks(np.arange(start, end, 0.04))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
ax.set_title("")
ax.set_xlabel('')
ax.set_ylabel('Classification accuracy')
fig.suptitle("")


# Set colors
colors = sns.color_palette(None, n_colors=3)
#colors.extend(pal[-2:])
children = ax.get_children()
for idx in range(2):
    box = children[idx*8]
    xs = box.get_xdata()
    ys = box.get_ydata()
    ax.fill_between((xs[0], xs[1]), ys[0], ys[2], color=colors[idx] + (0.5,))


# remove legend()
#ax.legend().set_visible(False)

fig.tight_layout()
fig.savefig('us8k_test_boxplot_3.png')























us8k_epoch_df = pandas.read_csv('epoch_us8k.csv')
esc50_epoch_df = pandas.read_csv('epoch_esc50.csv')

tick_labels = ['0.26M', '2.62M', '5.24M', '13.11M', '26.21M', '39.32M', '52.43M', '65.54M', '78.64M']

font = {'size' : 8}
matplotlib.rc('font', **font)

fig, axarr = plt.subplots(2, 1, figsize=(5, 5), sharex=True)

for df, ax, dataset_name, tick_interval, (start, end) in zip([us8k_epoch_df, esc50_epoch_df], axarr, ['UrbanSound8K', 'ESC-50'], [0.05, 0.05], [(0.55, 0.9), (0.55, 0.9)]):
    ax = df.boxplot(column='test_acc', by='# of samples', showmeans=True, ax=ax)
    ax.set_xticklabels(tick_labels, ha='right')
    ax.xaxis.set_tick_params(rotation=45)
    ax.set_ylim(start, end)
    ax.yaxis.set_ticks(np.arange(start, end, tick_interval))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
    ax.set_title(dataset_name)
    ax.set_xlabel('')
    ax.set_ylabel('Classification accuracy')

    # Set colors
    colors = sns.color_palette(None, n_colors=1)
    #colors.extend(pal[-2:])
    children = ax.get_children()
    for idx in range(len(tick_labels)):
        box = children[idx*8]
        xs = box.get_xdata()
        ys = box.get_ydata()
        ax.fill_between((xs[0], xs[1]), ys[0], ys[2], color=colors[0] + (0.5,))

fig.suptitle("")

fig.tight_layout()
fig.savefig('us8k_test_boxplot_4.png')
