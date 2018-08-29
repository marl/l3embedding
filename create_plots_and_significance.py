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
            data2[ident] = entry['test_acc']

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


def compute_stat_test(data2):
    keys = data2.keys()

    models = set()
    folds = set()
    a = {}

    for k in keys:
        ident = k[1:]
        fold = k[0]
        models.add(ident)
        folds.add(fold)

        if ident not in a:
            a[ident] = {}

        a[ident][fold] = data2[k]

    folds = list(sorted(list(folds)))
    models = list(models)

    num_folds = len(folds)

    measurements = {}

    for k1, k2 in itertools.product(models, models):
        if k1 == k2:
            continue

        k1_acc = []
        k2_acc = []
        for fold in range(1, num_folds+1):
            k1_acc.append(a[k1][fold])
            k2_acc.append(a[k2][fold])

        try:
            T, p = scipy.stats.wilcoxon(k1_acc, k2_acc)
        except:
            import pdb
            pdb.set_trace()

        measurements[(k1, k2)] = {'T': T, 'p': p}
    return measurements

us8k_data, us8k_data2 = get_entries('us8k', 'classifier', first_row=3)
esc50_data, esc50_data2 = get_entries('esc50', 'ESC50', first_row=3)
dcase2013_data, dcase2013_data2 = get_entries('dcase2013', 'DCASE', first_row=3)

us8k_stat = compute_stat_test(us8k_data2)
esc50_stat = compute_stat_test(esc50_data2)

print("US8K-L3")
pprint(us8k_stat)
print("ESC50-L3")
pprint(esc50_stat)

font = {'size' : 18}
matplotlib.rc('font', **font)

fig, axarr = plt.subplots(3, figsize=(10, 13), sharex=True)

tick_labels = []
for model_name in ["L", "M128", "M256"]:
    for width in ["6K", "512"]:
        for subset in ["Env", "Mus"]:
            tick_labels.append("{}/{}/{}".format(model_name, width, subset))

for data, ax, dataset_name, tick_interval in zip([us8k_data, esc50_data, dcase2013_data], axarr, ['UrbanSound8K', 'ESC-50', 'DCASE 2013'], [0.02, 0.02, 0.01]):
    df = pandas.DataFrame(data)
    ax = df.boxplot(column='test_acc', by=['embedding_model_type', 'embedding_length', 'audioset_subset'], figsize=(12,14), showmeans=True, ax=ax)

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
    pal = sns.color_palette("Paired", n_colors=10)
    colors = []
    for c in pal[:8]:
        colors.append(c)
        colors.append(c)
#colors.extend(pal[-2:])
    children = ax.get_children()
    for idx in range(12):
        box = children[idx*8]
        xs = box.get_xdata()
        ys = box.get_ydata()
        ax.fill_between((xs[0], xs[1]), ys[0], ys[2], color=colors[idx] + (0.5,))


# remove legend()
#ax.legend().set_visible(False)

fig.tight_layout()
fig.savefig('us8k_test_boxplot.png')










l3_model = ('original', 'music', '4_mel2')
us8k_data, us8k_data2, = get_entries('us8k', 'classifier', first_row=3, l3_only=False, l3_model=l3_model)
esc50_data, esc50_data2 = get_entries('esc50', 'ESC50', first_row=3, l3_only=False, l3_model=l3_model)
dcase2013_data, dcase2013_data2 = get_entries('dcase2013', 'DCASE', first_row=3, l3_only=False, l3_model=l3_model)

us8k_stat = compute_stat_test(us8k_data2)
print("US8K-AllEmbeddings")
pprint(us8k_stat)

esc50_stat = compute_stat_test(esc50_data2)
print("ESC50-AllEmbeddings")
pprint(esc50_stat)


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
    ax.set_xticklabels(["L3-M256/6K/Mus", "SoundNet", "VGGish"], ha='right')
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
us8k_data += us8k_aug_data
us8k_data2.update({k + ('augment',): v for k, v in us8k_aug_data2.items()})

us8k_stat = compute_stat_test(us8k_data2)
print("Augmented Trials")
pprint(us8k_stat)


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
