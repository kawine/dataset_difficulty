import pandas as pd
from scipy.special import softmax
import json
import numpy as np
import os
from scipy.stats import pearsonr


DATA = {}


def load_data():
    global DATA

    if DATA == {}:
        print("Loading data ...")
        for fn in os.listdir('PVI'):
            print(fn)
            model, suffix = fn[:-4].split('_', 1)
            
            if model not in DATA:
                DATA[model] = {}

            DATA[model][suffix] = pd.read_csv('PVI/' + fn)
            DATA[model][suffix] = DATA[model][suffix][DATA[model][suffix]['label'] >= 0]

    return DATA


def VI():
    """
    Return the v-info for every model for every standard setting.
    """
    load_data()
    table = DATA.copy()
    epochs = [ 'std', 'std2', 'std3', 'std10' ]
    
    for i in table:
        table[i] = { j : table[i][j]['PVI'].mean() for j in epochs }

    return pd.DataFrame.from_dict(table)


def VI_transform():
    """
    Return the v-info for every input transformation.
    """
    load_data()
    table = DATA.copy()
    transformations = [ 'std', 'random_order', 'length_only', 'hyp_only', 'prem_only', 'overlap_only' ]
    
    for i in table:
        table[i] = { j : table[i][j]['PVI'].mean() for j in transformations }

    return pd.DataFrame.from_dict(table)


def VI_transform_by_class(model='bert-base-cased'):
    """
    Return the v-info for every input transformation.
    """
    load_data()
    table = {}
    transformations = ['snli_std2_test', 'snli_shuffled_test', 'snli_hypothesis_test', 'snli_premise_test', 'snli_raw_overlap_test' ]
    
    for i in [0,1,2]:
        table[i] = { j : DATA[model][j][DATA[model][j]['label'] == i]['PVI'].mean() for j in transformations }

    return pd.DataFrame.from_dict(table)


def accuracy():
    """
    Return the accuracy for every model for every setting.
    """
    load_data()
    table = DATA.copy()
    epochs = [ 'std', 'std2', 'std3', 'std10' ]

    for i in table:
        table[i] = { j : table[i][j]['correct_yx'].mean() for j in epochs }

    return pd.DataFrame.from_dict(table)
        

def inter_model_corr(suffix="std"):
    """
    Return pearson correlation between the model-dependent PVI values for the setup defined by suffix.
    """
    load_data()
    table = {} 

    for i in DATA:
        table[i] = { j: pearsonr(DATA[i][suffix]['PVI'], DATA[j][suffix]['PVI'])[0] for j in DATA }

    return pd.DataFrame.from_dict(table)


def intra_model_corr(model, dataset='snli'):
    """
    Return pearson correlation between the PVI values of the model across epochs 1, 2, 3, and 10.
    """
    load_data()
    table = {}

    if dataset == 'snli':
        epochs = [ "snli_std_test", "snli_std2_test", "snli_std3_test", "snli_std5_test", "snli_std10_test" ]
    elif dataset == 'cola':
        epochs = [ f'cola_id_dev_std{i}' for i in [1,2,3,5,10] ]

    for i in epochs:
        table[i] = { j: pearsonr(DATA[model][i]["PVI"], DATA[model][j]["PVI"])[0] for j in epochs }

    return pd.DataFrame.from_dict(table)

    
def seed_corr(model):
    """
    Return pearson correlation between the PVI values of the model across different seeds.
    """
    load_data()
    table = {}
    seeds = [ f"snli_std_test_{seed}_0.99" for seed in ['b','c','d','e'] ]

    for i in seeds:
        table[i] = { j: pearsonr(DATA[model][i]["PVI"], DATA[model][j]["PVI"])[0] for j in seeds }
        
    return pd.DataFrame.from_dict(table)


def breakdown_by_class(suffix="std"):
    """
    Break down the VI by class.
    """
    load_data()
    table = {}

    for m in DATA:
        x = DATA[m][suffix]
        table[m] = { i: x[x['label'] == i]['PVI'].mean() for i in [0,1,2] }
        table[m]['ALL'] = x['PVI'].mean()

    return pd.DataFrame.from_dict(table)


def breakdown_by_correctness(suffix="std"):
    """
    Break down the VI by correctness.
    """
    load_data()
    table = {}

    for m in DATA:
        x = DATA[m][suffix]
        table[m] = { i: x[x['correct_yx'] == i]['PVI'].mean() for i in [0,1] }
        table[m]['ALL'] = x['PVI'].mean()

    return pd.DataFrame.from_dict(table)


def breakdown_by_agreement(suffix="std"):
    load_data()
    table = {}
    agreement_rates = []

    for row in open('data/snli_1.0/snli_1.0_test.jsonl'):
        row = json.loads(row)
        rate = sum(l == row['gold_label'] for l in row['annotator_labels']) / len(row['annotator_labels'])
        agreement_rates.append(rate)

    for m in DATA:
        x = DATA[m][suffix]
        x['agreement'] = agreement_rates
        table[m] = { f"[{round(i, 2)}, {round(i+0.2, 2)}]": x[(i <= x['agreement']) & (x['agreement'] < i+0.2)]['PVI'].mean() for i in np.arange(0.5, 1.01, 0.2) }

    return pd.DataFrame.from_dict(table)


def rerank(suffix="std", max_PVI=0.2):
    load_data()
    table = {}
    avg_PVI = np.mean([ DATA[m][suffix]['PVI'] for m in DATA ], axis=0)
    avg_PVI = np.array([ (x if x < max_PVI else np.inf) for x in avg_PVI ])

    for m in DATA:
        table[m] = { 'acc': DATA[m][suffix]["correct_yx"].mean() }

        for alpha in [0.5, 1.0, 1.5, 2.0]:
            weights = softmax(alpha * -1 * avg_PVI)
            table[m][f'acc (alpha = {alpha})'] = (DATA[m][suffix]["correct_yx"] * weights).sum()

    return pd.DataFrame.from_dict(table)


def compare_train_and_test():
    load_data()
    table = {}

    for m in DATA:
        table[m] = { 'test' : DATA[m]['std']['PVI'].mean(), 'train' : DATA[m]['std_train']['PVI'].mean() }

    return pd.DataFrame.from_dict(table)
