# core stuff
import pandas as pd
import numpy  as np
import csv

# scipy whatever
import scipy
from scipy import stats
import statsmodels.api as sm

# plotty stuff
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")


# CONSTANTS
WARMUP_PERIOD  =   4   # not really used
NUM_ITERATIONS = 100
SIM_TIME       =  30  # not really used
NUM_USERS      =  10

SAMPLE_SIZE    = 1000 # not really used
SEED_SAMPLING  =   42 # not really used

# DATA PATHs
DATA_PATH = "./data/"

MODE_DESCRIPTION = {
    'bin' : "Binomial CQIs",
    'uni' : "Uniform CQIs",
    'bin_old' : "Binomial CQIs (old)"
}

LAMBDA_DESCRIPTION = {
    'l01' : "λ = 0.1ms",
    'l09' : "λ = 0.9ms",
    'l1'  : "λ = 1.0ms",
    'l13' : "λ = 1.3ms",
    'l14' : "λ = 1.4ms",
    'l15' : "λ = 1.5ms",
    'l2'  : "λ = 2.0ms",
    'l5'  : "λ = 5.0ms"
}

MODE_PATH = {
    'log' : "lognormal/",
    'exp' : "exponential/",
    'nonmon' : "non_monitoring/"
}

LAMBDA_PATH = {
    'l01' : "lambda01/",
    'l09' : "lambda09/",
    'l1'  : "lambda1/",
    'l13' : "lambda13/",
    'l14' : "lambda14/",
    'l15' : "lambda15/",
    'l2'  : "lambda2/",
    'l5'  : "lambda5/"
}

CSV_PATH = {
    'sca' : "sca_res.csv",
    'vec' : "vec_res.csv"
}

CQI_CLASSES = [
    'LOW',
    'HIGH'
]

# Just to not fuck things up
np.random.seed(SEED_SAMPLING)

####################################################
#                       UTIL                       #
####################################################


####################################################
#                      PARSER                      #
####################################################

def parse_if_number(s):
    try:
        return float(s)
    except:
        return True if s == "true" else False if s == "false" else s if s else None


def parse_ndarray(s):
    return np.fromstring(s, sep=' ') if s else None


def parse_name_attr(s):
    return s.split(':')[0] if s else None


def parse_run(s):
    return int(s.split('-')[1]) if s else None


def vector_parse(cqi, pkt_lambda):
    path_csv = DATA_PATH + MODE_PATH[cqi] + LAMBDA_PATH[pkt_lambda] + CSV_PATH['vec']

    # vec files are huge, try to reduce their size ASAP!!
    data = pd.read_csv(path_csv,
                       delimiter=",", quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8',
                       usecols=['run', 'type', 'module', 'name', 'vecvalue', 'vectime'],
                       converters={
                           'run': parse_run,
                           'vectime': parse_ndarray,  # i guess
                           'vecvalue': parse_ndarray,
                           'name': parse_name_attr
                       }
                       )

    # remove useless rows
    data = data[data.type == 'vector']
    data.reset_index(inplace=True, drop=True)

    # rename vecvalue for simplicity...
    data = data.rename({'vecvalue': 'value', 'vectime': 'time'}, axis=1)
    return data[['run', 'name', 'time', 'value']].sort_values(['run', 'name'])


# Parse CSV file
def scalar_parse(cqi, pkt_lambda):
    path_csv = DATA_PATH + MODE_PATH[cqi] + LAMBDA_PATH[pkt_lambda] + CSV_PATH['sca']
    data = pd.read_csv(path_csv,
                       usecols=['run', 'type', 'name', 'value'],
                       converters={
                           'run': parse_run,
                           'name': parse_name_attr
                       }
                       )

    # remove useless rows (first 100-ish rows)
    data = data[data.type == 'scalar']
    data.reset_index(inplace=True, drop=True)

    # data['user'] = data.name.apply(lambda x: x.split['-'][1] if '-' in x else 'global')

    return data[['run', 'name', 'value']].sort_values(['run', 'name'])


def describe_attribute_sca(data, name, value='value'):
    # print brief summary of attribute name (with percentiles and stuff)
    print(data[data.name == name][value].describe(percentiles=[.25, .50, .75, .95]))
    return


def describe_attribute_vec(data, name, iteration=0):
    values = pd.Series(data[data.name == name].value.iloc[iteration])
    print(values.describe(percentiles=[.25, .50, .75, .95]))
    return


def vector_stats(data, group=False):
    # compute stats for each iteration
    stats = pd.DataFrame()
    stats['name'] = data.name
    stats['run'] = data.run
    stats['mean'] = data.value.apply(lambda x: x.mean())
    stats['max'] = data.value.apply(lambda x: x.max())
    stats['min'] = data.value.apply(lambda x: x.min())
    stats['std'] = data.value.apply(lambda x: x.std())
    stats['count'] = data.value.apply(lambda x: x.size)
    return stats.groupby(['name']).mean().drop('run', axis=1) if group else stats


def aggregate_users_signals(data, signal, users=range(0, NUM_USERS)):
    # meanResponseTime => dato il mean response time di un utente per ogni run, calcolo la media dei
    # mean response time dell'utente su tutte le run. E poi faccio la media per tutti gli utenti per
    # ottenere il mean responsetime medio per tutti gli utenti.
    return data[data.name.isin([signal + '-' + str(i) for i in users])].groupby('run').mean().describe(
        percentiles=[.25, .50, .75, .95])


def scalar_stats(data, attr=None, users=range(0, NUM_USERS)):
    stats = pd.DataFrame()
    attributes = data.name.unique() if attr is None else attr

    # STATS FOR EACH SIGNAL
    for attr in attributes:
        stats[attr] = data[data.name == attr].value.describe(percentiles=[.25, .50, .75, .95])

    # Aggregate dynamic stats (one signal per user):
    stats['meanResponseTime'] = aggregate_users_signals(data, 'responseTime', users)
    stats['meanThroughput'] = aggregate_users_signals(data, 'tptUser', users)
    stats['meanCQI'] = aggregate_users_signals(data, 'CQI', users)
    stats['meanNumberRBs'] = aggregate_users_signals(data, 'numberRBs', users)

    # Transpose...
    stats = stats.T

    # COMPUTE CI
    stats['ci95_l'] = stats['mean'] - 1.96 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci95_h'] = stats['mean'] + 1.96 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci99_l'] = stats['mean'] - 2.58 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci99_h'] = stats['mean'] + 2.58 * (stats['std'] / np.sqrt(stats['count']))
    return stats


def users_bandwidth_sca(data, group=False):
    stats = scalar_stats(data)
    index = [row for row in stats.index if row.startswith('tptUser-')]
    sel = stats.loc[index, :].reset_index()

    bandwidth = pd.DataFrame()
    bandwidth['user'] = sel['index'].str.split('-', expand=True)[1].astype(int)
    bandwidth['mean_Mbps'] = (sel['mean'] * 1000) / 125000
    bandwidth['max_Mbps'] = (sel['max'] * 1000) / 125000
    bandwidth['min_Mbps'] = (sel['min'] * 1000) / 125000

    bandwidth.index = bandwidth['user']
    bandwidth = bandwidth.drop('user', axis=1)
    return bandwidth

def main():
    print("\n\nPerformance Evaluation - Python Data Analysis\n")
    return

if __name__ == '__main__':
    main()


