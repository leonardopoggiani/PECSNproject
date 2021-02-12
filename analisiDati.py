import argparse
import csv
import math
import matplotlib.pyplot as plt
import numpy  as np
import pandas as pd
# import px
import pylab
import seaborn as sns
from scipy.stats import linregress as regr
import pprint
from pylab import *
import matplotlib.pyplot as plt
import seaborn as sns

color = sns.color_palette()
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px
from gekko import GEKKO

# import pandas_profiling

# seaborn settings, just to give a nicer look
sns.reset_defaults()
sns.set(
    rc={'figure.figsize': (7, 5)},
    style="white"
)

# matplotlib settings, when used
plt.rcParams['font.family'] = "serif"
plt.style.use('ggplot')

# CONSTANTS
WARMUP_PERIOD = 10000  # not really used
NUM_ITERATIONS = 5
SIM_TIME = 150000  # not really used
NUM_DATA_LINK = 1
NUM_AIRCRAFT = 1
SAMPLE_SIZE = 1000  # not really used
SEED_SAMPLING = 42  # not really used

# results csv path
DATA_PATH = "./simulations/results/"

# mode used for the creation of csv
MODE_DESCRIPTION = {
    'exp': "Exponential",
    'log': "Lognormal",
}

# interarrival time choosed
LAMBDA_DESCRIPTION = {
    'l01': "λ = 0.1ms",
    'l09': "λ = 0.9ms",
    'l1': "λ = 1.0ms",
    'l13': "λ = 1.3ms",
    'l14': "λ = 1.4ms",
    'l15': "λ = 1.5ms",
    'l2': "λ = 2.0ms",
    'l5': "λ = 5.0ms"
}

# to organize results
MODE_PATH = {
    'log': "lognormal/",
    'exp': "exponential/",
}

LAMBDA_PATH = {
    'l01': "lambda01/",
    'l09': "lambda09/",
    'l1': "lambda1/",
    'l13': "lambda13/",
    'l14': "lambda14/",
    'l15': "lambda15/",
    'l2': "lambda2/",
    'l5': "lambda5/"
}

CSV_PATH = {
    'sca': "sca_res.csv",
    'vec': "vec_res.csv"
}

CQI_CLASSES = [
    'LOW',
    'HIGH'
]

np.random.seed(SEED_SAMPLING)


####################################################
#                       PARSING                    #
####################################################
def parse_vector(s):
    return np.fromstring(s, sep=' ') if s else None


def parse_name(s):
    return s.split(':')[0] if s else None


def parse_run(s):
    return int(s.split('-')[2]) if s else None


def vector_parse(path_csv):
    data = pd.read_csv(path_csv,
                       delimiter=",", quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8',
                       usecols=['run', 'type', 'module', 'name', 'vecvalue', 'vectime'],
                       converters={
                           'run': parse_run,
                           'vectime': parse_vector,
                           'vecvalue': parse_vector,
                           'name': parse_name
                       })

    # remove useless rows
    data = data[data.type == 'vector']
    data.reset_index(inplace=True, drop=True)

    # rename vecvalue for simplicity...
    data = data.rename({'vecvalue': 'value', 'vectime': 'time'}, axis=1)
    df = data[['run', 'name', 'time', 'value']].sort_values(['run', 'name'])
    df.to_csv("cleanedData.csv", index=False)
    return data[['run', 'name', 'time', 'value']].sort_values(['run', 'name'])


# Parse CSV file
def scalar_parse(cqi, pkt_lambda):
    path_csv = DATA_PATH + MODE_PATH[cqi] + LAMBDA_PATH[pkt_lambda] + CSV_PATH['sca']
    data = pd.read_csv(path_csv,
                       usecols=['run', 'type', 'name', 'value'],
                       converters={
                           'run': parse_run,
                           'name': parse_name
                       }
                       )

    # remove useless rows (first 100-ish rows)
    data = data[data.type == 'scalar']
    data.reset_index(inplace=True, drop=True)

    # data['user'] = data.name.apply(lambda x: x.split['-'][1] if '-' in x else 'global')

    return data[['run', 'name', 'value']].sort_values(['run', 'name'])


####################################################
#                       UTIL                       #
####################################################

# mean of x
def running_avg(x):
    return np.cumsum(x) / np.arange(1, x.size + 1)


# window (N) mean of x
def winavg(x, N):
    xpad = np.concatenate((np.zeros(N), x))
    s = np.cumsum(xpad)
    ss = s[N:] - s[:-N]
    ss[N - 1:] /= N
    ss[:N - 1] /= np.arange(1, min(N - 1, ss.size) + 1)
    return ss


# keep only the data of the choosed attribute
def filter_data(data, attribute, start=0):
    sel = data[data.name == attribute]

    for i, row in sel.iterrows():
        tmp = np.where(row.time < start, np.nan, row.value)
        sel.at[i, 'value'] = tmp[~np.isnan(tmp)]
    return sel


def plot_mean_vectors_datalink(data, prefix, start=0, duration=SIM_TIME, iterations=None,
                               datalinks=range(0, NUM_DATA_LINK)):
    if iterations is None:
        iterations = [0]

    sel = data[data.name.str.startswith(prefix + '-')]

    for d in datalinks:
        usr = sel[sel.name == prefix + "-" + str(d)]
        for i in iterations:
            tmp = usr[(usr.run == i)]
            for row in tmp.itertuples():
                plt.plot(row.time, running_avg(row.value))

    # plot the data
    plt.xlim(start, duration)
    plt.show()
    return


def plot_mean_vectors(data, attribute, start=WARMUP_PERIOD, duration=SIM_TIME, iterations=None):
    if iterations is None:
        iterations = [0]

    sel = data[data.name == attribute]

    # plot a mean vector for each iteration
    for i in iterations:
        tmp = sel[sel.run == i]

        for row in tmp.itertuples():
            plt.plot(row.time, running_avg(row.value))

    # plot the data
    if attribute == "queueLength":
        plt.title("queueLength")
    elif attribute == "responseTime":
        plt.title("responseTime")
    elif attribute == "waitingTime":
        plt.title("waitingTime")
    elif attribute == "meanMalus":
        plt.title("meanMalus")
    elif attribute == "actualCapacity":
        plt.title("actualCapacity")
    elif attribute == "arrivalTime":
        plt.title("arrivalTime")
    elif attribute == "tDistribution":
        plt.title("tDistribution")
    elif attribute == "serviceTime":
        plt.title("serviceTime")
    elif attribute == "utilization":
        plt.title("utilization")

    plt.xlim(start, duration)
    plt.show()
    return


def describe_attribute_sca(data, name, value='value'):
    # print brief summary of attribute name (with percentiles and stuff)
    pprint.pprint(data[data.name == name][value].describe(percentiles=[.25, .50, .75, .95]))
    return


def describe_attribute_vec(data, name, iteration=0):
    values = pd.Series(data[data.name == name].value.iloc[iteration])
    pprint.pprint(values.describe(percentiles=[.25, .50, .75, .95]))
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


def aggregate_users_signals(data, signal, datalinks=range(0, NUM_DATA_LINK)):
    # meanResponseTime => dato il mean response time di un utente per ogni run, calcolo la media dei
    # mean response time dell'utente su tutte le run. E poi faccio la media per tutti gli utenti per
    # ottenere il mean responsetime medio per tutti gli utenti.
    return data[data.name.isin([signal + '-' + str(i) for i in datalinks])].groupby('run').mean().describe(
        percentiles=[.25, .50, .75, .95])


def scalar_stats(data, attr=None, datalinks=range(0, NUM_DATA_LINK)):
    stats = pd.DataFrame()
    attributes = data.name.unique() if attr is None else attr

    # STATS FOR EACH SIGNAL
    for attr in attributes:
        stats[attr] = data[data.name == attr].value.describe(percentiles=[.25, .50, .75, .95])

    # Aggregate dynamic stats (one signal per user):
    stats['meanResponseTime'] = aggregate_users_signals(data, 'responseTime', datalinks)
    stats['meanThroughput'] = aggregate_users_signals(data, 'tptUser', datalinks)
    stats['meanCQI'] = aggregate_users_signals(data, 'CQI', datalinks)
    stats['meanNumberRBs'] = aggregate_users_signals(data, 'numberRBs', datalinks)

    # Transpose...
    stats = stats.T

    # COMPUTE CI
    stats['ci95_l'] = stats['mean'] - 1.96 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci95_h'] = stats['mean'] + 1.96 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci99_l'] = stats['mean'] - 2.58 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci99_h'] = stats['mean'] + 2.58 * (stats['std'] / np.sqrt(stats['count']))
    return stats


####################################################
#                      LORENZ                      #
####################################################

def gini(data, precision=3):
    sorted_list = sorted(data)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(data) / 2.
    return round((fair_area - area) / fair_area, precision)


def lorenz_curve_sca(data, attribute, iterations=range(0, NUM_ITERATIONS)):
    # val = pd.DataFrame()
    sel = data[data.name.str.startswith(attribute + '-')]
    sel['user'] = sel.name.str.split('-', expand=True)[1].astype(int)
    sorted_data = pd.DataFrame()

    for r in iterations:
        tmp = sel[sel.run == r]
        sorted_data['run-' + str(r)] = np.sort(tmp.value.values)

    # return sorted_data
    plot_lorenz_curve(sorted_data.mean(axis=1))
    plt.plot([0, 1], [0, 1], 'k', alpha=0.85)
    plt.title("Lorenz Curve for " + attribute + " -  Gini: " + str(gini(sorted_data.mean(axis=1))))
    plt.show()
    return


def lorenz_curve_vec(data, attribute, name):
    # consider only the values for attribute
    clean_data = data[data.name == attribute]

    # for each iteration
    for i in range(0, len(clean_data)):
        # sort the data
        vec = clean_data.value.iloc[i]
        plot_lorenz_curve(vec)

    plt.plot([0, 1], [0, 1], 'k')
    plt.title("Lorenz Curve for " + attribute + ", " + name)
    plt.show()
    return


def all_lorenz(mode, lambda_val, attribute, iterations=range(0, NUM_ITERATIONS), save=False):
    data = scalar_parse(mode, lambda_val)

    # Plot the mean lorenz
    sel = data[data.name.str.startswith(attribute + '-')]
    sel['user'] = sel.name.str.split('-', expand=True)[1].astype(int)
    sorted_data = pd.DataFrame()

    for r in iterations:
        tmp = sel[sel.run == r]
        sorted_data['run-' + str(r)] = np.sort(tmp.value.values)
        plot_lorenz_curve(sorted_data['run-' + str(r)], color='grey', alpha=0.25)

    # return sorted_data
    plot_lorenz_curve(sorted_data.mean(axis=1))

    plt.plot([0, 1], [0, 1], 'k', alpha=0.85)
    plt.title(
        attribute + ": " + MODE_DESCRIPTION[mode] + ' and ' + LAMBDA_DESCRIPTION[lambda_val] + ' - Mean Gini: ' + str(
            gini(sorted_data.mean(axis=1))))

    if save:
        plt.savefig("lorenz_responseTime_" + mode + "_" + lambda_val + ".pdf")
        plt.clf()
    else:
        plt.show()

    return


####################################################
#                      ECDF                        #
####################################################


def ecdf_sca(data, attribute, aggregate=False, users=range(0, NUM_DATA_LINK), save=False):
    if aggregate:
        selected_ds = data[data.name.isin([attribute + '-' + str(i) for i in users])].groupby('run').mean()
    else:
        selected_ds = data[data.name == attribute]

    plot_ecdf(selected_ds.value.to_numpy())
    plt.title("ECDF for " + attribute + (" (aggregated mean)" if aggregate else ""))

    if save:
        plt.savefig("ecdf_" + attribute + ".pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()
    return


def plot_ecdf(data, name):
    # sort the values
    sorted_data = np.sort(data)

    # eval y
    n = sorted_data.size
    F_x = [sorted_data[sorted_data <= x].size / n for x in sorted_data]

    # plot the plot
    plt.step(sorted_data, F_x, linewidth=4, label=name)

    return


def plot_ecdf_vec(data, attribute, iteration=0, sample_size=1000, replace=False):
    # consider only what i need

    sample = data[data.name == attribute]
    sample = sample.value.iloc[iteration]

    # consider a sample
    if sample_size is not None:
        sample = sample[np.random.choice(sample.shape[0], sample_size, replace=replace)]

    plot_ecdf(sample)
    plt.title("ECDF for " + attribute)
    plt.show()
    # plt.savefig("./img/ecdf/responseTime-50ms")
    return


####################################################
#                      IID                         #
####################################################

def check_iid_sca(data, attribute, aggregate=False, users=range(0, NUM_DATA_LINK), save=False):
    if aggregate:
        samples = data[data.name.isin([attribute + '-' + str(i) for i in users])].groupby('run').mean()
    else:
        samples = data[data.name == attribute].value
    check_iid(samples, attribute, aggregate=aggregate, save=save)
    return


def check_iid_vec(data, attribute, iteration=0, sample_size=1000, seed=SEED_SAMPLING, save=False):
    samples = pd.Series(data[data.name == attribute].value.iloc[iteration])

    # consider a sample
    if sample_size is not None:
        samples = samples.sample(n=sample_size, random_state=seed)

    check_iid(samples, attribute, save)
    return


def check_iid(samples, attribute, aggregate=False, save=False):
    pd.plotting.lag_plot(samples)
    plt.title("Lag-Plot for " + attribute + (" (mean) " if aggregate else ""))

    if aggregate:
        plt.ylim(samples.min().value - samples.std().value, samples.max().value + samples.std().value)
        plt.xlim(samples.min().value - samples.std().value, samples.max().value + samples.std().value)
    else:
        plt.ylim(samples.min() - samples.std(), samples.max() + samples.std())
        plt.xlim(samples.min() - samples.std(), samples.max() + samples.std())

    if save:
        plt.savefig("iid_lagplot_" + attribute + ".pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()

    pd.plotting.autocorrelation_plot(samples)
    plt.title("Autocorrelation plot for " + attribute + (" (mean) " if aggregate else ""))
    if save:
        plt.savefig("iid_autocorrelation_" + attribute + ".pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()

    return


####################################################
####################################################
####################################################


def plot_winavg_vectors(data, attribute, start=0, duration=SIM_TIME, iterations=None, win=100):
    if iterations is None:
        iterations = [0]

    sel = data[data.name == attribute]

    # plot a mean vector for each iteration
    for i in iterations:
        tmp = sel[sel.run == i]
        for row in tmp.itertuples():
            plt.plot(row.time, winavg(row.value, win))

    # plot the data
    if attribute == "queueLength":
        plt.title("queueLength")
    elif attribute == "responseTime":
        plt.title("responseTime")
    elif attribute == "waitingTime":
        plt.title("waitingTime")
    elif attribute == "meanMalus":
        plt.title("meanMalus")
    elif attribute == "actualCapacity":
        plt.title("actualCapacity")
    elif attribute == "arrivalTime":
        plt.title("arrivalTime")
    elif attribute == "tDistribution":
        plt.title("tDistribution")
    elif attribute == "serviceTime":
        plt.title("serviceTime")
    elif attribute == "utilization":
        plt.title("utilization")
    elif attribute == "arrivalTime":
        plt.title("arrivalTime")

    # plot the data
    plt.xlim(start, duration)
    plt.show()
    return


def stats_to_csv():
    exp = {
        'uni': ['l09', 'l15', 'l2', 'l5'],
        'bin_old': ['l14', 'l15', 'l2', 'l5'],
        'bin': ['l15', 'l2', 'l5']
    }

    for m in exp.keys():
        for l in exp[m]:
            data = scalar_parse(m, l)
            stats = scalar_stats(data)
            stats.to_csv('stats_' + m + '_' + l + '.csv')
    return


def unibin_ci_plot(lambda_val, attr, bin_mode='bin', ci=95, save=False):
    # get the data...
    stats1 = scalar_stats(scalar_parse('uni', lambda_val))
    stats2 = scalar_stats(scalar_parse(bin_mode, lambda_val))

    bar1 = stats1['mean'][attr]
    bar2 = stats2['mean'][attr]

    error = np.array([bar1 - stats1['ci' + str(ci) + '_l'][attr], stats1['ci' + str(ci) + '_h'][attr] - bar1]).reshape(
        2, 1)
    plt.bar(MODE_DESCRIPTION['uni'], bar1, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=7)

    error = np.array([bar2 - stats2['ci' + str(ci) + '_l'][attr], stats2['ci' + str(ci) + '_h'][attr] - bar2]).reshape(
        2, 1)
    plt.bar(MODE_DESCRIPTION[bin_mode], bar2, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=7)

    # Show graphic
    plt.title("Comparison for " + attr + " and " + LAMBDA_DESCRIPTION[lambda_val])
    if save:
        plt.savefig("compare_unibin_" + attr + "_" + lambda_val + ".pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()
    return


def plot_to_img(mode, lambdas):
    for l in lambdas:
        all_lorenz(mode, l, 'responseTime', save=True)
    return


def histo_datalink(mode, lambda_val, attribute, ci=95, users=range(0, NUM_DATA_LINK), save=False):
    stats = scalar_stats(scalar_parse(mode, lambda_val))

    for u in users:
        attr = attribute + '-' + str(u)
        bar = stats['mean'][attr]
        error = np.array([bar - stats['ci' + str(ci) + '_l'][attr], stats['ci' + str(ci) + '_h'][attr] - bar]).reshape(
            2, 1)
        plt.bar('User ' + str(u), bar, yerr=error, align='center', alpha=0.95, ecolor='k', capsize=7)

    # Show graphic
    plt.title(attribute + ": " + MODE_DESCRIPTION[mode] + " and " + LAMBDA_DESCRIPTION[lambda_val])
    if save:
        plt.savefig("histousers_" + attribute + "_" + mode + "_" + lambda_val + ".pdf", bbox_inches="tight")
        plt.clf()
    else:
        plt.show()
    return


def scatterplot_mean(mode, lambda_val, x_attr, y_attr, group=None, hue='user',
                     col=None):
    data = tidy_scalar(mode, lambda_val)

    if group is not None:
        data = data.groupby(group).mean().reset_index()
        hue = group

    # cutie scatterplot
    sns.scatterplot(x=x_attr, y=y_attr, data=data, hue=hue)
    plt.title("Scatterplot " + x_attr + " - " + y_attr + " (" + MODE_DESCRIPTION[mode] + ", " + LAMBDA_DESCRIPTION[
        lambda_val] + ")")
    plt.show()

    # kind of regression plot?
    sns.lmplot(x=x_attr, y=y_attr, data=data, col=col)
    plt.show()

    sns.lmplot(x=x_attr, y=y_attr, data=data, lowess=True)
    plt.show()

    sns.jointplot(x=x_attr, y=y_attr, data=data, kind="reg")
    plt.show()

    return


def CQI_to_class(cqi):
    if cqi < 4: return CQI_CLASSES[0]
    # if cqi < 7: return 'MID'
    return CQI_CLASSES[1]


def class_plot(mode, lambda_val, y_attr='responseTime'):
    data = tidy_scalar(mode, lambda_val)

    sns.catplot(x='class', y=y_attr, hue='user', data=data, order=CQI_CLASSES)
    plt.title("Class plot for " + y_attr + " (" + MODE_DESCRIPTION[mode] + ", " + LAMBDA_DESCRIPTION[lambda_val] + ")")
    plt.show()

    sns.catplot(x='class', y=y_attr, hue='user', data=data, order=CQI_CLASSES, kind='bar', capsize=0.05)
    plt.title(
        "Class Barplot for " + y_attr + " (" + MODE_DESCRIPTION[mode] + ", " + LAMBDA_DESCRIPTION[lambda_val] + ")")
    plt.show()

    sns.catplot(x='class', y=y_attr, hue='user', data=data, order=CQI_CLASSES, kind='box')
    plt.title(
        "Class Boxplot for " + y_attr + " (" + MODE_DESCRIPTION[mode] + ", " + LAMBDA_DESCRIPTION[lambda_val] + ")")
    plt.show()

    return


def tidy_scalar(mode, lambda_val):
    tidy_data = pd.DataFrame()

    data = scalar_parse(mode, lambda_val)
    sel = data[data.name.str.contains('-')].reset_index().drop('index', axis=1)
    sel[['attr', 'user']] = sel.name.str.split('-', expand=True)
    sel = sel.drop('name', axis=1)

    tidy_data['user'] = 'user-' + sel[
        sel.attr == sel.attr.iloc[0]].user.values
    tidy_data['run'] = sel[sel.attr == sel.attr.iloc[0]].run.values  # same
    for attr_name in sel.attr.unique():
        tidy_data[attr_name] = sel[sel.attr == attr_name].value.values

    tidy_data['class'] = tidy_data['CQI'].apply(lambda x: CQI_to_class(x))
    return tidy_data


def histogram(df, nbin, name, k):
    plt.figure()
    n, bins, patches = plt.hist(
        df['Mean_' + name], nbin, density=True, facecolor='g', alpha=0.75)
    plt.title('Histogram of ' + name + ' ' + k + 's')
    plt.grid(True)
    plt.show()


'''
    look for x such that F(x) = P{X < x} = quantile, with an error of maxError
'''


def findQuantile(quantile, name, maxError):
    x = 0.0
    if name == 'serviceTime':
        error = quantile - plot_CDF("serviceTime")
        while error > maxError:
            x += 0.1 * error
            error = quantile - plot_CDF("serviceTime")
    elif name == 'distance':
        error = quantile - plot_CDF("serviceTime")
        while error > maxError:
            x += 0.3 * error
            error = quantile - plot_CDF("serviceTime")
    return x


'''
     find every quantile for both theoretical and sample distribution
'''


def fitDistribution(df, name, maxError):
    theoreticalQ = []
    sampleQ = []
    for i in range(1, len(df)):
        quantile = (i - 0.5) / len(df)
        sq = df[name].quantile(quantile)
        tq = findQuantile(quantile, name, maxError)
        sampleQ.append(sq)
        theoreticalQ.append(tq)
        pprint.pprint(quantile, tq, sq)
    return [theoreticalQ, sampleQ]


'''
     draw a qq plot
'''


def qqPlot(theoreticalQ, sampleQ, name):
    slope, intercept, r_value, p_value, std_err = regr(theoreticalQ, sampleQ)

    plt.figure()
    plt.scatter(theoreticalQ, sampleQ, s=0.8, label=name, c='blue')
    y = [x * slope + intercept for x in theoreticalQ]
    plt.plot(theoreticalQ, y, 'r', label='Trend line')
    plt.text(0, max(sampleQ) * 0.6, '\n\n$R^2$ = ' + str('%.6f' % r_value ** 2))
    if intercept > 0:
        plt.text(0, max(sampleQ) * 0.55, 'y = ' + str('%.6f' %
                                                      slope) + 'x + ' + str('%.6f' % intercept))
    else:
        plt.text(0, max(sampleQ) * 0.55, 'y = ' + str('%.6f' %
                                                      slope) + 'x ' + str('%.6f' % intercept))

    plt.xlabel('Theoretical Quantile')
    plt.ylabel('Sample Quantile')
    plt.title('QQ plot ' + name)
    plt.grid(True)
    plt.legend()


'''
    Find every column of dataframe that starts with "stat"
'''


def splitStat(df, stat):
    statDf = df[df.columns[pd.Series(df.columns).str.startswith(stat)]]
    return statDf


'''
    Sort every column of dataframe and perform a mean per row
'''


def meanPerRow(df, stat):
    orderedStat = df.apply(lambda x: x.sort_values().values)
    orderedStat = orderedStat.apply(lambda x: x.dropna())
    return pd.DataFrame(orderedStat.mean(axis=1), columns=[stat])


def vertical_mean_line(x, **kwargs):
    plt.axvline(x.mean(), linestyle="--",
                color=kwargs.get("color", "r"))
    txkw = dict(size=15, color=kwargs.get("color", "r"))

    label_x_pos_adjustment = 0.08  # this needs customization based on your data
    label_y_pos_adjustment = 5  # this needs customization based on your data
    if x.mean() < 6:  # this needs customization based on your data
        tx = "mean: {:.2f}\n(std: {:.2f})".format(x.mean(), x.std())
        plt.text(x.mean() + label_x_pos_adjustment, label_y_pos_adjustment, tx, **txkw)
    else:
        tx = "mean: {:.2f}\n  (std: {:.2f})".format(x.mean(), x.std())
        plt.text(x.mean() - 1.4, label_y_pos_adjustment, tx, **txkw)


def df_to_plotly(df):
    return {'z': df.values.tolist(),
            'x': df.columns.tolist(),
            'y': df.index.tolist()}


def plot_CDF(df, attribute, iteration=0):
    dataframe = df[df.name == attribute]

    X = dataframe.iloc[iteration].time
    Y = dataframe.iloc[iteration].value

    sorted = np.sort(Y)
    p = 1. * np.arange(len(Y)) / (len(Y) - 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.plot(p, sorted)
    ax1.set_xlabel('$p$')
    ax1.set_ylabel('$x$')

    ax2 = fig.add_subplot(122)
    ax2.plot(sorted, p)
    ax2.set_xlabel('$x$')
    ax2.set_ylabel('$p$')  # Plot both
    plot(X, Y)

    show()


def parse_module(s):
    if s == '':
        return None
    string = s.split('.')[1]
    if string != "controlTower":
        string += "." + s.split('.')[2]

    return string


def scalar_df_parse(path_csv):
    data = pd.read_csv(path_csv,
                       usecols=['run', 'type', 'module', 'name', 'value'],
                       converters={
                           'run': parse_run,
                           'name': parse_name,
                           'module': parse_module
                       }
                       )

    # remove useless rows (first 100-ish rows)
    data = data[data.type == 'scalar']
    data.reset_index(inplace=True, drop=True)

    # data['user'] = data.name.apply(lambda x: x.split['-'][1] if '-' in x else 'global')

    return data[['run', 'module', 'name', 'value']].sort_values(['run', 'name'])


def scalar_analysis(dataframe):
    dataframe.to_csv("scalar.csv", index=False)
    totaleServiceTime = []
    df = pd.DataFrame()
    thru = []
    srvTime = []
    packetReceived = []
    packetSent = []
    module = []
    queueLength = []

    for i in range(0, 10):
        dati = dataframe[dataframe.run == i]
        dati = dati[dati.name == "sentPackets"]

        dati2 = dataframe[dataframe.run == i]
        dati2 = dati2[dati2.name == "receivedPackets"]

        dati3 = dataframe[dataframe.run == i]
        dati3 = dati3[dati3.name == "serviceTime"]

        dati4 = dataframe[dataframe.run == i]
        dati4 = dati4[dati4.name == "queueLength"]

        pprint.pprint(
            f"{i}: {dati2['value'].sum()} packets received of"
            f" {dati['value'].sum()} packets sent, packet loss rate {dati2['value'].sum() / dati['value'].sum()}")

        packetReceived.append(dati2['value'].sum())
        packetSent.append(dati['value'].sum())

        tot = 0
        # totMalus = 0
        aircraft = dati['module'].iloc[0].split('.')[0]
        totale = []

        for j in range(0, dati['value'].size):
            to_control = dati['module'].iloc[j].split('.')[0]
            dataLink = dati['module'].iloc[j].split('.')[1]

            throughput = dati['value'].iloc[j] / 400
            serviceTime = (dati3['value'].iloc[j]) / 400
            queLen = (dati4['value'].iloc[j])

            if throughput != 0:
                pprint.pprint(f"throughput {dati['module'].iloc[j]}:  {throughput} packets/seconds")
                pprint.pprint(f"serviceTime of {dati3['module'].iloc[j]}: {serviceTime} seconds")
                pprint.pprint(f"queueLength of {dati3['module'].iloc[j]}: {queLen} seconds")

                totaleServiceTime.append(serviceTime)
                srvTime.append(serviceTime)
                thru.append(throughput)
                module.append(dati['module'].iloc[j])
                queueLength.append(dati4['value'].iloc[j])

                if aircraft == to_control:
                    tot += throughput
                else:
                    pprint.pprint(f"cumulative throughput {aircraft}:  {tot} packets/seconds")
                    # pprint.pprint(f"cumulative malus {aircraft}:  {totMalus} packets/seconds")

                    # if tot != 0:
                    # pprint.pprint(f"Percentage of malus over throughput {aircraft}:  {int((totMalus / tot) * 100)}% ")

                    aircraft = to_control
                    totale.append(tot)
                    totaleServiceTime.append(serviceTime)
                    # totaleMalus.append(totMalus)
                    tot = 0
                    # totMalus = 0

        # pprint.pprint(f"Total:  {totale} packets/seconds")
        # pprint.pprint(f"Total serviceTime:  {totaleServiceTime} seconds")

        # plt.plot(totaleServiceTime)
        # plt.plot(totaleMalus)
        # plt.title(f"Mean service time for iteration {i}")
        # plt.title(f"Throughput for {i}")
        # plt.savefig(f"./img/throughput/malus-50ms{i}.png")
        # plt.show()

    # pprint.pprint(f"Total serviceTime:  {totaleServiceTime} seconds")
    plt.plot(totaleServiceTime)
    plt.plot(queueLength)
    # plt.plot(totaleMalus)
    plt.title(f"Mean service time for iteration {i}")
    # plt.title(f"Throughput for {i}")
    # plt.savefig(f"./img/throughput/malus-50ms{i}.png")
    plt.show()
    df['aircraft'] = module
    # df['packetReceived'] = packetReceived
    # df['packetSent'] = packetSent
    df['throughput'] = thru
    df['serviceTime'] = srvTime
    df['queueLength'] = queueLength

    df.to_csv('test9.csv', index=False)


def aggregate_users_signals(data, signal, datalinks=range(0, NUM_DATA_LINK)):
    return data[data.name.isin([signal + '-' + str(i) for i in datalinks])].groupby('run').mean().describe(
        percentiles=[.25, .50, .75, .95])


def data_analysis(dataframe, attribute):
    stats = pd.DataFrame()
    pprint.pprint(dataframe['value'].mean())
    attributes = dataframe.name.unique()

    for attr in attributes:
        stats[attr] = dataframe[dataframe.name == attribute].value.describe(percentiles=[.25, .50, .75, .95])
    NUM_DATA_LINK = 10
    stats['mean' + attribute] = aggregate_users_signals(dataframe, attribute)
    stats['meanThroughput'] = aggregate_users_signals(dataframe, 'tptUser')
    stats['meanCQI'] = aggregate_users_signals(dataframe, 'CQI')
    stats['meanNumberRBs'] = aggregate_users_signals(dataframe, 'numberRBs')

    # Transpose...
    stats = stats.T

    # COMPUTE CI
    stats['ci95_l'] = stats['mean'] - 1.96 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci95_h'] = stats['mean'] + 1.96 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci99_l'] = stats['mean'] - 2.58 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci99_h'] = stats['mean'] + 2.58 * (stats['std'] / np.sqrt(stats['count']))
    return stats


def plot_ecdf_comparation(iteration=0, sample_size=1000, replace=False):
    df = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\lognormal\\scalar-50ms.csv")
    df1 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\lognormal\\scalar-35ms.csv")
    df2 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\lognormal\\scalar-20ms.csv")
    df3 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\lognormal\\scalar-10ms.csv")

    sample = df[df.name == "responseTime"]
    x = np.sort(sample['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="50ms")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    sample1 = df1[df1.name == "responseTime"]

    x = np.sort(sample1['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="35ms")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    sample2 = df2[df2.name == "responseTime"]

    x = np.sort(sample2['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="20ms")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    sample3 = df3[df3.name == "responseTime"]

    x = np.sort(sample3['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="10ms")
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

    stats = data_analysis(df, "responseTime")
    stats.to_csv('stats1.csv', index=False)
    stats = data_analysis(df1, "responseTime")
    stats.to_csv('stats2.csv', index=False)
    stats = data_analysis(df2, "responseTime")
    stats.to_csv('stats3.csv', index=False)
    stats = data_analysis(df3, "responseTime")
    stats.to_csv('stats4.csv', index=False)

    # data_analysis(df, "responseTime")
    # data_analysis(df1, "responseTime")
    # data_analysis(df2, "responseTime")

    '''
    sample = df[df.name == "responseTime"]
    sample = sample.value.iloc[iteration]

    # consider a sample
    if sample_size is not None:
        sample = sample[np.random.choice(sample.shape[0], sample_size, replace=replace)]

    plot_ecdf(sample, "10ms")
    plt.title("ECDF for responseTime")

    sample1 = df1[df1.name == "responseTime"]
    sample1 = sample1.value.iloc[iteration]

    # consider a sample
    if sample_size is not None:
        sample1 = sample1[np.random.choice(sample1.shape[0], sample_size, replace=replace)]

    plot_ecdf(sample1, "12ms")

    sample2 = df2[df2.name == "responseTime"]
    sample2 = sample2.value.iloc[iteration]

    # consider a sample
    if sample_size is not None:
        sample2 = sample2[np.random.choice(sample2.shape[0], sample_size, replace=replace)]

    plot_ecdf(sample2, "20ms")

    '''
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.title("Comparison for the waiting time ecdfs")
    plt.show()


def plot_lorenz_curve(data, name):
    # sort the data
    sample = data[data.name == "responseTime"]
    sorted_data = np.sort(sample['value'].dropna())

    # compute required stuff
    n = sorted_data.size
    T = sorted_data.sum()
    x = [i / n for i in range(0, n + 1)]
    y = sorted_data.cumsum() / T
    y = np.hstack((0, y))

    # plot
    plt.plot([0, 1], [0, 1], 'k')
    plt.title("Lorenz Curve for " + name)
    plt.plot(x, y, alpha=1, label=name)
    plt.show()

    return


def lorenz_curve_analysis():
    '''
    df = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\responseTime-7.5ms.csv")
    df4 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\responseTime-7.8ms.csv")
    df5 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\responseTime-7.7ms.csv")
    df0 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\responseTime-8.5ms.csv")
    df1 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\responseTime-8ms.csv")
    # df4 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\responseTime-9ms.csv")
    df2 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\responseTime-20ms.csv")
    df3 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\responseTime-35ms.csv")
    df6 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\responseTime-7.6ms.csv")
    df7 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\responseTime-7.9ms.csv")
    df1 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\queueLength-20ms.csv")
    df4 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\queueLength-35ms.csv")
    df5 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\queueLength-50ms.csv")
    df6 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\queueLength-8ms.csv")
    df7 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\queueLength-10ms.csv")
    '''

    df4 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\lognormal\\scalar-10ms.csv")
    df5 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\lognormal\\scalar-20ms.csv")
    df6 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\lognormal\\scalar-35ms.csv")
    df7 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\lognormal\\scalar-50ms.csv")
    # plot_lorenz_curve(df, "7.5ms")
    # plot_lorenz_curve(df0, "8.5ms")
    # plot_lorenz_curve(df0, "7ms")
    plot_lorenz_curve(df4, "10ms")
    plot_lorenz_curve(df5, "20ms")
    plot_lorenz_curve(df6, "35ms")
    plot_lorenz_curve(df7, "50ms")

    # plot_lorenz_curve(df2, "20ms")
    # plot_lorenz_curve(df3, "35ms")
    # plot_lorenz_curve(df4, "9ms")


def min_responseTime_validation():
    df = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\serviceTime-min-validation.csv")
    x = np.sort(df['value'].dropna())

    pprint.pprint(f"Minimum service time {x.min()}")


def main():
    pprint.pprint("Performance Evaluation - Python Data Analysis")

    # lorenz_curve_analysis()
    # min_responseTime_validation
    plot_ecdf_comparation()

    '''
    dataframe = scalar_df_parse()
    scalar_analysis(dataframe)

    df1 = pd.read_csv('test.csv')
    df2 = pd.read_csv('test2.csv')
    df3 = pd.read_csv('test3.csv')
    df5 = pd.read_csv('test5.csv')
    df6 = pd.read_csv('test6.csv')
    df7 = pd.read_csv('test7.csv')
    df8 = pd.read_csv('test8.csv')
    df9 = pd.read_csv('test9.csv')

    pprint.pprint(f"Mean service time nA=1, k=50ms: {df1['serviceTime'].mean()}")
    pprint.pprint(f"Mean service time nA=2, k=25ms: {df2['serviceTime'].mean()}")
    pprint.pprint(f"Mean service time nA=4, k=12.5ms: {df3['serviceTime'].mean()}")
    pprint.pprint(f"Mean service time nDL=10, k=50ms: {df5['serviceTime'].mean()}, mean queue length: {df5['queueLength'].mean()}")
    pprint.pprint(f"Mean service time nDL=20, k=25ms: {df6['serviceTime'].mean()}, mean queue length: {df6['queueLength'].mean()}")
    pprint.pprint(f"Mean service time nDL=50, k=25ms: {df7['serviceTime'].mean()}, mean queue length: {df7['queueLength'].mean()}")
    pprint.pprint(f"Mean service time nDL=200, k=25ms: {df8['serviceTime'].mean()}, mean queue length: {df8['queueLength'].mean()}")
    pprint.pprint(f"Mean service time nDL=1000, k=25ms: {df9['serviceTime'].mean()}, mean queue length: {df9['queueLength'].mean()}")
    '''
    # stats = responseTimeAnalysis(dataframe)
    # stats.to_csv("stats1.csv", index=False)
    # pprint.pprint(stats)
    # plot_ecdf_vec(dataframe, "responseTime", iteration=0, sample_size=1000, replace=False)

    # plot_ecdf_comparation()

    '''
    plot_mean_vectors(df, "arrivalTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9])
    plot_winavg_vectors(df, "arrivalTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9], win=5000)

    plot_mean_vectors(df, "meanMalus", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9])
    plot_winavg_vectors(df, "meanMalus", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9], win=5000)

    plot_mean_vectors(df, "tDistribution", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9])
    plot_winavg_vectors(df, "tDistribution", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9], win=5000)

    describe_attribute_vec(df, "arrivalTime", iteration=0)
    check_iid_vec(df, "arrivalTime", iteration=0, sample_size=1000, seed=42, save=False)
    lorenz_curve_vec(df, "serviceTime")

    stats = vector_stats(df, group=False)
    stats.to_csv("stats.csv",index=False)

    plot_mean_vectors(df, "responseTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9])
    plot_mean_vectors(df, "waitingTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9])
    plot_mean_vectors(df, "arrivalTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9])
    plot_mean_vectors(df, "serviceTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9])

    plot_winavg_vectors(df, "responseTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9], win=5000)
    plot_winavg_vectors(df, "waitingTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9], win=5000)
    plot_winavg_vectors(df, "arrivalTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9], win=5000)
    plot_winavg_vectors(df, "serviceTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9], win=5000)
    
    plot_ecdf_vec(df, "responseTime", iteration=0, sample_size=1000, replace=False)
    plot_ecdf_vec(df, "waitingTime", iteration=0, sample_size=1000, replace=False)
    plot_ecdf_vec(df, "arrivalTime", iteration=0, sample_size=1000, replace=False)

    print("Check IID")
    check_iid_vec(df, "responseTime", iteration=0, sample_size=1000, seed=42, save=False)
    check_iid_vec(df, "waitingTime", iteration=0, sample_size=1000, seed=42, save=False)

    print("Lorenz curve vectors")
    lorenz_curve_vec(df, "responseTime")

    pprint.pprint("Vectors stats")
    pprint.pprint(vector_stats(df, group=False))

    describe_attribute_vec(df, "responseTime", iteration=0)
    describe_attribute_vec(df, "waitingTime", iteration=0)
    describe_attribute_vec(df, "serviceTime", iteration=0)
    '''
    # dataframe = df[df.name == "queueLength"]
    # queueLength = pd.to_numeric(dataframe.iloc[0].value, errors='coerce')

    '''
    responseTime = pd.to_numeric(dataframe.iloc[0].value, errors='coerce')
    responseTimeCI = pd.to_numeric(dataframe.iloc[6], errors='coerce')
    queueLength = pd.to_numeric(dataframe.iloc[3], errors='coerce')
    queueLengthCI = pd.to_numeric(dataframe.iloc[4], errors='coerce')
    waitingTime = pd.to_numeric(dataframe.iloc[7], errors='coerce')
    waitingTimeCI = pd.to_numeric(dataframe.iloc[8], errors='coerce')
    '''
    '''
    k = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2]
    plt.style.use('ggplot')
    parser = argparse.ArgumentParser(description='Split Data by K')
    parser.add_argument(
        'dirPath', help='path of directory containing Results files')
    parser.add_argument(
        'distr', help='Distribution of Interarrival time [ exp | const ]')
    args = parser.parse_args()
    distribution = str(args.distr)
    path = str(args.dirPath)
    barWidth = 0.035
    errorBarStyle = dict(lw=0.5, capsize=2, capthick=0.5)
    '''
    '''
    # Plot Delay
    plt.figure(1)
    plt.errorbar(k, responseTime, yerr=responseTimeCI, fmt="--x",
                 markeredgecolor='red', linewidth=0.8, capsize=4, label='t= ' + str(t[i]))
    plt.xlabel('Interarrival Time [s]')
    plt.ylabel('Delay [s]')
    plt.ticklabel_format(axis='x', style='sci')
    plt.xticks(np.arange(0.5, 2.25, step=0.25))
    plt.title('End-to-End Delay')
    plt.legend()
    plt.grid(linestyle='--')
    '''

    # Plot Queue Length
    '''
    plt.figure(2)
    plt.errorbar(k, queueLength, yerr=2, fmt="--x",
                 markeredgecolor='red', linewidth=0.8, capsize=4, label='t= ')
    plt.xlabel('Interarrival Time [s]')
    plt.ylabel('Queue Length')
    plt.ticklabel_format(axis='x', style='sci')
    plt.xticks(np.arange(0.5, 2.25, step=0.25))
    plt.title('Queue Length Analysis')
    plt.legend()
    plt.grid(linestyle='--')
    '''
    '''
    # Bar plot Waiting Time Over Response Time
    plt.figure(3)
    x = [x - 0.04 * (2 - i) for x in k]
    plt.bar(x, responseTime, yerr=responseTimeCI,
            width=barWidth, error_kw=errorBarStyle, color='red')
    plt.bar(x, waitingTime, yerr=waitingTimeCI, width=barWidth,
            error_kw=errorBarStyle, label='t=' + str(t[i]), color=colors[i])
    plt.xlabel('Interarrival Time [s]')
    plt.ylabel('Time [s]')
    plt.xticks(np.arange(0.5, 2.25, step=0.25))
    plt.ticklabel_format(axis='x', style='sci')
    plt.title('Waiting Time over Response Time')
    plt.legend()
    plt.grid(linestyle='--')
    '''


if __name__ == '__main__':
    main()
