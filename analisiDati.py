import seaborn as sns
import matplotlib as plt
import csv
import pandas as pd
import pprint
import scipy.stats
import os
from pylab import *
import matplotlib.transforms as mtransforms
from matplotlib.pyplot import figure
from tqdm.notebook import tqdm, trange

color = sns.color_palette()

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
    data = data[data.name == "responseTime"]
    sorted_list = np.sort(data['value'])
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
    slope, intercept, r_value, p_value, std_err = 0  # regr(theoreticalQ, sampleQ)

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

    data['user'] = data.name.apply(lambda x: x.split['-'][1] if '-' in x else 'global')

    return data[['run', 'module', 'name', 'value']].sort_values(['run', 'name'])


def scalar_analysis(dataframe):
    # dataframe.to_csv("scalar.csv", index=False)
    totaleServiceTime = []
    df = pd.DataFrame()
    thru = []
    srvTime = []
    packetReceived = []
    packetSent = []
    module = []
    queueLength = []

    dati3 = dataframe[dataframe.name == "responseTime"]
    dati4 = dataframe[dataframe.name == "queueLength"]

    for i in range(0, 10):
        dati = dataframe[dataframe.run == i]
        dati = dati[dati.name == "sentPackets"]

        dati2 = dataframe[dataframe.run == i]
        dati2 = dati2[dati2.name == "receivedPackets"]

        for j in range(len(dati['value'])):
            pprint.pprint(
                f"{i}: {dati2['value'].sum()} packets received of"
                f" {dati['value'].sum()} packets sent, packet loss rate {1 - (dati2['value'].sum() / dati['value'].sum())}")

            packetReceived.append(dati2['value'].sum())
            packetSent.append(dati['value'].sum())

            tot = 0
            # totMalus = 0
            aircraft = dati['module'].iloc[0].split('.')[0]
            totale = []

            throughput = dati['value'].iloc[i] / 400
            serviceTime = (dati3['value'].iloc[i]) / 400
            queLen = (dati4['value'].iloc[i])

            if throughput != 0:
                pprint.pprint(f"throughput {dati['module'].iloc[i]}:  {throughput} packets/seconds")
                pprint.pprint(f"serviceTime of {dati3['module'].iloc[i]}: {serviceTime} seconds")
                pprint.pprint(f"queueLength of {dati3['module'].iloc[i]}: {queLen} packets")

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
    attributes = dataframe.name.unique()
    name = []

    for attr in attributes:
        name.append(attr)
        stats[attr] = dataframe[dataframe.name == attr].value.describe(percentiles=[.25, .50, .75, .95])

    # Transpose...
    stats = stats.T

    stats['name'] = name
    stats.set_index('name')
    pprint.pprint(stats)

    # COMPUTE CI
    stats['ci95_l'] = stats['mean'] - 1.96 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci95_h'] = stats['mean'] + 1.96 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci99_l'] = stats['mean'] - 2.58 * (stats['std'] / np.sqrt(stats['count']))
    stats['ci99_h'] = stats['mean'] + 2.58 * (stats['std'] / np.sqrt(stats['count']))
    return stats


def min_serviceTime_validation():
    df = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\serviceTime-min-validation.csv")
    x = np.sort(df['value'].dropna())

    pprint.pprint(f"Minimum service time {x.min()}")


def extract_2kr_matrix():
    path_csv = "C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\responseTime\\"
    list = os.listdir(path_csv)
    mean = []
    name = []
    name_errors = []
    errors = []

    for i in range(len(list)):
        with open(path_csv + list[i]) as file:
            df = scalar_df_parse(file)
            mean.append(df['value'].mean())
            name.append(list[i])

    df = pd.DataFrame()
    df['file'] = name
    df['mean'] = mean
    # df.to_csv("mean_values.csv", index=False)
    sum = 0
    somma = []
    media = []

    for j in range(100):
        for i in range(len(list)):
            with open(path_csv + list[i]) as file:
                df = scalar_df_parse(file)
                errors.append(df['value'].iloc[j] - mean[i])
                sum += pow((df['value'].iloc[j] - mean[i]), 2)

        name_errors.append(f"y{j} - ym")
        somma.append(sum)
        sum = 0

    df1 = pd.DataFrame()
    df1['iteration'] = name_errors
    df1['squared_sum_errors'] = somma
    df1.to_csv("errors.csv", index=False)


def mean_confidence_interval(data, confidence=0.99):
    data = data[data.name == 'responseTime']
    a = 1.0 * np.array(data['value'])
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return m, m - h, m + h


def plot_response_time(dataframe):
    dataframe.pivot(index='run', columns='name', values='value')
    del dataframe['run']
    df = dataframe.groupby('name').mean()
    pprint.pprint(df)


def scavetool_all():
    for i in number_datalink:
        os.system(
            'C:\\omnetpp-5.6.2\\bin\\scavetool.exe x ./simulations/results/Lognormal-capacity-'
            + str(i) + '-*.sca -o ./csv/pool_classico_varia_NA/Lognormal-capacity-' + str(i) + '.csv')
        os.system(
            '/home/leonardo/omnetpp-5.6.2/bin/scavetool x ./simulations/results/Exponential-capacity-'
            + str(i) + '-*.sca -o ./csv/pool_classico_varia_NA/Exponential-capacity-' + str(i) + '.csv')
        os.system(
            '/home/leonardo/omnetpp-5.6.2/bin/scavetool x ./simulations/results/Nonmonitoring-exponential-' +
            str(i) + '-*.sca -o ./csv/pool_classico_varia_NA/Nonmonitoring-exponential-' + str(i) + '.csv')

        os.system(
            '/home/leonardo/omnetpp-5.6.2/bin/scavetool x ./simulations/results/Nonmonitoring-lognormal-' +
            str(i) + '-*.sca -o ./csv/pool_classico_varia_NA/Nonmonitoring-lognormal-' + str(i) + '.csv')


def plot_scalar_mean(attribute):
    meanResponseTime = []
    meanResponseTimeLognormal = []
    index = []

    for i in interarrival_time:
        meanResponseTime.clear()
        index.clear()
        meanResponseTimeLognormal.clear()

        for m in monitoring_time:
            filename = './csv/' + 'Exponential-capacity-' + str(i) + "," + str(m) + '.csv'
            filename2 = './csv/' + 'Lognormal-capacity-' + str(i) + "," + str(m) + '.csv'

            index.append("m=" + str(m))

            with open(filename, 'r') as f:
                df = scalar_df_parse(filename)
                df = df[df.name == attribute]
                del df['run']
                meanResponseTime.append(df.value.mean())

            with open(filename2, 'r') as f:
                df = scalar_df_parse(filename2)
                df = df[df.name == attribute]
                del df['run']
                meanResponseTimeLognormal.append(df.value.mean())

        dataframe = pd.DataFrame()
        dataframe['file'] = index
        dataframe['meanResponseTimeExponential'] = meanResponseTime
        dataframe['meanResponseTimeLognormal'] = meanResponseTimeLognormal
        plt.plot(dataframe['meanResponseTimeExponential'], "g:o", label="exponential")
        plt.plot(dataframe['meanResponseTimeLognormal'], "r:o", label="lognormal")
        plt.fill_between(index, dataframe['meanResponseTimeExponential'], dataframe['meanResponseTimeLognormal']
                         , where=dataframe['meanResponseTimeExponential'] > dataframe['meanResponseTimeLognormal']
                         , facecolor='green', alpha=0.3)
        plt.fill_between(index, dataframe['meanResponseTimeExponential'], dataframe['meanResponseTimeLognormal']
                         , where=dataframe['meanResponseTimeExponential'] <= dataframe['meanResponseTimeLognormal']
                         , facecolor='red', alpha=0.3)
        plt.title('Response time for k=' + str(i) + "ms")
        plt.xticks(rotation=25)
        plt.xlabel("Value of m")
        plt.ylabel(attribute)

        plt.xticks([k for k in range(len(index))], [k for k in index])

        filename3 = './csv/' + 'Nonmonitoring-exponential-' + str(i) + '.csv'
        filename4 = './csv/' + 'Nonmonitoring-lognormal-' + str(i) + '.csv'

        with open(filename3, 'r') as f:
            df = scalar_df_parse(filename3)
            df = df[df.name == attribute]
            del df['run']
            plt.plot([df['value'].mean() for k in range(len(index))], "b:v", label="Non-monitoring exponential")

        with open(filename4, 'r') as f:
            df = scalar_df_parse(filename4)
            df = df[df.name == attribute]
            del df['run']
            plt.plot([df['value'].mean() for k in range(len(index))], "y:v", label="Non-monitoring lognormal")

        plt.legend(loc='upper left')
        plt.savefig(f'./analysis/immagini per clarissa/{attribute}/k={i}ms.png')
        plt.show()


plt.rcParams["figure.figsize"] = (8, 7)


def plot_everything_scenario():
    num_plots = 6

    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))

    meanResponseTime = []
    index = []

    for m in monitoring_time:
        index.append("m=" + str(m))

    dataframe = pd.DataFrame()
    dataframe['file'] = index

    for mode in modes:
        for i in interarrival_time:

            meanResponseTime.clear()

            for m in monitoring_time:
                if mode == 'Nonmonitoring-exponential-' or mode == 'Nonmonitoring-lognormal-':
                    filename = './csv/pool_max_40000/' + mode + str(i) + '.csv'
                else:
                    filename = './csv/pool_max_40000/' + mode + str(i) + "," + str(m) + '.csv'
                with open(filename, 'r') as f:
                    df = scalar_df_parse(filename)
                    df = df[df.name == 'queueLength']
                    del df['run']
                    meanResponseTime.append(df.value.mean())

            dataframe[f'k={i}'] = meanResponseTime
            plt.plot(dataframe[f'k={i}'], ":o", label=f"k={i}")

        plt.xticks([k for k in range(len(index))], [k for k in index])
        var = mode.split('-')[0]
        var2 = mode.split('-')[1]
        plt.title("scenario: " + var + " " + var2)
        plt.xticks(rotation=25)
        plt.xlabel("Value of m")
        plt.ylabel("Queue length")
        plt.legend(loc='upper left')
        plt.savefig(f"./analysis/immagini per clarissa/pool_40000/queueLength/{var}{var2}.png")
        plt.show()


def plot_everything_scenario_invertitoKM():
    num_plots = 6

    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))

    meanResponseTime = []
    index = []

    for k in interarrival_time:
        index.append("k=" + str(k))

    dataframe = pd.DataFrame()
    dataframe['file'] = index

    for mode in modes:
        for m in monitoring_time:

            meanResponseTime.clear()

            for i in interarrival_time:
                if mode == 'Nonmonitoring-exponential-' or mode == 'Nonmonitoring-lognormal-':
                    filename = './csv/pool_max_40000/' + mode + str(i) + '.csv'
                else:
                    filename = './csv/pool_max_40000/' + mode + str(i) + "," + str(m) + '.csv'
                with open(filename, 'r') as f:
                    df = scalar_df_parse(filename)
                    df = df[df.name == 'responseTime']
                    del df['run']
                    meanResponseTime.append(df.value.mean())

            dataframe[f'm={m}'] = meanResponseTime
            plt.plot(dataframe[f'm={m}'], ":o", label=f"m={m}")

        plt.xticks([k for k in range(len(index))], [k for k in index])
        var = mode.split('-')[0]
        var2 = mode.split('-')[1]
        plt.title("scenario: " + var + " " + var2)
        plt.xticks(rotation=25)
        plt.xlabel("Value of k")
        plt.ylabel("Response time")
        plt.legend(loc='upper left')
        plt.savefig(f"./analysis/immagini per clarissa/pool_40000/responseTime/invertitiKM/{var}{var2}.png")
        plt.show()


def plot_ecdf_single(iteration=0, sample_size=1000, replace=False):
    df = scalar_df_parse("csv/analisiScenario/20ms/Nonmonitoring-lognormal.csv")
    df1 = scalar_df_parse("csv/analisiScenario/35ms/Nonmonitoring-lognormal.csv")
    df2 = scalar_df_parse("csv/analisiScenario/50ms/Nonmonitoring-lognormal.csv")
    df3 = scalar_df_parse("csv/analisiScenario/10ms/Nonmonitoring-lognormal.csv")

    sample = df[df.name == "queueLength"]
    x = np.sort(sample['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="20ms")
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title("ECDF for 20ms")
    plt.savefig('./analysis/analisiScenario/queueLength/non-monitoring-lognormal/20ms.png')
    plt.show()

    sample1 = df1[df1.name == "queueLength"]

    x = np.sort(sample1['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="35ms")
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title("ECDF for 35ms")
    plt.savefig('./analysis/analisiScenario/queueLength/non-monitoring-lognormal/35ms.png')
    plt.show()

    sample2 = df2[df2.name == "queueLength"]

    x = np.sort(sample2['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="50ms")
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title("ECDF for 50ms")
    plt.savefig('./analysis/analisiScenario/queueLength/non-monitoring-lognormal/50ms.png')
    plt.show()

    sample3 = df3[df3.name == "queueLength"]

    x = np.sort(sample3['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="10ms")
    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title("ECDF for 10ms")
    plt.savefig('./analysis/analisiScenario/queueLength/non-monitoring-lognormal/10ms.png')
    plt.show()

    stats = data_analysis(df, "queueLength")
    stats.to_csv('./analysis/analisiScenario/queueLength/non-monitoring-lognormal/stats20ms.csv', index=False)
    stats = data_analysis(df1, "queueLength")
    stats.to_csv('./analysis/analisiScenario/queueLength/non-monitoring-lognormal/stats35ms.csv', index=False)
    stats = data_analysis(df2, "queueLength")
    stats.to_csv('./analysis/analisiScenario/queueLength/non-monitoring-lognormal/stats50ms.csv', index=False)
    stats = data_analysis(df3, "queueLength")
    stats.to_csv('./analysis/analisiScenario/queueLength/non-monitoring-lognormal/stats10ms.csv', index=False)


def plot_ecdf_comparation(iteration=0, sample_size=1000, replace=False):
    df = scalar_df_parse("csv/analisiScenario/20ms/Nonmonitoring-lognormal.csv")
    df1 = scalar_df_parse("csv/analisiScenario/35ms/Nonmonitoring-lognormal.csv")
    df2 = scalar_df_parse("csv/analisiScenario/50ms/Nonmonitoring-lognormal.csv")
    df3 = scalar_df_parse("csv/analisiScenario/10ms/Nonmonitoring-lognormal.csv")

    sample = df[df.name == "queueLength"]
    x = np.sort(sample['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="20ms")

    sample1 = df1[df1.name == "queueLength"]

    x = np.sort(sample1['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="35ms")

    sample2 = df2[df2.name == "queueLength"]

    x = np.sort(sample2['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="50ms")

    sample3 = df3[df3.name == "queueLength"]

    x = np.sort(sample3['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="10ms")

    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title("ECDF for queue length compared")
    plt.savefig('./analysis/analisiScenario/queueLength/non-monitoring-lognormal/comparison.png')
    plt.show()


def scenario_comparison():
    num_plots = 8

    colormap = plt.cm.gist_ncar
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, num_plots))))

    df = scalar_df_parse("csv/analisiScenario/20ms/Nonmonitoring-lognormal.csv")
    df1 = scalar_df_parse("csv/analisiScenario/20ms/Nonmonitoring-exponential.csv")

    df2 = scalar_df_parse("csv/analisiScenario/20ms/Exponential-capacity-0.5.csv")
    df3 = scalar_df_parse("csv/analisiScenario/20ms/Exponential-capacity-1.csv")
    df4 = scalar_df_parse("csv/analisiScenario/20ms/Exponential-capacity-4.csv")

    df5 = scalar_df_parse("csv/analisiScenario/20ms/Lognormal-capacity-0.5.csv")
    df6 = scalar_df_parse("csv/analisiScenario/20ms/Lognormal-capacity-1.csv")
    df7 = scalar_df_parse("csv/analisiScenario/20ms/Lognormal-capacity-4.csv")

    sample = df[df.name == "queueLength"]
    x = np.sort(sample['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="Non monitoring lognormal")

    sample1 = df1[df1.name == "queueLength"]

    x = np.sort(sample1['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="Non monitoring exponential")

    sample2 = df2[df2.name == "queueLength"]

    x = np.sort(sample2['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="Exponential m=0.5")

    sample3 = df3[df3.name == "queueLength"]

    x = np.sort(sample3['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="Exponential m=1")

    sample4 = df4[df4.name == "queueLength"]

    x = np.sort(sample4['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="Exponential m=4")

    sample5 = df5[df5.name == "queueLength"]

    x = np.sort(sample5['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="Lognormal m=0.5")

    sample6 = df6[df6.name == "queueLength"]

    x = np.sort(sample6['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="Lognormal m=1")

    sample7 = df7[df7.name == "queueLength"]

    x = np.sort(sample7['value'].dropna())
    n = x.size
    y = np.arange(1, n + 1) / n

    plt.scatter(x=x, y=y, label="Lognormal m=4")

    plt.legend(loc='best')
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('F(x)')
    plt.title("ECDF for queue length compared, k=20ms")
    plt.savefig('./analysis/analisiScenario/queueLength/comparison.png')
    plt.show()


def plot_response_time_variousT():
    dataframe = pd.DataFrame()
    index = []

    for k in capacity_change_time:
        for mode in modes:
            index.append(f"{mode},t={k}s")

    dataframe['index'] = index

    for mode in modes:

        index = []
        meanResponseTime = []

        for i in capacity_change_time:
            df = scalar_df_parse(
                f"C:\\Users\\Leonardo Poggiani\\Documents\\GitHub\\PECSNproject\\csv\\pool_classico_varia_T\\{mode}{i}.csv")
            response = df[df.name == "responseTime"]
            meanResponseTime.append(response.value.mean())
            index.append(i)

        plt.bar(index, meanResponseTime)
        plt.title(f"{mode}")
        plt.rcParams["figure.figsize"] = (12, 10)
        plt.xticks(rotation=25)
        plt.show()

    # pprint.pprint(meanResponseTime)
    # plt.xticks(x_pos, index)
    # plt.xticks([k for k in range(len(index))], [k for k in index])
    plt.xlabel("Value of t")
    plt.ylabel("Response time")
    plt.title("Comparison of various values of t")
    plt.legend(loc='best')
    plt.savefig(
        "C:\\Users\\Leonardo Poggiani\\Documents\\GitHub\\PECSNproject\\analysis\\variandoT\\responseTimeAlVariareDiT2.png")


def plot_response_time_variousNDL():
    dataframe = pd.DataFrame()
    index = []

    for k in number_datalink:
        index.append(f"nA={k}")

    dataframe['index'] = index

    for mode in modes:
        meanResponseTime = []
        var = mode.split("-")[0]
        var2 = mode.split("-")[1]

        for i in number_datalink:
            df = scalar_df_parse(f"csv/pool_classico_varia_NA/{mode}{i}.csv")
            response = df[df.name == "responseTime"]
            meanResponseTime.append(response.value.mean())

        dataframe[f'responseTime{mode}{i}'] = meanResponseTime

        plt.plot(meanResponseTime, ":o", label=f"{var} {var2}")

    plt.xticks([k for k in range(len(index))], [k for k in index])
    plt.xticks(rotation=25)
    plt.xlabel("Value of nA")
    plt.ylabel("Response time")
    plt.title("Comparison of various values of nA")
    plt.legend(loc='best')
    plt.savefig("./analysis/variandoNDLeNA/responseTimeAlVariareDinAT=2sk=20msm=3sx=0.05s.png")
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
    plt.title(name)
    plt.plot(x, y, alpha=1, label=name)
    plt.savefig(f"./analysis/analisiScenario/lorenzCurve/{name}.png")
    plt.show()
    return


def lorenz_curve_analysis():
    for k in interarrival_time:
        for mode in modes:
            for m in monitoring_time:
                if mode == 'Nonmonitoring-exponential' or mode == 'Nonmonitoring-lognormal':
                    df = scalar_df_parse(
                        f"./csv/analisiScenario/{k}ms/{mode}.csv")
                else:
                    df = scalar_df_parse(
                        f"./csv/analisiScenario/{k}ms/{mode}{m}.csv")

                plot_lorenz_curve(df, f"{mode}{k}ms,m={m}s,gini={gini(df)}")


####################################################
#                      IID                         #
####################################################

def check_iid_sca(data, attribute, mode, aggregate=False, users=range(0, 1000), save=False):
    if aggregate:
        samples = data.groupby('run').mean()
        pprint.pprint(samples)
        check_iid(samples, attribute, mode, aggregate=aggregate, save=save)
    return


def check_iid(samples, attribute, mode, aggregate=False, save=False):

    pd.plotting.lag_plot(samples)
    plt.title("Lag-Plot for " + attribute + (" (mean) " if aggregate else ""))

    if aggregate:
        plt.ylim(samples.min().value - samples.std().value, samples.max().value + samples.std().value)
        plt.xlim(samples.min().value - samples.std().value, samples.max().value + samples.std().value)

    plt.savefig(f"C:\\Users\\Leonardo Poggiani\\Documents\\GitHub\\PECSNproject\\analysis\\lagPlot\\responseTime\\{mode}.png")
    plt.show()

    pd.plotting.autocorrelation_plot(samples)
    plt.title("Autocorrelation plot for " + attribute + (" (mean) " if aggregate else ""))

    plt.savefig(f"C:\\Users\\Leonardo Poggiani\\Documents\\GitHub\\PECSNproject\\analysis\\autocorrelation\\responseTime\\{mode}.png")
    plt.show()

    return


def iid_grafici():
    for k in interarrival_time:
        for mode in modes:
            for t in monitoring_time:
                if mode == "Nonmonitoring-exponential" or mode == "Nonmonitoring-lognormal":
                    df = scalar_df_parse(
                        f"C:\\Users\\Leonardo Poggiani\\Documents\\GitHub\\PECSNproject\\csv\\analisiScenario\\{k}ms\\{mode}.csv")
                else:
                    df = scalar_df_parse(
                        f"C:\\Users\\Leonardo Poggiani\\Documents\\GitHub\\PECSNproject\\csv\\analisiScenario\\{k}ms\\{mode}{t}.csv")
                pprint.pprint(df)
                check_iid_sca(df, "responseTime", mode + str(k) + "-" + str(t), True)



def scavetool():
    for X in malus:
        os.system('/home/leonardo/omnetpp-5.6.2/bin/scavetool x ./simulations/results/Exponential-capacity-'
                  + str(X) + '-*.sca -o ./csv/pool_classico_vario_X/Exponential-capacity-' +
                  str(X) + '.csv')


capacity_change_time = [0.5, 1, 2, 5]
number_datalink = [1, 2, 16, 40]
modes = ['Exponential-capacity-', 'Lognormal-capacity-', 'Nonmonitoring-exponential', 'Nonmonitoring-lognormal']
interarrival_time = [9, 10, 13, 15, 20, 50]
monitoring_time = [0.5, 1.5, 4.5, 12]
malus = [0.01, 0.05, 0.5, 1, 2, 10]


def plot_response_time_various_X_k():
    dataframe = pd.DataFrame()
    index = []

    for k in interarrival_time:
        index.append(f"t={k}s")

    dataframe['index'] = index
    plt.xticks([k for k in range(len(index))], [k for k in index])

    for m in monitoring_time:
        for X in malus:
            meanResponseTime = []

            for k in interarrival_time:
                df = scalar_df_parse(
                    f"./csv/pool_classico_variano_X_k/m={m}s/{modes[0]}{k},{X}.csv")
                response = df[df.name == "queueLength"]
                meanResponseTime.append(response.value.mean())

            plt.plot(index,meanResponseTime,label=f"X={X}s")

        # plt.xticks(x_pos, index)
        # plt.xticks([k for k in range(len(index))], [k for k in index])
        plt.xlabel("Value of k")
        plt.ylabel("Queue length")
        plt.title(f"Comparison of various values of X, m={m}s")
        plt.legend(loc='best')
        # plt.savefig(f"./analysis/Experiment2/Queue Length Xvario-mfisso exponential_m{m}.png")
        plt.show()


def main():
    pprint.pprint("Performance Evaluation - Python Data Analysis")

    scavetool_all()

    plot_response_time_various_X_k()
    # iid_grafici()

    # lorenz_curve_analysis()

    # plot_everything_scenario()

    # plot_ecdf_comparation()
    # plot_ecdf_single()

    # scenario_comparison()

    # plot_response_time_variousT()

    # plot_response_time_variousNDL()

    # df = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\non-monitoring\\scalar-50ms.csv")
    # df = vector_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\exponential\\actualCapacity-50ms.csv")
    # df2 = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\lognormal\\scalar-50ms.csv")

    # min_responseTime_validation
    # plot_ecdf_comparation()

    # dataframe = scalar_df_parse("C:\\Users\\Leonardo Poggiani\\Desktop\\dataset\\v2\\test\\scalar.csv")
    # (dataframe)

    '''
    plot_mean_vectors(df, "arrivalTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9])
    plot_winavg_vectors(df, "arrivalTime", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9], win=5000)

    plot_mean_vectors(df, "meanMalus", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9])
    plot_winavg_vectors(df, "meanMalus", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9], win=5000)

    plot_mean_vectors(df, "tDistribution", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9])
    plot_winavg_vectors(df, "tDistribution", start=0, duration=400, iterations=[0, 1, 2, 3, 4, 5, 6 ,7 ,8 ,9], win=5000)

    describe_attribute_vec(df, "arrivalTime", iteration=0)
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
