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
    'bin' : "binomial/",
    'uni' : "uniform/",
    'bin_old' : "bin_old/"
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

def running_avg(x):
    return np.cumsum(x) / np.arange(1, x.size + 1)


def winavg(x, N):
    xpad = np.concatenate((np.zeros(N), x))  # pad with zeroes
    s = np.cumsum(xpad)
    ss = s[N:] - s[:-N]
    ss[N - 1:] /= N
    ss[:N - 1] /= np.arange(1, min(N - 1, ss.size) + 1)
    return ss


def filter_data(data, attribute, start=0):
    sel = data[data.name == attribute]

    for i, row in sel.iterrows():
        tmp = np.where(row.time < start, np.nan, row.value)
        sel.at[i, 'value'] = tmp[~np.isnan(tmp)]
    return sel


def plot_mean_vectors_user(data, prefix, start=0, duration=SIM_TIME, iterations=[0], users=range(0, NUM_USERS)):
    sel = data[data.name.str.startswith(prefix + '-')]

    for u in users:
        usr = sel[sel.name == prefix + "-" + str(u)]
        for i in iterations:
            tmp = usr[(usr.run == i)]
            for row in tmp.itertuples():
                plt.plot(row.time, running_avg(row.value))

    # plot the data
    plt.xlim(start, duration)
    plt.show()
    return


def parse_vector(s):
    return np.fromstring(s, sep=' ') if s else None


def parse_name(s):
    return s.split(':')[0] if s else None


def parse_run(s):
    return int(s.split('-')[6]) if s else None


def vector_parse():
    path_csv = "C:\\Users\\leona\\OneDrive\\Desktop\\exponential.csv"

    # vec files are huge, try to reduce their size ASAP!!
    data = pd.read_csv(path_csv,
                       delimiter=",", quoting=csv.QUOTE_NONNUMERIC, encoding='utf-8',
                       usecols=['run', 'type', 'module', 'name', 'vecvalue', 'vectime'],
                       converters={
                           'run': parse_run,
                           'vectime': parse_vector,  # i guess
                           'vecvalue': parse_vector,
                           'name': parse_name
                       })

    # remove useless rows
    data = data[data.type == 'vector']
    data.reset_index(inplace=True, drop=True)

    # rename vecvalue for simplicity...
    data = data.rename({'vecvalue': 'value', 'vectime': 'time'}, axis=1)
    return data[['run', 'name', 'time', 'value']].sort_values(['run', 'name'])


def plot_mean_vectors(data, attribute, start=10000, duration=150000, iterations=[0]):

    sel = data[data.name == attribute]

    # plot a mean vector for each iteration
    for i in iterations:
        tmp = sel[sel.run == (i)]

        for row in tmp.itertuples():
            plt.plot(row.time, running_avg(row.value))

    # plot the data
    if attribute == "queueLength":
        plt.title("queueLength")
    elif attribute == "responseTime":
        plt.title("responseTime")
    elif attribute == "waitingTime":
        plt.title("waitingTime")

    plt.xlim(start, duration)
    plt.show()
    return


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


####################################################
#                      PARSER                      #
####################################################


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


def lorenz_curve_sca(data, attribute, users=range(0, NUM_USERS), iterations=range(0, NUM_ITERATIONS)):
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


def lorenz_curve_vec(data, attribute):
    # consider only the values for attribute
    clean_data = data[data.name == attribute]

    # for each iteration
    for i in range(0, len(clean_data)):
        # sort the data
        vec = clean_data.value.iloc[i]
        plot_lorenz_curve(vec)

    plt.plot([0, 1], [0, 1], 'k')
    plt.title("Lorenz Curve for " + attribute)
    plt.show()
    return


def plot_lorenz_curve(data, color=None, alpha=1):
    # sort the data
    sorted_data = np.sort(data)

    # compute required stuff
    n = sorted_data.size
    T = sorted_data.sum()
    x = [i / n for i in range(0, n + 1)]
    y = sorted_data.cumsum() / T
    y = np.hstack((0, y))

    # plot
    plt.plot(x, y, color=color, alpha=alpha)
    return


def all_lorenz(mode, lambda_val, attribute, users=range(0, NUM_USERS), iterations=range(0, NUM_ITERATIONS), save=False):
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
#                      LORENZ                      #
####################################################


####################################################
#                      ECDF                        #
####################################################


def ecdf_sca(data, attribute, aggregate=False, users=range(0, NUM_USERS), save=False):
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


def plot_ecdf(data):
    # sort the values
    sorted_data = np.sort(data)

    # eval y
    n = sorted_data.size
    F_x = [(sorted_data[sorted_data <= x].size) / n for x in sorted_data]

    # plot the plot
    plt.step(sorted_data, F_x)
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
    return


####################################################
#                      ECDF                        #
####################################################


####################################################
#                      IID                         #
####################################################

def check_iid_sca(data, attribute, aggregate=False, users=range(0, NUM_USERS), save=False):
    if aggregate:
        samples = data[data.name.isin([attribute + '-' + str(i) for i in users])].groupby('run').mean()
    else:
        samples = data[data.name == attribute].value
    check_iid(samples, attribute, aggregate=aggregate, save=save)
    return


def check_iid_vec(data, attribute, iteration=0, sample_size=1000, seed=42, save=False):
    samples = pd.Series(data[data.name == attribute].value.iloc[iteration])

    # consider a sample
    if sample_size != None:
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
#                      IID                         #
####################################################

####################################################
####################################################
####################################################


def plot_winavg_vectors(data, attribute, start=0, duration=100, iterations=[0], win=1000):
    sel = data[data.name == attribute]

    # plot a mean vector for each iteration
    for i in iterations:
        tmp = sel[sel.run == (i)]
        for row in tmp.itertuples():
            plt.plot(row.time, winavg(row.value, win))

    # plot the data
    if attribute == "queueLength":
        plt.title("queueLength")
    elif attribute == "responseTime":
        plt.title("responseTime")
    elif attribute == "waitingTime":
        plt.title("waitingTime")

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


# this function works only with "nameSignal-user" kind of statistics
# (responseTime-#numuser or tptUser-#numuser)
def histo_users(mode, lambda_val, attribute, ci=95, users=range(0, NUM_USERS), save=False):
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


def scatterplot_mean(mode, lambda_val, x_attr, y_attr, users=range(0, NUM_USERS), save=False, group=None, hue='user',
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


# does it make sense?
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
        sel.attr == sel.attr.iloc[0]].user.values  # any dynsignal will be fine, because all of them have 100 instances
    tidy_data['run'] = sel[sel.attr == sel.attr.iloc[0]].run.values  # same
    for attr_name in sel.attr.unique():
        tidy_data[attr_name] = sel[sel.attr == attr_name].value.values

    tidy_data['class'] = tidy_data['CQI'].apply(lambda x: CQI_to_class(x))
    return tidy_data


def main():
    print("\n\nPerformance Evaluation - Python Data Analysis\n")

    df = vector_parse()

    plot_mean_vectors(df, "queueLength", start=10000, duration=150000, iterations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plot_mean_vectors(df, "responseTime", start=10000, duration=150000, iterations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plot_mean_vectors(df, "waitingTime", start=10000, duration=150000, iterations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    plot_winavg_vectors(df, "queueLength", start=10000, duration=150000, iterations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], win=1000)
    plot_winavg_vectors(df, "responseTime", start=10000, duration=150000, iterations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], win=1000)
    plot_winavg_vectors(df, "waitingTime", start=10000, duration=150000, iterations=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], win=1000)

    plot_ecdf_vec(df, "queueLength", iteration=0, sample_size=1000, replace=False)
    plot_ecdf_vec(df, "responseTime", iteration=0, sample_size=1000, replace=False)
    plot_ecdf_vec(df, "waitingTime", iteration=0, sample_size=1000, replace=False)

    check_iid_vec(df, "queueLength", iteration=0, sample_size=1000, seed=42, save=False)
    check_iid_vec(df, "responseTime", iteration=0, sample_size=1000, seed=42, save=False)
    check_iid_vec(df, "waitingTime", iteration=0, sample_size=1000, seed=42, save=False)

    lorenz_curve_vec(df, "queueLength")
    lorenz_curve_vec(df, "responseTime")
    lorenz_curve_vec(df, "waitingTime")

    vector_stats(df, group=False)

    describe_attribute_vec(df, "queueLength", iteration=0)
    describe_attribute_vec(df, "responseTime", iteration=0)
    describe_attribute_vec(df, "waitingTime", iteration=0)

    gini(df, precision=3)

if __name__ == '__main__':
    main()
