
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

from ml_utils import get_params, id_string, out_dir
from ml_preparation import feature_selection, data_balancing
from ml_classification import get_clf, hyperparam_opt, scorer, save_best_params
from datetime import datetime
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit

from matplotlib import pyplot
import pickle
import json
import gc
import os
import numpy as np


print("________________________________")
print("Starting")
# get pipeline customization parameters and setup output directories
params = get_params()
k = params["k"]
#k = 10
start_time = datetime.now()
time_str = start_time.strftime("%Y%m%d_%H%M%S")
job_id = id_string(params)
out_dir = out_dir(job_id + "__" + time_str)
perf_df = pd.DataFrame([])
precision_list = []
recall_list = []
fpr_list = []
tpr_list = []

def walk_forward_release(X, y, releases):
    """
    Generate train and test splits fro TimeSeriesSplit on releases.
    Train consists of a release or a list of successive releases, and
    the test set consist of the next release in time
    :param X: array-like of shape (n_samples, m_features)
    :param y: array-like of shape (n_samples,)
    :param releases : array-like of shape (n_samples,)
        Group labels for the samples used while splitting the dataset into
        train/test set.
        Must be a list of integer, i.e., [1, 1, 1, 2, 2, 3, 4, 4, etc.].
        Each integer denotes a release. Files within the same release have the same group id.
    """
    X, _, releases = indexable(X, y, releases)
    n_samples = _num_samples(X)
    n_folds = len(set(releases))  # Number of distinct groups (releases)

    if n_folds > n_samples:
        raise ValueError(f"Cannot have number of folds ={n_folds} greater than the number of samples: {n_samples}.")

    indices = np.arange(n_samples)
    offset = 0

    for _ in range(0, n_folds - 1):
        try:
            train_indices = [i for i, x in enumerate(releases) if x == releases[offset]]
            offset += len(train_indices)

            test_indices = [j for j, y in enumerate(releases) if y == releases[offset]]

            yield indices[:offset], indices[offset: offset + len(test_indices)]
        except IndexError:
            print("train_indices: ")
            print(train_indices)
            print("offset: ")
            print(offset)
            print("test_indices: ")
            print(test_indices)


def __init__(self):
    self.model = None

print("Started: " + start_time.strftime("%d %m %Y H%H:%M:%S"))
print("Results will be saved to dir: " + out_dir)
# get dataset
dataset_dir = "../dataset/isRefactored/" + params[
    "data"] + ".csv"  # Insert here your path

#this line is for RQ3    
df = pd.read_csv(dataset_dir)
df = df.fillna(0.0)

print("Loaded dataset of size: ")
print(df.shape)
print("Splitting dataset...")
# split dataset in k folds
unique_values_count = df['SHA'].nunique()


# Create column to group files belonging to the same release (identified by the commit hash)
df['group'] = df.SHA.astype('category').cat.rename_categories(range(1, df.SHA.nunique() + 1))

# Make sure the data is sorted by commit time (ascending)
df.sort_values(by=['SHA'], ascending=True)
df = df.reset_index(drop=True)

# Remove metadata
X, y = df.drop(['isRefactored'], axis=1), df.isRefactored.values.ravel()

releases = X.group.astype(int).tolist()
X.drop(['group'], axis=1, inplace=True)


folds = walk_forward_release(X,y,releases)

# del df
gc.collect()
i = 0
cols = df.select_dtypes([np.int64, np.int32]).columns
df[cols] = np.array(df[cols], dtype=float)

print_header = True

for train_index, test_index in folds:
    print("Round " + str(i + 1) + " of " + str(unique_values_count) + ": " + "Get data fold")

    # get train and test features and labels
    train = df.iloc[train_index]
    test = df.iloc[test_index]

    y_train = train["isRefactored"]
    y_train = y_train.astype(int)
    # this line is for RQ3
    X_train = train.drop(
        columns=['App','Repository','SHA','Tag','TestFilePath','isRefactored','group'])

    testset_sample = test[
        ['App','Repository','SHA','Tag','TestFilePath','isRefactored','group']]

    y_test = test["isRefactored"]
    y_test = y_test.astype(int)

    X_test = test.drop(
        columns=['App','Repository','SHA','Tag','TestFilePath','isRefactored','group'])
    
    print("Round " + str(i + 1) + " of " + str(unique_values_count) + ": " + "Data cleaning")
  

    columns_to_retain = X_train.columns
    X_test = X_test[columns_to_retain]

    print("Round " + str(i + 1) + " of " + str(unique_values_count) + ": " + "Feature selection")

    scaler = MinMaxScaler()
    scaled_values_train = scaler.fit_transform(X_train)
    scaled_values_test = scaler.fit_transform(X_test)

    X_train_scaled = pd.DataFrame(scaled_values_train, columns=X_train.columns)
    X_train = X_train_scaled


    X_test_scaled = pd.DataFrame(scaled_values_test, columns=X_test.columns)
    X_test = X_test_scaled
    print(X_test)

    data = []
    data = [columns_to_retain, mutual_info_classif(X_train[columns_to_retain], y_train, discrete_features = True)]


    data_T = pd.DataFrame(data).T
    data_T.columns = ["variable", "value"]


    data_filter = data_T[data_T.value > 0]
    print(data_T)
    print(data_filter)
    X_train = X_train[data_filter.variable]
    X_test = X_test[data_filter.variable]

    with open (out_dir + "IG/" + str(i) + ".txt", 'w') as f:
        dfAsString = data_T.to_string(header=False, index=False)
        f.write(dfAsString)
    del columns_to_retain
    gc.collect()


    # fix bug with numpy arrays
    X_train = X_train.values
    print(X_train)
    y_train = y_train.values.ravel()
    print(y_train)
    X_test = X_test.values
    y_test = y_test.values.ravel()

    # data balancing

    if not params["balancing"] == "none":
        print("Round " + str(i + 1) + " of " + str(unique_values_count) + ": " + "Data balancing")
        X_train, y_train = data_balancing(params["balancing"], X_train, y_train)

    # classifier 
    if not params["classifier"] == "none":
        clf_name = params["classifier"]
    else:
        clf_name = "dummy_random"
    clf = get_clf(clf_name)

    # hyperparameter opt

    if not params["optimization"] == "none" and not clf_name.startswith("dummy"):
        print("Round " + str(i + 1) + " of " + str(unique_values_count) + ": " + "Hyperparameters optimization")
        best_params = hyperparam_opt(clf, clf_name, params["optimization"], X_train, y_train)
        save_best_params(best_params, out_dir + "best_params/" + str(i))
        clf.set_params(**best_params)

    # validation 
    print("Round " + str(i + 1) + " of " + str(unique_values_count) + ": " + "Training")
    clf.fit(X_train, y_train)


    del X_train
    del y_train
    gc.collect()

    print("Round " + str(i + 1) + " of " + str(unique_values_count) + ": " + "Testing")
    fpr, tpr, res, y_pred = scorer(clf, clf_name, X_test, y_test)
    #precisionNew, recallNew, res, y_pred = scorer(clf, clf_name, X_test, y_test)
    y_pred = pd.DataFrame(y_pred, columns =["prediction"], index=testset_sample.index)
    y_pred.replace({0.0:False,1.0:True}, inplace=True)

    agreement = testset_sample.join(y_pred)
    mode = 'w' if print_header else 'a'
    agreement.to_csv(out_dir+"resultForTestCase.csv", mode=mode, header=print_header)
    print_header =False
    '''pyplot.figure()
    pyplot.plot(recallNew, precisionNew)
    pyplot.savefig(out_dir + "roc_curves/" + str(i) + ".png")'''
    fpr_list.append(fpr)
    tpr_list.append(tpr)

    perf_df = pd.concat([perf_df, res], ignore_index=True)
    perf_df = perf_df.dropna()

    del X_test
    del y_test
    gc.collect()

    i = i + 1


print("Saving performance")

sumTP = perf_df['tp'].sum()
sumFP = perf_df['fp'].sum()
sumTN = perf_df['tn'].sum()
sumFN = perf_df['fn'].sum()
meanPR = perf_df['precision'].mean()
meanRC = perf_df['recall'].mean()
meanACC = perf_df['accuracy'].mean()
meanIR = perf_df['inspection_rate'].mean()
meanF1 = perf_df['f1_score'].mean()
meanMCC = perf_df['mcc'].mean()
meanAUC = perf_df['auc_roc'].mean()

list = [sumTP, sumFP, sumTN, sumFN, meanPR, meanRC, meanACC, meanIR, meanF1, meanMCC, meanAUC]
#perf_df  = pd.read_csv('performance.csv')
perf_df = perf_df.append(pd.Series(list, index=perf_df.columns[:len(list)]), ignore_index=True)
perf_df.to_csv(out_dir + "performance.csv")


'''pyplot.figure()
for i in range(8):
    pyplot.plot(recallNew[i], precisionNew[i])
pyplot.savefig(out_dir + "pr_curves/all.png")'''

end_time = datetime.now()
print(params)

print("Ended: " + end_time.strftime("%d %m %Y H%H:%M:%S"))
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time}")
with open(out_dir + "elapsed_time.json", 'w') as f:
    f.write(json.dumps({"elapsed_time": f"{elapsed_time}"}))

print("________________________________")

