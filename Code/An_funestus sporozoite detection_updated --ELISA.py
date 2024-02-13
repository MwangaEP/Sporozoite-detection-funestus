# This programs is aiming to distinguish between Plasmodium infected and uninfected mosquitoes;

#%%
# Importing all modules

import os
import os.path
import io
import ast
import itertools
import collections
import json
from time import time
from tqdm import tqdm 

import numpy as np 
import pandas as pd 

import scipy.stats as stats
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

from random import randint
from collections import Counter 

import pickle

from sklearn.model_selection import (
                                        KFold, 
                                        ShuffleSplit, 
                                        train_test_split, 
                                        StratifiedShuffleSplit,
                                        cross_val_score,
                                        GridSearchCV,
                                        RandomizedSearchCV
                                    )


from sklearn.metrics import(
                                accuracy_score, 
                                classification_report, 
                                confusion_matrix, 
                                precision_recall_fscore_support
                            )

from sklearn.preprocessing import StandardScaler

from imblearn.under_sampling import (
                                        NearMiss, 
                                        CondensedNearestNeighbour, 
                                        OneSidedSelection
                                    )

from imblearn.ensemble import BalancedRandomForestClassifier 
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from sklearn import decomposition
from sklearn.feature_selection import SelectKBest

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.inspection import permutation_importance

from xgboost import XGBClassifier

import matplotlib.pyplot as plt # for making plots
import seaborn as sns
sns.set(
            context = "paper",
            style = "white",
            palette = "deep",
            font_scale = 2.0,
            color_codes = True,
            rc = ({"font.family": "Dejavu Sans"})
        )
%matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%

# define a convenient plotting function (confusion matrix)
def plot_confusion_matrix(
                            cm, classes,
                            normalise = True,
                            text = False,
                            title = 'Confusion matrix',
                            xrotation = 0,
                            yrotation = 0,
                            cmap = plt.cm.Blues,
                            printout = False
                        ):

    """
    This function prints and plots the confusion matrix.
    Normalisation can be applied by setting 'normalise=True'.
    """

    if normalise:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        if printout:
            print("Normalized confusion matrix")
        
    else:
        if printout:
            print('Confusion matrix')

    if printout:
        print(cm)

    # plt.figure(figsize=(6, 4))
    plt.imshow(
                cm, 
                interpolation = 'nearest', 
                cmap = cmap, 
                vmin = 0.2, 
                vmax = 1.0
            )

    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    # plt.set_ylim(len(classes)-0.5, -0.5)
    plt.xticks(tick_marks, classes, rotation=xrotation)
    plt.yticks(tick_marks, classes, rotation=yrotation)

    fmt = '.2f' if normalise else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]),
                                  range(cm.shape[1])):
            plt.text(
                        j, 
                        i, 
                        format(cm[i, j], fmt), 
                        horizontalalignment = "center",
                        color = "white" if cm[i, j] > thresh else "black"
                    )

    plt.tight_layout()
    plt.ylabel('True label', weight = 'bold')
    plt.xlabel('Predicted label', weight = 'bold')

#%%

#  Define the base directory
base_directory = r"C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML_final_analysis\Results\ELISA model_1"

# Create a function to generate paths within the base directory
def generate_path(*args):
    return os.path.join(base_directory, *args)


#%%
df = pd.read_csv(os.path.join("..", "Data", "sporozoite_full.csv"))
df.head()

#%%
# data shape
print(df.shape)

# Checking class distribution abd correlation in the data
Counter(df["Sporozoite"])


#%%
# Select vector of labels and matrix of features

X = df.iloc[:,7:] # matrix of features
features  = X
y = df["Sporozoite"] # vector of labels
X


#%%

# rescalling the data (undersampling the over respresented class - negative class)

rus = NearMiss(version = 2, n_neighbors = 3)
X_res, y_res = rus.fit_resample(X, y)
print(collections.Counter(y_res))

#  Get the indices of the samples that were not resampled
indices_not_resampled = np.setdiff1d(
                                        np.arange(len(X)), 
                                        rus.sample_indices_
                                    )

# Create a DataFrame with the remaining samples
remaining_samples = pd.DataFrame(
                                    X.iloc[indices_not_resampled], 
                                    columns = X.columns
                                )

remaining_samples['Sporozoite'] = y.iloc[indices_not_resampled]

# shift column 'Name' to first position
first_column = remaining_samples.pop('Sporozoite')
  
# insert column using insert(position,column_name, first_column) function
remaining_samples.insert(0, 'Sporozoite', first_column)

# data splitting 
X_train, X_val, y_train, y_val = train_test_split(
                                                    X_res, 
                                                    y_res, 
                                                    test_size= .1, 
                                                    random_state = 42, 
                                                    shuffle = True
                                                )

print('The shape of X train index : {}'.format(X_train.shape))
print('The shape of y train index : {}'.format(y_train.shape))
print('The shape of X val index : {}'.format(X_val.shape))
print('The shape of y val index : {}'.format(y_val.shape))


# prepare training data
training_temp = pd.concat(
                            [
                                y_train, 
                                X_train
                            ],
                            axis = 1
                        )

training_df = pd.concat(
                            [
                                training_temp, 
                                remaining_samples
                            ],
                            axis = 0
                        ).reset_index(drop = True)                  

#%%
# standardisation 

X_new = np.asarray(X_train)
y_new = np.asarray(y_train)
print('y labels : {}'.format(np.unique(y)))

scl = StandardScaler().fit(X = X_new)
X_scl  = scl.transform(X = X_new)

#%%

# Data splitting and defining models
num_folds = 5 # Spliting the training set into 6 parts
validation_size = 0.1 # defining the size of the validation set
seed = 42
SEED = np.random.randint(0, 81478)
scoring = 'accuracy' # score model accuracy

kf = KFold(
            n_splits = num_folds, 
            shuffle = True, 
            random_state = SEED
        )

# kf = StratifiedShuffleSplit(
#         n_splits=num_folds, test_size=validation_size, random_state=seed)


models = [] # telling python to create sub names models
models.append(
                (
                    "KNN", KNeighborsClassifier()
                )
            )
models.append(
                (
                    "LR", LogisticRegressionCV(
                                                multi_class = 'ovr', 
                                                cv = kf, 
                                                random_state = SEED, 
                                                max_iter = 2000
                                            )
                )
            )
models.append(
                (
                    "SVM", SVC(
                                random_state = SEED, 
                                kernel = 'linear', 
                                gamma = 'auto'
                            )
                )
            )
models.append(
                (
                    "XGB", XGBClassifier(
                                            random_state = SEED, 
                                            nthread = 1
                                        )
                )
            )
models.append(
                (
                    "RF", RandomForestClassifier(
                                                    random_state = SEED, 
                                                    n_estimators = 300
                                                )
                )
            )
models.append(
                (
                    "MLP", MLPClassifier(
                                            random_state = SEED, 
                                            max_iter = 1000,
                                            solver = 'adam',
                                            activation = 'logistic', 
                                            alpha = 0.01 )
                )
            )


#%%

# comparative evaluation of different classifiers
results = []
names = []

for name, model in models:
    cv_results = cross_val_score(
                                    model, 
                                    X_scl, 
                                    y_new, 
                                    cv = kf, 
                                    scoring = scoring
                                )

    results.append(cv_results)
    names.append(name)
    msg = "Cross val score for {0}: {1:.2%} ± {2:.2%}".format(
                                                                name, 
                                                                cv_results.mean(), 
                                                                cv_results.std()
                                                            )
    print(msg)


#%%
# plotting the results of the classifiers


results_df = pd.DataFrame(
                            results, 
                            columns = (0, 1, 2, 3, 4)
                        ).T # columns should correspond to the number of folds, k = 5

# rename columns to have number of components

# results_df.columns = ['KNN', 'LR', 'SVM', 'RF', 'XGB']
results_df.columns = names
results_df = pd.melt(results_df) # melt data frame into a long format. 

results_df.rename(
                    columns = {'variable':'Model', 'value':'Accuracy'}, 
                    inplace = True
                )


sns.boxplot(
                x = results_df['Model'], 
                y = results_df['Accuracy']
            )
sns.despine(offset=10, trim=True)
# plt.title("Algorithm comparison", weight="bold")
plt.xticks(rotation=90)
plt.yticks(np.arange(0.2, 1.0 + .05, step = 0.2))
plt.xlabel(' ')
plt.ylabel('Accuracy', weight = 'bold');
plt.savefig(
                generate_path("Algorithm selection_elisa.png"), 
                dpi = 500, 
                bbox_inches = "tight"
            )


#%%

# big LOOP
# TUNNING THE SELECTED MODEL

## Set validation procedure
num_folds = 5 # split training set into 5 parts for validation
num_rounds = 50 # increase this to 5 or 10 once code is bug-free
# seed = 4 # pick any integer. This ensures reproducibility of the tests
scoring = 'accuracy' # score model accuracy

# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results = [] # per class accuracy scores
save_predicted, save_true = [], [] # save predicted and true values for each loop

start = time()

# Specify model

classifier = XGBClassifier(random_state = seed, nthread=1)

# set hyparameter

estimators = [500, 1000]
rate = [0.05, 0.10, 0.15, 0.20, 0.30]
depth = [2, 3, 4, 5, 6, 8, 10, 12, 15]
child_weight = [1, 3, 5, 7]
gamma = [0.0, 0.1, 0.2, 0.3, 0.4]
bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]

random_grid = {
                'n_estimators': estimators, 
                'learning_rate': rate, 
                'max_depth': depth,
                'min_child_weight': child_weight, 
                'gamma': gamma, 
                'colsample_bytree': bytree
            } 

# Prepare data 
X = np.asarray(training_df.iloc[:,1:])
y = np.asarray(training_df['Sporozoite'])

# under-sample over-represented classes (Negative class)

for round in range (num_rounds):
    SEED = np.random.randint(0, 81478)

    X_resampled, y_resampled = rus.fit_resample(X, y)
    
    # cross validation and splitting of the validation set
    for train_index, test_index in kf.split(X_resampled, y_resampled):
        X_train, X_test = X_resampled[train_index], X_resampled[test_index]
        y_train, y_test = y_resampled[train_index], y_resampled[test_index]

        
        # standardise features using standard scaler

        X_train  = scl.transform(X = X_train)
        X_test = scl.transform(X = X_test)

        print('The shape of X train index : {}'.format(X_train.shape))
        print('The shape of y train index : {}'.format(y_train.shape))
        print('The shape of X test index : {}'.format(X_test.shape))
        print('The shape of y test index : {}'.format(y_test.shape))

        # # Specify model

        # classifier = RandomForestClassifier()

        # # Optimizing hyper-parameters for random forest

        # # Number of trees in random forest
        # n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
        # # Number of features to consider at every split
        # max_features = ['auto', 'sqrt']
        # # Maximum number of levels in tree
        # max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
        # max_depth.append(None)
        # # Minimum number of samples required to split a node
        # min_samples_split = [2, 5, 10]
        # # Minimum number of samples required at each leaf node
        # min_samples_leaf = [1, 2, 4]
        # # Method of selecting samples for training each tree
        # bootstrap = [True, False]
        
        # # Create the random grid
        # random_grid = {'n_estimators': n_estimators,
        #             'max_features': max_features,
        #             'max_depth': max_depth,
        #             'min_samples_split': min_samples_split,
        #             'min_samples_leaf': min_samples_leaf,
        #             'bootstrap': bootstrap}
        # print(random_grid)

        # generate models using all combinations of settings

        # RANDOMSED GRID SEARCH
        # Random search of parameters, using 5 fold cross validation, 
        # search across 100 different combinations, and use all available cores

        n_iter_search = 10
        rsCV = RandomizedSearchCV(
                                    verbose = 1,
                                    estimator = classifier, 
                                    param_distributions = random_grid, 
                                    n_iter = n_iter_search, 
                                    scoring = scoring, 
                                    cv = kf, 
                                    refit = True, 
                                    n_jobs = -1
                                )
        
        rsCV_result = rsCV.fit(X_train, y_train)

        # print out results and give hyperparameter settings for best one
        means = rsCV_result.cv_results_['mean_test_score']
        stds = rsCV_result.cv_results_['std_test_score']
        params = rsCV_result.cv_results_['params']
        for mean, stdev, param in zip(means, stds, params):
            print("%.2f (%.2f) with: %r" % (mean, stdev, param))

        # print best parameter settings
        print("Best: %.2f using %s" % (rsCV_result.best_score_,
                                    rsCV_result.best_params_))

        # Insert the best parameters identified by randomized grid search into the base classifier
        best_classifier = classifier.set_params(**rsCV_result.best_params_)
       
       
        best_classifier.fit(X_train, y_train)

        # predict test instances 

        y_pred = best_classifier.predict(X_test)
        # y_test = np.delete(y_res, train_index, axis=0)
        local_cm = confusion_matrix(y_test, y_pred)
        local_report = classification_report(y_test, y_pred)

        # zip predictions for all rounds for plotting averaged confusion matrix
        
        for predicted, true in zip(y_pred, y_test):
            save_predicted.append(predicted)
            save_true.append(true)

       # append feauture importances
        local_feat_impces = pd.DataFrame(
                                            best_classifier.feature_importances_,
                                            index = features.columns
                                        ).sort_values(by = 0, ascending = False)
    
        # summarizing results
        local_kf_results = pd.DataFrame(
                                            [
                                                ("Accuracy", accuracy_score(y_test, y_pred)), 
                                                ("TRAIN",str(train_index)), 
                                                ("TEST",str(test_index)), 
                                                ("CM", local_cm), 
                                                ("Classification report", local_report), 
                                                ("y_test", y_test),
                                                ("Feature importances", local_feat_impces.to_dict())
                                            ]
                                        ).T
        
        local_kf_results.columns = local_kf_results.iloc[0]
        local_kf_results = local_kf_results[1:]
        # kf_results = kf_results.append(local_kf_results)

        kf_results = pd.concat(
                                [kf_results, local_kf_results],
                                axis = 0,
                                join = 'outer'
                            ).reset_index(drop = True)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_test, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        kf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
    elapsed / 60, elapsed))


#%%

with open(generate_path('classifier.pkl'), 'wb') as fid:
     pickle.dump(best_classifier, fid)

#%%

# plot averaged confusion matrix for training
averaged_CM = confusion_matrix(save_true, save_predicted)

classes = np.unique(np.sort(y))

plt.rcParams["figure.figsize"] = [6,4]
plot_confusion_matrix(averaged_CM, classes)
plt.savefig(
                generate_path("_averaged_CM_full_wn_elisa.png"), 
                dpi = 500, 
                bbox_inches = "tight"
            )

#%%
# Results
kf_results.to_csv(
                    generate_path("crf_kfCV_record.csv"), 
                    index = False
                )
kf_results = pd.read_csv(generate_path("crf_kfCV_record.csv"))

# Accuracy distribution
crf_acc_distrib = kf_results["Accuracy"]
crf_acc_distrib.columns=["Accuracy"]
crf_acc_distrib.to_csv(
                            generate_path("crf_acc_distrib.csv"), 
                            header = True, 
                            index = False
                        )
crf_acc_distrib = pd.read_csv(generate_path("crf_acc_distrib.csv"))
crf_acc_distrib = np.round(crf_acc_distrib, 2)
print(crf_acc_distrib)

#%%
# plotting accuracy distribution
plt.figure(figsize=(2.25,3))
sns.displot(
                crf_acc_distrib, 
                kde = False, 
                bins = 12
            )
# plt.savefig("lgr_acc_distrib.png", bbox_inches="tight")

#%%
# class distribution 
class_names = np.unique(np.sort(y))
crf_per_class_acc_distrib = pd.DataFrame(kf_per_class_results, columns=class_names)
crf_per_class_acc_distrib.dropna().to_csv(generate_path("crf_per_class_acc_distrib.csv"))
crf_per_class_acc_distrib = pd.read_csv(generate_path("crf_per_class_acc_distrib.csv"), index_col = 0)
crf_per_class_acc_distrib = np.round(crf_per_class_acc_distrib, 2)
crf_per_class_acc_distrib_describe = crf_per_class_acc_distrib.describe()
crf_per_class_acc_distrib_describe.to_csv(generate_path("crf_per_class_acc_distrib.csv"))


#%%
# plotting class distribution
lgr_per_class_acc_distrib = pd.melt(
                                    crf_per_class_acc_distrib, 
                                    var_name = "status new"
                                )

plt.figure(figsize=(6,4))

sns.violinplot(
                x = "status new", 
                y = "value", 
                cut = 1, 
                data = lgr_per_class_acc_distrib
            )

sns.despine(left = True)
plt.xticks(rotation = 0, ha = "right")
plt.xticks()
plt.yticks(np.arange(0.2, 1.0 + .05, step = 0.2))
plt.xlabel(" ")
plt.ylabel("Prediction accuracy", weight = "bold")
plt.savefig(
                generate_path("per_class_acc_distrib_full_wn.png"), 
                dpi = 500, 
                bbox_inches="tight"
            )


#%% Feature Importances

# make this into bar with error bars across all best models

rskf_results = pd.read_csv(generate_path("crf_kfCV_record.csv"))

# All feat imp
all_featimp = pd.DataFrame(ast.literal_eval(rskf_results["Feature importances"][0]))

for featimp in rskf_results["Feature importances"][1:]:
    featimp = pd.DataFrame(ast.literal_eval(featimp))
    all_featimp = pd.concat(
                                [
                                    all_featimp, 
                                    featimp
                                ], 
                                axis = 1, 
                                ignore_index = True
                            )

all_featimp["mean"] = all_featimp.mean(axis = 1)
all_featimp["sem"] = all_featimp.sem(axis = 1)
all_featimp.sort_values(by = "mean", inplace = True)

featimp_global_mean = all_featimp["mean"].mean()
featimp_global_sem = all_featimp["mean"].sem()


fig = all_featimp["mean"][-50:].plot(
                                        figsize = (3, 14),
                                        kind = "barh",
                                        # orientation = 'vertical',
                                        legend = False,
                                        xerr = all_featimp["sem"],
                                        ecolor = 'k'
                                    )
plt.xlabel("Feature importance", weight = 'bold')
# plt.axvspan(xmin=0, xmax=featimp_global_mean+3*featimp_global_sem,facecolor='r', alpha=0.3)
# plt.axvline(x=featimp_global_mean, color="r", ls="--", dash_capstyle="butt")
sns.despine()

# Add mean accuracy of best models to plots
plt.annotate("Average Accuracy:\n{0:.3f} ± {1:.3f}".format(
                                                            crf_acc_distrib.mean()[0], 
                                                            crf_acc_distrib.sem()[0]
                                                            ), 
                                                            xy = (0.06, 0), color = "k")

plt.savefig(
                generate_path("_feature_impces_full_wn.png"), 
                dpi = 500, 
                bbox_inches = "tight"
            )

#%%
# Predict validation data

# Transform data using the mean and standard deviation from the model training data

trans_X_val = scl.transform(X = np.array(X_val))

# Predict test set

y_val_pred = best_classifier.predict(trans_X_val)

accuracy_val_elisa = accuracy_score(y_val, y_val_pred)
print("Accuracy: %.2f%%" % (accuracy_val_elisa * 100.0))

#%%
# Plotting confusion matrix for X_val 

plt.rcParams["figure.figsize"] = [6,4]
cm_val = confusion_matrix(y_val, y_val_pred)

plot_confusion_matrix(
                        cm_val, 
                        text = True, 
                        normalise = True, 
                        classes = class_names
                    )

plt.savefig(
                generate_path("CM_full_wn_val_elisa.png"), 
                dpi = 500, 
                bbox_inches = "tight"
            )

#%%
# Summarising precision, f_score, and recall for the validation set
cr_full_wn_val = classification_report(y_val, y_val_pred)
print('Classification report : {}'.format(cr_full_wn_val))

# save classification report to disk as a csv

cr_full_wn_val = pd.read_fwf(io.StringIO(cr_full_wn_val), header=0)
cr_full_wn_val = cr_full_wn_val.iloc[0:]
cr_full_wn_val.to_csv(generate_path("classification_report_full_wn_val_elisa.csv"))


#%%

# Collect all the important wavenumbers
important_wavenumb = pd.DataFrame(all_featimp["mean"][-100:])
important_wavenumb = important_wavenumb.reset_index()
important_wavenumb = important_wavenumb['index'].to_list()

with open(generate_path('important_wavenumbers.txt'), 'w') as outfile:
     json.dump(important_wavenumb, outfile)

# wnumber = [ int(x) for x in important_wavenumb]
# wnumber.sort
# print(wnumber)

###################################################################################################################

#%%

"""

# Predicting the sporozoite with age data included in test set

"""

# load age data
df_age = pd.read_csv(os.path.join("..", "Data", "wild_funestus_age.dat"), delimiter = '\t')

df_age = df_age.query("Cat3 == '14D'")
df_age = df_age.drop(['Cat1', 'Cat2', 'Cat3', 'Cat4', 'StoTime'], axis = 1)

df_age['Sporozoite'] = 'Negative'

# shift column 'Name' to first position
first_column = df_age.pop('Sporozoite')
  
# insert column using insert(position,column_name,
# first_column) function
df_age.insert(0, 'Sporozoite', first_column)

df_age.head()

#%%

new_val_df = pd.concat([y_val, X_val], axis = 1)

# # shift column 'Name' to first position
# first_column_ = new_val_df.pop('Sporozoite')
  
# # insert column using insert(position,column_name,
# # first_column) function
# new_val_df.insert(0, 'Sporozoite', first_column_)
new_val_df

#%%
# concatinate age data and PCR data

new_data_age_df = pd.concat(
                                [
                                    new_val_df, 
                                    df_age
                                ], 
                                axis = 0, 
                                join = 'outer'
                            )


print(collections.Counter(new_data_age_df['Sporozoite']))
new_data_age_df

#%%

X_val_2 = new_data_age_df.iloc[:,1:] # matrix of features
y_val_2 = new_data_age_df["Sporozoite"] # vector of labels
print(collections.Counter(y_val_2))

X_res_val, y_res_val = rus.fit_resample(X_val_2, y_val_2)
print(collections.Counter(y_res_val))

# standardize data
X_res_val_trans = scl.transform(np.asarray(X_res_val))
y_res_val = np.asarray(y_res_val)

#%%

# loading the classifier from the disk
# with open('E:\Sporozoite\Results\classifier.pkl', 'rb') as fid:
#      classifier_loaded = pickle.load(fid)

# generates output predictions based on the X_input passed

predictions = best_classifier.predict(X_res_val_trans)

# Examine the accuracy of the model to see whether age has an impact in the model  

accuracy = accuracy_score(y_res_val, predictions)
print("Accuracy:%.2f%%" %(accuracy * 100.0))

#%%
plt.rcParams["figure.figsize"] = [6,4]

cm_val_2 = confusion_matrix(y_res_val, predictions)
figure_name = 'age_sporozoite_elisa'
# visualize(figure_name, classes, y_res_val, predictions)

plot_confusion_matrix(
                        cm_val_2, 
                        text = True, 
                        normalise = True, 
                        classes = class_names
                    )
plt.savefig(
                generate_path("CM_age_sporozoite_elisa.png"), 
                dpi = 500, 
                bbox_inches = "tight"
            )

#%%
# Summarising precision, f_score, and recall for the validation set
cr_full_wn_val_age = classification_report(y_res_val, predictions)
print('Classification report : {}'.format(cr_full_wn_val_age))

# save classification report to disk as a csv

cr_full_wn_val_age = pd.read_fwf(io.StringIO(cr_full_wn_val_age), header=0)
cr_full_wn_val_age = cr_full_wn_val_age.iloc[0:]
cr_full_wn_val_age.to_csv(generate_path("classification_report_full_wn_val_age.csv"))

###################################################################################################################

#%%

"""

# Predicting PCR data using the model trained on ELISA data

"""
# Importing the full spectra dataset with biological attributes 

full_bio_attr_df = pd.read_csv(os.path.join("..", "Data", "Biological_attr.dat"), delimiter = '\t')
full_bio_attr_df.head()

#%%
# checking data shape
print(full_bio_attr_df.shape)

# Checking class distribution abd correlation in the data
Counter(full_bio_attr_df["Cat7"])

#%%

# Selecting/subsetting only head and thorax data (the one screened for infection)

head_and_thrx_df = full_bio_attr_df.query("Cat7 == 'HD'")
print('The shape of head and thorax data : {}'.format(head_and_thrx_df.shape))

# Observe first few observations
head_and_thrx_df.head()

# %%

# Import PCR results which contains the ID's of positive mosquitoes 
pcr_data_df = pd.read_csv(os.path.join("..", "Data", "PCR data-35cycles-Cy5-FAM.csv"))
pcr_data_df.head()

# %%

# Select a vector of sample ID from PCR data and use it to index all the positive 
# from the head and thorax data

positive_samples = pcr_data_df['Sample']
positive_samples_df = head_and_thrx_df.query("ID in @positive_samples")

# create a new column in positive samples dataframe and name the samples as positives
positive_samples_df['infection_status'] = 'Positive'

#%%

# Index all the negative from the head and thorax data
# Select all rows not in the list

negative_samples_df = head_and_thrx_df[~head_and_thrx_df['ID'].isin(positive_samples)]
negative_samples_df['infection_status'] = 'Negative'

# %%

# Concatinating positive and negative dataframes together
infection_data_df = pd.concat(
                                [
                                    positive_samples_df, 
                                    negative_samples_df
                                ], 
                                axis = 0, 
                                join = 'outer'
                            )
infection_data_df

# %%

infection_data_df = infection_data_df.drop(
                                            [
                                                'ID', 
                                                'Cat2', 
                                                'Cat3', 
                                                'Cat4', 
                                                'Cat5', 
                                                'Cat6', 
                                                'Cat7', 
                                                'StoTime'
                                            ], 
                                            axis = 1
                                        )
infection_data_df

#%%
# rescalling the data (undersampling the over respresented class - negative class)

X_val_pcr = infection_data_df.iloc[:,:-1] # select all columns except the last one

y_val_pcr = infection_data_df["infection_status"]

print('shape of X : {}'.format(X.shape))
print('shape of y : {}'.format(y.shape))

X_res_pcr, y_res_pcr = rus.fit_resample(X_val_pcr, y_val_pcr)
print(collections.Counter(y_res_pcr))

X_res_val_trans_pcr = scl.transform(np.asarray(X_res_pcr))
y_res_val_pcr = np.asarray(y_res_pcr)

#%%

# generates output predictions based on the X_input passed from PCR data

# with open("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML_final_analysis\Results\ELISA model\classifier.pkl", 'rb') as fid:
#      classifier_loaded = pickle.load(fid)

predictions_pcr = best_classifier.predict(X_res_val_trans_pcr)

# Examine the accuracy of the model  

accuracy_pcr = accuracy_score(y_res_val_pcr, predictions_pcr)
print("Accuracy:%.2f%%" %(accuracy_pcr * 100.0))

#%%

plt.rcParams["figure.figsize"] = [6,4]

class_names = np.unique(np.sort(y_res_val_pcr))
cm_val_3 = confusion_matrix(y_res_val_pcr, predictions_pcr)
# figure_name = 'pcr_sporozoite'
# visualize(figure_name, classes, y_res_val_elisa, predictions_elisa)

plot_confusion_matrix(
                        cm_val_3, 
                        text = True, 
                        normalise = True, 
                        classes = class_names
                    )
plt.savefig(
                generate_path("CM_full_wn_val_pcr.png"), 
                dpi = 500, 
                bbox_inches = "tight"
            )

#%%

# Summarising precision, f_score, and recall for the validation set
cr_full_wn_val_pcr = classification_report(y_res_val_pcr, predictions_pcr)
print('Classification report : {}'.format(cr_full_wn_val_pcr))

# save classification report to disk as a csv

cr_full_wn_val_pcr = pd.read_fwf(io.StringIO(cr_full_wn_val_pcr), header=0)
cr_full_wn_val_pcr = cr_full_wn_val_pcr.iloc[0:]
cr_full_wn_val_pcr.to_csv(generate_path("classification_report_full_wn_val_pcr.csv"))
