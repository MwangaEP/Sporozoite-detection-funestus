#%%
import os
import io
import json
import ast
import itertools
import collections
from time import time
from tqdm import tqdm

from itertools import cycle
import pickle
import random as rn
import datetime

import numpy as np 
import pandas as pd
from scipy.signal import savgol_filter

from random import randint
from collections import Counter 

from sklearn.model_selection import (
                                        ShuffleSplit, 
                                        train_test_split, 
                                        StratifiedKFold, 
                                        StratifiedShuffleSplit, 
                                        KFold
                                    ) 

from sklearn.model_selection import (
                                        RandomizedSearchCV, 
                                        GridSearchCV, 
                                        cross_val_score
                                    )

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
                                accuracy_score, 
                                confusion_matrix, 
                                classification_report, 
                                precision_recall_fscore_support
                            )

from imblearn.under_sampling import (
                                        RandomUnderSampler, 
                                        NearMiss, 
                                        InstanceHardnessThreshold
                                    )

from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

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
# %matplotlib inline
plt.rcParams["figure.figsize"] = [6,4]

#%%

#  Define the base directory
base_directory = r"C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML_final_analysis\Results\PCR model_final"

# Create a function to generate paths within the base directory
def generate_path(*args):
    return os.path.join(base_directory, *args)

# This normalizes the confusion matrix and ensures neat plotting for all outputs.
# Function for plotting confusion matrcies

def plot_confusion_matrix(
                            cm, classes,
                            normalize = True,
                            title = 'Confusion matrix',
                            xrotation=0,
                            yrotation=0,
                            cmap=plt.cm.Blues,
                            printout = False
                        ):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        if printout:
            print("Normalized confusion matrix")
    else:
        if printout:
            print('Confusion matrix')

    if printout:
        print(cm)
    
    plt.figure(figsize=(6,4))

    plt.imshow(
                cm, 
                interpolation = 'nearest', 
                vmin = .2, 
                vmax = 1.0,  
                cmap = cmap
            )
    
    # plt.title(title)
    plt.colorbar()
    classes = classes
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = xrotation)
    plt.yticks(tick_marks, classes, rotation = yrotation)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
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
    plt.savefig(
                    generate_path(
                                    "Confusion_Matrix_" 
                                    + figure_name 
                                    + "_" 
                                    + ".png"
                                ), 
                                dpi = 500, 
                                bbox_inches = "tight"
                )

#%%

def visualize(figure_name, classes, true, predicted):
    # Sort out predictions and true labels
    # for label_predictions_arr, label_true_arr, classes, outputs in zip(predicted, true, classes, outputs):
#     print('visualize predicted classes', predicted)
#     print('visualize true classes', true)
    classes_pred = np.asarray(predicted)
    classes_true = np.asarray(true)
    print(classes_pred.shape)
    print(classes_true.shape)
    cnf_matrix = confusion_matrix(classes_true, classes_pred, labels = classes)
    plot_confusion_matrix(cnf_matrix, classes)


#%%
# loading full spectra dataset

full_data_df = pd.read_csv(
                            os.path.join("..", "Data", "Biological_attr.dat"), 
                            delimiter = '\t'
                        )

full_data_df.head()

#%%
# checking data shape
print(full_data_df.shape)

# counting class distribution 
Counter(full_data_df["Cat7"])

#%%

# Selecting/subseting only head and thorax data

head_and_thrx_df = full_data_df.query("Cat7 == 'HD'")
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
# Select all rows not in the positive list

negative_samples_df = head_and_thrx_df[~head_and_thrx_df['ID'].isin(list(positive_samples))]
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

# %%

# define X (matrix of features) and y (list of labels)

X = infection_data_df.iloc[:,:-1] # select all columns except the last one 
y = infection_data_df["infection_status"]

print('shape of X : {}'.format(X.shape))
print('shape of y : {}'.format(y.shape))

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

remaining_samples['infection_status'] = y.iloc[indices_not_resampled]

# shift column 'Name' to first position
first_column = remaining_samples.pop('infection_status')
  
# insert column using insert(position,column_name, first_column) function
remaining_samples.insert(0, 'infection_status', first_column)


#%%

# Split data

X_train, X_test, y_train, y_test = train_test_split(
                                                        X_res, 
                                                        y_res, 
                                                        test_size= .1, 
                                                        random_state = 42, 
                                                        shuffle = True
                                                    )

print('The shape of X train index : {}'.format(X_train.shape))
print('The shape of y train index : {}'.format(y_train.shape))
print('The shape of X val index : {}'.format(X_test.shape))
print('The shape of y val index : {}'.format(y_test.shape))

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

# Standardisation

X_new = np.asarray(X_train)
y_new = np.asarray(y_train)
print('y labels : {}'.format(np.unique(y_new)))

# standardisation 
scaler = StandardScaler().fit(X = X_new)
X_transformed  = scaler.transform(X = X_new)

# %%

# Data splitting and defining models
num_folds = 5 # Spliting the training set into 5 parts
validation_size = 0.1 # defining the size of the validation set
seed = 42
SEED = np.random.randint(0, 81478)
scoring = 'accuracy' # score model accuracy

kf = KFold(
            n_splits = num_folds, 
            shuffle = True, 
            random_state = SEED
        )

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
                                            random_state = seed, 
                                            nthread = 1
                                        )
                )
            )
models.append(
                (
                    "RF", RandomForestClassifier(
                                                    random_state = seed, 
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
                                            alpha = 0.01
                                        )
                )
            )


#%%

# comparative evaluation of different classifiers

results = []
names = []

for name, model in models:
    cv_results = cross_val_score(
                                    model, 
                                    X_transformed, 
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

# plotting
plt.rcParams["figure.figsize"] = [6,4]

sns.boxplot(
                x = results_df['Model'], 
                y = results_df['Accuracy']
            )
sns.despine(offset = 10, trim = True)
# plt.title("Algorithm comparison", weight="bold")
plt.xticks(rotation = 90)
plt.yticks(np.arange(0.2, 1.0 + .05, step = 0.2))
plt.xlabel(' ')
plt.ylabel('Accuracy', weight = 'bold');
plt.savefig(
                generate_path("_algo_selection_.png"), 
                dpi = 500, 
                bbox_inches = "tight"
            )

#%%

# big LOOP
# TUNNING THE SELECTED MODEL

## Set validation procedure

num_rounds = 50 # increase this to 5 or 10 once code is bug-free
scoring = 'accuracy' # score model accuracy

# prepare matrices of results
kf_results = pd.DataFrame() # model parameters and global accuracy score
kf_per_class_results = [] # per class accuracy scores
sss_coef = pd.DataFrame()
save_predicted, save_true = [], [] # save predicted and true values for each loop

start = time()

# Specify model

classifier = XGBClassifier(random_state = seed, nthread = 1)

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
y = np.asarray(training_df['infection_status'])

for round in range (num_rounds):

    SEED = np.random.randint(0, 81478)

    X_resampled, y_resampled = rus.fit_resample(X, y)

    # cross validation and splitting of the validation set
    for train_index, test_index in kf.split(X_resampled, y_resampled):
        X_train_set, X_val = X_resampled[train_index], X_resampled[test_index]
        y_train_set, y_val = y_resampled[train_index], y_resampled[test_index]

            
        # standardise features using standard scaler

        X_train_set  = scaler.transform(X = X_train_set)
        X_val = scaler.transform(X = X_val)

        print('The shape of X train set : {}'.format(X_train_set.shape))
        print('The shape of y train set  : {}'.format(y_train_set.shape))
        print('The shape of X val set : {}'.format(X_val.shape))
        print('The shape of y val set : {}'.format(y_val.shape))

    
        # classifier = LogisticRegression(multi_class = 'ovr', random_state = seed, max_iter = 6000)


        # Optimizing hyper-parameters for random forest

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
        #                 'max_features': max_features,
        #                 'max_depth': max_depth,
        #                 'min_samples_split': min_samples_split,
        #                 'min_samples_leaf': min_samples_leaf,
        #                 'bootstrap': bootstrap}
        # # print(random_grid)

        # solvers = ['newton-cg', 'sag', 'saga', 'lbfgs']
        # c_values = [0.01, 0.1, 1, 10]     # 100, 50, 20,  10, 5]
                
        # # Create the random grid

        # random_grid = {'solver': solvers,
        #                'C': c_values}


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
                                    refit = True, 
                                    cv = kf
                                )
            
        rsCV_result = rsCV.fit(X_train_set, y_train_set)

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
        
        
        best_classifier.fit(X_train_set, y_train_set)

        # predict test instances 

        y_pred = best_classifier.predict(X_val)
        # y_test = np.delete(y_res, train_index, axis=0)
        local_cm = confusion_matrix(y_val, y_pred)
        local_report = classification_report(y_val, y_pred)

        # zip predictions for all rounds for plotting averaged confusion matrix
            
        for predicted, true in zip(y_pred, y_val):
            save_predicted.append(predicted)
            save_true.append(true)

        # append feauture importances
        local_feat_impces = pd.DataFrame(
                                            best_classifier.feature_importances_,
                                            index = infection_data_df.iloc[:,:-1].columns
                                        ).sort_values(by = 0, ascending = False)

        # summarizing results
        local_kf_results = pd.DataFrame(
                                            [
                                                ("Accuracy", accuracy_score(y_val, y_pred)), 
                                                ("TRAIN",str(train_index)), 
                                                ("TEST",str(test_index)), 
                                                ("CM", local_cm), 
                                                ("Classification report", local_report), 
                                                ("y_test", y_val),
                                                ("Feature importances", local_feat_impces.to_dict())
                                            ]
                                        ).T
            
        local_kf_results.columns = local_kf_results.iloc[0]
        local_kf_results = local_kf_results[1:]

        kf_results = pd.concat(
                                [kf_results, local_kf_results],
                                axis = 0,
                                join = 'outer'
                            ).reset_index(drop = True)

        # per class accuracy
        local_support = precision_recall_fscore_support(y_val, y_pred)[3]
        local_acc = np.diag(local_cm)/local_support
        kf_per_class_results.append(local_acc)

elapsed = time() - start
print("Time elapsed: {0:.2f} minutes ({1:.1f} sec)".format(
elapsed / 60, elapsed))


#%%
# plot confusion averaged for the validation set

classes = np.unique(np.sort(y_new))
figure_name = 'validation_model_train_data'
visualize(figure_name, classes, save_true, save_predicted)

#%%
# save the trained model to disk for future use

with open(generate_path('classifier.pkl'), 'wb') as fid:
     pickle.dump(best_classifier, fid)

# %%

# Results
kf_results.to_csv(generate_path("crf_kfCV_record.csv"), index = False)
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
plt.figure(figsize = (2.25,3))
sns.displot(
                crf_acc_distrib, 
                kde = False, 
                bins = 12
            )
# plt.savefig("lgr_acc_distrib.png", bbox_inches="tight")

#%%
# class distribution 
class_names = np.unique(np.sort(y))
crf_per_class_acc_distrib = pd.DataFrame(kf_per_class_results, columns = class_names)
crf_per_class_acc_distrib.dropna().to_csv(generate_path("crf_per_class_acc_distrib.csv"))
crf_per_class_acc_distrib = pd.read_csv(generate_path("crf_per_class_acc_distrib.csv", index_col = 0))
crf_per_class_acc_distrib = np.round(crf_per_class_acc_distrib, 2)
crf_per_class_acc_distrib_describe = crf_per_class_acc_distrib.describe()
crf_per_class_acc_distrib_describe.to_csv(generate_path("crf_per_class_acc_distrib.csv"))


#%%
# plotting class distribution
lgr_per_class_acc_distrib = pd.melt(
                                        crf_per_class_acc_distrib, 
                                        var_name = "status new"
                                    )

# Plotting
plt.figure(figsize=(6,4))

sns.violinplot(
                x = "status new", 
                y = "value", 
                cut = 0, 
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
                bbox_inches = "tight"
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

# # Add mean accuracy of best models to plots
# plt.annotate("Average Accuarcy:\n{0:.3f} ± {1:.3f}".format(
#                                                             crf_acc_distrib.mean()[0], 
#                                                             crf_acc_distrib.sem()[0]
#                                                         ), 
#                                                         xy = (0.06, 0), 
#                                                         color = "k"
#                                                     )


plt.savefig(
                generate_path("_feature_impces_full_wn.png"), 
                dpi = 500, 
                bbox_inches = "tight"
            )

#%%
# Another way to plot  feature importances

average_spectra = pd.DataFrame(
                                training_df.iloc[:,1:].mean().T
                            ).reset_index()

average_spectra.rename(
                        columns = {'index':'wavenumber', 0:'absorbance'}, 
                        inplace = True
                    )


# Average_spectra is your DataFrame with columns "wavenumber" and "absorbance"
wavenumbers = pd.to_numeric(average_spectra['wavenumber'])
absorbances = average_spectra['absorbance']

# We plot the result of the analysis and the selected wavenumbers
fig_wid = 25
fig_hei = 8

zord = 100
# Create a figure and axis
# fig = plt.figure(facecolor = 'w')
fig, ax = plt.subplots(
                        figsize = (fig_wid, fig_hei), 
                        facecolor = '#dddddd'
                    )

# Plot the spectra
for i in range(len(wavenumbers)):
    ax.plot(
                wavenumbers.sort_values(ascending = False), 
                absorbances, 
                'r',
                alpha = 0.8,
                label=f'Spectrum {i}', 
                zorder = zord
            )

    zord -= 1

# Set labels and title
ax.set_xlabel('Wavenumber')
ax.set_ylabel('Absorbance')
plt.xlim(4000, 500)

new_wavenumbers = pd.to_numeric(all_featimp["mean"][-50:].index)
feat_imp_mean = all_featimp["mean"][-50:].values

# Bar plot
# Normalize and scale feat_imp_mean
normalized_feat_imp_mean = feat_imp_mean / feat_imp_mean.max() * absorbances.max()
feat_imp_mean_scl = normalized_feat_imp_mean * 10

# Create a secondary y-axis
ax2 = ax.twinx()
# ax2.bar(new_wavenumbers, feat_imp_mean_scl, color='blue', alpha=0.7, width=10, align='center', label='Feature Importance')
# ax2.set_ylabel('Feature Importance (Scaled)')
# Bar plot on the secondary y-axis with fading color

# Bar plot on the secondary y-axis with single color (color blind friendly)
ax2.bar(
            new_wavenumbers, 
            feat_imp_mean_scl, 
            color = 'dodgerblue', 
            alpha = 0.8, 
            width = 10, 
            align = 'center', 
            label = 'Feature Importance'
        )

ax2.set_ylabel('Feature Importance (Scaled)')

# # Then the scores
# for n, i in enumerate(new_wavenumbers):
#     if feat_imp_mean[n] > 0.001:
#         try:

#             ppp = wavenumbers.index[i]
#             sc_y = np.mean(np.transpose(absorbances)[ppp])
#             if n < 25:
#                 tmp = Ellipse((i, sc_y), feat_imp_mean[n](wavenumbers[-1] - wavenumbers[0])*fig_hei/fig_wid, feat_imp_mean[n](0.30-0), color='paleturquoise', ec = 'black', zorder = zord)
#             else:
#                 tmp = Ellipse((i, sc_y), feat_imp_mean[n](wavenumbers[-1] - wavenumbers[0])*fig_hei/fig_wid, feat_imp_mean[n](0.30-0), color='paleturquoise', ec = 'black', zorder = zord)
#             ax.add_patch(tmp)
#             zord -= 1
#         except ValueError:
#             continue



# # Interpolate feat_imp_mean onto the existing wavenumbers range
# interp_feat_imp = np.interp(wavenumbers, new_wavenumbers, feat_imp_mean)

# normalized_feat_imp_mean = feat_imp_mean / feat_imp_mean.max() * absorbances.max()

# # for wavenumber, mean_value, norm_mean in zip(new_wavenumbers, interp_feat_imp, normalized_feat_imp_mean):
# #     ax.scatter(wavenumber, mean_value, s=norm_mean * 500, c='red', alpha=0.5, label='Mean Values')


# # Scatter plot along the line
# for wavenumber, mean_value, norm_mean in zip(new_wavenumbers, interp_feat_imp, normalized_feat_imp_mean):
#     ax.scatter(wavenumber, mean_value, s=norm_mean * 1000, c='red', alpha=0.8, label='Mean Values')

plt.savefig(
                generate_path("_feature_impces.png"), 
                dpi = 500, 
                bbox_inches = "tight"
            )


#%%
# Predict validation data

# Transform data using the mean and standard deviation from the model training data

# X_test = val_data.iloc[:,:-1]
# y_test = val_data['infection_status']

# scale data
trans_X_val = scaler.transform(X = np.asarray(X_test))

# with open("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML_final_analysis\Results\PCR model_1\classifier.pkl", 'rb') as fid:
#      classifier_loaded = pickle.load(fid)

# Predict the sporozoite infection in mosquitoes [PCR as true label]  
y_val_pred = best_classifier.predict(trans_X_val)

accuracy_test = accuracy_score(np.asarray(y_test), y_val_pred)
print("Accuracy: %.2f%%" % (accuracy_test * 100.0))

#%%
# Plotting confusion matrix for X_test 

plt.rcParams["figure.figsize"] = [6,4]

classes = np.unique(np.sort(y))
figure_name = 'unseen_prediction'
visualize(
            figure_name, 
            classes, 
            np.asarray(y_test), 
            y_val_pred
        )

#%%

# Summarising precision, f_score, and recall for the test set
cr_full_wn_val = classification_report(np.asarray(y_test), y_val_pred)
print('Classification report : {}'.format(cr_full_wn_val))

# save classification report to disk as a csv

cr_full_wn_val = pd.read_fwf(io.StringIO(cr_full_wn_val), header=0)
cr_full_wn_val = cr_full_wn_val.iloc[0:]
cr_full_wn_val.to_csv(generate_path("classification_report_full_wn_val_pcr.csv"))

#%%

# Collect all the important wavenumbers

important_wavenumb = pd.DataFrame(all_featimp["mean"][-100:])
important_wavenumb = important_wavenumb.reset_index()
important_wavenumb = important_wavenumb['index'].to_list()

with open(generate_path('important_wavenumbers.txt'), 'w') as outfile:
     json.dump(important_wavenumb, outfile)


#%%

"""

# Predicting the sporozoite with age data included in test set

"""

# load age data
df_age = pd.read_csv(os.path.join("..", "Data", "wild_funestus_age.dat"), delimiter = '\t')

df_age = df_age.query("Cat3 == '14D'")
df_age = df_age.drop(
                        [
                            'Cat1', 
                            'Cat2', 
                            'Cat3', 
                            'Cat4', 
                            'StoTime'
                        ], 
                        axis = 1
                    )

df_age['Sporozoite'] = 'Negative'

# shift column 'Name' to first position
first_column = df_age.pop('Sporozoite')
  
# insert column using insert(position,column_name,
# first_column) function
df_age.insert(0, 'Sporozoite', first_column)

df_age.head()

#%%

# combine the unseen matrix of features and list of labels in a single dataframe
new_val_df = pd.concat(
                        [
                            y_test, 
                            X_test
                        ], 
                        axis = 1
                    )

# shift column 'Name' to first position
first_column_ = new_val_df.pop('infection_status')
  
# insert column using insert(position,column_name,
# first_column) function
new_val_df.insert(0, 'Sporozoite', first_column_)
new_val_df

#%%
# concatinate age data and unseen PCR data

new_data_age_df = pd.concat(
                                [
                                    new_val_df, 
                                    df_age
                                ], 
                                axis = 0, 
                                join = 'outer'
                            )

print(new_data_age_df)
print(collections.Counter(new_data_age_df['Sporozoite']))


#%%

X_val_2 = new_data_age_df.iloc[:,1:] # matrix of features
y_val_2 = new_data_age_df["Sporozoite"] # vector of labels
print(collections.Counter(y_val_2))

X_res_val, y_res_val = rus.fit_resample(X_val_2, y_val_2)
y_res_count = collections.Counter(y_res_val)
print(y_res_count)

X_res_val_trans = scaler.transform(np.asarray(X_res_val))
y_res_val = np.asarray(y_res_val)

#%%

# loading the classifier from the disk
# with open("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\ML_final_analysis\Results\PCR model\classifier.pkl", 'rb') as fid:
#      classifier_loaded = pickle.load(fid)

# generates output predictions based on the X_input passed

predictions = best_classifier.predict(X_res_val_trans)

# Examine the accuracy of the model in predicting new samples

accuracy = accuracy_score(y_res_val, predictions)
print("Accuracy:%.2f%%" %(accuracy * 100.0))

#%%

# Plotting confusion matrix
plt.rcParams["figure.figsize"] = [6,4]

figure_name = 'age_sporozoite_PCR'
visualize(
            figure_name, 
            classes, 
            y_res_val, 
            predictions
        )

#%%
# Summarising precision, f_score, and recall for the validation set
cr_full_wn_val_age = classification_report(y_res_val, predictions)
print('Classification report : {}'.format(cr_full_wn_val_age))

# save classification report to disk as a csv

cr_full_wn_val_age = pd.read_fwf(io.StringIO(cr_full_wn_val_age), header=0)
cr_full_wn_val_age = cr_full_wn_val_age.iloc[0:]
cr_full_wn_val_age.to_csv(generate_path("classification_report_full_wn_val_age.csv"))


# %%

"""
    Use PCR model to predict elisa data
"""
# load ELISA infection data

elisa_df = pd.read_csv(os.path.join("..", "Data", "sporozoite_full.csv"))
elisa_df.head()

#%%

# data shape
print(elisa_df.shape)

# Checking class distribution abd correlation in the data
Counter(elisa_df["Sporozoite"])

#%%
# Select vector of labels and matrix of features

X_val_elisa = elisa_df.iloc[:,7:] # matrix of features
y_val_elisa = elisa_df["Sporozoite"] # vector of labels
X_val_elisa

#%%

# rescalling the data (undersampling the over respresented class - negative class)

X_res_elisa, y_res_elisa = rus.fit_resample(X_val_elisa, y_val_elisa)
print(collections.Counter(y_res_elisa))

X_res_val_trans_elisa = scaler.transform(np.asarray(X_res_elisa))
y_res_val_elisa = np.asarray(y_res_elisa)


#%%

# generates output predictions based on the X_input passed from PCR data

predictions_elisa = best_classifier.predict(X_res_val_trans_elisa)

# Examine the accuracy of the model 

accuracy_elisa = accuracy_score(y_res_val_elisa, predictions_elisa)
print("Accuracy:%.2f%%" %(accuracy_elisa * 100.0))

#%%

# Plotting confusion matrix
plt.rcParams["figure.figsize"] = [6,4]

figure_name = 'elisa_sporozoite'
visualize(
            figure_name, 
            classes, 
            y_res_val_elisa, 
            predictions_elisa
        )

#%%

# Summarising precision, f_score, and recall for the validation set
cr_full_wn_val_elisa = classification_report(y_res_val_elisa, predictions_elisa)
print('Classification report : {}'.format(cr_full_wn_val_elisa))

# save classification report to disk as a csv

cr_full_wn_val_elisa = pd.read_fwf(io.StringIO(cr_full_wn_val_elisa), header = 0)
cr_full_wn_val_elisa = cr_full_wn_val_elisa.iloc[0:]
cr_full_wn_val_elisa.to_csv(generate_path("classification_report_full_wn_val_elisa.csv"))

