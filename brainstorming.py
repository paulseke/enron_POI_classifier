# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 12:35:13 2016

@author: seke
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import pickle
sys.path.append("../tools/")
import copy
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pickle
import numpy as np
import pandas as pd
from ggplot import *



def dataframeTest_i(df, i):
    try:
        df[i]
        return True
    except KeyError:
        return False

def nan_or_not(all_df,column):
    column = all_df[column]
    return column!='NaN'

def nan_remover(all_df,column1,column2):
    no_nan_1 = nan_or_not(all_df,column1)
    no_nan_2 = nan_or_not(all_df,column2)
    all_df = all_df[no_nan_1]
    all_df = all_df[no_nan_2]
    return all_df

    
def dataReg(df):
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(df.iloc[1:(len(df)-1), 0].to_frame(), df.iloc[1:(len(df)-1), 1].to_frame())
    return reg
    
def outlierCleaner(df,column1,column2):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the value).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import pandas as pd
    import numpy as np
    import math
    df1 = nan_remover(df,column1,column2)
    df = pd.DataFrame(df1[[column1,column2,'poi','index']])
    df.columns = [0,1,2,3]
    ages_train = np.array(df[0])
    net_worths_train = np.array(df[1])
    reg = dataReg(df)
    errors = []
#    max_index = []
    for i in range(len(net_worths_train)):
        pred = reg.predict(ages_train[[i]])
        error = abs(net_worths_train[i] - pred[0][0])
#        error = math.pow(net_worths_train[i][0] - pred[0][0], 2)
        #error = net_worths_train[i][0] - pred[0][0]
        errors.append(error)
    # 10 % highest errors
    errors = pd.Series(errors)
    max_errors = errors.nlargest(int(math.floor(0.1*len(ages_train))))
    #max_errors = errors.nsmallest(int(math.floor(0.1*len(ages_train))))
#    max_index = max_errors.index.tolist()
        
    cleaned_data = []
    for i in range(len(ages_train)):
        if i in max_errors:
            continue
        else:
            datum = (ages_train[i], net_worths_train[i], df[2][i], df[3][i], errors[i])
            cleaned_data.append(datum)
    cleaned_data = pd.DataFrame(cleaned_data, columns = [column1,column2, 'poi','index','SE'])        
    #return errors
    return cleaned_data


def df_plotter(all_df,column1, column2, title_strg):
    p = ggplot(aes(x=column1, y=column2), data=all_df) +\
        geom_point(aes(color = 'poi')) +\
        ggtitle(title_strg) +\
        xlab(column1) +\
        ylab(column2) #+\
        #ylim(-10000, 600000) 
    return p



def test_feature(all_df,column_name):
    try:
        np.array(all_df[column_name])
        return True
    except TypeError:
        return False

        
def poi_counter(enron_data, names):
    counter = 0
    indexPoi = []
    namesPoi = []
    for i in range(len(names)):
        if enron_data[names[i]]["poi"] == 1.0:
            indexPoi.append(i)
            namesPoi.append(names[i])
            counter += 1
    return namesPoi, indexPoi, counter


def poi_counter2(column, namesPoi):
    names = list(column.index.values)
    counter = 0
    #poi = []
    for name in names:
        if name in namesPoi:
            #poi.append(i)
            counter += 1
    return counter

    
# number of missing values per feature
def inspect(all_df):
    features_list = list(all_df.columns.values)
    for i in range(len(features_list)):
        if test_feature(all_df, features_list[i]):
            column = copy.deepcopy(all_df[features_list[i]])
            no_nan = column!='NaN'
            column = column[no_nan]
            missing = len(all_df) - len(column)
            print " "
            print "missing data in {i}".format(**vars()), features_list[i],":", missing
            print "available data in {i}".format(**vars()), features_list[i],":", len(column)
            print "available POI in {i}".format(**vars()), features_list[i],":", poi_counter2(column, namesPoi)
        else:
            print "problem with column{i}:".format(**vars()), features_list[i]
            print " "
    return no_nan


    
def ratio_maker(i,df_to_keep,new_df,column1,column2):
    if dataframeTest_i(new_df['index'], i):
        # from cartesian coordinates to distance from the origin
        return new_df[column1][i]/new_df[column2][i]
    else:
         return 'NaN'
    
def ratio_maker2(i,df_to_keep,new_df,column1,column2):
    import numpy as np
    if dataframeTest_i(new_df['index'], i):
        # from cartesian coordinates to distance from the origin
        return np.sqrt((new_df[column1][i])**2+(new_df[column2][i])**2)
    else:
         return 'NaN'

def new_variableAdder(df_to_keep,new_df,column1,column2):
    new_variable = []
    for i in range(len(df_to_keep)):
        value = ratio_maker(i,df_to_keep,new_df,column1,column2)
        new_variable.append(value)
    df_to_keep[variable_name] = new_variable
    return df_to_keep



def remove_dict_key(dictionary, key):
    ''' removes a key without mutating the dictionary  '''  
    new_element = copy.deepcopy(dictionary)
    del new_element[key]
    return new_element

    
 
def get_index(indx):
    return features_list.index(indx)   
    
    

def get_bestParameters(clf, labels, features):
    from sklearn.cross_validation import StratifiedShuffleSplit
    from sklearn.grid_search import GridSearchCV
    
    param_grid = {"criterion" : ["gini", "entropy"],
                  "splitter" :   ["best", "random"]
                 }
    
    # store the split instance into cv and use to be used in GridSearchCV.
    cv = StratifiedShuffleSplit(labels, random_state=42)
    #features = MinMaxScaler().fit_transform(features)
    grid = GridSearchCV(clf, param_grid, cv = cv, scoring='f1')
    grid.fit(features, labels)
    return grid.best_params_    


def pcaPre_dTree(features_train, features_test, labels_train, labels_test,labels, features):
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.decomposition import PCA
    from sklearn.pipeline import Pipeline
#    from sklearn.metrics import accuracy_score 
#    from sklearn.metrics import recall_score
#    from sklearn.metrics import precision_score
#    from sklearn.preprocessing import MinMaxScaler
    
#    standard_train = MinMaxScaler().fit_transform(features_train)    
    pca = PCA(n_components = 'mle')
#    pca = PCA(n_components = 3)
    #'mle'
    DTC = DecisionTreeClassifier()
 
    bestParameters = get_bestParameters(DTC, labels, features)

    DTC = DecisionTreeClassifier(criterion = bestParameters['criterion'], 
                                 splitter = bestParameters['splitter'])
        
    estimators = [('reduce_dim', pca), ('dtc', DTC)]
    clf = Pipeline(estimators)
    clf.fit(features_train, labels_train) 
#    standard_test = MinMaxScaler().fit_transform(features_test)
#    pred = clf.predict(standard_test)
#    print "accuracy pca_dTree pipeline:", round(accuracy_score(pred, labels_test), 3)
#    print "recall:", recall_score(labels_test, pred, average='binary')
#    print "precision:", precision_score(labels_test, pred, average='binary')
    return clf
 
def min_max_scaling(features_train,features_test):
    from sklearn.preprocessing import MinMaxScaler as scaler    
    features_train = scaler().fit_transform(features_train)  
    features_test = scaler().fit_transform(features_test)
    return features_train, features_test

    
def kbest_dTree(labels, features,features_train, features_test, labels_train, labels_test):
    '''
    modified from scikit-learn.org/stable/auto_examples/feature_selection/
    feature_selection_pipeline.html#example-feature-selection-feature-selection-pipeline-py
    '''
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
         
    filteredBest = SelectKBest(f_classif, k=3)
    DTC = DecisionTreeClassifier()
    bestParameters = get_bestParameters(DTC, labels, features)
    DTC = DecisionTreeClassifier(criterion = bestParameters['criterion'], 
                                 splitter = bestParameters['splitter'])    
     
    estimators = [('best_features', filteredBest), ('dtc', DTC)]
    clf = Pipeline(estimators)
    clf.fit(features_train, labels_train) 
    
    score = filteredBest.scores_
    
    return bestParameters, score, clf


def kbest_dTree2(k_value,bestParameters,features_train, features_test,labels_train,labels_test):
    '''
    modified from scikit-learn.org/stable/auto_examples/feature_selection/
    feature_selection_pipeline.html#example-feature-selection-feature-selection-pipeline-py
    '''
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.pipeline import Pipeline
    
#    features_train = scaler().fit_transform(features_train)           
    filteredBest = SelectKBest(f_classif, k=k_value)
    DTC = DecisionTreeClassifier()
    DTC = DecisionTreeClassifier(criterion = bestParameters['criterion'], 
                                 splitter = bestParameters['splitter'])    
    
    estimators = [('best_features', filteredBest), ('dtc', DTC)]
    clf = Pipeline(estimators)
    clf.fit(features_train, labels_train) 
#    features_test = scaler().fit_transform(features_test)
    score = filteredBest.scores_

    return score, clf

 
def to_keep_list(lst):
    to_keep = [get_index(lst,'salary'),get_index(lst,'bonus'),get_index(lst,'poi'),get_index(lst,'total_payments'),
     get_index(lst,'restricted_stock'),get_index(lst,'from_poi_to_this_person'),get_index(lst,'total_stock_value'),get_index(lst,'exercised_stock_options'),get_index(lst,'long_term_incentive')]
    to_keep.sort()
    return to_keep


def new_dataDict(my_dataset):
    #del features_list
    dictionary = copy.deepcopy(my_dataset)
    names = dictionary.keys()
    features_list = dictionary[names[1]].keys()
    to_keep = to_keep_list(features_list)    
    for name in names:
        for i in range(len(dictionary[name])):
            if i not in to_keep:
                dictionary[name] = remove_dict_key(dictionary[name], features_list[i])
            else:
                continue
    return to_keep,dictionary


def new_dataDict2(my_dataset):
    #del features_list
    dictionary = copy.deepcopy(my_dataset)
    for name in names:
        data = my_dataset[name]
        dictionary[name] = {'salary':data['salary'],
                            'bonus':data['bonus'],
                            'poi':data['poi'],
                            'total_payments':data['total_payments'],
                            'restricted_stock':data['restricted_stock'],
                            'from_poi_to_this_person':data['from_poi_to_this_person'],
                            'total_stock_value':data['total_stock_value'],
                            'exercised_stock_options':data['exercised_stock_options'],
                            'long_term_incentive':data['long_term_incentive']
                            }

    return dictionary

    




### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# deep copy of data to keep original data untouched
enron_data = copy.deepcopy(data_dict) 
    


### Task 1: feature selection.
# number of persons in the dataset:
print "number of persons in the dataset:",len(data_dict)
# number of features per person
print "number of features per person:",len(data_dict['YEAP SOON'])

# getting names of people 
names = list(enron_data.keys())

# number of POIs in the dataset
namesPoi, indexPoi, poiNumber = poi_counter(enron_data, names)
nonPoiNumber = len(names) - poiNumber  
print "number of poi:",poiNumber


# getting all names of features
features_list = enron_data['YEAP SOON'].keys()


# building a pandas dataframe with all the data
all_df  = pd.DataFrame.from_dict(enron_data, orient='index')


# building dataframe for feature inspection
df_to_inspect = copy.deepcopy(all_df)
# removing features_list[16], boolean of poi status
# from df to inspect
del df_to_inspect[features_list[16]]
# inspection of all features
inspect(df_to_inspect)    
del df_to_inspect       

# removing features
# loan_advances: 142 missing, 1 POI and 3 non-POI remaining
del all_df[features_list[4]]
# 107 missing in deferral_payments, but kept because 5 POI in 39 remaining
# email_addresses (array of strings)
del all_df[features_list[6]]
# restricted_stock_deferred: 128 missing, 0 POI remaining
del all_df[features_list[7]]
# director_fees: 129 missing, 0 POI remaining
del all_df[features_list[20]]
# 97 missing in deferred_income, but kept because 11 POI in the 49 remaining

# adding an 'index' column
all_df['index']=range(len(all_df))

# updating feature list
del features_list
features_list = list(all_df.columns.values)
print " "
print "number of features remaining:", len(features_list)



# plotting features to have a feel of the data

# salary
# NOT GOOD DISCRIMINATORY POWER
i = 0
title_strg = features_list[i]
p = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)
# identifying the outlier(s)
column = copy.deepcopy(all_df[features_list[i]])
no_nan = all_df[column!='NaN']
column = no_nan[[features_list[i]]]
outlier = (column[column > 5000000]).dropna()
print " "
print "the outlier is:",outlier.index.tolist()
#Out[]:      ['TOTAL']

# removing the outlier
all_df = all_df.drop(outlier.index.tolist())
p0 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# to_messages
# NO GOOD DISCRIMINATORY POWER
i = 1
title_strg = features_list[i]
p1 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# deferral_payments
# MAY HAVE GOOD DISCRIMINATORY POWER
i = 2
title_strg = features_list[i]
p2 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# total_payments
# MAY HAVE A GOOD DISCRIMINATORY POWER
# a clear outlier (POI) not removed
i = 3
title_strg = features_list[i]
p3 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# bonus
# MAY HAVE A GOOD DISCRIMINATORY POWER
i = 4
title_strg = features_list[i]
p4 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# total_stock_value
# MAY HAVE INTERESTING DISCRIMINATORY POWER
i = 5
title_strg = features_list[i]
p5 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# shared_receipt_with_poi
# MAY HAVE INTERESTING DISCRIMINATORY POWER
i = 6
title_strg = features_list[i]
p6 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# long_term_incentive
# MAY HAVE INTERESTING DISCRIMINATORY POWER (NOT VERY GOOD)
i = 7
title_strg = features_list[i]
p7 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# exercised_stock_options
# MAY HAVE INTERESTING DISCRIMINATORY POWER
i = 8
title_strg = features_list[i]
p8 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# from_messages
# NO GOOD DISCRIMINATORY POWER
# To test in combination with from_poi
i = 9
title_strg = features_list[i]
p9 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# other
# NO GOOD DISCRIMINATORY POWER
i = 10
title_strg = features_list[i]
p10 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# from_poi_to_this_person
# MAY HAVE INTERESTING DISCRIMINATORY POWER
i = 11
title_strg = features_list[i]
p11 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# from_this_person_to_poi
#  MAY HAVE INTERESTING DISCRIMINATORY POWER
i = 12
title_strg = features_list[i]
p12 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# poi
#i = 13


# deferred_income
# MAY HAVE INTERESTING DISCRIMINATORY POWER (NOT SEEMS GOOD)
i = 14
title_strg = features_list[i]
p14 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# expenses
# MAY HAVE INTERESTING DISCRIMINATORY POWER
i = 15
title_strg = features_list[i]
p15 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)


# restricted_stock
# NO GOOD DISCRIMINATORY POWER
i = 16
title_strg = features_list[i]
p16 = df_plotter(all_df,features_list[len(features_list)-1]
                                      , features_list[i], title_strg)



    
# Inserting the selected features in a new dataframe
#df_to_keep = pd.DataFrame(all_df[[features_list[2],features_list[3],features_list[4],features_list[5],features_list[6],features_list[7],
#                                  features_list[8],features_list[11],features_list[12],features_list[14],features_list[15],'poi','index']])

df_to_keep = copy.deepcopy(all_df)
del all_df
    
# creating and evaluating new features

# ratio_from_poi_to_all_from_emails
# SEEMS TO HAVE A GOOD DISCRIMINATORY POWER
new_df = outlierCleaner(df_to_keep,features_list[9],features_list[11])
# removing other outliers
#new_df = new_df[new_df[features_list[9]]<500]

variable_name = 'ratio_from_poi_to_all_from_emails'
column1 = features_list[9]
column2 = features_list[11]
p17 = df_plotter(new_df,features_list[9], features_list[11], title_strg)

# Inserting the new feature to df_to_keep 
df_to_keep = new_variableAdder(df_to_keep,new_df,column1,column2)

# more plots and information 
title_strg = "from_poi_to_this_person in function of from_messages"
print " "
print "number of persons",variable_name,":",len(new_df)
print "number of POI:",poi_counter2(df_to_keep[variable_name], namesPoi)
#p17 = df_plotter(df_to_keep,'index', 'ratio_from_poi_to_all_from_emails', title_strg)



# ratio from_this_person_to_poi to to_messages
# SEEMS TO HAVE AN ACCEPTABLE DISCRIMINATORY POWER
new_df = outlierCleaner(df_to_keep,features_list[1],features_list[12])
# removing other outliers
#new_df = new_df[new_df[features_list[1]]<4000] 

variable_name = 'ratio_to_poi_to_all_sent_emails'
column1 = features_list[1]
column2 = features_list[12]
title_strg = "ratio_to_poi_to_all_sent_emails"
p18 = df_plotter(new_df,features_list[1], features_list[12], title_strg)

# Inserting the new feature to df_to_keep
df_to_keep = new_variableAdder(df_to_keep,new_df,column1,column2)
print " "
print "number of persons in",variable_name,":",len(new_df)
print "number of POI:",poi_counter2(df_to_keep[variable_name], namesPoi)
#p18 = df_plotter(df_to_keep,'index', variable_name, title_strg)



# updating feature list
del features_list
names = list(df_to_keep.index.values)
features_list = list(df_to_keep.columns.values)
# removing 'poi'
del features_list[features_list.index('poi')]
# adding 'poi' to first position in the list
features_list.insert(0, 'poi')

# defining my_dataset
my_dataset = df_to_keep.to_dict(orient='index')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)



# train/test spliting
# test size for train/test splitting set to 0.3 
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                            test_size=0.3, random_state=42)


# data preprocessing: scaling
features_train,features_test = min_max_scaling(features_train,features_test)

# decision tree classifier tuning using a k best selection
bestParameters,score,clf = kbest_dTree(labels, features,features_train, features_test, labels_train, labels_test)


## getting the scores and feature labels
#score_labels = []
#label_list = copy.deepcopy(features_list)
#del label_list[0]
#for i in range(len(score)):
#    score_labels.append([score[i], label_list[i], i])

#  Accuracy: 0.80860       Precision: 0.27632      Recall: 0.26900 F1: 0.27261     F2: 0.27043





# running again with only features with high k

# dropping the unused values

    
my_dataset = new_dataDict2(my_dataset)
names = dictionary.keys()
features_list = my_dataset[names[1]].keys()
# removing 'poi'
del features_list[features_list.index('poi')]
# adding 'poi' to first position in the list
features_list.insert(0, 'poi')


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# train/test spliting
# test size for train/test splitting set to 0.3 
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                            test_size=0.3, random_state=42)

# data scaling
features_train, features_test = min_max_scaling(features_train,features_test)
# a = pd.DataFrame(features_train)
# b = pd.DataFrame(features_test)
# features_train[0][1]



# decision tree classifier tuning using a k best selection
score,clf = kbest_dTree2(3,bestParameters,features_train, features_test,labels_train,labels_test)
#clf = pcaPre_dTree(features_train, features_test, labels_train, labels_test,labels, features)


# getting the scores and feature labels
score_labels = []
label_list = copy.deepcopy(features_list)
del label_list[0]
for i in range(len(score)):
    score_labels.append([score[i], label_list[i], i])

# no normalization
# Accuracy: 0.82327	Precision: 0.32750	Recall: 0.30900	F1: 0.31798	F2: 0.31253
# with normalization
# Accuracy: 0.81727	Precisi50on: 0.32646	Recall: 0.34850	F1: 0.33712	F2: 0.34386
# Accuracy: 0.81200       Precision: 0.30642      Recall: 0.32450 F1: 0.31520     F2: 0.32072
# Accuracy: 0.81327       Precision: 0.31654      Recall: 0.34550 F1: 0.33038     F2: 0.33929
#Accuracy: 0.81447       Precision: 0.32499      Recall: 0.36350 F1: 0.34317     F2: 0.35508



PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

def test_classifier(clf, data, labels, feature_list, folds = 1000):
    from sklearn.cross_validation import StratifiedShuffleSplit
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."


print "######### ALL DONE SUCCESSFULLY ##########"

test_classifier(clf, data, labels, features_list, folds = 1000)