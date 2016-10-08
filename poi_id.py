#!/usr/bin/python
"""
Classifier for detecting POI from ENRON dataset
using some financial informations and emails
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import pickle
sys.path.append("../tools/")

import copy
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np
import pandas as pd
from ggplot import *




### Some helpful functions
######################################################################################


def dataframeTest_i(df, i):
    try:
        df[i]
        return True
    except KeyError:
        return False


def listIndexTest(the_list, datum_index_to_find):
    try:
        the_list.index(datum_index_to_find)
        return True
    except ValueError:
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
    for i in range(len(net_worths_train)):
        pred = reg.predict(ages_train[[i]])
        error = abs(net_worths_train[i] - pred[0][0])
        errors.append(error)
    
    # removing data with 10 % highest errors
    errors = pd.Series(errors)
    max_errors = errors.nlargest(int(math.floor(0.1*len(ages_train))))
        
    cleaned_data = []
    for i in range(len(ages_train)):
        if i in max_errors:
            continue
        else:
            datum = (ages_train[i], net_worths_train[i], df[2][i], df[3][i], errors[i])
            cleaned_data.append(datum)
    cleaned_data = pd.DataFrame(cleaned_data, columns = [column1,column2, 'poi','index','SE'])        

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
    for name in names:
        if name in namesPoi:
            counter += 1
    return counter

    

def inspect(all_df):
    '''number of missing values per feature'''  

    features_list = list(all_df.columns.values)
    # removing boolean poi from df to inspect
    if listIndexTest(features_list, 'poi'):
        del features_list[features_list.index('poi')]
    # removing boolean 'email_addresses' from df to inspect
    if listIndexTest(features_list, 'email_addresses'):
        del features_list[features_list.index('email_addresses')]
        
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
#    return no_nan



    
def ratio_maker(i,df_to_keep,new_df,column1,column2):
    if dataframeTest_i(new_df['index'], i):
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

         

def new_variableAdder(variable_name,df_to_keep,new_df,column1,column2):
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
    
    
def get_bestParameters(labels, features):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.cross_validation import StratifiedShuffleSplit
    from sklearn.grid_search import GridSearchCV
    
    param_grid = {"criterion" : ["gini", "entropy"],
                  "splitter" :   ["best", "random"]
                 }
    DTC = DecisionTreeClassifier()
    folds = 1000
    cv = StratifiedShuffleSplit(labels,folds,random_state=42)
    grid = GridSearchCV(DTC, param_grid, cv = cv, scoring='f1')
    grid.fit(features, labels)
    return grid.best_params_    

    
 
def min_max_scaling(features_train,features_test):
    from sklearn.preprocessing import MinMaxScaler as scaler    
    features_train = scaler().fit_transform(features_train)  
    features_test = scaler().fit_transform(features_test)
    return features_train, features_test


def get_featuresScore(k_value,labels, features):
    from sklearn.feature_selection import SelectKBest, f_classif
    return SelectKBest(f_classif, k=k_value).fit_transform(features,labels)


def kbest_dTree(features_train, features_test, labels_train, labels_test):
    from sklearn.tree import DecisionTreeClassifier

    DTC = DecisionTreeClassifier()
    bestParameters = get_bestParameters(labels, features)
    DTC = DecisionTreeClassifier(criterion = bestParameters['criterion'], 
                                 splitter = bestParameters['splitter'])    
     
    DTC.fit(features_train, labels_train)       
    return bestParameters, DTC


 
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


def print_info_new_feature(variable_name,new_df,df_to_keep,namesPoi):
    print " "
    print "number of persons",variable_name,":",len(new_df)
    print "number of POI:",poi_counter2(df_to_keep[variable_name], namesPoi)    


def poi_to_first(feature_list):
    '''add 'poi' to first position in the feature list'''
    # removing 'poi'
    del feature_list[feature_list.index('poi')]
    # adding 'poi' to first position 
    feature_list.insert(0, 'poi')
    return feature_list



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



######################################################################################
    

###### Building a POI detector


### Task 1: feature selection

## Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


## dataset munging

# deleting awkward keys and entry with no data
data_dict = remove_dict_key(data_dict,'THE TRAVEL AGENCY IN THE PARK')
data_dict = remove_dict_key(data_dict,'LOCKHART EUGENE E')


# getting names of people 
names = list(data_dict.keys())
# getting feature list
features_list = data_dict[names[0]].keys()


## dataset information
# number of persons in the dataset:
print "number of persons in the dataset:",len(data_dict)
# number of features per person
print "number of features per person:",len(data_dict['YEAP SOON'])
# number of POIs in the dataset
namesPoi, indexPoi, poiNumber = poi_counter(data_dict, names)
nonPoiNumber = len(names) - poiNumber  
print "number of poi:",poiNumber



# feature inspection
# building a pandas dataframe with all the data
all_df  = pd.DataFrame.from_dict(data_dict, orient='index')
# inspection of all features
inspect(all_df) 

## removing unusable features from the dataframe
# loan_advances: 142 missing, 1 POI and 3 non-POI remaining
del all_df[features_list[features_list.index('loan_advances')]]
# 107 missing in deferral_payments, but kept because 5 POI in 39 remaining
# email_addresses (array of strings)
del all_df[features_list[features_list.index('email_address')]]
# restricted_stock_deferred: 128 missing, 0 POI remaining
del all_df[features_list[features_list.index('restricted_stock_deferred')]]
# director_fees: 129 missing, 0 POI remaining
del all_df[features_list[features_list.index('director_fees')]]
# 97 missing in deferred_income, but kept because 11 POI in the 49 remaining

# adding an 'index' column to track changes in the df
all_df['index']=range(len(all_df))



## updating feature list
del features_list
features_list = list(all_df.columns.values)
print " "
print "number of features remaining after wrangling:", len(features_list)




### Task 2: outlier removal
# plotting features to have a feel of the data

# salary
i = features_list.index('salary')
title_strg = features_list[i]
p0 = df_plotter(all_df,features_list[len(features_list)-1]
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
#p0 = df_plotter(all_df,features_list[len(features_list)-1]
#                                      , features_list[i], title_strg)

## NO OTHER KEY REMOVED FROM THE ENTIRE DATASET 
# (all plots and first sight data feeling in brainstorming.py)





### Task 3: Creation of new features
df_to_keep = copy.deepcopy(all_df)


## (1) ratio_from_poi_to_all_from_emails
ii = features_list.index('from_poi_to_this_person')
ij = features_list.index('from_messages')
new_df = outlierCleaner(df_to_keep,features_list[ii],features_list[ij])
# removing other outliers
new_df = new_df[new_df[features_list[ii]]<500]

#title_strg = "from_poi_to_this_person in function of from_messages"
#p17 = df_plotter(new_df,features_list[ii], features_list[ij], title_strg)

# Inserting the new feature to df_to_keep 
df_to_keep = new_variableAdder('ratio_from_poi_to_all_from_emails',
                               df_to_keep,new_df,features_list[ii],features_list[ij])
# new feature information 
print_info_new_feature('ratio_from_poi_to_all_from_emails',new_df,df_to_keep,namesPoi)



## (2) ratio from_this_person_to_poi to to_messages
ii = features_list.index('from_this_person_to_poi')
ij = features_list.index('to_messages')
new_df = outlierCleaner(df_to_keep,features_list[ii],features_list[ij])
# removing other outliers
new_df = new_df[new_df[features_list[ii]]<4000] 

#title_strg = "ratio_to_poi_to_all_sent_emails"
#p18 = df_plotter(new_df,features_list[ii], features_list[ij], title_strg)

# Inserting the new feature to df_to_keep
df_to_keep = new_variableAdder('ratio_to_poi_to_all_sent_emails',
                               df_to_keep,new_df,features_list[ii],features_list[ij])
# new feature information 
print_info_new_feature('ratio_to_poi_to_all_sent_emails',new_df,df_to_keep,namesPoi)



## updating feature list
del features_list
names = list(df_to_keep.index.values)
features_list = list(df_to_keep.columns.values)
# adding 'poi' to first position in the list
features_list = poi_to_first(features_list)


# storing data to dict my_dataset
my_dataset = df_to_keep.to_dict(orient='index')

# deleting the unused dataframes
del all_df
del new_df
del df_to_keep




### Task 4: Try a varity of classifiers
print " "
print "other classifiers tested. See brainstorming.py and previous version testingVariousClassifiers.py"
print " "


### Task 5: Tune your classifier to achieve better than .3 precision and recall 

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# getting the best features
features=get_featuresScore(3,labels, features)

# building a new dictionary only with features of interest
# and updating features_list
a = pd.DataFrame(labels, columns = ['poi'])
b = pd.DataFrame(features, columns = ['exercised_stock_options','bonus','total_stock_value'])
df = pd.concat([a,b], axis = 1)

features_list = df.columns.tolist()

df.index = names
my_dataset = df.to_dict(orient='index')


# train/test spliting
# test size for train/test splitting set to 0.3 
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, 
                                                                            test_size=0.3, random_state=42)


# data scaling to improve classifier accuracy
features_train,features_test = min_max_scaling(features_train,features_test)

# decision tree classifier tuning using a k best selection (optimized at k=3)
bestParameters, clf = kbest_dTree(features_train, features_test, labels_train, labels_test)



### testing the classifier
test_classifier(clf, data, labels, features_list, folds = 1000)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
dump_classifier_and_data(clf, my_dataset, features_list)

print "######### ALL DONE SUCCESSFULLY ##########"