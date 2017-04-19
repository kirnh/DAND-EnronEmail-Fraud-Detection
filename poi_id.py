#!/usr/bin/python

### Importing the required modules and functions
import sys
import pickle
sys.path.append("../tools/")
import pprint
from time import time

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Dealing with outliers
data_dict.pop("TOTAL")

### Some information about the dataset
print '\n----------------------------Dataset details:\n'
print 'Number of data points:', len(data_dict)
for i in data_dict:
	print 'Number of features:', len(data_dict[i])
	break

### Creating new feature/s
def compute_fraction(poi_messages, all_messages):
	if poi_messages == "NaN" or all_messages == "NaN":
		fraction = 0
	else:
		fraction = float(poi_messages) / all_messages
	return fraction

for name in data_dict:
	data_point = data_dict[name]
	data_point["fraction_from_poi"] = compute_fraction(data_point["from_poi_to_this_person"], data_point["to_messages"])
	data_point["fraction_to_poi"] = compute_fraction(data_point["from_this_person_to_poi"], data_point["from_messages"])

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Loading the list of features to be used
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 
'loan_advances', 'bonus', 'restricted_stock_deferred', 
'deferred_income', 'total_stock_value', 'expenses', 
'exercised_stock_options', 'other', 'long_term_incentive', 
'restricted_stock', 'director_fees', 'to_messages', 
'from_poi_to_this_person', 'from_messages', 
'from_this_person_to_poi', 'shared_receipt_with_poi', 
'fraction_from_poi', 'fraction_to_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
print 'Number of POIs', int(sum(labels)), "\n"

### Counting NaN values
def count_nan(dataset, feature):
	count = 0
	for name in dataset:
		if dataset[name][feature] == "NaN":
			count += 1
	return count 

print "----------------------------Counting missing values for features:\n"
for feature in features_list:
	if feature not in ["poi", "fraction_from_poi", "fraction_to_poi"]:
		print str(feature),": ", count_nan(data_dict, feature)

### Split data into training and testing set for analysing an example train/test split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Exploring some algorithms to find best ones for further tuning 
print "\n----------------------------Testing basic models:"
print "\nGaussianNB\n"
gnb = GaussianNB()
gnb.fit(features_train, labels_train)
pred = gnb.predict(features_test)
print "accuracy_score:", accuracy_score(labels_test, pred)
print "recall_score:", recall_score(labels_test, pred)
print "precision_score:", precision_score(labels_test, pred)

print "\nDecisionTreeClassifier\n"
dt = DecisionTreeClassifier(random_state = 24)
dt.fit(features_train, labels_train)
pred = dt.predict(features_test)
print "accuracy_score:", accuracy_score(labels_test, pred)
print "recall_score:", recall_score(labels_test, pred)
print "precision_score:", precision_score(labels_test, pred)

### Comparing optimized model of DecisionTreeClassifier with GaussianNB to choose the bests
dt = DecisionTreeClassifier(random_state = 24)
param_grid_dt = [{'criterion': ('gini', 'entropy'),
				'min_samples_split':[2, 10, 20],
              	'max_depth':[10,15,20,25,30],
              	'max_leaf_nodes':[5,10,30]}]

print "\n----------------------------Tuned DecisionTreeClassifier:\n"
grid = GridSearchCV(dt, param_grid = param_grid_dt, scoring = 'f1')
grid.fit(features_train, labels_train)
dt = grid.best_estimator_
pred = dt.predict(features_test)
print "\naccuracy_score:", accuracy_score(labels_test, pred)
print "recall_score:", recall_score(labels_test, pred)
print "precision_score:", precision_score(labels_test, pred), '\n'
print dt

### Creating selector
skb = SelectKBest()

### Choosing the best value of k parameter of SelectKBest using GridSearchCV
print "\n----------------------------After including SelectKBest in GridSearchCV:\n"
param_grid = [{'skb__k': range(1, 22)}]
pipe = Pipeline([('skb', skb),
				('dt', dt)])
grid = GridSearchCV(pipe, param_grid = param_grid, scoring = 'f1')
grid.fit(features_train, labels_train)
dt = grid.best_estimator_
pred = dt.predict(features_test)
print "accuracy_score:", accuracy_score(labels_test, pred)
print "recall_score:", recall_score(labels_test, pred)
print "precision_score:", precision_score(labels_test, pred), '\n'
print dt

### Univariate feature selection using SelectKBest
skb = SelectKBest(k = 10)
selectedFeatures = skb.fit(features_train, labels_train)
features = selectedFeatures.transform(features)
scores = list(skb.scores_)
print "\n----------------------------Printing feature scores from SelectKBest:\n"
for i in scores:
	index = scores.index(i)
	print features_list[index + 1], ": ", i

print "\n----------------------------Printing 10 best features selected by SelectKBest:\n"
feature_names = [features_list[i + 1] for i in selectedFeatures.get_support(indices=True)]
features_list = ['poi']
for i in feature_names:
	print i
	features_list.append(i)

### Crossvalidating the model using StratifiedShuffleSplit and GridSearchCV
sss = StratifiedShuffleSplit(n_splits = 100, random_state = 24)
dt = DecisionTreeClassifier(random_state = 24)
pipe_dt = Pipeline([('dt', dt)])
param_grid_dt = [{'dt__criterion': ('gini', 'entropy'),
				'dt__min_samples_split':[2, 10, 20],
              	'dt__max_depth':[10,15,20,25,30],
              	'dt__max_leaf_nodes':[5,10,30]}]
grid = GridSearchCV(pipe, param_grid = param_grid_dt, cv = sss, scoring = 'f1')
grid.fit(features, labels)
clf = grid.best_estimator_

### Validating the models using test_classifier function from tester module
from tester import test_classifier

print "\n----------------------------Testing final DecisionTreeClassifier:\n"
test_classifier(clf, my_dataset, features_list, folds = 1000)

### Dumping required data into files
dump_classifier_and_data(clf, my_dataset, features_list)
