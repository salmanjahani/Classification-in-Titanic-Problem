# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:56:11 2019

@author: jahani
"""

# Import necessary modules for data analysis and data visualization. 
# Data analysis modules
# Pandas is probably the most popular and important modules for any work related to data management. 
import pandas as pd
# numpy is a great library for doing mathmetical operations. 
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import mean_absolute_error, accuracy_score


from scipy.stats import randint, uniform

# Some visualization libraries
from matplotlib import pyplot as plt
import seaborn as sns

## Some other snippit of codes to get the setting right 
## This is so that the chart created by matplotlib can be shown in the jupyter notebook. 
import warnings ## importing warnings library. 
warnings.filterwarnings('ignore') ## Ignore warning
import os ## imporing os

## Importing the datasets
train = pd.read_csv("C:\\Users\\jahan\\input\\train.csv")
test = pd.read_csv("C:\\Users\\jahan\\input\\test.csv")

## Take a look at the overview of the dataset. 
train.sample(5)

test.sample(5)

print ("The shape of the train data is (row, column):"+ str(train.shape))
print (train.info())
print ("The shape of the test data is (row, column):"+ str(test.shape))
print (test.info())

passengerid = test.PassengerId
## We will drop PassengerID and Ticket since it will be useless for our data. 
#train.drop(['PassengerId'], axis=1, inplace=True)
#test.drop(['PassengerId'], axis=1, inplace=True)

print (train.info())
print ("*"*40)
print (test.info())


total = train.isnull().sum().sort_values(ascending = False)
percent = round(train.isnull().sum().sort_values(ascending = False)/len(train)*100, 2)
pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])


total = test.isnull().sum().sort_values(ascending = False)
percent = round(test.isnull().sum().sort_values(ascending = False)/len(test)*100, 2)
pd.concat([total, percent], axis = 1,keys= ['Total', 'Percent'])


percent = pd.DataFrame(round(train.Embarked.value_counts(dropna=False, normalize=True)*100,2))
## creating a df with th
total = pd.DataFrame(train.Embarked.value_counts(dropna=False))
## concating percent and total dataframe

total.columns = ["Total"]
percent.columns = ['Percent']
pd.concat([total, percent], axis = 1)

train[train.Embarked.isnull()]


fig, ax = plt.subplots(figsize=(16,12),ncols=2)
ax1 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=train, ax = ax[0]);
ax2 = sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=test, ax = ax[1]);
ax1.set_title("Training Set", fontsize = 18)
ax2.set_title('Test Set',  fontsize = 18)
fig.show()


train.Embarked.fillna("C", inplace=True)

print("Train Cabin missing: " + str(train.Cabin.isnull().sum()/len(train.Cabin)))
print("Test Cabin missing: " + str(test.Cabin.isnull().sum()/len(test.Cabin)))

## Concat train and test into a variable "all_data"
survivers = train.Survived

train.drop(["Survived"],axis=1, inplace=True)

all_data = pd.concat([train,test], ignore_index=False)

## Assign all the null values to N
all_data.Cabin.fillna("N", inplace=True)


all_data.Cabin = [i[0] for i in all_data.Cabin]

with_N = all_data[all_data.Cabin == "N"]

without_N = all_data[all_data.Cabin != "N"]

all_data.groupby("Cabin")['Fare'].mean().sort_values()
def cabin_estimator(i):
    a = 0
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<38:
        a = "T"
    elif i>=38 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a

##applying cabin estimator function. 
with_N['Cabin'] = with_N.Fare.apply(lambda x: cabin_estimator(x))

## getting back train. 
all_data = pd.concat([with_N, without_N], axis=0)

## PassengerId helps us separate train and test. 
all_data.sort_values(by = 'PassengerId', inplace=True)

## Separating train and test from all_data. 
train = all_data[:891]

test = all_data[891:]

# adding saved target variable with train. 
train['Survived'] = survivers


test[test.Fare.isnull()]

missing_value = test[(test.Pclass == 3) & (test.Embarked == "S") & (test.Sex == "male")].Fare.mean()
## replace the test.fare null values with test.fare mean
test.Fare.fillna(missing_value, inplace=True)

print ("Train age missing value: " + str((train.Age.isnull().sum()/len(train))*100)+str("%"))
print ("Test age missing value: " + str((test.Age.isnull().sum()/len(test))*100)+str("%"))


## dropping the three outliers where Fare is over $500 
train = train[train.Fare < 500]
## factor plot
sns.factorplot(x = "Parch", y = "Survived", data = train,kind = "point",size = 8)
plt.title("Factorplot of Parents/Children survived", fontsize = 25)
plt.subplots_adjust(top=0.85)





# Placing 0 for female and 
# 1 for male in the "Sex" column. 
train['Sex'] = train.Sex.apply(lambda x: 0 if x == "female" else 1)
test['Sex'] = test.Sex.apply(lambda x: 0 if x == "female" else 1)


###########Part 5: Feature Engineering

# Creating a new colomn with a 
train['name_length'] = [len(i) for i in train.Name]
test['name_length'] = [len(i) for i in test.Name]

def name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'good'
    else:
        a = 'long'
    return a


train['nLength_group'] = train['name_length'].map(name_length_group)
test['nLength_group'] = test['name_length'].map(name_length_group)

## Here "map" is python's built-in function. 
## "map" function basically takes a function and 
## returns an iterable list/tuple or in this case series. 
## However,"map" can also be used like map(function) e.g. map(name_length_group) 
## or map(function, iterable{list, tuple}) e.g. map(name_length_group, train[feature]]). 
## However, here we don't need to use parameter("size") for name_length_group because when we 
## used the map function like ".map" with a series before dot, we are basically hinting that series 
## and the iterable. This is similar to .append approach in python. list.append(a) meaning applying append on list. 
## cuts the column by given bins based on the range of name_length
#group_names = ['short', 'medium', 'good', 'long']
#train['name_len_group'] = pd.cut(train['name_length'], bins = 4, labels=group_names)

## get the title from the name
train["title"] = [i.split('.')[0] for i in train.Name]
train["title"] = [i.split(',')[1] for i in train.title]
test["title"] = [i.split('.')[0] for i in test.Name]
test["title"]= [i.split(',')[1] for i in test.title]
#rare_title = ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col']
#train.Name = ['rare' for i in train.Name for j in rare_title if i == j]
## train Data
train["title"] = [i.replace('Ms', 'Miss') for i in train.title]
train["title"] = [i.replace('Mlle', 'Miss') for i in train.title]
train["title"] = [i.replace('Mme', 'Mrs') for i in train.title]
train["title"] = [i.replace('Dr', 'rare') for i in train.title]
train["title"] = [i.replace('Col', 'rare') for i in train.title]
train["title"] = [i.replace('Major', 'rare') for i in train.title]
train["title"] = [i.replace('Don', 'rare') for i in train.title]
train["title"] = [i.replace('Jonkheer', 'rare') for i in train.title]
train["title"] = [i.replace('Sir', 'rare') for i in train.title]
train["title"] = [i.replace('Lady', 'rare') for i in train.title]
train["title"] = [i.replace('Capt', 'rare') for i in train.title]
train["title"] = [i.replace('the Countess', 'rare') for i in train.title]
train["title"] = [i.replace('Rev', 'rare') for i in train.title]



#rare_title = ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col']
#train.Name = ['rare' for i in train.Name for j in rare_title if i == j]
## test data
test['title'] = [i.replace('Ms', 'Miss') for i in test.title]
test['title'] = [i.replace('Dr', 'rare') for i in test.title]
test['title'] = [i.replace('Col', 'rare') for i in test.title]
test['title'] = [i.replace('Dona', 'rare') for i in test.title]
test['title'] = [i.replace('Rev', 'rare') for i in test.title]

## Family_size seems like a good feature to create
train['family_size'] = train.SibSp + train.Parch+1
test['family_size'] = test.SibSp + test.Parch+1
def family_group(size):
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a
train['family_group'] = train['family_size'].map(family_group)
test['family_group'] = test['family_size'].map(family_group)


train['is_alone'] = [1 if i<2 else 0 for i in train.family_size]
test['is_alone'] = [1 if i<2 else 0 for i in test.family_size]

train.drop(['Ticket'], axis=1, inplace=True)

test.drop(['Ticket'], axis=1, inplace=True)


train['calculated_fare'] = train.Fare/train.family_size
test['calculated_fare'] = test.Fare/test.family_size


def fare_group(fare):
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a

train['fare_group'] = train['calculated_fare'].map(fare_group)
test['fare_group'] = test['calculated_fare'].map(fare_group)
train.drop(['PassengerId'], axis=1, inplace=True)

test.drop(['PassengerId'], axis=1, inplace=True)

train = pd.get_dummies(train, columns=['title',"Pclass", 'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=False)
test = pd.get_dummies(test, columns=['title',"Pclass",'Cabin','Embarked','nLength_group', 'family_group', 'fare_group'], drop_first=False)
train.drop(['family_size','Name', 'Fare','name_length'], axis=1, inplace=True)
test.drop(['Name','family_size',"Fare",'name_length'], axis=1, inplace=True)

 
train = pd.concat([train[["Survived", "Age", "Sex","SibSp","Parch"]], train.loc[:,"is_alone":]], axis=1)
test = pd.concat([test[["Age", "Sex"]], test.loc[:,"SibSp":]], axis=1)


from sklearn.ensemble import RandomForestRegressor

## writing a function that takes a dataframe with missing values and outputs it by filling the missing values. 
def completing_age(df):
    ## gettting all the features except survived
    age_df = df.loc[:,"Age":] 
    
    temp_train = age_df.loc[age_df.Age.notnull()] ## df with age values
    temp_test = age_df.loc[age_df.Age.isnull()] ## df without age values
    
    y = temp_train.Age.values ## setting target variables(age) in y 
    x = temp_train.loc[:, "Sex":].values
    
    rfr = RandomForestRegressor(n_estimators=1500, n_jobs=-1)
    rfr.fit(x, y)
    
    predicted_age = rfr.predict(temp_test.loc[:, "Sex":])
    
    df.loc[df.Age.isnull(), "Age"] = predicted_age
    

    return df

## Implementing the completing_age function in both train and test dataset. 
completing_age(train)
completing_age(test);

plt.subplots(figsize = (22,10),)
sns.distplot(train.Age, bins = 100, kde = True, rug = False, norm_hist=False);


def age_group_fun(age):
    a = ''
    if age <= 1:
        a = 'infant'
    elif age <= 4: 
        a = 'toddler'
    elif age <= 13:
        a = 'child'
    elif age <= 18:
        a = 'teenager'
    elif age <= 35:
        a = 'Young_Adult'
    elif age <= 45:
        a = 'adult'
    elif age <= 55:
        a = 'middle_aged'
    elif age <= 65:
        a = 'senior_citizen'
    else:
        a = 'old'
    return a
        
## Applying "age_group_fun" function to the "Age" column.
train['age_group'] = train['Age'].map(age_group_fun)
test['age_group'] = test['Age'].map(age_group_fun)

## Creating dummies for "age_group" feature. 
train = pd.get_dummies(train,columns=['age_group'], drop_first=True)
test = pd.get_dummies(test,columns=['age_group'], drop_first=True);

"""train.drop('Age', axis=1, inplace=True)
test.drop('Age', axis=1, inplace=True)"""

train = train.drop(['Cabin_A','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Cabin_T','nLength_group_long', 'nLength_group_medium', 'nLength_group_short','nLength_group_good',
       'family_group_large', 'family_group_loner', 'family_group_small',
       'fare_group_Very_low', 'fare_group_high', 'fare_group_low',
       'fare_group_mid', 'fare_group_very_high', 'age_group_adult',
       'age_group_child', 'age_group_infant', 'age_group_middle_aged',
       'age_group_old', 'age_group_senior_citizen', 'age_group_teenager',
       'age_group_toddler'], axis = 1)
test = test.drop(['Cabin_A','Cabin_B','Cabin_C','Cabin_D','Cabin_E','Cabin_F','Cabin_G','Cabin_T','nLength_group_long', 'nLength_group_medium', 'nLength_group_short','nLength_group_good',
       'family_group_large', 'family_group_loner', 'family_group_small',
       'fare_group_Very_low', 'fare_group_high', 'fare_group_low',
       'fare_group_mid', 'fare_group_very_high', 'age_group_adult',
       'age_group_child', 'age_group_infant', 'age_group_middle_aged',
       'age_group_old', 'age_group_senior_citizen', 'age_group_teenager',
       'age_group_toddler'], axis = 1)

# separating our independent and dependent variable
X = train.drop(['Survived'], axis = 1)
y = train["Survived"]



#age_filled_data_nor = NuclearNormMinimization().complete(df1)
#Data_1 = pd.DataFrame(age_filled_data, columns = df1.columns)
#pd.DataFrame(zip(Data["Age"],Data_1["Age"],df["Age"]))



from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(X,y,test_size = .33, random_state = 0)

train.sample()

headers = train_x.columns 

train_x.head()

# Feature Scaling
## We will be using standardscaler to transform
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

## transforming "train_x"
train_x = sc.fit_transform(train_x)
## transforming "test_x"
test_x = sc.transform(test_x)

## transforming "The testset"
test = sc.transform(test)



pd.DataFrame(train_x, columns=headers).head()



######Models#######
# Number of cross-validation folds
k_folds = 10

n_estimators = 100

random_state = 1

# Create a dictionary containing the instance of the models, scores, mean accuracy and standard deviation
classifiers = {
    'name': ['LogReg', 'KNN'],
    'models': [LogisticRegression(random_state=random_state),
               KNeighborsClassifier()], 
    'scores': [],
    'acc_mean': [],
    'acc_std': []
}



# Run cross-validation and store the scores

for model in classifiers['models']:
    if __name__ == '__main__':
        score = cross_val_score(model, train_x, train_y, cv=k_folds)
        classifiers['scores'].append(score)
        classifiers['acc_mean'].append(score.mean())
        classifiers['acc_std'].append(score.std())    

    # Create a nice table with the results
classifiers_df = pd.DataFrame({
    'Model Name': classifiers['name'],
    'Accuracy': classifiers['acc_mean'],
    'Std': classifiers['acc_std']
}, columns=['Model Name', 'Accuracy', 'Std']).set_index('Model Name')

classifiers_df.sort_values('Accuracy', ascending=False)


cross_val_score(KNeighborsClassifier(), train_x, train_y, cv=k_folds)

# Utility function to report best scores
# Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html#sphx-glr-auto-examples-model-selection-plot-randomized-search-py
def report(results, n_top=3, limit=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        if limit is not None:
            candidates = candidates[:limit]
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.4f} (std: {1:.4f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print()

# Number of iterations
n_iter_search = 200

logreg = LogisticRegression(random_state=random_state)
rand_param = {
    'penalty': ['l1'],
    'C': uniform(0.01, 10)
}

logreg_search = RandomizedSearchCV(logreg, param_distributions=rand_param, n_iter=n_iter_search, cv=k_folds)
logreg_search.fit(train_x, train_y)
report(logreg_search.cv_results_)

logreg_best = logreg_search.best_estimator_
print(X.columns) 
print('Coefficient of LR:', logreg_best.coef_)

logreg_pred = logreg_best.predict(test_x)

print ("Logistic Regression's accuracy Score is: {}".format(round(accuracy_score(logreg_pred, test_y),4)))


from sklearn.metrics import roc_curve, auc
#plt.style.use('seaborn-pastel')
y_score = logreg_best.predict_proba(test_x)[:,1]
FPR, TPR, _ = roc_curve(test_y, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[8,8])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive', fontsize = 18)
plt.ylabel('True Positive', fontsize = 18)
plt.title('ROC for Logistic Regression', fontsize= 18)
plt.show()

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(test_y, y_score)
PR_AUC = auc(recall, precision)
print (PR_AUC)


plt.figure(figsize=[8,8])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for Logistic Regression', fontsize=18)
plt.legend(loc="lower right")
plt.show()


## KNN
knn = KNeighborsClassifier()
rand_param = {
    'n_neighbors': randint(1, 25),
    'leaf_size': randint(1, 50),
    'weights': ['uniform', 'distance']
}

knn_search = RandomizedSearchCV(knn, param_distributions=rand_param, n_iter=n_iter_search, cv=k_folds)#, n_jobs=4, verbose=1)
knn_search.fit(train_x, train_y)
report(knn_search.cv_results_)

knn_best = knn_search.best_estimator_

knn_pred = knn_best.predict(test_x)


print ("KNN's accuracy Score is: {}".format(round(accuracy_score(knn_pred, test_y),4)))

#plt.style.use('seaborn-pastel')
y_score = knn_best.predict_proba(test_x)
y_score = y_score[:, 1]

FPR, TPR, _ = roc_curve(test_y, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[8,8])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive', fontsize = 18)
plt.ylabel('True Positive', fontsize = 18)
plt.title('ROC for KNN', fontsize= 18)
plt.show()

from sklearn.metrics import precision_recall_curve

precision, recall, _ = precision_recall_curve(test_y, y_score)
PR_AUC = auc(recall, precision)
print (PR_AUC)

plt.figure(figsize=[8,8])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for KNN', fontsize=18)
plt.legend(loc="lower right")
plt.show()


## SVC
svc = SVC(random_state=random_state, probability=True)
rand_param = {
    'C': uniform(0.01, 10),
    'gamma': uniform(0.01, 10)
 }

svc_search = RandomizedSearchCV(svc, param_distributions=rand_param, n_iter=n_iter_search, cv=k_folds, n_jobs=4, verbose=1)
svc_search.fit(train_x, train_y)
report(svc_search.cv_results_)

svc_best = svc_search.best_estimator_
svc_pred = svc_best.predict(test_x)


print ("SVC's accuracy Score is: {}".format(round(accuracy_score(svc_pred, test_y),4)))
y_score = svc_best.predict_proba(test_x)
y_score = y_score[:, 1]

FPR, TPR, _ = roc_curve(test_y, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[8,8])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive', fontsize = 18)
plt.ylabel('True Positive', fontsize = 18)
plt.title('ROC for SVC', fontsize= 18)
plt.show()

precision, recall, _ = precision_recall_curve(test_y, y_score)
PR_AUC = auc(recall, precision)
print (PR_AUC)

plt.figure(figsize=[8,8])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for SVC', fontsize=18)
plt.legend(loc="lower right")
plt.show()


## GP
from sklearn.gaussian_process import GaussianProcessClassifier
GaussianProcessClassifier = GaussianProcessClassifier()
GaussianProcessClassifier.fit(train_x, train_y)
GP_pred = GaussianProcessClassifier.predict(test_x)
GP_pred2 = GaussianProcessClassifier.predict(train_x)


print ("GP's accuracy Score is: {}".format(round(accuracy_score(GP_pred2, train_y),4)))
print ("GP's accuracy Score is: {}".format(round(accuracy_score(GP_pred, test_y),4)))
y_score = GaussianProcessClassifier.predict_proba(test_x)
y_score = y_score[:, 1]

FPR, TPR, _ = roc_curve(test_y, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[8,8])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive', fontsize = 18)
plt.ylabel('True Positive', fontsize = 18)
plt.title('ROC for GP', fontsize= 18)
plt.show()

precision, recall, _ = precision_recall_curve(test_y, y_score)
PR_AUC = auc(recall, precision)
print (PR_AUC)

plt.figure(figsize=[8,8])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for GP', fontsize=18)
plt.legend(loc="lower right")
plt.show()

## Adaboost
ada = AdaBoostClassifier(random_state=random_state, n_estimators=n_estimators)
rand_param = {
    'learning_rate': uniform(0.1, 10),
}

ada_search = RandomizedSearchCV(ada, param_distributions=rand_param, n_iter=n_iter_search, cv=k_folds, n_jobs=4, verbose=1)
ada_search.fit(train_x, train_y)
report(ada_search.cv_results_)

ada_best = ada_search.best_estimator_
ada_pred = ada_best.predict(test_x)


print ("Adaboost's accuracy Score is: {}".format(round(accuracy_score(svc_pred, test_y),4)))
y_score = svc_best.predict_proba(test_x)
y_score = y_score[:, 1]

FPR, TPR, _ = roc_curve(test_y, y_score)
ROC_AUC = auc(FPR, TPR)
print (ROC_AUC)

plt.figure(figsize =[8,8])
plt.plot(FPR, TPR, label= 'ROC curve(area = %0.2f)'%ROC_AUC, linewidth= 4)
plt.plot([0,1],[0,1], 'k--', linewidth = 4)
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive', fontsize = 18)
plt.ylabel('True Positive', fontsize = 18)
plt.title('ROC for Adaboost', fontsize= 18)
plt.show()

precision, recall, _ = precision_recall_curve(test_y, y_score)
PR_AUC = auc(recall, precision)
print (PR_AUC)

plt.figure(figsize=[8,8])
plt.plot(recall, precision, label='PR curve (area = %0.2f)' % PR_AUC, linewidth=4)
plt.xlabel('Recall', fontsize=18)
plt.ylabel('Precision', fontsize=18)
plt.title('Precision Recall Curve for SVC', fontsize=18)
plt.legend(loc="lower right")
plt.show()

#####################################
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(train_x, train_y)
GNB_pred = gaussian.predict(test_x)
GNB_pred2 = gaussian.predict(train_x)

print ("GNB's accuracy Score is: {}".format(round(accuracy_score(GNB_pred2, train_y),4)))
print ("GNB's accuracy Score is: {}".format(round(accuracy_score(GNB_pred, test_y),4)))


###########
#Random Forrest
from sklearn.ensemble import RandomForestClassifier
n_estimators = [140,145,150,155,160];
max_depth = range(1,10);
criterions = ['gini', 'entropy'];
cv = 10


parameters = {'n_estimators':n_estimators,
              'max_depth':max_depth,
              'criterion': criterions
              
        }
grid = GridSearchCV(estimator=RandomForestClassifier(max_features='auto'),
                                 param_grid=parameters,
                                 cv=cv,
                                 n_jobs = -1)
grid.fit(train_x, train_y) 

print (grid.best_score_)
print (grid.best_params_)
print (grid.best_estimator_)
rf_grid = grid.best_estimator_
rf_grid.score(test_x,test_y)
rf_pred =rf_grid.predict(test_x)
######################
####################################



## Ensemble
# Similarity for target = 0
pred_df = pd.DataFrame({
    'LR': logreg_pred[test_y==0],
    'KNN': knn_pred[test_y==0],
    'SVC': svc_pred[test_y==0],
    'GP': GP_pred[test_y==0],
    'AB': ada_pred[test_y==0],
    'GNB': GNB_pred[test_y==0],
    'RF': rf_pred[test_y==0]
})

jsim_df = pd.DataFrame(np.nan, columns=pred_df.columns, index=pred_df.columns)
for i in pred_df.columns:
    for j in pred_df.loc[:, i:].columns:
        jsim_df.loc[i, j] = jaccard_similarity_score(pred_df[i], pred_df[j])
        jsim_df.loc[j, i] = jsim_df.loc[i, j]

plt.figure(figsize=(20,10)) 
ax = plt.axes()
sns.set(font_scale=3)
sns.heatmap(jsim_df, linewidths=0.1, ax = ax, vmax=1.0, vmin=0, square=True, linecolor='white', annot=True, cmap='coolwarm')
ax.set_title('Similarity for Not Survived')
plt.show()

# Similarity for target = 0
pred_df = pd.DataFrame({
    'LR': logreg_pred[test_y==1],
    'KNN': knn_pred[test_y==1],
    'SVC': svc_pred[test_y==1],
    'GP': GP_pred[test_y==1],
    'AB': ada_pred[test_y==1],
    'GNB': GNB_pred[test_y==1],
    'RF': rf_pred[test_y==1]
})

jsim_df = pd.DataFrame(np.nan, columns=pred_df.columns, index=pred_df.columns)
for i in pred_df.columns:
    for j in pred_df.loc[:, i:].columns:
        jsim_df.loc[i, j] = jaccard_similarity_score(pred_df[i], pred_df[j])
        jsim_df.loc[j, i] = jsim_df.loc[i, j]

plt.figure(figsize=(20,10)) 
ax = plt.axes()
sns.set(font_scale=3)
sns.heatmap(jsim_df, linewidths=0.1, ax = ax, vmax=1.0, vmin=0, square=True, linecolor='white', annot=True, cmap='coolwarm')
ax.set_title('Similarity for Survived')
plt.show()

## Ensemble
estimators = [
    ('Logistic Regression', logreg_best),
    ('KNN', knn_best),
    ('SVC', svc_best),
    ('GP', GaussianProcessClassifier),
    ('AB', ada_best),
    ('GNB', gaussian),
    ('RF', rf_grid)
    ]

eclf = VotingClassifier(estimators=estimators)
ensemble_param = {'voting': ['hard', 'soft']}

eclf_search = GridSearchCV(eclf, param_grid=ensemble_param, cv=10, n_jobs=4, verbose=1)
eclf_search.fit(train_x, train_y)
report(eclf_search.cv_results_)

eclf_best = eclf_search.best_estimator_
eclf_pred = eclf_best.predict(test_x)

print ("Ensemble's accuracy Score is: {}".format(round(accuracy_score(eclf_pred, test_y),4)))
