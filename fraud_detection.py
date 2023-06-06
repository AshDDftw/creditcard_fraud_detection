#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries and modules

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
from collections import Counter

import warnings
warnings.filterwarnings("ignore")


# In[2]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier, IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier as KNN


# In[3]:


from xgboost import XGBClassifier


# In[4]:


# from imblearn.under_sampling import NearMiss
# from imblearn.over_sampling import RandomOverSampler


# ## Problem Statement

# Credit risk is associated with the possibility of a client failing to meet contractual obligations,
# such as mortgages, credit card debts, and other types of loans. The dataset contains transactions
# made by credit cards in September 2013 by European cardholders. This dataset presents
# transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions.
# The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all
# transactions. These datasets are hard to handle. You have to predict whether, given the details
# about the credit card, it is real or fake.

# ## Loading dataset and preprocessing

# In[5]:


df = pd.read_csv('creditcard.csv')
df_raw = df.copy()


# In[6]:


df


# ### Data Analysis

# In[7]:


df.info()


# ### Checking the number of NULL data entries (if any)

# In[8]:


df.isnull().sum()


# ### Number of unique class labels and their corresponding number of data points

# In[9]:


classes = df.Class.unique()
print(list(classes))
classes_sum = [df[df['Class']==x].shape[0] for x in classes]
print(classes_sum)


# ### Plotting Correlation Heatmap

# In[10]:


df_corr = df.corr()


# In[11]:


fig = plt.figure(figsize=(50,40))
sns.heatmap(df_corr, annot=True)
plt.title('Correlation Heatmap')
plt.show()


# ### Data Visualization

# ### Plotting dataset histogram

# In[12]:


df.hist(figsize=(25,25))
plt.show()


# ### Plotting counts of class labels

# In[13]:


def plot_counts(df, feature):
    '''
    Input: dataframe, feature
    
    Plots the number of counts for each unique entry in the input feature column of the input dataframe
    '''
    uniquearray = df[feature].unique()
    
    uniquedict = dict()
    for each in uniquearray:
        uniquedict[each] = df[df[feature]==each].shape[0]
        
    names = list(uniquedict.keys())
    values = list(uniquedict.values())
    
    plt.bar(range(len(uniquearray)), values, tick_label=names)
    plt.ylabel('count')
    plt.xlabel(feature)
    
    for each in range(len(uniquearray)):
        plt.text(each, values[each]+3000, values[each], ha = 'center')
    
    plt.show()
    


# In[14]:


fig = plt.figure(figsize=(10, 7))
plot_counts(df, 'Class')


# ### Dropping the unneccessary columns (not useful in training the model)

# In[15]:


columns_dropped = ['Time']


# In[16]:


df = df.drop(columns=columns_dropped, axis=1)


# In[17]:


df.describe()


# In[18]:


df.columns


# In[19]:


features = df.columns.tolist()[:-1]
# features


# ### Applying standardization on the dataset

# In[20]:


scaler = StandardScaler()


# In[21]:


X = df.copy()[features]
y = df.Class


# In[22]:


X['Amount'] = scaler.fit_transform(np.array(X['Amount']).reshape(-1,1))
# X_scaled = scaler.fit_transform(X)
# X_scaled = X.copy()
# X_unscaled = X.copy()
# X_scaled = pd.DataFrame(X_scaled)
# X_scaled.columns = features
# X = X_scaled


# In[23]:


X


# ### Train-test split

# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


df_train = X_train.copy()
df_train['Class'] = y_train

df_test = X_test.copy()
df_test['Class'] = y_test


# ## Outlier removal (using Isolation Forest method)

# In[26]:


df_out = df_raw.copy()


# In[27]:


X_out = df_out.drop(['Class'], axis=1)
y_out = df_out['Class']


# In[28]:


iso_forest = IsolationForest(n_estimators=100, max_samples='auto', contamination=float(0.01),max_features=0.1)
iso_forest.fit(df_out)


# In[29]:


scores = iso_forest.decision_function(df_out)


# In[30]:


y_pred_out = iso_forest.predict(df_out)


# In[31]:


df_out['scores'] = scores


# In[32]:


df_out['is_outlier'] = y_pred_out


# In[33]:


print('Number of anomalies detected:', sum(df_out['is_outlier'] == -1))


# ### Before outlier removal

# In[34]:


sns.boxplot(data=df_raw['Amount'], color="#a2d2ff").set_title("Amount")

plt.tight_layout
plt.show()


# ### After outlier removal

# In[35]:


anomaly = df_out.loc[df_out['is_outlier']==-1]
anomaly_index = list(anomaly.index)

df_out = df_out.drop(anomaly_index, axis = 0).reset_index(drop=True)

sns.boxplot(data=df_out['Amount'], color="#a2d2ff").set_title('Amount')

plt.tight_layout()
plt.show()


# # Methods to handle imbalanced dataset

# In[36]:


def Visualizing(d, target):
    '''
    Input: training dataset and labels
    
    Plots the scatter plot of transformed input datapoints for each class
    after applying PCA (n_components=2) on the input data
    '''
    pca = PCA(n_components=2)
    dataset = pca.fit_transform(d)
    # print(dataset)
    d = pd.DataFrame(data=dataset, columns=['x1','x2'])
    y = pd.DataFrame(data=target, columns=['Class']).reset_index(drop=True)

    plt.figure(figsize=(7,7))
    plt.scatter(d['x1'][y.Class==1], d['x2'][y.Class==1], color='red')
    plt.scatter(d['x1'][y.Class==0], d['x2'][y.Class==0], color='blue')
    plt.show()
    


# In[37]:


def PCA_KNN(X_train, y_train, X_test, y_test, n_neighbors=2):
    '''
    Input: training dataset and labels, test dataset and labels, n_neighbors for KNN classifier
    
    Applies PCA(n_components=2) on the input training dataset and uses it to transform test dataset,
    displays results of classification on the transformed test data using KNN classifier
    '''
    pca = PCA(n_components=2)
    dataset = pca.fit_transform(X_train)
    d = pd.DataFrame(data=dataset, columns=['x1','x2'])

    X_test_pca = pca.transform(X_test)

    knn = KNN(n_neighbors=1)
    knn.fit(d, y_train)
    y_pred = knn.predict(X_test_pca)
    print(f'accuracy score: {accuracy_score(y_test, y_pred)}')
    print(f'\nf1-score: {f1_score(y_test, y_pred)}')
    print('\nClassification report:-')
    print(classification_report(y_test, y_pred))
    


# ## Undersamping (from scratch)

# In[38]:


count_class_0, count_class_1 = df_train.Class.value_counts()

df_class_0 = df_train[df_train['Class'] == 0]
df_class_1 = df_train[df_train['Class'] == 1]


# In[39]:


df_class_0_US = df_class_0.sample(count_class_1)
df_US = pd.concat([df_class_0_US, df_class_1], axis=0)

print('Random Under-sampling:')
print(df_US.Class.value_counts())

X_train_US = df_US.drop('Class',axis='columns')
y_train_US = df_US['Class']


# In[40]:


# ns = NearMiss(sampling_strategy=1)
# X_train_US, y_train_US = ns.fit_resample(X_train, y_train)

print(f'The number of classes before fit {Counter(y_train)}')
print(f'The number of classes after fit {Counter(y_train_US)}')


# In[41]:


Visualizing(X_train_US, y_train_US)


# In[42]:


PCA_KNN(X_train_US, y_train_US, X_test, y_test, n_neighbors=10)


# ## Oversampling (from scratch)

# In[43]:


df_class_1_OS = df_class_1.sample(int(count_class_0*.75), replace=True)
df_OS = pd.concat([df_class_0, df_class_1_OS], axis=0)

print('Random over-sampling:')
print(df_OS.Class.value_counts())
X_train_OS = df_OS.drop(columns=['Class'])
y_train_OS = df_OS.Class


# In[44]:


# os = RandomOverSampler(sampling_strategy=0.75)
# X_train_OS, y_train_OS = os.fit_resample(X_train, y_train)

print("The number of classes before fit {}".format(Counter(y_train)))
print("The number of classes after fit {}".format(Counter(y_train_OS)))


# In[45]:


Visualizing(X_train_OS, y_train_OS)


# In[46]:


PCA_KNN(X_train_OS, y_train_OS, X_test, y_test)


# ## Self-written customized functions for showing the results of classification techniques

# In[47]:


def plot_confusion_matrix(conf_mat):
    '''
    Input: numpy array of confusion matrix
    
    Displays the confustion matrix heatmap of the input confusion matrix
    '''
    group_names = ['True Negatives','False Postives','False Negatives','True Positives']
    group_counts = [f'{value}' for value in conf_mat.flatten()]
    group_percentages = [f'{round(value, 5)}%' for value in conf_mat.flatten()/np.sum(conf_mat)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(conf_mat, annot=labels, fmt='', cmap='Blues')


# In[48]:


def return_classification_results(model, X_train, y_train, X_test, y_test):
    '''
    Input: classifier model, train datasets and labels, test datasets and labels
    
    Displays the classification results of the model trained on the input training input on the test input
    '''
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f'accuracy score: {model.score(X_test, y_test)}')
    print(f'\nf1 score: {f1_score(y_test, y_pred)}')
    
    print('\nConfusion Matrix:-')
    fig = plt.figure(figsize=(5, 3))
    conf_mat = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(conf_mat)
    plt.show()
    
    print('\nClassification Report:-')
    print(classification_report(y_test, y_pred))
    


# ## Linear Discriminant Analysis

# In[49]:


def lda_testing(X_train, y_train, X_test, y_test, n_components=None):
    '''
    Input: training data and labels, test data and labels,
           number of components (default value = None)
    
    Trains a linear discriminant analysis model on the training inputs
    and returns the results of the classification performed on the testing inputs
    '''
    model = LinearDiscriminantAnalysis(n_components=n_components)
    return_classification_results(model, X_train, y_train, X_test, y_test)


# ## Random Forest Classifier

# In[50]:


def RandomForest_testing(X_train, y_train, X_test, y_test, n_estimators=100):
    '''
    Input: training data and labels, test data and labels, number of estimators
    
    Trains a random forest classifier model on the training inputs
    and returns the results of the classification performed on the testing inputs
    '''
    model = RandomForestClassifier(n_estimators=n_estimators)
    return_classification_results(model, X_train, y_train, X_test, y_test)


# ## Bagging Classifier

# In[51]:


DTC = DecisionTreeClassifier()
RFC = RandomForestClassifier()

def bagging_testing(X_train, y_train, X_test, y_test, n_estimators=100, estimator=DTC):
    '''
    Input: training data and labels, test data and labels, number of estimators
    
    Trains a random bagging classifier model on the training inputs
    and returns the results of the classification performed on the testing inputs
    '''
    model = BaggingClassifier(base_estimator=estimator, n_estimators=n_estimators)
    return_classification_results(model, X_train, y_train, X_test, y_test)


# ## XGBoost Classifier

# In[52]:


def xgboost_testing(X_train, y_train, X_test, y_test, n_estimators=100):
    '''
    Input: training data and labels, test data and labels, number of estimators
    
    Trains a xgboost classifier model on the training inputs
    and returns the results of the classification performed on the testing inputs
    '''
    model = XGBClassifier(n_estimators=n_estimators)
    return_classification_results(model, X_train, y_train, X_test, y_test)


# ## Self-implemented Ensemble Methods (from scratch)

# In[53]:


def return_randomforest(train_X, train_y, n_estimators=100):
    '''
    Input: training data and label, number of estimators
    
    Returns a random forest classifier model trained on the input data
    '''
    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(train_X, train_y)
    
    return model


# In[54]:


def voting_classifier(test_X, test_y, estimators):
    '''
    Input: testing data and labels,
           estimator models to be used in the voting classifier
           
    Displays the classification results of the voting classifier on the test data,
    the voting classifier returns the most occuring prediction out of each of the estimator models
    on each of the data entries of the test input
    '''
    y_preds = list()
    for each in range(len(estimators)):
        y_pred = estimators[each].predict(test_X)
        y_preds.append(y_pred)
        
    y_preds = np.array(y_preds)
    
    voting_results = list()
    
    for each in range(len(y_preds.T)):
        counts = np.bincount(y_preds.T[each])
        voting_results.append(np.argmax(counts))
    
    print(f'accuracy score: {accuracy_score(test_y, voting_results)}')
    print(f'\nf1 score: {f1_score(test_y, voting_results)}')
    
    print('\nConfusion Matrix:-')
    fig = plt.figure(figsize=(5, 3))
    conf_mat = confusion_matrix(test_y, voting_results)
    plot_confusion_matrix(conf_mat)
    plt.show()
    
    print('\nClassification Report:-')
    print(classification_report(y_test, y_pred))
    


# In[55]:


def fraud_dataset(df_train, y_train):
    '''
    Input: train data with labels, train labels
    
    Returns train dataframe containing only fraud data entries (Class=1)
    '''
    df_fraud = df_train[y_train==1].copy()
    return df_fraud


# In[56]:


def not_fraud_dataset(df_train, y_train):
    '''
    Input: train data with labels, train labels
    
    Returns train dataframe containing only non-fraud data entries (Class=0)
    '''
    df_notfraud = df_train[y_train==0].copy()
    df_notfraud = df_notfraud.sample(frac=1)
    return df_notfraud


# In[57]:


def Ensemble_Classifier_1(num, df_fraud, df_notfraud, X_test, y_test, features, class_label):
    '''
    Input: number of splits, fraud data points, non-fraud data points,
           test dataset and labels, array containing feature names, label of the class column in the dataframe
           
    Displays the result of our first ensemble approach,
    the approach deals with splitting the non-fraud datapoints from the training set into desired number of splits
    and concatenating each of the split with all the fraud datapoints of the training set and then using each split for
    training random forest classifiers which are further used as estimators in the previously implemented voting classifier
    '''
    dim = df_notfraud.shape[0]

    dataframes = list()
    start = 0
    Dim = dim//num

    for each in range(num):
        if each == num-1:
            dataframes.append(df_notfraud.iloc[start:dim].copy())
        else:
            dataframes.append(df_notfraud.iloc[start:start+Dim].copy())

        df_temp = dataframes[each].copy()
        df_temp = df_temp.append(df_fraud, ignore_index=True)
        df_temp = df_temp.sample(frac=1)
        dataframes[each] = df_temp

        start += Dim
        
    estimators_voting = list()

    for each in range(num):
        temp = dataframes[each].copy()
        temp_X = temp[features]
        temp_y = temp['Class']

        model = return_randomforest(temp_X, temp_y, n_estimators=10)
        estimators_voting.append(model)
        
    voting_classifier(X_test, y_test, estimators=estimators_voting)


# In[58]:


def Ensemble_Classifier_2(num, df_fraud, df_notfraud, X_test, y_test, features, class_label):
    '''
    Displays the result of our second ensemble approach,
    the approach deals with splitting the non-fraud datapoints and fraud datapoints from the training set into 
    desired number of splits and concatenating each of the split with one of the fraud datapoint split of the 
    training set and then using each of the split for training random forest classifiers which are further used as 
    estimators in the previously implemented voting classifier
    '''
    fraud_dim = df_fraud.shape[0]
    notfraud_dim = df_notfraud.shape[0]

    start = 0
    fraudDim = fraud_dim//num

    fraud_dataframes = []

    for each in range(num):
        if each == num-1:
            fraud_dataframes.append(df_fraud.iloc[start:fraud_dim].copy())
        else:
            fraud_dataframes.append(df_fraud.iloc[start:start+fraudDim].copy())

        start += fraudDim


    start = 0
    notFraudDim = notfraud_dim//num

    dataframes = []

    for each in range(num):
        if each == num-1:
            dataframes.append(df_notfraud.iloc[start:notfraud_dim].copy())
        else:
            dataframes.append(df_notfraud.iloc[start:start+notFraudDim].copy())

        df_temp = dataframes[each].copy()
        df_temp = df_temp.append(fraud_dataframes[each], ignore_index=True)
        df_temp = df_temp.sample(frac=1)
        dataframes[each] = df_temp

        start += notFraudDim
        
    estimators_voting = []

    for each in range(num):
        temp = dataframes[each].copy()
        temp_X = temp[features]
        temp_y = temp['Class']

        model = return_randomforest(temp_X, temp_y, n_estimators=10)

        estimators_voting.append(model)
        
    voting_classifier(X_test, y_test, estimators=estimators_voting)


# # Classifier Models testing on the dataset and the ttransformed datasets

# ## LDA

# #### for standardized dataset

# In[59]:


lda_testing(X_train, y_train, X_test, y_test)


# #### for undersampled dataset

# In[60]:


lda_testing(X_train_US, y_train_US, X_test, y_test)


# #### for oversampled dataset

# In[61]:


lda_testing(X_train_OS, y_train_OS, X_test, y_test)


# ---

# ## Random Forest

# #### for standardized dataset

# In[62]:


RandomForest_testing(X_train, y_train, X_test, y_test, n_estimators=25)


# #### for undersampled dataset

# In[63]:


RandomForest_testing(X_train_US, y_train_US, X_test, y_test, n_estimators=25)


# #### for oversampled dataset

# In[64]:


RandomForest_testing(X_train_OS, y_train_OS, X_test, y_test, n_estimators=25)


# ---

# ## Bagging

# #### for standardized dataset

# In[65]:


bagging_testing(X_train, y_train, X_test, y_test, n_estimators=10)


# #### for undersampled dataset

# In[66]:


bagging_testing(X_train_US, y_train_US, X_test, y_test, n_estimators=10)


# #### for oversampled dataset

# In[67]:


bagging_testing(X_train_OS, y_train_OS, X_test, y_test, n_estimators=10)


# ---

# ## XGBoost

# #### for standardized dataset

# In[68]:


xgboost_testing(X_train, y_train, X_test, y_test, n_estimators=10)


# #### for undersampled dataset

# In[69]:


xgboost_testing(X_train_US, y_train_US, X_test, y_test, n_estimators=10)


# #### for oversampled dataset

# In[70]:


xgboost_testing(X_train_OS, y_train_OS, X_test, y_test, n_estimators=10)


# ---

# ## Ensemble Methods

# #### for standardized dataset

# In[71]:


df_fraud = fraud_dataset(df_train, y_train)
df_notfraud = not_fraud_dataset(df_train, y_train)

Ensemble_Classifier_1(5, df_fraud, df_notfraud, X_test, y_test, features, 'Class')
print()
print('-'*100)
print()
Ensemble_Classifier_2(5, df_fraud, df_notfraud, X_test, y_test, features, 'Class')


# #### for undersampled dataset

# In[72]:


df_train_US = X_train_US.copy()
df_train_US['Class'] = y_train_US

df_fraud_US = fraud_dataset(df_train_US, y_train_US)
df_notfraud_US = not_fraud_dataset(df_train_US, y_train_US)

Ensemble_Classifier_1(5, df_fraud_US, df_notfraud_US, X_test, y_test, features, 'Class')
print()
print('-'*100)
print()
Ensemble_Classifier_2(5, df_fraud_US, df_notfraud_US, X_test, y_test, features, 'Class')


# #### for oversampled dataset

# In[73]:


df_train_OS = X_train_OS.copy()
df_train_OS['Class'] = y_train_OS

df_fraud_OS = fraud_dataset(df_train_OS, y_train_OS)
df_notfraud_OS = not_fraud_dataset(df_train_OS, y_train_OS)

Ensemble_Classifier_1(5, df_fraud_OS, df_notfraud_OS, X_test, y_test, features, 'Class')
print()
print('-'*100)
print()
Ensemble_Classifier_2(5, df_fraud_OS, df_notfraud_OS, X_test, y_test, features, 'Class')


# ## Overall pipeline

# In[74]:


all_features = df_raw.columns.tolist()[:-1]
# all_features


# In[75]:


final_scaler = StandardScaler()
final_scaler.fit(np.array(df_raw['Amount']).reshape(-1,1))


# In[76]:


final_model = RandomForestClassifier(n_estimators=25)
final_model.fit(X, y)


# In[77]:


def Pipeline(testset, final_model=final_model, final_scaler=final_scaler):
    '''
    Input: test dataset, final model
    
    Returns numpy array of predictions by the finally trained model
    '''
    testset = pd.DataFrame(testset).copy()
    testset.columns = all_features
    testset = testset.drop(columns=['Time'], axis=1)
    testset['Amount'] = final_scaler.transform(np.array(testset['Amount']).reshape(-1,1))
    
    predictions = np.array(final_model.predict(testset))
    
    return predictions
    


# In[ ]:




