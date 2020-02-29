import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# download the dataset
# !wget -O loan_train.csv https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/loan_train.csv


df = pd.read_csv('loan_train.csv')
# print(df.keys())

df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
# print(df.head(-2))
# print(df.keys())
# print(df['loan_status'].value_counts())

#--------------------Data visualization and pre-processing

# bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
# g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
# g.map(plt.hist, 'Principal', bins=bins, ec="k")
# g.axes[-1].legend()
# plt.show()

# bins = np.linspace(df.age.min(), df.age.max(), 10)
# g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
# g.map(plt.hist, 'age', bins=bins, ec="k")
# g.axes[-1].legend()
# plt.show()

#---------------Pre-processing: Feature selection/extraction

df['dayofweek'] = df['effective_date'].dt.dayofweek
# bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
# g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
# g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
# g.axes[-1].legend()
# plt.show()

# We see that people who get the loan at the end of the week dont pay it off,
# so lets use Feature binarization to set a threshold values less then day 4
df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
# print(df.head())


# ---------------------------Convert Categorical features to numerical values
# By gender
# print(df.groupby(['Gender'])['loan_status'].value_counts(normalize=True))

# Lets convert male to 0 and female to 1:
df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
# print(df.head())

# by education
# print(df.groupby(['education'])['loan_status'].value_counts(normalize=True))

# conver categorical varables to binary variables and append them to the feature Data Frame
# print(df[['Principal','terms','age','Gender','education']].head())
Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
# print(Feature.head())


# ----------------------     Feature selection

X = Feature
y = df['loan_status'].values

# ----------------------      Normalize Data
# Data Standardization give data zero mean and unit variance (technically should be done after train test split )

# X= preprocessing.StandardScaler().fit(X).transform(X)
# print(X)


 # --------------      KNN      --------------
X_train_knn, X_test_knn, y_train_knn, y_test_knn = \
    train_test_split( X, y, test_size=0.2, random_state=4)
# print ('Train set:', X_train_knn.shape,  y_train_knn.shape)
# print ('Test set:', X_test_knn.shape,  y_test_knn.shape)

# k = 4
# #Train Model and Predict
# neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train_knn,y_train_knn)
# # print(neigh)
# yhat = neigh.predict(X_test_knn)
# # print(yhat[0:5])
# # print("Train set Accuracy: ", metrics.accuracy_score(y_train_knn, neigh.predict(X_train_knn)))
# # print("Test set Accuracy: ", metrics.accuracy_score(y_test_knn, yhat))
# # k=4
# # Train set Accuracy:  0.8152173913043478
# # Test set Accuracy:  0.6857142857142857

# find the best K
Ks = 10
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))
ConfustionMx = [];
for n in range(1, Ks):
    # Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(X_train_knn, y_train_knn)
    yhat = neigh.predict(X_test_knn)
    mean_acc[n - 1] = metrics.accuracy_score(y_test_knn, yhat)

    std_acc[n - 1] = np.std(yhat == y_test_knn) / np.sqrt(yhat.shape[0])

# print(mean_acc)

# # plot model accuracy fot different K
# plt.plot(range(1,Ks),mean_acc,'g')
# plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
# plt.legend(('Accuracy ', '+/- 3xstd'))
# plt.ylabel('Accuracy ')
# plt.xlabel('Number of Nabors (K)')
# plt.tight_layout()
# plt.show()
# print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

#  -----------    decision tree    -----------


X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X, y, test_size=0.3, random_state=7)
# print(X_train_dt.shape)
# print(y_train_dt.shape)
# print(X_test_dt.shape)
# print(y_test_dt.shape)

#  an instance of the DecisionTreeClassifier called load_tree.
load_tree = DecisionTreeClassifier(criterion="entropy", max_depth = 2)
# print(load_tree) # it shows the default parameters

# fit the data with the training feature matrix and training response vector
load_tree.fit(X_train_dt,y_train_dt)

predTree = load_tree.predict(X_test_dt)
# print (predTree [0:5])
# print (y_test_dt [0:5])

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test_dt, predTree))


