import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

## importing dataset
df = pd.read_csv('customer_validation.csv')

## classsifying features and labels
X = df.iloc[:,3:13].values
y = df.iloc[:,13].values

## label encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
encode_X_country = LabelEncoder()
encode_X_gender = LabelEncoder()
X[:,1] = encode_X_country.fit_transform(X[:,1])
X[:,2] = encode_X_gender.fit_transform(X[:,2])
onehotencode = OneHotEncoder(categorical_features=[1])
X = onehotencode.fit_transform(X).toarray()
X = X[:,1:12]

## train and test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

## feature scaling
from sklearn.preprocessing import StandardScaler
scale_X =  StandardScaler()
X_train = scale_X.fit_transform(X_train)
X_test = scale_X.transform(X_test)

## building ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from keras.layers import Dense
from keras.models import Sequential

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier)

## Optimizing the model
parameters = {'batch_size':[2,8,16,32], 'epochs':[100,250,500], 'optimizer':['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train) 

best_paramters = grid_search.best_params_
best_accuracy = grid_search.best_score_

accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)

best_paramters = grid_search.best_params_
best_accuracy = grid_search.best_score_























