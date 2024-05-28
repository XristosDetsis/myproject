#Load the Iris dataset:
from sklearn import datasets

iris=datasets.load_iris()
X = iris.data
Y = iris.target
print(X.shape)
print(Y.shape)


#Split the dataset into training and testing sets:
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_test.shape ,X_train.shape ,Y_test.shape ,Y_train.shape)


#Train a RandomForestClassifier:
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train, Y_train)

print(clf.feature_importances_)
print(clf.predict(X_test))#προβλεπω τα test
print(Y_test)#κανονικες τιμες test
print(clf.score(X_test, Y_test))#ποσοστο επιτυχιας στα test

#Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
import numpy as np

max_features_range = np.arange(1,5,1)
n_estimators_range = np.arange(10,210,10)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

rf=RandomForestClassifier()

grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)

grid.fit(X_train, Y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))


#Analyze the results of GridSearchCV:
import pandas as pd

grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()
grid_reset = grid_contour.reset_index()
grid_pivot = grid_reset.pivot(index='max_features', columns='n_estimators', values='Accuracy')
grid_pivot = grid_reset.pivot_table(index='max_features', columns='n_estimators', values='Accuracy')
print(grid_pivot) # φτιαχνω πινακα