from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

#Load the Iris dataset:
iris=datasets.load_iris()
X = iris.data
Y = iris.target
print("Διαστάσεις Πίνακα Χαρακτηριστικών =",X.shape)
print("Διαστάσεις Πίνακα Στόχων =",Y.shape)

#Split the dataset into training and testing sets:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
print(X_test.shape ,X_train.shape ,Y_test.shape ,Y_train.shape)


#Train a RandomForestClassifier:
clf = RandomForestClassifier()
clf.fit(X_train, Y_train)

K=iris.feature_names
L=clf.feature_importances_
print("Σημασία χαρακτηριστικών:")
for i in range(4):
    print(f"{K[i]} = {L[i]}")
print("Προβλέψεις της RandomForest για το δείγμα(X_test)=", clf.predict(X_test))
print("Kανονικές τιμές του δείγματος(Y_test)=", Y_test)
print("ποσοστο επιτυχιας στα test της RandomForest = ",clf.score(X_test, Y_test))


#Hyperparameter tuning using GridSearchCV
max_features_range = np.arange(1,5,1)
n_estimators_range = np.arange(10,210,10)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)


rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)

grid.fit(X_train, Y_train)

print("Οι καλύτερες παραμετροί είναι %s με επίδοση %0.3f"
      % (grid.best_params_, grid.best_score_))


#Analyze the results of GridSearchCV:
grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()
grid_reset = grid_contour.reset_index()
grid_pivot = grid_reset.pivot(index='max_features', columns='n_estimators', values='Accuracy')
grid_pivot = grid_reset.pivot_table(index='max_features', columns='n_estimators', values='Accuracy')
print("Πίνακας παραμέτρων:")
print(grid_pivot)
