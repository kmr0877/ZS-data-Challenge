from pandas import Series
import pandas as pd
import pylab as pl
import numpy as np
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks 
from sklearn import svm
from sklearn.metrics import roc_auc_score
from imblearn.combine import SMOTETomek


df = pd.read_csv("train.csv")

# NOrmalize
for idx in range(1, 27):
	var = "var{}".format(idx)
	norm = (df[var] - df[var].mean()) / (df[var].max() - df[var].min())
	df[var] = norm

y = df['Disease_Flag'].as_matrix()
del(df['Disease_Flag'])
del(df['Resident ID'])

X = df.as_matrix()

# Make a copy for final training
X_original = X.copy()
y_original = y.copy()


# Run experiments on train/test split
from sklearn.model_selection import train_test_split
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33)

#from imblearn.ensemble import BalancedBaggingClassifier
clf = LinearClassifier(base_estimator=GradientBoostingClassifier(),
								ratio='auto',
                                 replacement=False,
                                 random_state=1)

clf.fit(X_res, y_res)


train_prob = clf.predict_proba(X_res)
prob_y_2 = [p[1] for p in train_prob]
print(clf.score(X_res, y_res))
print(roc_auc_score(y_res, prob_y_2))


test_prob = clf.predict_proba(X_test)
prob_y_2 = [p[1] for p in test_prob]
print(clf.score(X_test, y_test))
print(roc_auc_score(y_test, prob_y_2))



# Run the final model on the complete training set
clf.fit(X_original, y_original)

test = pd.read_csv("test.csv")
for idx in range(1, 27):
	var = "var{}".format(idx)
	norm = (test[var] - test[var].mean()) / (test[var].max() - test[var].min())
	test[var] = norm


del(test['Resident ID'])
yy = test.as_matrix()

prob = clf.predict_proba(yy)

f = open('run.csv', 'w')
for i,pp in enumerate(prob):
	f.write("Row{},{}\n".format(i+1, pp[1]))
f.close()
