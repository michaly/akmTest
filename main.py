import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib osx
from pandas.plotting import scatter_matrix



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

np.random.seed(2)
# Printing config:
pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.expand_frame_repr', False) # for printing full objects


data = pd.read_csv("~/PycharmProjects/akamai/data.csv")
data.head()
data.tail()
print(data.shape) # (145818, 17)

labelCol = 'target'

data.info() # Checking cols data type and existence of missing values (by comparing the # of values in a col to the df # of rows).
# country = US :      47616
# non-missing county: 47616
# Meaning - there are no actual missing values under 'county'

# Basic statistics:
print("% of target (label) = 1: " + str(100 * np.sum(data[labelCol])/data.shape[0]))
print(data.describe(include = [np.number])) # Print summary of numeric features
print(data.describe(include = ['O'])) # Print summary of non-numeric features

#data.dtypes#.index
#data.select_dtypes(['float64','int64'])
# TODO: get col names by dtype = 'object' and not hard coded.
nonNumericCols = ['path', 'user_agent', 'country', 'city', 'county', 'timezone']

for c in nonNumericCols:
    print('\ncolumn : ' + c)
    print(data[c].value_counts(normalize = True))

# Convert categorical features to dummy features (we don't treat 'path' as categorical, it is better to find some sort of topic from the path and use it as features:
# TODO: extract keywords / topics from 'path'
categoricalCols = nonNumericCols[:]  # Copy list by value
categoricalCols.remove('path')

X = pd.get_dummies(data = data, columns = categoricalCols)
scatter_matrix(X)
plt.show()

y = X[labelCol]
X.drop(columns = [labelCol, 'path'], inplace = True) # remove 'path' till we extract data from it
X.shape
X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 2)
print("Dimentions: \nX_train - " + str(X_train.shape) + " ; X_test - " + str(X_test.shape) +
      "\ny_train - " + str(y_train.shape) + "    ; y_test - " + str(y_test.shape) +
      "\n# of target = 1: train - " + str(np.sum(y_train)) + " ; test - " + str(np.sum(y_test)) +
      "\n% of target = 1: train - " + str(100*np.sum(y_train)/len(y_train)) + " ; test - " + str(100*np.sum(y_test)/len(y_test)))




model = LogisticRegression()
results = cross_val_score(model, X_train, y_train)#, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


