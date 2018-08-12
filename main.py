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


def dataAnalysis(data_df, labelCol):
    print("data head :\n" + str(data_df.head()))
    print("data tail :\n" + str(data_df.tail()))
    print("data shape : " + str(data_df.shape))  # (145818, 17)

    #labelCol = 'target'

    print(data_df.info())  # Checking cols data type and existence of missing values (by comparing the # of values in a col to the df # of rows).
    # country = US :      47616
    # non-missing county: 47616
    # Meaning - there are no actual missing values under 'county'

    # Basic statistics:
    print("% of target (label) = 1: " + str(100 * np.sum(data_df[labelCol]) / data_df.shape[0]))
    print(data_df.describe(include = [np.number]))  # Print summary of numeric features
    print(data_df.describe(include = ['O']))  # Print summary of non-numeric features

    # data.dtypes#.index
    # data.select_dtypes(['float64','int64'])
    # TODO: get col names by dtype = 'object' and not hard coded.
    nonNumericCols = ['user_agent', 'country', 'city', 'county', 'timezone'] # , 'path']

    for c in nonNumericCols:
        print('\ncolumn : ' + c)
        print(data_df[c].value_counts(normalize = True))

    return True

def preprocData(data_df, labelColName):
    import seaborn as sns
    import re

    # Convert categorical features to dummy features (we don't treat 'path' as categorical, it is better to find some sort of topic from the path and use it as features:
    # TODO: extract keywords / topics from 'path'
    #categoricalCols = nonNumericCols[:]  # Copy list by value
    #categoricalCols.remove('path')
    categoricalCols = ['country', 'city', 'county']#, 'timezone', 'user_agent']

    pp_data_df = pd.get_dummies(data = data_df, columns = categoricalCols, drop_first = True)

    # Correlation analysis:
    abs_corr_df = pp_data_df.corr().abs() # correlation matrix

    # the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1):
    ordered_abs_corr = (abs_corr_df.where(np.triu(np.ones(abs_corr_df.shape), k = 1).astype(np.bool)).stack().sort_values(ascending = False))

    # Find redundant features:
    pure_dep = ordered_abs_corr[ordered_abs_corr == 1] # series object with 2 indices (2 features)
    # DEBUG: print("abs(correlation) == 1 :\n%s" % pure_dep)
    pure_dep_index_1 = pure_dep.index.get_level_values(0)
    pure_dep_index_2 = pure_dep.index.get_level_values(1)
    # Define the redundant features list:
    if(any(re.search(labelColName, i) != None for i in pure_dep_index_1) == False): # If 'target' is not in pure_dep first index
        removeCols = list(pure_dep_index_1)
    elif(any(re.search(labelColName, i) != None for i in pure_dep_index_2) == False):  # If 'target' is not in pure_dep second index
        removeCols = list(pure_dep_index_2)
    else: # 'target' is in both indices - not supported for now.
        print("%s is in both pure_dep indices. This scenario is not yet supported - exiting.")
        return False

    removeCols += ['port'] # std = 0, one value - all (-1)
    removeCols += ['user_agent'] # perfect predictor
    removeCols += ['latitude', 'longitude', 'timezone'] # high correlations
    # Remove rare features (<= 10 values):
    count_col_values = pp_data_df.astype(bool).sum(axis = 0)
    removeCols += list(count_col_values[count_col_values <= 10].index)
    removeCols = set(removeCols) # Remove duplicates


    print(("Delete %d out of %d columns.") % (len(removeCols), pp_data_df.shape[1]))
    pp_data_df.drop(columns = removeCols, inplace = True)


    #pp_data_df.drop(columns = ['respose', 'time_total'], inplace = True)

    # plot the heatmap after some feature reduction:
    corr_df = pp_data_df.corr()
    corr_df.where(np.triu(np.ones(corr_df.shape), k = 1).astype(np.bool)).stack().sort_values(ascending = False)

    sns.heatmap(corr_df, xticklabels = corr_df.columns, yticklabels = corr_df.columns)

    # Check multicollinearity by the eigenvalues of the correlation matrix
    #eigval, eigvec = np.linalg.eig(corr_df)
    # There are still 2 very close eigenvalues to zero. The corresponding eigenvectors can detail the dependency
    # We better use additional feature reduction before applying sensitive models.


    return pp_data_df

def splitTrainTest(data_df, labelColName):

    y = data_df[labelColName]
    X = data_df.drop(columns = [labelColName, 'path'])#, inplace = True)  # remove 'path' till we extract data from it
    print("dims : data_df - " +str(data_df.shape) + " ; X - " + str(X.shape))
    #X.columns
    #X.corr()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 2)
    print("Dimentions: \nX_train - " + str(X_train.shape) + " ; X_test - " + str(X_test.shape) +
          "\ny_train - " + str(y_train.shape) + "    ; y_test - " + str(y_test.shape) +
          "\n# of target = 1: train - " + str(np.sum(y_train)) + " ; test - " + str(np.sum(y_test)) +
          "\n% of target = 1: train - " + str(100 * np.sum(y_train) / len(y_train)) + " ; test - " + str(100 * np.sum(y_test) / len(y_test)))

    return (X_train, X_test, y_train, y_test)

data = pd.read_csv("~/PycharmProjects/akamai/data.csv")

labelCol = 'target'

dataAnalysis(data, labelCol)
ppData = preprocData(data)
(X_train, X_test, y_train, y_test) = splitTrainTest(ppData, labelCol)


########################################################################################
########################################################################################

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


########################################################################################
########################################################################################



# Convert categorical features to dummy features (we don't treat 'path' as categorical, it is better to find some sort of topic from the path and use it as features:
# TODO: extract keywords / topics from 'path'
categoricalCols = nonNumericCols[:]  # Copy list by value
categoricalCols.remove('path')

X = pd.get_dummies(data = data, columns = categoricalCols)


########################################################################################
########################################################################################

y = X[labelCol]
X.drop(columns = [labelCol, 'path'], inplace = True) # remove 'path' till we extract data from it
X.shape
X.columns
X.corr()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 2)
print("Dimentions: \nX_train - " + str(X_train.shape) + " ; X_test - " + str(X_test.shape) +
      "\ny_train - " + str(y_train.shape) + "    ; y_test - " + str(y_test.shape) +
      "\n# of target = 1: train - " + str(np.sum(y_train)) + " ; test - " + str(np.sum(y_test)) +
      "\n% of target = 1: train - " + str(100*np.sum(y_train)/len(y_train)) + " ; test - " + str(100*np.sum(y_test)/len(y_test)))

#scatter_matrix(X_train)
#plt.show()


model = LogisticRegression()
results = cross_val_score(model, X_train, y_train)#, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


