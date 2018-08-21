import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib osx

# Printing config:
pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.expand_frame_repr', False) # for printing full objects

def colNameListByDType(df, numericCols=True):
    # # # # # # # # # #
    # Finds the names of numeric/non-numeric columns of a dataframe
    # Args:
    #       df - (pandas dataframe)
    #       numericCols - (bool), True - for numerical columns, False - for non-numerical columns
    # Return:
    #       col_name_list - (list of strings), the matched columns name
    # # # # # # # # # #
    from pandas.api.types import is_numeric_dtype

    col_name_list = list()
    for col in df.columns:
        if(numericCols): # if the numeric columns are required
            if(is_numeric_dtype(df[col]) == True):
                col_name_list += [col]
        else:   # the non-numeric columns are required
            if (is_numeric_dtype(df[col]) == False):
                col_name_list += [col]

    # apply doesn't work with is_numeric_dtype for some reason!
    #if(numericCols):
    #    col_name_list = df.columns[df.apply(lambda x: is_numeric_dtype(x))]
    #else:
    #    col_name_list = df.columns[~np.array(df.apply(is_numeric_dtype))]

    return col_name_list

def dataAnalysis(data_df, labelColName):
    # # # # # # # # # #
    # Analyse the dataset
    # Args:
    #       data_df - (pandas dataframe), the dataset
    #       labelColName - (string), the column name that represents the label (0/1)
    # Return:
    #       True (for tracking failures)
    # # # # # # # # # #
    import random
    from pandas.plotting import scatter_matrix
    from scipy.stats import normaltest

    print("Data head :\n" + str(data_df.head()))
    print("Data shape : " + str(data_df.shape))  # (145818, 17)

    print(data_df.info())  # Checking cols data type and existence of missing values (by comparing the # of values in a col to the df # of rows).
    """ country = US :      47616
        non-missing county: 47616
        Meaning - there are no actual missing values under 'county' """

    # Basic statistics:
    print("% of target (label) = 1: " + str(100 * np.sum(data_df[labelColName]) / data_df.shape[0]))
    print(data_df.describe(include=[np.number]))  # Print summary of numeric features
    print(data_df.describe(include=['O']))  # Print summary of non-numeric features

    non_numeric_cols = colNameListByDType(data_df, numericCols=False)

    for col in non_numeric_cols:
        print('\nUnique value counts of column : ' + col)
        print(data_df[col].value_counts(normalize=True))

    # Check if the numeric columns in the dataset are coming from normal dist
    # (will help us decide how to scale the features later on):
    numeric_cols = colNameListByDType(data_df, numericCols=True)
    alpha = 1e-3
    for i in numeric_cols:
        k2, p = normaltest(data_df[i])  # null hypothesis: the feature comes from a normal distribution
        if (p < alpha):  # The null hypothesis can be rejected
            print("It is most likely that %s is not coming from a normal distribution, pval = %.4f" % (i, p))
        else:  # The null hypothesis cannot be rejected
            print("We cannot reject the hypothesis that %s is coming from a normal distribution, pval = %.4f" % (i, p))

    # Scatter plot of numeric features:
    numeric_cols = list(set(['port']).symmetric_difference(numeric_cols)) # 'port' has zero variance, ignore it in scatter plot
    sample_idx = random.sample(range(0, data_df.shape[0] - 1), 500)
    scatter_matrix(data_df.loc[sample_idx, numeric_cols])
    """ It seems like 'total_ime' has the greatest separation with respect to 'target'. """

    return True

def colNamesToFilter(df, method, min_val):
    # # # # # # # # # #
    # Finds out which columns should be filtered (according to the specified method and min_val).
    # Args:
    #       df - (pandas dataframe), the dataset
    #       method - (string), possible values: 'freq' - frequency
    #                                           'std' - standard deviation
    #       min_val - (float), columns with method value lower or equal to this argument will be filtered
    # Return:
    #       f_cols - (list of strings), list of column names to filter
    # # # # # # # # # #
    f_cols = []
    if(method == 'freq'):
        cols_method = df.astype(bool).sum(axis=0)
    elif(method == 'std'):
        cols_method = df.std(axis=0)
    else:
        print("colNamesToFilter warning: method is empty, no filter will be applied.")
    f_cols += list(cols_method[cols_method <= min_val].index)
    return f_cols

def extractFeaturesFromPath(path_sr):
    # # # # # # # # # #
    # Extract new features from the elements constructing a 'path' (column)
    # Args:
    #       path_sr - (pandas series), the 'path' column (a column of strings with '/' delimiter)
    # Return:
    #       dummies_df - (pandas dataframe), the new extracted features (original indices for future merge)
    # # # # # # # # # #

    # path_sr = data['path'].copy(deep=True)
    delimiter = "/"
    if(all(path_sr.str.startswith(delimiter))): # If all path values start with "/" - remove it. Otherwise, it may mean something, so don't.
        path_sr = path_sr.str[1:]
    path_sr[path_sr.str.endswith(delimiter)] = path_sr[path_sr.str.endswith(delimiter)].apply(lambda x: x + "empty_end") # Add 'empty_end' string to paths that end with "/", may be represent a directory.

    path_sr = path_sr.apply(lambda x: list(set(x.split(delimiter)))) # Split the path by "/"
    path_df = pd.DataFrame({'idx' : path_sr.index, 'path' : path_sr})

    dummies_df = pd.get_dummies(
        path_df.join(pd.Series(path_df['path'].apply(pd.Series).stack().reset_index(1, drop=True),
                          name='splittedPath')).drop('path', axis=1).rename(columns={'splittedPath': 'path'}),
        columns=['path']).groupby('idx', as_index=False).sum().drop(columns=['idx'])
    remove_cols = colNamesToFilter(dummies_df, method='freq', min_val=10.)
    dummies_df.drop(columns=remove_cols, inplace=True)
    print("Adding %d features extracted from 'path'." % dummies_df.shape[1])
    return dummies_df

def featureDistPlot(df):
    # # # # # # # # # #
    # Plot the pdf of the features in a dataset
    # Args:
    #       df - (pandas dataframe), the dataset, where the columns represent the features
    # Return:
    #       True (for tracking failures)
    # # # # # # # # # #
    import seaborn as sns

    f, axes = plt.subplots(1, df.shape[1], figsize=(20, 3), sharex=True)
    for i in range(0, df.shape[1]):
        sns.distplot(df[df.columns[i]], ax=axes[i])
    return True

def featureScaling(data_df, fe_to_scale_list):
    # # # # # # # # # #
    # Scale (MinMax) the specified columns of a dataset [0,1]
    # Note: we don't perform standardization because the continues features are not normally distributed.
    # Args:
    #       data_df - (pandas dataframe), the dataset, where the columns represent the features
    #       fe_to_scale_list - (list of strings), feature names to scale.
    # Return:
    #       s_data_df - (pandas dataframe), the scaled dataset
    # # # # # # # # # #
    from sklearn.preprocessing import MinMaxScaler
    # fe_to_scale_list = colNameListByDType(data_df.drop(columns=['target','port']), numericCols = True)
    s_data_df = data_df.copy(deep=True)

    # Plot the feature distribution to analyse if log-transform is required:
    min_max_scaler = MinMaxScaler()
    s_data_df[fe_to_scale_list] = min_max_scaler.fit_transform(s_data_df[fe_to_scale_list])
    # Plot the features (fe_to_scale_list) dist:
    featureDistPlot(s_data_df[fe_to_scale_list])
    """Log-transform won't help us in this case to approximate to normal distribution."""

    return s_data_df

def preprocData(data_df, labelColName):
    # # # # # # # # # #
    # Preprocess dataset - feature extraction, feature reduction and feature scaling
    # Note: we didn't clean outliers because the numerical features were not normally distributed, even after log transformation.
    #       Also, no missing data handling is done, because there are no such cases in our dataset.
    # Args:
    #       data_df - (pandas dataframe), the 'path' column (a column of strings with '/' delimiter)
    #       labelColName - (string), the column name that represents the label (0/1)
    # Return:
    #       pp_data_df - (pandas dataframe), the preprocessed dataset
    # # # # # # # # # #
    import re

    # data_df = data.copy(deep=True); labelColName = labelCol

    """ 
    # maybe we can do this step AFTER the feature reduction - 
    check the numeric features dist - if it is normal or can be transformed to normal. 
    otherwise it is not clear how to perform the outliers detection.
    # CLEAN OUTLIERS:
    numericalCols = ['fuel', 'source_port', 'contacts_n', 'latitude', 'longitude', 'bandwidth', 'respose', 'download_time', 'port', 'time_total']
    std_threshold = 4
    # TODO: if the numerical features are not normally distributed - is it correct to remove outliers by SD?
    """
    # FEATURE EXTRACTION:
    # Convert categorical features to dummy features:
    categorical_cols = list(set(['path']).symmetric_difference(colNameListByDType(data_df, numericCols=False)))
    pp_data_df = pd.get_dummies(data=data_df, columns=categorical_cols, drop_first=True)

    # Extract features (keywords) from 'path':
    path_fe_df = extractFeaturesFromPath(pp_data_df['path'])
    pp_data_df = pd.merge(pp_data_df, path_fe_df, left_index=True, right_index=True)


    # FEATURE REDUCTION:
    remove_cols = []
    # Correlation analysis:
    abs_corr_df = pp_data_df.corr().abs() # (absolute value) correlation matrix
    # the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1):
    ordered_abs_corr = (abs_corr_df.where(np.triu(np.ones(abs_corr_df.shape), k = 1).astype(np.bool)).stack().sort_values(ascending = False))

    # Find redundant features:
    corr_threshold = 0.9
    corr_fe = ordered_abs_corr[ordered_abs_corr >= corr_threshold] # series object with 2 indices (2 features)
    # print("DEBUG: abs(correlation) == 1 :\n%s" % corr_fe)
    corr_fe_index_1 = corr_fe.index.get_level_values(0)
    corr_fe_index_2 = corr_fe.index.get_level_values(1)
    # Define the redundant features list:
    if(any(re.search(labelColName, i) != None for i in corr_fe_index_1) == False): # If 'target' is not in corr_fe first index
        remove_cols += list(corr_fe_index_1)
    elif(any(re.search(labelColName, i) != None for i in corr_fe_index_2) == False):  # If 'target' is not in corr_fe second index
        remove_cols += list(corr_fe_index_2)
    else: # 'target' is in both indices - not supported for now.
        print("%s is in both corr_fe indices. This scenario is not yet supported - no high-correlation filter will be applied.")

    # Std filter:
    remove_cols += colNamesToFilter(pp_data_df, method='std', min_val=0.01)
    # Frequency filter (most likely these features will be filtered in the previous filter):
    remove_cols += colNamesToFilter(pp_data_df, method='freq', min_val=10.)
    remove_cols += ['path']
    remove_cols = set(remove_cols) # Remove duplicates

    print(("Delete %d out of %d columns.") % (len(remove_cols), pp_data_df.shape[1]))
    pp_data_df.drop(columns=remove_cols, inplace=True)
    print("Preprocessed data shape after feature reduction - " + str(pp_data_df.shape))

    # Multicollinearity filter (detected by existence of tiny eigenvalues of the correlation matrix)
    """ 
        Existence of tiny eigenvalues (very close to zero) implies multicollinearity. 
        The corresponding eigenvectors can detail the dependency of the features.
        """
    corr_df = pp_data_df.drop(columns=[labelColName]).corr()
    remove_cols = []
    eigval, eigvec = np.linalg.eig(corr_df)
    min_eigval = 0.001
    small_eigval_idx = np.where(eigval <= min_eigval)[0]
    for i in small_eigval_idx:
        #print("DEBUG: Eigenvector of eigenvalue %.15f :\n%s\n" % (eigval[i], eigvec[:, i]))
        remove_cols += list(corr_df.columns[np.where(np.abs(eigvec[:, i]) >= 0.01)[0]][:-1]) # Keep only one of the features and remove the rest
        print("DEBUG: multicollinearity (major) relation: %s" % list(corr_df.columns[np.where(np.abs(eigvec[:, i]) >= 0.1)[0]]))

    remove_cols = set(remove_cols) # Remove duplicates
    print(("Multicollinearity filter: delete %d out of %d columns.") % (len(remove_cols), pp_data_df.shape[1]))
    print("DEBUG: remove_cols = %s" % remove_cols)
    pp_data_df.drop(columns=remove_cols, inplace=True)
    print("Preprocessed data shape after feature reduction - " + str(pp_data_df.shape))

    # FEATURE SCALING:
    fe_to_scale_list = pp_data_df.columns[np.array(pp_data_df.apply(lambda x: x.min() < 0)) | np.array(pp_data_df.apply(lambda x: x.max() > 1))]
    pp_data_df = featureScaling(pp_data_df, fe_to_scale_list)

    return pp_data_df


def defineClassifiers():
    # # # # # # # # # #
    # Define classifiers - Decision Tree, Random Forest, SVM
    # Args:
    #       Nothing
    # Return:
    #       clf_list - (list of sklearn classifier objects BEFORE FITTING)
    # # # # # # # # # #
    from sklearn import tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC

    dt = tree.DecisionTreeClassifier(max_depth=1)  # , min_samples_leaf=100)
    rf = RandomForestClassifier(max_depth=2, random_state=0)  # , min_samples_leaf=100))
    svm = SVC(kernel='linear')

    clf_list = [dt, rf, svm]

    return clf_list

def evaluateClassifiers(X, y, clf_list):
    # # # # # # # # # #
    # Evaluate classifiers that were NOT FITTED - precision, recall, f1
    # Args:
    #       X - (pandas dataframe), the design matrix (without train/test split)
    #       y - (pandas series), the labels (without train/test split)
    #       clf_list - (list of sklearn classifier objects BEFORE FITTING)
    # Return:
    #       best_clf - (sklearn classifier object) best classifier - the one with max(E[F1_scores])
    # # # # # # # # # #
    from sklearn.model_selection import cross_validate, ShuffleSplit

    cv = ShuffleSplit(n_splits=5, test_size=0.4, random_state=0)
    scoring = ('precision', 'recall', 'f1')
    f1_scores = []
    for clf in clf_list:
        print("\nEvaluate classifier: %s" % type(clf))
        # for regular K-fold use c=k:
        scores = cross_validate(clf, X, y, cv=cv, scoring=scoring)
        f1_scores.append(scores["test_f1"].mean())
        for s in scoring:
            print("(class 1) %s: %f" % (s, scores["test_" + s].mean()))
    f1_scores = np.array(f1_scores)
    best_clf = clf_list[np.where(f1_scores == max(f1_scores))[0][0]]
    return best_clf

data = pd.read_csv("./data.csv")

label_col = 'target'

dataAnalysis(data, label_col)
ppData = preprocData(data, label_col)
clf_list = defineClassifiers()
X = ppData.drop(columns=label_col)
y = ppData[label_col]
clf = evaluateClassifiers(X, y, clf_list)

print("We recommend using %s classifier for detecting %s" % (clf, label_col))

