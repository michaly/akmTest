import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib osx



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

# %reset # clear all variables

np.random.seed(2) # For the purpose of reproducing the results detailed in the report
# Printing config:
pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.expand_frame_repr', False) # for printing full objects

def colNameListByDType(df, numericCols = True):
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
    print(data_df.describe(include = [np.number]))  # Print summary of numeric features
    print(data_df.describe(include = ['O']))  # Print summary of non-numeric features

    non_numeric_cols = colNameListByDType(data_df, numericCols = False)

    for col in non_numeric_cols:
        print('\nUnique value counts of column : ' + col)
        print(data_df[col].value_counts(normalize = True))

    # Check if the numeric columns in the dataset are coming from normal dist
    # (will help us decide how to scale the features later on):
    numeric_cols = colNameListByDType(data_df, numericCols = True)
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
    """kw_set = set()
    for i in path_sr:
        #print(i)
        kw_set = set(list(kw_set) + list(i))
        #print("kw_set: " + str(kw_set))"""

    url_df = pd.DataFrame({'idx' : path_sr.index, 'path' : path_sr})

    dummies_df = pd.get_dummies(
        url_df.join(pd.Series(url_df['path'].apply(pd.Series).stack().reset_index(1, drop = True),
                          name = 'splitedPath')).drop('path', axis = 1).rename(columns = {'splitedPath': 'path'}),
        columns = ['path']).groupby('idx', as_index = False).sum().drop(columns = ['idx'])

    # Drop the features containing less than 10 occurrences:
    #count_col_values = dummies_df.astype(bool).sum(axis = 0)
    #remove_cols = list(count_col_values[count_col_values <= 10].index)
    #dummies_df.drop(columns = remove_cols, inplace = True)

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

    f, axes = plt.subplots(1, df.shape[1], figsize = (20, 3), sharex = True)
    for i in range(0, df.shape[1]):
        sns.distplot(df[df.columns[i]], ax = axes[i])
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
    s_data_df = data_df.copy(deep = True)

    # Plot the feature distribution to analyse if log-transform is required:
    min_max_scaler = MinMaxScaler()
    s_data_df[fe_to_scale_list] = min_max_scaler.fit_transform(s_data_df[fe_to_scale_list])
    # Plot the features (fe_to_scale_list) dist:
    featureDistPlot(s_data_df[fe_to_scale_list])
    """Log-transform won't help us in this case to approximate to normal distribution."""

    # Perform log-transform to approximate to normal dist
    # Scale to positive feature (min value = 1), if not already:
    # for i in fe_to_scale_list:
    #    min_val = s_data_df[i].min()
    #    if(min_val <= 0.):
    #        s_data_df[i] = s_data_df[i].apply(lambda x: x + (1 - min_val))
    # s_data_df[fe_to_scale_list] = s_data_df[fe_to_scale_list].transform(np.log)

    return s_data_df

def preprocData(data_df, labelColName):
    # # # # # # # # # #
    # Preprocess dataset - feature extraction and feature reduction
    # Note: we didn't clean outliers because the numerical features were not normally distributed, even after log transformation.
    #       Also, no missing data handling is done, because there are no such cases in our dataset.
    # Args:
    #       data_df - (pandas dataframe), the 'path' column (a column of strings with '/' delimiter)
    #       labelColName - (string), the column name that represents the label (0/1)
    # Return:
    #       pp_data_df - (pandas dataframe), the preprocessed dataset
    # # # # # # # # # #

    #import seaborn as sns # For plotting the heatmap of correlation between the columns
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
    categorical_cols = list(set(['path']).symmetric_difference(colNameListByDType(data_df, numericCols = False)))
    pp_data_df = pd.get_dummies(data = data_df, columns = categorical_cols, drop_first = True)

    # Extract features (keywords) from 'path':
    path_fe_df = extractFeaturesFromPath(pp_data_df['path'])
    pp_data_df = pd.merge(pp_data_df, path_fe_df, left_index = True, right_index = True)


    # FEATURE REDUCTION:
    # Correlation analysis:
    abs_corr_df = pp_data_df.corr().abs() # (absolute value) correlation matrix

    # the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1):
    ordered_abs_corr = (abs_corr_df.where(np.triu(np.ones(abs_corr_df.shape), k = 1).astype(np.bool)).stack().sort_values(ascending = False))

    # Find redundant features:
    corr_threshold = 0.9
    corr_fe = ordered_abs_corr[ordered_abs_corr >= corr_threshold] # series object with 2 indices (2 features)
    # DEBUG: print("abs(correlation) == 1 :\n%s" % corr_fe)
    corr_fe_index_1 = corr_fe.index.get_level_values(0)
    corr_fe_index_2 = corr_fe.index.get_level_values(1)
    # Define the redundant features list:
    if(any(re.search(labelColName, i) != None for i in corr_fe_index_1) == False): # If 'target' is not in corr_fe first index
        remove_cols = list(corr_fe_index_1)
    elif(any(re.search(labelColName, i) != None for i in corr_fe_index_2) == False):  # If 'target' is not in corr_fe second index
        remove_cols = list(corr_fe_index_2)
    else: # 'target' is in both indices - not supported for now.
        print("%s is in both corr_fe indices. This scenario is not yet supported - exiting.")
        return False

    #remove_cols += ['port'] # std = 0, one value - all (-1)
    #remove_cols += ['user_agent'] # perfect predictor
    #remove_cols += ['latitude', 'longitude', 'timezone'] # high correlations

    # Std filter:
    cols_std = pp_data_df.std(axis = 0)
    min_std = 0.01
    remove_cols += list(cols_std[cols_std <= min_std].index)
    # Frequency filter (most likely these features will be filtered in the previous filter):
    count_col_values = pp_data_df.astype(bool).sum(axis = 0)
    min_freq = 10
    remove_cols += list(count_col_values[count_col_values <= min_freq].index)
    remove_cols += ['path']
    remove_cols = set(remove_cols) # Remove duplicates


    print(("Delete %d out of %d columns.") % (len(remove_cols), pp_data_df.shape[1]))
    pp_data_df.drop(columns = remove_cols, inplace = True)
    print("Preprocessed data shape after feature reduction - " + str(pp_data_df.shape))

    # plot the heatmap after feature reduction:
    corr_df = pp_data_df.drop(columns = [labelColName]).corr()
    #sns.heatmap(corr_df, xticklabels = corr_df.columns, yticklabels = corr_df.columns)

    # Multicollinearity filter (detected by existence of tiny eigenvalues of the correlation matrix)
    remove_cols = []
    eigval, eigvec = np.linalg.eig(corr_df)
    min_eigval = 0.001
    small_eigval_idx = np.where(eigval <= min_eigval)[0]
    for i in small_eigval_idx:
        #print("DEBUG: Eigenvector of eigenvalue %.15f :\n%s\n" % (eigval[i], eigvec[:, i]))
        remove_cols += list(corr_df.columns[np.where(np.abs(eigvec[:, i]) >= 0.1)[0]][:-1]) # Keep only one of the features and remove the rest
        print("DEBUG: multicollinearity (major) relation: %s" % list(corr_df.columns[np.where(np.abs(eigvec[:, i]) >= 0.1)[0]]))
    """ There are still 2 very close eigenvalues to zero. The corresponding eigenvectors can detail the dependency.
     Also, we still observe high correlation (corr_df), so it is not surprising. However these correlations seems like 
     good predictors and in contrary to 'user_agent' (perfect predictor), for example, which is most likely due to 
     data collection bias. 
     We better use additional feature reduction before applying sensitive models. """

    remove_cols = set(remove_cols) # Remove duplicates
    print(("Multicollinearity filter: delete %d out of %d columns.") % (len(remove_cols), pp_data_df.shape[1]))
    print("DEBUG: remove_cols = %s" % remove_cols)
    pp_data_df.drop(columns = remove_cols, inplace = True)
    print("Preprocessed data shape after feature reduction - " + str(pp_data_df.shape))

    # FEATURE SCALING:
    fe_to_scale_list = pp_data_df.columns[np.array(pp_data_df.apply(lambda x: x.min() < 0)) | np.array(pp_data_df.apply(lambda x: x.max() > 1))]
    pp_data_df = featureScaling(pp_data_df, fe_to_scale_list)

    return pp_data_df

def splitTrainTest(data_df, labelColName):

    y = data_df[labelColName]
    X = data_df.drop(columns = [labelColName])#, inplace = True)
    print("dims : data_df - %s ; X - %s" % (data_df.shape, X.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 2)
    print("Dimentions: \nX_train - %s ; X_test - %s\ny_train - %s    ; y_test - %s" % (X_train.shape, X_test.shape, y_train.shape, y_test.shape))
    print("Target = 1 (freq):    train - %d  ; test - %d\nTarget = 1 (percent): train - %.3f ; test - %.3f"
          % (np.sum(y_train), np.sum(y_test), 100 * np.sum(y_train) / len(y_train), 100 * np.sum(y_test) / len(y_test)))

    return (X_train, X_test, y_train, y_test)

def trainClassifiers(X_train, y_train):

    # Decision Tree:
    from sklearn import tree
    import graphviz
    dt_clf = tree.DecisionTreeClassifier()
    dt_clf = dt_clf.fit(X_train, y_train)
    dot_data = tree.export_graphviz(dt_clf, out_file = None, feature_names = X_train.columns,
                         #class_names=iris.target_names,
                         filled = True, rounded = True,
                         special_characters = True)
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree_graph")

    # Random Forest:
    from sklearn.ensemble import RandomForestClassifier
    rf_clf = RandomForestClassifier(max_depth = 1, random_state = 0)
    print("Random Forest features importance: \n%s" % rf_clf.feature_importances_)

    # Naive Bayes:
    from sklearn.naive_bayes import GaussianNB
    gnb_clf = GaussianNB()

    clf_list = [rf_clf, gnb_clf]
    for clf in clf_list:
        clf.fit(X_train, y_train)

    return

def evaluateClassifiers(X_test, y_test, clf):
    from sklearn.metrics import confusion_matrix

    print("Evaluate classifier: %s" % type(clf))
    y_pred = clf.predict(X_test)
    print(confusion_matrix(y_test, y_pred))

    return True

data = pd.read_csv("./data.csv")

labelCol = 'target'

dataAnalysis(data, labelCol)
ppData = preprocData(data, labelCol)
(X_train, X_test, y_train, y_test) = splitTrainTest(ppData, labelCol)





model = LogisticRegression()
results = cross_val_score(model, X_train, y_train)#, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


