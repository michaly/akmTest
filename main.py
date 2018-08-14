import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib osx



from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression



np.random.seed(2) # For the purpose of reproducing the results detailed in the report
# Printing config:
pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.expand_frame_repr', False) # for printing full objects


def dataAnalysis(data_df, labelColName):
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

    # Analyse (potential) categorical features:
    # TODO: get col names by dtype = 'object' and not hard coded.
    nonNumericCols = ['user_agent', 'country', 'city', 'county', 'timezone']

    for c in nonNumericCols:
        print('\nUnique value counts of column : ' + c)
        print(data_df[c].value_counts(normalize = True))

    return True

def preprocData(data_df, labelColName):
    #import seaborn as sns # For plotting the heatmap of correlation between the columns
    import re
    import random
    from pandas.plotting import scatter_matrix

    # Convert categorical features to dummy features (we don't treat 'path' as categorical, it is better to find some
    # sort of topic from the path and use it as features:
    # TODO: extract keywords / topics from 'path'

    categoricalCols = ['country', 'city', 'county']

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
    print("Preprocessed data shape after feature reduction - " + str(pp_data_df.shape))

    # plot the heatmap after feature reduction:
    #corr_df = pp_data_df.corr()
    #sns.heatmap(corr_df, xticklabels = corr_df.columns, yticklabels = corr_df.columns)

    # Check multicollinearity by the eigenvalues of the correlation matrix
    #eigval, eigvec = np.linalg.eig(corr_df)
    #small_eigval_idx = np.where(eigval < 0.0001)[0]
    #for i in small_eigval_idx:
    #    print("Eigenvector of eigenvalue %.15f :\n%s\n" % (eigval[i], eigvec[:, i]))
    """ There are still 2 very close eigenvalues to zero. The corresponding eigenvectors can detail the dependency.
     Also, we still observe high correlation (corr_df), so it is not surprising. However these correlations seems like 
     good predictors and in contrary to 'user_agent' (perfect predictor), for example, which is most likely due to 
     data collection bias. 
     We better use additional feature reduction before applying sensitive models. """

    # Scatter plot of highly correlated features with 'target' that could be great predictors:
    corr_cols = [labelColName, 'time_total', 'respose', 'fuel', 'contacts_n']
    sample_idx = random.sample(range(0, pp_data_df.shape[0] - 1), 1000)
    scatter_matrix(pp_data_df.loc[sample_idx, corr_cols])
    """ It seems like 'total_ime' has the greatest separation with respect to 'target'. """

    return pp_data_df

def featureScaling(data_df, fe_to_scale_list):
    #from sklearn.preprocessing import StandardScaler
    from sklearn import preprocessing
    import seaborn as sns  # For plotting the heatmap of correlation between the columns
    # fe_to_scale_list = ['fuel', 'source_port', 'contacts_n', 'bandwidth', 'respose', 'download_time', 'time_total']
    s_data_df = data_df.copy(deep = True)
    for f in fe_to_scale_list:
        plt.hist(data_df.loc[:,f])

    s_data_df['contacts_n'] = s_data_df['contacts_n'].transform(lambda x: x + np.ones(len(x)))
    #s_data_df[fe_to_scale_list] = s_data_df[fe_to_scale_list].transform(lambda x: np.log(x + np.ones(len(x))))
    s_data_df[fe_to_scale_list] = s_data_df[fe_to_scale_list].transform(np.log)
    # Plot the feature distribution to analyse if log-transform is required:
    min_max_scaler = preprocessing.MinMaxScaler()
    s_data_df[fe_to_scale_list] = min_max_scaler.fit_transform(s_data_df[fe_to_scale_list])

    f, axes = plt.subplots(1, 7, figsize=(20, 3), sharex=True)
    color_list = ['skyblue', 'olive', 'red', 'gold', 'teal', 'pink', 'green']
    for i in range(0, len(fe_to_scale_list)):
        sns.distplot(s_data_df[fe_to_scale_list[i]], color = color_list[i], ax=axes[i])
    #sns.distplot(df["sepal_width"], color="olive", ax=axes[0, 1])
    #sns.distplot(df["petal_length"], color="gold", ax=axes[1, 0])
    #sns.distplot(df["petal_width"], color="teal", ax=axes[1, 1])

    scaler = StandardScaler().fit(X)
    rescaledX = scaler.transform(X)
    # summarize transformed data
    numpy.set_printoptions(precision=3)
    print(rescaledX[0:5, :])

    return s_data_df

def splitTrainTest(data_df, labelColName):

    y = data_df[labelColName]
    X = data_df.drop(columns = [labelColName, 'path'])#, inplace = True)  # remove 'path' till we extract data from it
    print("dims : data_df - " +str(data_df.shape) + " ; X - " + str(X.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 2)
    print("Dimentions: \nX_train - " + str(X_train.shape) + " ; X_test - " + str(X_test.shape) +
          "\ny_train - " + str(y_train.shape) + "    ; y_test - " + str(y_test.shape) +
          "\n# of target = 1: train - " + str(np.sum(y_train)) + " ; test - " + str(np.sum(y_test)) +
          "\n% of target = 1: train - " + str(100 * np.sum(y_train) / len(y_train)) + " ; test - " + str(100 * np.sum(y_test) / len(y_test)))

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

data = pd.read_csv("~/PycharmProjects/akamai/data.csv")

labelCol = 'target'

dataAnalysis(data, labelCol)
ppData = preprocData(data, labelCol)
(X_train, X_test, y_train, y_test) = splitTrainTest(ppData, labelCol)





model = LogisticRegression()
results = cross_val_score(model, X_train, y_train)#, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)") % (results.mean()*100.0, results.std()*100.0)


