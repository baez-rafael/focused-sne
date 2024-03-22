import math
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

'''
In this file we have all the preprocessing methods
1- Intrinsic dimensionality
2- define feature and target
3- identify numeric and categorical features
'''

"""
Python Implementation of 'Maximum Likelihood Estimation of Intrinsic Dimension' Elizaveta Levina and Peter J. Bickel, Advances in neural information processing systems, 2005
----------
The goal is to estimate intrinsic dimensionality of data, the estimation of dimensionality is scale dependent
(depending on how much you zoom into the data distribution you can find different dimesionality), so they
propose to average it over different scales, the interval of the scales [k1, k2] are the only parameters of the algorithm.
This code also provides a way to repeat the estimation with bootstrapping to estimate uncertainty.
"""

def load_data(data, filetype):
    """
    loads data from CSV file into dataframe.
    Parameters
    ----------
    data : String
        path to input dataset
    filetype : String
        type of file
    Returns
    ----------
    df_Data : dataframe
        dataframe with data
    """

    if filetype == "csv":
        df_Data = pd.read_csv(data)
    else:
        df_Data = pd.read_excel(data)
    df_Data = df_Data.sample(1000)
    df_Data.reset_index(inplace = True)
    return df_Data

def intrinsic_dim_sample_wise(X, k=5):
    """
    Computes intrinsic dimensionality based on sample.
    
    Parameters
    ----------
    X : dataframe, dataset
    k : integer, number of neighbors
    
    Returns
    ----------
    intdim_sample : integer, intrinsic dimensionality based on sample
    """
    neighb = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dist, ind = neighb.kneighbors(X)
    dist = dist[:, 1:]
    dist = dist[:, 0:k]
    assert dist.shape == (X.shape[0], k)
    assert np.all(dist > 0)
    d = np.log(dist[:, k - 1: k] / dist[:, 0:k-1])
    d = d.sum(axis=1) / (k - 2)
    d = 1. / d
    intdim_sample = d
    return intdim_sample

def intrinsic_dim_scale_interval(X, k1=10, k2=20):
    """
    Computes intrinsic dimensionality for a given scale interval.
    
    Parameters
    ----------
    X : dataframe, dataset
    k1 : integer, number of neighbors (start of range)
    k2 : integer, number of neighbors (end of range)
    
    Returns
    ----------
    intdim_k : integer, intrinsic dimensionality for a given scale interval
    """

    X = pd.DataFrame(X).drop_duplicates().values # remove duplicates in case you use bootstrapping
    intdim_k = []
    for k in range(k1, k2 + 1):
        m = intrinsic_dim_sample_wise(X, k).mean()
        intdim_k.append(m)
    return intdim_k

def repeated(func, X, nb_iter=100, random_state=None, verbose=0, mode='bootstrap', **func_kw):
    """
    Computes intrinsic dimensionality in an iterative way.
    
    Parameters
    ----------
    X : dataframe, dataset
    
    Returns
    ----------
    int_dim : integer, intrinsic dimensionality
    """

    if random_state is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_state)
    nb_examples = X.shape[0]
    results = []

    iters = range(nb_iter)
    if verbose > 0:
        iters = tqdm(iters)    
    for i in iters:
        if mode == 'bootstrap':
            Xr = X[rng.randint(0, nb_examples, size=nb_examples)]
        elif mode == 'shuffle':
            ind = np.arange(nb_examples)
            rng.shuffle(ind)
            Xr = X[ind]
        elif mode == 'same':
            Xr = X
        else:
            raise ValueError('unknown mode : {}'.format(mode))
        results.append(func(Xr, **func_kw))

    intdim_k_repeated = np.array(results)

    histogram = np.histogram(intdim_k_repeated.mean(axis=1), bins=12)
    int_dim_original = histogram[1][np.argsort(histogram[0])[len(histogram[0])-1]]

    ## Identify the intrinsic dimensionality from the flaot value (i.e. ceiling for more than 0.5, and floor otherwise)

    if int_dim_original - math.floor(int_dim_original) < 0.5:
        int_dim = math.floor(int_dim_original)
    else:
        int_dim = math.ceil(int_dim_original)

    return int_dim


def set_features_and_target(df_Data):
    """
    Sets the features and targets from the dataframes.
    Parameters
    ----------
    df_Data : dataframe
        input dataset
    Returns
    ----------
    data_features : nD array
        matrix with data features
    data_target : list
        list of original labels
    """
    list_Data = list(df_Data)
    len_list = len(list_Data)
    features = list_Data[0:len(list_Data)-1]
    target = []
    target.append(list_Data[len(list_Data)-1])
    print("Features: ", features)
    print("Target: ", target)
    return features, target


def identify_and_transform_features(df, model_features):
    """
    Distinguigh the numeric and categorical variables and separate them.
    Parameters
    ----------
    df : dataframe, input dataset
    model_features: array, name of all features in the dataset

    Returns
    ----------
    X_transformed: nD array, tranformed data
    categorical_features: list, indexes of categorical features
    numeric_features: list, indexes of numeric features
    categorical_names: list, names of categorical features
    """

    ## Identify the categorical features

    likely_cat = {}
    categorical_features = []
    numeric_features = []
    for var in model_features:
        # likely_cat[var] = 1.*df[var].nunique()/df[var].count() < 0.2 #or some other threshold
        # if 1.*df[var].nunique()/df[var].count() < 0.2:
        #     categorical_features.append(df.columns.get_loc(var))
        # else:
            numeric_features.append(df.columns.get_loc(var))
                   
    ## Set up categorical names for the Explainer

    categorical_names = {}
    for feature in categorical_features:
        le = LabelEncoder()
        le.fit(df.values[:, feature])
        df.values[:, feature] = le.transform(df.values[:, feature])
        categorical_names[feature] = le.classes_
        
    ## Prepare the numeric and categorical features for NearestNeighbors

    numeric = []
    cat = []
    for feature in categorical_features:
        cat.append(df.columns[feature])
        
    for feature in numeric_features:
        numeric.append(df.columns[feature])

    ## transforming the numeric and categorical attributes:
    ## numerical attributes: quantile transformation
    ## categorical attributes: One-hot-encoding

    numeric_transformer = Pipeline(steps=[('scaler', QuantileTransformer())])
    cat_transformer = Pipeline(steps=[('onehot', OneHotEncoder())])

    prep = ColumnTransformer(transformers=[('num', numeric_transformer, numeric), 
                                           ('cat', cat_transformer, cat)], sparse_threshold=0)

    ## Transformed data, where the numeric and categorical features have this encoding done.
    X_df = pd.DataFrame(df, columns=model_features)
    X_transformed = prep.fit_transform(X_df)

    return X_transformed, categorical_features, numeric_features, categorical_names
