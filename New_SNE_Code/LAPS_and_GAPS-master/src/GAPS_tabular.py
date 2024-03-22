"""
Functions for explaining classifiers that use tabular data (matrices).
"""
import collections
import copy
from functools import partial
import json
import warnings

import numpy as np
import sklearn
import sklearn.preprocessing
from sklearn.utils import check_random_state
from sklearn.neighbors import NearestNeighbors

from .discretize import QuartileDiscretizer, DecileDiscretizer, EntropyDiscretizer, BaseDiscretizer
from . import GAPS_Explanation

def is_notebook():
  try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
      return True   # Jupyter notebook or qtconsole
    elif shell == 'TerminalInteractiveShell':
      return False  # Terminal running IPython
    else:
      return False  # Other type (?)
  except NameError:
    return False
if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class GapsExplainer(object):
    """Explains predictions on tabular (i.e. matrix) data.
    For numerical features, perturb them by sampling from a Normal(0,1) and
    doing the inverse operation of mean-centering and scaling, according to the
    means and stds in the training data. For categorical features, perturb by
    sampling according to the training distribution, and making a binary
    feature that is 1 when the value is the same as the instance being
    explained."""

    def __init__(self,
                 training_data,
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 class_names=None,
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None):
        
        """Init function.

        Parameters:
            training_data: numpy 2d array
            training_labels: labels for training data. Not required, but may be
                used by discretizer.
            feature_names: list of names (strings) corresponding to the columns
                in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            categorical_names: map from int to list of names, where
                categorical_names[x][y] represents the name of the yth value of
                column x.
            class_names: list of class names, ordered according to whatever the
                classifier is using. If not present, class names will be '0',
                '1', ...
            discretize_continuous: if True, all non-categorical features will
                be discretized into quartiles.
            discretizer: only matters if discretize_continuous is True. Options
                are 'quartile', 'decile', 'entropy' or a BaseDiscretizer
                instance.
            sample_around_instance: if True, will sample continuous features
                in perturbed samples from a normal centered at the instance
                being explained. Otherwise, the normal is centered on the mean
                of the feature data.
            random_state: an integer or numpy.RandomState that will be used to
                generate random numbers. If None, the random state will be
                initialized using the internal numpy seed.
        """
        self.random_state = check_random_state(random_state)
        self.categorical_names = categorical_names or {}
        self.sample_around_instance = sample_around_instance
        
        if categorical_features is None:
            categorical_features = []
        if feature_names is None:
            feature_names = [str(i) for i in range(training_data.shape[1])]

        self.categorical_features = list(categorical_features)
        self.feature_names = list(feature_names)

        self.discretizer = None
        if discretize_continuous and not sp.sparse.issparse(training_data):
            # Set the discretizer if training data stats are provided
            if self.training_data_stats:
                discretizer = StatsDiscretizer(training_data, self.categorical_features,
                                               self.feature_names, labels=training_labels,
                                               data_stats=self.training_data_stats)

            if discretizer == 'quartile':
                self.discretizer = QuartileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif discretizer == 'decile':
                self.discretizer = DecileDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif discretizer == 'entropy':
                self.discretizer = EntropyDiscretizer(
                        training_data, self.categorical_features,
                        self.feature_names, labels=training_labels)
            elif isinstance(discretizer, BaseDiscretizer):
                self.discretizer = discretizer
            else:
                raise ValueError('''Discretizer must be 'quartile',''' +
                                 ''' 'decile', 'entropy' or a''' +
                                 ''' BaseDiscretizer instance''')
            self.categorical_features = list(range(training_data.shape[1]))

            # Get the discretized_training_data when the stats are not provided
            if(self.training_data_stats is None):
                discretized_training_data = self.discretizer.discretize(
                    training_data)
        

        self.scaler = None
        self.class_names = class_names
        self.scaler = sklearn.preprocessing.StandardScaler(with_mean=False)
        self.scaler.fit(training_data)
        self.feature_values = {}
        self.feature_frequencies = {}

        for feature in self.categorical_features:
            column = training_data[:, feature]

            feature_count = collections.Counter(column)
            values, frequencies = map(list, zip(*(feature_count.items())))

            self.feature_values[feature] = values
            self.feature_frequencies[feature] = (np.array(frequencies) /
                                                 float(sum(frequencies)))
            self.scaler.mean_[feature] = 0
            self.scaler.scale_[feature] = 1
            

    @staticmethod
    def convert_and_round(values):
        return ['%.2f' % v for v in values]

            
    def generate_perturbed_neighborhood_global(self,
                         training_data,
                         transformed_data,
                         embedding,
                         data_row_indexes,
                         model_features,
                         categorical_features,
                         numeric_features,
                         nbrs=5,
                         num_features=5,
                         num_samples=100, verbose = False):
        """Generates generates perturbed neighborhoods for multiple points
        by randomly perturbing features from the instance (see __data_inverse). 

        Parameters:
            training_data: numpy 2d array, original dataset
            transformed_data: numpy 2d array, transformed original data
            embedding: numpy 2d array, embedding obtained after executing DR on training data
            data_row_indexes: 1d numpy array, indexes corresponding to a set of rows from the original dataset
            model_features: list of names (strings) corresponding to the columns in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            numeric_features: list of indices (ints) corresponding to the
                numeric columns.
            nbrs: integer, number of neighbors for knn search 
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            

        Returns:
            -------------------
        """

        local_feature_contributions = np.zeros((len(data_row_indexes),training_data.shape[1]))
        local_feature_contributions_embd = np.zeros((len(data_row_indexes),training_data.shape[1]))
        neighbors_local = np.zeros((len(data_row_indexes),nbrs))
        neighbors_embd_local = np.zeros((len(data_row_indexes),nbrs))

        neighbors = []
        neighbors_embd = []
        #print(len(data_row_indexes))
        #num_samples = 1000/(len(data_row_indexes)*nbrs)
        #num_samples = 100/(len(data_row_indexes)*nbrs)
        print(f'num_samples = {num_samples}')
        #num_samples = num_samples
        knn = NearestNeighbors(leaf_size=30, n_neighbors=nbrs, p=2, radius=1.0, algorithm='ball_tree')
        knn.fit(transformed_data)
        for index in range (0,len(data_row_indexes)):
            globals() ["neighbors_"+str(index)] = knn.kneighbors(transformed_data, return_distance=False)[data_row_indexes[index]]
            globals() ["data_"+str(index)] = globals() ["data_embd_"+str(index)] = globals() ["inverse_"+str(index)] = globals() ["inverse_embd_"+str(index)] = np.array([training_data[data_row_indexes[index]]])
        knn.fit(embedding)
        for index in range (0,len(data_row_indexes)):
            globals() ["neighbors_embd_"+str(index)] = knn.kneighbors(embedding, return_distance=False)[data_row_indexes[index]]
        for index in range (0,len(data_row_indexes)):

            #if verbose:
            #  print(globals() ["neighbors_"+str(index)])
            for neighbor in globals() ["neighbors_"+str(index)]:
                neighbors.append(neighbor)
        #if verbose:    
          #print("......................................")
        
        for index in range (0,len(data_row_indexes)):
            #if verbose:
            #  print(globals() ["neighbors_embd_"+str(index)])
            for neighbor in globals() ["neighbors_embd_"+str(index)]:
                neighbors_embd.append(neighbor)
        if verbose:
          bar = tqdm(range(0, len(data_row_indexes)))
        else:
          bar = range(0, len(data_row_indexes))
        for index in bar:
            for globals() ["point_"+str(index)], globals() ["point_embd_"+str(index)] in zip(globals() ["neighbors_"+str(index)], globals() ["neighbors_embd_"+str(index)]):
                #if verbose:
                #  print("Neighbors for data-point", globals() ["point_"+str(index)], globals() ["point_embd_"+str(index)])

                globals() ["temp_data_"+str(index)], globals() ["temp_inverse_"+str(index)] = self.__data_inverse(training_data[globals() ["point_"+str(index)]], num_samples)
                globals() ["temp_embd_data_"+str(index)],  globals() ["temp_inverse_embd_"+str(index)] = self.__data_inverse(training_data[globals() ["point_embd_"+str(index)]], num_samples)
                #if verbose:
                #  print(globals() ["data_"+str(index)].shape)
                #  print(globals() ["temp_data_"+str(index)].shape)
                globals() ["data_"+str(index)] = np.concatenate((globals() ["data_"+str(index)],globals() ["temp_data_"+str(index)]), axis=0)
                globals() ["data_embd_"+str(index)] = np.concatenate((globals() ["data_embd_"+str(index)],globals() ["temp_embd_data_"+str(index)]), axis=0)
                

                globals() ["inverse_"+str(index)] = np.concatenate((globals() ["inverse_"+str(index)],globals() ["temp_inverse_"+str(index)]), axis=0)
                globals() ["inverse_embd_"+str(index)] = np.concatenate((globals() ["inverse_embd_"+str(index)],globals() ["temp_inverse_embd_"+str(index)]), axis=0)
                       
                #if globals() ["point_"+str(index)] == globals() ["neighbors_"+str(index)][0] and globals()["point_embd_"+str(index)] == globals() ["neighbors_embd_"+str(index)][0]:
                #    continue
                #if globals() ["point_"+str(index)] != globals() ["neighbors_"+str(index)][0]:
                #    globals() ["data_"+str(index)] = np.concatenate((globals() ["data_"+str(index)],np.array([training_data[globals() ["point_"+str(index)]]])), axis=0)
                #    globals() ["inverse_"+str(index)] = np.concatenate((globals() ["inverse_"+str(index)],np.array([training_data[globals() ["point_"+str(index)]]])), axis=0)
                #if globals() ["point_embd_"+str(index)] != globals() ["neighbors_embd_"+str(index)][0]:
                #    globals() ["data_embd_"+str(index)] = np.concatenate((globals() ["data_embd_"+str(index)],np.array([training_data[globals() ["point_embd_"+str(index)]]])), axis=0)
                #    globals() ["inverse_embd_"+str(index)] = np.concatenate((globals() ["inverse_embd_"+str(index)],np.array([training_data[globals() ["point_embd_"+str(index)]]])), axis=0)

                globals() ["scaled_data_"+str(index)] = (globals()["data_"+str(index)] - self.scaler.mean_) / self.scaler.scale_
                globals() ["scaled_data_embd_"+str(index)] = (globals()["data_embd_"+str(index)] - self.scaler.mean_) / self.scaler.scale_


            '''
            Run Feature Contributions for each point inside the outer for loop
            '''

            corr_feat_dist, _, _, _, _ = GAPS_Explanation.explain_point_local(training_data[data_row_indexes[index]], globals() ["neighbors_"+str(index)], globals() ["scaled_data_"+str(index)], model_features, categorical_features, numeric_features, budget=10)
            corr_feat_dist_embd, _, _, _, _ = GAPS_Explanation.explain_point_local(training_data[data_row_indexes[index]], globals() ["neighbors_embd_"+str(index)], globals() ["scaled_data_embd_"+str(index)], model_features, categorical_features, numeric_features, budget=10)
            neighbors_local[index] = globals() ["neighbors_"+str(index)]
            neighbors_embd_local[index] = globals() ["neighbors_embd_"+str(index)]
            local_feature_contributions[index] = corr_feat_dist
            local_feature_contributions_embd[index] = corr_feat_dist_embd

        scaled_data = globals() ["scaled_data_"+str(0)]
        scaled_data_embd = globals() ["scaled_data_embd_"+str(0)]

        for index in range (1,len(data_row_indexes)):
            scaled_data = np.concatenate((scaled_data,globals() ["scaled_data_"+str(index)]), axis=0)
            scaled_data_embd = np.concatenate((scaled_data_embd,globals() ["scaled_data_embd_"+str(index)]), axis=0)


        # print("until now it all works!")
        
        return neighbors, neighbors_embd, scaled_data, scaled_data_embd, local_feature_contributions, local_feature_contributions_embd, neighbors_local, neighbors_embd_local
        
    def explain_instance_local(self,
                         training_data,
                         transformed_data,
                         embedding,
                         data_row_indexes,
                         model_features,
                         categorical_features,
                         numeric_features,
                         nbrs=5,
                         num_features=5,
                         num_samples=1000):
        """Runs LAPS on each point in the selected subset.

        Args:
            training_data: numpy 2d array, original dataset
            transformed_data: numpy 2d array, transformed original data
            embedding: numpy 2d array, embedding obtained after executing DR on training data
            data_row_indexes: 1d numpy array, indexes corresponding to a set of rows from the original dataset
            model_features: list of names (strings) corresponding to the columns in the training data.
            categorical_features: list of indices (ints) corresponding to the
                categorical columns. Everything else will be considered
                continuous. Values in these columns MUST be integers.
            numeric_features: list of indices (ints) corresponding to the
                numeric columns.
            nbrs: integer, number of neighbors for knn search 
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            -------------------
        """
        num_samples = int(5000/nbrs)
        
        knn = NearestNeighbors(leaf_size=30, n_neighbors=nbrs, p=2, radius=1.0, algorithm='ball_tree')
        knn.fit(transformed_data)
        neighbors = knn.kneighbors(transformed_data, return_distance=False)[data_row_index]
        
        knn.fit(embedding)
        neighbors_embd = knn.kneighbors(embedding, return_distance=False)[data_row_index]
        
        data = data_embd = np.array([data_row])
        inverse = inverse_embd = np.array([data_row])
        
        for point, point_embd in zip(neighbors, neighbors_embd):
            print(point, point_embd)
            temp_data, temp_inverse = self.__data_inverse(training_data[point], num_samples)
            temp_embd_data, temp_embd_inverse = self.__data_inverse(training_data[point_embd], num_samples)
            data = np.concatenate((data,temp_data), axis=0)
            data_embd = np.concatenate((data_embd,temp_embd_data), axis=0)
            inverse = np.concatenate((inverse,temp_inverse), axis=0)
            inverse_embd = np.concatenate((inverse_embd,temp_embd_inverse), axis=0)
            if point == neighbors[0] and point_embd == neighbors_embd[0]:
                continue
            if point != neighbors[0]:
                data = np.concatenate((data,np.array([training_data[point]])), axis=0)
                inverse = np.concatenate((inverse,np.array([training_data[point]])), axis=0)
            if point_embd != neighbors_embd[0]:
                data_embd = np.concatenate((data_embd,np.array([training_data[point_embd]])), axis=0)
                inverse_embd = np.concatenate((inverse_embd,np.array([training_data[point_embd]])), axis=0)
        
        '''
        Calculating distance with Euclidean distance for all instances right now
        '''
        
        scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        
        distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
        ).ravel()
        
        scaled_data_embd = (data_embd - self.scaler.mean_) / self.scaler.scale_
        
        distances_embd = sklearn.metrics.pairwise_distances(
                scaled_data_embd,
                scaled_data_embd[0].reshape(1, -1),
                metric=distance_metric
        ).ravel()
        
        
        print(inverse.shape)
        print(inverse_embd.shape)
        print("until now it all works!")
        
        return neighbors, neighbors_embd, scaled_data, scaled_data_embd



    def __data_inverse(self,
                       data_row,
                       num_samples):
        """Generates a neighborhood around a point.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.

        Parameters:
            data_row: 1d numpy array, corresponding to a row
            num_samples: size of the neighborhood to learn the linear model

        Returns:
            A tuple (data, inverse), where:
                data: dense num_samples * K matrix, where categorical features
                are encoded with either 0 (not equal to the corresponding value
                in data_row) or 1. The first row is the original instance.
                inverse: same as data, except the categorical features are not
                binary, but categorical (as the original data)
        """
        data = np.zeros((int(num_samples), data_row.shape[0]))
        
        categorical_features = range(data_row.shape[0])
        if self.discretizer is None:
            data = self.random_state.normal(
                    0, 1, int(num_samples) * data_row.shape[0]).reshape(
                    int(num_samples), data_row.shape[0])
            if self.sample_around_instance:
                data = data * self.scaler.scale_ + data_row
            else:
                data = data * self.scaler.scale_ + self.scaler.mean_
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=int(num_samples),
                                                      replace=True, p=freqs)
            binary_column = np.array([1 if x == first_row[column]
                                      else 0 for x in inverse_column])
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse

