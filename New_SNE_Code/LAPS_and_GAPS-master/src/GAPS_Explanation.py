import numpy as np
from .Distances import gower_distances
import math
from scipy.spatial import distance
from sklearn.manifold import Isomap
from sklearn.utils import validation
from sklearn.metrics import pairwise, pairwise_distances
from scipy.spatial.distance import pdist, wminkowski

unsquareform = lambda a: a[np.nonzero(np.triu(a, 1))]

def get_local_explanations_for_GAPS(X_df, ld_embedding, data_instance_numbers, local_feature_contributions):
    """
    Provides explanations on each point in the selected subset for local explanations.
    
    Parameters:
    -----------------
    X_df: nD array, original data
    ld_embedding: nD array, low dimensional embedding of the original data
    data_instance_numbers: array, subset of data points to be analyzed
    local_feature_contributions: dict, influence of features in the local neighborhoods
    
    Returns:
    -----------------    
    sorted_index_combinations: nD array, sorted pairs of indexes according to ascending order
                                of Geodesic distances in the original data
    sorted_index_combinations_embd: nD array, sorted pairs of indexes according to ascending order
                                of Geodesic distances in the embedding
    """

    int_dim = 2
    unsquareform = lambda a: a[np.nonzero(np.triu(a, 1))]

    temp_data = np.zeros((len(data_instance_numbers),local_feature_contributions.shape[1]))
    temp_data_embd = np.zeros((len(data_instance_numbers),int_dim))

    for index in range(0,len(data_instance_numbers)):
        temp_data[index] = X_df.values[index]
    for index in range(0,int_dim):
        temp_data_embd[index] = ld_embedding[index]

    iso_embd = Isomap(n_components=2, n_neighbors =1)

    X_trans = iso_embd.fit_transform(temp_data)        
    distance_matrix = iso_embd.dist_matrix_
    iso_embd.dist_matrix_[iso_embd.dist_matrix_ == 0] = -9999
    distances = unsquareform(iso_embd.dist_matrix_)

    embd_trans = iso_embd.fit_transform(temp_data_embd)        
    distance_matrix_embd = iso_embd.dist_matrix_
    iso_embd.dist_matrix_[iso_embd.dist_matrix_ == 0] = -9999
    distances_embd = unsquareform(iso_embd.dist_matrix_)

    index_combinations=[]
    for row in range(0,len(distance_matrix)):
        for col in range(row+1,len(distance_matrix[0])):
            index_combinations.append([row,col])

    sorted_indexes = np.argsort(distances)
    sorted_index_combinations = []

    sorted_indexes_embd = np.argsort(distances_embd)
    sorted_index_combinations_embd = []

    for index in sorted_indexes_embd:
        sorted_index_combinations_embd.append(index_combinations[index])

    for index in sorted_indexes:
        sorted_index_combinations.append(index_combinations[index])

    return sorted_index_combinations, sorted_index_combinations_embd

def explain_point_local(data_row, neighbors, oversampled_data, model_features, categorical_features, numeric_features, budget=999, show_pos_neg = False):
    """
    Provides explanations on each point in the selected subset for local explanations.
    
    Parameters:
    -----------------
    data_row: integer, the index of each individual data-point in the subset
    neighbors: array, list of nearest neighbors of the point
    oversampled_data:  nD array, approximated neighborhood of the point
    model_features: list, name of all features of the model
    categorical_features: list, name of categorical features
    numeric_features: name of all numeric features
    budget: integer, number of features to be displayed as output
    show_pos_neg: boolean, show both positive and negative influences of features
    
    Returns:
    -----------------
    corr_feat_dist: nD array, correlations between the features contribution and distances
    feature_dict: dict, dictionary of feature influences
    feature_distance_contribution: nD array, contribution of each features in the relative distances between points
    dvs_matrix: nD array, distance variance score matrix
    sorted_indexes: nD array, sorted indexes of data points according to proximity
    """

    distances = []
    #weights = []
    cat_features = []
    
    if len(categorical_features) != 0:
        for i in range (0, oversampled_data.shape[1]):
            cat_features.append(True) if i in categorical_features else cat_features.append(False)

   # for index in range (0,len(oversampled_data.T)):
    #    weights.append(1/float(len(oversampled_data.T)))
    for index in range(0, len(oversampled_data)):
        point_vectors = np.array([oversampled_data[0], oversampled_data[index]])
        if len(categorical_features) == 0:
            #distances.append(pdist(point_vectors,wminkowski,2, weights)[0])
            distances.append(pdist(point_vectors)[0])
        else:
            #distances.append(Distances.gower_distances(point_vectors, w=weights, categorical_features=cat_features)[0][1])
            distances.append(gower_distances(point_vectors, categorical_features=cat_features)[0][1])

    sorted_indexes = np.argsort(distances)
    sorted_distances = np.sort(distances)       
    
    ## Order neighbors based on sorted positions

    sorted_neighbors = np.empty((0,data_row.shape[0]), float)
    for index in sorted_indexes:
        sorted_neighbors = np.append(sorted_neighbors , np.array([oversampled_data[index,:]]), axis=0)
        
    ## calculate feature difference matrix
    
    feature_difference = np.zeros((sorted_neighbors.shape[0]-1, sorted_neighbors.shape[1]))
    
    for col_index in range(0,sorted_neighbors.shape[1]):
        if col_index in numeric_features:
            for row_index in range(0,sorted_neighbors.shape[0]-1):
                feature_difference[row_index][col_index] = (abs(sorted_neighbors[row_index+1][col_index] - sorted_neighbors[row_index][col_index])
                                                                    / (np.max(sorted_neighbors[:,col_index])-np.min(sorted_neighbors[:,col_index])))
        else:
            for row_index in range(0,sorted_neighbors.shape[0]-1):
                feature_difference[row_index][col_index] = 0 if sorted_neighbors[row_index+1][col_index] == sorted_neighbors[row_index][col_index] else 1
    
    ## calculate the feature contribution matrix and distance variance score matrix
    
    feature_distance_contribution = np.zeros((sorted_neighbors.shape[0]-1, sorted_neighbors.shape[1]))
    dvs_matrix = np.zeros((sorted_neighbors.shape[0]-1, sorted_neighbors.shape[1]))
    for col_index in range(0,sorted_neighbors.shape[1]):
        for row_index in range(0,sorted_neighbors.shape[0]-1):
            feature_distance_contribution[row_index][col_index] = feature_difference[row_index][col_index] / sorted_distances[row_index]
            dvs_matrix[row_index][col_index] = feature_distance_contribution[row_index][col_index] * np.var(sorted_neighbors[:,col_index])
    
    ## calculate feature covariance with distances -> Experiment a little bit with this
    feature_distance_contribution = np.nan_to_num(feature_distance_contribution)
    corr_feat_dist = np.zeros(feature_difference.shape[1])
    for col_index in range(0,feature_difference.shape[1]):
        corr_feat_dist[col_index] = np.corrcoef(feature_difference[:,col_index],sorted_distances[:-1])[0][1]
    
    if budget != 999 and show_pos_neg == False:
        sub_features = [np.argsort(abs(corr_feat_dist), axis=0)[::-1][x] for x in range(0,budget)]
        feature_dict= dict(zip([model_features[x] for x in sub_features], [corr_feat_dist[x] for x in sub_features]))
    elif budget != 999 and show_pos_neg == True:
        sub_features = [np.argsort(corr_feat_dist, axis=0)[::-1][x] for x in range(0,len(corr_feat_dist))]
        feature_dict= dict(zip([model_features[x] for x in sub_features[0:int(budget/2)]], [corr_feat_dist[x] for x in sub_features[0:int(budget/2)]]))
        feature_dict.update(zip([model_features[x] for x in sub_features[(len(sub_features)-int(budget/2)): len(sub_features)]], [corr_feat_dist[x] for x in sub_features[(len(sub_features)-int(budget/2)): len(sub_features)]]))
    else:
        feature_dict= dict(zip(model_features, corr_feat_dist))
    
    return corr_feat_dist, feature_dict, feature_distance_contribution, dvs_matrix, sorted_indexes

def compute_local_divergences(neighbors_local, neighbors_embd_local, local_feature_contributions, local_feature_contributions_embd):
    """
    Compute local Divergence.
    
    Parameters:
    -----------------
    neighbors_local: array, list of nearest neighbors of a single data point in the subset in the original data
    neighbors_embd_local: array, list of nearest neighbors of a single data point in the subset in the embedding
    local_feature_contributions: dict, feature influence contributions for a single point in the original data
    local_feature_contributions_embd: dict, feature influence contributions for a single point in the embedding

    Returns:
    -----------------
    local_divergences: array, list of local diveregnce scores for all points in the subset
    """

    disagreements_1 = []
    disagreements_2 = []
    disagreements_3 = []
    local_divergences =[]
    for corr_feat_dist, corr_feat_dist_embd in zip(local_feature_contributions, local_feature_contributions_embd):
        disagreement1 = pdist([corr_feat_dist, corr_feat_dist_embd], metric='euclidean')
        disagreements_1.append(1/(1+disagreement1[0]))

    for neighbors, neighbors_embd in zip(neighbors_local, neighbors_embd_local):    
        disagreement2 = len(neighbors)-len(np.intersect1d(neighbors, neighbors_embd))
        disagreements_2.append(float(disagreement2)/len(neighbors))

    for neighbors, neighbors_embd in zip(neighbors_local, neighbors_embd_local): 
        disagree = 0
        for index in range (0,len(neighbors)):
            if neighbors[index] != neighbors_embd[index]:
                disagree = disagree + 1
        disagreements_3.append(float(disagree)/len(neighbors))
        
    for index in range(0,len(disagreements_1)):
        local_divergences.append((1/float(3))*disagreements_1[index] + (1/float(3))*disagreements_2[index] + (1/float(3))*disagreements_3[index])

    return local_divergences
