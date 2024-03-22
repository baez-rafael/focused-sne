import numpy as np
from .Distances import gower_distances
import math
from scipy.spatial import distance
from sklearn.manifold import Isomap
from sklearn.utils import validation
from sklearn.metrics import pairwise, pairwise_distances
from scipy.spatial.distance import pdist, wminkowski

unsquareform = lambda a: a[np.nonzero(np.triu(a, 1))]

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


def explain_point_local(data_row, neighbors, oversampled_data, model_features, categorical_features, numeric_features, budget=999, show_pos_neg = False):
    """
    Provides explanations on the preserved local structure.
    
    Parameters:
    -----------------
    data_row: integer, the index of the data-point to be explained
    neighbors: array, list of nearest neighbors of the chosen row
    oversampled_data:  nD array, approximated neighborhood of the chosen point
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
    cat_features = []
    
    if len(categorical_features) != 0:
        for i in range (0, oversampled_data.shape[1]):
            cat_features.append(True) if i in categorical_features else cat_features.append(False)
    for index in range(0, len(oversampled_data)):
        point_vectors = np.array([oversampled_data[0], oversampled_data[index]])
        if len(categorical_features) == 0:
            distances.append(pdist(point_vectors)[0])
        else:
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
    
    ## calculate feature covariance with distances
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

def explain_point_global(oversampled_data, model_features, categorical_features, numeric_features, budget=999, show_pos_neg = False, verbose = False):
        
    """
    Provides explanations on the preserved global structure.
    
    Parameters:
    -----------------
    oversampled_data:  nD array, approximated neighborhood of the chosen point
    model_features: list, name of all features of the model
    categorical_features: list, name of categorical features
    numeric_features: name of all numeric features
    budget: integer, number of features to be displayed as output
    show_pos_neg: boolean, show both positive and negative influences of features
    
    Returns:
    -----------------
    corr_feat_dist: nD array, correlations between the features contribution and distances
    feature_dict: dict, dictionary of feature influences
    sorted_indexes: nD array, sorted indexes of data points according to proximity
    """

    index_combinations = []
    
    # distances = distance.pdist(oversampled_data,'minkowski', p=2)
    # distance_matrix = distance.squareform(distances)
    # distance_matrix[distance_matrix==0] = -9999

    embedding = Isomap(n_components=2)
    X_transformed = embedding.fit_transform(oversampled_data)
    distance_matrix = embedding.dist_matrix_
    embedding.dist_matrix_[embedding.dist_matrix_ == 0] = -9999
    distances = unsquareform(embedding.dist_matrix_)
    print('Step 1/4')
    if verbose:
      bar = tqdm(range(0, len(distance_matrix)))
    else:
      bar = range(0, len(distance_matrix))    
    for row in bar:
        
        for col in range(row+1,len(distance_matrix[0])):
            index_combinations.append([row,col])

    sorted_indexes = np.argsort(distances)
    sorted_distances = np.sort(distances)       
    
    #print(len(distances))
    ## Order index combinations based on sorted positions
    print('Step 2/4')
    sorted_index_combinations = []
    for index in sorted_indexes:
        sorted_index_combinations.append(index_combinations[index])
 
  ## calculate feature difference matrix
    print('Step 3/4')
    feature_difference = np.zeros((len(sorted_index_combinations), oversampled_data.shape[1]))
    if verbose:
      bar = tqdm(range(0, len(sorted_index_combinations)))
    else:
      bar = range(0, len(sorted_index_combinations))
    for row_index in bar:
        row1, row2 = sorted_index_combinations[row_index]
        for col_index in range (0,oversampled_data.shape[1]):
            if col_index in numeric_features:
                feature_difference[row_index][col_index] = abs(oversampled_data[row1][col_index] - oversampled_data[row2][col_index])
            else:
                feature_difference[row_index][col_index] = 0 if oversampled_data[row1][col_index] == oversampled_data[row2][col_index] else 1
 
    print("Step 4/4")
    
    corr_feat_dist = np.zeros(feature_difference.shape[1])
    for col_index in range(0,feature_difference[:-1].shape[1]):
        corr_feat_dist[col_index] = np.corrcoef(feature_difference[:,col_index],sorted_distances)[0][1]
        
    if budget != 999 and show_pos_neg == False:
        sub_features = [np.argsort(abs(corr_feat_dist), axis=0)[::-1][x] for x in range(0,budget)]
        feature_dict= dict(zip([model_features[x] for x in sub_features], [corr_feat_dist[x] for x in sub_features]))
    elif budget != 999 and show_pos_neg == True:
        sub_features = [np.argsort(corr_feat_dist, axis=0)[::-1][x] for x in range(0,len(corr_feat_dist))]
        feature_dict= dict(zip([model_features[x] for x in sub_features[0:(budget/2)]], [corr_feat_dist[x] for x in sub_features[0:(budget/2)]]))
        feature_dict.update(zip([model_features[x] for x in sub_features[(len(sub_features)-(budget/2)): len(sub_features)]], [corr_feat_dist[x] for x in sub_features[(len(sub_features)-(budget/2)): len(sub_features)]]))
    else:
        feature_dict= dict(zip(model_features, corr_feat_dist))
    
    return corr_feat_dist, feature_dict, sorted_index_combinations
    
def compute_local_divergence(corr_feat_dist, corr_feat_dist_embd, neighbors, neighbors_embd):

    """
    Compute LAPS Divergence.
    
    Parameters:
    -----------------
    corr_feat_dist: nD array, correlations between the original feature contribution and distances
    corr_feat_dist: nD array, correlations between the embedding feature contribution and distances
    neighbors: array, list of nearest neighbors of the chosen row in the original data
    neighbors_embd: array, list of nearest neighbors of the chosen row in the embedding
    
    Returns:
    -----------------
    components: String, individual components of teh local divergence
    divergence: float, final diveregnce score
    """

    components = ""

    disagreement_component1 = pdist([corr_feat_dist, corr_feat_dist_embd], metric='euclidean')
    disagreement_component1 = 1/(1+disagreement_component1[0])

    components = components + str(disagreement_component1)


    disagreement_component2 = len(neighbors)-len(np.intersect1d(neighbors, neighbors_embd))
    disagreement_component2 = disagreement_component2/len(neighbors)

    components = components + "," + str(disagreement_component2)

    disagree = 0
    for index in range (0,len(neighbors)):
        if neighbors[index] != neighbors_embd[index]:
            disagree = disagree + 1
    disagreement_component3 = float(disagree)/len(neighbors)

    components = components + "," + str(disagreement_component3)

    divergence = (1/3)*disagreement_component1 + (1/3)*disagreement_component2 + (1/3)*disagreement_component3

    return components, divergence

def compute_global_divergence(corr_feat_dist, corr_feat_dist_embd, neighbors, neighbors_embd, local_divergences):

    """
    Compute LAPS Divergence.
    
    Parameters:
    -----------------
    corr_feat_dist: nD array, correlations between the original feature contribution and distances
    corr_feat_dist: nD array, correlations between the embedding feature contribution and distances
    neighbors: array, list of nearest neighbors of the chosen row in the original data
    neighbors_embd: array, list of nearest neighbors of the chosen row in the embedding
    local_divergence: array, list of local diveregences of all selected points in the subset
    
    Returns:
    -----------------
    components: String, individual components of teh global divergence
    overall_divergence: float, final global diveregnce score
    """

    components = ""
    disagreements1 = pdist([corr_feat_dist, corr_feat_dist_embd], metric='euclidean')
    disagreements1 = 1/(1+disagreements1[0])
    components = components + str(disagreements1)

    disagreement2 = len(neighbors)-len(np.intersect1d(neighbors, neighbors_embd))
    disagreement2 = disagreement2/len(neighbors)
    components = components + "," + str(disagreement2)

    disagree = 0
    for index in range (0,len(neighbors)):
        if neighbors[index] != neighbors_embd[index]:
            disagree = disagree + 1
    disagreement3 = float(disagree)/len(neighbors)
    components = components + "," + str(disagreement3)

    divergence = (1/3)*disagreements1 + (1/3)*disagreement2 + (1/3)*disagreement3


    overal_divergence = 0
    for index in range(0,len(local_divergences)):

        overal_divergence = overal_divergence + (local_divergences[index] *(local_divergences[index]/divergence))
    
    overal_divergence = (1/(1+overal_divergence))
    return components, overal_divergence

