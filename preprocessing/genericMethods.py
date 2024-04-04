## imports for running DR algorithms

import math
import numpy as np
import pandas as pd

## imports for rep-data points and subset selection

from sklearn import svm
import scipy.spatial as spatial
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances_argmin_min

## imports from my files

from .DR_algorithms import run_DR_Algorithm
from .Preprocessing import set_features_and_target, identify_and_transform_features

## My methods

def get_dataset(dataset):
	"""
    Read dataset into Pandas dataframe

    Parameters:
    -----------------
    dataset : String, name of dataset

    Returns:
    -----------------
    data : dataframe, dataset
    """

	data = pd.read_csv('data/'+dataset+'.csv')
	return data

def find_density_based_outliers(data, budget):
	"""
    Find outliers based on: neighborhood density in the data

    Parameters:
    -----------------
    data : dataframe, dataset
    budget: integer, number of outliers the user would want to see

    Returns:
    -----------------
    outliers : array, array of outliers
    """

	density = []
	radius = 100
	
	point_tree = spatial.cKDTree(data)
	for index in range(0,data.shape[0]):
		density.append(len(point_tree.indices[point_tree.query_ball_point(data[index], radius)]))
	
	outliers = np.argsort(np.array(density))[::-1][:budget]
	
	return outliers

def find_points_with_highest_density(X_df, budget):
	"""
    Find points with the highest density in the data

    Parameters:
    -----------------
    X_df : dataframe, original dataset
    budget: integer, number of points the user would want to see

    Returns:
    -----------------
    highly_dense_points : array, array of highly dense points
    """

	density = []
	point_tree = spatial.cKDTree(X_df.values)
	for index in range(0,X_df.shape[0]):
	        density.append(len(point_tree.indices[point_tree.query_ball_point(X_df.values[index], 100)]))
	highly_dense_points = np.argsort(np.array(density))[::-1][:budget]

	return highly_dense_points

def find_misplaced_points(X_transformed, ld_embedding, budget):
	"""
    Find points with the lowest trustworthiness in the data

    Parameters:
    -----------------
    X_transformed: nD array, original dataset
    ld_embedding: nD array, low dimensional embedding of the dataset
    budget: integer, number of points the user would want to see

    Returns:
    -----------------
    misplaced_points : array, array of misplaced points
    """

	misplaced_neighbor_count = []

	for index in range (0,X_transformed.shape[0]):
		knn = NearestNeighbors(leaf_size=30, n_neighbors=15, p=2, radius=1.0, algorithm='ball_tree')

		knn.fit(X_transformed)
		neighbors = knn.kneighbors(X_transformed, return_distance=False)[index]

		knn.fit(ld_embedding)
		neighbors_embd = knn.kneighbors(ld_embedding, return_distance=False)[index]

		misplaced_neighbor_count.append(len(neighbors)-len(np.intersect1d(neighbors, neighbors_embd)))

	misplaced_points = np.argsort(np.array(misplaced_neighbor_count))[::-1][:budget]

	return misplaced_points

def find_points_close_to_decision_boundary(X_df, y, budget):
	"""
    Find points that are the closest to the decision boundary in the data

    Parameters:
    -----------------
    X_df: dataframe, original dataset
    y: array, target column in the data
    budget: integer, number of points the user would want to see

    Returns:
    -----------------
    points_close_to_decision_boundary : array, array of points close to decision boundary
    """

	svc = svm.SVC(kernel='linear').fit(X_df, y)
	decision = svc.decision_function(X_df)
	w_norm = np.linalg.norm(svc.coef_)
	dist = decision / w_norm

	points_close_to_decision_boundary = np.argsort(dist)[:budget]

	return points_close_to_decision_boundary

def find_cluster_centres(X_df, clusters=2):
	"""
    Find points that are the cluster centers in the data

    Parameters:
    -----------------
    X_df: dataframe, original dataset
    clusters: integer, number of clusters the user would want to see

    Returns:
    -----------------
    closest : array, array of points that are cluster centers
    """

	kmeans = KMeans(n_clusters=clusters, random_state=0).fit(X_df)
	closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_df)
	return closest


def generate_dissimilarity_density_based_subset(budget, X_df):
	"""
    Find data subset based on: neighborhood density and dissimilarity in the data

    Parameters:
    -----------------
    budget: integer, number of points the user would want to see in the subset
    X_df : dataframe, dataset

    Returns:
    -----------------
    selected : array, array containing subset
    """

	k = 10 ## Future work: optimize k for coverage

	selected = []
	active_status = {}
	density = []
	point_tree = spatial.cKDTree(X_df.values)
	knn = NearestNeighbors(leaf_size=30, n_neighbors=k, p=2, radius=1.0, algorithm='ball_tree')
	knn.fit(X_df)
    
    ## Find density of all points
    
	for index in range(0,X_df.shape[0]):
		density.append(len(point_tree.indices[point_tree.query_ball_point(X_df.values[index], 100)]))
		active_status.update({index : 'active'})
    # end for
    
    ## order points based on density
	ordered_density = np.argsort(np.array(density))[::-1]  ## ordered in descending order

    ## select the point with the highest density
	for index in range(0,X_df.shape[0]):
		active_array = []
        
        ## if the point is active and is not already a part of selected
		if active_status[ordered_density[index]] == 'active' and ordered_density[index] not in selected:
            
            ## Add point into selected
			selected.append(ordered_density[index])  
			nbrs = knn.kneighbors(X_df, return_distance=False)[ordered_density[index]]
            
            # Find its neighbors and change their status to inactive
			for nbr_index in range(0,len(nbrs)):
				active_status[nbrs[nbr_index]] = 'inactive'
            
            # for all the remaining active points in the active_status array, find their distances with the selected point
			for active_index in range(0,X_df.shape[0]):
				if active_status[active_index] == 'active':
					active_array.append(X_df.values[active_index])
           

			distances = pairwise_distances(
									active_array,
									X_df.values[ordered_density[index]].reshape(1, -1),
									metric='euclidean'
									).ravel()
            
            # Add the point with the highest distance from selected to the selected array
            # find its nearest neighbors and turn them to inactive
			counter = 0
			for active_index in range(0,X_df.shape[0]):
				if active_status[active_index] == 'active':
					counter = counter + 1
					if counter == np.argmax(distances):
						selected.append(active_index)
						nbrs_new = knn.kneighbors(X_df, return_distance=False)[active_index]
            
						for nbr_index in range(0,len(nbrs_new)):
							active_status[nbrs_new[nbr_index]] = 'inactive'
            
            # if all active points are exhausted and budget has not yet reached
            #make everything active again and start from the beginning
            
			if index == X_df.shape[0] and len(selected) < budget:
				for active_index in range(0,X_df.shape[0]):
					active_status[active_index] = 'active'
				index = 0
				continue
            
            # otherwise stop
			if len(selected) > budget:
				selected = selected[:-1]
            
			if len(selected) == budget:
				break
                
    # return the selected subset of datapoints        
	return selected

def generate_cluster_based_subset(budget, X_df):
	"""
    Find data subset based on: clusters in the data

    Parameters:
    -----------------
    budget: integer, number of points the user would want to see in the subset
    X_df : dataframe, dataset

    Returns:
    -----------------
    subset : array, array containing subset
    """

	kmeans = KMeans(n_clusters=2, random_state=0).fit(X_df)
	closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X_df)
	closest

	knn = NearestNeighbors(leaf_size=30, n_neighbors=15, p=2, radius=1.0, algorithm='ball_tree')
	knn.fit(X_df)
	nbrs = knn.kneighbors(X_df, return_distance=False)[closest[0]]
	nbrs1 = knn.kneighbors(X_df, return_distance=False)[closest[1]]

	if budget % 2 == 0:
		half_budget_1 = int(budget/2)
		half_budget_2 = half_budget_1
	else:
		half_budget_1 = int(budget/2)
		half_budget_2 = budget - half_budget_1

	subset = np.concatenate([np.array(nbrs[:half_budget_1]),np.array(nbrs1[:half_budget_2])]).tolist()
	
	return subset

def generate_representative_points(budget, data, data_transformed, target, ld_embedding):
	"""
    Identify representative points from the data

    Parameters:
    -----------------
    budget: integer, number of points the user would want to see in the subset
    data : dataframe, dataset
    data_transformed: nD array, transformed dataset
    target: aray, target column in the data
    ld_embedding: nD array, low dimensional embedding

    Returns:
    -----------------
    rep_points : nD array, nD array containing all possible representative points
    """

	rep_points = []

	rep_points.append(find_density_based_outliers(data.values, budget).tolist())
	rep_points.append(find_points_with_highest_density(data, budget).tolist())
	rep_points.append(find_misplaced_points(data_transformed, ld_embedding, budget).tolist())
	rep_points.append(find_points_close_to_decision_boundary(data, target, budget).tolist())
	rep_points.append(find_cluster_centres(data).tolist())

	return rep_points

def generate_representative_subset(budget, data):
	"""
    Identify representative subsets from the data

    Parameters:
    -----------------
    budget: integer, number of points the user would want to see in the subset
    data : dataframe, dataset

    Returns:
    -----------------
    rep_subsets : nD array, nD array containing all possible representative subsets
    """

	rep_subsets = []

	rep_subsets.append(generate_dissimilarity_density_based_subset(budget, data))
	rep_subsets.append(generate_cluster_based_subset(budget, data))

	return rep_subsets

def compute_coverage(X_df, rep_subset):
    nn_array = []

    knn = NearestNeighbors(leaf_size=30, n_neighbors=10, p=2, radius=1.0, algorithm='ball_tree')
    knn.fit(X_df)

    for index in range(0,len(rep_subset)):
        nn_array.append(knn.kneighbors(X_df, return_distance=False)[index])

    cc_nn_array = np.concatenate(nn_array, axis=0 )

    coverage = len(list(set(cc_nn_array))) / X_df.shape[0]
    print("Coverage of density and dissimilarity based subset is: %.2f" % (coverage * 100) + "%")

