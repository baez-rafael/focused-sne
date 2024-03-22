import numpy as np
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from sklearn.neighbors import NearestNeighbors as KNN2  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html  and  https://stackoverflow.com/questions/21052509/sklearn-knn-usage-with-a-user-defined-metric
import os
import pickle
import matplotlib.pyplot as plt
import glob
from sklearn.metrics.pairwise import pairwise_kernels


class new_SNE:

    def __init__(self, X, n_comp = 2, lr = 0.1, max_iter = 1000, checkpoint = 20):
        self.X = X
        self.lr = lr
        self.n_comp = n_comp
        self.n_images = X.shape[0]
        self.max_iter = max_iter
        self.checkpoint = checkpoint
    def fit_transform(self):
        X_transformed = np.random.rand(self.n_images, self.n_comp)
        print('Calculating p')
        dist_matrix_orig = self.get_distances(data = self.X)
        dist_matrix_orig /= np.mean(dist_matrix_orig)
        p_matrix = np.zeros((self.n_images, self.n_images))
        for i in range(self.n_images):
          print(f'---Processing Image {i}')
          sigma = 1 / (2**0.5)
          d_square = (dist_matrix_orig[:,i] ** 2) / (2 * (sigma**2))
          d_square_noDiagElem = np.delete(d_square, i)
          denom = np.sum(np.exp(d_square_noDiagElem*-1))
          for j in range(self.n_images):
            if i != j:
              num = np.exp(d_square[j]*-1)
              if i < 2:
                print(f'{num} / {denom}')
              p = num / denom
            else:
              p = 0
            p_matrix[i, j] = p
        cost_iters = np.zeros((self.checkpoint, 1))
        update = 0
        iter_ind = -1

        while True:
          iter_ind += 1
          #--- Update alpha ---
          if iter_ind < 250:
            alpha = 0.5
          else:
            alpha = 0.8
          print(f'Iteration {iter_ind} ...')
          dist_matrix_emb = self.get_distances(data = X_transformed)
          #dist_matrix_emb = dist_matrix_emb/np.mean(dist_matrix_emb)
          q_matrix = np.zeros((self.n_images, self.n_images))
          for i in range(self.n_images):
            d_square = (dist_matrix_emb[:,i] ** 2)
            d_square_noDiagElem = np.delete(d_square, i)
            denom = np.sum(np.exp(d_square_noDiagElem*-1))
            for j in range(self.n_images):
              if i != j:
                num = np.exp(d_square[j]*-1)
                q = num / denom
              else:
                q = 0
              q_matrix[i,j] = q
          for i in range(self.n_images):
            X_i_prev_iter = X_transformed[i,:].reshape((-1,1))
            gradient = np.zeros((2,1))
            for j in range(self.n_images):
              X_j_prev_iter = X_transformed[j,:].reshape((-1,1))
              p_ij = p_matrix[i,j]
              p_ji = p_matrix[j,i]
              q_ij = q_matrix[i,j]
              q_ji = q_matrix[j,i]
              sumOfDists = (p_ij - q_ij + p_ji - q_ji)
              #print(sumOfDists.shape)
              #print(gradient.shape)
              #print(X_i_prev_iter.shape)
              gradient += sumOfDists*(X_i_prev_iter - X_j_prev_iter)
            gradient *= 2
            update = -(self.lr*gradient) + (alpha * update)
            X_i_transformed = X_i_prev_iter + update
            X_transformed[i,:] = X_i_transformed.ravel()
          #--- Jitter ---
          if iter_ind < 50:
            for a in range(self.X.shape[0]):
              noise = np.random.normal(0,0.1,self.n_comp)
              X_transformed[a, :] += noise
          #--- Cost ---
          cost = 0
          for i in range(self.n_images):
            for j in range(self.n_images):
              if i != j:
                p_ij = p_matrix[i,j]
                q_ij = q_matrix[i,j]
                if p_ij != 0 and q_ij != 0:
                  cost += (p_ij * np.log10(p_ij) - (p_ij*np.log10(q_ij)))
          print(f'----Cost: {cost}')
          cost_iters[iter_ind%self.checkpoint] = cost
          if self.max_iter is not None:
            if iter_ind > self.max_iter:
              return X_transformed, p_matrix
    def get_distances(self, data):
        distance_matrix = KNN(X = data, 
                              n_neighbors = data.shape[0]-1, 
                              mode = 'distance', 
                              include_self = False, 
                              n_jobs=-1)
        distance_matrix = distance_matrix.toarray()
        return distance_matrix