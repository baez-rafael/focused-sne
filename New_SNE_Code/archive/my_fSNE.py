import numpy as np
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
from sklearn.neighbors import NearestNeighbors as KNN2  # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html  and  https://stackoverflow.com/questions/21052509/sklearn-knn-usage-with-a-user-defined-metric
import os
import pickle
import matplotlib.pyplot as plt
import glob
from sklearn.metrics.pairwise import pairwise_kernels


class my_fSNE:

    def __init__(self, X, n_comp = 2, lr = 0.1, max_iter = 1000, checkpoint = 20, POI = [0], lambdas = [1,1,1]):
        self.X = X
        self.lr = lr
        self.n_comp = n_comp
        self.n_images = X.shape[0]
        self.max_iter = max_iter
        self.checkpoint = checkpoint
        self.POI = POI
        self.lambdas = lambdas
    def fit_transform(self):
        POI = self.POI
        lambdas = self.lambdas
        X_transformed = np.random.rand(self.n_images, self.n_comp)
        print('Calculating p')
        dist_matrix_orig = self.get_distances(data = self.X)
        #dist_matrix_orig /= np.mean(dist_matrix_orig)
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
              p = num / denom
            else:
              p = 0
            p_matrix[i, j] = p
        cost_iters = np.zeros((self.checkpoint, 1))
        update = 0
        iter_ind = -1
        notPOI = list(np.delete(np.array([i for i in range(self.X.shape[0])]),POI))

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
          q_orig = np.zeros((self.n_images, self.n_images))
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
              q_orig[i,j] = q
          
          for i in range(self.n_images):
            X_i_prev_iter = X_transformed[i,:].reshape((-1,1))
            gradientA = np.zeros((2,1))
            gradientB = np.zeros((2,1))
            gradientC = np.zeros((2,1))
            for j in range(self.n_images):
              X_j_prev_iter = X_transformed[j,:].reshape((-1,1))
              p_ij = p_matrix[i,j]
              p_ji = p_matrix[j,i]
              q_ij = q_matrix[i,j]
              q_ji = q_matrix[j,i]
              if j in POI:
                prod1 = np.sum(p_matrix[POI,i])
                prod2 = np.sum(p_matrix.T[i, POI])
                sumOfDists = (p_ij - q_ij*prod1 + p_ji - q_ji*prod2)
                gradientA += sumOfDists*(X_i_prev_iter - X_j_prev_iter)
              else:
                if i in POI:
                  prod1 = np.sum(p_matrix[notPOI,i])
                  prod2 = np.sum(p_matrix.T[i, notPOI])
                  sumOfDists = (p_ij - q_ij*prod1 + p_ji - q_ji*prod2)
                  gradientB += sumOfDists*(X_i_prev_iter - X_j_prev_iter)
                else:
                  prod1 = np.sum(p_matrix[notPOI,i])
                  prod2 = np.sum(p_matrix.T[i, notPOI])
                  sumOfDists = (p_ij - q_ij*prod1 + p_ji - q_ji*prod2)
                  gradientC += sumOfDists*(X_i_prev_iter - X_j_prev_iter)
              #sumOfDists = (p_ij - q_ij*prod1 + p_ji - q_ji*prod2)
              #gradient += sumOfDists*(X_i_prev_iter - X_j_prev_iter)
            gradientA *= 2*lambdas[0]
            gradientB *= 2*lambdas[1]
            gradientC *= 2*lambdas[2]
            gradient = gradientA + gradientB + gradientC
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
          logPQ = np.log10(p_matrix/q_orig)
          logPQ_man = np.zeros((self.n_images, self.n_images))
          np.fill_diagonal(logPQ, 0)
          term1 = logPQ[POI]
          term2 = logPQ[notPOI,:][:,POI]
          term3 = logPQ[notPOI,:][:,notPOI]
          sumTerm1 = np.sum(p_matrix[POI]*term1)*lambdas[0]
          sumTerm2 = np.sum(p_matrix[notPOI,:][:,POI]*term2)*lambdas[1]
          sumTerm3 = np.sum(p_matrix[notPOI,:][:,notPOI]*term3)*lambdas[2]
          cost += sumTerm1+sumTerm2+sumTerm3

          print(f'----Cost: {cost}')
          cost_iters[iter_ind%self.checkpoint] = cost
          if self.max_iter is not None:
            if iter_ind > self.max_iter:
              return X_transformed#, p_matrix, #, logPQ, logPQ_man
    def get_distances(self, data):
        distance_matrix = KNN(X = data, 
                              n_neighbors = data.shape[0]-1, 
                              mode = 'distance', 
                              include_self = False, 
                              n_jobs=-1)
        distance_matrix = distance_matrix.toarray()
        return distance_matrix