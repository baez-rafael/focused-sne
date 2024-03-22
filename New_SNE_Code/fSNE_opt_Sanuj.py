import numpy as np
from sklearn.neighbors import kneighbors_graph as KNN
import torch
from time import time
def get_distances_btw_points(data_matrix):
    # data_matrix: rows are features and columns are samples
    n_samples = data_matrix.shape[1]
    distance_matrix = KNN(X=data_matrix.T, n_neighbors=n_samples-1, mode='distance', include_self=False, n_jobs=-1)
    distance_matrix = distance_matrix.toarray()
    return distance_matrix
    
# def _quickDistSquare(X: np.ndarray):
#     D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)**2
#     return D

def _x2p(X: np.ndarray):
    D = get_distances_btw_points(X.T)**2#_quickDistSquare(X)
    (n, d) = X.shape
    sigma2 = 0.5
    dist = -D / (2 * sigma2) 
    exp_all = np.exp(dist)
    np.fill_diagonal(exp_all, 0)
    denom = np.sum(exp_all, axis = 1)
    P = np.divide(exp_all, denom).T
    np.fill_diagonal(P, 0)
    return P



def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P

# https://github.com/mxl1990/tsne-pytorch
def x2p_torch(X, tol=1e-4, perplexity=50.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    # print("Computing pairwise distances...")
    (n, d) = X.shape

    X = torch.from_numpy(X)
    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        # if i % 500 == 0:
        #     print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        thisP = thisP.float()
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P.numpy()


class fSNE:
    def __init__(self, X, n_comp = 2, lr = 0.1, max_iter = 1000, POI = [0], lambdas = 1):
        self.X = X
        self.lr = lr
        self.n_comp = n_comp
        self.n_images = X.shape[0]
        self.max_iter = max_iter
        self.POI = POI
        self.lambdas = lambdas
    def fit_transform(self):
        #start = time()
        POI = self.POI
        lambdas = self.lambdas
        
        #X_transformed = np.random.rand(self.n_images, self.n_comp)
        X_transformed = np.random.normal(size=(self.n_images, self.n_comp), scale=1e-4)
        print("Calculating P-Matrix")
        # p_matrix = _x2p(self.X)

        p_matrix = x2p_torch(self.X)
        
        update = np.zeros((self.n_images,self.n_comp))

        iter_ind = -1
        notPOI = list(np.delete(np.array([i for i in range(self.X.shape[0])]),POI))

        
        cached_POI = {}
        for i in range(self.n_images):
          cached_POI[i] = np.sum(p_matrix[POI,i])
        
        cached_nPOI = {}
        for i in range(self.n_images):
          cached_nPOI[i] = np.sum(p_matrix[notPOI,i])
        dist_emb = get_distances_btw_points(X_transformed.T)**2#_quickDistSquare(X_transformed)
        dist = dist_emb * -1
        exp_all = np.exp(dist) + 1e-12
        np.fill_diagonal(exp_all, 0)
        denom = np.sum(exp_all, axis = 1) 
        q_matrix = np.divide(exp_all, denom).T
        np.fill_diagonal(q_matrix, 0)
        #print(f'Setup completed in {time()-start} seconds')
        while iter_ind < self.max_iter:
          #start = time()
          iter_ind += 1
          alpha = 0.8
          # if iter_ind < 250:
          #  alpha = 0.5
          # else:
          #  alpha = 0.8
          print(f'Iterations {iter_ind} ...')
         

                  
        #############################
          Y = X_transformed
          dPQ = (p_matrix - q_matrix) + (p_matrix - q_matrix).T
          dY = lambdas * 2 * np.sum(np.expand_dims(dPQ, 1).transpose((0,2,1)) * (np.expand_dims(Y, 1) - Y), axis = 1) 
          
          X_transformed_updated = np.random.normal(size=(self.n_images, self.n_comp), scale=1e-4)
          
          for i in POI:
            X_i_prev_iter = X_transformed[i,:]#.reshape((-1,1))
  
            sum2 = 0
            for i_p in POI:
              X_i_p_prev_iter = X_transformed[i_p,:]#.reshape((-1,1))
              sum2 += cached_POI[i_p]*-2*q_matrix[i,i_p]*(X_i_p_prev_iter - X_i_prev_iter)
            #sum1 = np.expand_dims(dY[i], 1)
            sum1= dY[i]
            sum2 *= (lambdas-1)
            sum2 = sum2.ravel()
            update[i] = -(self.lr*(sum1 + sum2)) + (alpha * update[i])
            X_i_transformed = X_i_prev_iter + update[i]
            X_transformed_updated[i,:] = X_i_transformed
          #print(f'First loop took {start-time()} seconds')
          #start = time()
          for i in notPOI:
            X_i_prev_iter = X_transformed[i,:]#.reshape((-1,1))
            #sum1 = 0
            sum2 = 0
            sum3 = 0
            sum4 = 0
            sum5 = 0
            for j in range(self.n_images):
              X_j_prev_iter = X_transformed[j,:]#.reshape((-1,1))
              #sum1 += 2*dPQ[j,i]*(X_i_prev_iter - X_j_prev_iter)
              sum2 += 2*(X_i_prev_iter - X_j_prev_iter)*q_matrix[j,i]*cached_POI[i]
              
            for j in notPOI:
              X_j_prev_iter = X_transformed[j,:]#.reshape((-1,1))
              X_i_minus_X_j = X_i_prev_iter - X_j_prev_iter
              sum3 += -2*(X_i_minus_X_j)*p_matrix[i,j]*(1-q_matrix[i,j])#TODO
              sum5 += -2*p_matrix[j,i]*(X_i_minus_X_j)
            for i_p in notPOI:
              X_i_p_prev_iter = X_transformed[i_p,:]#.reshape((-1,1))
              sum4 += cached_nPOI[i_p]*-2*q_matrix[i,i_p]*(X_i_p_prev_iter - X_i_prev_iter)
            # sum1 = np.expand_dims(dY[i], 1)
            sum1 = dY[i]
            sum2 *= (lambdas-1)

            sum3 *= (lambdas-1)
            sum3 = sum3.ravel()

            sum4 *= (lambdas-1)
            sum4 = sum4.ravel()

            sum5 *= (lambdas-1)
            sum5 = sum5.ravel()

            update[i] = -(self.lr*(sum1+sum2+sum3+sum4+sum5)) + (alpha * update[i])
            X_i_transformed = X_i_prev_iter + update[i]
            X_transformed_updated[i,:] = X_i_transformed
          #print(f'Second Loop took {time() -start} seconds')
          X_transformed = X_transformed_updated
        #############################
          
          #--- Jitter ---
          #if iter_ind < 50:
          #  for a in range(self.X.shape[0]):
          #    noise = np.random.normal(0,0.1,self.n_comp)
          #    X_transformed[a, :] += noise
          #--- Cost ---

          dist_emb = get_distances_btw_points(X_transformed.T)**2#_quickDistSquare(X_transformed)
          dist = dist_emb * -1
          exp_all = np.exp(dist) + 1e-12
          np.fill_diagonal(exp_all, 0)
          denom = np.sum(exp_all, axis = 1) 
          q_matrix = np.divide(exp_all, denom).T
          np.fill_diagonal(q_matrix, 0)

          cost = 0
          logPQ = np.log(p_matrix/q_matrix)
          np.fill_diagonal(logPQ, 0)
          term1 = logPQ[POI]
          term2 = logPQ[notPOI,:][:,POI]
          term3 = logPQ[notPOI,:][:,notPOI]
          sumTerm1 = np.sum(p_matrix[POI]*term1)*lambdas
          sumTerm2 = np.sum(p_matrix[notPOI,:][:,POI]*term2)*lambdas
          sumTerm3 = np.sum(p_matrix[notPOI,:][:,notPOI]*term3)
          cost += sumTerm1+sumTerm2+sumTerm3
          print(f'----Cost: {cost:.4f}')

        return X_transformed
        