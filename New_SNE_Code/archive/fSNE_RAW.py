import numpy as np

def _quickDistSquare(X: np.ndarray):
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)**2
    return D
def _x2p(X: np.ndarray):
    D = _quickDistSquare(X)
    (n, d) = X.shape
    sigma = 1 / (2**0.5)
    dist = D / (2 * sigma**2) * -1
    exp_all = np.exp(dist)
    np.fill_diagonal(exp_all, 0)
    denom = np.sum(exp_all, axis = 1)
    P = np.divide(exp_all.T, denom).T
    np.fill_diagonal(P, 0)
    return P


class fSNE:
    def __init__(self, X, n_comp = 2, lr = 0.1, max_iter = 1000, POI = [0], lambdas = [1,1,1]):
        self.X = X
        self.lr = lr
        self.n_comp = n_comp
        self.n_images = X.shape[0]
        self.max_iter = max_iter
        self.POI = POI
        self.lambdas = lambdas
    def fit_transform(self):
        POI = self.POI
        lambdas = self.lambdas
        
        X_transformed = np.random.rand(self.n_images, self.n_comp)
        print("Calculating P-Matrix")
        p_matrix = _x2p(self.X)
        
        update = 0
        iter_ind = -1
        notPOI = list(np.delete(np.array([i for i in range(self.X.shape[0])]),POI))
        #cache = [0 for i in range(self.n_images)]
        #for i in range(cache):
          #temp = p_matrix[]
          #cache[i] = np.
        while iter_ind < self.max_iter:
          iter_ind += 1
          if iter_ind < 250:
            alpha = 0.5
          else:
            alpha = 0.8
          print(f'Iterations {iter_ind} ...')
          dist_emb = _quickDistSquare(X_transformed)
          dist = dist_emb * -1
          exp_all = np.exp(dist)
          np.fill_diagonal(exp_all, 0)
          denom = np.sum(exp_all, axis = 1)
          q_matrix = np.divide(exp_all.T, denom).T
          np.fill_diagonal(q_matrix, 0)

                  
        #############################
          Y = X_transformed
          dPQ = (p_matrix - q_matrix) + (p_matrix - q_matrix).T
          
          for i in POI:
            X_i_prev_iter = X_transformed[i,:].reshape((-1,1))
            sum1 = 0
            sum2 = 0
            for j in range(self.n_images):
              X_j_prev_iter = X_transformed[j,:].reshape((-1,1))
              sum1 += 2*dPQ[j,i]*(X_i_prev_iter - X_j_prev_iter)
            for i_p in POI:
              X_i_p_prev_iter = X_transformed[i_p,:].reshape((-1,1))
              for j_p in POI:
                sum2 += p_matrix[j_p,i_p]*-2*q_matrix[i,i_p]*(X_i_p_prev_iter - X_i_prev_iter)
            sum1 *= lambdas[0]
            sum2 *= (1-lambdas[0])
            update = -(self.lr*(sum1+sum2)) + (alpha * update)
            X_i_transformed = X_i_prev_iter + update
            X_transformed[i,:] = X_i_transformed.ravel()
          for i in notPOI:
            X_i_prev_iter = X_transformed[i,:].reshape((-1,1))
            sum1 = 0
            sum2 = 0
            sum3 = 0
            sum4 = 0
            sum5 = 0
            for j in range(self.n_images):
              X_j_prev_iter = X_transformed[j,:].reshape((-1,1))
              sum1 += 2*dPQ[j,i]*(X_i_prev_iter - X_j_prev_iter)
              sum2 += 2*(X_i_prev_iter - X_j_prev_iter)*q_matrix[j,i]*np.sum(p_matrix[POI,:][:,i])
            for j in notPOI:
              X_j_prev_iter = X_transformed[j,:].reshape((-1,1))
              sum3 += 2*(X_j_prev_iter - X_i_prev_iter)*p_matrix[i,j]*(1-q_matrix[i,j])
              sum5 += 2*p_matrix[j,i]*(X_i_prev_iter - X_j_prev_iter)
            for i_p in notPOI:
              aux_sum = 0
              X_i_p_prev_iter = X_transformed[i_p,:].reshape((-1,1))
              for j_p in notPOI:
                aux_sum += p_matrix[j_p, i_p]*-2*(X_i_p_prev_iter-X_i_prev_iter)*q_matrix[i,i_p]
              sum4 += aux_sum
            sum1 *= lambdas[0]
            sum2 *= (1-lambdas[0])
            sum3 *= (1-lambdas[0])
            sum4 *= (1-lambdas[0])
            sum5 *= (1-lambdas[0])
            update = -(self.lr*(sum1+sum2+sum3+sum4-sum5)) + update#+ (alpha * update)
            X_i_transformed = X_i_prev_iter + update
            X_transformed[i,:] = X_i_transformed.ravel()
        # dY1_POI = lambdas[0] * 2 * np.sum(dPQ.expand_dims(1).transpose((0,2,1)) * (Y.expand_dims(1) - Y), axis = 1)
        # dY2_POI = (lambdas[0]-1) * 
        # p_matrix[POI,:][:,POI] - 2 * ()
        # gradientA = dY1_POI + dY2_POI






        #############################
          #--- Jitter ---
          #if iter_ind < 50:
          #  for a in range(self.X.shape[0]):
          #    noise = np.random.normal(0,0.1,self.n_comp)
          #    X_transformed[a, :] += noise
          #--- Cost ---
          cost = 0
          logPQ = np.log10(p_matrix/q_matrix)
          np.fill_diagonal(logPQ, 0)
          term1 = logPQ[POI]
          term2 = logPQ[notPOI,:][:,POI]
          term3 = logPQ[notPOI,:][:,notPOI]
          sumTerm1 = np.sum(p_matrix[POI]*term1)*lambdas[0]
          sumTerm2 = np.sum(p_matrix[notPOI,:][:,POI]*term2)*lambdas[0]
          sumTerm3 = np.sum(p_matrix[notPOI,:][:,notPOI]*term3)
          cost += sumTerm1+sumTerm2+sumTerm3
          print(f'----Cost: {cost}')
        return X_transformed
        