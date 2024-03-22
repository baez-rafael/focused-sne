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
          logPQ = np.log10(p_matrix/q_matrix)
          np.fill_diagonal(logPQ, 0)
          term1 = logPQ[POI]
          term2 = logPQ[notPOI,:][:,POI]
          term3 = logPQ[notPOI,:][:,notPOI]
          sumTerm1 = np.sum(p_matrix[POI]*term1)*lambdas[0]
          sumTerm2 = np.sum(p_matrix[notPOI,:][:,POI]*term2)*lambdas[1]
          sumTerm3 = np.sum(p_matrix[notPOI,:][:,notPOI]*term3)*lambdas[2]
          cost += sumTerm1+sumTerm2+sumTerm3
          print(f'----Cost: {cost}')
        return X_transformed
        