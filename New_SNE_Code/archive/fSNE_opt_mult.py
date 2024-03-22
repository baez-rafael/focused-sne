import numpy as np
import multiprocessing
from multiprocessing import Pool
from contextlib import closing

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
    def __init__(self, X, n_comp = 2, lr = 0.1, max_iter = 1000, POI = [0], lambdas = 1):
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
          alpha = 0.5
          #if iter_ind < 25:
          #  alpha = 0.5
          #else:
          #  alpha = 0.8
          print(f'Iterations {iter_ind} ...')
          dist_emb = _quickDistSquare(X_transformed)
          dist = dist_emb * -1
          exp_all = np.exp(dist)
          np.fill_diagonal(exp_all, 0)
          denom = np.sum(exp_all, axis = 1)
          q_matrix = np.divide(exp_all.T, denom).T
          np.fill_diagonal(q_matrix, 0)

                  
        
          Y = X_transformed
          dPQ = (p_matrix - q_matrix) + (p_matrix - q_matrix).T
          dY = lambdas * 2 * np.sum(np.expand_dims(dPQ, 1).transpose((0,2,1)) * (np.expand_dims(Y, 1) - Y), axis = 1) 
	  def POI_loop(point, x_transformed, grad1, p, q, poi, l, lr):
              #update = 0
              x_i_prev = x_transformed[point,:].reshape((-1,1))
              aux_sum = 0
              for i_p in poi:
                  x_ip_prev = x_transformed[i_p,:].reshape((-1,1))
                  aux_sum += np.sum(p[poi, i_p])*-2*q[point, i_p]*(x_i_prev - x_ip_prev)
              sum1 = np.expand_dims(grad1, 1)
              aux_sum *= (1-l)
              update = -(lr*(sum1 + aux_sum))# + (a*update))
              return (x_i_prev + update)
          def args_unpack(args):
              return POI_loop(*args)
          all_args = []
          #print(type(POI))
          #for P in POI:
          #    all_args.append(((P, X_transformed, dY[P], p_matrix, q_matrix, POI, lambdas, self.lr)))
          all_args = [(P, X_transformed, dY[P], p_matrix, q_matrix, POI, lambdas, self.lr) for P in POI]
          print(all_args[0])
          with Pool(processes=3) as pool:
              results = pool.map(args_unpack, all_args)
          print(results)
          for i in POI:
            X_i_prev_iter = X_transformed[i,:].reshape((-1,1))
            sum2 = 0
            #test = 0
            for i_p in POI:
              X_i_p_prev_iter = X_transformed[i_p,:].reshape((-1,1))
              sum2 += np.sum(p_matrix[POI,i_p])*-2*q_matrix[i,i_p]*(X_i_p_prev_iter - X_i_prev_iter)
            sum1 = np.expand_dims(dY[i], 1)
            sum2 *= (1-lambdas)
            update = -(self.lr*(sum1 + sum2)) + (alpha * update)
            X_i_transformed = X_i_prev_iter + update
            
            X_transformed[i,:] = X_i_transformed.ravel()
          for i in notPOI:
            X_i_prev_iter = X_transformed[i,:].reshape((-1,1))
            #sum1 = 0
            sum2 = 0
            sum3 = 0
            sum4 = 0
            sum5 = 0
            for j in range(self.n_images):
              X_j_prev_iter = X_transformed[j,:].reshape((-1,1))
              #sum1 += 2*dPQ[j,i]*(X_i_prev_iter - X_j_prev_iter)
              sum2 += 2*(X_i_prev_iter - X_j_prev_iter)*q_matrix[j,i]*np.sum(p_matrix[POI,:][:,i])
            for j in notPOI:
              X_j_prev_iter = X_transformed[j,:].reshape((-1,1))
              sum3 += 2*(X_j_prev_iter - X_i_prev_iter)*p_matrix[i,j]*(1-q_matrix[i,j])
              sum5 += 2*p_matrix[j,i]*(X_i_prev_iter - X_j_prev_iter)
            for i_p in notPOI:
              X_i_p_prev_iter = X_transformed[i_p,:].reshape((-1,1))
              sum2 += np.sum(p_matrix[:,i_p][notPOI])*-2*q_matrix[i,i_p]*(X_i_p_prev_iter - X_i_prev_iter)
            for i_p in notPOI:
              X_i_p_prev_iter = X_transformed[i_p,:].reshape((-1,1))
              sum4 += np.sum(p_matrix[notPOI,i_p])*-2*q_matrix[i,i_p]*(X_i_p_prev_iter - X_i_prev_iter)
              #for j_p in notPOI:
              #  test += p_matrix[j_p, i_p]*-2*(X_i_p_prev_iter-X_i_prev_iter)*q_matrix[i,i_p]
            sum1 = np.expand_dims(dY[i], 1)
            sum2 *= (1-lambdas)
            sum3 *= (1-lambdas)
            sum4 *= (1-lambdas)
            sum5 *= (1-lambdas)
            update = -(self.lr*(sum1+sum2+sum3+sum4-sum5)) + (alpha * update)
            X_i_transformed = X_i_prev_iter + update
            X_transformed[i,:] = X_i_transformed.ravel()

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
          sumTerm1 = np.sum(p_matrix[POI]*term1)*lambdas
          sumTerm2 = np.sum(p_matrix[notPOI,:][:,POI]*term2)*lambdas
          sumTerm3 = np.sum(p_matrix[notPOI,:][:,notPOI]*term3)
          cost += sumTerm1+sumTerm2+sumTerm3
          print(f'----Cost: {cost}')
        return X_transformed
        