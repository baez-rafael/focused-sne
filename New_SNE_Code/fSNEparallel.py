from multiprocessing import Pool, cpu_count, Process, Queue, Manager
from multiprocessing.sharedctypes import RawArray
import numpy as np
from sklearn.neighbors import kneighbors_graph as KNN
from time import time
import torch
from math import ceil

def get_distances_btw_points(data_matrix):
    # data_matrix: rows are features and columns are samples
    n_samples = data_matrix.shape[1]
    distance_matrix = KNN(X=data_matrix.T, n_neighbors=n_samples-1, mode='distance', include_self=False, n_jobs=-1)
    distance_matrix = distance_matrix.toarray()
    return distance_matrix
#def _quickDistSquare(X: np.ndarray):
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

# def POI_loop(points, x_transformed, sne, p, q, poi, cache_poi, up, yi_minus_yj, l, lr, a, res):
#     #update = 0
#     print('In POI loop')
#     x_transformed = np.array(x_transformed)
#     sne = np.array(sne)
#     p = np.array(p)
#     q = np.array(q)
#     up = np.array(up)
#     yi_minus_yj = np.array(yi_minus_yj)
#     for point in points:
#       sum2 = 0
#       for i_p in poi:
#           sum2 += cache_poi[i_p]*-2*q[point,i_p]*(yi_minus_yj[i_p, point])
#       sum1 = sne[point] 
#       sum2 *= (l-1)
#       aux_update = -(lr*(sum1 + sum2)) + (a * up[point])
#       aux_transformed = x_transformed[point] + aux_update
#       res[point] = (aux_transformed, aux_update)
#     print("Batch Done")
def POI_loop(points, x_transformed, sne, p, q, poi, cache_poi, up, yi_minus_yj, l, lr, a, res):
    #print('In POI loop')
    start = time()
    x_transformed = np.array(x_transformed)
    sne = np.array(sne)
    p = np.array(p)
    q = np.array(q)
    up = np.array(up)
    yi_minus_yj = np.array(yi_minus_yj)
    for point in points:
      sum1 = sne[point]
      sum2 = (l-1) * np.sum(-2*np.expand_dims(cache_poi[poi],1)*np.expand_dims(q[point, poi],1)*yi_minus_yj[poi,point],axis=0)
      aux_update = -(lr*(sum1 + sum2)) + (a * up[point])
      aux_transformed = x_transformed[point] + aux_update
      res[point] = (aux_transformed, aux_update)
    #print(f"Batch Done in {time()-start} seconds")
      #return (point, aux_transformed, aux_update)    
# def notPOI_loop(points, x_transformed, sne, p, q, notpoi, cache_poi, cache_notpoi, up, yi_minus_yj, l, lr, a, res):
#     print('In nPOI Loop')
#     x_transformed = np.array(x_transformed)
#     sne = np.array(sne)
#     p = np.array(p)
#     q = np.array(q)
#     up = np.array(up)
#     yi_minus_yj = np.array(yi_minus_yj)
#     for point in points:
#       sum2 = 0
#       sum3 = 0
#       sum4 = 0
#       sum5 = 0
#       for j in range(x_transformed.shape[0]):
#           x_i_minus_x_j = yi_minus_yj[point, j]
#           sum2 += 2*(yi_minus_yj[point, j])*q[j,point]*cache_poi[point]
#       #for j in notpoi:
#           if j in notpoi:
#           #x_j_prev = x_transformed[j, :]
#           #x_i_minus_x_j = x_i_prev - x_j_prev
#               sum3 += -2*(yi_minus_yj[point, j])*p[point, j]*(1-q[point, j])
#               sum4 += cache_notpoi[j]*-2*q[point, j]*(yi_minus_yj[j, point])
#               sum5 += -2*p[j, point]*yi_minus_yj[point, j]
#       #for i_p in notpoi:
#           #x_i_p_prev = x_transformed[i_p, :]
#           #sum4 += cache_notpoi[i_p]*-2*q[point, i_p]*(x_i_p_prev - x_i_prev)
#       sum1 = sne[point]    
#       sum2 *= (l-1)
#       sum3 *= (l-1)
#       sum4 *= (l-1)
#       sum5 *= (l-1)
#       aux_update = -(lr*(sum1+sum2+sum3+sum4+sum5)) + (a*up[point])
#       aux_transformed = x_transformed[point] + aux_update
#       res[point] = (aux_transformed, aux_update)
#     print("Batch Done")
def notPOI_loop(points, x_transformed, sne, p, q, notpoi, cache_poi, cache_notpoi, up, yi_minus_yj, l, lr, a, res):
    #print('In nPOI Loop')
    start = time()
    x_transformed = np.array(x_transformed)
    sne = np.array(sne)
    p = np.array(p)
    q = np.array(q)
    up = np.array(up)
    yi_minus_yj = np.array(yi_minus_yj)
    for point in points:
      sum1 = sne[point]   
      sum2 = (l - 1) * np.sum(2*yi_minus_yj[point,:]*np.expand_dims(q[:,point],1)*cache_poi[point], axis = 0)
      sum3 = (l - 1) * np.sum(-2*yi_minus_yj[point, notpoi]*np.expand_dims(p[point, notpoi],1)*(1-np.expand_dims(q[point, notpoi], 1)), axis = 0)
      sum4 = (l - 1) * np.sum(-2*np.expand_dims(cache_notpoi[notpoi],1)*np.expand_dims(q[point, notpoi],1)*(-1*yi_minus_yj[point, notpoi]), axis = 0)
      sum5 = (l - 1) * np.sum(-2*np.expand_dims(p[notpoi, point],1)*yi_minus_yj[point, notpoi], axis = 0)
      aux_update = -(lr*(sum1+sum2+sum3+sum4+sum5)) + (a*up[point])
      aux_transformed = x_transformed[point] + aux_update
      res[point] = (aux_transformed, aux_update)
    #print(f"Batch Done in {time()-start} seconds")
      #return (point, aux_transformed, aux_update)
class fSNE:
    def __init__(self, X, n_comp = 2, lr = 0.1, max_iter = 1000, POI = [0], lambdas = 1, cpus = None):
        self.X = X
        self.lr = lr
        self.n_comp = n_comp
        self.n_images = X.shape[0]
        self.max_iter = max_iter
        self.POI = POI
        self.lambdas = lambdas
        self.cpus = cpus
    def fit_transform(self):
        POI = self.POI
        lambdas = self.lambdas
        
        #X_transformed = np.random.rand(self.n_images, self.n_comp)
        X_transformed = np.random.normal(size=(self.n_images, self.n_comp), scale=1e-4)
        print("Calculating P-Matrix")
        #p_matrix = _x2p(self.X)
        p_matrix = x2p_torch(self.X)
        
        update = np.zeros((self.n_images,self.n_comp))

        iter_ind = -1
        notPOI = list(np.delete(np.array([i for i in range(self.X.shape[0])]),POI))

        cached_POI = np.zeros(self.n_images)
        cached_nPOI = np.zeros(self.n_images)
        for i in range(self.n_images):
            cached_POI[i] = np.sum(p_matrix[POI, i])
        for i in range(self.n_images):
            cached_nPOI[i] = np.sum(p_matrix[notPOI, i])
        # cached_POI = {}
        # for i in range(self.n_images):
        #   cached_POI[i] = np.sum(p_matrix[POI,i])
        
        # cached_nPOI = {}
        # for i in range(self.n_images):
        #   cached_nPOI[i] = np.sum(p_matrix[notPOI,i])
        print('Calculating Q-Matrix')
        dist_emb = get_distances_btw_points(X_transformed.T)**2
        #dist_emb = _quickDistSquare(X_transformed)
        dist = dist_emb * -1
        exp_all = np.exp(dist) + 1e-12
        np.fill_diagonal(exp_all, 0)
        denom = np.sum(exp_all, axis = 1) 
        q_matrix = np.divide(exp_all, denom).T
        np.fill_diagonal(q_matrix, 0)

        aux = cpu_count()
        if self.cpus is None:
          cpu = aux
        else:
          if self.cpus > aux:
            cpu = aux
          else:
            cpu = self.cpus
        print(f'Using {cpu} cores')
        batch = int(ceil(self.n_images/(cpu)))
        #batch = 1000
        print(f'Using a batch size of {batch}')

        while iter_ind < self.max_iter:
          start = time()
          iter_ind += 1
          alpha = 0.8
          # if iter_ind < 250:
          #   alpha = 0.5
          # else:
          #   alpha = 0.8
          print(f'Iterations {iter_ind} ...')
         

                  
        #############################
          Y = X_transformed
          dPQ = (p_matrix - q_matrix) + (p_matrix - q_matrix).T
          Yi_minus_Yj = np.expand_dims(Y, 1) - Y
          dY = lambdas * 2 * np.sum(np.expand_dims(dPQ, 1).transpose((0,2,1)) * (np.expand_dims(Y, 1) - Y), axis = 1) 
          
          X_transformed_updated = np.random.normal(size=(self.n_images, self.n_comp), scale=1e-4)
          
          results = Manager().dict()
          shared_X = Manager().list(X_transformed)
          shared_sne = Manager().list(dY)
          shared_p = Manager().list(p_matrix)
          shared_q = Manager().list(q_matrix)
          shared_update = Manager().list(update)
          shared_yi_minus_yj = Manager().list(Yi_minus_Yj)
          #tmp = np.ctypeslib.as_ctypes(q_matrix)
          #shared_q = RawArray(tmp._type_, tmp)
                          #(point, x_transformed, sum1, p,    q,        poi, cache_poi,  up,        l,      lr,       a)
          # args_poi_loop = [(P, X_transformed, dY, p_matrix, q_matrix, POI, cached_POI, update, Yi_minus_Yj, lambdas, self.lr, alpha, results) for P in POI]
          #                    #(point, x_transformed, sum1, p,    q,        poi, cached_poi, cache_notpoi,  up,        l,      lr,       a)
          # args_notpoi_loop = [(P, X_transformed, dY, p_matrix, q_matrix, POI, cached_nPOI, cached_POI, update, Yi_minus_Yj, lambdas, self.lr, alpha, results) for P in notPOI]
          # results_poi = 0
          # results_npoi = 0
          # with Pool(processes = cpu) as pool:
          #     results_npoi = pool.starmap(notPOI_loop, args_notpoi_loop, chunksize = batch)
          #     results_poi = pool.starmap(POI_loop, args_poi_loop, chunksize = batch)

          def split_into_batch(work, batch_size):
            return [work[x:x+batch_size] for x in range(0, len(work), batch_size)]
          split_loop_1 = split_into_batch(POI, batch)
          split_loop_2 = split_into_batch(notPOI, batch)
          all_loop = split_loop_2 + split_loop_1
          jobs = []
          
          for i in range(len(all_loop)):
            #print(f'Batch {i}')
            if i < len(split_loop_2):
              #(all_loop[i], X_transformed, dY[P], p_matrix, q_matrix, POI, cached_nPOI, cached_POI, update[P], lambdas, self.lr, alpha, results)
              p = Process(target = notPOI_loop, args = (all_loop[i], shared_X, shared_sne, shared_p, shared_q, POI, cached_nPOI, cached_POI, shared_update, shared_yi_minus_yj, lambdas, self.lr, alpha, results))
              jobs.append(p)
              p.start()
            else:
              #(all_loop[i], X_transformed, dY[P], p_matrix, q_matrix, POI, cached_POI, update[P], lambdas, self.lr, alpha, results)
              p = Process(target = POI_loop, args = (all_loop[i], shared_X, shared_sne, shared_p, shared_q, POI, cached_POI, shared_update, shared_yi_minus_yj, lambdas, self.lr, alpha, results))
              jobs.append(p)
              p.start()
          for proc in jobs:
            proc.join()
          for P in results:
            X_transformed_updated[P,:] = results[P][0]
            update[P] = results[P][1]

          # for i, P in enumerate(POI):
          #     #print(P)
          #     X_transformed_updated[P,:] = results_poi[i][1]
          #     update[P] = results_poi[i][2]
          # for i, nP in enumerate(notPOI):
          #     X_transformed_updated[nP,:] = results_npoi[i][1]
          #     update[nP] = results_npoi[i][2]
          X_transformed = X_transformed_updated
        #############################
          
          #--- Jitter ---
          #if iter_ind < 50:
          #  for a in range(self.X.shape[0]):
          #    noise = np.random.normal(0,0.1,self.n_comp)
          #    X_transformed[a, :] += noise
          #--- Cost ---
          #print(X_transformed[:5])
          dist_emb = get_distances_btw_points(X_transformed.T)**2
          #dist_emb = _quickDistSquare(X_transformed)
          dist = dist_emb * -1
          exp_all = np.exp(dist) + 1e-12
          np.fill_diagonal(exp_all, 0)
          denom = np.sum(exp_all, axis = 1) 
          q_matrix = np.divide(exp_all, denom).T
          np.fill_diagonal(q_matrix, 0)


          logPQ = p_matrix*np.log(p_matrix/q_matrix)
          np.fill_diagonal(logPQ, 0)
          term1 = logPQ[POI]
          term2 = logPQ[notPOI,:][:,POI]
          term3 = logPQ[notPOI,:][:,notPOI]
          sumTerm1 = np.sum(term1)*lambdas
          sumTerm2 = np.sum(term2)*lambdas
          sumTerm3 = np.sum(term3)
          cost = sumTerm1+sumTerm2+sumTerm3
          print(f'----Cost: {cost:.4f}, That took {(time()-start):.4f} seconds')
          #print(f'That took {(time()-start):.4f} seconds')
        return X_transformed