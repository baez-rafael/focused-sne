import scipy.spatial.distance
import numpy as np
import torch
import sys
from time import time

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

def _quickDistSquare(X):
    X = X.cpu().numpy()
    aux = scipy.spatial.distance.pdist(X,'minkowski', p=2)
    D = scipy.spatial.distance.squareform(aux)
    return D**2
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
    return P
def x2q(X, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    D = _quickDistSquare(X)
    minusD = D*-1
    exp_all = np.exp(minusD) + 1e-12
    np.fill_diagonal(exp_all, 0)
    denom = np.sum(exp_all, axis = 1)
    Q = np.divide(exp_all, denom)
    Q = torch.tensor(Q)
    return Q.T
class fSNE_torch:
    def __init__(self, X, n_comp = 2, lr = 0.1, max_iter = 1000, POI = [0], lambdas = 1):
        self.X = X
        self.n_comp = n_comp
        self.n_samples = X.shape[0]
        self.lr = lr
        self.max_iter = max_iter
        self.POI = POI
        self.lambdas = lambdas
    def fit_transform(self, verbose = True):
        #start = time()
        POI = self.POI
        lambdas = self.lambdas
        
        notPOI = list(np.delete(np.array([i for i in range(self.n_samples)]),POI))
        if isinstance(self.X, np.ndarray):
          X = torch.tensor(self.X)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f"using {device}", file=sys.stderr)
        X = X.to(device)

        #X_transformed = np.random.normal(scale = 1e-4, size= self.X.shape)
        #X_updated = np.zeros((self.n_samples, self.n_comp))
        X_transformed = torch.randn(self.n_samples, self.n_comp, device = device)
        X_updated = torch.zeros(self.n_samples, self.n_comp, device = device)

        p_matrix = x2p_torch(self.X)
        p_matrix = p_matrix.to(device)
        #update = np.zeros((self.n_samples, self.n_comp))
        update = torch.zeros(self.n_samples, self.n_comp, device = device)

        cache_POI = np.zeros(self.n_samples)
        cache_nPOI = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            cache_POI[i] = np.sum(p_matrix[POI, i].cpu().numpy())
        for i in range(self.n_samples):
            cache_nPOI[i] = np.sum(p_matrix[notPOI, i].cpu().numpy())
        cache_POI = torch.Tensor(cache_POI).to(device)
        cache_nPOI = torch.Tensor(cache_nPOI).to(device)
        q_matrix = x2q(X_transformed)
        q_matrix = q_matrix.to(device)
        bar = range(self.max_iter)
        if verbose:
          bar = tqdm(bar)
        #print(f'Setup took {time()-start} seconds')
        for it in bar:
          #start = time()
          if it < 250:
            alpha = 0.5
          else:
            alpha = 0.8
          Y = X_transformed
          Yi_minus_Yj = Y.unsqueeze(1) - Y
          PQ = (p_matrix - q_matrix) + (p_matrix - q_matrix).T
          #sneGrad = lambdas * 2 * np.sum(np.expand_dims(PQ, 1).transpose((0,2,1)) * (np.expand_dims(Y, 1) - Y), axis = 1)
          sneGrad = 2 * lambdas * torch.sum(PQ.unsqueeze(1).transpose(2,1) * Yi_minus_Yj, dim = 1)
          
          for i in POI:
            auxPOIsum2 = torch.sum(cache_POI[POI].unsqueeze(1)*-2*q_matrix[i,POI].unsqueeze(1)*Yi_minus_Yj[POI,i], dim = 0)
            sum1 = sneGrad[i]
            sum2 = auxPOIsum2*(lambdas - 1)
            update[i] = -(self.lr * (sum1 + sum2)) + (alpha * update[i])
            aux = X_transformed[i,:] + update[i]
            X_updated[i, :] = aux
          #   sum2 = 0
          #   for i_p in POI:
          #     sum2 += cache_POI[i_p]*-2*q_matrix[i, i_p]*Yi_minus_Yj[i_p,i]
          #   sum2 *= (lambdas - 1)
          for i in notPOI:
            auxNPOIsum2 = torch.sum(2*Yi_minus_Yj[i,:]*q_matrix[:,i].unsqueeze(1)*cache_POI[i], dim = 0)
            auxNPOIsum3 = torch.sum(-2*Yi_minus_Yj[i,notPOI]*p_matrix[i,notPOI].unsqueeze(1)*(1-q_matrix[i,notPOI]).unsqueeze(1), dim = 0)
            auxNPOIsum4 = torch.sum(-2*cache_nPOI[notPOI].unsqueeze(1)*q_matrix[i, notPOI].unsqueeze(1)*(-1*Yi_minus_Yj[i,notPOI]), dim = 0)
            auxNPOIsum5 = torch.sum(-2*p_matrix[notPOI, i].unsqueeze(1)*Yi_minus_Yj[i,notPOI], dim = 0)
            # for j in range(self.n_samples):
            #   sum2 += 2 * Yi_minus_Yj[i,j] * q_matrix[j,i]*cache_POI[i]
            #sum3 = 0
            #sum4 = 0
            #sum5 = 0
            #for j in notPOI:
            #  sum3 += -2*Yi_minus_Yj[i,j] * p_matrix[i,j] * (1-q_matrix[i,j])
            #  sum4 += -2*cache_nPOI[j]*q_matrix[i,j]*(-1*Yi_minus_Yj[i,j])
            #  sum5 += -2*p_matrix[j,i]*(Yi_minus_Yj[i,j])
            sum1 = sneGrad[i]
            sum2 = auxNPOIsum2 * (lambdas - 1)
            sum3 = auxNPOIsum3 * (lambdas - 1)
            sum4 = auxNPOIsum4 * (lambdas - 1)
            sum5 = auxNPOIsum5 * (lambdas - 1)
            #sum3 *= (lambdas - 1)
            #sum4 *= (lambdas - 1)
            #sum5 *= (lambdas - 1)
            #print(f'New: {asum3}, Old: {sum3}')
            update[i] = -(self.lr*(sum1+sum2+sum3+sum4+sum5)) + (alpha*update[i])
            aux = X_transformed[i,:] + update[i]
            X_updated[i,:] = aux
          #print(f'Second loop took {time()-start} seconds')
          X_transformed = X_updated
          q_matrix = x2q(X_transformed)
          q_matrix = q_matrix.to(device)
          if verbose:
            logPQ = p_matrix*torch.log(p_matrix/q_matrix)
            logPQ.fill_diagonal_(0)
            term1 = logPQ[POI]
            term2 = logPQ[notPOI, :][:, POI]
            term3 = logPQ[notPOI, :][:, notPOI]
            sumTerm1 = torch.sum(term1) * lambdas
            sumTerm2 = torch.sum(term2) * lambdas
            sumTerm3 = torch.sum(term3)
            C = torch.sum(sumTerm1 + sumTerm2 + sumTerm3)
            bar.set_description(f'Cost: {C.cpu().item():.3f}')
        return X_transformed.cpu().numpy()         