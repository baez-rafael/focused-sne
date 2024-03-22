import scipy.spatial.distance
import numpy as np
import torch
import sys
from time import time
import gc

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
def x2p_torch(X, tol=1e-5, perplexity=30.0):
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
    del D
    del Di
    del beta
    gc.collect()
    # Return final Pc-matrix
    return (P+1e-24)
def x2q(X, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    #X = torch.from_numpy(X)
    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)
    minusD = D*-1
    exp_all = torch.exp(minusD) + 1e-24
    exp_all.fill_diagonal_(0)
    denom = torch.sum(exp_all, dim = 1)
    Q = torch.divide(exp_all, denom)
    # D = _quickDistSquare(X)
    # minusD = D*-1
    # exp_all = np.exp(minusD) + 1e-24
    # np.fill_diagonal(exp_all, 0)
    # denom = np.sum(exp_all, axis = 1)
    # Q = np.divide(exp_all, denom)
    # Q = torch.tensor(Q)
    del sum_X
    del D
    del minusD
    del exp_all
    del denom
    gc.collect()
    return Q.T

class fSNE_torch:
    def __init__(self, X, n_comp = 2, lr = 0.1, max_iter = 1000, POI = [0], ES=300, tol=1e-5, perp = 30.0, lambdas = 1):
        self.X = X
        self.n_comp = n_comp
        self.n_samples = X.shape[0]
        self.lr = lr
        self.max_iter = max_iter
        self.POI = POI
        self.lambdas = lambdas
        self.ES = ES
        self.tol = tol
        self.perp = perp
    def fit_transform(self, verbose = True):
        POI = self.POI
        lambdas = self.lambdas
        
        notPOI = list(np.delete(np.array([i for i in range(self.n_samples)]),POI))
        if isinstance(self.X, np.ndarray):
          X = torch.tensor(self.X)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       # device = "cpu"
        if verbose:
            print(f"using {device}", file=sys.stderr)
        X = X.to(device)
        X_transformed = torch.randn(self.n_samples, self.n_comp, dtype = torch.float64, device = device)
        X_updated = torch.zeros(self.n_samples, self.n_comp, dtype = torch.float64, device = device)
        p_matrix = x2p_torch(self.X, perplexity = self.perp)  
        p_matrix = p_matrix.to(device)
        update = torch.zeros(self.n_samples, self.n_comp, dtype = torch.float64, device = device)

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
        del X
        torch.cuda.empty_cache()
        gc.collect()
        
        if self.ES == 0:
            lowest_cost = -1e24
        else:
            lowest_cost = 1e24
        no_progress = 0
       
        
        bar = range(self.max_iter)
        if verbose:
          bar = tqdm(bar, position=0, leave=True)
        for it in bar:
          torch.cuda.empty_cache()
          #alpha = 0.5
          if it < 250:
            alpha = 0.5
          else:
            alpha = 0.8
          Y = X_transformed
          Yi_minus_Yj = Y.unsqueeze(1) - Y
          
          PQ = (p_matrix - q_matrix) + (p_matrix - q_matrix).T
          sneGrad = 2 * lambdas * torch.sum(PQ.unsqueeze(-1) * Yi_minus_Yj, dim = 1)
          loop1Grad1 = sneGrad[POI]
          loop1Grad2 = (lambdas - 1) * torch.sum(cache_POI[POI].unsqueeze(1) * -2 * q_matrix[POI][:,POI].unsqueeze(-1) * Yi_minus_Yj[POI][:,POI].transpose(1,0), dim = 1)
          
          update[POI] = -(self.lr*(loop1Grad1 + loop1Grad2)) + (alpha * update[POI])
          aux = X_transformed[POI,:] + update[POI]
          X_updated[POI,:] = aux
          
          
          Yi_minus_Yj_nPOI = Yi_minus_Yj[notPOI][:,notPOI]
          p_matrix_nPOI = p_matrix[notPOI][:, notPOI].unsqueeze(-1)
          q_matrix_nPOI = q_matrix[notPOI][:, notPOI].unsqueeze(-1)

          loop2Grad1 = sneGrad[notPOI]
          loop2Grad2 = (lambdas - 1)*torch.sum((2*Yi_minus_Yj[notPOI].transpose(1,0)*q_matrix[:,notPOI].unsqueeze(-1)*cache_POI[notPOI].unsqueeze(1)), dim = 0)
          loop2Grad3 = (lambdas - 1)*torch.sum(-2*Yi_minus_Yj_nPOI*p_matrix_nPOI*(1-q_matrix_nPOI), dim = 1)
          loop2Grad4 = (lambdas - 1)*torch.sum(-2*cache_nPOI[notPOI].unsqueeze(1)*q_matrix_nPOI*(-1*Yi_minus_Yj_nPOI), dim = 1)
          loop2Grad5 = (lambdas - 1)*torch.sum(2*p_matrix_nPOI*(Yi_minus_Yj_nPOI), dim = 0)
          
          update[notPOI] = -(self.lr*(loop2Grad1+loop2Grad2+loop2Grad3+loop2Grad4+loop2Grad5)) + (alpha*update[notPOI])
          aux = X_transformed[notPOI,:] + update[notPOI]
          X_updated[notPOI,:] = aux
          
          del q_matrix
          torch.cuda.empty_cache()
          X_transformed = X_updated
          q_matrix = x2q(X_transformed)
          q_matrix = q_matrix.to(device) + 1e-24
          
          logPQ = (p_matrix*torch.log(p_matrix/q_matrix))
          logPQ.fill_diagonal_(0)
          np_logPQ = logPQ.cpu().numpy()
        
          term1 = logPQ[POI]
          term2 = logPQ[notPOI, :][:, POI]
          term3 = logPQ[notPOI, :][:, notPOI]
          sumTerm1 = torch.sum(term1) * lambdas
          sumTerm2 = torch.sum(term2) * lambdas
          sumTerm3 = torch.sum(term3)
          C = torch.sum(sumTerm1 + sumTerm2 + sumTerm3)
          if C.cpu().item() != C.cpu().item():
            print()
            print('An unexpected error occured')
            print(f'C = {C.cpu().item()}')
            return X_transformed.cpu().numpy()
          if (lowest_cost - C - self.tol) < 0:
              no_progress +=1
              #print(no_progress)
          else:
              lowest_cost = C
              no_progress = 0
          if no_progress >= self.ES:
              print()
              print(f'No progress in the last {self.ES} iterations')
              return X_transformed.cpu().numpy()
          bar.set_description(f'Cost: {C.cpu().item():.5f}')
        return X_transformed.cpu().numpy()#, p_matrix.cpu().numpy(), q_matrix.cpu().numpy()         
