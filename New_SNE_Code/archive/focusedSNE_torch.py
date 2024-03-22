from ctypes import POINTER
import sys
from typing import Union

from sklearn.neighbors import kneighbors_graph as KNN
import numpy as np
import torch

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
def quickDist(X: torch.Tensor, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
  (n,d) = X.shape
  x_norm = (X**2).sum(1).view(-1,1)
  y_t = torch.transpose(X,0,1)
  y_norm = x_norm.view(1,-1)
  dist = x_norm + y_norm - 2.0 * torch.mm(X, y_t)
  #d = torch.clamp(dist, 1e-12, np.inf)
  #d.fill_diagonal_(0)
  D = torch.zeros(n,n, device = device)
  D[:,:] = dist
  return D
def _x2p_torch(X: torch.Tensor, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    D = quickDist(X)
    (n,d) = X.shape
    std_D = D/torch.mean(D)
    #std_D = D
    sigma = 1 / (2**0.5)
    dist_square = std_D / (2 * sigma**2) * -1
    exp_all = torch.exp(dist_square)
    exp_all.fill_diagonal_(0)
    denom = torch.sum(exp_all, dim = 0)
    p = torch.div(exp_all.T, denom).T
    p.fill_diagonal_(0)
    P = torch.zeros(n, n, device = device)
    P[:,:] = p
    # Return final P-matrix and Distance^2 matrix
    return P, D

def _tsne(X: Union[torch.Tensor, np.ndarray], no_dims: int = 2, POI = [0], lambdas = [2,1], initial_dims: int = 50, lr: float = 0.1, max_iter: int = 1000, verbose: bool = False):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    if not isinstance(no_dims, int) or no_dims <= 0:
        raise ValueError("dims must be positive integer")
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"using {device}", file=sys.stderr)

    X = X.to(device)

    if verbose:
        print("initializing...", file=sys.stderr)
    # Initialize variables
    (n, d) = X.shape
    alpha = 0.5
    final_alpha = 0.8
    update = 0
    Y = torch.randn(n, no_dims, device=device)
    iY = torch.zeros(n, no_dims, device=device)
    notPOI = list(np.delete(np.array([i for i in range(n)]),POI))
    # Compute P-values
    if verbose:
        print("computing p-values...", file=sys.stderr)
    P, _ = _x2p_torch(X, device=device)
    PG1 = torch.sum(P)
    #P = torch.max(P, torch.tensor(1e-21, device=P.device))  # (N, N)

    if verbose:
        print("fitting...", file=sys.stderr)

    bar = range(max_iter)

    if verbose:
        bar = tqdm(bar)

    for it in bar:
        # print(Y)
        # Compute Q Distances
        D_Y = quickDist(Y)
        #std_D_Y = D_Y/torch.mean(D_Y)
        std_D_Y = D_Y
        dist_square = std_D_Y * -1
        exp_all = torch.exp(dist_square)
        exp_all.fill_diagonal_(0)
        denom = torch.sum(exp_all, dim = 0)
        q = torch.div(exp_all.T, denom).T
        q.fill_diagonal_(0)
        Q = torch.zeros(n,n, device=device)
        Q[:,:] = q
        grad1 = P[POI,:]
        grad2 = P[notPOI, :][:,POI]
        grad3 = P[notPOI, :][:,notPOI]

        sumGrad1 = torch.sum(grad1, dim = 0)
        sumGrad2 = torch.sum(grad2, dim = 0)
        sumGrad3 = torch.sum(grad3, dim = 0)
        
        Q[POI,:] *= sumGrad1
        Q[notPOI,:][:,POI] *= sumGrad2
        Q[notPOI,:][:,notPOI] *= sumGrad3
        
        PQ = P-Q
        dPQ = PQ + PQ.T
        #print()
        #print('dPQ shape')
        # print(dPQ[POI,:].unsqueeze(1).transpose(2,1).shape)
        #print('y shape')
        # print((Y.unsqueeze(1) - Y)[POI,:].shape)
        sum1 = torch.sum(dPQ[POI,:].unsqueeze(1).transpose(2,1)*(Y.unsqueeze(1) - Y)[POI,:],dim=1)
        sum2 = torch.sum(dPQ[notPOI,:][:,POI].unsqueeze(1).transpose(2,1)*(Y.unsqueeze(1) - Y)[notPOI,:][:,POI],dim=1)
        sum3 = torch.sum(dPQ[notPOI,:][:,notPOI].unsqueeze(1).transpose(2,1)*(Y.unsqueeze(1) - Y)[notPOI,:][:,notPOI],dim=1)
        #sum1 = torch.sum(dPQ[POI,:].unsqueeze(1).transpose(2,1)*(Y.unsqueeze(1) - Y)[POI,:],dim=1)
        # print(sum1.shape)
        # print(sum2.shape)
        # print(sum3.shape)
        dY = torch.zeros(n, no_dims, device = device)
        dY[POI, :] += sum1*lambdas[0]*2
        dY[notPOI,:] += sum2*lambdas[0]*2
        dY[notPOI,:] += sum3*lambdas[1]*2
        #sum1 = torch.sum(dPQ[POI,:].unsqueeze(1).transpose(2,1)*(Y.unsqueeze(1) - Y), dim = 1)
        #dPQ[POI,:] *= lambdas[0]
        #dPQ[notPOI, :][:,POI] *= lambdas[1]
        #dPQ[notPOI, :][:, notPOI] *= lambdas[2]
        #dY = 2 * torch.sum(
        #  dPQ.unsqueeze(1).transpose(2,1) * (Y.unsqueeze(1) - Y), dim = 1
        #)
        #print(dY.shape)
        # Perform the update
        if it < 250:
            momentum = alpha
        else:
            momentum = final_alpha
        iY = -(lr*dY) + (momentum * iY)
        Y = Y + iY
        # Some Jitter
        if it < 50:
          noise = np.random.normal(0, 0.1, Y.shape)
          Y += torch.tensor(noise, device = device)
        # Compute current value of cost function
        if verbose:
            logPQ = torch.log10(P/q)
            #print(f'logPQ : {logPQ.item()}')
            logPQ.fill_diagonal_(0)
            term1 = logPQ[POI]
            term2 = logPQ[notPOI,:][:,POI]
            term3 = logPQ[notPOI,:][:,notPOI]
            sumTerm1 = torch.sum(P[POI]*term1)*lambdas[0]
            sumTerm2 = torch.sum(P[notPOI,:][:,POI]*term2)*lambdas[0]
            sumTerm3 = torch.sum(P[notPOI,:][:,notPOI]*term3)*lambdas[1]
            C = sumTerm1+sumTerm2+sumTerm3
            bar.set_description(f"error: {C.cpu().item():.3f}")

        # Stop lying about P-values
        #if it == 100:
        #    P = P / 4.

    # Return solution
    return Y.detach().cpu().numpy()
class TorchfSNE:
    def __init__(
            self,
            lr: float = 0.1,
            n_iter: int = 1000,
            n_components: int = 2,
            POI = [0],
            lambdas = [2,1], #,0.33],
            initial_dims: int = 50,
            verbose: bool = False
    ):
        self.lr = lr
        self.n_iter = n_iter
        self.n_components = n_components
        self.POI = POI
        self.lambdas = lambdas
        self.initial_dims = initial_dims
        self.verbose = verbose

    # noinspection PyPep8Naming,PyUnusedLocal
    def fit_transform(self, X, y=None):
        """
        Learns the t-stochastic neighbor embedding of the given data.
        :param X: ndarray or torch tensor (n_samples, *)
        :param y: ignored
        :return: ndarray (n_samples, n_components)
        """
        with torch.no_grad():
            return _tsne(
                X,
                no_dims=self.n_components,
                POI = self.POI,
                lambdas = self.lambdas,
                initial_dims=self.initial_dims,
                lr=self.lr,
                verbose=self.verbose,
                max_iter=self.n_iter
            )