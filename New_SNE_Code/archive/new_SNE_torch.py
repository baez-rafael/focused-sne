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
  sum_X = torch.sum(X*X, 1)
  D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)
  D.fill_diagonal_(0)
  return D
def _x2p_torch(X: torch.Tensor, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    D = quickDist(X)
    (n,d) = X.shape
    sigma = 1 / (2**0.5)
    dist_square = D / (2 * (sigma**2)) * -1
    exp_all = torch.exp(dist_square)
    exp_all.fill_diagonal_(0)
    denom = torch.sum(exp_all, dim = 0)
    p = torch.div(exp_all.T, denom).T
    p.fill_diagonal_(0)
    P = torch.zeros(n, n, device = device)
    P[:,:] = p
    # Return final P-matrix and Distance^2 matrix
    return P, D

def _tsne(X: Union[torch.Tensor, np.ndarray], no_dims: int = 2, initial_dims: int = 50, lr: float = 0.1, max_iter: int = 1000, verbose: bool = False):
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

    # Compute P-values
    if verbose:
        print("computing p-values...", file=sys.stderr)
    P, _ = _x2p_torch(X, device=device)
    #P = torch.max(P, torch.tensor(1e-21, device=P.device))  # (N, N)

    if verbose:
        print("fitting...", file=sys.stderr)

    bar = range(max_iter)

    if verbose:
        bar = tqdm(bar)

    for it in bar:
        if it < 250:
            momentum = alpha
        else:
            momentum = final_alpha
        # Compute Q Distances
        D_Y = quickDist(Y)
        #print(D_Y)
        dist_square = D_Y * -1
        exp_all = torch.exp(dist_square)
        exp_all.fill_diagonal_(0)
        denom = torch.sum(exp_all, dim = 0)
        q = torch.div(exp_all.T, denom).T
        q.fill_diagonal_(0)
        Q = torch.zeros(n, n, device = device)
        Q[:,:] = q
        # Compute gradient
        PQ = P - Q
        dPQ = (PQ + PQ.T)
        dY = 2 * torch.sum(
          dPQ.unsqueeze(1).transpose(2,1) * (Y.unsqueeze(1) - Y), dim = 1
        )

        #dY = 2 * torch.sum(
        #  dPQ.unsqueeze(1).transpose(2,1) * (Y.unsqueeze(1) - Y), dim = 1
        #)
        # Perform the update

        #print(dY.shape)
        #print(iY.shape)
        iY = -(lr*dY) + (momentum * iY)
        Y = Y + iY
        # Some Jitter
        if it < 50:
          noise = np.random.normal(0, 0.1, Y.shape)
          Y += torch.tensor(noise, device = device)
        # Compute current value of cost function
        if verbose:
            logPQ = torch.log10(P/Q)
            #print(f'logPQ : {logPQ.item()}')
            logPQ.fill_diagonal_(0)
            C = torch.sum(P * logPQ)
            bar.set_description(f"error: {C.cpu().item():.3f}")

        # Stop lying about P-values
        #if it == 100:
        #    P = P / 4.

    # Return solution
    return Y.detach().cpu().numpy()
class TorchTSNE:
    def __init__(
            self,
            lr: float = 0.1,
            n_iter: int = 1000,
            n_components: int = 2,
            initial_dims: int = 50,
            verbose: bool = False
    ):
        self.lr = lr
        self.n_iter = n_iter
        self.n_components = n_components
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
                initial_dims=self.initial_dims,
                lr=self.lr,
                verbose=self.verbose,
                max_iter=self.n_iter
            )