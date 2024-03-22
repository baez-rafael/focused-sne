import sys
from typing import Union

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
def _quickDistSquare(X: torch.Tensor, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)
    D.fill_diagonal_(0)
    return D
def _x2p(X: torch.Tensor, device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    D = _quickDistSquare(X)
    (n, d) = X.shape
    sigma = 1 / (2**0.5)
    dist = D / (2 * sigma**2) * -1
    exp_all = torch.exp(dist)
    exp_all.fill_diagonal_(0)
    denom = torch.sum(exp_all, dim = 0)
    p = torch.div(exp_all.T, denom).T
    p.fill_diagonal_(0)
    P = torch.zeros(n, n, device = device)
    P[:,:] = p
    return P
def _SNE(X: Union[torch.Tensor, np.ndarray], no_dims: int = 2, lr: float = 0.1, max_iter: int = 1000, verbose: bool = False):
    if not isinstance(no_dims, int) or no_dims <= 0:
        raise ValueError("Dims must be positive integer")
    if isinstance(X, np.ndarray):
        X = torch.tensor(X)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"using {device}", file=sys.stderr)
    X = X.to(device)
    if verbose:
        print("initializing...", file=sys.stderr)
    
    # Initialize Variables
    (n, d) = X.shape
    alpha = 0.5
    f_alpha = 0.8
    Y = torch.randn(n, no_dims, device = device)
    iY = torch.zeros(n, no_dims, device = device)
    
    # P-Matrix
    if verbose:
        print("computing p-values...", file=sys.stderr)
    P = _x2p(X, device = device)

    if verbose:
        print("fiting...", file = sys.stderr)
    
    bar = range(max_iter)
    if verbose:
        bar = tqdm(bar)
    
    for it in bar:
        if it < 250:
            momentum = alpha
        else:
            momentum = f_alpha
        # Q-Matrix
        D_Y = _quickDistSquare(Y)
        dist = D_Y * -1
        exp_all = torch.exp(dist)
        exp_all.fill_diagonal_(0)
        denom = torch.sum(exp_all, dim = 0)
        q = torch.div(exp_all.T, denom).T
        q.fill_diagonal_(0)
        Q = torch.zeros(n, n, device = device)
        Q[:,:] = q
        PQ = P - Q
        dPQ = PQ + PQ.T
        dY = 2 * torch.sum(
          dPQ.unsqueeze(1).transpose(2,1) * (Y.unsqueeze(1) - Y), dim = 1
        )
        iY = -(lr*dY) + (momentum * iY)
        Y = Y + iY
        
        # Jitter
        if it < 50:
            noise = np.random.normal(0, 0.1, Y.shape)
            Y += torch.tensor(noise, device = device)
        if verbose:
            logPQ = torch.log10(P/Q)
            logPQ.fill_diagonal_(0)
            C = torch.sum(P*logPQ)
            bar.set_description(f'error: {C.cpu().item():.3f}')
    return Y.detach().cpu().numpy()
class TorchSNE:
    def __init__(
        self,
        lr: float = 0.1,
        n_iter: int = 1000,
        n_components: int = 2,
        verbose: bool = False        
    ):
        self.lr = lr
        self.n_iter = n_iter
        self.n_components = n_components
        self.verbose = verbose
    def fit_transform(self, X):
        with torch.no_grad():
            return _SNE(
                X,
                no_dims = self.n_components,
                lr = self.lr,
                verbose = self.verbose,
                max_iter = self.n_iter
            )
    