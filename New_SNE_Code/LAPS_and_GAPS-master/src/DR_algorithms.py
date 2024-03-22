
import numpy as np
import trimap
import umap.umap_ as umap
#import umap
#from MulticoreTSNE import MulticoreTSNE as M_TSNE
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, Isomap, MDS

def run_DR_Algorithm(name, data_features):

    """
    Runs each DR algorithm and returns the embedding.
    
    Parameters:
    -----------------
    name : String, name of algorithm
    data_features : nD array, original features
    
    Returns:
    -----------------
    points : nD array
        embedding
    """
    int_dim=2

    if name == "UMAP":
        reducer = umap.UMAP(n_neighbors=15, n_components=int_dim)
        points = reducer.fit_transform(data_features)
        
    elif name == "tSNE":
        tsne = TSNE(n_components=int_dim, perplexity=30)
        points = tsne.fit_transform(data_features)
        
    
    elif name == "PCA":
        pca = PCA(n_components=int_dim)
        points = pca.fit_transform(data_features)
        

    elif name == "Trimap":
        points = trimap.TRIMAP().fit_transform(data_features)


    #elif name == "M_Core_tSNE":
    #    tsne = M_TSNE(n_components=int_dim, perplexity=30, n_jobs=8)
    #    points = tsne.fit_transform(data_features)


    elif name == "MDS":
        mds = MDS(n_components=int_dim)
        points = mds.fit_transform(data_features)


    elif name == "Isomap":
        isomap = Isomap(n_components=int_dim)
        points = isomap.fit_transform(data_features)


    elif name == "KernelPCA":
        kpca = KernelPCA(n_components=int_dim)
        points = kpca.fit_transform(data_features)

    return points














