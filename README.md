# fSNE - Focused Stochastic Neighbor Embedding for Better Preserving Points of Interest
Pytorch implementation for focused-SNE (fSNE).

fSNE was designed to deal with the problem of focused dimensionality reduction. Given an original high-dimensional dataset and a set of points of interest, 
we want to find 2- or 3-dimensional embeddings of the original data such that the information loss 
in the local neighborhoods surrounding the points of interest is minimized as much as possible. In other words, if the information loss is inevitable, it should not happen around the points of interest.

# Environment
Tested on:-

**Local Machine -**
<ul>
  <li>Pytorch 1.7.1+cu110</li>
</ul>

# Running fSNE
Check out - src/Run_fSNE_SNE.ipynb to see an example of how to run fSNE_torch
[fsne](https://github.com/baez-rafael/focused-sne/blob/main/src/Run_fSNE_SNE.ipynb)

# Visualizations
Here are some example visualizations showcasing fSNE in action - <br/>
![vis1](/visualizations/original_fsne_sne_20News.png)
![vis1](/visualizations/original_fsne_sne_mnist.png)
![vis1](/visualizations/original_fsne_sne_Wine.png)

# Citation 
```
@inproceedings{ramirez2022focused,
  title={Focused Stochastic Neighbor Embedding for Better Preserving Points of Interest},
  author={Ramirez, Rafael Baez and Kumar, Sanuj and Le, Tuan MV and Cao, Huiping},
  booktitle={2022 IEEE/ACM International Conference on Big Data Computing, Applications, and Technologies (BDCAT)},
  pages={259--264},
  year={2022},
  organization={IEEE}
}
```
