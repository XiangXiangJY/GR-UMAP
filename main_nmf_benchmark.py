# -*- coding: utf-8 -*-
"""
Updated on 2/29/2024 
@author: yutah
"""
import numpy as np
from algorithm.auxilary import load_X, load_y, drop_sample, makeFolder
from algorithm.clustering import computeClusteringScore
from algorithm.nmf import NMF
from algorithm.rnmf import rNMF
from algorithm.graphs import computeGraphLaplacian
from algorithm.gnmf import GNMF
from algorithm.rgnmf import rGNMF
from algorithm.initialization import initialize
import argparse, os

def write_results(outfile, param, ari, nmi, purity, acc):
    file = open(outfile, 'w')
    file.writelines('Param,ARI,NMI' + '\n')
    file.writelines('%s,%.4f,%.4f,%.4f,%.4f'%(param, np.mean(ari), np.mean(nmi), np.mean(purity), np.mean(acc)))
    file.close()
    return

data_path = './SingleCellDataProcess/data/' 
data_process_path = './SingleCellDataProcess/'
init_path = './initialization/'

parser = argparse.ArgumentParser(description='Topological Nonnegative matrix factorization. There are 2 primary types, the TNMF and rTNMF. Additionally, 2 subtypes, the k-NN based filtration, and the cutoff based persistent Laplacian')
parser.add_argument('--data', type=str, default = 'GSE94820')
args = parser.parse_args()



X = load_X(args.data, data_path, data_process_path)
y = load_y(args.data, data_path, data_process_path)
if args.data != 'GSE57249':
    X, y = drop_sample(X, y)
    
X = np.log10(1+X)
X = X/ np.linalg.norm(X, axis = 0) [None , :]    
n_clusters = np.unique(y).shape[0]

M, N = X.shape
    





print('Computing NMF--------------------------------------------')
W, H = initialize(init_path, args.data, X)
myNMF = NMF(n_components = W.shape[1])
W, H = myNMF.fit_transform(X, W, H)
Ht = H.T
ari_nmf, nmi_nmf, purity_nmf, acc_nmf, LABELS = computeClusteringScore(Ht, y, max_state = 1)



print('Computing rNMF--------------------------------------------')
W, H = initialize(init_path, args.data, X)
myrNMF = rNMF(n_components = W.shape[1])
W, H = myrNMF.fit_transform(X, W, H)
Ht = H.T
ari_rnmf, nmi_rnmf, purity_rnmf, acc_rnmf, LABELS = computeClusteringScore(Ht, y, max_state = 1)




L, A, D = computeGraphLaplacian(X.T, n_neighbors = 8)

print('Computing GNMF--------------------------------------------')
W, H = initialize(init_path, args.data, X)
myGNMF = GNMF(n_components = W.shape[1], l = 1.)
W, H = myGNMF.fit_transform(X, W, H, L, A, D)
Ht = H.T
ari_gnmf, nmi_gnmf, purity_gnmf, acc_gnmf, LABELS = computeClusteringScore(Ht, y, max_state = 1)

print('Computing rGNMF--------------------------------------------')
W, H = initialize(init_path, args.data, X)
myrGNMF = rGNMF(n_components = W.shape[1], l = 1.)
W, H = myrGNMF.fit_transform(X, W, H, L, A, D)
Ht = H.T
ari_rgnmf, nmi_rgnmf, purity_rgnmf, acc_rgnmf, LABELS = computeClusteringScore(Ht, y, max_state = 1)




print('Scores for %s '%args.data )
print('Method \t ARI \t\t NMI \t\t Purity \t ACC \n')
print('NMF \t %.4f \t %.4f \t %.4f \t %.4f'%(ari_nmf.round(4), nmi_nmf.round(4), purity_nmf.round(4), acc_nmf.round(4)))
print('rNMF \t %.4f \t %.4f \t %.4f \t %.4f'%(ari_rnmf.round(4), nmi_rnmf.round(4), purity_rnmf.round(4), acc_rnmf.round(4)))
print('GNMF\t %.4f \t %.4f \t %.4f \t %.4f'%(ari_gnmf.round(4), nmi_gnmf.round(4), purity_gnmf.round(4), acc_gnmf.round(4)))
print('rGNMF\t %.4f \t %.4f \t %.4f \t %.4f'%(ari_rgnmf.round(4), nmi_rgnmf.round(4), purity_rgnmf.round(4), acc_rgnmf.round(4)))
