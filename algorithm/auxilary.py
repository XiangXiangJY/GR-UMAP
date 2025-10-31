# -*- coding: utf-8 -*-
"""
Updated on 2/29/2024 
@author: yutah
"""

import numpy as np
import pandas as pd
import csv
import os
from sklearn.cluster import KMeans

'''
This file is an auxilary file used for preprocessing of the data
'''

def makeFolder(outpath):
    try:
        os.makedirs(outpath)
    except:
        return
    return

def load_X(data, inpath, data_process_path):
    #load the data. Also, download the data if it is not available
    if not os.path.exists(inpath + data + '/%s_data.csv'%(data)):
        os.system('python %s/main.py --data %s --process_directory %s --outpath %s'%(data_process_path, data, data_process_path, inpath) )
    inpath = inpath + data + '/'
    X = pd.read_csv(inpath + '%s_data.csv'%(data))
    X = X.values[:, 1:].astype(float)
    return X




def load_y(data, inpath, data_process_path):
    #load the labels. Also, download the data if it is not available
    if not os.path.exists(inpath + data + '/%s_labels.csv'%(data)):
        os.system('python %s/main.py --data %s --process_directory %s --outpath %s'%(data_process_path, data, data_process_path, inpath) )
    inpath = inpath + data + '/'
    y = pd.read_csv(inpath + '%s_labels.csv'%(data))
    y = np.array(list(y['Label'])).astype(int)
    return y



def drop_sample(X, y):
    # Removed samples of class whose class size is less than 15
    original = X.shape[1]
    labels = np.unique(y)
    good_index = []
    for l in labels:
        index = np.where(y == l)[0]
        if index.shape[0] > 15:
            good_index.append(index)
        else:
            print('label %d removed'%(l))
    good_index = np.concatenate(good_index)
    good_index.sort()
    new = good_index.shape[0]
    print(original - new, 'samples removed')
    return X[:, good_index], y[good_index]


