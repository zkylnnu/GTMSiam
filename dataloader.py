import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
from numba import jit
def simple_data_generator(X1, X2, y, batch_size, shuffle = True, till_end = False):

    data_length = len(y)
    indexes = np.array(list(range(data_length)))
    if shuffle:
        np.random.shuffle(indexes)
    num_batches = data_length // batch_size
    for epi in range(num_batches):
        selected = indexes[epi*batch_size:(epi+1)*batch_size]
        X_batch1 = X1[selected]
        X_batch2 = X2[selected]       
        y_batch = y[selected]
        yield X_batch1, X_batch2, y_batch
    if till_end:
        if data_length % batch_size != 0:
            selected = indexes[batch_size * num_batches:]
            X_batch1 = X1[selected]
            X_batch2 = X2[selected]
            y_batch = y[selected]
            yield X_batch1, X_batch2, y_batch

def simple_data_generator_test(X1, X2, y, realcoords, batch_size = 24, shuffle = True, till_end = False):

    data_length = len(y)
    indexes = np.array(list(range(data_length)))
    if shuffle:
        np.random.shuffle(indexes)
    num_batches = data_length // batch_size
    for epi in range(num_batches):
        selected = indexes[epi*batch_size:(epi+1)*batch_size]
        X_batch1 = X1[selected]
        X_batch2 = X2[selected]
        y_batch = y[selected]
        order = realcoords[selected]
        yield X_batch1, X_batch2, y_batch, order
    if till_end:
        if data_length % batch_size != 0:
            selected = indexes[batch_size * num_batches:]
            X_batch1 = X1[selected]
            X_batch2 = X2[selected]
            y_batch = y[selected]
            yield X_batch1, X_batch2, y_batch




