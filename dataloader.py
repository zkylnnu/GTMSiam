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
        #A = graph_embbing(X_batch_slic,batch_size)
        #adj = notlocal_aru_pos(X_batch_slic, batch_size)
        y_batch = y[selected]
        #adj = graph_embbing(X_batch1, X_batch2, batch_size)

        #X_batch1 = X_batch1 + X_batch1_sm
        #X_batch2 = X_batch2 + X_batch2_sm
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
        #X_batch1_sm = notlocal_aru(X_batch1, batch_size)
        #X_batch2_sm = notlocal_aru(X_batch2, batch_size)
        #X_batch3_sm = notlocal_aru(X_batch_slic, batch_size)
        y_batch = y[selected]
        #A = graph_embbing(X_batch_slic, batch_size)
        order = realcoords[selected]
        #adj = notlocal_aru_pos(X_batch_slic)
        #Channel_adj = chan_aru(X_batch_slic, batch_size)
        #adj = graph_embbing(X_batch1, X_batch2, batch_size)

        #X_batch1 = X_batch1 + X_batch1_sm
        #X_batch2 = X_batch2 + X_batch2_sm
        yield X_batch1, X_batch2, y_batch, order
    if till_end:
        if data_length % batch_size != 0:
            selected = indexes[batch_size * num_batches:]
            X_batch1 = X1[selected]
            X_batch2 = X2[selected]
            y_batch = y[selected]
            yield X_batch1, X_batch2, y_batch

def graph_embbing(X, batch_size):    # (128,49,198)
    X1 = X # (128,49,198)
    patchsize = X.shape[1]
    # X_batch = X.transpose((2, 0, 1))
    C1 = np.mean(X, axis=2)  # (128,49,1)
    C1 = np.mean(C1, axis=1)  # (128,1,1)
    C2 = np.reshape(C1, [batch_size])  # (128)
    A = np.zeros([batch_size, batch_size])

    for i in range(batch_size):
        pix1 = C2[i]
        for j in range(patchsize):
            pix2 = C2[j]
            if A[i, j] != 0:
                continue
            diss = np.exp(-np.sum(np.square(pix1 - pix2)) / 10 ** 2)
            A[ i, j] = A[ j, i] = diss
    #adj1 = preprocessing.scale(A, axis=1)
    return A
@jit
def notlocal_aru(X_batch,batchsize):
    X_batcha = X_batch  # (128,49,198)
    X_1c = np.mean(X_batcha,axis=2)  # (128,49,1)
    X_1c = np.reshape(X_1c,[batchsize,-1])   # (128,49)
    X_similar = np.zeros_like(X_batch)
    best_list = np.zeros(batchsize)
    for i in range(batchsize):
        patch1 = X_1c[i]
        lowdiss = 10e6
        best_sm = 0
        for j in range(batchsize):
            if i==j:
                continue
            patch2 = X_1c[j]
            diff = patch1 - patch2
            diss = np.multiply(diff,diff)
            total_diss = np.sum(diss)
            if total_diss<lowdiss:
                lowdiss = total_diss
                best_sm = j
        best_list[i] = best_sm
        X_similar[i] = X_batch[best_sm]
    return best_list

def chan_aru(X, batchsize):
    channel = X.shape[2]
    X_batch = X.transpose((2, 0, 1))     # (198,128,49)
    C1 = np.mean(X_batch, axis=2)        # (198,128,1)
    C2 = np.reshape(C1,[channel,-1])     # (198,128)
    X_similar = np.zeros_like(X_batch)
    best_list = np.zeros(channel)
    for i in range(channel):
        c1 = C2[i]
        lowdiss = 10e9
        best_sm = 0
        for j in range(channel):
            if i==j:
                continue
            c2 = C2[j]
            diff = c1 - c2
            diss = np.multiply(diff,diff)
            total_diss = np.sum(diss)
            if total_diss<lowdiss:
                lowdiss = total_diss
                best_sm = j
        best_list[i] = best_sm
        X_similar[i] = X_batch[best_sm]

    return best_list

def Calcullate_linjie(X,batchsize):
    X1 = np.mean(X,axis=2)                # (128,49,1)
    X2 = np.reshape(X1,[batchsize,-1])    # (128,49)
    X3 = np.mean(X2,axis=1)               # (128,1)
    A = np.zeros([batchsize, batchsize], dtype=np.float32) # (128,128)
    for i in range(batchsize):
        for j in range(batchsize):
            pix1 = X3[i]
            pix2 = X3[j]
            diss = np.exp(-np.sum(np.square(pix1 - pix2)) / 10 ** 2)
            A[i,j]=A[j,i]=diss
    adj1 = preprocessing.scale(A, axis=1)
    return adj1

@jit
def notlocal_aru_pos(X_batch, batch_size):
    batchsize = X_batch.shape[0]
    length = X_batch.shape[1]
    channel = X_batch.shape[2]
    X_batcha = X_batch  # (128,49,198)
    X_1c = np.mean(X_batcha,axis=2)  # (128,49,1)
    X_1c = np.reshape(X_1c,[batchsize,-1])   # (128,49)
    #X_1c = X_1c.transpose(1,0)   # (25, 128)
    new_length = X_1c.shape[0]
    best = np.zeros_like(X_1c)


    for i in range(batchsize):    # 128
        patch1 = X_1c[i]          # 25
        totalduibi = np.zeros_like(X_1c)
        lowpos = np.zeros_like(patch1)
        for j in range(batchsize):
            if i == j:
                continue
            patch2 = X_1c[j]    # 取出第j个patch
            diss = patch1 - patch2
            diss = np.multiply(diss, diss)
            totalduibi[j] = diss
        totalduibi[i] = 10e6
        min11 = np.argmin(totalduibi,axis=0)
        best[i] = min11
    return best



