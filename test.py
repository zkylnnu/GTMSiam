import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from numba import jit
from scipy.io import savemat

from data_pre import load_dataset, sampling, get_mask_onehot, Grammar, zeropad_to_max_len, Data_Generator
from dataloader import simple_data_generator_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Draw_Classification_Map(label, scale: float = 4.0, dpi: int = 400):
    '''
    get classification map , then save to given path
    :param label: classification label, 2D
    :param name: saving path and file's name
    :param scale: scale of image. If equals to 1, then saving-size is just the label-size
    :param dpi: default is OK
    :return: null
    '''
    fig, ax = plt.subplots()
    numlabel = np.array(label)
    numlabel = numlabel.astype(np.int16)
    #numlabel = np.where(numlabel > 1, 0, 1)
    # savemat('gra.mat', {'a': numlabel})
    plt.imshow(numlabel, cmap='gray')
    # v = spy.imshow(classes=numlabel.astype(np.int16), fignum=fig.number)

    # spy.imshow(numlabel)
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.set_size_inches(label.shape[1] * scale / dpi, label.shape[0] * scale / dpi)
    foo_fig = plt.gcf()  # 'get current figure'
    # cmap = plt.cm.get_cmap('gray')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig('figures/new.png', format='png', transparent=True, dpi=dpi, pad_inches=0)
    pass

def cal_kappa(pred, truth):

    pred = pred.detach().cpu().numpy()
    truth = truth.detach().cpu().numpy()
    cal_label = 1 - truth
    cal_pred = 1 - pred
    tp = np.sum(np.logical_and(cal_pred, cal_label))
    tn = np.sum(np.logical_not(np.logical_or(cal_pred, cal_label)))
    fp = np.sum(np.logical_and(np.logical_xor(cal_pred, cal_label), cal_pred))
    fn = np.sum(np.logical_and(np.logical_xor(cal_pred, cal_label), cal_label))

    return tp, tn, fp, fn

def test(X1, X2, test_coords, y_test, net, arg, y):
    X_test1, X_test2 = Grammar(X1, X2, test_coords, method="rect 5")
    total_test_eq = 0
    total_tp = total_tn = total_fp = total_fn = 0
    X_test_shape = X_test1.shape
    y = 2 - y
    pred = y
    realcoords = test_coords - 4
    if len(X_test_shape) == 4:
        X_test1 = np.reshape(X_test1, [X_test_shape[0], X_test_shape[1] * X_test_shape[2], X_test_shape[3]])
        X_test2 = np.reshape(X_test2, [X_test_shape[0], X_test_shape[1] * X_test_shape[2], X_test_shape[3]])
    X_test1 = zeropad_to_max_len(X_test1, max_len=arg.max_len)
    X_test2 = zeropad_to_max_len(X_test2, max_len=arg.max_len)

    test_generator = simple_data_generator_test(X_test1, X_test2, y_test, realcoords, batch_size=arg.batch_size, shuffle=True)
    torch.cuda.empty_cache()
    net.load_state_dict(torch.load("model\\best_model.pt"))
    net.eval()
    for ind, (X_batch1, X_batch2, y_batch, order) in enumerate(test_generator):
        with torch.no_grad():
            X_batch1 = torch.from_numpy(X_batch1.astype(np.float32)).to(device)
            X_batch2 = torch.from_numpy(X_batch2.astype(np.float32)).to(device)
            y_batch = torch.from_numpy(y_batch.astype(np.float32)).to(device)
            output = net(X_batch1, X_batch2)
            logits = torch.argmax(output, dim=1)
            TP, TN, FP, FN = cal_kappa(logits, y_batch)
            total_tp = total_tp + TP
            total_tn = total_tn + TN
            total_fp = total_fp + FP
            total_fn = total_fn + FN
            locpu = logits.detach().cpu()
            loo = 1 - locpu
            for j in range(len(loo)):
                a = order[j][0]  
                b = order[j][1]  
                pred[a][b] = loo[j]
    Draw_Classification_Map(pred)
    total_num = total_tp + total_tn + total_fp + total_fn
    total_num2 = total_num
    test_OA = (total_tp + total_tn) / total_num
    suanzi1 = (total_tp +total_fp)/total_num * (total_tp+total_fn)/total_num
    suanzi2 = (total_tn + total_fp)/total_num * (total_tn + total_fn)/total_num
    PRE = suanzi1 + suanzi2
    test_kappa = (test_OA - PRE)/(1-PRE)
    print(total_tp,total_tn,total_fp,total_fn)
    del net

    return output, test_OA, test_kappa
