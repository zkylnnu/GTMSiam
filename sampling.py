import numpy as np
import random

def get_coordinates_labels(y_hsi):   #  gt图 （463,241）
    max_label = np.max(y_hsi)        #  找最大值 2
    row_coords = []                  # 行坐标
    col_coords = []                  # 列坐标
    labels = []
    for lbl in range(1, max_label+1):
        real_label = lbl - 1
        lbl_locs = np.where(y_hsi == lbl)  # 找出第lbl类的全部索引
        row_coords.append(lbl_locs[0])     # 将索引的行坐标写入
        col_coords.append(lbl_locs[1])     # 将索引的列坐标写入
        length = len(lbl_locs[0])          # 第lbl类的数量
        labels.append(np.array([real_label]*length))   # 以length为长度的列表，值为类标签 0 / 1
    row_coords = np.expand_dims(np.concatenate(row_coords), axis=-1)   # 先连接行坐标， 再添加一个维度  （111583， 1）
    col_coords = np.expand_dims(np.concatenate(col_coords), axis=-1)   # 先连接列坐标， 再添加一个维度  （111583， 1）
    return np.concatenate([row_coords, col_coords], axis=-1), np.concatenate(labels)  # 连接行坐标列坐标   （111583,2）  连接对应标签（111583,1）

def get_train_test(data, data_labels, val_size, test_size = None, shuffle = True): # data行列坐标  data_labels 对应标签
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    val_data = []
    val_labels = []
    limited_num = None
    #test_size = None
    unique_labels = np.unique(data_labels)   # 找出不重复元素
    for index, label in enumerate(unique_labels):
        masks = (data_labels == label)     # 找出第label类对应掩码
        length = masks.sum()               # 第label类标签数量
        if test_size:       # 如果指定了测试比例
            nb_test = int(test_size * length)  #     该类数量 * 测试比例
            nb_val = int(val_size * length)
            test_indexes = np.random.choice(length, (nb_test,), replace=False)  # 在length内随机生成测试样本数量个随机数
            test_indexes = test_indexes.tolist()
            val_indexes = random.sample(test_indexes, nb_val)
            train_indexes = np.array([i for i in range(length) if i not in test_indexes]) # 上一行未被选取的即为训练集索引
            val_indexes = np.array(val_indexes)
            test_indexes = np.array(test_indexes)
        if limited_num:
            assert (test_size is None)
            if label==0:
                nb_train=3313   #97
            if label==1:
                nb_train=3919   #1018
            #nb_train = limited_num
            train_indexes = np.random.choice(length, (nb_train,), replace=False)
            val_indexes = np.random.choice(length, (1500,), replace=False)
            test_indexes = np.array([i for i in range(length) if i not in train_indexes])
            train_labels.extend(data_labels[masks][train_indexes])  # 将该类训练样本标签写入训练集标签
            test_labels.extend(data_labels[masks][test_indexes])  # 将该类测试样本标签写入测试集标签
            val_labels.extend(data_labels[masks][val_indexes])  # 将该类验证样本标签写入验证集标签
        else:
            train_labels.extend(data_labels[masks][train_indexes])  # 将该类训练样本标签写入训练集标签
            test_labels.extend(data_labels[masks][test_indexes])    # 将该类测试样本标签写入测试集标签
            val_labels.extend(data_labels[masks][val_indexes])      # 将该类验证样本标签写入验证集标签

        train_data.append(data[masks][train_indexes])        # 将该类训练样本位置写入训练集坐标集
        test_data.append(data[masks][test_indexes])          # 将该类测试样本位置写入测试集坐标集
        val_data.append(data[masks][val_indexes])            # 将该类验证样本位置写入验证集坐标集

    train_data = np.concatenate(train_data, axis=0)          # 连接训练集各类坐标 (3750, 2)
    train_labels = np.array(train_labels)                    # 连接训练集各类标签 (3750)
    test_data = np.concatenate(test_data, axis=0)            # 连接测试集各类坐标 (107833, 2)
    test_labels = np.array(test_labels)                      # 连接测试集各类标签 (107833)
    val_data = np.concatenate(val_data, axis=0)  # 连接测试集各类坐标 (107833, 2)
    val_labels = np.array(val_labels)  # 连接测试集各类标签 (107833)
    if shuffle:
        train_shuffle = np.random.permutation(len(train_labels))  # 生成打乱随机数
        train_data = train_data[train_shuffle]                    # 打乱训练集
        train_labels = train_labels[train_shuffle]
        test_shuffle = np.random.permutation(len(test_labels))    # 打乱测试集
        test_data = test_data[test_shuffle]
        test_labels = test_labels[test_shuffle]
        val_shuffle = np.random.permutation(len(val_labels))  # 打乱测试集
        val_data = val_data[val_shuffle]
        val_labels = val_labels[val_shuffle]
    return train_data, train_labels, test_data, test_labels, val_data, val_labels

def get_train_val_test(data, data_labels, test_size = None, shuffle = True): # data行列坐标  data_labels 对应标签
    train_data = []
    train_labels = []
    val_data = []
    test_data = []
    test_labels = []
    val_labels = []
    limited_num = None
    unique_labels = np.unique(data_labels)   # 找出不重复元素
    for index, label in enumerate(unique_labels):
        masks = (data_labels == label)     # 找出第label类对应掩码
        length = masks.sum()               # 第label类标签数量
        if test_size:       # 如果指定了测试比例
            nb_test = int(test_size * length)  #     该类数量 * 测试比例
            test_indexes = np.random.choice(length, (nb_test,), replace=False)  # 在length内随机生成测试样本数量个随机数
            train_indexes = np.array([i for i in range(length) if i not in test_indexes]) # 上一行未被选取的即为训练集索引
        if limited_num:
            assert (test_size is None)
            nb_train = limited_num
            train_indexes = np.random.choice(length, (nb_train,), replace=False)
            test_indexes = np.array([i for i in range(length) if i not in train_indexes])
        else:
            train_labels.extend(data_labels[masks][train_indexes])  # 将该类训练样本标签写入训练集标签
            test_labels.extend(data_labels[masks][test_indexes])    # 将该类测试样本标签写入测试集标签
        train_data.append(data[masks][train_indexes])        # 将该类训练样本位置写入训练集坐标集
        test_data.append(data[masks][test_indexes])          # 将该类测试样本位置写入测试集坐标集
    train_data = np.concatenate(train_data, axis=0)          # 连接训练集各类坐标 (3750, 2)
    train_labels = np.array(train_labels)                    # 连接训练集各类标签 (3750)
    test_data = np.concatenate(test_data, axis=0)            # 连接测试集各类坐标 (107833, 2)
    test_labels = np.array(test_labels)                      # 连接测试集各类标签 (107833)
    if shuffle:
        train_shuffle = np.random.permutation(len(train_labels))  # 生成打乱随机数
        train_data = train_data[train_shuffle]                    # 打乱训练集
        train_labels = train_labels[train_shuffle]
        test_shuffle = np.random.permutation(len(test_labels))    # 打乱测试集
        test_data = test_data[test_shuffle]
        test_labels = test_labels[test_shuffle]
    return train_data, train_labels, test_data, test_labels