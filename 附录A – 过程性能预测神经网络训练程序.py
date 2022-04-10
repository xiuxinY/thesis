# -*- coding: utf-8 -*-
'''
论文题目：基于深度学习的PSA过程优化与控制
作    者：余秀鑫
单    位：天津大学化工学院
'''
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn import preprocessing
import time
import pandas as pd


def vANN(name, nodes):
    thisdata = name
    print(
        "================================Train ANN for:"
        + name
        + "=============================="
    )

    def R2_Adj(y_true, y_pred):
        n, k = len(y_true), 7
        R2 = r2_score(y_true, y_pred)
        R2_Adj = 1 - (1 - R2) * (n - 1) / (n - k - 1)
        return R2_Adj

    time_start = time.time()

    # 给定训练集
    path = "../results_of_sample1000\DL/"
    x_train = np.load("../results_of_sample1000\DL\Input.npy")
    y_train = np.load(path + thisdata + ".npy")

    scaler_x = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(x_train)
    x_train_std = scaler_x.transform(x_train)
    scaler_y = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(y_train)
    y_train_std = scaler_y.transform(y_train)

    np.random.seed(666)  # 使用相同的seed，保证输入特征和标签一一对应
    np.random.shuffle(x_train)
    np.random.seed(666)
    np.random.shuffle(x_train_std)
    np.random.seed(666)
    np.random.shuffle(y_train)
    np.random.seed(666)
    np.random.shuffle(y_train_std)
    tf.random.set_seed(666)

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(nodes, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )  # 定义模型

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss=tf.keras.losses.mse,
    )  # 定义损失函数和优化器

    history = model.fit(
        x_train_std,
        y_train_std,
        batch_size=64,
        epochs=2000,
        validation_split=0.2,
        validation_freq=1,
    )  # 训练模型

    model.summary()  # 打印模型概况

    time_end = time.time()
    print("Time cost", time_end - time_start)

    y_pred_std = model.predict(x_train_std, batch_size=16)
    acc = R2_Adj(y_train_std, y_pred_std)
    print("R2_Adj_std: ", acc)

    y_pred = scaler_y.inverse_transform(y_pred_std)
    acc = R2_Adj(y_train, y_pred)
    print("R2_Adj_scl: ", acc)

    return acc


time1 = time.time()
keys = [
    "Energy",
    "CO2_Purity",
    "CO2_Recovery",
    "CO2_Productivity",
    "H2_Purity",
    "H2_Recovery",
    "H2_Productivity",
]

nodes = np.arange(5, 41, 1)  # 节点数从5到41
single_result = pd.DataFrame(np.zeros((len(nodes), len(keys))), columns=keys)

for key in keys:
    for i in range(len(nodes)):
        single_result.loc[i, key] = vANN(key, nodes[i])

time2 = time.time()
print("标量预测ANN节点数探究的总耗时：", time2 - time1)

nodes_df = pd.DataFrame(nodes, columns=['Nodes'])
results = pd.concat([nodes_df, single_result], axis=1)
writer = pd.ExcelWriter("scalar_result_nodes_3.xlsx")
results.to_excel(writer, "page_1")
writer.save()
