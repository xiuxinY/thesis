import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from sklearn.metrics import r2_score
import datetime


def getData(DataDir='ProcessPerformancesSamples/', performance_name=None):
    x = np.load(DataDir + 'Input.npy')
    y = np.load(DataDir + performance_name + '.npy')

    scaler_x = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(x)
    x_train = scaler_x.transform(x)
    scaler_y = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(y)
    y_train = scaler_y.transform(y)

    np.random.seed(666)
    np.random.shuffle(x)
    np.random.seed(666)
    np.random.shuffle(x_train)
    np.random.seed(666)
    np.random.shuffle(y)
    np.random.seed(666)
    np.random.shuffle(y_train)

    return x_train, y_train


def getModel(nodes_num=17):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(nodes_num, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss=tf.keras.losses.mse,
    )

    return model


def R2_Adj(y_true, y_pred):
    n, k = len(y_true), 7
    R2 = r2_score(y_true, y_pred)
    R2_Adj = 1 - (1 - R2) * (n - 1) / (n - k - 1)
    return R2_Adj


def create_checkpoint_callback(Dir='Checkpoints/', performance_name=None, node_num=None):
    CHECKPOINT_DIR = Dir + performance_name + '_' + \
        str(node_num) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_DIR + '.h5',
                                                                   save_weights_only=True,
                                                                   monitor='val_loss',
                                                                   mode='min',
                                                                   save_best_only=True)
    print(f"Saving Checkpoint files to: {CHECKPOINT_DIR}")
    return model_checkpoint_callback


def Model_Train(model, x_train, y_train):
    PARAMS = {
        'nb_epoch': 2000,
        'batch_size': 64,
        'verbose': 0,
        'validation_split': 0.2,
        'early_stopping_patience': 30,
        'reduce_lr_patience': 10,
    }

    # Define callbacks.
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                               patience=PARAMS['early_stopping_patience'],
                                                               verbose=1)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                              factor=0.5,
                                                              patience=PARAMS['reduce_lr_patience'],
                                                              min_lr=1e-10,
                                                              verbose=1)

    history = model.fit(
        x_train,
        y_train,
        epochs=PARAMS['nb_epoch'],
        batch_size=PARAMS['batch_size'],
        verbose=PARAMS['verbose'],
        validation_split=0.2,
        validation_freq=1,
        callbacks=[early_stopping_callback,
                   reduce_lr_callback, model_checkpoint_callback],
    )

    y_pred = model.predict(x_train, batch_size=16)
    acc = R2_Adj(y_train, y_pred)

    return acc, history


def history_save(Dir='History/', performance_name=None, node_num=None, history=None):
    history_df = pd.DataFrame(history.history)
    history_df['epoch'] = history.epoch
    history_df.to_csv(Dir + performance_name + '_' + str(node_num) + '_' +
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', index=False)
    print(f"Saving History files to: {Dir}")


def result_save(Dir='Results/', performance_name=None, node_num=None, acc=None):
    with open(Dir + performance_name + '_' + str(node_num) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.txt', 'w') as f:
        f.write(str(acc))
    print(f"Saving Result files to: {Dir}")

def model_save(Dir='Models/', performance_name=None, node_num=None, model=None):
    model.save(Dir + performance_name + '_' + str(node_num) + '.h5')
    print(f"Saving Model files to: {Dir}")

if __name__ == '__main__':
    keys = [
        "Energy",
        "CO2_Purity",
        "CO2_Recovery",
        "CO2_Productivity",
        "H2_Purity",
        "H2_Recovery",
        "H2_Productivity",
    ]

    nodes = 17  # 节点数确定为17
    single_result = pd.DataFrame(
        np.zeros((1, len(keys))), columns=keys)

    for key in keys:
            print('Training ' + key + ' with ' + str(nodes) + ' nodes')
            x_train, y_train = getData(performance_name=key)
            model = getModel(nodes_num=nodes)
            model_checkpoint_callback = create_checkpoint_callback(
                performance_name=key, node_num=nodes)
            acc, history = Model_Train(model, x_train, y_train)
            history_save(performance_name=key,
                         node_num=nodes, history=history)
            result_save(performance_name=key, node_num=nodes, acc=acc)
            model_save(performance_name=key, node_num=nodes, model=model)
            single_result.loc[0, key] = acc
            

    nodes_df = pd.DataFrame([nodes], columns=['Nodes'])
    results = pd.concat([nodes_df, single_result], axis=1)
    writer = pd.ExcelWriter("scalar_result_final.xlsx")
    results.to_excel(writer, "page_1")
    writer.save()
