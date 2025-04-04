import os
import glob
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from wandb.integration.keras import WandbCallback
import wandb
from collections import Counter

# 自定义 WandbCallback，覆盖 on_train_batch_end 以跳过图记录
class CustomWandbCallback(WandbCallback):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._step = 0

    def on_train_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        wandb.log(logs, step=self._step)
        self._step += 1


# 新增：解析 TFRecord 文件的函数
def parse_tfrecord_function(record, filename, table, num_classes):
    """
    解析单个TFRecord记录，并利用文件路径提取标签信息。
    假设TFRecord中存储了'event_grid'和'shape'两个字段，
    文件路径格式为 ".../类别名/xxx.tfrecord"，则类别名在倒数第二个位置。
    """
    features = {
        'event_grid': tf.io.FixedLenFeature([], tf.string),
        'shape': tf.io.FixedLenFeature([3], tf.int64)
    }
    # 解析单个样本
    example = tf.io.parse_single_example(record, features)
    event_grid = tf.io.decode_raw(example['event_grid'], tf.float32)
    shape = example['shape']
    event_grid = tf.reshape(event_grid, shape)
    event_grid = tf.image.resize(event_grid, (224, 224))
    event_grid = event_grid / 255.0

    # 从文件路径中提取标签
    parts = tf.strings.split(filename, os.sep)
    label_str = parts[-2]  # 假设类别名在倒数第二个位置
    label_int = table.lookup(label_str)
    label = tf.one_hot(label_int, depth=num_classes)
    return event_grid, label

def create_dataset_tf2(data_dir, batch_size, seed=42):
    """
    构建 TFRecord 数据集：
    - 使用 glob 获取所有 TFRecord 文件（排除 .ipynb_checkpoints 文件夹下的文件）
    - 根据父文件夹名称（类别名）做分层抽样
    - 利用 tf.data.TFRecordDataset 读取每个文件，并通过 interleave 方式将文件名附带到每条记录中
    - 最后 map 到解析函数中
    """
    # 获取所有TFRecord文件
    all_files = glob.glob(os.path.join(data_dir, "*/*.tfrecord"))
    all_files = [f for f in all_files if ".ipynb_checkpoints" not in f]

    # 获取有效类别（文件夹名称）并排序
    valid_classes = sorted([cls for cls in os.listdir(data_dir)
                             if os.path.isdir(os.path.join(data_dir, cls)) and cls != '.ipynb_checkpoints'])
    print("有效类别:", valid_classes)

    # 构造类别查找表
    keys = tf.constant(valid_classes)
    vals = tf.constant(range(len(valid_classes)), dtype=tf.int32)
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals), default_value=-1)
    num_classes = len(valid_classes)

    # 根据文件的父目录名称获取标签信息，用于 stratified 分层抽样
    labels = [os.path.basename(os.path.dirname(f)) for f in all_files]

    # 分层划分训练集和验证集
    train_files, val_files = train_test_split(
        all_files, test_size=0.2, random_state=seed, stratify=labels)
    
    print("训练集样本数量:", len(train_files))
    print("验证集样本数量:", len(val_files))
    print("训练集类别分布:", Counter([os.path.basename(os.path.dirname(f)) for f in train_files]))
    print("验证集类别分布:", Counter([os.path.basename(os.path.dirname(f)) for f in val_files]))

    # 定义一个函数：读取每个文件的TFRecord，并附带文件名信息
    def process_file(filename):
        ds = tf.data.TFRecordDataset(filename)
        # 将每个记录和对应的文件名打包在一起
        ds = ds.map(lambda record: (record, filename))
        return ds

    # 构建训练集
    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    train_ds = train_ds.interleave(lambda x: process_file(x),
                                   cycle_length=tf.data.AUTOTUNE,
                                   block_length=1)
    train_ds = train_ds.map(lambda record, filename: parse_tfrecord_function(record, filename, table, num_classes),
                            num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(1000, seed=seed).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    # 构建验证集
    val_ds = tf.data.Dataset.from_tensor_slices(val_files)
    val_ds = val_ds.interleave(lambda x: process_file(x),
                               cycle_length=tf.data.AUTOTUNE,
                               block_length=1)
    val_ds = val_ds.map(lambda record, filename: parse_tfrecord_function(record, filename, table, num_classes),
                        num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

def train():
    wandb.init()
    config = wandb.config

    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    data_paths = {
        "timeStack1281281": "/data/timeStack_1281281_tf",
    }
    data_name = config.get("data_name", "timeStack1281281")
    learning_rate = config.get("learning_rate", 0.0003)
    epochs = config.get("epochs", 90)
    batch_size = config.get("batch_size", 16)
    patience = config.get("patience", 100)
    min_delta = config.get("min_delta", 0.01)

    l2_reg = config.get("dense_l2", 1e-4)  # 这里的 1e-4 是兜底默认值，可根据需要修改
    dropout_rate = config.get("dense_dropout", 0.3)

    train_dataset, val_dataset = create_dataset_tf2(data_paths[data_name], batch_size, seed)

    input_tensor = Input(shape=(224, 224, 3))
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=input_tensor)
    base_model.trainable = True
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    def scheduler(epoch, lr):
        if epoch < 20:
            return learning_rate * (epoch + 1) / 20
        elif epoch < 80:
            T = 60
            cos_inner = np.pi * (epoch - 20) / T
            return learning_rate * (np.cos(cos_inner) + 1) / 2
        else:
            return learning_rate * 0.01
    lr_callback = LearningRateScheduler(scheduler)

    os.makedirs("models", exist_ok=True)
    checkpoint_path = f"models/best{data_name}_{learning_rate}_{batch_size}_{dropout_rate}_{l2_reg}_mbNetV2.h5"
    early_stop = EarlyStopping(monitor='val_accuracy', min_delta=min_delta, patience=patience, restore_best_weights=True)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

    # 使用自定义的 WandbCallback，禁用模型图记录
    wandb_callback = CustomWandbCallback(save_model=False, log_graph=False)

    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[lr_callback, early_stop, checkpoint, wandb_callback]
    )

    eval_results = model.evaluate(val_dataset)
    wandb.log({"Test Loss": eval_results[0], "Test Accuracy": eval_results[1]*100})

    valid_classes = sorted([cls for cls in os.listdir(data_paths[data_name])
                        if os.path.isdir(os.path.join(data_paths[data_name], cls)) and cls != '.ipynb_checkpoints'])
    y_preds = []
    y_trues = []
    for images, labels in val_dataset:
        preds = model.predict(images)
        y_preds.extend(np.argmax(preds, axis=1))
        y_trues.extend(np.argmax(labels.numpy(), axis=1))
    cm = metrics.confusion_matrix(y_trues, y_preds)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=valid_classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    wandb.log({"Confusion Matrix": wandb.Image("confusion_matrix.png")})

    wandb.finish()
