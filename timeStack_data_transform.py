import os
import struct
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def read_aedat_file(filename):
    """读取单个 .aedat 文件，返回解析后的数据"""
    with open(filename, 'rb') as f:
        header_lines = []
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                raise ValueError(f"文件 {filename} 中没有找到二进制数据区，请检查文件格式")

            try:
                decoded_line = line.decode('ascii', errors='strict')
            except UnicodeDecodeError:
                # 无法ASCII解码，意味着此处已是二进制数据开始
                f.seek(pos)
                break

            stripped_line = decoded_line.strip()
            if stripped_line.startswith('#'):
                header_lines.append(stripped_line)
            else:
                f.seek(pos)
                break

        data_start_index = f.tell()  # 数据区起始偏移
        data = f.read()

    event_size = 8
    num_events = len(data) // event_size

    timestamps = []
    xs = []
    ys = []

    for i in range(num_events):
        event_data = data[i * event_size:(i + 1) * event_size]
        # 以大端序解析 address 和 timestamp
        address, t = struct.unpack('>ii', event_data)
        x = (address >> 1) & 0x7F
        y = (address >> 8) & 0x7F

        xs.append(x)
        ys.append(y)
        timestamps.append(t)

    return {
        'header': header_lines,
        'timestamps': timestamps,
        'xs': xs,
        'ys': ys
    }

def process_and_save_event_count_tf(input_base_folder, output_base_folder, grid_size=(128, 128), num_time_bins=1):
    """
    将 aedat 文件转换为事件计数网格，并保存为 TFRecord 格式。
    
    改进点：利用 NumPy 向量化处理事件数据，使用 np.histogramdd 快速统计
    """
    if not os.path.exists(output_base_folder):
        os.makedirs(output_base_folder)

    # 遍历类别文件夹
    class_folders = [f for f in os.listdir(input_base_folder) if os.path.isdir(os.path.join(input_base_folder, f))]
    for class_folder in tqdm(class_folders, desc="Processing Categories", unit="category"):
        input_folder = os.path.join(input_base_folder, class_folder)
        output_folder = os.path.join(output_base_folder, class_folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历当前类别下的 .aedat 文件
        files = [f for f in os.listdir(input_folder) if f.endswith('.aedat')]
        for filename in tqdm(files, desc=f"Processing {class_folder}", unit="file", leave=False):
            filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename.replace('.aedat', '.tfrecord'))

            # 读取 aedat 数据
            data = read_aedat_file(filepath)
            xs = data['xs']
            ys = data['ys']
            timestamps = data['timestamps']

            # 计算时间窗口边界
            t_min, t_max = np.min(timestamps), np.max(timestamps)
            time_bin_edges = np.linspace(t_min, t_max, num=num_time_bins + 1)

            # 向量化计算所有事件的时间索引
            t_indices = np.searchsorted(time_bin_edges, timestamps, side='right') - 1

            # 构造事件坐标数组，形状为 (N, 3) -> [x, y, t_index]
            event_coords = np.stack([xs, ys, t_indices], axis=1)

            # 定义每个维度的 bin 边界（注意边界需要多一格）
            bins = [np.arange(0, grid_size[0] + 1),
                    np.arange(0, grid_size[1] + 1),
                    np.arange(0, num_time_bins + 1)]
            # 使用 np.histogramdd 统计事件分布，结果形状为 (grid_size[0], grid_size[1], num_time_bins)
            event_count_grid, _ = np.histogramdd(event_coords, bins=bins)
            event_count_grid = event_count_grid.astype(np.int32)

            # 如果只有一个时间窗口，则将单通道复制三份，形成3通道数据
            if num_time_bins == 1:
                event_count_grid = np.repeat(event_count_grid, 3, axis=-1)

            # 转换为 TensorFlow 张量（后续序列化时使用）
            event_tensor = tf.convert_to_tensor(event_count_grid, dtype=tf.float32)

            # 序列化为 TFRecord 示例
            serialized_example = serialize_example(event_tensor.numpy())
            with tf.io.TFRecordWriter(output_filepath) as writer:
                writer.write(serialized_example)

def serialize_example(event_grid):
    """
    将事件计数网格序列化为 TFRecord 格式。
    保存两个字段：
      - 'event_grid': 以 bytes 格式存储的事件计数网格数据
      - 'shape': 记录网格的形状信息
    """
    feature = {
        'event_grid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[event_grid.tobytes()])),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=event_grid.shape))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()



# 参数设置
input_base_folder = r"C:\Users\Lem17\Master Thesis\Data processing\data_aedat2"
output_base_folder = r"D:\Dataset\eventData_dataset\timeStack_1281281_tf"
num_time_bins = 1  # 此处设为1时会自动复制为3通道
grid_size = (128, 128)

process_and_save_event_count_tf(input_base_folder, output_base_folder, grid_size=grid_size, num_time_bins=num_time_bins)
