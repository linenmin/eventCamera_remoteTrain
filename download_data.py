import os
import gdown

# Google Drive 共享文件的 ID（从你的链接提取）
FILE_ID = "1dZMB9xU46GxzqZIJaLOssSm3KF8UL1gS"
OUTPUT_PATH = "data/timeStack_data_1281281.zip"

# 确保 data 目录存在
os.makedirs("data", exist_ok=True)

# 下载数据集
gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", OUTPUT_PATH, quiet=False)

print("数据下载完成！请手动解压 data/timeStack_1281281_tf.zip")
