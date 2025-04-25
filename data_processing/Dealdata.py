# 数据处理接口
import os
import numpy as np
from .Data_conversion import to_numpy
from .Filtering import ECG_Filter

# 定义保存函数
def save_processed_data(data, original_mat_path):
    # 生成npy保存路径（同目录、同名、不同后缀）
    save_dir = os.path.dirname(original_mat_path)
    base_name = os.path.splitext(os.path.basename(original_mat_path))[0]
    save_path = os.path.join(save_dir, f"{base_name}.npy")
    
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存数据
    np.save(save_path, data)
    
    # 打印信息
    print(f"保存处理后的数据: {os.path.basename(save_path)}\n"
    f"—— 保存路径: {save_path}\n"
    f"—— shape={data.shape} | dtype={data.dtype}\n")
    return save_path

# 处理数据并保存
def ECG_Datadeal(data_path):
    # 1) 格式转换：mat/npy ——> npy
    data_npy = to_numpy(data_path)

    # 2) 滤波去噪
    data_filter = ECG_Filter(data_npy , 400)

    # 3) Z-score标准化
    data_mean = np.mean(data_filter)
    data_std  = np.std(data_filter)
    data  = (data_filter - data_mean) / data_std

    print(f"数据处理完毕!")
    
    # 4) 保存数据并返回保存路径
    return save_processed_data(data, data_path)

if __name__ == "__main__":
    train_path = os.path.join("data", "train", "traindata.mat")
    test_path = os.path.join("data", "test", "testdata.mat")
    ECG_Datadeal(train_path)
    ECG_Datadeal(test_path)
