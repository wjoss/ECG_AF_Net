import os
import h5py
import scipy.io
import numpy as np
from typing import Optional


class MatConversionError(Exception):
     # MAT 文件转换到 NumPy 时发生的错误
    pass

def to_numpy(filepath: str,
             var_name: Optional[str] = None,
             save_path: Optional[str] = None,
             overwrite: bool = True
             ) -> np.ndarray:
    """
    功能：
        将 .mat 或 .npy 转为 .npy 并返回数组;如果输入是 .npy，则直接加载。
    参数:
        filepath: 输入文件路径（.mat 或 .npy）
        var_name: MAT 文件中要转换的变量名
        save_path: 输出 .npy 路径（默认与输入同名同目录）
        overwrite: 是否覆盖已存在文件
    返回:
        np.ndarray
    """
    filepath = os.path.abspath(filepath)
    ext = os.path.splitext(filepath)[1].lower()

    # 1) 直接加载 .npy
    if ext == '.npy':
        data = np.load(filepath)
        print(f"加载 {filepath} 文件"
              f"—— shape={data.shape} | dtype={data.dtype}")
        return data

    # 2) 确定输出路径
    if save_path is None:
        save_path = os.path.splitext(filepath)[0] + '.npy'
    else:
        save_path = os.path.abspath(save_path)
    if os.path.exists(save_path) and not overwrite:
        raise MatConversionError(f"保存路径已存在：{save_path}")

    # 3) 读取 MAT
    try:
        # 3.1 HDF5 v7.3+
        with h5py.File(filepath, 'r') as f:
            keys = [k for k, obj in f.items() if isinstance(obj, h5py.Dataset)]
            if var_name is None:
                if len(keys) != 1:
                    raise MatConversionError("HDF5 文件有多个数据集，请指定 var_name")
                var_name = keys[0]
            data = f[var_name][()]
            data = data.T  # 转置，与原数据相同
            print(f"读取 HDF5 格式文件{var_name}.mat成功!\n"
            f"—— 路径：{filepath}\n"
            f"—— shape={data.shape} | dtype={data.dtype}")

    except OSError:
        # 3.2 旧版 MAT
        mat = scipy.io.loadmat(filepath)
        user_vars = [k for k in mat if not k.startswith('__')]
        if var_name is None:
            if len(user_vars) != 1:
                raise MatConversionError("MAT 文件有多个变量，请指定 var_name")
            var_name = user_vars[0]
        if var_name not in user_vars:
            raise MatConversionError(f"变量不存在：{var_name}")
        data = mat[var_name]
        print(f"读取 旧版MAT 格式文件{var_name}.mat成功!\n"
              f"—— 路径：{filepath}\n"
              f"—— shape={data.shape} | dtype={data.dtype}")

    return data

if __name__ == "__main__":
    train_path = os.path.join("data", "train", "traindata.mat")
    test_path = os.path.join("data", "test", "testdata.mat")
    to_numpy(train_path)
    to_numpy(test_path)
