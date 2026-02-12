# 采样初始状态的脚本
# 这个脚本会创建一个 OffScreenRenderEnv 环境，强制在 reset() 时随机生成初始状态，并采样指定数量的状态。
# ~/libero/scripts/sample_init_states.py 放置路径
import argparse
import os
import numpy as np
import torch
from libero.libero.envs import OffScreenRenderEnv
import robosuite.utils.transform_utils as T

def sample_init_states(bddl_file, num_samples, save_path):
    print(f"正在为 {bddl_file} 采样 {num_samples} 个初始状态...")
    
    # 创建环境，强制不使用预设的 init states，这样 reset() 时会随机生成新状态
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        control_freq=20,
    )

    init_states = []
    
    for i in range(num_samples):
        try:
            env.reset()
            # 获取当前的模拟器内部状态 (xml, data, etc.)
            # get_sim_state() 是 robosuite/libero 的底层 API
            state = env.sim.get_state().flatten()
            init_states.append(state)
            print(f"[Success] Sampled state {i+1}/{num_samples}")
        except Exception as e:
            print(f"[Error] Failed to sample state {i+1}: {e}")

    # 转换为 numpy 数组
    init_states = np.array(init_states)
    
    # 按照 Libero 的标准格式保存 (通常是 torch 保存)
    # 官方 init 文件通常是 torch.Tensor 或 numpy array
    # 这里我们保存为 torch 格式以保持兼容性
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(torch.from_numpy(init_states), save_path)
    
    print(f"完成！已保存到: {save_path}")
    print(f"状态形状: {init_states.shape}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bddl_file", type=str, required=True, help="BDDL 文件的绝对或相对路径")
    parser.add_argument("--num_samples", type=int, default=50, help="要采样的初始状态数量")
    parser.add_argument("--save_path", type=str, required=True, help="保存 .init 文件的路径")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.bddl_file):
        raise FileNotFoundError(f"找不到 BDDL 文件: {args.bddl_file}")
        
    sample_init_states(args.bddl_file, args.num_samples, args.save_path)