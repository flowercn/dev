#!/usr/bin/env python3
"""
验证 FOG 测量模长 vs 地球自转角速度

关键发现：CSV 中存的是角速度 (rad/s)，不是角增量！
C++ 代码读取后乘以 ts 得到角增量。
"""
import numpy as np
import pandas as pd

# 读取数据
csv_path = "/home/v/dev/hybrid_ins_cpp/fog3h.csv"
df = pd.read_csv(csv_path, header=None)

# CSV 数据格式: wx, wy, wz (rad/s), ax, ay, az (m/s^2)
w_raw = df.iloc[:, :3].values  # 角速度 (rad/s)

print(f"数据量: {len(w_raw)} 样本, {len(w_raw)/400/3600:.2f} 小时")

# FOG 平均角速度 (rad/s)
w_fog_avg = w_raw.mean(axis=0)
print("\n" + "=" * 60)
print("FOG 平均角速度:")
dph = 3600 / np.deg2rad(1)  # deg/h per rad/s
print(f"  wx = {w_fog_avg[0] * dph:.4f} deg/h")
print(f"  wy = {w_fog_avg[1] * dph:.4f} deg/h")  
print(f"  wz = {w_fog_avg[2] * dph:.4f} deg/h")

# FOG 模长
fog_norm = np.linalg.norm(w_fog_avg)
print(f"\n  |w_fog| = {fog_norm * dph:.4f} deg/h")

# 理论地球自转
wie = 7.2921151467e-5  # rad/s
print(f"\n理论地球自转:")
print(f"  wie = {wie * dph:.4f} deg/h")

# 南京位置的 wie 在 n 系分量
lat = 32.0 * np.deg2rad(1)  # 南京纬度
wie_n = np.array([0, wie * np.cos(lat), wie * np.sin(lat)])
print(f"  wie_n (deg/h): [{wie_n[0]*dph:.4f}, {wie_n[1]*dph:.4f}, {wie_n[2]*dph:.4f}]")

# 差异
diff = fog_norm - wie
print(f"\n模长差异:")
print(f"  |w_fog| - wie = {diff * dph:.4f} deg/h")
print(f"  相对误差 = {diff / wie * 100:.4f} %")

# 分量对比
print("\n" + "=" * 60)
print("分量对比 (假设姿态接近水平):")
print(f"  FOG_x  = {w_fog_avg[0] * dph:.4f} deg/h")
print(f"  FOG_y  = {w_fog_avg[1] * dph:.4f} deg/h  (理论 wie*cos(lat) = {wie_n[1]*dph:.4f})")
print(f"  FOG_z  = {w_fog_avg[2] * dph:.4f} deg/h  (理论 wie*sin(lat) = {wie_n[2]*dph:.4f})")

# 估算 FOG 零偏 (假设姿态完全水平，航向=0)
eb_x = w_fog_avg[0] - 0  # wie_n_x = 0
eb_y = w_fog_avg[1] - wie_n[1]
eb_z = w_fog_avg[2] - wie_n[2]
print(f"\n估算零偏 eb (假设水平姿态):")
print(f"  eb_x = {eb_x * dph:.4f} deg/h")
print(f"  eb_y = {eb_y * dph:.4f} deg/h")
print(f"  eb_z = {eb_z * dph:.4f} deg/h")
print(f"  |eb| = {np.sqrt(eb_x**2 + eb_y**2 + eb_z**2) * dph:.4f} deg/h")

print("\n" + "=" * 60)
print("结论:")
print(f"  FOG 测量的模长 = {fog_norm * dph:.4f} deg/h")
print(f"  地球自转模长   = {wie * dph:.4f} deg/h")
print(f"  差异 = {diff * dph:.4f} deg/h ({diff/wie*100:.2f}%)")
print("=" * 60)
