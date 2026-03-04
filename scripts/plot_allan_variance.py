#!/usr/bin/env python3
"""
Allan Variance Plot Generator
绘制陀螺和加计的Allan方差曲线，用于噪声特性分析
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import argparse

# ==========================================
# [配置] SCI 顶刊级绘图风格
# ==========================================
def set_sci_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix' 
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.top'] = True
    plt.rcParams['ytick.right'] = True

def read_data(path):
    """读取IMU数据文件 (gx, gy, gz, ax, ay, az)"""
    if not os.path.exists(path):
        if os.path.exists(os.path.basename(path)):
            path = os.path.basename(path)
        else:
            print(f"[Error] File not found: {path}")
            return None

    print(f"Loading {path} ...")
    data = []
    try:
        with open(path, 'r') as f:
            sample_line = f.readline()
            f.seek(0)
            if not sample_line: 
                return None
            
            use_comma = ',' in sample_line
            success_count = 0
            
            for i, line in enumerate(f):
                line = line.strip()
                if not line: 
                    continue
                if line[0].isalpha() or line.startswith('%') or line.startswith('#'): 
                    continue

                parts = line.split(',') if use_comma else line.split()
                try:
                    if len(parts) == 6:
                        # 格式: gx, gy, gz, ax, ay, az
                        gx, gy, gz = float(parts[0]), float(parts[1]), float(parts[2])
                        ax, ay, az = float(parts[3]), float(parts[4]), float(parts[5])
                    elif len(parts) >= 7:
                        # 格式: t, gx, gy, gz, ax, ay, az
                        gx, gy, gz = float(parts[1]), float(parts[2]), float(parts[3])
                        ax, ay, az = float(parts[4]), float(parts[5]), float(parts[6])
                    else:
                        continue
                    
                    data.append([gx, gy, gz, ax, ay, az])
                    success_count += 1
                except ValueError:
                    continue
            
            print(f"[Info] Loaded {success_count} lines.")
    except Exception as e:
        print(f"[Error] {e}")
        return None
    
    return np.array(data)

def simple_allan_variance(data, fs):
    """计算Allan方差"""
    N = len(data)
    if N < 100: 
        return [], []
    
    max_tau_power = int(np.log10(N/2))
    m_list = np.unique(np.logspace(0, max_tau_power, 50).astype(int))
    taus, adev = [], []
    
    print(f"Calculating Allan Variance ({N} pts)...", end='', flush=True)
    for m in m_list:
        tau = m / fs
        if m * 2 > N: 
            break
        
        n_sub = int(N // m)
        data_sub = data[:n_sub*m].reshape((n_sub, m))
        avg_rates = np.mean(data_sub, axis=1) 
        diffs = np.diff(avg_rates)            
        var = 0.5 * np.mean(diffs**2)
        dev = np.sqrt(var)
        taus.append(tau)
        adev.append(dev)
        print(".", end='', flush=True)
    
    print(" Done.")
    return np.array(taus), np.array(adev)

def plot_gyro_allan(data, fs, output_file="fig_allan_variance_gyro.png"):
    """绘制陀螺Allan方差曲线"""
    print("\n=== Plotting Gyro Allan Variance ===")
    set_sci_style()
    
    titles = ['Gyro X', 'Gyro Y', 'Gyro Z']
    colors = ['#D62728', '#2CA02C', '#1F77B4']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    results = []
    
    for i in range(3):
        # 陀螺数据：rad/s -> deg/h
        gyro_data = data[:, i] * (180.0 / math.pi)
        taus, adev = simple_allan_variance(gyro_data, fs)
        
        if len(taus) == 0: 
            continue
        
        adev_h = adev * 3600.0  # deg/s -> deg/h
        ax.loglog(taus, adev_h, label=titles[i], color=colors[i], alpha=0.9)
        
        # 计算指标
        idx_1s = np.abs(taus - 1.0).argmin()
        arw_val = adev_h[idx_1s] / 60.0  # ARW at tau=1s
        min_adev = np.min(adev_h)
        results.append({'axis': titles[i], 'arw': arw_val, 'bias_inst': min_adev})
    
    ax.set_xlabel(r'Averaging Time $\tau$ (s)', fontweight='bold')
    ax.set_ylabel(r'Allan Deviation $\sigma_A$ ($^\circ$/h)', fontweight='bold')
    ax.grid(True, which="major", ls="-", alpha=0.6)
    ax.grid(True, which="minor", ls=":", alpha=0.4)
    
    legend = ax.legend(loc='upper right', framealpha=1.0, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to '{output_file}'")
    
    # 打印统计结果
    print("\nGyro Statistics:")
    for r in results:
        print(f"  {r['axis']}: ARW = {r['arw']:.6f} deg/√h, Bias Instability = {r['bias_inst']:.6f} deg/h")

def plot_acc_allan(data, fs, output_file="fig_allan_variance_acc.png"):
    """绘制加计Allan方差曲线"""
    print("\n=== Plotting Accelerometer Allan Variance ===")
    set_sci_style()
    
    titles = ['Acc X', 'Acc Y', 'Acc Z']
    colors = ['#D62728', '#2CA02C', '#1F77B4']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    results = []
    
    for i in range(3):
        # 加计数据：m/s^2 -> ug
        acc_data = data[:, i+3] * (1e6 / 9.80665)
        taus, adev = simple_allan_variance(acc_data, fs)
        
        if len(taus) == 0: 
            continue
        
        ax.loglog(taus, adev, label=titles[i], color=colors[i], alpha=0.9)
        
        # 计算指标
        idx_1s = np.abs(taus - 1.0).argmin()
        vrw_val = adev[idx_1s]  # VRW at tau=1s
        min_adev = np.min(adev)
        results.append({'axis': titles[i], 'vrw': vrw_val, 'bias_inst': min_adev})
    
    ax.set_xlabel(r'Averaging Time $\tau$ (s)', fontweight='bold')
    ax.set_ylabel(r'Allan Deviation $\sigma_A$ ($\mu g$)', fontweight='bold')
    ax.grid(True, which="major", ls="-", alpha=0.6)
    ax.grid(True, which="minor", ls=":", alpha=0.4)
    
    legend = ax.legend(loc='upper right', framealpha=1.0, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to '{output_file}'")
    
    # 打印统计结果
    print("\nAccelerometer Statistics:")
    for r in results:
        print(f"  {r['axis']}: VRW = {r['vrw']:.4f} ug, Bias Instability = {r['bias_inst']:.4f} ug")

def main():
    parser = argparse.ArgumentParser(description='Generate Allan Variance plots for IMU data')
    parser.add_argument('-f', '--file', type=str, 
                        default='/home/v/dev/hybrid_ins_cpp/fog_part1.csv',
                        help='Input CSV file path')
    parser.add_argument('-s', '--fs', type=float, default=400.0,
                        help='Sampling rate (Hz)')
    parser.add_argument('-t', '--type', type=str, default='both',
                        choices=['gyro', 'acc', 'both'],
                        help='Plot type: gyro, acc, or both')
    parser.add_argument('-o', '--output-dir', type=str, default='.',
                        help='Output directory')
    
    args = parser.parse_args()
    
    # 读取数据
    data = read_data(args.file)
    if data is None or len(data) == 0:
        print("[Error] Failed to read data.")
        return 1
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 绘图
    if args.type in ['gyro', 'both']:
        output_file = os.path.join(args.output_dir, 'fig_allan_variance_gyro.png')
        plot_gyro_allan(data, args.fs, output_file)
    
    if args.type in ['acc', 'both']:
        output_file = os.path.join(args.output_dir, 'fig_allan_variance_acc.png')
        plot_acc_allan(data, args.fs, output_file)
    
    print("\n✅ All plots generated successfully!")
    return 0

if __name__ == "__main__":
    exit(main())
