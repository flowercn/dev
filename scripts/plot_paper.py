import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import os
import glob
import re

# ==========================================
# [配置] 论文级绘图风格
# ==========================================
def set_paper_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'Liberation Serif', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix' 
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#D62728', '#2CA02C', '#1F77B4', '#FF7F0E', '#9467BD', '#8C564B'])

# ==========================================
# [工具] 智能文件选择
# ==========================================
def select_file_smart():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw() 
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(
            title="Select CSV Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        root.destroy()
        if file_path: return file_path
    except:
        pass 

    import glob
    csv_files = glob.glob("*.csv")
    csv_files.sort()
    
    print("\nFound CSV files in current directory:")
    if not csv_files:
        print("  (No .csv files found)")
    else:
        for i, f in enumerate(csv_files):
            print(f"  [{i+1}] {f}")
    
    while True:
        user_input = input("\nEnter file number or full path: ").strip()
        if not user_input: return None
        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(csv_files): return csv_files[idx]
        elif os.path.exists(user_input):
            return user_input
        else:
            print("Invalid input.")

# ==========================================
# [核心] 数据标准化
# ==========================================
def normalize_dataframe(df):
    df.columns = df.columns.str.strip()
    
    # 别名映射
    alias_map = {
        'time': ['t', 'Time', 'sec'],
        'north_m': ['lat_err', 'North', 'north'], 
        'east_m':  ['lon_err', 'East', 'east'],
        'h_m':     ['h_err', 'Height', 'Alt'],
        'drift':   ['Drift', 'pos_err'],
        'vn':  ['vN', 'VN', 'vel_n'],
        've':  ['vE', 'VE', 'vel_e'],
        'vd':  ['vD', 'VD', 'vel_d', 'vU'],
        'roll': ['Roll', 'phi'],
        'pitch':['Pitch', 'theta'],
        'yaw':  ['Yaw', 'psi'],
        'bg_x': ['eb_x', 'ebX', 'gyro_bias_x'],
        'bg_y': ['eb_y', 'ebY', 'gyro_bias_y'],
        'bg_z': ['eb_z', 'ebZ', 'gyro_bias_z'],
        'ba_x': ['db_x', 'dbX', 'acc_bias_x'],
        'ba_y': ['db_y', 'dbY', 'acc_bias_y'],
        'ba_z': ['db_z', 'dbZ', 'acc_bias_z'],
    }

    for std_name, aliases in alias_map.items():
        if std_name in df.columns: continue
        for alias in aliases:
            if alias in df.columns:
                df.rename(columns={alias: std_name}, inplace=True)
                break
                
    if 'vU' in df.columns and 'vd' not in df.columns:
        df['vd'] = -df['vU']
        
    return df

# ==========================================
# [功能1] 单文件详细绘图
# ==========================================
def plot_smart_single(file_path):
    if not file_path or not os.path.exists(file_path): return

    print(f"\n[Single Mode] Processing {os.path.basename(file_path)}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}"); return

    df = normalize_dataframe(df)
    
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f"{base_name}_figs"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    set_paper_style()

    if 'time' in df.columns: t = df['time']
    else: t = df.index
    
    # 【修改点1】：将秒转换为小时
    t_rel = (t - t.iloc[0]) / 3600.0  

    # 1. 轨迹与位置
    if 'north_m' in df.columns and 'east_m' in df.columns:
        # Fig 1: 轨迹
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(df['east_m'], df['north_m'], c=t_rel, cmap='viridis', s=5, rasterized=True)
        ax.plot(df['east_m'].iloc[0], df['north_m'].iloc[0], 'rx', markersize=10, mew=2, label='Start')
        ax.plot(df['east_m'].iloc[-1], df['north_m'].iloc[-1], 'k+', markersize=10, mew=2, label='End')
        ax.set_xlabel('East Error (m)'); ax.set_ylabel('North Error (m)')
        ax.set_title('Horizontal Position Drift')
        ax.axis('equal'); 
        # 【修改点2】：Colorbar 标签改为 Time (h)
        plt.colorbar(sc, label='Time (h)')
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_traj.png", dpi=300); plt.close()

        # Fig 2: 位置三轴
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), sharex=True)
        ax1.plot(t_rel, df['north_m'], label='North'); ax1.set_ylabel('North (m)'); 
        ax2.plot(t_rel, df['east_m'], 'tab:orange', label='East'); ax2.set_ylabel('East (m)'); 
        # 【修改点3】：X轴标签改为 Time (h)
        ax2.set_xlabel('Time (h)')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_pos_err.png", dpi=300); plt.close()

    # 2. 速度三轴
    if {'vn', 've', 'vd'}.issubset(df.columns):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
        ax1.plot(t_rel, df['vn']); ax1.set_ylabel(r'$V_N$ (m/s)')
        ax2.plot(t_rel, df['ve'], 'tab:orange'); ax2.set_ylabel(r'$V_E$ (m/s)')
        ax3.plot(t_rel, df['vd'], 'tab:green'); ax3.set_ylabel(r'$V_D$ (m/s)')
        # 【修改点4】：X轴标签改为 Time (h)
        ax3.set_xlabel('Time (h)'); plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_vel.png", dpi=300); plt.close()

    # 3. 姿态三轴
    if {'roll', 'pitch', 'yaw'}.issubset(df.columns):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
        ax1.plot(t_rel, df['roll'], 'tab:red'); ax1.set_ylabel('Roll (deg)')
        ax2.plot(t_rel, df['pitch'], 'tab:green'); ax2.set_ylabel('Pitch (deg)')
        ax3.plot(t_rel, df['yaw']); ax3.set_ylabel('Yaw (deg)')
        # 【修改点5】：X轴标签改为 Time (h)
        ax3.set_xlabel('Time (h)'); plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_att.png", dpi=300); plt.close()

    # 4. 零偏六轴
    has_gb = {'bg_x', 'bg_y', 'bg_z'}.issubset(df.columns)
    has_ab = {'ba_x', 'ba_y', 'ba_z'}.issubset(df.columns)
    
    if has_gb or has_ab:
        rows = (1 if has_gb else 0) + (1 if has_ab else 0)
        fig, axes = plt.subplots(rows, 1, figsize=(8, 4*rows), sharex=True)
        if rows == 1: axes = [axes]
        
        idx = 0
        if has_ab: # 加计
            axes[idx].plot(t_rel, df['ba_x'], label='X'); axes[idx].plot(t_rel, df['ba_y'], label='Y'); axes[idx].plot(t_rel, df['ba_z'], label='Z')
            axes[idx].set_ylabel(r'Acc Bias ($\mu g$)'); axes[idx].legend(ncol=3); axes[idx].set_title('Accelerometer Bias')
            idx += 1
        if has_gb: # 陀螺
            axes[idx].plot(t_rel, df['bg_x'], label='X'); axes[idx].plot(t_rel, df['bg_y'], label='Y'); axes[idx].plot(t_rel, df['bg_z'], label='Z')
            axes[idx].set_ylabel(r'Gyro Bias ($^\circ$/h)'); axes[idx].legend(ncol=3); axes[idx].set_title('Gyroscope Bias')
            
        # 【修改点6】：X轴标签改为 Time (h)
        axes[-1].set_xlabel('Time (h)'); plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_bias.png", dpi=300); plt.close()

    print(f"✅ [Single] Figures saved to '{output_dir}/'")

# ==========================================
# [功能2] 多组对比绘图 (全面增强)
# ==========================================
def plot_comparison_all(files):
    if not files: return
    
    print(f"\n[Comparison Mode] Processing {len(files)} files...")
    datasets = []
    
    for f in files:
        try:
            df = pd.read_csv(f)
            df = normalize_dataframe(df)
            label = os.path.basename(f).replace('.csv', '').replace('nav_res_', '').replace('nav_', '')
            if 'ug' in label: 
                val = label.replace('ug', '')
                label = fr"VRW={val}$\mu g$"
            elif 'final' in label:
                label = "Baseline"
            datasets.append({'label': label, 'df': df})
        except: pass
            
    output_dir = "comparison_figs"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    set_paper_style()

    # --- 1. 轨迹对比 ---
    fig, ax = plt.subplots(figsize=(8, 7))
    for ds in datasets:
        df = ds['df']
        if 'north_m' in df.columns:
            ax.plot(df['east_m'], df['north_m'], label=ds['label'], linewidth=2, alpha=0.8)
            ax.plot(df['east_m'].iloc[-1], df['north_m'].iloc[-1], 'o', markersize=5)
    ax.set_xlabel('East Error (m)'); ax.set_ylabel('North Error (m)')
    ax.set_title('Trajectory Comparison'); ax.axis('equal'); ax.legend()
    plt.savefig(f"{output_dir}/comp_traj.png", dpi=300); plt.close()

    # --- 2. 位置对比 (仅 North / East) ---
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ds in datasets:
        df = ds['df']; t = df['time'] / 3600.0
        # 北向
        axes[0].plot(t, df['north_m'], label=ds['label'])
        # 东向
        axes[1].plot(t, df['east_m'], label=ds['label'])

    axes[0].set_ylabel('North Error (m)'); axes[0].legend(loc='upper right')
    axes[1].set_ylabel('East Error (m)'); axes[1].set_xlabel('Time (h)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comp_pos.png", dpi=300); plt.close()

    # --- 3. 速度对比 (ENU) ---
    fig, axes = plt.subplots(2,1, figsize=(10, 10), sharex=True)
    for ds in datasets:
        df = ds['df']; t = df['time'] / 3600.0
        if 'vn' in df.columns: axes[0].plot(t, df['vn'], label=ds['label'])
        if 've' in df.columns: axes[1].plot(t, df['ve'], label=ds['label'])
    
    axes[0].set_ylabel(r'$V_N$ (m/s)'); axes[0].legend(loc='upper right')
    axes[1].set_ylabel(r'$V_E$ (m/s)'); axes[1].set_xlabel('Time (h)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comp_vel.png", dpi=300); plt.close()

    # --- 4. 姿态对比 (RPY 三轴) ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ds in datasets:
        df = ds['df']; t = df['time'] / 3600.0
        
        # 乘以 3600 将度转换为角秒
        if 'roll' in df.columns: axes[0].plot(t, df['roll'] * 3600.0, label=ds['label'])
        if 'pitch' in df.columns: axes[1].plot(t, df['pitch'] * 3600.0, label=ds['label'])
        if 'yaw' in df.columns: axes[2].plot(t, df['yaw'] * 3600.0, label=ds['label'])
    
    # 修改标签单位为 arcsec
    axes[0].set_ylabel('Roll (arcsec)'); axes[0].legend(loc='upper right')
    axes[1].set_ylabel('Pitch (arcsec)')
    axes[2].set_ylabel('Yaw (arcsec)'); axes[2].set_xlabel('Time (h)')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comp_att.png", dpi=300); plt.close()

    # --- 5. 零偏对比 (Acc 3轴 + Gyro 3轴) ---
    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    # 左列：加计 (ug)
    for ds in datasets:
        df = ds['df']; t = df['time'] / 3600.0
        if 'ba_x' in df.columns: axes[0,0].plot(t, df['ba_x'], label=ds['label'])
        if 'ba_y' in df.columns: axes[1,0].plot(t, df['ba_y'], label=ds['label'])
        if 'ba_z' in df.columns: axes[2,0].plot(t, df['ba_z'], label=ds['label'])
    
    axes[0,0].set_title('Accelerometer Bias (ug)'); axes[0,0].set_ylabel('X')
    axes[1,0].set_ylabel('Y'); axes[2,0].set_ylabel('Z'); axes[2,0].set_xlabel('Time (h)')
    axes[0,0].legend(loc='best')

    # 右列：陀螺 (deg/h)
    for ds in datasets:
        df = ds['df']; t = df['time'] / 3600.0
        if 'bg_x' in df.columns: axes[0,1].plot(t, df['bg_x'], label=ds['label'])
        if 'bg_y' in df.columns: axes[1,1].plot(t, df['bg_y'], label=ds['label'])
        if 'bg_z' in df.columns: axes[2,1].plot(t, df['bg_z'], label=ds['label'])

    axes[0,1].set_title('Gyroscope Bias (deg/h)'); axes[0,1].set_ylabel('X')
    axes[1,1].set_ylabel('Y'); axes[2,1].set_ylabel('Z'); axes[2,1].set_xlabel('Time (h)')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/comp_bias.png", dpi=300); plt.close()

    print(f"✅ [Comparison] All figures saved to '{output_dir}/'")

# ==========================================
# [入口] 智能判断模式
# ==========================================
if __name__ == "__main__":
    comp_files = glob.glob("nav_res_*.csv")
    comp_files.sort()

    print("\n========================================")
    print("      Hybrid Navigation Plotter         ")
    print("========================================")

    if len(comp_files) > 1:
        print(f" Detected {len(comp_files)} comparison datasets:")
        for f in comp_files: print(f"  - {f}")
        print("\n Options:")
        print(" [1] Analyze Single File (Detail Mode)")
        print(" [2] Compare All Datasets (Full 6-Axis Comparison)")
        
        choice = input("\nSelect Option [2]: ").strip()
        if choice == '1':
            path = select_file_smart()
            plot_smart_single(path)
        else:
            plot_comparison_all(comp_files)
    else:
        path = None
        if len(sys.argv) > 1: path = sys.argv[1]
        if not path: path = select_file_smart()
        plot_smart_single(path)