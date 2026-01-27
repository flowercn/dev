import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import os

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
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

# ==========================================
# [工具] 智能文件选择 (GUI -> CLI 自动降级)
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
        pass # GUI 失败则静默进入 CLI 模式

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
# [核心] 数据标准化 (处理各种命名差异)
# ==========================================
def normalize_dataframe(df):
    # 1. 去除空格
    df.columns = df.columns.str.strip()
    
    # 2. 定义同义词映射表 {标准名: [可能的别名]}
    alias_map = {
        'time': ['t', 'Time', 'sec'],
        'lat': ['Latitude', 'Lat'],
        'lon': ['Longitude', 'Lon'],
        'h':   ['Height', 'Alt', 'H'],
        'vn':  ['vN', 'VN', 'vel_n'],
        've':  ['vE', 'VE', 'vel_e'],
        'vd':  ['vD', 'VD', 'vel_d'],
        'roll': ['Roll', 'phi'],
        'pitch':['Pitch', 'theta'],
        'yaw':  ['Yaw', 'psi'],
        # 零偏别名
        'bg_x': ['ebX', 'wb_x', 'gyro_bias_x'],
        'bg_y': ['ebY', 'wb_y', 'gyro_bias_y'],
        'bg_z': ['ebZ', 'wb_z', 'gyro_bias_z'],
        'ba_x': ['dbX', 'fb_x', 'acc_bias_x'],
        'ba_y': ['dbY', 'fb_y', 'acc_bias_y'],
        'ba_z': ['dbZ', 'fb_z', 'acc_bias_z'],
    }

    # 3. 执行重命名
    for std_name, aliases in alias_map.items():
        if std_name in df.columns: continue # 已经是标准名
        for alias in aliases:
            if alias in df.columns:
                print(f"  [Mapping] '{alias}' -> '{std_name}'")
                df.rename(columns={alias: std_name}, inplace=True)
                break
    
    # 4. 特殊处理：vU (Up) -> vd (Down)
    if 'vU' in df.columns and 'vd' not in df.columns:
        print("  [Mapping] 'vU' -> 'vd' (Inverted sign)")
        df['vd'] = -df['vU']
    
    return df

# ==========================================
# [工具] 经纬度转平面坐标
# ==========================================
def llh_to_enu(df):
    if not {'lat', 'lon', 'h'}.issubset(df.columns): return df
    
    lat0, lon0 = df['lat'].iloc[0] * np.pi/180.0, df['lon'].iloc[0] * np.pi/180.0
    Re, e2 = 6378137.0, 0.00669437999014
    slat = np.sin(lat0)
    RM = Re * (1 - e2) / np.power(1 - e2 * slat**2, 1.5)
    RN = Re / np.sqrt(1 - e2 * slat**2)

    df['north_m'] = (df['lat'] * np.pi/180.0 - lat0) * (RM + df['h'])
    df['east_m']  = (df['lon'] * np.pi/180.0 - lon0) * (RN + df['h']) * np.cos(lat0)
    return df

# ==========================================
# [绘图] 主逻辑
# ==========================================
def plot_smart(file_path):
    if not file_path or not os.path.exists(file_path): return

    print(f"\nReading {os.path.basename(file_path)}...")
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading CSV: {e}"); return

    # --- 关键步骤：标准化列名 ---
    df = normalize_dataframe(df)
    cols = set(df.columns)
    print(f"Normalized columns: {list(cols)}")

    # 准备目录
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = f"{base_name}_figs"
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    set_paper_style()

    # 时间轴
    if 'time' in df.columns: t = df['time']
    else: t = df.index
    t_rel = t - t.iloc[0]

    # --- 1. 轨迹与位置误差 ---
    if {'lat', 'lon', 'h'}.issubset(cols):
        print("Plotting: Position...")
        df = llh_to_enu(df)
        
        # Fig 1: 轨迹
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(df['east_m'], df['north_m'], c=t_rel, cmap='viridis', s=5, rasterized=True)
        ax.plot(df['east_m'].iloc[0], df['north_m'].iloc[0], 'rx', markersize=10, mew=2, label='Start')
        ax.plot(df['east_m'].iloc[-1], df['north_m'].iloc[-1], 'k+', markersize=10, mew=2, label='End')
        ax.set_xlabel('East Error (m)'); ax.set_ylabel('North Error (m)')
        ax.set_title('Horizontal Position Drift')
        ax.axis('equal'); plt.colorbar(sc, label='Time (s)')
        plt.legend(); plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_traj.png", dpi=300); plt.close()

        # Fig 2: 位置时序
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        ax1.plot(t_rel, df['north_m'], label='North'); ax1.set_ylabel('North (m)'); ax1.legend(loc='upper right')
        ax2.plot(t_rel, df['east_m'], 'tab:orange', label='East'); ax2.set_ylabel('East (m)'); ax2.legend(loc='upper right')
        ax2.set_xlabel('Time (s)'); plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_pos_err.png", dpi=300); plt.close()

    # --- 2. 速度 ---
    if {'vn', 've', 'vd'}.issubset(cols):
        print("Plotting: Velocity...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
        ax1.plot(t_rel, df['vn']); ax1.set_ylabel(r'$V_N$ (m/s)')
        ax2.plot(t_rel, df['ve'], 'tab:orange'); ax2.set_ylabel(r'$V_E$ (m/s)')
        ax3.plot(t_rel, df['vd'], 'tab:green'); ax3.set_ylabel(r'$V_D$ (m/s)')
        ax3.set_xlabel('Time (s)'); plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_vel.png", dpi=300); plt.close()
    else:
        print(f"Skipping Velocity (Missing vn/ve/vd). Found: {[c for c in cols if 'v' in c.lower()]}")

    # --- 3. 姿态 ---
    if {'roll', 'pitch', 'yaw'}.issubset(cols):
        print("Plotting: Attitude...")
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 6), sharex=True)
        ax1.plot(t_rel, df['roll'], 'tab:red'); ax1.set_ylabel('Roll (deg)')
        ax2.plot(t_rel, df['pitch'], 'tab:green'); ax2.set_ylabel('Pitch (deg)')
        ax3.plot(t_rel, df['yaw']); ax3.set_ylabel('Yaw (deg)')
        ax3.set_xlabel('Time (s)'); plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_att.png", dpi=300); plt.close()

    # --- 4. 零偏 ---
    has_gb = {'bg_x', 'bg_y', 'bg_z'}.issubset(cols)
    has_ab = {'ba_x', 'ba_y', 'ba_z'}.issubset(cols)
    if has_gb or has_ab:
        print("Plotting: Bias...")
        rows = (1 if has_gb else 0) + (1 if has_ab else 0)
        fig, axes = plt.subplots(rows, 1, figsize=(7, 3*rows), sharex=True)
        if rows == 1: axes = [axes]
        idx = 0
        if has_gb:
            axes[idx].plot(t_rel, df['bg_x'], label='X'); axes[idx].plot(t_rel, df['bg_y'], label='Y'); axes[idx].plot(t_rel, df['bg_z'], label='Z')
            axes[idx].set_ylabel(r'Gyro Bias ($^\circ$/h)'); axes[idx].legend(ncol=3)
            idx += 1
        if has_ab:
            axes[idx].plot(t_rel, df['ba_x'], label='X'); axes[idx].plot(t_rel, df['ba_y'], label='Y'); axes[idx].plot(t_rel, df['ba_z'], label='Z')
            axes[idx].set_ylabel(r'Acc Bias ($\mu g$)'); axes[idx].legend(ncol=3)
        axes[-1].set_xlabel('Time (s)'); plt.tight_layout()
        plt.savefig(f"{output_dir}/fig_bias.png", dpi=300); plt.close()
    else:
        # Debug info for bias
        print(f"Skipping Bias. Check cols. Gyro-like found: {[c for c in cols if 'eb' in c or 'bg' in c]}")

    print(f"\n✅ Done! Figures saved to '{output_dir}/'")

if __name__ == "__main__":
    path = None
    if len(sys.argv) > 1: path = sys.argv[1]
    if not path: path = select_file_smart()
    plot_smart(path)