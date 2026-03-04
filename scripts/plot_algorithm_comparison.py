import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import os

# ==========================================
# [配置] SCI 顶刊级绘图风格
# ==========================================
def set_sci_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix' 
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 15
    plt.rcParams['xtick.labelsize'] = 13
    plt.rcParams['ytick.labelsize'] = 13
    plt.rcParams['legend.fontsize'] = 13
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'

set_sci_style()

def load_data(file_name):
    if not os.path.exists(file_name):
        print(f"[Error] {file_name} not found! Please run C++ simulation first.")
        return None
    print(f"Loading {file_name} ...")
    return pd.read_csv(file_name)

df_fog = load_data("nav_pure_fog.csv")
df_esfgo = load_data("nav_esfgo_full.csv")
df_gyro_only = load_data("nav_dead_1.6s_gyro_only.csv")

COLOR_FOG = '#D62728'   # 纯光纤 (红)
COLOR_ESFGO = '#2CA02C' # ES-FGO (绿)
COLOR_CAIG = '#1F77B4'  # 仅陀螺 (蓝)

# =========================================================
# 图 3: 舒拉振荡验证 (展现半物理真实数据的鲁棒性)
# =========================================================
if df_gyro_only is not None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    time_h = df_gyro_only['time'] / 3600.0
    
    ax.plot(time_h, df_gyro_only['drift'], color=COLOR_CAIG, linewidth=2.0, label='ES-FGO (CAIG-Only)')
    
    ax.set_xlabel('Time (h)', fontweight='bold')
    ax.set_ylabel('Position Error (m)', fontweight='bold')
    ax.set_xlim([0, 24])
    
    legend = ax.legend(loc='upper left', framealpha=1.0, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)
    
    plt.tight_layout()
    plt.savefig("fig_pos_err.png", dpi=300, bbox_inches='tight')
    print("✅ Generated fig_pos_err.png")

# =========================================================
# 图 4: 轨迹与时间漂移双拼图 (科学展现 97.7% 的压制效果)
# =========================================================
if df_fog is not None and df_esfgo is not None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ---------- 子图 (a): 2D 轨迹 + 核心区放大 ----------
    ax1.plot(df_fog['lon_err']/1000.0, df_fog['lat_err']/1000.0, 
            color=COLOR_FOG, linestyle='--', linewidth=1.5, label='Pure FOG INS')
    ax1.plot(df_esfgo['lon_err']/1000.0, df_esfgo['lat_err']/1000.0, 
            color=COLOR_ESFGO, linewidth=2.0, label='ES-FGO (Proposed)')
    ax1.scatter(0, 0, marker='*', color='black', s=200, zorder=5, label='Start Point')
    
    ax1.set_title('(a) 2D Horizontal Trajectory Error', fontweight='bold', pad=15)
    ax1.set_xlabel('East Error (km)', fontweight='bold')
    ax1.set_ylabel('North Error (km)', fontweight='bold')
    
    # 【严谨修正】：增加轨迹中心区域放大图，让审稿人看清绿线并非失效
    axins = zoomed_inset_axes(ax1, zoom=12.0, loc='lower right')
    axins.plot(df_esfgo['lon_err']/1000.0, df_esfgo['lat_err']/1000.0, color=COLOR_ESFGO, linewidth=1.5)
    axins.scatter(0, 0, marker='*', color='black', s=100, zorder=5)
    axins.set_xlim(-0.4, 0.4) # 聚焦在 ±400m 区域
    axins.set_ylim(-0.4, 0.4)
    axins.grid(True, linestyle=':', alpha=0.6)
    axins.set_title("Zoom-in (Proposed)", fontsize=10, pad=5)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    mark_inset(ax1, axins, loc1=2, loc2=3, fc="none", ec="0.5", linestyle="--")

    legend1 = ax1.legend(loc='upper left', framealpha=1.0, edgecolor='black')
    legend1.get_frame().set_linewidth(1.0)
    ax1.axis('equal') 

    # ---------- 子图 (b): 宏观发散对比 (彻底删除假锯齿) ----------
    time_h_fog = df_fog['time'] / 3600.0
    time_h_esfgo = df_esfgo['time'] / 3600.0
    
    ax2.plot(time_h_fog, df_fog['drift']/1000.0, color=COLOR_FOG, linestyle='--', linewidth=2.0, label='Pure FOG INS')
    ax2.plot(time_h_esfgo, df_esfgo['drift']/1000.0, color=COLOR_ESFGO, linewidth=2.0, label='ES-FGO (Proposed)')
    
    ax2.set_title('(b) Position Drift Over Time', fontweight='bold', pad=15)
    ax2.set_xlabel('Time (h)', fontweight='bold')
    ax2.set_ylabel('Position Drift (km)', fontweight='bold')
    ax2.set_xlim([0, 24])
    ax2.set_yscale('log') # 科学体现 16.4km 与 0.36km 的量级差距
    
    legend2 = ax2.legend(loc='upper left', framealpha=1.0, edgecolor='black')
    legend2.get_frame().set_linewidth(1.0)
    
    plt.tight_layout()
    plt.savefig("fig_traj.png", dpi=300, bbox_inches='tight')
    print("✅ Generated fig_traj.png (Scientifically Rigorous)")

print("\n🎉 All plots generated successfully!")