import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
import os

# ==========================================
# [配置] 顶级学术期刊绘图风格 (IEEE / SCI)
# ==========================================
def set_paper_style():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
    plt.rcParams['mathtext.fontset'] = 'stix' 
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    
set_paper_style()

# 数据文件路径
file_gyro_only = "nav_dead_1.6s_gyro_only.csv"
file_gyro_acc = "nav_dead_1.6s_gyro_acc.csv"

# 统一输出目录
output_dir = "figs_paper_final"
os.makedirs(output_dir, exist_ok=True)

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"[错误] 找不到文件: {filepath}")
        return None
    df = pd.read_csv(filepath)
    df.columns = [col.strip() for col in df.columns]
    df['time_h'] = df['time'] / 3600.0
    return df[df['time_h'] <= 24.0].copy()

def main():
    print("Loading datasets...")
    df_only = load_data(file_gyro_only)
    df_full = load_data(file_gyro_acc)

    if df_only is None or df_full is None:
        print("数据缺失，无法绘图。")
        return

    # =========================================================================
    # Figure 3: 消融实验 (拆分 East Error 和 North Error)
    # =========================================================================
    print("Generating Figure 3 (Schuler Oscillation East/North)...")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    
    # 画东向和北向，不再合成为 Drift
    ax3.plot(df_only['time_h'], df_only['lon_err'], color='#D62728', alpha=0.9, label='East Error')
    ax3.plot(df_only['time_h'], df_only['lat_err'], color='#1F77B4', alpha=0.9, label='North Error')
    
    ax3.set_xlabel('Time (h)', fontweight='bold')
    ax3.set_ylabel('Position Error (m)', fontweight='bold')
    ax3.set_xlim(0, 24)
    ax3.legend(loc='upper right', framealpha=0.9, edgecolor='black')
    
    plt.tight_layout()
    fig3.savefig(f"{output_dir}/fig3_schuler_oscillation.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # =========================================================================
    # Figure 4: 全状态平滑对比 (2D平面 + 1D微观放大)
    # =========================================================================
    print("Generating Figure 4 (Performance Comparison with Micro-Zoom)...")
    fig4, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    
    label_gyro_only = "ES-FGO (CAIG Only)"
    label_gyro_acc = "ES-FGO (CAIG + CAIA)"

    # --- Subplot (a): 2D 平面位置误差 ---
    ax_2d = axes[0]
    ax_2d.plot(df_only['lon_err'], df_only['lat_err'], color='#D62728', alpha=0.8, label=label_gyro_only)
    ax_2d.plot(df_full['lon_err'], df_full['lat_err'], color='#1F77B4', alpha=0.9, label=label_gyro_acc)
    ax_2d.set_xlabel('East Error (m)', fontweight='bold')
    ax_2d.set_ylabel('North Error (m)', fontweight='bold')
    ax_2d.set_title('(a) 2D Horizontal Trajectory Error', fontsize=14, pad=10)
    ax_2d.set_aspect('equal', adjustable='datalim')
    ax_2d.legend(loc='upper right', framealpha=0.9)

    # --- Subplot (b): 综合位置漂移曲线 ---
    ax_time = axes[1]
    ax_time.plot(df_only['time_h'], df_only['drift'], color='#D62728', alpha=0.8, label=label_gyro_only)
    ax_time.plot(df_full['time_h'], df_full['drift'], color='#1F77B4', linewidth=2.5, label=label_gyro_acc)
    ax_time.set_xlabel('Time (h)', fontweight='bold')
    ax_time.set_ylabel('Position Drift (m)', fontweight='bold')
    ax_time.set_xlim(0, 24)
    ax_time.set_title('(b) Position Drift vs. Time', fontsize=14, pad=10)
    ax_time.legend(loc='upper right', framealpha=0.9)

    # 【神来之笔：微观级别的 Inset 放大】
    try:
        # 在图的上方中间放置放大图
        axins = inset_axes(ax_time, width="40%", height="35%", loc='upper center')
        
        # 精准定位：提取第 12.0 小时附近的 30 秒数据 (比如 43200 秒 ~ 43230 秒)
        zoom_start_sec = 12.0 * 3600
        zoom_end_sec = zoom_start_sec + 30.0 # 仅放大 30 秒，包含 15 个原子更新周期
        
        mask_ins_only = (df_only['time'] >= zoom_start_sec) & (df_only['time'] <= zoom_end_sec)
        mask_ins_full = (df_full['time'] >= zoom_start_sec) & (df_full['time'] <= zoom_end_sec)
        
        # 放大图的横坐标我们直接用“秒(s)”，这样更直观
        time_zoom_only = df_only['time'][mask_ins_only]
        time_zoom_full = df_full['time'][mask_ins_full]
        
        # 加入 marker='.' 可以看出你的数据记录频率
        axins.plot(time_zoom_only, df_only['drift'][mask_ins_only], color='#D62728', alpha=0.8, marker='o', markersize=2)
        axins.plot(time_zoom_full, df_full['drift'][mask_ins_full], color='#1F77B4', linewidth=2, marker='o', markersize=3)
        
        axins.set_xlim(zoom_start_sec, zoom_end_sec)
        
        # 自动调整Y轴缩放
        y_min = df_full['drift'][mask_ins_full].min() * 0.95
        y_max = df_only['drift'][mask_ins_only].max() * 1.05
        axins.set_ylim(y_min, y_max)
        
        # 取消放大图的坐标轴刻度数字，只看趋势
        axins.set_xticks([])
        axins.set_yticks([])
        axins.set_title("30s Micro-Zoom", fontsize=10, pad=3)
        axins.grid(True, linestyle=':', alpha=0.6)
        
        # 画出指示框 (连接放大图和主图)
        # 注意：主图的 x 轴是小时，放大图的 x 轴是秒，mark_inset 可能会画歪，
        # 所以我们这里直接通过纯粹的视觉框出大致位置即可，也可以注释掉下面这一行
        # mark_inset(ax_time, axins, loc1=3, loc2=4, fc="none", ec="0.5", lw=1)
        
    except Exception as e:
        print(f"画放大图时发生警告: {e}")

    plt.tight_layout()
    fig4.savefig(f"{output_dir}/fig4_smoothing_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)

    print(f"\n✅ 绘图完成！图片已保存在 '{output_dir}/' 目录下。")

if __name__ == "__main__":
    main()