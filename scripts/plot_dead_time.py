import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re

# ==========================================
# [配置] 绘图风格
# ==========================================
def set_paper_style():
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['lines.linewidth'] = 2.0
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.4
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['axes.unicode_minus'] = False 
    
    fonts = ['WenQuanYi Micro Hei', 'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    for f in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [f]
            break
        except:
            continue
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD'])

# ==========================================
# [核心] 数据清洗
# ==========================================
def process_dataframe(df, cycle_time=2.0):
    # 1. 清洗列名
    df.columns = df.columns.str.strip()
    
    # 2. 智能降采样 (防止死区残差被重复累计)
    mask = (np.abs(df['time'] % cycle_time) < 0.01) | (np.abs(df['time'] % cycle_time - cycle_time) < 0.01)
    df_resampled = df[mask].drop_duplicates(subset='time').copy()
    
    # =======================================================
    # [兼容性补丁] 适配老版本 CSV (res_int_x -> res_gyro_x)
    # =======================================================
    if 'res_int_x' in df_resampled.columns and 'res_gyro_x' not in df_resampled.columns:
        print("  -> 检测到老版本格式 (res_int_*)，自动映射为陀螺残差。")
        df_resampled['res_gyro_x'] = df_resampled['res_int_x']
        df_resampled['res_gyro_y'] = df_resampled['res_int_y']
        df_resampled['res_gyro_z'] = df_resampled['res_int_z']
    
    # 3. 陀螺累计残差 (弧度 -> 角秒 ″)
    if 'res_gyro_x' in df_resampled.columns:
        rad2arcsec = (180.0 / np.pi) * 3600.0
        df_resampled['cum_gyro_x'] = df_resampled['res_gyro_x'].cumsum() * rad2arcsec
        df_resampled['cum_gyro_y'] = df_resampled['res_gyro_y'].cumsum() * rad2arcsec
        df_resampled['cum_gyro_z'] = df_resampled['res_gyro_z'].cumsum() * rad2arcsec
    
    # 4. 加计累计残差 (m/s)
    if 'res_acc_x' in df_resampled.columns:
        df_resampled['cum_acc_x'] = df_resampled['res_acc_x'].cumsum()
        df_resampled['cum_acc_y'] = df_resampled['res_acc_y'].cumsum()
        df_resampled['cum_acc_z'] = df_resampled['res_acc_z'].cumsum()

    return df, df_resampled

# ==========================================
# [主程序] 绘图逻辑
# ==========================================
def main():
    set_paper_style()
    
    search_pattern = "nav_dead_*.csv"
    files = glob.glob(search_pattern)
    files.sort()
    
    print(f"=== 死区效应分析绘图工具 (兼容版) ===")
    
    if not files:
        print(f"未找到数据文件 ({search_pattern})")
        return

    datasets = []
    print(f"正在处理 {len(files)} 个文件...")

    for f in files:
        try:
            raw_df = pd.read_csv(f)
            match = re.search(r"nav_dead_(\d+\.?\d*)s", f)
            if match:
                t_active = float(match.group(1))
                t_dead = 2.0 - t_active
                label = f"死区 {t_dead:.1f}s (工作 {t_active}s)"
                sort_key = t_dead
            else:
                label = f
                sort_key = 0
            
            full_df, resampled_df = process_dataframe(raw_df)
            
            datasets.append({
                'label': label,
                'full': full_df,
                'resampled': resampled_df,
                'sort': sort_key
            })
            
        except Exception as e:
            print(f"处理 {f} 失败: {e}")

    datasets.sort(key=lambda x: x['sort'], reverse=True)
    
    output_dir = "figs_dead_time"
    if not os.path.exists(output_dir): os.makedirs(output_dir)

    # -----------------------------------------
    # 图 1: 位置误差对比 (始终绘制)
    # -----------------------------------------
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ds in datasets:
        df = ds['full']
        t = df['time'] / 3600.0
        ax1.plot(t, df['lat_err'], label=ds['label'])
        ax2.plot(t, df['lon_err'], label=ds['label'])
    
    ax1.set_title('位置误差对比')
    ax1.set_ylabel('北向误差 (m)')
    ax1.legend(loc='upper left', ncol=2)
    ax2.set_ylabel('东向误差 (m)')
    ax2.set_xlabel('时间 (h)')
    ax2.set_xlim(0, 24)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_位置误差.png", dpi=300)
    print(f"已生成: {output_dir}/1_位置误差.png")

    # -----------------------------------------
    # 图 2: 陀螺累计残差 (如果存在)
    # -----------------------------------------
    if any('cum_gyro_x' in d['resampled'].columns for d in datasets):
        fig2, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        for ds in datasets:
            df = ds['resampled']
            if 'cum_gyro_x' not in df.columns: continue
            
            t = df['time'] / 3600.0
            axes[0].plot(t, df['cum_gyro_x'], label=ds['label'])
            axes[1].plot(t, df['cum_gyro_y'], label=ds['label'])
            axes[2].plot(t, df['cum_gyro_z'], label=ds['label'])
            
        axes[0].set_title('陀螺盲区累计残差')
        axes[0].set_ylabel('X轴角度漂移 (″)')
        axes[0].legend(loc='upper left')
        axes[1].set_ylabel('Y轴角度漂移 (″)')
        axes[2].set_ylabel('Z轴角度漂移 (″)')
        axes[2].set_xlabel('时间 (h)')
        axes[2].set_xlim(0, 24)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/2_陀螺残差.png", dpi=300)
        print(f"已生成: {output_dir}/2_陀螺残差.png")
    else:
        print("未找到陀螺残差数据，跳过图2绘制。")

    # -----------------------------------------
    # 图 3: 加计累计残差 (如果存在)
    # -----------------------------------------
    if any('cum_acc_x' in d['resampled'].columns for d in datasets):
        fig3, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        for ds in datasets:
            df = ds['resampled']
            if 'cum_acc_x' not in df.columns: continue
            
            t = df['time'] / 3600.0
            axes[0].plot(t, df['cum_acc_x'], label=ds['label'])
            axes[1].plot(t, df['cum_acc_y'], label=ds['label'])
            axes[2].plot(t, df['cum_acc_z'], label=ds['label'])
            
        axes[0].set_title('加计盲区累计残差')
        axes[0].set_ylabel('X轴速度漂移 (m/s)')
        axes[0].legend(loc='upper left')
        axes[1].set_ylabel('Y轴速度漂移 (m/s)')
        axes[2].set_ylabel('Z轴速度漂移 (m/s)')
        axes[2].set_xlabel('时间 (h)')
        axes[2].set_xlim(0, 24)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/3_加计残差.png", dpi=300)
        print(f"已生成: {output_dir}/3_加计残差.png")
    else:
        print("未找到加计残差数据 (老版本可能无此项)，跳过图3绘制。")

if __name__ == "__main__":
    main()


# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # ==========================================
# # [配置] 绘图风格 (符合 SCI 论文规范)
# # ==========================================
# def set_paper_style():
#     plt.rcParams['font.size'] = 12
#     plt.rcParams['axes.labelsize'] = 14
#     plt.rcParams['axes.titlesize'] = 16
#     plt.rcParams['legend.fontsize'] = 12
#     plt.rcParams['xtick.labelsize'] = 12
#     plt.rcParams['ytick.labelsize'] = 12
#     plt.rcParams['lines.linewidth'] = 2.0
#     plt.rcParams['axes.grid'] = True
#     plt.rcParams['grid.alpha'] = 0.4
#     plt.rcParams['grid.linestyle'] = '--'
#     plt.rcParams['axes.unicode_minus'] = False 
    
#     # 尝试加载常用字体
#     fonts = ['Times New Roman', 'DejaVu Sans', 'Arial']
#     for f in fonts:
#         try:
#             plt.rcParams['font.family'] = f
#             break
#         except:
#             continue
#     # 使用比较学术的配色方案
#     plt.rcParams['axes.prop_cycle'] = plt.cycler(color=['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD'])

# # ==========================================
# # [核心] 数据加载与标签映射
# # ==========================================
# def load_datasets():
#     """
#     在这里手动配置你要画的 csv 文件路径，以及它在论文图例中的名字。
#     键 (Key)   : csv 文件的相对或绝对路径
#     值 (Value) : 显示在图表上的标签 (Legend)
#     """
#     target_files = {
#         # 如果需要加入纯光纤作为 Baseline，取消下面这行的注释即可
#         # "nav_30h_final.csv": "Pure FOG (Baseline)", 
        
#         "nav_dead_1.6s_gyro_only.csv": "ES-FGO (CAIG Only)",
#         "nav_dead_1.6s_gyro_acc.csv": "ES-FGO (CAIG + CAIA)"
#     }

#     datasets = []
#     for file_path, label in target_files.items():
#         if os.path.exists(file_path):
#             print(f"[Load] 成功加载文件: {file_path} -> 标签: {label}")
#             df = pd.read_csv(file_path)
            
#             # 清洗列名，去除多余空格
#             df.columns = [col.strip() for col in df.columns]
            
#             datasets.append({
#                 'label': label,
#                 'df': df
#             })
#         else:
#             print(f"[Warning] 未找到文件，跳过: {file_path}")
            
#     return datasets

# # ==========================================
# # [绘图] 主函数
# # ==========================================
# def main():
#     set_paper_style()
#     output_dir = "figs_paper"
#     os.makedirs(output_dir, exist_ok=True)
    
#     datasets = load_datasets()
#     if not datasets:
#         print("没有成功加载任何数据，退出程序。")
#         return

#     # -----------------------------------------
#     # 图 1: 核心对比 - 24小时位置漂移 (舒拉振荡)
#     # -----------------------------------------
#     fig1, ax1 = plt.subplots(figsize=(8, 6))
    
#     for ds in datasets:
#         df = ds['df']
#         if 'time' not in df.columns or 'drift' not in df.columns:
#             continue
        
#         time_hours = df['time'] / 3600.0
#         # 仅截取 24 小时内的数据
#         mask = time_hours <= 24.0
        
#         ax1.plot(time_hours[mask], df['drift'][mask], label=ds['label'])
        
#     ax1.set_xlabel('Time (h)')
#     ax1.set_ylabel('Position Drift (m)')
#     ax1.set_xlim(0, 24)
#     # ax1.set_ylim(0, 800) # 根据需要取消注释以固定 Y 轴
#     ax1.legend(loc='upper left')
    
#     plt.tight_layout()
#     plt.savefig(f"{output_dir}/Fig_Position_Drift.png", dpi=300)
#     print(f"[Save] 已保存图表: {output_dir}/Fig_Position_Drift.png")

#     # -----------------------------------------
#     # 图 2: 姿态误差对比 (如果有需要)
#     # -----------------------------------------
#     if all(col in datasets[0]['df'].columns for col in ['pitch', 'roll', 'yaw']):
#         fig2, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
        
#         for ds in datasets:
#             df = ds['df']
#             time_hours = df['time'] / 3600.0
#             mask = time_hours <= 24.0
            
#             # 乘以 3600 转为角秒 ('')，或者保留度 (deg)
#             axes[0].plot(time_hours[mask], df['pitch'][mask] * 3600, label=ds['label'])
#             axes[1].plot(time_hours[mask], df['roll'][mask] * 3600, label=ds['label'])
#             axes[2].plot(time_hours[mask], df['yaw'][mask] * 3600, label=ds['label'])
            
#         axes[0].set_ylabel('Pitch Error (arcsec)')
#         axes[1].set_ylabel('Roll Error (arcsec)')
#         axes[2].set_ylabel('Yaw Error (arcsec)')
#         axes[2].set_xlabel('Time (h)')
#         axes[2].set_xlim(0, 24)
#         axes[0].legend(loc='upper right')
        
#         plt.tight_layout()
#         plt.savefig(f"{output_dir}/Fig_Attitude_Error.png", dpi=300)
#         print(f"[Save] 已保存图表: {output_dir}/Fig_Attitude_Error.png")

# if __name__ == "__main__":
#     main()