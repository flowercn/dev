import numpy as np
import matplotlib.pyplot as plt
import math
import os

# --- 配置 ---
FILE_PATH = "../fog3h.csv" # 支持相对或绝对路径
FS = 400.0                 # 采样率 (Hz)

def read_data(path):
    # 路径检查
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
            # 预览第一行用于Debug
            sample_line = f.readline()
            f.seek(0)
            if not sample_line: return None
            
            # 判断分隔符 (你的转换脚本生成的是逗号分隔)
            use_comma = ',' in sample_line
            delimiter_name = "Comma" if use_comma else "Whitespace"
            print(f"[Debug] Format: {delimiter_name}, Preview: {sample_line.strip()[:50]}...")

            success_count = 0
            
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                # 跳过可能存在的表头
                if line[0].isalpha() or line.startswith('%') or line.startswith('#'): continue

                parts = line.split(',') if use_comma else line.split()
                
                # --- 核心适配逻辑 ---
                # 情况A: 6列数据 (gx, gy, gz, ax, ay, az) -> 你的情况
                # 情况B: 7列数据 (t, gx, gy, gz, ax, ay, az) -> 以前的情况
                
                try:
                    if len(parts) == 6:
                        # 无时间戳，直接读前3列
                        gx = float(parts[0])
                        gy = float(parts[1])
                        gz = float(parts[2])
                    elif len(parts) >= 7:
                        # 有时间戳，读第2,3,4列
                        gx = float(parts[1])
                        gy = float(parts[2])
                        gz = float(parts[3])
                    else:
                        continue # 列数不够，跳过

                    # --- 单位转换 ---
                    # 你的CSV里加速度是 9.74 (m/s^2)，说明陀螺通常是 rad/s
                    # Allan 方差图标准单位是 deg/h，计算需要先转成 deg/s
                    # 1 rad/s = (180/pi) deg/s
                    scale_factor = 180.0 / math.pi
                    
                    data.append([gx * scale_factor, gy * scale_factor, gz * scale_factor])
                    success_count += 1
                    
                except ValueError:
                    continue

            print(f"[Info] Loaded {success_count} lines.")
            
    except Exception as e:
        print(f"[Error] {e}")
        return None
        
    return np.array(data)

def simple_allan_variance(data, fs):
    N = len(data)
    if N < 100: return [], []
    
    max_tau_power = int(np.log10(N/2))
    m_list = np.unique(np.logspace(0, max_tau_power, 50).astype(int))
    
    taus = []
    adev = []
    
    print(f"Calculating Allan Variance ({N} pts)...", end='', flush=True)
    
    for m in m_list:
        tau = m / fs
        if m * 2 > N: break
        
        n_sub = int(N // m)
        data_sub = data[:n_sub*m].reshape((n_sub, m))
        avg_rates = np.mean(data_sub, axis=1) # 求平均速率
        diffs = np.diff(avg_rates)            # 相邻差分
        var = 0.5 * np.mean(diffs**2)
        dev = np.sqrt(var)
        
        taus.append(tau)
        adev.append(dev)
        print(".", end='', flush=True)
        
    print(" Done.")
    return np.array(taus), np.array(adev)

def main():
    data = read_data(FILE_PATH)
    if data is None or len(data) == 0:
        print("[Fatal] No data loaded.")
        return

    titles = ['Gyro X', 'Gyro Y', 'Gyro Z']
    colors = ['r', 'g', 'b']
    
    plt.figure(figsize=(10, 8))
    results = []

    for i in range(3):
        gyro_data = data[:, i] # 单位: deg/s
        taus, adev = simple_allan_variance(gyro_data, FS)
        
        if len(taus) == 0: continue

        adev_h = adev * 3600.0 # 转换 y轴单位: deg/s -> deg/h
        
        plt.loglog(taus, adev_h, label=titles[i], color=colors[i])
        
        # 拟合参数
        # 1. ARW (角度随机游走): 斜率 -0.5, 取 tau=1 附近
        idx_1s = np.abs(taus - 1.0).argmin()
        arw_val = adev_h[idx_1s] / 60.0 # deg/h -> deg/sqrt(h)
        
        # 2. Bias Instability: 曲线底部最小值
        min_adev = np.min(adev_h)
        
        results.append({'axis': titles[i], 'arw': arw_val, 'bias_inst': min_adev})

    plt.grid(True, which="both", ls="-", color='0.65')
    plt.title(f'Allan Deviation - {os.path.basename(FILE_PATH)}')
    plt.xlabel('Tau (s)')
    plt.ylabel('Allan Deviation (deg/h)')
    plt.legend()
    
    print("\n" + "="*60)
    print(" >>> AUTOMATIC PARAMETER SUGGESTION <<<")
    print("="*60)
    
    avg_arw = np.mean([r['arw'] for r in results])
    avg_bi = np.mean([r['bias_inst'] for r in results])
    
    print(f"{'Axis':<10} | {'ARW (deg/sqrt(h))':<20} | {'Bias Instability (deg/h)':<25}")
    print("-" * 60)
    for r in results:
        print(f"{r['axis']:<10} | {r['arw']:.6f}{'':<12} | {r['bias_inst']:.6f}")
    print("-" * 60)
    
    print(f"\n[Recommendation for C++ Config]")
    print(f"1. cfg.web_psd (ARW) [Use Avg ARW]:")
    print(f"   Value = {avg_arw:.6f} * glv.deg / sqrt(3600.0);")
    print(f"\n2. cfg.eb_sigma (P0) [Use 10x Bias Instability]:")
    print(f"   Value = {avg_bi*10:.6f} * glv.deg / 3600.0;")

    output_img = "allan_plot.png"
    plt.savefig(output_img)
    print(f"\nPlot saved to '{output_img}'. Check this image for the curves!")

if __name__ == "__main__":
    main()