import numpy as np
import matplotlib.pyplot as plt
import math
import os

# --- 配置 ---
FILE_PATH = "/home/v/dev/hybrid_ins_cpp/fog3h.csv"
FS = 400.0  # 采样率 (Hz)

def read_data(path):
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
            if not sample_line: return None
            
            use_comma = ',' in sample_line
            delimiter_name = "Comma" if use_comma else "Whitespace"
            print(f"[Debug] Format: {delimiter_name}, Preview: {sample_line.strip()[:50]}...")

            success_count = 0
            
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                if line[0].isalpha() or line.startswith('%') or line.startswith('#'): continue

                parts = line.split(',') if use_comma else line.split()
                
                try:
                    if len(parts) == 6:
                        # 无时间戳: gx, gy, gz, ax, ay, az
                        ax = float(parts[3])
                        ay = float(parts[4])
                        az = float(parts[5])
                    elif len(parts) >= 7:
                        # 有时间戳: t, gx, gy, gz, ax, ay, az
                        ax = float(parts[4])
                        ay = float(parts[5])
                        az = float(parts[6])
                    else:
                        continue

                    # 单位: m/s^2 -> ug (微g)
                    # 1 m/s^2 = 1/9.8 g = 1e6/9.8 ug
                    scale_factor = 1e6 / 9.80665
                    
                    data.append([ax * scale_factor, ay * scale_factor, az * scale_factor])
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
        avg_rates = np.mean(data_sub, axis=1)
        diffs = np.diff(avg_rates)
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

    titles = ['Acc X', 'Acc Y', 'Acc Z']
    colors = ['r', 'g', 'b']
    
    plt.figure(figsize=(10, 8))
    results = []

    for i in range(3):
        acc_data = data[:, i]  # 单位: ug
        taus, adev = simple_allan_variance(acc_data, FS)
        
        if len(taus) == 0: continue

        plt.loglog(taus, adev, label=titles[i], color=colors[i])
        
        # 拟合参数
        # 1. VRW (速度随机游走): 斜率 -0.5, 取 tau=1 附近
        idx_1s = np.abs(taus - 1.0).argmin()
        vrw_val = adev[idx_1s]  # ug at tau=1s
        # VRW 单位转换: ug -> ug/sqrt(Hz) = ug * sqrt(tau) at tau=1
        vrw_ugpsqrtHz = vrw_val
        
        # 2. Bias Instability: 曲线底部最小值
        min_adev = np.min(adev)
        
        results.append({'axis': titles[i], 'vrw': vrw_ugpsqrtHz, 'bias_inst': min_adev})

    plt.grid(True, which="both", ls="-", color='0.65')
    plt.title(f'Accelerometer Allan Deviation - {os.path.basename(FILE_PATH)}')
    plt.xlabel('Tau (s)')
    plt.ylabel('Allan Deviation (ug)')
    plt.legend()
    
    print("\n" + "="*60)
    print(" >>> ACCELEROMETER PARAMETER SUMMARY <<<")
    print("="*60)
    
    avg_vrw = np.mean([r['vrw'] for r in results])
    avg_bi = np.mean([r['bias_inst'] for r in results])
    
    print(f"{'Axis':<10} | {'VRW (ug/sqrt(Hz))':<20} | {'Bias Instability (ug)':<25}")
    print("-" * 60)
    for r in results:
        print(f"{r['axis']:<10} | {r['vrw']:.2f}{'':<16} | {r['bias_inst']:.2f}")
    print("-" * 60)
    print(f"{'Average':<10} | {avg_vrw:.2f}{'':<16} | {avg_bi:.2f}")
    
    print(f"\n[Reference: Accelerometer Grade]")
    print(f"  Navigation grade: VRW < 10 ug/sqrt(Hz), BI < 10 ug")
    print(f"  Tactical grade:   VRW < 100 ug/sqrt(Hz), BI < 100 ug")
    print(f"  Consumer grade:   VRW > 100 ug/sqrt(Hz), BI > 100 ug")
    
    print(f"\n[Your Accelerometer]")
    if avg_vrw < 10 and avg_bi < 10:
        print(f"  Grade: Navigation (excellent)")
    elif avg_vrw < 100 and avg_bi < 100:
        print(f"  Grade: Tactical (good)")
    else:
        print(f"  Grade: Consumer/Industrial")

    output_img = "allan_acc_plot.png"
    plt.savefig(output_img)
    print(f"\nPlot saved to '{output_img}'")

if __name__ == "__main__":
    main()
