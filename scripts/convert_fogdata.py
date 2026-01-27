import scipy.io as sio
import pandas as pd
import numpy as np
import os

INPUT_MAT_FILE = '/home/v/桌面/Shared/CAI_FOG/fog3h.mat' 
OUTPUT_CSV_FILE = '/home/v/dev/hybrid_ins_cpp/build/fog3h.csv'
VAR_NAME = 'fog' 

def convert():
    if not os.path.exists(INPUT_MAT_FILE):
        print(f"Error: File not found at {INPUT_MAT_FILE}")
        return

    print(f"Loading {INPUT_MAT_FILE}...")
    mat = sio.loadmat(INPUT_MAT_FILE)
    
    if VAR_NAME not in mat:
        print(f"Error: Variable '{VAR_NAME}' not found in .mat file.")
        print(f"Available variables: {mat.keys()}")
        return

    data = mat[VAR_NAME]
    
    # 虽然这里指定了 columns，但在保存时会被忽略
    df = pd.DataFrame(data, columns=['gx', 'gy', 'gz', 'ax', 'ay', 'az'])
    
    print(f"Saving to {OUTPUT_CSV_FILE}...")
    
    # 关键修改：header=False 禁止写入列名
    df.to_csv(OUTPUT_CSV_FILE, index=False, float_format='%.12f', header=False)
    
    print(f"Done. Saved {len(df)} rows without header.")

if __name__ == "__main__":
    convert()