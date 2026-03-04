import os

# 配置
input_file = "/home/v/dev/hybrid_ins_cpp/fog30h.csv"
lines_per_chunk = 8640000 # 6小时 * 3600秒 * 400Hz
output_prefix = "fog_part"

def split_csv():
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        return

    print(f"Splitting {input_file}...")
    
    chunk_idx = 1
    line_count = 0
    
    # 打开源文件
    with open(input_file, 'r') as f_in:
        f_out = open(f"{output_prefix}{chunk_idx}.csv", 'w')
        print(f"Writing {output_prefix}{chunk_idx}.csv ...")
        
        for line in f_in:
            f_out.write(line)
            line_count += 1
            
            # 切分点
            if line_count >= lines_per_chunk:
                f_out.close()
                chunk_idx += 1
                line_count = 0
                
                # 如果还有数据，开新文件
                f_out = open(f"{output_prefix}{chunk_idx}.csv", 'w')
                print(f"Writing {output_prefix}{chunk_idx}.csv ...")

        # 关闭最后一个文件
        if not f_out.closed:
            f_out.close()

    print("Done! Files created:")
    for i in range(1, chunk_idx + 1):
        print(f"  - {output_prefix}{i}.csv")

if __name__ == "__main__":
    split_csv()