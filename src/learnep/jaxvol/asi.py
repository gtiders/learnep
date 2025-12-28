"""
Active Set Inverse (ASI) 文件 I/O 模块。
负责读取和保存 active_set.asi 文件格式。
"""
import numpy as np

def save_asi(asi, filename="active_set.asi"):
    """
    Save the Active Set Inverse dictionary to a file.
    Format:
    Element Rows Cols
    val1
    val2
    ...
    """
    with open(filename, "w") as f:
        for k, v in asi.items():
            f.write(f"{k} {v.shape[0]} {v.shape[1]}\n")
            # Using v.flatten() but ensuring we write floats
            np.savetxt(f, v.flatten(), fmt="%.18e")

def load_asi(filename):
    """
    Load the Active Set Inverse dictionary from a file.
    """
    ret = {}
    with open(filename, "r") as f:
        while True:
            line1 = f.readline()
            if not line1:
                break
            parts = line1.strip().split()
            if not parts: 
                break
                
            element = parts[0]
            rows, cols = int(parts[1]), int(parts[2])
            size = rows * cols
            
            # Read 'size' lines
            # This is slow if reading line by line for large matrices.
            # But the format is line-separated values...
            # We can use np.fromfile or readlines, but since files are interleaved with headers,
            # we have to be careful. If the format strictly follows the header line then raw data...
            
            # Let's read the specific number of lines required.
            data = []
            for _ in range(size):
                line = f.readline()
                data.append(float(line))
            
            ret[element] = np.array(data).reshape((rows, cols))
            
    return ret
