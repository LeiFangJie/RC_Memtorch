"""
此模块负责解析 CHB-MIT 头皮脑电图数据库中的 summary.txt 文件，
以提取癫痫发作的时间标注，并获取每位患者的 .edf 文件路径列表。
"""
import os
import re
from config import DATA_DIR

def parse_summary_file(patient_id):
    """
    解析指定患者的 summary.txt 文件，提取癫痫发作标注。
    
    参数:
        patient_id (str): 患者 ID (例如: 'chb01')
        
    返回:
        dict: 字典类型，键为 EDF 文件名，值为该文件中癫痫发作的 (start_time, end_time) 元组列表。
    """
    summary_path = os.path.join(DATA_DIR, patient_id, f"{patient_id}-summary.txt")
    if not os.path.exists(summary_path):
        print(f"Warning: Summary file not found at {summary_path}")
        return {}

    seizures_dict = {}
    current_file = None
    
    with open(summary_path, 'r') as f:
        lines = f.readlines()
        
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Match File Name
        file_match = re.match(r"File Name:\s*(.*\.edf)", line)
        if file_match:
            current_file = file_match.group(1)
            seizures_dict[current_file] = []
            continue
            
        # Match Number of Seizures
        num_seizures_match = re.match(r"Number of Seizures in File:\s*(\d+)", line)
        if num_seizures_match and current_file:
            num_seizures = int(num_seizures_match.group(1))
            
            # If there are seizures, they will be listed in the following lines
            if num_seizures > 0:
                # We will look for "Seizure Start Time" and "Seizure End Time"
                # They can be "Seizure Start Time: XXX seconds" or "Seizure 1 Start Time: XXX seconds"
                pass # We handle this by just matching all seizure times until next file
                
        # Match Seizure Start Time
        start_match = re.search(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", line)
        if start_match and current_file:
            start_time = int(start_match.group(1))
            
            # The next line should be the end time
            if i + 1 < len(lines):
                next_line = lines[i+1].strip()
                end_match = re.search(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", next_line)
                if end_match:
                    end_time = int(end_match.group(1))
                    seizures_dict[current_file].append((start_time, end_time))
                    
    return seizures_dict

def get_edf_paths(patient_id):
    """
    获取指定患者的所有 .edf 文件的绝对路径列表。
    
    参数:
        patient_id (str): 患者 ID (例如: 'chb01')
        
    返回:
        list: .edf 文件的绝对路径列表，按字母顺序排序。
    """
    patient_dir = os.path.join(DATA_DIR, patient_id)
    if not os.path.exists(patient_dir):
        return []
        
    edf_files = [f for f in os.listdir(patient_dir) if f.endswith('.edf')]
    edf_paths = [os.path.join(patient_dir, f) for f in edf_files]
    return sorted(edf_paths)

if __name__ == "__main__":
    # Test parser
    sample_patient = "chb01"
    seizures = parse_summary_file(sample_patient)
    for f, s in seizures.items():
        if s:
            print(f"{f}: {s}")
