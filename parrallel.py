# -*- coding:UTF-8 -*-
import concurrent.futures  
import os  
from preprocess import preprocess  # 假设这是你的预处理函数  
  
def process_file(file_path):  
    # 这里调用你的预处理函数  
    preprocess(file_path)  
  
def process_files_in_parallel(file_prefix, num_processes=None):  
    # 如果num_processes为None，则使用所有可用的CPU核心  
    if num_processes is None:  
        num_processes = os.cpu_count()  
  
    list1 = list(range(53,68))
    # 获取所有子文件的路径  
    file_paths = [f"{file_prefix}_{i}.jsonl" for i in list1]  # 假设有1000个子文件，你可能需要动态确定这个数  
  
    # 使用ProcessPoolExecutor并行处理文件  
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:  
        futures = [executor.submit(process_file, file_path) for file_path in file_paths]  
        for future in concurrent.futures.as_completed(futures):  
            try:  
                # 获取处理结果（如果有的话），或者处理异常  
                result = future.result()  
            except Exception as exc:  
                print(f"生成了一个异常: {exc}")  
  
# 使用示例  
process_files_in_parallel("zh_cc", num_processes=15)