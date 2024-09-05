from pathlib import Path
import os
import shutil
import psutil

def kill_processes_using_file(file_path):
    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
        try:
            for file in proc.info['open_files'] or []:
                if file.path.startswith(file_path):
                    proc.kill()  # Terminate the process using the file
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

# delete chroma db folder
def delete_data_folder(folder_path):
    kill_processes_using_file(folder_path)
    if os.path.exists(folder_path):
        try:
            shutil.rmtree(folder_path)
            print('deleted')
        except Exception as e:
            print(f'Failed to delete folder: {e}')

dir = 'C:/Users/HomePC/Desktop/project_law/src/chroma_db'

if os.path.exists(dir):
    delete_data_folder(dir)
