import shutil 
import os
import random
from pathlib import Path

def split_folder(original_data: str, 
                 split_folder:str, 
                 ratio: str):
    
    if not os.path.exists(split_folder):
        os.makedirs(split_folder, exist_ok=True)
    for folder in os.listdir(original_data):
        for subfolder in os.listdir(os.path.join(original_data, folder)):
            list_subfolder_path = list(Path(original_data, folder, subfolder).glob('*.jpg'))
            percent = int(len(list_subfolder_path)*ratio)
            subsample = random.sample(population=list_subfolder_path, k=percent)
            new_subfolder = Path(split_folder, folder, subfolder)
            if not os.path.exists(new_subfolder):
                os.makedirs(new_subfolder, exist_ok=True)
            for file in subsample:
                shutil.copy(file, new_subfolder)


def count_image(data: str):
    for folder in os.listdir(data):
        for subfolder in os.listdir(os.path.join(data, folder)):
            print(f'{subfolder} in {folder} set: {len(os.listdir(os.path.join(data, folder, subfolder)))} images')


def compose_class(
        original_data: str,
        new_folder:str, 
        new_class: str,
        big_class: list):
    
    if not os.path.exists(new_folder):
        os.makedirs(new_folder, exist_ok=True)
    
    new_list = []
    for folder in os.listdir(original_data):
        if folder in big_class:
            new_list += list(Path(original_data, folder).glob('*.jpg'))
            
            new_class_path = os.path.join(new_folder, new_class)
        
            if not os.path.exists(new_class_path):
                os.makedirs(new_class_path, exist_ok=True)

    for file in new_list:
        shutil.copy(file, new_class_path)


def custom_folder(target_dir: str):
    ori_path = Path(target_dir)
    paths_train = ori_path / 'train' 
    paths_test = ori_path / 'test' 
    
    #Create the folder of  training set
    if paths_train.is_dir():
        shutil.rmtree(paths_train)
    paths_train.mkdir(parents = True, exist_ok = True)    
    
    #Create the folder of  training set
    if paths_test.is_dir():
        shutil.rmtree(paths_test)
    paths_test.mkdir(parents = True, exist_ok = True)  

    for subfolders in os.listdir(ori_path):
        if subfolders == 'test' or subfolders =='train':
            continue
        file_paths = os.path.join(ori_path, subfolders)
        class_name = str(subfolders).split('\\')[-1]
        sub_paths_train = Path(os.path.join(file_paths,class_name))
        
        if sub_paths_train.is_dir():
            shutil.rmtree(sub_paths_train)
        sub_paths_train.mkdir(parents = True, exist_ok = True)    
        
        filenames = os.listdir(file_paths)
        random.shuffle(filenames)
        split_up_ratio = 0.8
        train_split_idx = int(len(filenames) * split_up_ratio)
        train_filenames = filenames[:train_split_idx]
        
        for filename in train_filenames:
            if filename != class_name:
                filename_path = os.path.join(file_paths,filename)
                shutil.move(filename_path, sub_paths_train)
        
        shutil.move(str(sub_paths_train), str(paths_train))
        shutil.move(str(file_paths),str(paths_test))
         

if __name__ == '__main__':
    pass


        

