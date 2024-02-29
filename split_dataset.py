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
    
    
    for folder in os.listdir(original_data):
        new_list = []
        for subfolder in os.listdir(os.path.join(original_data, folder)):
            if subfolder in big_class:
                new_list += list(Path(original_data, folder, subfolder).glob('*.jpg'))
                
                new_class_path = os.path.join(new_folder, folder, new_class)
                
                if not os.path.exists(new_class_path):
                    os.makedirs(new_class_path, exist_ok=True)
    
        for file in new_list:
            shutil.copy(file, new_class_path)

def split_one_class(original_data, new_folder, split_class, ratio):
    if not os.path.exists(split_class):
        os.makedirs(split_class, exist_ok=True)
    
    for folder in os.listdir(original_data):
        for subfolder in os.listdir(os.path.join(original_data, folder)):
            if subfolder == split_class:
                list_subfolder_path = list(Path(original_data, folder, subfolder).glob('*.jpg'))
                percent = int(len(list_subfolder_path)*ratio)
                subsample = random.sample(population=list_subfolder_path, k=percent)
                new_subfolder = Path(new_folder, folder, split_class)
                
                if not os.path.exists(new_subfolder):
                    os.makedirs(new_subfolder, exist_ok=True)
                
                for file in subsample:
                    shutil.copy(file, new_subfolder)

            
    
            

if __name__ == '__main__':
    # split_folder(original_data='images',
    #              split_folder='new_split', 
    #              ratio=0.1)
    
    # count_image(data='new_split')

    compose_class(original_data='images',
                  new_folder='1',
                  new_class='A',
                  big_class=['area', 'heatmap', 'horizontal_bar', 'horizontal_interval','manhattan', 'map', 
                             'pie', 'scatter', 'scatter-line', 'surface', 'venn', 'vertical_bar', 'vertical_box', 'vertical_interval'])

    # count_image(data='new2')
    # composed_class = ['area', 'surface', 'heatmap']
    # compose_class(original_data='images', new_class='A', new_folder='new2', big_class=composed_class)
    # split_one_class(original_data='images',new_folder='new2' ,split_class='vertical_bar', ratio=0.7)

        

