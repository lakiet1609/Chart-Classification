import os
import torch
from PIL import Image
from pathlib import Path
from collections import Counter
import json
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import random
import yaml
import shutil
import albumentations as A
import cv2
import uuid

# from torch.utils.tensorboard import SummaryWriter

def find_classes(target_dir):
    classes = sorted([name for name in os.listdir(target_dir)])
    if not classes:
        raise FileNotFoundError(f'Could not find the file in folder {target_dir}')
    else:
        class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx

def save_model(model,
               target_dir,
               epoch,
               optimizer,
               acc,
               best_acc,
               loss,
               min_loss,
               args):

    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)

    checkpoint = {
        'epoch': epoch+1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    torch.save(checkpoint, os.path.join(target_dir, 'last.pt'))

    if acc > best_acc and loss < min_loss :
        torch.save(checkpoint, os.path.join(target_dir, 'best.pt'))
        best_acc = acc
        min_loss = loss

def tensorboard(train_loss, test_loss, test_acc, epoch, args):
    writer = SummaryWriter(args.tensorboard)
    writer.add_scalars(main_tag='Loss',
                       tag_scalar_dict={'train_loss': train_loss,
                                        'test_loss': test_loss},
                       global_step=epoch)

    writer.add_scalars(main_tag='Accuracy',
                       tag_scalar_dict={'test_acc': test_acc},
                       global_step=epoch)

def read_json(path):
    with open(path) as data:
        f = json.load(data)
        output = f['task1']['output']['chart_type']
    return output

def read_label_custom(path):
    label_list = []
    exception_list = ['vertical bar', 'vertical interval', 'horizontal bar', 'horizontal interval', 'vertical box', 'scatter-line']
    replace_list = ['vertical_bar', 'vertical_interval', 'horizontal_bar', 'horizontal_interval', 'vertical_box', 'scatter_line']
    A = ['line', 'scatter', 'scatter_line', 'horizontal_interval']
    B = ['area', 'surface', 'heatmap']
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        label = read_json(file_path)
        if label in exception_list:
            index = exception_list.index(label)
            label = replace_list[index]
        if label not in B:
            label = 'unknown'
        label_list.append(label)
    return label_list

def read_label_class(path, class_list):
    label_list = []
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        label = read_json(file_path)
        if label in class_list:
            label_list.append(label)
    return label_list

def read_label(path, class_list, new_class):
    label_list = []
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        label = read_json(file_path)
        if label in class_list:
            label = new_class
        
        label_list.append(label)
    
    return label_list

def predict_custom_img(model,
                       image_path,
                       class_names,
                       image_size,
                       transform,
                       device):
    
    img = Image.open(image_path).convert('RGB')
    
    if transform:
        img_transform = transform(img)
    else:
        img_transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    model.to(device)
    model.eval()
    with torch.inference_mode():
        img_transformation = img_transform.unsqueeze(0)
        img_transformation = model(img_transformation.to(device))
        
    img_pred_prob = torch.softmax(img_transformation, dim=1)
    img_pred_label = torch.argmax(img_pred_prob, dim=1)
    
    probs = img_pred_prob.max()
    label = class_names[img_pred_label.item()]
    
    return probs, label

def pred_distribution(prediction_list, label_list, true_pred):
    classes = sorted(list(Counter(label_list).keys()))
    predictions = [Counter(prediction_list)[item] for item in classes]
    labels = [Counter(label_list)[item] for item in classes]
    actual_pred = [true_pred[item] for item in classes]
    counts = {
        'Predictions': np.array(predictions),
        'Correct Predictions': np.array(actual_pred),
        'Ground Truths': np.array(labels),
    }
    width = 0.6  # the width of the bars: can also be len(x) sequence
    fig, ax = plt.subplots()
    bottom = np.zeros(len(classes))

    for pred, pred_count in counts.items():
        p = ax.bar(classes, pred_count, width, label=pred, bottom=bottom)
        bottom += pred_count
        ax.bar_label(p, label_type='center')

    ax.set_title('Ratio of predictions over ground truths on the test set')
    ax.legend()
    plt.savefig('pred.png')

def visualize_test_pred(test_imgs,
                        test_labels,
                        model,
                        image_size,
                        class_names,
                        num_images,
                        transform,
                        device):
    
    test_path = list(Path(test_imgs).glob('*.jpg'))
    test_samples = random.sample(population=test_path,k=num_images)
    
    fig = plt.figure(figsize=(20,8))
    columns = 4
    rows = 4
    fig.subplots_adjust(wspace=1.2, hspace=0.8)
    for i in range(1, columns*rows+1):
        test_sample = [sample for sample in test_samples]
        image = plt.imread(test_sample[i-1])
        probs, pred = predict_custom_img(model=model,
                                         image_path=test_sample[i-1],
                                         image_size=image_size,
                                         class_names=class_names,
                                         transform=transform,
                                         device=device)
        
        label = test_sample[i-1].stem + '.json'
        label_path = Path(test_labels, label)
        actual = read_json(label_path)
        fig.add_subplot(rows, columns, i)
        plt.title(f'Actual value: {actual} \nPrediction: {pred} with probability {probs:.3f} ')    
        plt.savefig(image)
    plt.show()

def evaluate_score(label: list,
                   output: list,
                   prob: list):
    
    assert len(label) == len(output)
    FP_label_idx = []
    TP_prob = []
    FP_prob = []
    classes = sorted(list(Counter(label).keys()))
    count = 0
    true_prediction = {k: 0 for k in classes}
    for i in range(len(output)):
        if label[i] == output[i]:
            count += 1
            true_prediction[label[i]] += 1
            TP_prob.append(prob[i])
        else:
            FP_label_idx.append(i)
            FP_prob.append(prob[i])
                
    
    ratio = round(float((count/len(label))*100),1)
    average_TP_prob = round(sum(TP_prob)/len(TP_prob),2)
    average_FP_prob = round(sum(FP_prob)/len(FP_prob),2)
    
    print(f'Number of total images: {len(output)}')
    print(f'Number of correct predictions: {count}')
    print(f'Prediction ratio on the total evaluation set: {ratio}%')
    print(f'Average confidence score of True Positive predictions: {average_TP_prob}')
    print(f'Average confidence score of False Positive predictions: {average_FP_prob}')
    
    print(Counter(label))
    print(true_prediction)
    
    
    return true_prediction, FP_label_idx, (average_TP_prob, average_FP_prob)

def read_yaml(path):
    with open(path) as yaml_file:
        content = yaml.safe_load(yaml_file)
    return content

def save_file(file, save_entity, class_id):
    with open(f'{save_entity}/{class_id}.json', 'w') as f:
        json.dump(file, f)
        
def load_file(file):
    with open(file, 'rb') as f:
        output = json.load(f)
    return output

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

def split_one_class(original_data, new_class, split_class, ratio):
    for folder in os.listdir(original_data):
        if folder == split_class:
            list_subfolder_path = list(Path(original_data, folder).glob('*.jpg'))
            percent = int(len(list_subfolder_path)*ratio)
            subsample = random.sample(population=list_subfolder_path, k=percent)
            new_subfolder = Path(original_data, new_class)
            
            if not os.path.exists(new_subfolder):
                os.makedirs(new_subfolder, exist_ok=True)
            
            for file in subsample:
                shutil.copy(file, new_subfolder)

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

def augment_wrong_images(new_folder, evaluation_path, evaluation_label, label_idx, specific_class):
    new_class_path = os.path.join(new_folder, specific_class)
    os.makedirs(new_class_path, exist_ok=True)
    
    wrong_predicted_images = [sorted(os.listdir(evaluation_path))[i] for i in label_idx]
    for image in wrong_predicted_images:
        image_name = str(image).split('.')[0]
        image_label = image_name + '.json'
        lable_path = os.path.join(evaluation_label, image_label)
        image_path = os.path.join(evaluation_path, image)
        label = read_json(lable_path)
        if label == specific_class:
            shutil.copy(image_path, new_class_path)

def split_image_test(new_original, images_folder, label_folder, class_list):
    new_img_folder = os.path.join(new_original, 'images')
    if not os.path.exists(new_img_folder):
        os.makedirs(new_img_folder, exist_ok=True)
    
    new_label_folder = os.path.join(new_original, 'labels')
    if not os.path.exists(new_label_folder):
        os.makedirs(new_label_folder, exist_ok=True)
    
    for file in os.listdir(label_folder):
        file_path = os.path.join(label_folder, file)
        label = read_json(file_path)
        if label in class_list:
            file_name = str(file).split('.')[0]
            file_image = file_name + '.jpg'
            image_path = os.path.join(images_folder, file_image)
            shutil.copy(file_path, new_label_folder)
            shutil.copy(image_path, new_img_folder)

def data_augmentation(original_path: str,
                      classes_list: list):
    
    # Define the directory containing the original dataset
    original_dataset_dir_list = [f'{original_path}/{class_name}' for class_name in classes_list]


    # Define augmentation pipeline
    augmentation_pipeline = A.Compose([
        A.RandomRotate90(),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomGamma(p=0.5),
        A.Blur(blur_limit=1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.5, rotate_limit=45, p=0.5),
        A.RandomScale(p=0.5),
    ])

    # Load the original images
    image_files_list = [os.listdir(specific_class) for specific_class in original_dataset_dir_list]

    # Apply augmentation to each image in the dataset
    for i in range(len(image_files_list)):
        for file_path in image_files_list[i]:
            image_path = os.path.join(original_dataset_dir_list[i], file_path)
            image = cv2.imread(image_path)

            # Apply augmentation
            augmented = augmentation_pipeline(image=image)
            augmented_image = augmented['image']
            new_image_name = str(uuid.uuid4()) + ".jpg"
            new_img_path = os.path.join(original_dataset_dir_list[i], new_image_name)

            cv2.imwrite(new_img_path, augmented_image)

    print(f"Augmentation and sampling complete")
        
