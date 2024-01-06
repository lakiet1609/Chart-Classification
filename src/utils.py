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
from torch.utils.tensorboard import SummaryWriter

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


def read_label(path):
    label_list = []
    exception_list = ['vertical bar', 'vertical interval', 'horizontal bar', 'horizontal interval', 'vertical box']
    replace_list = ['vertical_bar', 'vertical_interval', 'horizontal_bar', 'horizontal_interval', 'vertical_box']
    for file in sorted(os.listdir(path)):
        file_path = os.path.join(path, file)
        label = read_json(file_path)
        if label in exception_list:
            index = exception_list.index(label)
            label = replace_list[index]
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
    plt.show()

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
        plt.imshow(image)
    plt.show()

 
def evaluate_score(label: list,
                   output: list):
    
    assert len(label) == len(output)
    classes = sorted(list(Counter(label).keys()))
    count = 0
    true_prediction = {k: 0 for k in classes}
    for i in range(len(output)):
        if label[i] == output[i]:
            count += 1
            true_prediction[label[i]] += 1
    
    print(f'Number of total images: {len(output)}')
    print(f'Number of correct predictions: {count}')
    
    ratio = round(float((count/len(label))*100),1)
    
    print(f'Prediction ratio on the total evaluation set: {ratio}%')
    return true_prediction
