from PIL import Image
from torch import nn
from torchvision import transforms
import torch
import argparse
import torchvision
from torchmetrics import F1Score
import os
import sys
sys.path.insert(0, '../src')

from utils import find_classes, read_label, evaluate_score, predict_custom_img, pred_distribution, visualize_test_pred


def get_args():
    parser = argparse.ArgumentParser('Evaluate classification model')
    parser.add_argument('--load-checkpoint', '-c', type=str, default=r'models\best4.pt', help='path to saved the checkpoint')
    parser.add_argument('--evaluation-path', '-e', type=str, default=r'image_test\images', help='Images path to evaluate')
    parser.add_argument('--evaluation-label', '-v', type=str, default=r'image_test\labels', help='Label path to evaluate')
    parser.add_argument('--train-path', '-t',  type=str, default=r'images\train', help='Images path to train')
    args, unknown = parser.parse_known_args()
    return args

def evaluate(args):
    #Parameters
    train_path = args.train_path
    evaluation_path = args.evaluation_path
    evaluation_label = args.evaluation_label
    load_checkpoint = args.load_checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Classes
    classes, _ = find_classes(train_path)

    #Load model
    weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
    model = torchvision.models.efficientnet_v2_s(weights=weights).to(device=device)
    auto_transform = weights.transforms()

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.5, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=len(classes), bias=True)).to(device=device)

    #Load model
    checkpoint = torch.load(load_checkpoint)
    model.load_state_dict(checkpoint['model'])
    
    predictions = []
    for img in sorted(os.listdir(evaluation_path)):
        img_path = os.path.join(evaluation_path, img)
        _, label = predict_custom_img(model=model,
                                      image_path=img_path,
                                      class_names=classes,
                                      image_size=None,
                                      transform=auto_transform,
                                      device=device) 
        predictions.append(label)
    
    labels = read_label(evaluation_label) 
    true_pred = evaluate_score(label=labels, output=predictions)

    pred_distribution(predictions, labels, true_pred)
    
    # while len(evaluation_path):
    #     visualize_test_pred(test_imgs=evaluation_path,
    #                         test_labels= evaluation_label,
    #                         model=model,
    #                         image_size=None,
    #                         class_names=classes,
    #                         num_images=16,
    #                         transform=auto_transform,
    #                         device=device)
    # return predictions, labels

if __name__ == '__main__':
    args = get_args()
    evaluate(args)


        
