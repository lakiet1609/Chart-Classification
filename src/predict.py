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

from utils import find_classes, read_label, evaluate_score


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

def get_args():
    parser = argparse.ArgumentParser('Evaluate classification model')
    parser.add_argument('--load-checkpoint', '-c', type=str, default='models/resnet.pt', help='path to saved the checkpoint')
    parser.add_argument('--evaluation-path', '-e', type=str, default='images_test/images', help='Images path to evaluate')
    parser.add_argument('--evaluation-label', '-v', type=str, default='images_test/labels', help='Label path to evaluate')
    parser.add_argument('--train-path', '-t',  type=str, default='images/train', help='Images path to train')
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
    weights = torchvision.models.ResNet101_Weights.DEFAULT
    model = torchvision.models.resnet101(weights=weights).to(device=device)
    auto_transform = weights.transforms()


    #Load model
    checkpoint = torch.load(load_checkpoint)
    model.load_state_dict(checkpoint['model'], strict=False)
    
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

    return predictions, labels

if __name__ == '__main__':
    args = get_args()
    evaluate(args=args)
    predictions, labels = evaluate(args=args)
    evaluate_score(label=labels, output=predictions)


        
