from torchvision import transforms
import torch
import argparse
import torchvision
import os
import sys
sys.path.insert(0, '../src')

from utils import *
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


def get_args():
    parser = argparse.ArgumentParser('Evaluate classification model')
    parser.add_argument('--model-name', '-n', type=str, default='B', help='batch size of the dataset')
    parser.add_argument('--image-size', '-i', type=int, default=384, help='Image size to resize')
    parser.add_argument('--save-entity', '-s', type=str, default='entity', help='Saving the entity')
    parser.add_argument('--load-entity', '-l', type=str, default=None, help='Loading the entity')
    parser.add_argument('--load-checkpoint', '-c', type=str, default='model_B/best.pt', help='path to saved the checkpoint')
    parser.add_argument('--evaluation-path', '-e', type=str, default='image_test/images', help='Images path to evaluate')
    parser.add_argument('--evaluation-label', '-v', type=str, default='image_test/labels', help='Label path to evaluate')
    parser.add_argument('--train-path', '-t',  type=str, default='new_b_class/train', help='Images path to train')
    args, unknown = parser.parse_known_args()
    return args


def prediction(args):
    #Parameters
    model_name = args.model_name
    train_path = args.train_path
    evaluation_path = args.evaluation_path
    evaluation_label = args.evaluation_label
    load_checkpoint = args.load_checkpoint
    save_entity = args.save_entity
    load_entity = args.load_entity
    composed_class_B = ['area', 'heatmap', 'surface']
    composed_class_A = ['horizontal interval', 'scatter', 'line', 'scatter-line']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Classes
    classes, _ = find_classes(train_path)
    print(classes)
    num_classes = len(classes)

    WeightsEnum.get_state_dict = get_state_dict
    
    #Preprocess for each composed classes    
    if model_name is None:
        weights = torchvision.models.EfficientNet_B3_Weights.DEFAULT
        model = torchvision.models.efficientnet_b3(weights=weights).to(device=device)
        image_transformation = weights.transforms()
        image_size = 320
        in_features = 1536
        drop_out_ratio = 0.5
    
    else: 
        assert model_name is not None
        weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
        model = torchvision.models.efficientnet_b4(weights=weights).to(device=device)
        in_features = 1792
        image_size = 384
        
        if model_name == 'B' or model_name == 'B_1':
            center_crop_ratio = 0.9
            drop_out_ratio = 0.5
        
        elif model_name == 'A' or model_name == 'A_1' or model == 'A_2':
            center_crop_ratio = 0.7
            drop_out_ratio = 0.2
        
        image_transformation = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.CenterCrop((int(image_size*center_crop_ratio), int(image_size*center_crop_ratio))),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
    


    model.classifier = torch.nn.Sequential(
    torch.nn.Dropout(p=drop_out_ratio, inplace=True),
    torch.nn.Linear(in_features=in_features, out_features=num_classes, bias=True)).to(device=device)

    #Load model
    checkpoint = torch.load(load_checkpoint)
    model.load_state_dict(checkpoint['model'])

    
    #Predict custom image on different category
    if load_entity is None:
        data_list = sorted(os.listdir(evaluation_path))
    else:
        data_list = load_file(load_entity)
    
    predictions = []
    probs = []
    labels = []
    label_paths_model_name = []
    label_paths_others = [] 
    for img in data_list:
        image_name = str(img).split('.')[0]
        label_file = image_name + '.json'
        img_path = os.path.join(evaluation_path, img)
        label_path = os.path.join(evaluation_label, label_file)
        label = read_json(label_path)
        prob, pred = predict_custom_img(model=model,
                                    image_path=img_path,
                                    class_names=classes,
                                    image_size=None,
                                    transform=image_transformation,
                                    device=device)
        
        if model_name == 'B':
            if label in composed_class_B:
                label = 'B'
            if pred == 'B':
                label_paths_model_name.append(img)
            else:
                label_paths_others.append(img)
        
        elif model_name == 'A':
            if pred == 'A_1':
                label_paths_model_name.append(img)
            else:
                label_paths_others.append(img)
        
        elif model_name == 'base' and pred == 'A':
            if label in composed_class_A:
                label = 'A'
            label_paths_model_name.append(img)
        
        probs.append(prob)
        labels.append(label)
        predictions.append(pred)

    #Saving entity for later inference process
    os.makedirs(save_entity, exist_ok=True)
    if len(label_paths_model_name) != 0:
        if model_name == 'B':
            save_file(file=label_paths_model_name,save_entity=save_entity, class_id=model_name)
        elif model_name == 'A':
            save_file(file=label_paths_model_name,save_entity=save_entity, class_id=str(model_name + '_1'))
        else:
            save_file(file=label_paths_model_name,save_entity=save_entity, class_id='A')
        
    if len(label_paths_others) != 0:    
        if model_name == 'B':
            save_file(file=label_paths_others,save_entity=save_entity, class_id='unidentified')
        elif model_name == 'A':
            save_file(file=label_paths_others,save_entity=save_entity, class_id=str(model_name + '_2'))

    #Evaluation score on the custom set
    evalution_score, _ = evaluate_score(label=labels, output=predictions, prob=probs)
    
    #Visualization of correct prediction distribution on the custom test set
    pred_distribution(predictions, labels, evalution_score)
    
    #Image prediction visualization
    while len(evaluation_path):
        visualize_test_pred(test_imgs=evaluation_path,
                            test_labels= evaluation_label,
                            model=model,
                            image_size=None,
                            class_names=classes,
                            num_images=16,
                            transform=image_transformation,
                            device=device)
    return predictions, labels
    

if __name__ == '__main__':
    args = get_args()
    prediction(args)


        
 