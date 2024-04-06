import torch
import torch.nn as nn
import torchvision
from torchmetrics import F1Score
from torchvision import transforms
from timeit import default_timer as timer
import argparse
import os
import sys
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url

sys.path.insert(0, '../src')
from utils import find_classes, data_augmentation
from data_loader import Load_Data
from engine import Training


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


def get_args():
    parser = argparse.ArgumentParser('Train classification model')
    parser.add_argument('--model-name', '-n', type=str, default='B', help='batch size of the dataset')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='batch size of the dataset')
    parser.add_argument('--epochs', '-e', type=int, default=2, help='Epochs to train')
    parser.add_argument('--image-size', '-i', type=int, default=384, help='Input image size')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, help='Learning rate while training')
    parser.add_argument('--train-path', '-p', type=str, default='dataset_2/train', help='train path')
    parser.add_argument('--test-path', '-q', type=str, default='dataset_2/test', help='test path')
    parser.add_argument('--save-path', '-s', type=str, default='models', help='saving path')
    parser.add_argument('--load-checkpoint', '-c', type=str, default=None, help='path to saved the checkpoint')
    parser.add_argument('--tensorboard', '-t', type=str, default='tensorboard', help='path to tensorboard')
    args, unknown = parser.parse_known_args()
    return args


def main(args):
    #Parameters
    train_path = args.train_path
    test_path = args.test_path
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    model_name = args.model_name
    num_workers = os.cpu_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Find number of classes
    classes, _ = find_classes(train_path)
    num_classes = len(classes)

    #Load model
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
        
        elif model_name != 'B' and model_name != 'B_1':
            center_crop_ratio = 0.7
            drop_out_ratio = 0.2
            if model_name == 'A':
                # Augmentation using for Scatter Line and Horizontal Interval class
                data_augmentation(classes_list=['scatter-line', 'horizontal_interval'],
                                  original_path=train_path)
        
        
        image_transformation = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.CenterCrop((int(image_size*center_crop_ratio), int(image_size*center_crop_ratio))),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])])
        

    #Load data
    train_dataloader, test_dataloader = Load_Data(train_dir=train_path,
                                                  test_dir=test_path,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  transform=image_transformation)


    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=drop_out_ratio, inplace=True),
        torch.nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
    ).to(device=device)


    #Compile model
    loss_function = nn.CrossEntropyLoss()
    accuracy = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(device=device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)


    #Load model
    if args.load_checkpoint:
        checkpoint = torch.load(args.load_checkpoint)
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
    else:
        start_epoch = 0

    #Training process
    start_time = timer()
    model_results = Training(model=model,
                             train_dataloader=train_dataloader,
                             test_dataloader=test_dataloader,
                             loss_function=loss_function,
                             optimizer=optimizer,
                             accuracy=accuracy,
                             epochs=epochs,
                             start_epoch=start_epoch,
                             args=args,
                             device=device)

    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")


if __name__ == '__main__':
  args = get_args()
  results = main(args)
  
  