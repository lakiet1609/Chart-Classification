import torch
import torch.nn as nn
import torchvision
from torchmetrics import F1Score
from timeit import default_timer as timer
import argparse
import os
import sys
sys.path.insert(0, '../src')

from utils import find_classes
from data_loader import Load_Data
from engine import Training

def get_args():
    parser = argparse.ArgumentParser('Train classification model')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='batch size of the dataset')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='Epochs to train')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, help='Learning rate while training')
    parser.add_argument('--train-path', '-p', type=str, default=r'images/train', help='train path')
    parser.add_argument('--test-path', '-q', type=str, default=r'images/test', help='test path')
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
    num_workers = os.cpu_count()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    #Find number of classes
    classes, _ = find_classes(train_path)
    num_classes = len(classes)

    #Load model
    weights = torchvision.models.ResNet101_Weights.DEFAULT
    model = torchvision.models.resnet101(weights=weights).to(device=device)

    #Get the transform from the pretrained_weights
    auto_transform = weights.transforms()

    #Load data
    train_dataloader, test_dataloader = Load_Data(train_dir=train_path,
                                                  test_dir=test_path,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers,
                                                  transform=auto_transform)


    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
    ).to(device=device)

    #Compile model
    loss_function = nn.CrossEntropyLoss()
    accuracy = F1Score(task='multiclass', num_classes=num_classes, average='micro').to(device=device)
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

