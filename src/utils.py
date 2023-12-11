import os
import torch
from pathlib import Path
import shutil
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

