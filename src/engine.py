import torch
import torchvision
from tqdm.autonotebook import tqdm
from utils import save_model, tensorboard

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_step(model,
               dataloader,
               loss_function,
               optimizer,
               device,
               ):
    
    model.to(device)
    model.train()
    train_loss = 0
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_function(y_pred,y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader)
    return train_loss
    
def test_step(model,
              dataloader,
              loss_function,
              accuracy,
              device):
    
    model.to(device)
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = loss_function(test_pred, y)
            test_loss += loss.item()
            test_pred_class = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
            test_acc += accuracy(test_pred_class, y)
        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
    return test_loss, test_acc

def Training(model,
             train_dataloader,
             test_dataloader,
             loss_function,
             optimizer,
             accuracy,
             device,
             epochs,
             start_epoch,
             args):
    
    best_acc = 0
    min_loss = 1
    results = {
        'train_loss': [],
        'test_loss': [],
        'test_acc': []
    }
    
    for epoch in tqdm(range(start_epoch, epochs), colour='CYAN'):
        train_loss = train_step(model=model,
                                dataloader=train_dataloader,
                                loss_function=loss_function,
                                optimizer=optimizer,
                                device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_function=loss_function,
                                        accuracy=accuracy,
                                        device=device)
        
        save_model(model=model,
                   target_dir=args.save_path,
                   epoch=epoch,
                   optimizer=optimizer,
                   acc=test_acc,
                   best_acc=best_acc,
                   loss=test_loss,
                   min_loss=min_loss,
                   args=args)
        
        tensorboard(train_loss=train_loss,
                    test_loss=test_loss,
                    test_acc=test_acc,
                    epoch=epoch,
                    args=args)
        
        
        print(
            f'Epoch: {epoch+1} |'
            f'train_loss: {train_loss} |'
            f'test_loss: {test_loss} |'
            f'test_acc: {test_acc} |'
        )
              
        
        results['train_loss'].append(train_loss)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        
    return results