# from pynput.mouse import Controller, Button
# import time
# mouse = Controller()
# while True:
#     mouse.click(Button.left, 1)
#     print('Clicked')
#     time.sleep(180)

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)