from pynput.mouse import Controller, Button
import time

mouse = Controller()

while True:
    mouse.click(Button.left, 1)
    print('clicked')
    time.sleep(180)