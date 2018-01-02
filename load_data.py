import os
import glob
import numpy as np
import pandas as pd
import cv2


def load_data():
    cars = glob.glob('vehicles/**/*.png', recursive=True)
    notcars = glob.glob('non-vehicles/**/*.png', recursive=True)
    
    vehicles = []
    no_vehicles = []
    for car in cars:
        img = cv2.imread(car)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        vehicles.append(img)

    for nocar in notcars:
        img = cv2.imread(nocar)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        no_vehicles.append(img)
   

    return vehicles, no_vehicles
