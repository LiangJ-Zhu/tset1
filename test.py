import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from PIL import Image

'''img = Image.open("E:/Data/python/0316/origin.jpg")

arr = np.array(img)
print(arr.shape,arr.dtype)
print(arr)'''

a=np.arange(30).reshape(2,3,5)
b=np.mean(a,2)
c=a.reshape(-1,2)
d=a.reshape(2,-1)
print(a)
print(b)
print(c)
print(d)