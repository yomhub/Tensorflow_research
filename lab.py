import os, sys
import tensorflow as tf
import numpy as np
from mydataset.ctw import CTW
# from mydataset.svt import SVT
# from lib.model.faster_rcnn import Faster_RCNN, RCNNLoss
# from lib.tflib.evaluate_tools import draw_boxes, save_image
# from lib.tflib.log_tools import _auto_histogram
# import matplotlib.pyplot as plt
import argparse
# 

if __name__ == "__main__":
  npz = np.zeros((10,10,3))
  npo = np.ones((5,5,3))

  npz[1:6,1:6,:]=npo

  print(npz)
  