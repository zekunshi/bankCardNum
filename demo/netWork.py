from __future__ import print_function


import glob
import os
import shutil
import sys


import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

sys.path.append(os.getcwd())


from flask import Flask, render_template, request

from web.Controller import getPictureFromNet


if __name__ == '__main__':
    getPictureFromNet()


