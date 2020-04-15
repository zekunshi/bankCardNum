import os
import tensorflow as tf
from crnn.modules import CRNN
from crnn import config as crnn_config

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def train():
    crnn = CRNN(image_shape=crnn_config.image_shape,
                min_len=crnn_config.min_len,
                max_len=crnn_config.max_len,
                lstm_hidden=crnn_config.lstm_hidden,
                pool_size=crnn_config.pool_size,
                learning_decay_rate=crnn_config.learning_decay_rate,
                learning_rate=crnn_config.learning_rate,
                learning_decay_steps=crnn_config.learning_decay_steps,
                mode=crnn_config.mode,
                dict=crnn_config.dict,
                is_training=True,
                train_label_path=crnn_config.train_label_path,
                train_images_path=crnn_config.train_images_path,
                charset_path=crnn_config.charset_path)


    crnn.train(epoch=crnn_config.epoch,
               batch_size=crnn_config.batch_size,
               train_images_path=crnn_config.train_images_path,
               train_label_path=crnn_config.train_label_path,
               restore=True,
               fonts=crnn_config.fonts,
               logs_path=crnn_config.logs_path,
               models_path=crnn_config.models_path)

