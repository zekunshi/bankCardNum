from __future__ import print_function

import os
import tensorflow as tf
from crnn.modules import CRNN
from multiprocessing import Pool
from crnn import config as crnn_config
from crnn.data_generator import load_images
from crnn.data_generator import captcha_batch_gen


import glob
import os
import shutil
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

sys.path.append(os.getcwd())

from ctpn.lib.fast_rcnn.config import cfg, cfg_from_file
from ctpn.lib.fast_rcnn.test import _get_blobs
from ctpn.lib.text_connector.detectors import TextDetector
from ctpn.lib.text_connector.text_connect_cfg import Config as TextLineCfg
from ctpn.lib.rpn_msr.proposal_layer_tf import proposal_layer
from ctpn.ctpn.getNumPlace import draw_boxes,resize_im


from ctpn.lib.fast_rcnn.config import cfg, cfg_from_file
def clearData():

    f1 = open(r'.\demo\test_results\result.txt','w+')

    f1.truncate()
    f1.close()
    f2 = open(r'.\crnn\data\predict_label.txt', 'r+')

    f2.truncate()
    f2.close()
    num_names = os.listdir('./cardNum')
    result_data = os.listdir('./data/results')
    testResults=os.listdir('./demo/test_results')
    i = 0
    while i < len(num_names):
        os.remove('./cardNum' + '//' + num_names[i])
        i += 1

    i = 0
    while i < len(result_data):
        os.remove('./data/results' + '//' + result_data[i])
        i += 1

    i = 0
    while i < len(testResults):
        os.remove('./demo/test_results' + '//' + testResults[i])
        i += 1

def predict():

    clearData()

    cfg_from_file(r'./ctpn/ctpn/text.yml')

    # init session
    config = tf.ConfigProto(allow_soft_placement=True)

    config.gpu_options.per_process_gpu_memory_fraction = 0.4

    sess = tf.Session(config=config)
    with gfile.FastGFile(r'.\ctpn\ctpn\data\ctpn.pb', 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
    sess.run(tf.global_variables_initializer())

    input_img = sess.graph.get_tensor_by_name('Placeholder:0')
    output_cls_prob = sess.graph.get_tensor_by_name('Reshape_2:0')
    output_box_pred = sess.graph.get_tensor_by_name('rpn_bbox_pred/Reshape_1:0')

    im_names = os.listdir('./demo/test_images')
    index = 0
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(('finding card number for {:s}'.format(im_name)))
        img = cv2.imread('./demo/test_images' + '//' + im_name)
        img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
        blobs, im_scales = _get_blobs(img, None)
        if cfg.TEST.HAS_RPN:
            im_blob = blobs['data']
            blobs['im_info'] = np.array(
                [[im_blob.shape[1], im_blob.shape[2], im_scales[0]]],
                dtype=np.float32)
        cls_prob, box_pred = sess.run([output_cls_prob, output_box_pred], feed_dict={input_img: blobs['data']})
        rois, _ = proposal_layer(cls_prob, box_pred, blobs['im_info'], 'TEST', anchor_scales=cfg.ANCHOR_SCALES)

        scores = rois[:, 0]
        boxes = rois[:, 1:5] / im_scales[0]
        textdetector = TextDetector()
        boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
        draw_boxes(img, im_name, boxes, scale, index)
        index += 1

    print('recognizing card number:')


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
                train_label_path=crnn_config.predict_label_path,
                train_images_path=crnn_config.predict_images_path,
                charset_path=crnn_config.charset_path)

    result = crnn.predict(epoch=crnn_config.epoch,
               batch_size=crnn_config.batch_size,
               train_images_path=crnn_config.cardNum_path,
               train_label_path=crnn_config.predict_label_path,
               restore=True,
               fonts=crnn_config.fonts,
               logs_path=crnn_config.logs_path,
               models_path=crnn_config.models_path)
    return result


if __name__ == '__main__':
    predict()