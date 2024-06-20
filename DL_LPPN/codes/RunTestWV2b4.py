# conda activate LYpy3.7tf1.14np1.16
# cd/d E:\TQW\FusionEvaluateExperiment\MethodDL_PanNet
# python RunFusionHypothesis.py

# ---------------------------------------------------------------
# Copyright (c) 2021, Cheng Jin, Liang-Jian Deng, Ting-Zhu Huang,
# Gemine Vivone, All rights reserved.
#
# This work is licensed under GNU Affero General Public License 
# v3.0 International To view a copy of this license, see the 
# LICENSE file.
#
# This file is running on WorldView-3 dataset. For other dataset
# (i.e., QuickBird and GaoFen-2), please change the corresponding
# inputs.
# ---------------------------------------------------------------

import os
import scipy.io as sio
import numpy as np
import tensorflow as tf
import WV2b4_model
import time
import h5py

import os, sys, fnmatch

tf.compat.v1.reset_default_graph() # tf.reset_default_graph()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 隐去warning
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1' # 默认，显示所有信息 
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 warning 和 Error 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' # 只显示 Error 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) #上面代码仅仅能去除普通的pycharm的警告信息，对于tf产生的警告信息根本没用，如果使用的是tf1.0不用 加compat.v1 该代码是直接将提示信息设置为只提醒error类型的信息，不提醒warning类型

"""
# 原test.py代码       

"""

# if __name__ == '__main__':
def run_main(test_data,model_directory,testOutput_data):

    # test_data = './test_data/GaoFen-2.mat'
    # model_directory = './models/'

    tf.reset_default_graph()

    data = sio.loadmat(test_data)
    # data = mat73.loadmat(test_data)
    
    # data normalization

    ms = data['ms'][...]  # MS image
    ms = np.array(ms, dtype=np.float32) / 2047. # 2047 for WorldView-3, QuickBird datasets, 1023 for GaoFen-2 datasets
    ms = ms[np.newaxis, :, :, :]

    lms = data['lms'][...]  # up-sampled LRMS image
    lms = np.array(lms, dtype=np.float32) / 2047. # 2047 for WorldView-3, QuickBird datasets, 1023 for GaoFen-2 datasets
    lms = lms[np.newaxis, :, :, :]

    pan = data['pan'][...]  # pan image
    pan = np.array(pan, dtype=np.float32) / 2047. # 2047 for WorldView-3, QuickBird datasets, 1023 for GaoFen-2 datasets
    pan = pan[np.newaxis, :, :, np.newaxis]

    h = pan.shape[1]  # height
    w = pan.shape[2]  # width

    # placeholder for tensor
    pan_p = tf.placeholder(shape=[1, h, w, 1], dtype=tf.float32)
    # ms_p = tf.placeholder(shape=[1, h / 4, w / 4, 8], dtype=tf.float32)    
    ms_p = tf.placeholder(shape=[1, h / 4, w / 4, 4], dtype=tf.float32)    
    # lms_p = tf.placeholder(shape=[1, h, w, 8], dtype=tf.float32)
    lms_p = tf.placeholder(shape=[1, h, w, 4], dtype=tf.float32)

    # 输出张量用于调试
    # print('p_hp shape:', pan_p.shape)
    # print('m_hp shape:', ms_p.shape)
    # print('lms_p shape:', lms_p.shape)

    output_pyramid = WV2b4_model.LPPN(pan_p, ms_p)
    
    # output  = tf.clip_by_value(output_pyramid[4], 0, 1)  # final output
    output  = tf.clip_by_value(output_pyramid[4], 0, 2047)  # final output
    # 输出张量用于调试
    # print('p_hp shape:', pan_p.shape)
    # print('m_hp shape:', ms_p.shape)
    # print('lms_p shape:', lms_p.shape)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # loading  model
        if tf.train.get_checkpoint_state(model_directory):
            ckpt = tf.train.latest_checkpoint(model_directory)
            saver.restore(sess, ckpt)
            print("load new model")

        else:
            ckpt = tf.train.get_checkpoint_state(model_directory + "pre-trained/")
            saver.restore(sess,
                          ckpt.model_checkpoint_path)
            print("load pre-trained model")
        start_time = time.time()
        final_output = sess.run(output, feed_dict={pan_p: pan, lms_p: lms, ms_p: ms})
        end_time = time.time()
        print('running time: ', end_time-start_time)
        # sio.savemat('./result/output_GF.mat', {'output_LPPN': final_output[0, :, :, :]})
        sio.savemat(testOutput_data, {'output': final_output[0, :, :, :]})


