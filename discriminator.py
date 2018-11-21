# -*- coding: utf-8 -*-

########## discriminator.py ##########
#
# MLP-GAN Discriminator
# 
#
# created 2018/9/28 Takeyuki Watadani @ UT Radiology
#
########################################

import config as cf

import tensorflow as tf
import numpy as np

import os.path
import io
import math
import random
import zipfile

logger = cf.LOGGER

class Discriminator:
    '''MLP-GANのDiscriminatorを記述するクラス'''


    def __init__(self, global_step):
        self.global_step_tensor = global_step

        #self.is_input_fromdataset = None # 入力画像の選択用bool placeholder
        self.from_dataset = None # データセットからの入力画像
        self.from_generator = None # generatorからの入力画像
        self.output = None # 出力の確率
        self.optimizer = None
        self.train_op = None

        self.input_selector = None # generatorからの時は True

    def define_forward(self, input, vreuse = None):
        '''判定する計算を返す'''
        
        with tf.variable_scope('D_network', reuse=vreuse):

            inflatten = tf.reshape(input, shape=(-1, cf.PIXELSIZE * cf.PIXELSIZE))


            # ver. 5 ネットワーク規模の拡大
            nilayers = 5
            iunits = cf.PIXELSIZE * cf.PIXELSIZE
            ilayers = []
            incons = 100
            
            for i in range(nilayers):
                if i == 0:
                    iin = inflatten
                else:
                    iin = ilayers[-1]

                interlayer = tf.layers.dense(inputs = iin,
                                             units = iunits,
                                             activation = tf.nn.leaky_relu,
                                             kernel_initializer = tf.initializers.random_uniform(minval=-incons/iunits, maxval=incons/iunits),
                                             name='D_interlayer_' + str(i))
                dropout = tf.layers.dropout(inputs = interlayer,
                                            rate = cf.DROPOUT_RATE,
                                            name='D_inter_dropout_' + str(i))
                ilayers.append(dropout)

            nlayers = round(math.log2(cf.PIXELSIZE * cf.PIXELSIZE / cf.LATENT_VECTOR_SIZE)) + 2

            layers = []
            for i in range(nlayers):
                if i == 0:
                    lin = ilayers[-1]
                else:
                    lin = layers[-1]

                nunits = (cf.PIXELSIZE * cf.PIXELSIZE) // (2 ** (i+1))
            
                perceptron_layer = tf.layers.dense(inputs = lin,
                                                   units = nunits,
                                                   activation = tf.nn.leaky_relu,
                                                   kernel_initializer=tf.initializers.random_uniform(minval=-(1.0/nunits), maxval=(1.0/nunits)),
                                                   name='D_perceptron_layer_' + str(i))
                dropout = tf.layers.dropout(inputs = perceptron_layer, rate=cf.DROPOUT_RATE, name='D_dropout_' + str(i))
                layers.append(dropout)

            fully_connected = tf.layers.dense(inputs = layers[-1],
                                              units = 1,
                                              kernel_initializer=tf.initializers.random_uniform(minval=-(1.0/255.0), maxval=(1.0/255.0)),
                                              activation = tf.nn.sigmoid,
                                              name='D_fully_connected')
            
            adjusted = fully_connected

            return adjusted

    def define_graph(self):
        '''discriminatorの計算グラフを定義する'''

        self.from_dataset = tf.placeholder(dtype=tf.float32, shape=(None, cf.PIXELSIZE, cf.PIXELSIZE), name='D_input_image')
        
        epsilon = 0.0000001

        random1 = tf.random_normal(shape=(cf.MINIBATCHSIZE, 1), mean=0.0, stddev=0.015, dtype=tf.float32,
                                   name='random_1')
        random2 = tf.random_normal(shape=(cf.MINIBATCHSIZE, 1), mean=0.0, stddev=0.015, dtype=tf.float32,
                                   name='random_2')

        self.p_real = tf.maximum(tf.add(self.define_forward(self.from_dataset, vreuse=False), random1),
                                  epsilon, name='p_real')
        self.p_fake = self.define_forward(self.from_generator, vreuse=True)

        self.p_fake_for_loss = tf.maximum(tf.add(self.p_fake, random2), epsilon,
                                           name='p_fake_for_loss')

        self.loss = -tf.reduce_mean(tf.log(0.0001 + self.p_real) - tf.log(0.0001 + self.p_fake_for_loss), name='D_loss')

        D_vars = [x for x in tf.trainable_variables() if 'D_' in x.name]
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=cf.LEARNING_RATE, name='D_optimizer')
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor, var_list=D_vars, name='D_train_op')

    def set_input_from_generator(self, generator):
        self.from_generator = generator.output
        return

    @staticmethod
    def create_minibatch():
        '''データセットからミニバッチを作成する'''
        zippath = os.path.join(cf.DATASET_PATH, cf.TRAIN_PREFIX + '.zip')
        with zipfile.ZipFile(zippath, 'r') as zf:

            nplist = zf.namelist()
            npsampled = random.sample(nplist, cf.MINIBATCHSIZE)

            minibatch = []

            for i in range(cf.MINIBATCHSIZE):
                bytes = zf.read(npsampled[i])
                buf = io.BytesIO(bytes)
                minibatch.append(np.load(buf))
        return minibatch
        
