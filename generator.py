# -*- coding: utf-8 -*-

########## generator.py ##########
#
# MLP-GAN Generator
# 
#
# created 2018/9/28 Takeyuki Watadani @ UT Radiology
#
########################################

import config as cf

import numpy as np
import tensorflow as tf

import math

logger = cf.LOGGER

class Generator:
    '''MLP-GANのGeneratorを記述するクラス'''

    def __init__(self, discriminator, global_step):
        self.D = discriminator # 対となるDiscriminator


        self.global_step_tensor = global_step

        self.latent = None # 入力のlatent vector
        self.output = None # 出力のピクセルデータ
        self.loss = None # 損失関数
        self.optimizer = None # オプティマイザ
        self.train_op = None # 学習オペレーション

        np.random.seed()

    def define_forward(self):
        '''latent vectorを与えてforward propagationを行い、
        生成画像を返すTFの計算'''

        with tf.variable_scope('G_network', reuse=tf.AUTO_REUSE):

            ncons = 100.0
            nlayers = round(math.log2(cf.PIXELSIZE * cf.PIXELSIZE / cf.LATENT_VECTOR_SIZE)) + 2
            
            layers = []
            for i in range(nlayers):
                if i == 0:
                    pin = self.latent
                    #shape(minibatch, cf.LATENT_VECTOR_SIZE) 値域[0-1]
                else:
                    pin = layers[-1]
                nunits = cf.LATENT_VECTOR_SIZE * (2 ** (i+1))
                perceptron_layer = tf.layers.dense(inputs = pin,
                                                   units =nunits,
                                                   activation=tf.nn.leaky_relu,
                                                   kernel_initializer=tf.initializers.random_uniform(minval=-ncons/nunits, maxval=ncons/nunits),
                                                   name='G_perceptron_layer_' + str(i))
                bn = tf.layers.batch_normalization(inputs = perceptron_layer, name='G_batchnormalization_' + str(i))
                dropout = tf.layers.dropout(inputs=bn,
                                            rate=cf.DROPOUT_RATE, name='G_dropout_' + str(i))

                layers.append(dropout)
                logger.info('layers[' + str(i) + ']: ' + str(layers[-1].shape))

            # ver. 5 ネットワーク規模の拡大
            nilayers = 8
            nicons = 100.0
            iunits = 4096
            ilayers = []
            for i in range(nilayers):
                if i == 0:
                    iin = layers[-1]
                else:
                    iin = ilayers[-1]
            
                interlayer = tf.layers.dense(inputs = iin,
                                             units = iunits,
                                             activation=tf.nn.leaky_relu,
                                             kernel_initializer = tf.initializers.random_uniform(minval=-nicons/iunits, maxval=nicons/iunits),
                                             name='G_interlayer_' + str(i))
                bn = tf.layers.batch_normalization(inputs = interlayer,
                                                   name = 'G_inter_batchnormalization_' + str(i))
                dropout = tf.layers.dropout(inputs = bn,
                                            rate = cf.DROPOUT_RATE,
                                            name='G_inter_dropout_' + str(i))
                ilayers.append(dropout)

            nunits = cf.PIXELSIZE * cf.PIXELSIZE
            fully_connected = tf.layers.dense(inputs = ilayers[-1],
                                              units=nunits,
                                              kernel_initializer=tf.initializers.random_uniform(minval=-ncons/nunits, maxval =ncons/nunits),
                                              activation=tf.nn.tanh,
                                              name='G_fully_connected')

            constant1 = tf.constant(255.0 / 2.0, dtype=tf.float32, shape=(cf.MINIBATCHSIZE, cf.PIXELSIZE * cf.PIXELSIZE))
            constant2 = tf.ones(shape=(cf.MINIBATCHSIZE, cf.PIXELSIZE * cf.PIXELSIZE))

            preout = tf.multiply(tf.add(fully_connected, constant2), constant1, name='G_preout')

            self.output = tf.reshape(preout,
                                     shape=(-1, cf.PIXELSIZE, cf.PIXELSIZE),
                                     name='G_output')

    def define_graph(self):
        '''generatorのネットワークを記述する'''

        self.latent = tf.placeholder(dtype=tf.float32, shape=(None, cf.LATENT_VECTOR_SIZE), name='G_latent_vector')
        logger.info('latent vector: ' + str(self.latent.shape))


        self.define_forward()

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=cf.LEARNING_RATE, name='G_optimizer')

        self.D.set_input_from_generator(self)
        return


    def define_graph_postD(self):

        self.loss = tf.reduce_mean(-tf.log(0.0001 + self.D.p_fake_for_loss), name='G_loss')
        G_vars = [x for x in tf.trainable_variables() if 'G_' in x.name]
        logger.info('G_vars: ' + str(len(G_vars)))
        logger.info('trainable_variables: ' + str(tf.trainable_variables()))
        logger.info('G_vars: ' + str(G_vars))
        print(G_vars)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor, var_list=G_vars, name='G_train_op')
        self.mean_D_score = tf.reduce_mean(self.D.p_fake)
        return

    @staticmethod
    def generate_latent_vector():
        '''numpy形式でlatent vectorをランダム生成する
        出力の値域は[0, 1]'''
        return np.random.rand(1, cf.LATENT_VECTOR_SIZE)
        

