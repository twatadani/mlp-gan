# -*- coding: utf-8 -*-

##### config.py #####
# MLP-GAN用のコンフィグファイル
#
# created 2018/9/28 Takeyuki Watadani @ UT Radiology
##########
import numpy as np

# このセッションのデスクリプション
DESC = 'MLP-GAN-ver5'
DESC += '-01-lungHRCT-interlayer'

# 結果格納用ディレクトリ
RESULTDIR = './results/'

# データセット格納用のパス
DATASET_PATH = './datasets/'

# 教師用データセットとなる画像を保存しているパス
TRAIN_DATA_PATH = '/Users/watadani/OneDrive/src/lungHRCT_jpg/'

# 画像の拡張子
TRAIN_DATA_EXT = '.jpg'

# 教師用データセットを作成する画像数
TRAIN_DATA_NUMBER = 20000

# 今回用いる画像のピクセルサイズ
PIXELSIZE = 32

# 何ミニバッチごとにサンプルを保存するか
SAVE_SAMPLE_MINIBATCH = 250

# 教師用 保存データセットのプレフィクス
TRAIN_PREFIX = 'HRCT32'

# NNのドロップアウト率
DROPOUT_RATE = 0.5

# 学習のミニバッチサイズ
MINIBATCHSIZE = 32

# 学習のmax epoch
MAXEPOCH = 10000

# latent vectorの次元数
LATENT_VECTOR_SIZE = 128

# 学習が偏ったときにGとDの片方だけを集中学習するかどうか 基本はFalse
USE_INTENSIVE_TRAINING = False

# 学習率
LEARNING_RATE = 0.0012

# キューの設定 (通常は変更不要)
QUEUEMAX = 150
QUEUEBUFSIZE = 50
QUEUEBATCHSIZE = 20

# スレッドのスリープ間隔 (通常は変更不要)
SLEEP_INTERVAL = 0.8

# ログ用の設定
import logging
import os.path
from datetime import datetime
LOGGER = logging.getLogger(DESC)
LOGGER.setLevel(10)
logdir = os.path.join(RESULTDIR, DESC)
if not os.path.exists(logdir):
    os.makedirs(logdir)
logfilename = 'log-'
now = datetime.now()
logfilename += str(now.year) + str(now.month) + str(now.day) + '-' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.txt'
logfile = os.path.join(logdir, logfilename)
if os.path.exists(logfile):
    os.remove(logfile)
fhandler = logging.FileHandler(logfile)
LOGGER.addHandler(fhandler)
shandler = logging.StreamHandler()
LOGGER.addHandler(shandler)


