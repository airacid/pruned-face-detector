#-*-coding:utf-8-*-

import os
import numpy as np
from easydict import EasyDict as edict

config = edict()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"          ##if u use muti gpu set them visiable there and then set config.TRAIN.num_gpu

config.TRAIN = edict()

#### below are params for dataiter
config.TRAIN.process_num = 3                      ### process_num for data provider
config.TRAIN.prefetch_size = 100                  ### prefect Q size for data provider

config.TRAIN.num_gpu = 1                         ##match with   os.environ["CUDA_VISIBLE_DEVICES"]
config.TRAIN.log_interval = 10

config.MODEL = edict()
#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#

config.TRAIN.epoch = 10                                         # <20 epoch
config.TRAIN.batch_size = 2
config.TRAIN.lr_value_every_step = [0.001,0.001,0.0005]     # learning rate value
config.TRAIN.lr_decay_every_step = [1000,10000]             # learning rate decay step
config.MODEL.model_path = './face_detector_pruned/'                     # save directory
config.MODEL.pretrained_model='./face_detector/optimized/epoch_5_val_loss_3.ckpt' # 기존에 학습한 face detector 모델의 ckpt 파일 경로로 설정
config.TRAIN.pruned_alpha = 1.                              # alpha
config.TRAIN.pruned_ratio = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.] # pruned ratio


#---------------------------------------------------------------------------------------------#
#---------------------------------------------------------------------------------------------#
config.TRAIN.weight_decay_factor = 5.e-4
config.TRAIN.train_set_size=13000              ###widerface train size
config.TRAIN.val_set_size=3000                 ###widerface val size

config.TRAIN.iter_num_per_epoch = config.TRAIN.train_set_size // config.TRAIN.num_gpu // config.TRAIN.batch_size
config.TRAIN.val_iter=config.TRAIN.val_set_size// config.TRAIN.num_gpu // config.TRAIN.batch_size

config.TRAIN.vis=False                                    ##check data flag

config.TRAIN.norm='BN'    ##'GN' OR 'BN'
config.TRAIN.lock_basenet_bn=False
config.TRAIN.frozen_stages=-1   ##no freeze

config.TEST = edict()
config.TEST.parallel_iterations=8
config.TEST.score_thres = 0.05
config.TEST.iou_thres = 0.3
config.TEST.max_detect = 1500

config.DATA = edict()
config.DATA.root_path=''
config.DATA.train_txt_path='train.txt'
config.DATA.val_txt_path='val.txt'
config.DATA.num_category=1                                  ###face 1  voc 20 coco 80
config.DATA.num_class = config.DATA.num_category + 1        # +1 background

config.DATA.PIXEL_MEAN = [123., 116., 103.]                 ###rgb
config.DATA.PIXEL_STD = [58., 57., 57.]

config.DATA.hin = 640  # input size
config.DATA.win = 640
config.DATA.max_size=[config.DATA.hin,config.DATA.win]  ##h,w
config.DATA.cover_small_face=5                          ###cover the small faces

config.DATA.mutiscale=False                #if muti scale set False  then config.DATA.MAX_SIZE will be the inputsize
config.DATA.scales=(320,640)

# anchors -------------------------
config.ANCHOR = edict()
config.ANCHOR.rect=True
config.ANCHOR.rect_longer=True       ####    make anchor h/w=1.5
config.ANCHOR.ANCHOR_STRIDE = 16
config.ANCHOR.ANCHOR_SIZES = (16,32,64, 128, 256, 512)   # sqrtarea of the anchor box
config.ANCHOR.ANCHOR_STRIDES = (4, 8,16, 32, 64, 128)  # strides for each FPN level. Must be the same length as ANCHOR_SIZES
config.ANCHOR.ANCHOR_RATIOS = (1., 4.) ######           1:2 in size,
config.ANCHOR.POSITIVE_ANCHOR_THRESH = 0.35
config.ANCHOR.NEGATIVE_ANCHOR_THRESH = 0.35
config.ANCHOR.AVG_MATCHES=20
config.ANCHOR.super_match=True

from lib.core.anchor.anchor import Anchor

config.ANCHOR.achor=Anchor()


#vgg as basemodel. if vgg, set config.TRAIN.norm ='None', achieves fddb 0.987

config.TRAIN.norm='None'
config.MODEL.l2_norm=[10,8,5]
config.MODEL.continue_train=True ### revover from a trained model
config.MODEL.net_structure='vgg_16'

config.MODEL.fpn_dims=[256,512,512,1024,512,256]
config.MODEL.fem_dims=512



config.MODEL.fpn=False
config.MODEL.dual_mode=True
config.MODEL.maxout=True
config.MODEL.max_negatives_per_positive= 3.0

