import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.core.model.net.arg_scope.resnet_args_cope import resnet_arg_scope

from lib.core.model.net.vgg.pruned_vgg import vgg_16
from lib.core.model.net.FEM import create_fem_net

from train_config import config as cfg

def l2_normalization(x, scale, name):
    with tf.variable_scope(name):
        x = scale*tf.nn.l2_normalize(x, axis=-1)
    return x

def extra_feature(x, endpoint):

    extra_fm=[]
    with tf.variable_scope('extra'):
        x=slim.conv2d(x, 256, [1, 1],stride=1,
                    activation_fn=tf.nn.relu,
                    normalizer_fn=None,
                    scope='extra_conv1')
        endpoint.append(x)
        x = slim.conv2d(x, 512, [3, 3],stride=2,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        scope='extra_conv2')
        endpoint.append(x)
        extra_fm.append(x)

        x = slim.conv2d(x, 128, [1, 1], stride=1,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        scope='extra_conv3')
        endpoint.append(x)
        x = slim.conv2d(x, 256, [3, 3], stride=2,
                        activation_fn=tf.nn.relu,
                        normalizer_fn=None,
                        scope='extra_conv4')
        endpoint.append(x)
        extra_fm.append(x)

    return extra_fm, endpoint


def vgg_ssd(image,L2_reg,is_training=True):
    with slim.arg_scope(resnet_arg_scope(weight_decay=L2_reg, bn_is_training=is_training, bn_trainable=True)):

        net,end_points=vgg_16(image,num_classes=None,global_pool=False,spatial_squeeze=False,fc_conv_padding='SAME')
        endpoint = []
        for k,v in end_points.items():
            # print('mobile backbone output:',k,v)
            endpoint.append(v)
        ###add conv6, conv7
        conv6 = slim.conv2d(net, 1024, [3, 3],
                          activation_fn=tf.nn.relu,
                          normalizer_fn=None,
                          scope='fc6')
        endpoint.append(conv6)
        conv7 = slim.conv2d(conv6, 1024, [1, 1],
                          activation_fn=tf.nn.relu,
                          normalizer_fn=None,
                          scope='fc7')
        endpoint.append(conv7)
        extra_fms, endpoint=extra_feature(conv7, endpoint)

        vgg_fms = [end_points['vgg_16/conv3/conv3_3'],
                   end_points['vgg_16/conv4/conv4_3'],
                   end_points['vgg_16/conv5/conv5_3'],
                   conv7]+extra_fms


        # print(vgg_fms)

        vgg_fms[0] = l2_normalization(vgg_fms[0], scale=cfg.MODEL.l2_norm[0], name='of0')
        vgg_fms[1] = l2_normalization(vgg_fms[1], scale=cfg.MODEL.l2_norm[1], name='of1')
        vgg_fms[2] = l2_normalization(vgg_fms[2], scale=cfg.MODEL.l2_norm[2], name='of2')
        if cfg.MODEL.fpn:

            enhanced_fms = create_fem_net(vgg_fms, L2_reg, is_training)

            enhanced_fms[0] = l2_normalization(enhanced_fms[0], scale=cfg.MODEL.l2_norm[0], name='ef0')
            enhanced_fms[1] = l2_normalization(enhanced_fms[1], scale=cfg.MODEL.l2_norm[1], name='ef1')
            enhanced_fms[2] = l2_normalization(enhanced_fms[2], scale=cfg.MODEL.l2_norm[2], name='ef2')
        else:
            enhanced_fms=None
    return vgg_fms,enhanced_fms, endpoint
