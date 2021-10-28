# -*-coding:utf-8-*-

import sys

sys.path.append('.')
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import numpy as np
import cv2

from lib.dataset.dataietr import DataIter
from lib.core.model.net.pruned_ssd import DSFD
from pruned_train_config import config as cfg

from lib.helper.logger import logger


class trainner():
    def __init__(self):
        self.train_ds = DataIter(cfg.DATA.root_path, cfg.DATA.train_txt_path, training_flag=True)
        self.val_ds = DataIter(cfg.DATA.root_path, cfg.DATA.val_txt_path, training_flag=False)

        self.inputs = []
        self.outputs = []
        self.val_outputs = []
        self.ite_num = 1

        self._graph = tf.Graph()

        self.summaries = []

        self.ema_weights = False

    def get_opt(self):

        with self._graph.as_default():
            ##set the opt there
            global_step = tf.get_variable(
                'global_step', [],
                initializer=tf.constant_initializer(0), dtype=tf.int32, trainable=False)

            # Decay the learning rate
            lr = tf.train.piecewise_constant(global_step,
                                             cfg.TRAIN.lr_decay_every_step,
                                             cfg.TRAIN.lr_value_every_step
                                             )
            opt = tf.train.MomentumOptimizer(lr, momentum=0.9, use_nesterov=False)
            return opt, lr, global_step

    def load_weight(self):

        with self._graph.as_default():

            if cfg.MODEL.continue_train:
                #restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                for v in tf.global_variables():
                    if 'moving_mean' in v.name or 'moving_variance' in v.name:
                        variables_restore.append(v)
                saver2 = tf.train.Saver(variables_restore, max_to_keep=None)
                saver2.restore(self.sess, cfg.MODEL.pretrained_model)

            elif cfg.MODEL.pretrained_model is not None:
                #restore the params
                variables_restore = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=cfg.MODEL.net_structure)

                for v in tf.global_variables():
                    if 'moving_mean' in v.name or 'moving_variance' in v.name:
                        if cfg.MODEL.net_structure in v.name:
                            variables_restore.append(v)
                # print(variables_restore)

                variables_restore_n = [v for v in variables_restore if
                                       'GN' not in v.name]  # Conv2d_1c_1x1 Bottleneck
                # print(variables_restore_n)
                saver2 = tf.train.Saver(variables_restore_n, max_to_keep=None)
                saver2.restore(self.sess, cfg.MODEL.pretrained_model)
            else:
                logger.info('no pretrained model, train from sctrach')
                # Build an initialization operation to run below.

    def frozen(self):
        with self._graph.as_default():

            variables_need_grads = []

            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

            for v in variables:
                training_flag = True
                if cfg.TRAIN.frozen_stages >= 0:

                    if '%s/conv1' % cfg.MODEL.net_structure in v.name:
                        training_flag = False

                for i in range(1, 1 + cfg.TRAIN.frozen_stages):
                    if '%s/block%d' % (cfg.MODEL.net_structure, i) in v.name:
                        training_flag = False
                        break

                if training_flag:
                    variables_need_grads.append(v)
                else:
                    v_stop = tf.stop_gradient(v)
            return variables_need_grads

    def add_summary(self, event):
        self.summaries.append(event)

    def tower_loss(self, scope, images, labels, boxes, L2_reg, training):
        """Calculate the total loss on a single tower running the model.

        Args:
          scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
          images: Images. 4D tensor of shape [batch_size, height, width, 3].
          labels: Labels. 1D tensor of shape [batch_size].

        Returns:
           Tensor of shape [] containing the total loss for a batch of data
        """

        # Build the portion of the Graph calculating the losses. Note that we will
        # assemble the total_loss using a custom function below.
        ssd = DSFD()
        reg_loss, cla_loss, endpoint = ssd.forward(images, boxes, labels, L2_reg, training)
        print(endpoint)
        regularization_losses = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')

        # regularization_losses = tf.reduce_sum(tf.zeros([1]))

        lowest_loss = []
        for num,v in enumerate(endpoint):
            loss = tf.reduce_max(v, axis=[1, 2])

            lowest_k = tf.cast(tf.floor(tf.scalar_mul(cfg.TRAIN.pruned_ratio[num], tf.cast(tf.shape(loss)[1], dtype=tf.float32))),
                               dtype=tf.int32)  # hyperparameter
            sorted_loss = tf.sort(loss)
            sorted_loss = tf.reduce_mean(sorted_loss[:, 0:lowest_k])
            sorted_loss = tf.where(tf.is_nan(sorted_loss), tf.zeros_like(sorted_loss), sorted_loss)
            lowest_loss.append(sorted_loss)
        lowest_loss = tf.add_n(lowest_loss)

        pruned_loss = tf.scalar_mul(cfg.TRAIN.pruned_alpha, lowest_loss)

        return reg_loss, cla_loss, regularization_losses, pruned_loss, endpoint

    def average_gradients(self, tower_grads):
        """Calculate the average gradient for each shared variable across all towers.

        Note that this function provides a synchronization point across all towers.

        Args:
          tower_grads: List of lists of (gradient, variable) tuples. The outer list
            is over individual gradients. The inner list is over the gradient
            calculation for each tower.
        Returns:
           List of pairs of (gradient, variable) where the gradient has been averaged
           across all towers.
        """

        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                try:
                    g = tf.clip_by_value(g, -5., 5.)
                    expanded_g = tf.expand_dims(g, 0)
                except:
                    print(_)
                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def build(self):

        with self._graph.as_default(), tf.device('/cpu:0'):

            # Create an optimizer that performs gradient descent.
            opt, lr, global_step = self.get_opt()

            ##some global placeholder
            L2_reg = tf.placeholder(tf.float32, name="L2_reg")
            training = tf.placeholder(tf.bool, name="training_flag")

            total_loss_to_show = 0.
            images_place_holder_list = []
            labels_place_holder_list = []
            boxes_place_holder_list = []

            weights_initializer = slim.xavier_initializer()
            biases_initializer = tf.constant_initializer(0.)
            biases_regularizer = tf.no_regularizer
            weights_regularizer = tf.contrib.layers.l2_regularizer(L2_reg)

            # Calculate the gradients for each model tower.
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(cfg.TRAIN.num_gpu):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % (i)) as scope:
                            with slim.arg_scope([slim.model_variable, slim.variable], device='/cpu:0'):

                                images_ = tf.placeholder(tf.float32, [None, None, None, 3], name="images")
                                boxes_ = tf.placeholder(tf.float32, [cfg.TRAIN.batch_size, None, 4], name="input_boxes")
                                labels_ = tf.placeholder(tf.int64, [cfg.TRAIN.batch_size, None], name="input_labels")
                                ###total anchor

                                images_place_holder_list.append(images_)
                                labels_place_holder_list.append(labels_)
                                boxes_place_holder_list.append(boxes_)

                                with slim.arg_scope([slim.conv2d, slim.conv2d_in_plane, \
                                                     slim.conv2d_transpose, slim.separable_conv2d,
                                                     slim.fully_connected],
                                                    weights_regularizer=weights_regularizer,
                                                    biases_regularizer=biases_regularizer,
                                                    weights_initializer=weights_initializer,
                                                    biases_initializer=biases_initializer):
                                    reg_loss, cla_loss, l2_loss, pruned_loss, endpoint = self.tower_loss(
                                        scope, images_, labels_, boxes_, L2_reg, training)

                                    ##use muti gpu ,large batch
                                    if i == cfg.TRAIN.num_gpu - 1:
                                        total_loss = tf.add_n([reg_loss, cla_loss, l2_loss, pruned_loss])
                                    else:
                                        total_loss = tf.add_n([reg_loss, cla_loss, pruned_loss])
                                        # raise Exception('no reg loss!!!!')
                                total_loss_to_show += total_loss
                                # Reuse variables for the next tower.
                                tf.get_variable_scope().reuse_variables()

                                ##when use batchnorm, updates operations only from the
                                ## final tower. Ideally, we should grab the updates from all towers
                                # but these stats accumulate extremely fast so we can ignore the
                                #  other stats from the other towers without significant detriment.
                                bn_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)

                                # Retain the summaries from the final tower.
                                self.summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                                ###freeze some params
                                train_var_list = self.frozen()
                                # Calculate the gradients for the batch of data on this CIFAR tower.
                                grads = opt.compute_gradients(total_loss, train_var_list)

                                # Keep track of the gradients across all towers.
                                tower_grads.append(grads)
            # We must calculate the mean of each gradient. Note that this is the
            # synchronization point across all towers.
            grads = self.average_gradients(tower_grads)

            # Add a summary to track the learning rate.
            self.add_summary(tf.summary.scalar('learning_rate', lr))
            self.add_summary(tf.summary.scalar('total_loss', total_loss_to_show))
            self.add_summary(tf.summary.scalar('loc_loss', reg_loss))
            self.add_summary(tf.summary.scalar('cla_loss', cla_loss))
            self.add_summary(tf.summary.scalar('l2_loss', l2_loss))

            # # Add histograms for gradients.
            # for grad, var in grads:
            #     if grad is not None:
            #         self.add_summary(tf.summary.histogram(var.op.name + '/gradients', grad))

            # Apply the gradients to adjust the shared variables.
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            for var in tf.trainable_variables():
                self.add_summary(tf.summary.histogram(var.op.name, var))

            if self.ema_weights:
                # Track the moving averages of all trainable variables.
                variable_averages = tf.train.ExponentialMovingAverage(
                    0.9, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                # Group all updates to into a single train op.
                train_op = tf.group(apply_gradient_op, variables_averages_op, *bn_update_ops)
            else:
                train_op = tf.group(apply_gradient_op, *bn_update_ops)

            ###set inputs and ouputs
            self.inputs = [images_place_holder_list,
                           boxes_place_holder_list,
                           labels_place_holder_list,
                           L2_reg,
                           training]
            self.outputs = [train_op,
                            total_loss_to_show,
                            reg_loss,
                            cla_loss,
                            l2_loss,
                            pruned_loss,
                            endpoint,
                            lr]
            self.val_outputs = [total_loss_to_show,
                                reg_loss,
                                cla_loss,
                                l2_loss,
                                pruned_loss,
                                endpoint,
                                lr]

            tf_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False)
            tf_config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=tf_config)

            ##init all variables
            init = tf.global_variables_initializer()
            self.sess.run(init)
            ######

    def calculate_flops(self, endpoint):
        flops = 0
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var= [i for i in var if not 'ssdout' in i.name]
        var = [i for i in var if 'weights' in i.name]

        for i in range(len(endpoint)):
            if i ==0:
                flops += endpoint[i].shape[1] *var[i].shape[0]*var[i].shape[1] * endpoint[i].shape[2] * 3 * endpoint[i].shape[3]
            else:
                flops += endpoint[i].shape[1]*var[i].shape[0]*var[i].shape[1]*endpoint[i].shape[2]*endpoint[i-1].shape[3]*endpoint[i].shape[3]

        # print(flops)
        return flops

    def calculate_pruned_flops(self, endpoint):
        flops = 0
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var= [i for i in var if not 'ssdout' in i.name]
        var = [i for i in var if 'weights' in i.name]
        # channel
        # [print(np.shape(i)) for i in endpoint]
        og_channel = [np.amax(i, axis=(1, 2)) for i in endpoint]
        # [print(np.shape(i)) for i in og_channel]
        max_endpoint = [np.count_nonzero(i,axis=1) for i in og_channel]
        # og_channel = [np.shape(i)[1] for i in og_channel]
        # print(og_channel)
        # print(max_endpoint)
        avg_channel = [np.average(i) for i in max_endpoint]
        # print(avg_channel)

        for j in range(max_endpoint[0].shape[0]):
            for i in range(len(max_endpoint)):
                if i ==0:
                    flops += endpoint[i].shape[1]*endpoint[i].shape[2]*var[i].shape[0]*var[i].shape[1]*3*max_endpoint[i][j]
                else:
                    flops += endpoint[i].shape[1]*endpoint[i].shape[2]*var[i].shape[0]*var[i].shape[1]*max_endpoint[i-1][j]*max_endpoint[i][j]

        return int(flops)//len(max_endpoint[0])/127822233600 *100, avg_channel

    def calculate_avgpruned_flops(self, endpoint):
        flops = 0
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var= [i for i in var if not 'ssdout' in i.name]
        var = [i for i in var if 'weights' in i.name]
        # channel
        # [print(np.shape(i)) for i in endpoint]
        # if len(endpoint[0].shape) == 1:
        #     avg_channel = endpoint
        # else:
        avg_channel = [np.amax(i, axis=(0, 1, 2)) for i in endpoint]
        # avg_channel = [np.amax(i, axis=(0, 1, 2)) for i in endpoint]
        # [print(np.shape(i)) for i in avg_channel]
        avg_endpoint = [np.count_nonzero(i) for i in avg_channel]
        # og_channel = [np.shape(i)[1] for i in og_channel]
        # print(og_channel)
        # print(avg_endpoint)

        # avg_channel = [np.average(i) for i in max_endpoint]
        # print(avg_channel)

        # for j in range(avg_endpoint[0].shape[0]):
        for i in range(len(avg_endpoint)):
            if i ==0:
                flops += endpoint[i].shape[1]*endpoint[i].shape[2]*var[i].shape[0]*var[i].shape[1]*3*avg_endpoint[i]
            else:
                flops += endpoint[i].shape[1]*endpoint[i].shape[2]*var[i].shape[0]*var[i].shape[1]*avg_endpoint[i-1]*avg_endpoint[i]
        # print(int(flops)/127822233600 *100)
        return int(flops)/127822233600 *100, avg_endpoint

    def calculate_pruned_parameters(self,endpoint):
        num_param = 0
        og_param = 0
        var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var = [i for i in var if not 'ssdout' in i.name]
        var = [i for i in var if 'weights' in i.name]
        # print(var)
        # channel
        # if len(endpoint[0].shape) ==1:
        #     avg_channel = endpoint
        # else:
        avg_channel = [np.amax(i, axis=(0, 1, 2)) for i in endpoint]
            # [print(np.shape(i)[0]) for i in avg_channel]
        avg_endpoint = [np.count_nonzero(i) for i in avg_channel]
        # print(avg_endpoint)
        for i in range(len(avg_endpoint)):
            if i == 0:
                og_param += var[i].shape[0] * var[i].shape[1] * 3 * np.shape(avg_channel[i])[0]
                num_param += var[i].shape[0] * var[i].shape[1] * np.shape(avg_channel[i-1])[0] * np.shape(avg_channel[i])[0]

            else:
                og_param +=var[i].shape[0] * var[i].shape[1] * np.shape(avg_channel[i-1])[0] * np.shape(avg_channel[i])[0]
                num_param += var[i].shape[0] * var[i].shape[1] * avg_endpoint[i - 1] * avg_endpoint[i]

        return int(num_param)/int(og_param)*100

    def train_loop(self):
        """Train faces data for a number of epoch."""

        self.build()
        self.load_weight()

        with self._graph.as_default():
            # Create a saver.
            self.saver = tf.train.Saver(tf.global_variables())

            # Build the summary operation from the last tower summaries.
            self.summary_op = tf.summary.merge(self.summaries)

            self.summary_writer = tf.summary.FileWriter(cfg.MODEL.model_path, self.sess.graph)

            min_loss_control = 1000.
            for epoch in range(cfg.TRAIN.epoch):
                self._train(epoch)
                val_loss, val_param, val_mac, val_channel = self._val(epoch)
                logger.info('**************'
                            'val_loss %f val_param %f val_mac %f ' % (val_loss, val_param, val_mac))
                logger.info('/'.join(map(str, val_channel)))

                # tmp_model_name=cfg.MODEL.model_path + \
                #               'epoch_' + str(epoch ) + \
                #               'L2_' + str(cfg.TRAIN.weight_decay_factor) + \
                #               '.ckpt'
                # logger.info('save model as %s \n'%tmp_model_name)
                # self.saver.save(self.sess, save_path=tmp_model_name)
                save_model_name = cfg.MODEL.model_path + \
                                  'epoch_' + str(epoch) + '_val_loss_'+str(val_loss)+'_val_mac_'+str(val_mac)+'.ckpt'
                self.saver.save(self.sess, save_path=save_model_name)
                # 낮을 때만 save하는 코드 지우고 개수제한 풀기
                if min_loss_control > val_loss:
                    min_loss_control = val_loss

                    low_loss_model_name = cfg.MODEL.model_path + '/optimized/' + \
                                          'epoch_' + str(epoch) + \
                                          '_val_loss_' + str(min_loss_control) +'_val_mac_' +str(val_mac)+'.ckpt'
                    logger.info('A new low loss model  saved as %s \n' % low_loss_model_name)
                    self.saver.save(self.sess, save_path=low_loss_model_name)

            self.sess.close()

    def _train(self, _epoch):
        for step in range(cfg.TRAIN.iter_num_per_epoch):
            self.ite_num += 1

            ########show_flag check the data
            if cfg.TRAIN.vis:
                example_image, example_all_boxes, example_all_labels = next(self.train_ds)

                print(example_all_boxes.shape)
                for i in range(example_all_boxes.shape[0]):
                    img = example_image[i]
                    box_encode = example_all_boxes[i]
                    label = example_all_labels[i]

                    print(np.sum(label[label > 0]))
                    # for j in range(label.shape[0]):
                    #     if label[j]>0:
                    #         print(box_encode[j])
                    cv2.namedWindow('img', 0)
                    cv2.imshow('img', img)
                    cv2.waitKey(0)

            else:

                start_time = time.time()
                feed_dict = {}
                examples = next(self.train_ds)
                for n in range(cfg.TRAIN.num_gpu):
                    feed_dict[self.inputs[0][n]] = examples[0][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
                    feed_dict[self.inputs[1][n]] = examples[1][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
                    feed_dict[self.inputs[2][n]] = examples[2][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]

                feed_dict[self.inputs[3]] = cfg.TRAIN.weight_decay_factor
                feed_dict[self.inputs[4]] = True

                fetch_duration = time.time() - start_time

                start_time2 = time.time()
                _, total_loss_value, reg_loss_value, cla_loss_value, l2_loss_value,pruned_loss_value, endpoint_value, lr_value = \
                    self.sess.run([*self.outputs],
                                  feed_dict=feed_dict)

                duration = time.time() - start_time2
                run_duration = duration
                if self.ite_num % cfg.TRAIN.log_interval == 0:
                    num_examples_per_step = cfg.TRAIN.batch_size * cfg.TRAIN.num_gpu
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = duration / cfg.TRAIN.num_gpu
                    # pruned flops and parameters
                    # print(endpoint_value[0].shape)
                    # print(len(endpoint_value[0].shape))
                    pruned_param = self.calculate_pruned_parameters(endpoint_value)
                    avg_mac, avg_channel = self.calculate_avgpruned_flops(endpoint_value)

                    format_str = ('epoch %d: iter %d, '
                                  'total_loss=%.6f '
                                  'reg_loss=%.6f '
                                  'cla_loss=%.6f '
                                  'pruned_loss=%.6f '
                                  'avgMAC ratio=%.4f '
                                  'pruned_param=%.4f '
                                  'l2_loss=%.6f '
                                  'learning rate =%e '
                                  '(%.1f examples/sec; %.3f sec/batch) \navg_channel=%s'

                                  )
                    logger.info(format_str % (_epoch,
                                              self.ite_num,
                                              total_loss_value,
                                              reg_loss_value,
                                              cla_loss_value,
                                              pruned_loss_value,
                                              avg_mac, pruned_param,
                                              l2_loss_value,
                                              lr_value,
                                              examples_per_sec,
                                              sec_per_batch, '/'.join(map(str, avg_channel)) ))

                if self.ite_num % 100 == 0:
                    summary_str = self.sess.run(self.summary_op, feed_dict=feed_dict)
                    self.summary_writer.add_summary(summary_str, self.ite_num)

    def _val(self, _epoch):

        all_total_loss = 0

        for step in range(cfg.TRAIN.val_iter):

            feed_dict = {}
            examples = next(self.val_ds)
            for n in range(cfg.TRAIN.num_gpu):
                feed_dict[self.inputs[0][n]] = examples[0][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
                feed_dict[self.inputs[1][n]] = examples[1][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]
                feed_dict[self.inputs[2][n]] = examples[2][n * cfg.TRAIN.batch_size:(n + 1) * cfg.TRAIN.batch_size]

            feed_dict[self.inputs[3]] = 0
            feed_dict[self.inputs[4]] = False

            total_loss_value, reg_loss_value, cla_loss_value, l2_loss_value, pruned_loss_value, endpoint_value, lr_value = \
                self.sess.run([*self.val_outputs],
                              feed_dict=feed_dict)
            if step==0:
                endpoint_list = endpoint_value
            else:
                for i in range(len(endpoint_list)):
                    endpoint_list[i]+= endpoint_value[i]
            all_total_loss += total_loss_value

        pruned_param = self.calculate_pruned_parameters(endpoint_list)

        avg_mac, avg_channel = self.calculate_avgpruned_flops(endpoint_list)
        return all_total_loss / cfg.TRAIN.val_iter , pruned_param, avg_mac, avg_channel

    def train(self):
        self.train_loop()






