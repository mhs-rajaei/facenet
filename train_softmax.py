"""Training a face recognizer with TensorFlow using softmax cross entropy loss
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import argparse
from scipy import misc

import h5py
import math
import tensorflow.contrib.slim as slim

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

from importlib.machinery import SourceFileLoader
facenet = SourceFileLoader('facenet', os.path.join(PROJECT_PATH, 'facenet.py')).load_module()
lfw = SourceFileLoader('lfw', os.path.join(PROJECT_PATH, 'lfw.py')).load_module()
network = SourceFileLoader('inception_resnet_v1', os.path.join(PROJECT_PATH, 'models/inception_resnet_v1.py')).load_module()
verification = SourceFileLoader('verification', os.path.join(PROJECT_PATH, 'verification.py')).load_module()
eval_data_reader = SourceFileLoader('eval_data_reader', os.path.join(PROJECT_PATH, 'data/eval_data_reader.py')).load_module()

log_path = os.path.join(PROJECT_PATH, 'logs')
models_path = os.path.join(PROJECT_PATH, 'models')
# train_dataset_path = r'F:\Documents\JetBrains\PyCharm\OFR\images\1024First_lfw_160'
# train_dataset_path = r'F:\Documents\JetBrains\PyCharm\OFR\images\200_lfw_192'
train_dataset_path = r'F:\Documents\JetBrains\PyCharm\OFR\images\200END_lfw_160_train'
# eval_dir_path = r'F:\Documents\JetBrains\PyCharm\OFR\images\lfw_160'
eval_dir_path = r'F:\Documents\JetBrains\PyCharm\OFR\images\200END_lfw_160_test_Copy'
learning_rate_schedule_decay_path = os.path.join(PROJECT_PATH, 'data/learning_rate_schedule_classifier_casia.txt')
eval_pairs_path = os.path.join(PROJECT_PATH, 'data/200END_lfw_160_test_Copy_pairs.txt')
print(PROJECT_PATH)
print(log_path)
print(models_path)
print(train_dataset_path)
print(learning_rate_schedule_decay_path)

class Args:
    prelogits_norm_p = 1.0
    prelogits_norm_loss_factor = 5e-4
    center_loss_alfa = 0.95
    center_loss_factor = 0.0
    epoch_size = 2
    learning_rate_decay_factor = 1.0
    learning_rate_decay_epochs = 100
    moving_average_decay = 0.9999
    optimizer = 'ADAM'
    log_histograms = True
    gpu_memory_fraction = 1.0
    validate_every_n_epochs =1
    max_nrof_epochs = 1000
    use_fixed_image_standardization = True
    eval_dir = eval_dir_path

    seed = 313
    image_size = 160
    batch_size = 8
    learning_rate = -1

    embedding_size = 512
    keep_probability = 0.4
    weight_decay = 5e-4

    logs_base_dir = log_path

    models_base_dir = models_path

    data_dir = train_dataset_path

    nrof_preprocess_threads = 10

    filter_filename = False

    validation_set_split_ratio = 0.07

    pretrained_model = False

    learning_rate_schedule_file = learning_rate_schedule_decay_path
    random_rotate = True
    random_crop = True
    random_flip = True

    min_nrof_val_images_per_class = 0
    prelogits_hist_max = 10.0

    alpha = 0.2
    people_per_batch = 45
    images_per_person = 40

    eval_distance_metric = 1
    eval_use_flipped_images = True
    eval_subtract_mean = True
    eval_nrof_folds = 2

    eval_dataset = eval_dir_path
    eval_pair = eval_pairs_path
    save = False

def main(args):
  
    subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), subdir)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)

    stat_file_name = os.path.join(log_dir, 'stat.h5')

    # Write arguments to a text file
    facenet.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    facenet.store_revision_info(PROJECT_PATH, log_dir, ' '.join(sys.argv))

    np.random.seed(seed=args.seed)
    random.seed(args.seed)
    dataset = facenet.get_dataset(args.data_dir)

    if args.filter_filename:
        dataset = filter_dataset(dataset, os.path.expanduser(args.filter_filename), 
            args.filter_percentile, args.filter_min_nrof_images_per_class)
        
    if args.validation_set_split_ratio > 0.0:
        train_set, val_set = facenet.split_dataset(dataset, args.validation_set_split_ratio, args.min_nrof_val_images_per_class, 'SPLIT_IMAGES')
    else:
        train_set, val_set = dataset, []
        
    nrof_classes = len(train_set)
    
    print('Model directory: %s' % model_dir)
    print('Log directory: %s' % log_dir)
    pre_trained_model = None
    if args.pretrained_model:
        pre_trained_model = os.path.expanduser(args.pretrained_model)
        print('Pre-trained model: %s' % pre_trained_model)
    
    if args.eval_dir:
        print('Evaluation directory: %s' % args.eval_dir)
        # Read the file containing the pairs used for testing
        pairs = lfw.read_pairs(os.path.expanduser(args.eval_pair))
        # Get the paths for the corresponding images
        eval_paths, eval_actual_issame = lfw.get_paths(os.path.expanduser(args.eval_dir), pairs)

        # 2.2 prepare custom validate dataset
        ver_list = []
        ver_name_list = []
        print('begin db %s convert.' % args.eval_dataset)
        data_set = eval_data_reader.load_eval_datasets(args)
        ver_list.append(data_set)
        ver_name_list.append(args.eval_dataset)


    with tf.Graph().as_default():
        tf.set_random_seed(args.seed)
        global_step = tf.Variable(0, trainable=False)

        # Get a list of image paths and their labels
        image_list, _label_list, name_dict, index_dict = facenet.get_image_paths_and_labels(train_set, path=args.data_dir)
        assert len(image_list) > 0, 'The training set should not be empty'

        label_list = []
        if os.name == 'nt':
            for path in image_list:
                label_list.append(int(name_dict[path.split('\\')[len(args.data_dir.split('\\'))]]))
        else:
            for path in image_list:
                label_list.append(int(name_dict[path.split('/')[len(args.data_dir.split('/'))]]))

        # tf_gen_dataset_perfarmance(image_list, label_list, args.nrof_preprocess_threads, args.image_size, BATCH_SIZE=32, repeat_count=-1)
        tf_dataset_train = facenet.tf_gen_dataset(image_list, label_list, args.nrof_preprocess_threads, args.image_size,
                                   method='cache_slices', BATCH_SIZE=args.batch_size, repeat_count=1)

        val_image_list, _val_label_list = facenet.get_image_paths_and_labels(val_set)

        val_label_list = []
        if os.name == 'nt':
            for path in val_image_list:
                val_label_list.append(int(name_dict[path.split('\\')[len(args.data_dir.split('\\'))]]))
        else:
            for path in val_image_list:
                val_label_list.append(int(name_dict[path.split('/')[len(args.data_dir.split('/'))]]))

        tf_dataset_validation = facenet.tf_gen_dataset(val_image_list, val_label_list, args.nrof_preprocess_threads, args.image_size,
                                                       method='cache_slices', BATCH_SIZE=len(val_image_list), repeat_count=1)

        # Create a queue that produces indices into the image_list and label_list 
        # labels = ops.convert_to_tensor(label_list, dtype=tf.int32)
        # range_size = array_ops.shape(labels)[0]
        # index_queue = tf.train.range_input_producer(range_size, num_epochs=None,
        #                      shuffle=True, seed=None, capacity=32)
        #
        # index_dequeue_op = index_queue.dequeue_many(args.batch_size*args.epoch_size, 'index_dequeue')

        learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        # batch_size_placeholder = tf.placeholder(tf.int32, name='batch_size')
        phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
        # image_paths_placeholder = tf.placeholder(tf.string, shape=(None,1), name='image_paths')
        # labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
        # control_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='control')

        # img = misc.imread(image_list[0])
        # _shape = list(img.shape)
        # _shape.insert(0, None)
        # image_batch = tf.placeholder(shape=(_shape), dtype=tf.float32)
        image_batch = tf.placeholder(shape=([None, args.image_size, args.image_size, 3]), dtype=tf.float32)

        label_batch = tf.placeholder(dtype=tf.int32, shape=[None, ])

        # nrof_preprocess_threads = 4
        # input_queue = data_flow_ops.FIFOQueue(capacity=2000000,
        #                             dtypes=[tf.string, tf.int32, tf.int32],
        #                             shapes=[(1,), (1,), (1,)],
        #                             shared_name=None, name=None)
        # enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder, control_placeholder], name='enqueue_op')
        # image_batch, label_batch = facenet.create_input_pipeline(input_queue, image_size, nrof_preprocess_threads, batch_size_placeholder)
        #
        # image_batch = tf.identity(image_batch, 'image_batch')
        # image_batch = tf.identity(image_batch, 'input')
        # label_batch = tf.identity(label_batch, 'label_batch')
        
        print('Number of classes in training set: %d' % nrof_classes)
        print('Number of examples in training set: %d' % len(image_list))

        print('Number of classes in validation set: %d' % len(val_set))
        print('Number of examples in validation set: %d' % len(val_image_list))
        
        print('Building training graph')
        
        # Build the inference graph
        prelogits, _ = network.inference(image_batch, args.keep_probability, 
            phase_train=phase_train_placeholder, bottleneck_layer_size=args.embedding_size, 
            weight_decay=args.weight_decay)
        logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None, 
                weights_initializer=slim.initializers.xavier_initializer(), 
                weights_regularizer=slim.l2_regularizer(args.weight_decay),
                scope='Logits', reuse=False)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

        # Norm for the prelogits
        eps = 1e-4
        prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits)+eps, ord=args.prelogits_norm_p, axis=1))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * args.prelogits_norm_loss_factor)

        # Add center loss
        prelogits_center_loss, _ = facenet.center_loss(prelogits, label_batch, args.center_loss_alfa, nrof_classes)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * args.center_loss_factor)

        learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,
            args.learning_rate_decay_epochs*args.epoch_size, args.learning_rate_decay_factor, staircase=True)
        tf.summary.scalar('learning_rate', learning_rate)

        # Calculate the average cross entropy loss across the batch
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=label_batch, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)
        
        correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(label_batch, tf.int64)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)
        
        # Calculate the total losses
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')

        # Build a Graph that trains the model with one batch of examples and updates the model parameters
        train_op = facenet.train(total_loss, global_step, args.optimizer, 
            learning_rate, args.moving_average_decay, tf.global_variables(), args.log_histograms)
        
        # Create a saver
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Start running operations on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        # coord = tf.train.Coordinator()
        # tf.train.start_queue_runners(coord=coord, sess=sess)

        with sess.as_default():

            if pre_trained_model:
                print('Restoring pretrained model: %s' % pre_trained_model)
                saver.restore(sess, pre_trained_model)

            # Training and validation loop
            print('Running training')

            args.epoch_size = int(np.ceil(len(image_list) / args.batch_size))
            nrof_steps = args.max_nrof_epochs * args.epoch_size
            nrof_val_samples = int(math.ceil(args.max_nrof_epochs / args.validate_every_n_epochs))   # Validate every validate_every_n_epochs
            stat = {
                'loss': np.zeros((nrof_steps,), np.float32),
                'center_loss': np.zeros((nrof_steps,), np.float32),
                'reg_loss': np.zeros((nrof_steps,), np.float32),
                'xent_loss': np.zeros((nrof_steps,), np.float32),
                'prelogits_norm': np.zeros((nrof_steps,), np.float32),
                'accuracy': np.zeros((nrof_steps,), np.float32),
                'val_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_xent_loss': np.zeros((nrof_val_samples,), np.float32),
                'val_accuracy': np.zeros((nrof_val_samples,), np.float32),
                'eval_accuracy': np.zeros((args.max_nrof_epochs,), np.float32),
                'eval_valrate': np.zeros((args.max_nrof_epochs,), np.float32),
                'learning_rate': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_train': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_validate': np.zeros((args.max_nrof_epochs,), np.float32),
                'time_evaluate': np.zeros((args.max_nrof_epochs,), np.float32),
                'prelogits_hist': np.zeros((args.max_nrof_epochs, 1000), np.float32),
              }

            train_iterator = tf_dataset_train.make_initializable_iterator()
            train_next_element = train_iterator.get_next()
            sess.run(train_iterator.initializer)

            validation_iterator = tf_dataset_validation.make_initializable_iterator()
            validation_next_element = validation_iterator.get_next()
            sess.run(validation_iterator.initializer)

            for epoch in range(1, args.max_nrof_epochs+1):
                step = sess.run(global_step, feed_dict=None)
                # Train for one epoch
                t = time.time()

                # cont = train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
                #     learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, global_step,
                #     total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file,
                #     stat, cross_entropy_mean, accuracy, learning_rate,
                #     prelogits, prelogits_center_loss, args.random_rotate, args.random_crop, args.random_flip, prelogits_norm, args.prelogits_hist_max,
                #              args.use_fixed_image_standardization)

                cont = train(args, sess, epoch, train_next_element, train_iterator, image_batch, label_batch, image_list,
                             label_list, learning_rate_placeholder, phase_train_placeholder, global_step,
                             total_loss, train_op, summary_op, summary_writer, regularization_losses, args.learning_rate_schedule_file,
                             stat, cross_entropy_mean, accuracy, learning_rate,
                             prelogits, prelogits_center_loss, args.random_rotate, args.random_crop, args.random_flip, prelogits_norm,
                             args.prelogits_hist_max, args.use_fixed_image_standardization)

                stat['time_train'][epoch-1] = time.time() - t
                
                if not cont:
                    break
                  
                t = time.time()
                if len(val_image_list)>0 and ((epoch-1) % args.validate_every_n_epochs == args.validate_every_n_epochs-1 or epoch==args.max_nrof_epochs):
                    # validate(args, sess, epoch, val_image_list, val_label_list, enqueue_op, image_paths_placeholder, labels_placeholder,
                    #          control_placeholder, phase_train_placeholder, batch_size_placeholder, stat, total_loss, regularization_losses,
                    #          cross_entropy_mean, accuracy, args.validate_every_n_epochs, args.use_fixed_image_standardization)

                    validate(name_dict, index_dict, args, sess, epoch, validation_next_element, validation_iterator,
                             embeddings, image_batch, label_batch, val_image_list, val_label_list, phase_train_placeholder, stat, total_loss,
                             regularization_losses, cross_entropy_mean, accuracy, args.validate_every_n_epochs,
                             args.use_fixed_image_standardization)

                stat['time_validate'][epoch-1] = time.time() - t
                if args.save:
                    # Save variables and the metagraph if it doesn't exist already
                    save_variables_and_metagraph(sess, saver, summary_writer, model_dir, subdir, epoch)

                # Evaluate on LFW
                t = time.time()
                if args.eval_dir:
                    # evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder,
                    #          control_placeholder, embeddings, label_batch, eval_paths, actual_issame, args.eval_batch_size, args.eval_nrof_folds,
                    #          log_dir, step, summary_writer, stat, epoch, args.eval_distance_metric, args.eval_subtract_mean,
                    #          args.eval_use_flipped_images, args.use_fixed_image_standardization)

                    # results = evaluate(sess, phase_train_placeholder, embeddings, label_batch, eval_paths, eval_actual_issame, args.eval_batch_size,
                    #                    args.eval_nrof_folds, log_dir, step, summary_writer, stat, epoch,
                    #                    args.eval_distance_metric, args.eval_subtract_mean, args.eval_use_flipped_images,
                    #                    args.use_fixed_image_standardization)
                    feed_dict_test = {phase_train_placeholder: False}
                    results = verification.ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=epoch, sess=sess,
                                                    embedding_tensor=embeddings, batch_size=args.batch_size, feed_dict=feed_dict_test,
                                                    input_placeholder=image_batch)
                    print('test accuracy is: ', str(results[0]))
                    if max(results) > 0.996:
                        print('best accuracy is %.5f' % max(results))

                    print()

                stat['time_evaluate'][epoch-1] = time.time() - t
                if args.save:
                    print('Saving statistics')
                    with h5py.File(stat_file_name, 'w') as f:
                        for key, value in stat.items():
                            f.create_dataset(key, data=value)
    
    return model_dir


def find_threshold(var, percentile):
    hist, bin_edges = np.histogram(var, 100)
    cdf = np.float32(np.cumsum(hist)) / np.sum(hist)
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2
    #plt.plot(bin_centers, cdf)
    threshold = np.interp(percentile*0.01, cdf, bin_centers)
    return threshold


def filter_dataset(dataset, data_filename, percentile, min_nrof_images_per_class):
    with h5py.File(data_filename,'r') as f:
        distance_to_center = np.array(f.get('distance_to_center'))
        label_list = np.array(f.get('label_list'))
        image_list = np.array(f.get('image_list'))
        distance_to_center_threshold = find_threshold(distance_to_center, percentile)
        indices = np.where(distance_to_center>=distance_to_center_threshold)[0]
        filtered_dataset = dataset
        removelist = []
        for i in indices:
            label = label_list[i]
            image = image_list[i]
            if image in filtered_dataset[label].image_paths:
                filtered_dataset[label].image_paths.remove(image)
            if len(filtered_dataset[label].image_paths)<min_nrof_images_per_class:
                removelist.append(label)

        ix = sorted(list(set(removelist)), reverse=True)
        for i in ix:
            del(filtered_dataset[i])

    return filtered_dataset


# def train(args, sess, epoch, image_list, label_list, index_dequeue_op, enqueue_op, image_paths_placeholder, labels_placeholder,
#       learning_rate_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder, step,
#       loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file,
#       stat, cross_entropy_mean, accuracy,
#       learning_rate, prelogits, prelogits_center_loss, random_rotate, random_crop, random_flip, prelogits_norm, prelogits_hist_max, use_fixed_image_standardization):
def train(args, sess, epoch, train_next_element, train_iterator, image_batch, label_batch, image_list, label_list,
          learning_rate_placeholder, phase_train_placeholder, step,
          loss, train_op, summary_op, summary_writer, reg_losses, learning_rate_schedule_file,
          stat, cross_entropy_mean, accuracy,
          learning_rate, prelogits, prelogits_center_loss, random_rotate, random_crop, random_flip, prelogits_norm, prelogits_hist_max,
          use_fixed_image_standardization):

    # batch_number = 0
    #
    # if args.learning_rate > 0.0:
    #     lr = args.learning_rate
    # else:
    #     lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)
    #
    # if lr<=0:
    #     return False
    #
    # index_epoch = sess.run(index_dequeue_op)
    # label_epoch = np.array(label_list)[index_epoch]
    # image_epoch = np.array(image_list)[index_epoch]
    #
    # # Enqueue one epoch of image paths and labels
    # labels_array = np.expand_dims(np.array(label_epoch),1)
    # image_paths_array = np.expand_dims(np.array(image_epoch),1)
    # control_value = facenet.RANDOM_ROTATE * random_rotate + facenet.RANDOM_CROP * random_crop + facenet.RANDOM_FLIP * random_flip + facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    # control_array = np.ones_like(labels_array) * control_value
    # sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})
    #
    # # Training loop
    # train_time = 0
    # while batch_number < args.epoch_size:
    #     start_time = time.time()
    #     feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder:True, batch_size_placeholder:args.batch_size}
    #     tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, learning_rate, prelogits_norm, accuracy, prelogits_center_loss]
    #     if batch_number % 100 == 0:
    #         loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_, summary_str = sess.run(tensor_list + [summary_op], feed_dict=feed_dict)
    #         summary_writer.add_summary(summary_str, global_step=step_)
    #     else:
    #         loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_ = sess.run(tensor_list, feed_dict=feed_dict)
    #
    #     duration = time.time() - start_time
    #     stat['loss'][step_-1] = loss_
    #     stat['center_loss'][step_-1] = center_loss_
    #     stat['reg_loss'][step_-1] = np.sum(reg_losses_)
    #     stat['xent_loss'][step_-1] = cross_entropy_mean_
    #     stat['prelogits_norm'][step_-1] = prelogits_norm_
    #     stat['learning_rate'][epoch-1] = lr_
    #     stat['accuracy'][step_-1] = accuracy_
    #     stat['prelogits_hist'][epoch-1,:] += np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]
    #
    #     duration = time.time() - start_time
    #     print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
    #           (epoch, batch_number+1, args.epoch_size, duration, loss_, cross_entropy_mean_, np.sum(reg_losses_), accuracy_, lr_, center_loss_))
    #     batch_number += 1
    #     train_time += duration
    # # Add validation loss and accuracy to summary
    # summary = tf.Summary()
    # #pylint: disable=maybe-no-member
    # summary.value.add(tag='time/total', simple_value=train_time)
    # summary_writer.add_summary(summary, global_step=step_)
    # return True


# def validate(args, sess, epoch, image_list, label_list, enqueue_op, image_paths_placeholder, labels_placeholder, control_placeholder,
#              phase_train_placeholder, batch_size_placeholder, stat, loss, regularization_losses, cross_entropy_mean, accuracy,
#              validate_every_n_epochs, use_fixed_image_standardization):

    batch_number = 0

    if args.learning_rate > 0.0:
        lr = args.learning_rate
    else:
        lr = facenet.get_learning_rate_from_file(learning_rate_schedule_file, epoch)

    if lr <= 0:
        return False

    train_time = 0
    start_time = time.time()

    while True:
        try:

            images, labels = sess.run(train_next_element)

            feed_dict = {learning_rate_placeholder: lr, phase_train_placeholder: True, image_batch: images, label_batch: labels}

            tensor_list = [loss, train_op, step, reg_losses, prelogits, cross_entropy_mean, learning_rate, prelogits_norm, accuracy,
                           prelogits_center_loss]
            if batch_number % 10 == 0:
                loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_, summary_str = \
                    sess.run(tensor_list + [summary_op], feed_dict=feed_dict)

                summary_writer.add_summary(summary_str, global_step=step_)
            else:
                loss_, _, step_, reg_losses_, prelogits_, cross_entropy_mean_, lr_, prelogits_norm_, accuracy_, center_loss_ = \
                    sess.run(tensor_list, feed_dict=feed_dict)

            # duration = time.time() - start_time
            stat['loss'][step_ - 1] = loss_
            stat['center_loss'][step_ - 1] = center_loss_
            stat['reg_loss'][step_ - 1] = np.sum(reg_losses_)
            stat['xent_loss'][step_ - 1] = cross_entropy_mean_
            stat['prelogits_norm'][step_ - 1] = prelogits_norm_
            stat['learning_rate'][epoch - 1] = lr_
            stat['accuracy'][step_ - 1] = accuracy_
            stat['prelogits_hist'][epoch - 1, :] += \
                np.histogram(np.minimum(np.abs(prelogits_), prelogits_hist_max), bins=1000, range=(0.0, prelogits_hist_max))[0]

            duration = time.time() - start_time
            print('Epoch: [%d][%d/%d]\tTime %.3f\tLoss %2.3f\tXent %2.3f\tRegLoss %2.3f\tAccuracy %2.3f\tLr %2.5f\tCl %2.3f' %
                  (epoch, batch_number + 1, args.epoch_size, duration, loss_, cross_entropy_mean_, np.sum(reg_losses_), accuracy_, lr_,
                   center_loss_))
            batch_number += 1
            train_time += duration

        except tf.errors.OutOfRangeError:
            print('tf.errors.OutOfRangeError, Reinitialize train_iterator')
            sess.run(train_iterator.initializer)
            break

    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    # pylint: disable=maybe-no-member
    summary.value.add(tag='time/total', simple_value=train_time)
    summary_writer.add_summary(summary, global_step=step_)

    return True

def validate(name_dict, index_dict, args, sess, epoch, validation_next_element, validation_iterator, embeddings,
             image_batch, label_batch, image_list, label_list, phase_train_placeholder, stat, loss, regularization_losses,
             cross_entropy_mean, accuracy, validate_every_n_epochs,
             use_fixed_image_standardization):
    print('Running forward pass on validation set')

    # nrof_batches = len(label_list) // args.eval_batch_size
    # nrof_images = nrof_batches * args.eval_batch_size
    #
    # # Enqueue one epoch of image paths and labels
    # labels_array = np.expand_dims(np.array(label_list[:nrof_images]),1)
    # image_paths_array = np.expand_dims(np.array(image_list[:nrof_images]),1)
    # control_array = np.ones_like(labels_array, np.int32)*facenet.FIXED_STANDARDIZATION * use_fixed_image_standardization
    # sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})

    # loss_array = np.zeros((nrof_batches,), np.float32)
    # xent_array = np.zeros((nrof_batches,), np.float32)
    # accuracy_array = np.zeros((nrof_batches,), np.float32)

    loss_array = []
    xent_array = []
    accuracy_array = []

    ## Training loop
    # start_time = time.time()
    # for i in range(nrof_batches):
    #     feed_dict = {phase_train_placeholder:False, batch_size_placeholder:args.eval_batch_size}
    #     loss_, cross_entropy_mean_, accuracy_ = sess.run([loss, cross_entropy_mean, accuracy], feed_dict=feed_dict)
    #     loss_array[i], xent_array[i], accuracy_array[i] = (loss_, cross_entropy_mean_, accuracy_)
    #     if i % 10 == 9:
    #         print('.', end='')
    #         sys.stdout.flush()
    # print('')

    # duration = time.time() - start_time

    # val_index = (epoch-1)//validate_every_n_epochs
    # stat['val_loss'][val_index] = np.mean(loss_array)
    # stat['val_xent_loss'][val_index] = np.mean(xent_array)
    # stat['val_accuracy'][val_index] = np.mean(accuracy_array)
    #
    # print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f' %
    #       (epoch, duration, np.mean(loss_array), np.mean(xent_array), np.mean(accuracy_array)))

    # Training loop
    start_time = time.time()
    # sess.run(iterator.initializer)
    num_batch = 0
    while True:
        try:
            images, labels = sess.run(validation_next_element)
            feed_dict = {phase_train_placeholder: False, image_batch: images, label_batch: labels}

            emb_array, loss_, cross_entropy_mean_, accuracy_ = sess.run([embeddings, loss, cross_entropy_mean, accuracy], feed_dict=feed_dict)

            # tpr, fpr, accuracy_1, val, val_std, far = verification.evaluate(name_dict, index_dict, emb_array, images, labels,
            #                                                                 nrof_folds=args.eval_nrof_folds)
            # evaluate_1(emb_array, labels)

            # print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy_1), np.std(accuracy_1)))
            # print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            # print('fpr and tpr: %1.3f %1.3f' % (np.mean(fpr, 0), np.mean(tpr, 0)))

            loss_array.append(loss_)
            xent_array.append(cross_entropy_mean_)
            accuracy_array.append(accuracy_)
            # loss_array[i], xent_array[i], accuracy_array[i] = (loss_, cross_entropy_mean_, accuracy_)
            # if num_batch % 10 == 9:
            #     print('.', end='')
            #     sys.stdout.flush()

            num_batch += 1
        except tf.errors.OutOfRangeError:
            print('tf.errors.OutOfRangeError, Reinitialize validation_iterator')
            sess.run(validation_iterator.initializer)
            # print('')
            break

    duration = time.time() - start_time

    val_index = (epoch - 1) // validate_every_n_epochs
    stat['val_loss'][val_index] = np.mean(np.array(loss_array))
    stat['val_xent_loss'][val_index] = np.mean(np.array(xent_array))
    stat['val_accuracy'][val_index] = np.mean(np.array(accuracy_array))

    print('Validation Epoch: %d\tTime %.3f\tLoss %2.3f\tXent %2.3f\tAccuracy %2.3f' %
          (epoch, duration, np.mean(np.array(loss_array)), np.mean(np.array(xent_array)), np.mean(np.array(accuracy_array))))


# def evaluate(sess, enqueue_op, image_paths_placeholder, labels_placeholder, phase_train_placeholder, batch_size_placeholder, control_placeholder,
#         embeddings, labels, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step, summary_writer, stat, epoch, distance_metric,
#              subtract_mean, use_flipped_images, use_fixed_image_standardization):
def evaluate(sess, phase_train_placeholder, embeddings, label_batch, image_paths, actual_issame, batch_size, nrof_folds, log_dir, step,
             summary_writer, stat, epoch,
             distance_metric,
             subtract_mean, use_flipped_images, use_fixed_image_standardization):

    start_time = time.time()
    # Run forward pass to calculate embeddings
    print('Runnning forward pass on LFW images')

    # Enqueue one epoch of image paths and labels
    nrof_embeddings = len(actual_issame)*2  # nrof_pairs * nrof_images_per_pair
    nrof_flips = 2 if use_flipped_images else 1
    nrof_images = nrof_embeddings * nrof_flips
    labels_array = np.expand_dims(np.arange(0,nrof_images),1)
    image_paths_array = np.expand_dims(np.repeat(np.array(image_paths),nrof_flips),1)
    control_array = np.zeros_like(labels_array, np.int32)
    if use_fixed_image_standardization:
        control_array += np.ones_like(labels_array)*facenet.FIXED_STANDARDIZATION
    if use_flipped_images:
        # Flip every second image
        control_array += (labels_array % 2)*facenet.FLIP
    sess.run(enqueue_op, {image_paths_placeholder: image_paths_array, labels_placeholder: labels_array, control_placeholder: control_array})

    embedding_size = int(embeddings.get_shape()[1])
    assert nrof_images % batch_size == 0, 'The number of LFW images must be an integer multiple of the LFW batch size'
    nrof_batches = nrof_images // batch_size
    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    for i in range(nrof_batches):
        feed_dict = {phase_train_placeholder:False, batch_size_placeholder:batch_size}
        emb, lab = sess.run([embeddings, labels], feed_dict=feed_dict)
        lab_array[lab] = lab
        emb_array[lab, :] = emb
        if i % 10 == 9:
            print('.', end='')
            sys.stdout.flush()
    print('')
    embeddings = np.zeros((nrof_embeddings, embedding_size*nrof_flips))
    if use_flipped_images:
        # Concatenate embeddings for flipped and non flipped version of the images
        embeddings[:,:embedding_size] = emb_array[0::2,:]
        embeddings[:,embedding_size:] = emb_array[1::2,:]
    else:
        embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the' \
                                                                    ' input pipeline'
    _, _, accuracy, val, val_std, far = lfw.evaluate(embeddings, actual_issame, nrof_folds=nrof_folds, distance_metric=distance_metric,
                                                     subtract_mean=subtract_mean)

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
    eval_time = time.time() - start_time
    # Add validation loss and accuracy to summary
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='eval/accuracy', simple_value=np.mean(accuracy))
    summary.value.add(tag='eval/val_rate', simple_value=val)
    summary.value.add(tag='time/eval', simple_value=eval_time)
    summary_writer.add_summary(summary, step)
    with open(os.path.join(log_dir,'eval_result.txt'),'at') as f:
        f.write('%d\t%.5f\t%.5f\n' % (step, np.mean(accuracy), val))
    stat['eval_accuracy'][epoch-1] = np.mean(accuracy)
    stat['eval_valrate'][epoch-1] = val


def save_variables_and_metagraph(sess, saver, summary_writer, model_dir, model_name, step):
    # Save the model checkpoint
    print('Saving variables')
    start_time = time.time()
    checkpoint_path = os.path.join(model_dir, 'model-%s.ckpt' % model_name)
    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
    save_time_variables = time.time() - start_time
    print('Variables saved in %.2f seconds' % save_time_variables)
    metagraph_filename = os.path.join(model_dir, 'model-%s.meta' % model_name)
    save_time_metagraph = 0  
    if not os.path.exists(metagraph_filename):
        print('Saving metagraph')
        start_time = time.time()
        saver.export_meta_graph(metagraph_filename)
        save_time_metagraph = time.time() - start_time
        print('Metagraph saved in %.2f seconds' % save_time_metagraph)
    summary = tf.Summary()
    #pylint: disable=maybe-no-member
    summary.value.add(tag='time/save_variables', simple_value=save_time_variables)
    summary.value.add(tag='time/save_metagraph', simple_value=save_time_metagraph)
    summary_writer.add_summary(summary, step)
  

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--logs_base_dir', type=str, 
        help='Directory where to write event logs.', default='~/logs/facenet')
    parser.add_argument('--models_base_dir', type=str,
        help='Directory where to write trained models and checkpoints.', default='~/models/facenet')
    parser.add_argument('--gpu_memory_fraction', type=float,
        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    parser.add_argument('--pretrained_model', type=str,
        help='Load a pretrained model before training starts.')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory containing aligned face patches.',
        default='~/datasets/casia/casia_maxpy_mtcnnalign_182_160')
    parser.add_argument('--model_def', type=str,
        help='Model definition. Points to a module containing the definition of the inference graph.', default='models.inception_resnet_v1')
    parser.add_argument('--max_nrof_epochs', type=int,
        help='Number of epochs to run.', default=500)
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--epoch_size', type=int,
        help='Number of batches per epoch.', default=1000)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    parser.add_argument('--random_crop', 
        help='Performs random cropping of training images. If false, the center image_size pixels from the training images are used. ' +
         'If the size of the images in the data directory is equal to image_size no cropping is performed', action='store_true')
    parser.add_argument('--random_flip', 
        help='Performs random horizontal flipping of training images.', action='store_true')
    parser.add_argument('--random_rotate', 
        help='Performs random rotations of training images.', action='store_true')
    parser.add_argument('--use_fixed_image_standardization', 
        help='Performs fixed standardization of images.', action='store_true')
    parser.add_argument('--keep_probability', type=float,
        help='Keep probability of dropout for the fully connected layer(s).', default=1.0)
    parser.add_argument('--weight_decay', type=float,
        help='L2 weight regularization.', default=0.0)
    parser.add_argument('--center_loss_factor', type=float,
        help='Center loss factor.', default=0.0)
    parser.add_argument('--center_loss_alfa', type=float,
        help='Center update rate for center loss.', default=0.95)
    parser.add_argument('--prelogits_norm_loss_factor', type=float,
        help='Loss based on the norm of the activations in the prelogits layer.', default=0.0)
    parser.add_argument('--prelogits_norm_p', type=float,
        help='Norm to use for prelogits norm loss.', default=1.0)
    parser.add_argument('--prelogits_hist_max', type=float,
        help='The max value for the prelogits histogram.', default=10.0)
    parser.add_argument('--optimizer', type=str, choices=['ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'],
        help='The optimization algorithm to use', default='ADAGRAD')
    parser.add_argument('--learning_rate', type=float,
        help='Initial learning rate. If set to a negative value a learning rate ' +
        'schedule can be specified in the file "learning_rate_schedule.txt"', default=0.1)
    parser.add_argument('--learning_rate_decay_epochs', type=int,
        help='Number of epochs between learning rate decay.', default=100)
    parser.add_argument('--learning_rate_decay_factor', type=float,
        help='Learning rate decay factor.', default=1.0)
    parser.add_argument('--moving_average_decay', type=float,
        help='Exponential decay for tracking of training parameters.', default=0.9999)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--nrof_preprocess_threads', type=int,
        help='Number of preprocessing (data loading and augmentation) threads.', default=4)
    parser.add_argument('--log_histograms', 
        help='Enables logging of weight/bias histograms in tensorboard.', action='store_true')
    parser.add_argument('--learning_rate_schedule_file', type=str,
        help='File containing the learning rate schedule that is used when learning_rate is set to to -1.', default='data/learning_rate_schedule.txt')
    parser.add_argument('--filter_filename', type=str,
        help='File containing image data used for dataset filtering', default='')
    parser.add_argument('--filter_percentile', type=float,
        help='Keep only the percentile images closed to its class center', default=100.0)
    parser.add_argument('--filter_min_nrof_images_per_class', type=int,
        help='Keep only the classes with this number of examples or more', default=0)
    parser.add_argument('--validate_every_n_epochs', type=int,
        help='Number of epoch between validation', default=5)
    parser.add_argument('--validation_set_split_ratio', type=float,
        help='The ratio of the total dataset to use for validation', default=0.0)
    parser.add_argument('--min_nrof_val_images_per_class', type=float,
        help='Classes with fewer images will be removed from the validation set', default=0)
 
    # Parameters for validation on LFW
    parser.add_argument('--eval_pairs', type=str,
        help='The file containing the pairs to use for validation.', default='data/pairs.txt')
    parser.add_argument('--eval_dir', type=str,
        help='Path to the data directory containing aligned face patches.', default='')
    parser.add_argument('--eval_batch_size', type=int,
        help='Number of images to process in a batch in the LFW test set.', default=100)
    parser.add_argument('--eval_nrof_folds', type=int,
        help='Number of folds to use for cross validation. Mainly used for testing.', default=10)
    parser.add_argument('--eval_distance_metric', type=int,
        help='Type of distance metric to use. 0: Euclidian, 1:Cosine similarity distance.', default=0)
    parser.add_argument('--eval_use_flipped_images',
        help='Concatenates embeddings for the image and its horizontally flipped counterpart.', action='store_true')
    parser.add_argument('--eval_subtract_mean',
        help='Subtract feature mean before calculating distance.', action='store_true')
    return parser.parse_args(argv)
  

def tf_gen_dataset_perfarmance(image_list, label_list, nrof_preprocess_threads, image_size, BATCH_SIZE=32, repeat_count=1):
    """
    Performance of tensorflow dataset generation
    :return:
    """
    """
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    tfrecord:
    image_count: 1805
    ^^^^^^^^^^^^^^:: timeit method 2 ::^^^^^^^^^^^^^^
    non_equal_counter: 1119
    Num Batch:  1140
    1140 batches: 2.122495651245117 s
    17187.31437 Images/s
    Total time: 2.122495651245117s
    <=><=><=><=><=><=><=><=><=><=>
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    cache_zip:
    image_count: 1805
    ::::::::::::::::::::::::::::::::in memory cache::::::::::::::::::::::::::::::::
    ^^^^^^^^^^^^^^:: timeit method 2 ::^^^^^^^^^^^^^^
    non_equal_counter: 1119
    Num Batch:  1140
    1140 batches: 3.9507839679718018 s
    9233.61042 Images/s
    Total time: 3.9507839679718018s
    <=><=><=><=><=><=><=><=><=><=>
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    cache_slices:
    image_count: 1805
    ::::::::::::::::::::::::::::::::in memory cache::::::::::::::::::::::::::::::::
    ^^^^^^^^^^^^^^:: timeit method 2 ::^^^^^^^^^^^^^^
    non_equal_counter: 1119
    Num Batch:  1140
    1140 batches: 3.991459608078003 s
    9139.51376 Images/s
    Total time: 3.991459608078003s
    <=><=><=><=><=><=><=><=><=><=>
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    slices:
    image_count: 1805
    ^^^^^^^^^^^^^^:: timeit method 2 ::^^^^^^^^^^^^^^
    non_equal_counter: 1119
    Num Batch:  1140
    1140 batches: 22.685077667236328 s
    1608.10558 Images/s
    Total time: 22.685077667236328s
    <=><=><=><=><=><=><=><=><=><=>
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    zip:
    image_count: 1805
    ^^^^^^^^^^^^^^:: timeit method 2 ::^^^^^^^^^^^^^^
    non_equal_counter: 1105
    Num Batch:  1140
    1140 batches: 25.535181999206543 s
    1428.61719 Images/s
    Total time: 25.535181999206543s
    <=><=><=><=><=><=><=><=><=><=>
    --------------------------------------------------
    """
    for method in ['tfrecord', 'cache_slices', 'slices']:
        print(f"{'@' * 100}")
        print(f'{method}:')
        _ = facenet.tf_gen_dataset(image_list, label_list, nrof_preprocess_threads, image_size, method=method, BATCH_SIZE=BATCH_SIZE, performance=True,
                                   repeat_count=repeat_count)
    print(f"{'-' * 50}")

    return


if __name__ == '__main__':
    args_obj = Args()
    main(args_obj)
