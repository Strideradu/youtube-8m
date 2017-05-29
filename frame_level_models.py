# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import util_conv
import grid_rnn_cell as grid

import tensorflow.contrib.slim as slim
from tensorflow import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")
flags.DEFINE_integer("num_random_frames", 128, "Number of random frames.")
flags.DEFINE_bool("grid_weights_tied", False, "Tie the time and depth weights for less overhead")
flags.DEFINE_string("weight_initializer", "uniform_unit_scaling_initializer",
                    "Weight initializing method, only in use now for the LSTM")
flags.DEFINE_integer("attention_length", 8, "Size of attention window.")
flags.DEFINE_integer("num_shifts", 512, "number of shift.")


class FrameLevelLogisticModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a logistic classifier over the average of the
        frame-level features.

        This class is intended to be an example for implementors of frame level
        models. If you want to train a model over averaged features it is more
        efficient to average them beforehand rather than on the fly.

        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        feature_size = model_input.get_shape().as_list()[2]

        denominators = tf.reshape(
            tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
        avg_pooled = tf.reduce_sum(model_input,
                                   axis=[1]) / denominators

        output = slim.fully_connected(
            avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(1e-8))
        return {"predictions": output}


class DbofModel(models.BaseModel):
    """Creates a Deep Bag of Frames model.

    The model projects the features for each frame into a higher dimensional
    'clustering' space, pools across frames in that space, and then
    uses a configurable video-level model to classify the now aggregated features.

    The model will randomly sample either frames or sequences of frames during
    training to speed up convergence.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_frames,
                     iterations=None,
                     add_batch_norm=None,
                     sample_random_frames=None,
                     cluster_size=None,
                     hidden_size=None,
                     is_training=True,
                     **unused_params):
        iterations = iterations or FLAGS.iterations
        add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
        random_frames = sample_random_frames or FLAGS.sample_random_frames
        cluster_size = cluster_size or FLAGS.dbof_cluster_size
        hidden1_size = hidden_size or FLAGS.dbof_hidden_size

        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        if random_frames:
            model_input = utils.SampleRandomFrames(model_input, num_frames,
                                                   iterations)
        else:
            model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                     iterations)
        max_frames = model_input.get_shape().as_list()[1]
        feature_size = model_input.get_shape().as_list()[2]
        reshaped_input = tf.reshape(model_input, [-1, feature_size])
        tf.summary.histogram("input_hist", reshaped_input)

        if add_batch_norm:
            reshaped_input = slim.batch_norm(
                reshaped_input,
                center=True,
                scale=True,
                is_training=is_training,
                scope="input_bn")

        cluster_weights = tf.get_variable("cluster_weights",
                                          [feature_size, cluster_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
        tf.summary.histogram("cluster_weights", cluster_weights)
        activation = tf.matmul(reshaped_input, cluster_weights)
        if add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="cluster_bn")
        else:
            cluster_biases = tf.get_variable("cluster_biases",
                                             [cluster_size],
                                             initializer=tf.random_normal(stddev=1 / math.sqrt(feature_size)))
            tf.summary.histogram("cluster_biases", cluster_biases)
            activation += cluster_biases
        activation = tf.nn.relu6(activation)
        tf.summary.histogram("cluster_output", activation)

        activation = tf.reshape(activation, [-1, max_frames, cluster_size])
        activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [cluster_size, hidden1_size],
                                          initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
        tf.summary.histogram("hidden1_weights", hidden1_weights)
        activation = tf.matmul(activation, hidden1_weights)
        if add_batch_norm:
            activation = slim.batch_norm(
                activation,
                center=True,
                scale=True,
                is_training=is_training,
                scope="hidden1_bn")
        else:
            hidden1_biases = tf.get_variable("hidden1_biases",
                                             [hidden1_size],
                                             initializer=tf.random_normal_initializer(stddev=0.01))
            tf.summary.histogram("hidden1_biases", hidden1_biases)
            activation += hidden1_biases
        activation = tf.nn.relu6(activation)
        tf.summary.histogram("hidden1_output", activation)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=activation,
            vocab_size=vocab_size,
            **unused_params)


class LstmModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers

        iterations = FLAGS.num_random_frames
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomSequence(model_input, num_frames,
                                                 iterations)
        batch_size = tf.shape(model_input)[0]
        num_frames = tf.fill([batch_size], iterations)

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.DropoutWrapper(
                    tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, reuse=tf.get_variable_scope().reuse),
                    input_keep_prob=0.5, output_keep_prob=0.5)
                for _ in range(number_of_layers)
                ])

        loss = 0.0

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                           sequence_length=num_frames,
                                           dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=state[-1].h,
            vocab_size=vocab_size,
            **unused_params)


class TemporalPoolingCNNModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        max_frame = 128
        num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
        model_input = utils.SampleRandomFrames(model_input, num_frames, max_frame)
        # max_frame = model_input.get_shape().as_list()[1]
        image = tf.reshape(model_input, [-1, 32, 32])
        image = tf.expand_dims(image, 3)
        with slim.arg_scope([slim.conv2d],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005),
                            normalizer_fn=slim.batch_norm):
            net = slim.conv2d(image, 32, [5, 5], padding='VALID', scope='conv1')
            net = slim.relu(net, 32, scope='relu1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.conv2d(net, 64, [5, 5], padding='VALID', scope='conv2')
            net = slim.relu(net, 64, scope='relu2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.conv2d(net, 128, [5, 5], padding='VALID', scope='conv3')
            net = slim.relu(net, 128, scope='relu3')
            net = tf.squeeze(net, [1, 2], name='squeezed')
            print(net)

        net = tf.reshape(net, [-1, max_frame, 128])
        net = utils.FramePooling(net, 'max')
        net = slim.fully_connected(net, 512, scope='fc4')
        print(net)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=net,
            vocab_size=vocab_size,
            **unused_params)


class GridLstmModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of grid-LSTM Units to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        num_shifts = FLAGS.num_shifts

        stacked_grid_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.GridLSTMCell(
                    lstm_size, forget_bias=1.0, use_peepholes=True, state_is_tuple=False,
                    num_frequency_blocks=[num_shifts])
                for _ in range(number_of_layers)], state_is_tuple=False)

        loss = 0.0
        outputs, state = tf.nn.dynamic_rnn(stacked_grid_lstm, model_input,
                                           sequence_length=num_frames,
                                           dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=state,
            vocab_size=vocab_size,
            **unused_params)


class AttentionLstmModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        weight_initializer = FLAGS.weight_initializer
        attention_length = FLAGS.attention_length

        if weight_initializer == 'random':
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.AttentionCellWrapper(
                        tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False,
                                                initializer=tf.truncated_normal_initializer(stddev=1e-3),
                                                reuse=tf.get_variable_scope().reuse), attn_length=attention_length,
                        reuse=tf.get_variable_scope().reuse)
                    for _ in range(number_of_layers)
                    ], state_is_tuple=False)
        else:  # uniform weight initializations by default, for some reason
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.AttentionCellWrapper(
                        tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False,
                                                reuse=tf.get_variable_scope().reuse), attn_length=attention_length,
                        reuse=tf.get_variable_scope().reuse)
                    for _ in range(number_of_layers)
                    ], state_is_tuple=False)

        loss = 0.0

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                           sequence_length=num_frames,
                                           dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=state,
            vocab_size=vocab_size,
            **unused_params)


class TFLstmModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        weight_initializer = FLAGS.weight_initializer

        if weight_initializer == 'random':
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.TimeFreqLSTMCell(
                        lstm_size, forget_bias=1.0, state_is_tuple=False,
                        initializer=tf.truncated_normal_initializer(stddev=1e-3))
                    for _ in range(number_of_layers)
                    ], state_is_tuple=False)
        else:  # uniform weight initializations by default, for some reason
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.TimeFreqLSTMCell(
                        lstm_size, forget_bias=1.0, state_is_tuple=False)
                    for _ in range(number_of_layers)
                    ], state_is_tuple=False)

        loss = 0.0

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                           sequence_length=num_frames,
                                           dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=state[-1].h,
            vocab_size=vocab_size,
            **unused_params)


class Grid2LstmModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of grid-LSTM Units to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers

        stacked_grid_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                grid.Grid2LSTMCell(
                    lstm_size, forget_bias=1.0, use_peepholes=True, tied=FLAGS.grid_weights_tied, state_is_tuple=False,
                    output_is_tuple=False)
                for _ in range(number_of_layers)], state_is_tuple=False)

        loss = 0.0
        outputs, state = tf.nn.dynamic_rnn(stacked_grid_lstm, model_input,
                                           sequence_length=num_frames,
                                           dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=state,
            vocab_size=vocab_size,
            **unused_params)


class PeeholeLstmModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        weight_initializer = FLAGS.weight_initializer

        if weight_initializer == 'random':
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False, use_peepholes=True,
                                         initializer=tf.truncated_normal_initializer(stddev=1e-3),
                                         reuse=tf.get_variable_scope().reuse)
                 for _ in range(number_of_layers)
                 ], state_is_tuple=False)
        else:  # uniform weight initializations by default, for some reason
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False, use_peepholes=True,
                                            reuse=tf.get_variable_scope().reuse)
                    for _ in range(number_of_layers)
                    ], state_is_tuple=False)

        loss = 0.0

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                           sequence_length=num_frames,
                                           dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=state,
            vocab_size=vocab_size,
            **unused_params)


class LayerNormLstmModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_size, forget_bias=1.0, dropout_keep_prob=0.5,
                                                      reuse=tf.get_variable_scope().reuse)
                for _ in range(number_of_layers)
                ])

        loss = 0.0

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                           sequence_length=num_frames,
                                           dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=state[-1].h,
            vocab_size=vocab_size,
            **unused_params)


class BiLstmModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """

        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers

        stacked_lstm_fw = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False, reuse=tf.get_variable_scope().reuse)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

        stacked_lstm_bw = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0, state_is_tuple=False, reuse=tf.get_variable_scope().reuse)
                for _ in range(number_of_layers)
                ], state_is_tuple=False)

        # lstm_fw = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False)

        # lstm_bw = tf.contrib.rnn.BasicLSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False)

        loss = 0.0

        outputs, state = tf.nn.bidirectional_dynamic_rnn(
            stacked_lstm_fw,
            stacked_lstm_bw,
            model_input,
            sequence_length=num_frames,
            dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        # As we have Bi-LSTM, we have two output, which are not connected. So
        # merge them
        # state = tf.concat(2, state)
        combined_state = tf.add(state[0], state[1])

        return aggregated_model().create_model(
            model_input=combined_state,
            vocab_size=vocab_size,
            **unused_params)


class PeeholeLstmModel2(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers
        weight_initializer = FLAGS.weight_initializer

        if weight_initializer == 'random':
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False, use_peepholes=True,
                                         initializer=tf.truncated_normal_initializer(stddev=1e-3),
                                         reuse=tf.get_variable_scope().reuse)
                 for _ in range(number_of_layers)
                 ], state_is_tuple=False)
        else:  # uniform weight initializations by default, for some reason
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False, use_peepholes=True,
                                            reuse=tf.get_variable_scope().reuse)
                    for _ in range(number_of_layers)
                    ], state_is_tuple=False)

        loss = 0.0

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                           sequence_length=num_frames,
                                           dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        h_all = []
        state_len = state.get_shape().as_list()[1]
        for i in range(state_len):
            h_all.append(state[i].h)

        mean_h = tf.reduce_mean(h_all)

        return aggregated_model().create_model(
            model_input=mean_h,
            vocab_size=vocab_size,
            **unused_params)


class CNNLstmModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.
        Args:
          model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                       input features.
          vocab_size: The number of classes in the dataset.
          num_frames: A vector of length 'batch' which indicates the number of
               frames for each video (before padding).
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          'batch_size' x 'num_classes'.
        """
        """4 different cnn layers into one rnn with sequence length 4"""
        model_input = tf.reshape(model_input, [-1, 32, 32, 300])
        print model_input.shape
        p1conv1 = tf.layers.conv2d(
            inputs=model_input,
            filters=300,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        p1pool1 = tf.layers.max_pooling2d(inputs=p1conv1, pool_size=[2, 2], strides=2)
        p1conv2 = tf.layers.conv2d(
            inputs=p1pool1,
            filters=600,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)
        p1pool2 = tf.layers.max_pooling2d(inputs=p1conv2, pool_size=[2, 2], strides=2)

        outputconv = tf.reshape(p1pool2, [-1, 64, 600])
        lstm_size = 64
        number_of_layers = FLAGS.lstm_layers

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

        loss = 0.0

        outputs, state = tf.nn.dynamic_rnn(stacked_lstm, outputconv,
                                           sequence_length=num_frames[:600],
                                           dtype=tf.float32)

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=state[-1].h,
            vocab_size=vocab_size,
            **unused_params)


class SeqCNNModel(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Creates a model which uses a stack of LSTMs to represent the video.
                Args:
                  model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                               input features.
                  vocab_size: The number of classes in the dataset.
                  num_frames: A vector of length 'batch' which indicates the number of
                       frames for each video (before padding).
                Returns:
                  A dictionary with a tensor containing the probability predictions of the
                  model in the 'predictions' key. The dimensions of the tensor are
                  'batch_size' x 'num_classes'.
                """
        filter_sizes = [3, 4, 5]
        num_filters = 128
        feature_size = model_input.get_shape().as_list()[2]

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            conv1 = tf.layer.conv2d(input=model_input,
                                    filters=num_filters,
                                    kernel_size=[filter_size, feature_size],
                                    padding="same",
                                    activation=tf.nn.relu,
                                    bias_initializer=tf.zeros_initializer()
                                    )
            pool = tf.layers.max_pooling2d(inputs=conv1,
                                           pool_size=[num_frames - filter_size + 1, 1],
                                           padding="valid",
                                           strides=1)
            pooled_outputs.append(pool)

        num_filters_total = num_filters * len(filter_sizes)
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

        dropout = tf.layers.dropout(h_pool_flat,
                                    rate=0.5,
                                    )

        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=dropout,
            vocab_size=vocab_size,
            **unused_params)
