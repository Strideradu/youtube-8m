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
            model_input=state[-1].h,
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
                [   tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False, use_peepholes = True,
                                            initializer=tf.truncated_normal_initializer(stddev=1e-3),
                                            reuse=tf.get_variable_scope().reuse)
                    for _ in range(number_of_layers)
                    ], state_is_tuple=False)
        else:  # uniform weight initializations by default, for some reason
            stacked_lstm = tf.contrib.rnn.MultiRNNCell(
                [
                    tf.contrib.rnn.LSTMCell(lstm_size, forget_bias=1.0, state_is_tuple=False, use_peepholes = True,
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
    """Test class
    all models take in
    1. input tensor, 3d, [-1, 300, 1024]
    2. number of classes
    3. number of frames (max) - 300
    output
    dictionary. key: 'prediction', value: train_op
      try to wrap with tf.Prints
    """

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """Bidirectional RNN
        y_t = softmax(W_y(hf_t + hb_t) + b_y)
        t = timestep
        y = feature
        """
        lstm_size = FLAGS.lstm_cells
        number_of_layers = FLAGS.lstm_layers

        forward_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(
                lstm_size, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
             for _ in range(number_of_layers)
             ])
        backward_stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(
                lstm_size, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)
             for _ in range(number_of_layers)
             ])

        loss = 0.0
        outputs, state = tf.nn.bidirectional_dynamic_rnn(forward_stacked_lstm,
                                                         backward_stacked_lstm,
                                                         model_input,
                                                         sequence_length=num_frames,
                                                         dtype=tf.float32)

        combined_state = tf.add(state[0][-1].h, state[-1][-1].h)
        bi_weights = tf.get_variable("bi_weights", [1024, 512],
                                     initializer=tf.random_normal_initializer(stddev=0.1))
        bi_bias = tf.get_variable("bi_bias", [512],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
        softmax = tf.nn.softmax(tf.matmul(combined_state, bi_weights) + bi_bias)

        # combined_state = tf.Print(combined_state, [tf.shape(combined_state)], 'combined=', summarize=10)

        aggregated_model = getattr(video_level_models, FLAGS.video_level_classifier_model)

        return aggregated_model().create_model(
            model_input=softmax,
            vocab_size=vocab_size,
            **unused_params)


class DCModel(models.BaseModel):
    """
    Psuedo deepmind
    """

    def create_model(self, model_input, vocab_size, num_frames, **unused_params):
        """
        1 layer of 1d conv
        24 layers of dilated conv
        2 1x1 conv blocks w relu
        softmax output / throw to MoeModel
        Args:
          model_input: tf Tensor
          vocab_size: int, 4716
          num_frame: int, max number of frames. 300

        Returns:
          prediction: either from softmax or MoE
        """
        self.filter_width = 2
        self.num_of_blocks = 4
        self.dilations = [1, 2, 4, 6, 8, 16]
        self._define_variables()

        # causal layer
        z = self._causal_conv(model_input, self.var.get('causal_conv'), dilation=1)

        receptive_field = (2 - 1) * sum(self.dilations) * self.num_of_blocks + 1
        output_width = tf.shape(model_input)[1] - receptive_field + 1

        # dilation stack
        with tf.name_scope('dilated_stack'):
            skip = 0
            for i in range(self.num_of_blocks):
                for dil in self.dilations:
                    with tf.name_scope('layer_{}_{}'.format(i, dil)):
                        z, s = self._res_block(z, dil, i, output_width)
                    skip += s

        # post-skip 1d convolutions
        with tf.name_scope('post_processing'):
            transformed1 = tf.nn.relu(skip)
            conv1 = tf.nn.conv1d(transformed1, self.var.get("conv1"), stride=1, padding="SAME")
            conv1 = tf.add(conv1, self.var.get('bias1'))

            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, self.var.get("conv2"), stride=1, padding="SAME")
            conv2 = tf.add(conv2, self.var.get('bias2'))
            conv2 = tf.Print(conv2, [conv2], 'cov2shape = ')

            # average pool over 60 timesteps
            # avg2 = tf.reduce_max(conv2, axis=1)
            # avg2 = tf.Print(avg2, [avg2], 'avg')
            # proba = tf.cast(tf.nn.softmax(avg2), tf.float32)

        # return {"predictions": proba}
        aggregated_model = getattr(video_level_models,
                                   FLAGS.video_level_classifier_model)
        return aggregated_model().create_model(
            model_input=conv2,
            vocab_size=vocab_size,
            **unused_params)

    def _res_block(self, input_tensor, dilation, block_number, output_width):
        """performs dilated conv and gating + skip connection
        refer to section 2.4 of deepmind wavenet paper
        https://arxiv.org/pdf/1609.03499.pdf

        Args:
          input_tensor: 3d tf tensor
          dilation: int
          block_number: int, chunk w dilated layer belongs to
          output_width: int
        """
        with tf.name_scope('res_block_{}_{}'.format(block_number, dilation)):
            gate_output = self._causal_conv(input_tensor,
                                            self.var.get('gate_{}_{}'.format(block_number, dilation)),
                                            dilation)
            filter_output = self._causal_conv(input_tensor,
                                              self.var.get('filter_{}_{}'.format(block_number, dilation)),
                                              dilation)

            gate_bias = self.var.get('gate_bias_{}_{}'.format(block_number, dilation))
            filter_bias = self.var.get('filter_bias_{}_{}'.format(block_number, dilation))
            gate_output = tf.add(gate_output, gate_bias)
            filter_output = tf.add(filter_output, filter_bias)

            joined = tf.tanh(filter_output) * tf.sigmoid(gate_output)

            # out = tf.nn.conv1d(joined,
            #                  self.var.get('output_{}_{}'.format(block_number, dilation)),
            #                  stride=1,
            #                  padding='SAME')
            # out = tf.Print(out, [tf.shape(out)])
            # 1x1 conv output
            transformed = tf.nn.conv1d(joined,
                                       self.var.get('dense_{}_{}'.format(block_number, dilation)),
                                       stride=1, padding="SAME", name="dense")

            # 1x1 conv skip connection
            out_skip = tf.slice(joined, [0, 0, 0], [-1, 90, -1])
            skip_contrib = tf.nn.conv1d(out_skip,
                                        self.var.get('skip_{}_{}'.format(block_number, dilation)),
                                        stride=1, padding='SAME', name='skip')
            # add bias
            dense_bias = self.var.get('dense_bias_{}_{}'.format(block_number, dilation))
            skip_bias = self.var.get('skip_bias_{}_{}'.format(block_number, dilation))
            transformed = tf.add(transformed, dense_bias)
            skip_contrib = tf.add(skip_contrib, skip_bias)

            input_cut = tf.shape(input_tensor)[1] - tf.shape(transformed)[1]
            input_tensor = tf.slice(input_tensor, [0, input_cut, 0], [-1, -1, -1])
            residual = input_tensor + transformed

            return residual, skip_contrib

    def _define_variables(self):
        """define all filters w name scope within a dictionary
        gate & filter undergoes sigmoid and tanh -> zero center
        skip & dense are initialized positive
        """
        self.var = {}
        with tf.name_scope('causal'):
            self.var['causal_conv'] = tf.get_variable('causal_conv',
                                                      [self.filter_width, 1024, 512],
                                                      initializer=tf.random_normal_initializer(stddev=0.1))

        with tf.name_scope('dilate_stack'):
            for block in range(self.num_of_blocks):
                for dilation in self.dilations:
                    gate_name = "gate_{}_{}".format(block, dilation)
                    filter_name = "filter_{}_{}".format(block, dilation)
                    # output_name = "output_{}_{}".format(block, dilation)
                    skip_name = "skip_{}_{}".format(block, dilation)
                    dense_name = "dense_{}_{}".format(block, dilation)

                    self.var[gate_name] = tf.get_variable(gate_name,
                                                          [self.filter_width, 512, 512],
                                                          initializer=tf.random_normal_initializer(stddev=0.1))

                    self.var[filter_name] = tf.get_variable(filter_name,
                                                            [self.filter_width, 512, 512],
                                                            initializer=tf.random_normal_initializer(stddev=0.1))

                    # self.var[output_name] = tf.get_variable(output_name,
                    #                          [self.filter_width, 512, 512],
                    #                          initializer=tf.random_normal_initializer(stddev=0.01))

                    self.var[skip_name] = tf.get_variable(skip_name, [1, 512, 512],
                                                          initializer=tf.random_normal_initializer(stddev=0.01))

                    self.var[dense_name] = tf.get_variable(dense_name, [1, 512, 512],
                                                           initializer=tf.random_normal_initializer(stddev=0.01))

                    # add biases
                    gate_bias = "gate_bias_{}_{}".format(block, dilation)
                    filter_bias = "filter_bias_{}_{}".format(block, dilation)
                    skip_bias = "skip_bias_{}_{}".format(block, dilation)
                    dense_bias = "dense_bias_{}_{}".format(block, dilation)

                    self.var[gate_bias] = tf.get_variable(gate_bias, [512],
                                                          initializer=tf.constant_initializer(0.0))

                    self.var[filter_bias] = tf.get_variable(filter_bias, [512],
                                                            initializer=tf.constant_initializer(0.0))

                    self.var[skip_bias] = tf.get_variable(skip_bias, [512],
                                                          initializer=tf.constant_initializer(0.1))

                    self.var[dense_bias] = tf.get_variable(dense_bias, [512],
                                                           initializer=tf.constant_initializer(0.1))

        with tf.name_scope('post_process'):
            self.var['conv1'] = tf.get_variable('conv1', [1, 512, 512],
                                                initializer=tf.random_normal_initializer(stddev=0.01))
            self.var['conv2'] = tf.get_variable('conv2', [1, 512, 4716],
                                                initializer=tf.random_normal_initializer(stddev=0.01))
            self.var['bias1'] = tf.get_variable('bias1', [512],
                                                initializer=tf.constant_initializer(0.1))
            self.var['bias2'] = tf.get_variable('bias2', [4716],
                                                initializer=tf.constant_initializer(0.1))

    # Helper Functions
    def _causal_conv(self, value, filter_, dilation, name='causal_conv'):
        """Performs 1d convolution

        For dilated conv, function performs reshaping and un-reshaping to create the
        dilated effect.
        Args:
          value: 3d tf tensor
          filter_: 3d tf variable
          dilation: int
          name: string

        Returns:
          result: 3d tf tensor
        """
        with tf.name_scope(name):
            filter_width = tf.shape(filter_)[0]
            if dilation > 1:
                transformed = self._time_to_batch(value, dilation)
                conv = tf.nn.conv1d(transformed, filter_, stride=1, padding='VALID')
                restored = self._batch_to_time(conv, dilation)
            else:
                restored = tf.nn.conv1d(value, filter_, stride=1, padding='VALID')

            out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
            result = tf.slice(restored, [0, 0, 0], [-1, out_width, -1])
            return result

    def _time_to_batch(self, value, dilation, name=None):
        """value shape [1, 300, 1024] or [num_sample, timesteps, channels]
        Convert 3d tensor into dilated form
        """
        with tf.name_scope('time_to_batch'):
            shape = tf.shape(value)
            pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
            padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
            reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
            transposed = tf.transpose(reshaped, [1, 0, 2])
            return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

    def _batch_to_time(self, value, dilation, name=None):
        """Convert back to original tensor"""
        with tf.name_scope('batch_to_time'):
            shape = tf.shape(value)
            prepared = tf.reshape(value, [dilation, -1, shape[2]])
            transposed = tf.transpose(prepared, perm=[1, 0, 2])
            return tf.reshape(transposed, [tf.div(shape[0], dilation), -1, shape[2]])
