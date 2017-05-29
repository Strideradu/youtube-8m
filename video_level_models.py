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

"""Contains model definitions."""
import math

import models
import tensorflow as tf
import utils

from tensorflow import flags
import tensorflow.contrib.slim as slim

FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "moe_num_mixtures", 2,
    "The number of mixtures (excluding the dummy 'expert') used for MoeModel.")


class LogisticModel(models.BaseModel):
    """Logistic model with L2 regularization."""

    def create_model(self, model_input, vocab_size, l2_penalty=1e-8, **unused_params):
        """Creates a logistic model.

        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.

        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        output = slim.fully_connected(
            model_input, vocab_size, activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(l2_penalty))
        return {"predictions": output}


class MoeModel(models.BaseModel):
    """A softmax over a mixture of logistic models (with L2 regularization)."""

    def create_model(self,
                     model_input,
                     vocab_size,
                     num_mixtures=None,
                     l2_penalty=1e-8,
                     **unused_params):
        """Creates a Mixture of (Logistic) Experts model.

         The model consists of a per-class softmax distribution over a
         configurable number of logistic classifiers. One of the classifiers in the
         mixture is not trained, and always predicts 0.

        Args:
          model_input: 'batch_size' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
          num_mixtures: The number of mixtures (excluding a dummy 'expert' that
            always predicts the non-existence of an entity).
          l2_penalty: How much to penalize the squared magnitudes of parameter
            values.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes.
        """
        num_mixtures = num_mixtures or FLAGS.moe_num_mixtures

        gate_activations = slim.fully_connected(
            model_input,
            vocab_size * (num_mixtures + 1),
            activation_fn=None,
            biases_initializer=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="gates")
        expert_activations = slim.fully_connected(
            model_input,
            vocab_size * num_mixtures,
            activation_fn=None,
            weights_regularizer=slim.l2_regularizer(l2_penalty),
            scope="experts")

        gating_distribution = tf.nn.softmax(tf.reshape(
            gate_activations,
            [-1, num_mixtures + 1]))  # (Batch * #Labels) x (num_mixtures + 1)
        expert_distribution = tf.nn.sigmoid(tf.reshape(
            expert_activations,
            [-1, num_mixtures]))  # (Batch * #Labels) x num_mixtures

        final_probabilities_by_class_and_batch = tf.reduce_sum(
            gating_distribution[:, :num_mixtures] * expert_distribution, 1)
        final_probabilities = tf.reshape(final_probabilities_by_class_and_batch,
                                         [-1, vocab_size])
        return {"predictions": final_probabilities}


class Model3nn2048BnReluSkipDouble(models.BaseModel):
    def create_model(self, model_input, vocab_size, num_hidden_units=2048, l2_penalty=1e-8, prefix='', **unused_params):
        """Creates a logistic model.
        Args:
          model_input: 'batch' x 'num_features' matrix of input features.
          vocab_size: The number of classes in the dataset.
        Returns:
          A dictionary with a tensor containing the probability predictions of the
          model in the 'predictions' key. The dimensions of the tensor are
          batch_size x num_classes."""

        # Initialize weights for projection
        w_s = tf.Variable(tf.random_normal(shape=[1152, 2048], stddev=0.01))
        input_projected = tf.matmul(model_input, w_s)

        hidden1 = tf.layers.dense(
            inputs=model_input, units=num_hidden_units, activation=None,
            kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix + 'fc_1')

        bn1 = slim.batch_norm(hidden1, scope=prefix + 'bn1')

        relu1 = tf.nn.relu(bn1, name=prefix + 'relu1')

        input_projected_plus_h1 = tf.add(input_projected, relu1)

        hidden2 = tf.layers.dense(
            inputs=input_projected_plus_h1, units=num_hidden_units, activation=None,
            kernel_regularizer=slim.l2_regularizer(l2_penalty), name=prefix + 'fc_2')

        bn2 = slim.batch_norm(hidden2, scope=prefix + 'bn2')

        relu2 = tf.nn.relu(bn2, name=prefix + 'relu2')

        input_projected_plus_h2 = tf.add_n([input_projected, input_projected_plus_h1, relu2])
        # input_projected_plus_h2 = tf.add(input_plus_h1, relu2)

        output = slim.fully_connected(
            input_projected_plus_h2, vocab_size, activation_fn=tf.nn.sigmoid,
            weights_regularizer=slim.l2_regularizer(l2_penalty), scope=prefix + 'fc_3')

        weights_norm = tf.add_n(tf.losses.get_regularization_losses())

        return {"predictions": output, "regularization_loss": weights_norm}
        # return {"predictions": output}
