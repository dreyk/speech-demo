# coding=utf-8
"""
Based on the paper:

http://arxiv.org/pdf/1601.06581v2.pdf

And some improvements from :

https://arxiv.org/pdf/1609.05935v2.pdf

This model is:

Acoustic RNN trained with ctc loss
"""

import tensorflow as tf
from tensorflow.python.client import timeline
from tensorflow.python.training.summary_io import SummaryWriterCache
import numpy as np
import time
import os
from datetime import datetime
import logging
from random import randint
import util.audioprocessor as audioprocessor
import util.dataprocessor as dataprocessor
import horovod.tensorflow as hvd


class AcousticModel(object):
    def __init__(self, num_layers, hidden_size, batch_size, max_input_seq_length,
                 max_target_seq_length, input_dim, normalization, num_labels):
        """
        Initialize the acoustic rnn model parameters

        Parameters
        ----------
        :param num_layers: number of lstm layers
        :param hidden_size: size of hidden layers
        :param batch_size: number of training examples fed at once
        :param max_input_seq_length: maximum length of input vector sequence
        :param max_target_seq_length: maximum length of ouput vector sequence
        :param input_dim: dimension of input vector
        :param normalization: boolean indicating whether or not to normalize data in a input batch
        :param num_labels: the numbers of output labels
        """
        # Store model's parameters
        self.iterator_handle = None
        self.use_local = None
        self.t_iterator_init = None
        self.v_iterator_init = None
        self.handle_train = None
        self.handle_v = None
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.max_input_seq_length = max_input_seq_length
        self.max_target_seq_length = max_target_seq_length
        self.input_dim = input_dim
        self.normalization = normalization
        self.num_labels = num_labels

        # Create object's variables for tensorflow ops
        self.rnn_state_zero_op = None
        self.rnn_keep_state_op = None
        self.saver_op = None

        # Create object's variable for result output
        self.prediction = None

        # Create object's variables for placeholders
        self.input_keep_prob_ph = self.output_keep_prob_ph = None
        self.inputs_ph = self.input_seq_lengths_ph = self.labels_ph = None

        # Create object's variables for dataset's iterator input
        self.iterator_get_next_op = None
        self.is_training_var = tf.Variable(initial_value=False, trainable=False, name="is_training_var",collections=self.use_local, dtype=tf.bool)

        # Create object's variable for hidden state
        self.rnn_tuple_state = None

        # Create object's variables for training
        self.input_keep_prob = self.output_keep_prob = None
        self.global_step = None
        self.learning_rate_var = None
        # Create object variables for tensorflow training's ops
        self.learning_rate_decay_op = None
        self.accumulated_mean_loss = self.acc_mean_loss_op = self.acc_mean_loss_zero_op = None
        self.accumulated_error_rate = self.acc_error_rate_op = self.acc_error_rate_zero_op = None
        self.mini_batch = self.increase_mini_batch_op = self.mini_batch_zero_op = None
        self.acc_gradients_zero_op = self.accumulate_gradients_op = None
        self.train_step_op = None

        # Create object's variables for tensorboard
        self.tensorboard_dir = None
        self.timeline_enabled = False
        self.train_summaries_op = None
        self.test_summaries_op = None
        self.summary_writer_op = None

        # Create object's variables for status checking
        self.rnn_created = False

    def create_forward_rnn(self):
        """
        Create the forward-only RNN

        Parameters
        -------
        :return: the logits
        """
        if self.rnn_created:
            logging.fatal("Trying to create the acoustic RNN but it is already.")

        # Set placeholders for input
        self.inputs_ph = tf.placeholder(tf.float32, shape=[self.max_input_seq_length, None, self.input_dim],
                                        name="inputs_ph")

        self.input_seq_lengths_ph = tf.placeholder(tf.int32, shape=[None], name="input_seq_lengths_ph")

        # Build the RNN
        self.global_step, logits, self.prediction, self.rnn_keep_state_op, self.rnn_state_zero_op,\
            _, _, self.rnn_tuple_state = self._build_base_rnn(self.inputs_ph, self.input_seq_lengths_ph, True)

        return logits

    def create_training_rnn(self,is_mpi, input_keep_prob, output_keep_prob, grad_clip, learning_rate, lr_decay_factor,
                            use_iterator=False):
        """
        Create the training RNN

        Parameters
        ----------
        :param input_keep_prob: probability of keeping input signal for a cell during training
        :param output_keep_prob: probability of keeping output signal from a cell during training
        :param grad_clip: max gradient size (prevent exploding gradients)
        :param learning_rate: learning rate parameter fed to optimizer
        :param lr_decay_factor: decay factor of the learning rate
        :param use_iterator: if True then plug an iterator.get_next() operation for the input of the model, if None
                            placeholders are created instead
        """
        if self.rnn_created:
            logging.fatal("Trying to create the acoustic RNN but it is already.")

        # Store model parameters
        self.is_mpi = is_mpi
        #self.use_local =[tf.GraphKeys.LOCAL_VARIABLES]

        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        if use_iterator is True:
            mfcc_batch, input_lengths, label_batch = self.iterator_get_next_op
            # Pad if the batch is not complete
            padded_mfcc_batch = tf.pad(mfcc_batch, [[0, self.batch_size - tf.size(input_lengths)], [0, 0], [0, 0]])
            # Transpose padded_mfcc_batch in order to get time serie as first dimension
            # [batch_size, time_serie, input_dim] ====> [time_serie, batch_size, input_dim]
            inputs = tf.transpose(padded_mfcc_batch, perm=[1, 0, 2])
            # Pad input_seq_lengths if the batch is not complete
            input_seq_lengths = tf.pad(input_lengths, [[0, self.batch_size - tf.size(input_lengths)]])

            # Label tensor must be provided as a sparse tensor.
            idx = tf.where(tf.not_equal(label_batch, 0))
            sparse_labels = tf.SparseTensor(idx, tf.gather_nd(label_batch, idx),
                                            [self.batch_size, self.max_target_seq_length])
            # Pad sparse_labels if the batch is not complete
            sparse_labels, _ = tf.sparse_fill_empty_rows(sparse_labels, self.num_labels - 1)
        else:
            # Set placeholders for input
            self.inputs_ph = tf.placeholder(tf.float32, shape=[self.max_input_seq_length, None, self.input_dim],
                                            name="inputs_ph")

            self.input_seq_lengths_ph = tf.placeholder(tf.int32, shape=[None], name="input_seq_lengths_ph")
            self.labels_ph = tf.placeholder(tf.int32, shape=[None, self.max_target_seq_length],
                                            name="labels_ph")
            inputs = self.inputs_ph
            input_seq_lengths = self.input_seq_lengths_ph
            label_batch = self.labels_ph

            # Label tensor must be provided as a sparse tensor.
            # First get indexes from non-zero positions
            idx = tf.where(tf.not_equal(label_batch, 0))
            # Then build a sparse tensor from indexes
            sparse_labels = tf.SparseTensor(idx, tf.gather_nd(label_batch, idx),
                                            [self.batch_size, self.max_target_seq_length])

        self.global_step, logits, prediction, self.rnn_keep_state_op, self.rnn_state_zero_op, self.input_keep_prob_ph,\
            self.output_keep_prob_ph, self.rnn_tuple_state = self._build_base_rnn(inputs, input_seq_lengths, False)

        # Add the train part to the network
        self.learning_rate_var = self._add_training_on_rnn(logits, grad_clip, learning_rate, lr_decay_factor,
                                                           sparse_labels, input_seq_lengths, prediction)


    def _build_base_rnn(self, inputs, input_seq_lengths, forward_only=True):
        """
        Build the Acoustic RNN

        Parameters
        ----------
        :param inputs: inputs to the RNN
        :param input_seq_lengths: vector containing the length of each input from 'inputs'
        :param forward_only: whether the RNN will be used for training or not (if true then add a dropout layer)
        
        Returns
        ----------
        :returns logits: each char probability for each timestep of the input, for each item of the batch
        :returns prediction: the best prediction for the input
        :returns rnn_keep_state_op: a tensorflow op to save the RNN internal state for the next batch
        :returns rnn_state_zero_op: a tensorflow op to reset the RNN internal state to zeros
        :returns input_keep_prob_ph: a placeholder for input_keep_prob of the dropout layer
                                     (None if forward_only is True)
        :returns output_keep_prob_ph: a placeholder for output_keep_prob of the dropout layer
                                      (None if forward_only is True)
        :returns rnn_tuple_state: the RNN internal state
        """
        # Define a variable to keep track of the learning process step
        global_step = tf.train.get_or_create_global_step()

        # If building the RNN for training then create dropout rate placeholders
        input_keep_prob_ph = output_keep_prob_ph = None
        if not forward_only:
            with tf.name_scope('dropout'):
                # Create placeholders, used to override values when running on the test set
                input_keep_prob_ph = tf.placeholder(tf.float32)
                output_keep_prob_ph = tf.placeholder(tf.float32)

        # Define cells of acoustic model
        with tf.variable_scope('LSTM'):
            # Create each layer
            layers_list = []
            for _ in range(self.num_layers):
                cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)

                # If building the RNN for training then add a dropoutWrapper to the cells
                if not forward_only:
                    with tf.name_scope('dropout'):
                        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=input_keep_prob_ph,
                                                             output_keep_prob=output_keep_prob_ph)
                layers_list.append(cell)

            # Store the layers in a multi-layer RNN
            cell = tf.contrib.rnn.MultiRNNCell(layers_list, state_is_tuple=True)

        # Build the input layer between input and the RNN
        with tf.variable_scope('Input_Layer'):
            w_i = tf.get_variable("input_w", [self.input_dim, self.hidden_size], tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_i = tf.get_variable("input_b", [self.hidden_size], tf.float32,
                                  initializer=tf.constant_initializer(0.0))

        # Apply the input layer to the network input to produce the input for the rnn part of the network
        rnn_inputs = [tf.matmul(tf.squeeze(i, axis=[0]), w_i) + b_i
                      for i in tf.split(axis=0, num_or_size_splits=self.max_input_seq_length, value=inputs)]
        # Switch from a list to a tensor
        rnn_inputs = tf.stack(rnn_inputs)

        # Add a batch normalization layer to the model if needed
        if self.normalization:
            with tf.name_scope('Normalization'):
                epsilon = 1e-3
                # Note : the tensor is [time, batch_size, input vector] so we go against dim 1
                batch_mean, batch_var = tf.nn.moments(rnn_inputs, [1], shift=None, name="moments", keep_dims=True)
                rnn_inputs = tf.nn.batch_normalization(rnn_inputs, batch_mean, batch_var, None, None,
                                                       epsilon, name="batch_norm")

        # Define some variables to store the RNN state
        # Note : tensorflow keep the state inside a batch but it's necessary to do this in order to keep the state
        #        between batches, especially when doing live transcript
        #        Another way would have been to get the state as an output of the session and feed it every time but
        #        this way is much more efficient
        with tf.variable_scope('Hidden_state'):
            state_variables = []
            for state_c, state_h in cell.zero_state(self.batch_size, tf.float32):
                state_variables.append(tf.nn.rnn_cell.LSTMStateTuple(
                    tf.Variable(state_c, trainable=False,collections=self.use_local),
                    tf.Variable(state_h, trainable=False,collections=self.use_local)))
            # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
            rnn_tuple_state = tuple(state_variables)

        # Build the RNN
        with tf.name_scope('LSTM'):
            rnn_output, new_states = tf.nn.dynamic_rnn(cell, rnn_inputs, sequence_length=input_seq_lengths,
                                                       initial_state=rnn_tuple_state, time_major=True)

        # Define an op to keep the hidden state between batches
        update_ops = []
        for state_variable, new_state in zip(rnn_tuple_state, new_states):
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(new_state[0]),
                               state_variable[1].assign(new_state[1])])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        rnn_keep_state_op = tf.tuple(update_ops)

        # Define an op to reset the hidden state to zeros
        update_ops = []
        for state_variable in rnn_tuple_state:
            # Assign the new state to the state variables on this layer
            update_ops.extend([state_variable[0].assign(tf.zeros_like(state_variable[0])),
                               state_variable[1].assign(tf.zeros_like(state_variable[1]))])
        # Return a tuple in order to combine all update_ops into a single operation.
        # The tuple's actual value should not be used.
        rnn_state_zero_op = tf.tuple(update_ops)

        # Build the output layer between the RNN and the char_map
        with tf.variable_scope('Output_layer'):
            w_o = tf.get_variable("output_w", [self.hidden_size, self.num_labels], tf.float32,
                                  initializer=tf.contrib.layers.xavier_initializer())
            b_o = tf.get_variable("output_b", [self.num_labels], tf.float32,
                                  initializer=tf.constant_initializer(0.0))

        # Compute the logits (each char probability for each timestep of the input, for each item of the batch)
        logits = tf.stack([tf.matmul(tf.squeeze(i, axis=[0]), w_o) + b_o
                          for i in tf.split(axis=0, num_or_size_splits=self.max_input_seq_length, value=rnn_output)])

        # Compute the prediction which is the best "path" of probabilities for each item of the batch
        decoded, _log_prob = tf.nn.ctc_beam_search_decoder(logits, input_seq_lengths)
        # Set the RNN result to the best path found
        prediction = tf.to_int32(decoded[0])

        return global_step, logits, prediction, rnn_keep_state_op, rnn_state_zero_op,\
            input_keep_prob_ph, output_keep_prob_ph, rnn_tuple_state

    def _add_training_on_rnn(self, logits, grad_clip, learning_rate, lr_decay_factor,
                             sparse_labels, input_seq_lengths, prediction):
        """
        Build the training add-on of the Acoustic RNN
        
        This add-on offer ops that can be used to train the network :
          * self.learning_rate_decay_op : will decay the learning rate
          * self.acc_mean_loss_op : will compute the loss and accumulate it over multiple mini-batchs
          * self.acc_mean_loss_zero_op : will reset the loss accumulator to 0
          * self.acc_error_rate_op : will compute the error rate and accumulate it over multiple mini-batchs
          * self.acc_error_rate_zero_op : will reset the error_rate accumulator to 0
          * self.increase_mini_batch_op : will increase the mini-batchs counter
          * self.mini_batch_zero_op : will reset the mini-batchs counter
          * self.acc_gradients_zero_op : will reset the gradients
          * self.accumulate_gradients_op : will compute the gradients and accumulate them over multiple mini-batchs
          * self.train_step_op : will clip the accumulated gradients and apply them on the RNN

        Parameters
        ----------
        :param logits: the output of the RNN before the beam search
        :param grad_clip: max gradient size (prevent exploding gradients)
        :param learning_rate: learning rate parameter fed to optimizer
        :param lr_decay_factor: decay factor of the learning rate
        :param sparse_labels: the labels in a sparse tensor
        :param input_seq_lengths: vector containing the length of each input from 'inputs'
        :param prediction: the predicted label given by the RNN

        Returns
        -------
        :returns: tensorflow variable keeping the current learning rate
        """
        # Define the variable for the learning rate
        learning_rate_var = tf.Variable(float(learning_rate), trainable=False, name='learning_rate')
        # Define an op to decrease the learning rate
        self.learning_rate_decay_op = learning_rate_var.assign(tf.multiply(learning_rate_var, lr_decay_factor))

        # Compute the CTC loss between the logits and the truth for each item of the batch
        with tf.name_scope('CTC'):
            ctc_loss = tf.nn.ctc_loss(sparse_labels, logits, input_seq_lengths, ignore_longer_outputs_than_inputs=True)

            # Compute the mean loss of the batch (only used to check on progression in learning)
            # The loss is averaged accross the batch but before we take into account the real size of the label
            mean_loss = tf.reduce_mean(tf.truediv(ctc_loss, tf.to_float(input_seq_lengths)))

            # Set an accumulator to sum the loss between mini-batchs
            self.accumulated_mean_loss = tf.Variable(0.0, trainable=False,collections=self.use_local)
            self.acc_mean_loss_op = self.accumulated_mean_loss.assign_add(mean_loss)
            self.acc_mean_loss_zero_op = self.accumulated_mean_loss.assign(tf.zeros_like(self.accumulated_mean_loss))

        # Compute the error between the logits and the truth
        with tf.name_scope('Error_Rate'):
            error_rate = tf.reduce_mean(tf.edit_distance(prediction, sparse_labels, normalize=True))

            # Set an accumulator to sum the error rate between mini-batchs
            self.accumulated_error_rate = tf.Variable(0.0, trainable=False,collections=self.use_local)
            self.acc_error_rate_op = self.accumulated_error_rate.assign_add(error_rate)
            self.acc_error_rate_zero_op = self.accumulated_error_rate.assign(tf.zeros_like(self.accumulated_error_rate))

        # Count mini-batchs
        with tf.name_scope('Mini_batch'):
            # Set an accumulator to count the number of mini-batchs in a batch
            # Note : variable is defined as float to avoid type conversion error using tf.divide
            self.mini_batch = tf.Variable(0.0, trainable=False,collections=self.use_local)
            self.increase_mini_batch_op = self.mini_batch.assign_add(1)
            self.mini_batch_zero_op = self.mini_batch.assign(tf.zeros_like(self.mini_batch))

        # Compute the gradients
        trainable_variables = tf.trainable_variables()
        with tf.name_scope('Gradients'):
            opt = tf.train.AdamOptimizer(learning_rate_var)
            #if self.is_mpi is True:
            #    opt = hvd.DistributedOptimizer(opt)

            gradients = opt.compute_gradients(ctc_loss, trainable_variables)

            # Define a list of variables to store the accumulated gradients between batchs
            accumulated_gradients = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False,collections=self.use_local)
                                     for tv in trainable_variables]

            # Define an op to reset the accumulated gradient
            self.acc_gradients_zero_op = [tv.assign(tf.zeros_like(tv)).op for tv in accumulated_gradients]

            # Define an op to accumulate the gradients calculated by the current batch with
            # the accumulated gradients variable
            self.accumulate_gradients_op = [accumulated_gradients[i].assign_add(gv[0]).op
                                            for i, gv in enumerate(gradients)]

            # Define an op to apply the result of the accumulated gradients
            if self.is_mpi is True:
                with tf.name_scope('Gradients_Allreduce'):
                    reduced_gradients = [hvd.allreduce(v) for v in accumulated_gradients]
                    clipped_gradients, _norm = tf.clip_by_global_norm(reduced_gradients, grad_clip)
                    self.train_step_op = opt.apply_gradients([(clipped_gradients[i], gv[1]) for i, gv in enumerate(gradients)],
                                                             global_step=self.global_step)
            else:
                clipped_gradients, _norm = tf.clip_by_global_norm(accumulated_gradients, grad_clip)
                self.train_step_op = opt.apply_gradients([(clipped_gradients[i], gv[1]) for i, gv in enumerate(gradients)],
                                                     global_step=self.global_step)
        return learning_rate_var

    def add_tensorboard(self, tensorboard_dir, timeline_enabled=False):
        """
        Add the tensorboard operations to the acoustic RNN
        This method will add ops to feed tensorboard
          self.train_summaries_op : will produce the summary for a training step
          self.test_summaries_op : will produce the summary for a test step
          self.summary_writer_op : will write the summary to disk

        Parameters
        ----------
        :param session: the tensorflow session
        :param tensorboard_dir: path to tensorboard directory
        :param tb_run_name: directory name for the tensorboard files inside tensorboard_dir, if None a default dir
                            will be created
        :param timeline_enabled: enable the output of a trace file for timeline visualization
        """
        if tensorboard_dir is None:
            return
        self.tensorboard_dir = tensorboard_dir
        self.timeline_enabled = timeline_enabled

        # Define GraphKeys for TensorBoard
        graphkey_training = tf.GraphKeys()
        graphkey_test = tf.GraphKeys()

        # Learning rate
        tf.summary.scalar('Learning_rate', self.learning_rate_var, collections=[graphkey_training, graphkey_test])

        # Loss
        with tf.name_scope('Mean_loss'):
            mean_loss = tf.divide(self.accumulated_mean_loss, self.mini_batch)
            tf.summary.scalar('Training', mean_loss, collections=[graphkey_training])
            tf.summary.scalar('Test', mean_loss, collections=[graphkey_test])

        # Accuracy
        with tf.name_scope('Accuracy_-_Error_Rate'):
            mean_error_rate = tf.divide(self.accumulated_error_rate, self.mini_batch)
            tf.summary.scalar('Training', mean_error_rate, collections=[graphkey_training])
            tf.summary.scalar('Test', mean_error_rate, collections=[graphkey_test])

        # Hidden state
        with tf.name_scope('RNN_internal_state'):
            for idx, state_variable in enumerate(self.rnn_tuple_state):
                tf.summary.histogram('Training_layer-{0}_cell_state'.format(idx), state_variable[0],
                                     collections=[graphkey_training])
                tf.summary.histogram('Test_layer-{0}_cell_state'.format(idx), state_variable[0],
                                     collections=[graphkey_test])
                tf.summary.histogram('Training_layer-{0}_hidden_state'.format(idx), state_variable[1],
                                     collections=[graphkey_training])
                tf.summary.histogram('Test_layer-{0}_hidden_state'.format(idx), state_variable[1],
                                     collections=[graphkey_test])

        self.train_summaries_op = tf.summary.merge_all(key=graphkey_training)
        self.test_summaries_op = tf.summary.merge_all(key=graphkey_test)


    def get_learning_rate(self):
        return self.learning_rate_var.eval()

    def set_learning_rate(self, sess, learning_rate):
        assign_op = self.learning_rate_var.assign(learning_rate)
        sess.run(assign_op)

    def set_is_training(self, sess, is_training):
        assign_op = self.is_training_var.assign(is_training)
        sess.run(assign_op)

    @staticmethod
    def initialize(sess):
        # Initialize variables
        sess.run(tf.global_variables_initializer())

    def save(self, session, checkpoint_dir):
        if self.saver_op != None:
            # Save the model
            checkpoint_path = os.path.join(checkpoint_dir, "acousticmodel.ckpt")
            self.saver_op.save(session, checkpoint_path, global_step=self.global_step)
            logging.info("Checkpoint saved")

    def restore(self, session, checkpoint_dir):
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        # Restore from checkpoint (will overwrite variables)
        if ckpt:
            self.saver_op.restore(session, ckpt.model_checkpoint_path)
            logging.info("Restored model parameters from %s (global_step id %d)", ckpt.model_checkpoint_path,
                         self.global_step.eval(session=session))
        else:
            logging.info("Created model with fresh parameters.")
        return

    @staticmethod
    def calculate_wer(first_string, second_string):
        """
        Source : https://martin-thoma.com/word-error-rate-calculation/

        Calculation of WER with Levenshtein distance.

        Works only for strings up to 254 characters (uint8).
        O(nm) time ans space complexity.

        Parameters
        ----------
        first_string : string
        second_string : string

        Returns
        -------
        int

        Examples
        --------
        > calculate_wer("who is there", "is there")
        1
        > calculate_wer("who is there", "")
        3
        > calculate_wer("", "who is there")
        3
        """
        # initialisation
        r = first_string.split()
        h = second_string.split()

        d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint8)
        d = d.reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1):
            for j in range(len(h) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return d[len(r)][len(h)]

    @staticmethod
    def calculate_cer(first_string, second_string):
        """
        Calculation of Character Error Rate (CER).

        Works only for strings up to 65635 elements (uint16).

        Parameters
        ----------
        first_string : string
        second_string : string

        Returns
        -------
        int

        Examples
        --------
        > calculate_cer("who is there", "whois there")
        0
        > calculate_cer("who is there", "who i thre")
        2
        > calculate_cer("", "who is there")
        10
        """
        # initialisation
        r = list(first_string.replace(" ", ""))
        h = list(second_string.replace(" ", ""))

        d = np.zeros((len(r) + 1) * (len(h) + 1), dtype=np.uint16)
        d = d.reshape((len(r) + 1, len(h) + 1))
        for i in range(len(r) + 1):
            for j in range(len(h) + 1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r) + 1):
            for j in range(1, len(h) + 1):
                if r[i - 1] == h[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        return d[len(r)][len(h)]

    def run_step(self, session, compute_gradients=True, run_options=None, run_metadata=None):
        """
        Returns:
        mean of ctc_loss
        """
        # Base output is to accumulate loss, error_rate, increase the mini-batchs counter and keep the hidden state for
        # next batch
        output_feed = [self.acc_mean_loss_op, self.acc_error_rate_op,
                       self.increase_mini_batch_op, self.rnn_keep_state_op]

        if compute_gradients:
            # Add the update operation
            output_feed.append(self.accumulate_gradients_op)
            # and feed the dropout layer the keep probability values
            input_feed = {self.input_keep_prob_ph: self.input_keep_prob,
                          self.output_keep_prob_ph: self.output_keep_prob,
                          self.iterator_handle: self.handle_train}
        else:
            # No need to apply a dropout, set the keep probability to 1.0
            input_feed = {self.input_keep_prob_ph: 1.0, self.output_keep_prob_ph: 1.0,
                          self.iterator_handle: self.handle_v}

        # Actually run the tensorflow session
        start_time = time.time()
        logging.debug("Starting a step")
        session.run(output_feed, input_feed, options=run_options, run_metadata=run_metadata)
        mini_batch_num = self.mini_batch.eval(session=session)
        logging.debug("Step duration : %.2f", time.time() - start_time)
        return mini_batch_num

    def start_batch(self, session, is_training, run_options=None, run_metadata=None):
        output = [self.acc_error_rate_zero_op, self.acc_mean_loss_zero_op, self.mini_batch_zero_op]

        ##self.set_is_training(session, is_training)
        if is_training:
            output.append(self.acc_gradients_zero_op)

        session.run(output, options=run_options, run_metadata=run_metadata)
        return

    def end_batch(self, session, is_training, run_options=None, run_metadata=None, rnn_state_reset_ratio=1.0):
        # Get each accumulator's value and compute the mean for the batch
        output_feed = [self.accumulated_mean_loss, self.accumulated_error_rate, self.mini_batch, self.global_step]

        # If in training...
        if is_training:
            # Append the train_step_op (this will apply the gradients)
            output_feed.append(self.train_step_op)
            # Reset the hidden state at the given random ratio (default to always)
            if randint(1, 1 // rnn_state_reset_ratio) == 1:
                output_feed.append(self.rnn_state_zero_op)
        if self.tensorboard_dir is not None:
            if is_training:
                output_feed.append(self.train_summaries_op)
            else:
                output_feed.append(self.test_summaries_op)

        outputs = session.run(output_feed, options=run_options, run_metadata=run_metadata)
        accumulated_loss = outputs[0]
        accumulated_error_rate = outputs[1]
        batchs_count = outputs[2]
        global_step = outputs[3]

        if self.tensorboard_dir is not None:
            summary = outputs[-1]
            if self.summary_writer_op is None:
                self.summary_writer_op = SummaryWriterCache.get(self.tensorboard_dir)
            self.summary_writer_op.add_summary(summary, global_step)

        mean_loss = accumulated_loss / batchs_count
        mean_error_rate = accumulated_error_rate / batchs_count
        return mean_loss, mean_error_rate, global_step

    def process_input(self, session, inputs, input_seq_lengths, run_options=None, run_metadata=None):
        """
        Returns:
          Output vector
        """
        input_feed = {self.inputs_ph: np.array(inputs), self.input_seq_lengths_ph: np.array(input_seq_lengths)}

        if (self.input_keep_prob_ph is not None) and (self.output_keep_prob_ph is not None):
            input_feed[self.input_keep_prob_ph] = 1.0
            input_feed[self.output_keep_prob_ph] = 1.0

        output_feed = [self.prediction]
        outputs = session.run(output_feed, input_feed, options=run_options, run_metadata=run_metadata)
        predictions = session.run(tf.sparse_tensor_to_dense(outputs[0], default_value=self.num_labels,
                                                            validate_indices=True),
                                  options=run_options, run_metadata=run_metadata)
        return predictions

    def evaluate_full(self, sess, eval_dataset, input_seq_length, signal_processing, char_map,
                      run_options=None, run_metadata=None):
        # Create an audio_processor
        audio_processor = audioprocessor.AudioProcessor(input_seq_length, signal_processing)

        wer_list = []
        cer_list = []
        file_number = 0
        input_feat_vecs = []
        input_feat_vec_lengths = []
        labels = []
        for file, label, _ in eval_dataset:
            feat_vec, feat_vec_length = audio_processor.process_audio_file(file)
            file_number += 1
            label_data_length = len(label)
            if (label_data_length > self.max_target_seq_length) or\
               (feat_vec_length > self.max_input_seq_length):
                logging.warning("Warning - sample too long : %s (input : %d / text : %s)",
                                file, feat_vec_length, label_data_length)
            else:
                logging.debug("Processed file %d / %d", file_number, len(eval_dataset))
                input_feat_vecs.append(feat_vec)
                input_feat_vec_lengths.append(feat_vec_length)
                labels.append(label)

            # If we reached the last file then pad the lists to obtain a full batch
            if file_number == len(eval_dataset):
                for i in range(self.batch_size - len(input_feat_vecs)):
                    input_feat_vecs.append(np.zeros([self.max_input_seq_length,
                                                     audio_processor.feature_size]))
                    input_feat_vec_lengths.append(0)
                    labels.append("")

            if len(input_feat_vecs) == self.batch_size:
                # Run the batch
                logging.debug("Running a batch")
                input_feat_vecs = np.swapaxes(input_feat_vecs, 0, 1)
                predictions = self.process_input(sess, input_feat_vecs, input_feat_vec_lengths,
                                                 run_options=run_options, run_metadata=run_metadata)
                for index, prediction in enumerate(predictions):
                    transcribed_text = dataprocessor.DataProcessor.get_labels_str(char_map, prediction)
                    true_label = labels[index]
                    if len(true_label) > 0:
                        nb_words = len(true_label.split())
                        nb_chars = len(true_label.replace(" ", ""))
                        wer_list.append(self.calculate_wer(transcribed_text, true_label) / float(nb_words))
                        cer_list.append(self.calculate_cer(transcribed_text, true_label) / float(nb_chars))
                # Reset the lists
                input_feat_vecs = []
                input_feat_vec_lengths = []
                labels = []

        wer = (sum(wer_list) * 100) / float(len(wer_list))
        cer = (sum(cer_list) * 100) / float(len(cer_list))
        return wer, cer

    def run_evaluation(self, sess, run_options=None, run_metadata=None):
        start_time = time.time()
        logging.info("Start evaluating...")

        # Start a new batch
        self.start_batch(sess, False, run_options=run_options, run_metadata=run_metadata)

        try:
            while True:
                self.run_step(sess, False, run_options=run_options, run_metadata=run_metadata)
        except tf.errors.OutOfRangeError:
            logging.debug("Dataset empty, exiting evaluation step")

        # Close the batch (always reset the RNN state after each batch in evaluation mode)
        mean_loss, mean_error_rate, current_step = self.end_batch(sess, False, run_options=run_options,
                                                                  run_metadata=run_metadata,
                                                                  rnn_state_reset_ratio=1.0)
        logging.info("Evaluation at step %d : loss %.5f - error_rate %.5f - duration %.2f",
                     current_step, mean_loss, mean_error_rate, time.time() - start_time)

        return mean_loss, mean_error_rate, current_step



    @staticmethod
    def build_dataset_from_records(input_set, batch_size, max_input_seq_length):
        audio_dataset = tf.data.TFRecordDataset([f for f in input_set])
        # Separate each data from the input list
        feature = {'audio': tf.FixedLenFeature([], tf.string),
                   'label': tf.FixedLenFeature([], tf.string),
                   'length': tf.FixedLenFeature([], tf.int64)}
        def _parse(f):
            features = tf.parse_single_example(f, features=feature)
            audio = tf.reshape(tf.decode_raw(features['audio'], tf.float32),[-1,120])
            labels = tf.decode_raw(features['label'], tf.int32)
            length = tf.cast(features['length'], tf.int32)
            return audio,length,labels


        audio_dataset = audio_dataset.map(_parse).prefetch(30)

        # Batch the datasets
        audio_dataset = audio_dataset.padded_batch(batch_size, padded_shapes=([max_input_seq_length, None],
                                                                              tf.TensorShape([]),
                                                                              [None]))

        return audio_dataset

    @staticmethod
    def build_dataset(input_set, batch_size, max_input_seq_length, max_target_seq_length,
                      signal_processing, char_map):
        # Separate each data from the input list
        audio_and_label_set = [[item[0], item[1]] for item in input_set]
        audio_dataset = tf.data.Dataset.from_tensor_slices(audio_and_label_set)

        # Read audio data and convert string labels
        def _read_audio_and_transcode_label(filename_label):
            # Need to convert back to string because tf.py_func changed it to a numpy array
            filename = str(filename_label[0], encoding='UTF-8')
            label = str(filename_label[1], encoding='UTF-8')
            audio_processor = audioprocessor.AudioProcessor(max_input_seq_length, signal_processing)
            audio_decoded, audio_length = audio_processor.process_audio_file(filename)
            label_transcoded = dataprocessor.DataProcessor.get_str_labels(char_map, label)
            return np.array(audio_decoded, dtype=np.float32), np.array(audio_length, dtype=np.int32), \
                   np.array(label_transcoded, dtype=np.int32)

        audio_dataset = audio_dataset.map(lambda filename_label: tuple(tf.py_func(_read_audio_and_transcode_label,
                                                                                  [filename_label],
                                                                                  [tf.float32, tf.int32, tf.int32])),
                                          num_parallel_calls=2).prefetch(30)

        # Batch the datasets
        audio_dataset = audio_dataset.padded_batch(batch_size, padded_shapes=([max_input_seq_length, None],
                                                                              tf.TensorShape([]),
                                                                              [None]))

        # Convert the labels' batch to a sparse tensor
        # TODO : support will probably be ok with TF v1.5
        # def _convert_labels_to_sparse(audio, audio_lengths, dense_labels):
        #    idx = tf.where(tf.not_equal(dense_labels, 0))
        #    sparse_labels = tf.SparseTensor(idx, tf.gather_nd(dense_labels, idx), [max_target_seq_length, batch_size])
        #    return audio, audio_lengths, sparse_labels
        # audio_dataset = audio_dataset.map(_convert_labels_to_sparse)

        # TODO : add a filter for files which are too long (currently de-structuring with Dataset.filter is not
        #        supported in python3)

        return audio_dataset

    def simple_shuffle_batch(self,source, capacity,out_types,out_shape,batch_size=1):
        # Create a random shuffle queue.
        queue = tf.RandomShuffleQueue(capacity=capacity,
                                      min_after_dequeue=batch_size,
                                      dtypes=out_types)

        # Create an op to enqueue one item.
        enqueue = queue.enqueue(source)

        # Create a queue runner that, when started, will launch 4 threads applying
        # that enqueue op.
        num_threads = 4
        qr = tf.train.QueueRunner(queue, [enqueue] * num_threads)

        # Register the queue runner so it can be found and started by
        # `tf.train.start_queue_runners` later (the threads are not launched yet).
        tf.train.add_queue_runner(qr)

        # Create an op to dequeue a batch
        return queue.dequeue()

    def add_dataset_input_queue(self, dataset):

        iterator = dataset.repeat().make_one_shot_iterator().get_next()
        self.iterator_get_next_op = self.simple_shuffle_batch(iterator,10,dataset.output_types,dataset.output_shapes,3)
        return None

    def add_datasets_input_queue(self, dataset,valid_dataset):

        iterator = dataset.repeat().make_one_shot_iterator().get_next()
        self.iterator_get_next_op = self.simple_shuffle_batch(iterator,10,dataset.output_types,dataset.output_shapes,3)
        return None, None

    def add_dataset_input(self, train_dataset):
        self.add_datasets_input(train_dataset,None)
    def add_datasets_input(self, train_dataset, valid_dataset):
        """
        Add training and evaluation datasets for input to the model
        Warning : returned iterators must be initialized before use : "tf.Session.run(iterator.initializer)" on each

        Parameters
        ----------
        :param train_dataset: a tensorflow Dataset
        :param valid_dataset: a tensorflow Dataset
        :return t_iterator: tensorflow Iterator for the train dataset
        :return v_iterator: tensorflow Iterator for the valid dataset
        """
        t_iterator = None
        v_iterator = None
        if train_dataset is not None:
            t0_iterator = train_dataset.make_initializable_iterator()
            self.t_iterator_init = t0_iterator.initializer
            t_iterator  = t0_iterator.string_handle()

        if valid_dataset is not None:
            v0_iterator = valid_dataset.make_initializable_iterator()
            self.v_iterator_init = v0_iterator.initializer
            v_iterator  = v0_iterator.string_handle()

        self.iterator_handle = tf.placeholder(tf.string, shape=[])
        iterator = tf.contrib.data.Iterator.from_string_handle(self.iterator_handle, train_dataset.output_types, train_dataset.output_shapes)
        self.iterator_get_next_op = iterator.get_next()
        return t_iterator, v_iterator

    def _write_timeline(self, run_metadata, inter_time, action=""):
        logging.debug("--- Action %s duration : %.4f", action, time.time() - inter_time)

        if self.tensorboard_dir is None:
            logging.warning("Could not write timeline, a tensorboard_dir is required in config file")
            return

        # Create the Timeline object, and write it to a json
        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        logging.info('Writing to timeline-' + action + '.ctf.json')
        with open(self.tensorboard_dir + '/' + 'timeline-' + action + '.ctf.json', 'w') as trace_file:
            trace_file.write(trace.generate_chrome_trace_format())
        return time.time()

    def run_train_step(self, sess, mini_batch_size, rnn_state_reset_ratio, run_options=None, run_metadata=None):
        """
        Run a single train step 

        Parameters
        ----------
        :param sess: a tensorflow session
        :param mini_batch_size: the number of batchs to run before applying the gradients
        :param rnn_state_reset_ratio: the ratio to which the RNN internal state will be reset to 0
             example: 1.0 mean the RNN internal state will be reset at the end of each batch
             example: 0.25 mean there is 25% chances that the RNN internal state will be reset at the end of each batch
        :param run_options: options parameter for the sess.run calls
        :param run_metadata: run_metadata parameter for the sess.run calls
        :returns float mean_loss: mean loss for the train batch run
        :returns float mean_error_rate: mean error rate for the train batch run
        :returns int current_step: new value of the step counter at the end of this batch
        :returns bool dataset_empty: `True` if the dataset was emptied during the batch
        """
        start_time = inter_time = time.time()
        dataset_empty = False

        # Start a new batch
        self.start_batch(sess, True, run_options=run_options, run_metadata=run_metadata)
        if self.timeline_enabled:
            inter_time = self._write_timeline(run_metadata, inter_time, "start_batch")

        # Run multiple mini-batchs inside the train step
        mini_batch_num = 0
        try:
            for i in range(mini_batch_size):
                # Run a step on a batch and keep the loss
                mini_batch_num = self.run_step(sess, True, run_options=run_options, run_metadata=run_metadata)
                if self.timeline_enabled:
                    inter_time = self._write_timeline(run_metadata, inter_time, "step-" + str(i))
        except tf.errors.OutOfRangeError:
            logging.debug("Dataset empty, exiting train step")
            dataset_empty = True

        # Close the batch if at least a mini-batch was completed
        if mini_batch_num > 0:
            mean_loss, mean_error_rate, current_step = self.end_batch(sess, True, run_options=run_options,
                                                                      run_metadata=run_metadata,
                                                                      rnn_state_reset_ratio=rnn_state_reset_ratio)
            if self.timeline_enabled:
                _ = self._write_timeline(run_metadata, inter_time, "end_batch")

            # Step result
            logging.info("Batch %d : loss %.5f - error_rate %.5f - duration %.2f",
                         current_step, mean_loss, mean_error_rate, time.time() - start_time)

            return mean_loss, mean_error_rate, current_step, dataset_empty
        else:
            return 0.0, 0.0, self.global_step.eval(session=sess), dataset_empty
