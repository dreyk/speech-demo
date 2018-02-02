# coding=utf-8
"""
Main program to use the speech recognizer.
"""

from models.AcousticModel import AcousticModel
from models.LanguageModel import LanguageModel
from models.SpeechRecognizer import SpeechRecognizer
import tensorflow as tf
import numpy as np
import util.hyperparams as hyperparams
import util.audioprocessor as audioprocessor
import util.dataprocessor as dataprocessor
import argparse
import logging
from random import shuffle
import sys
from glob import glob
import horovod.tensorflow as hvd


def main():
    prog_params = parse_args()
    serializer = hyperparams.HyperParameterHandler(prog_params['config_file'])
    hyper_params = serializer.get_hyper_params()
    audio_processor = audioprocessor.AudioProcessor(hyper_params["max_input_seq_length"],
                                                    hyper_params["signal_processing"])
    # Get the input dimension for the RNN, depend on the chosen signal processing mode
    hyper_params["input_dim"] = audio_processor.feature_size

    speech_reco = SpeechRecognizer(hyper_params["language"])
    hyper_params["char_map"] = speech_reco.get_char_map()
    hyper_params["char_map_length"] = speech_reco.get_char_map_length()

    if prog_params['start_ps'] is True:
        start_ps_server(prog_params)
    if prog_params['save_acoustic'] is True:
        if hyper_params["dataset_size_ordering"] in ['True', 'First_run_only']:
            ordered = True
        else:
            ordered = False
        train_set, test_set = speech_reco.load_acoustic_dataset(hyper_params["training_dataset_dirs"],
                                                                hyper_params["test_dataset_dirs"],
                                                                hyper_params["training_filelist_cache"],
                                                                ordered,
                                                                hyper_params["train_frac"])
        save_acoustic_rnn(train_set, hyper_params, prog_params)
    elif (prog_params['train_acoustic'] is True) or (prog_params['dtrain_acoustic'] is True):
        if hyper_params["dataset_size_ordering"] in ['True', 'First_run_only']:
            ordered = True
        else:
            ordered = False
        #train_set = glob(prog_params['train_set'])
        train_set, test_set = speech_reco.load_acoustic_dataset(hyper_params["training_dataset_dirs"],
                                                                hyper_params["test_dataset_dirs"],
                                                                hyper_params["training_filelist_cache"],
                                                                ordered,
                                                                hyper_params["train_frac"])
        if prog_params['test_set'] is not None:
            test_set = prog_params['test_set']
        if prog_params['train_acoustic'] is True:
            train_acoustic_rnn(train_set, test_set, hyper_params, prog_params)
        else:
            distributed_train_acoustic_rnn(train_set, test_set, hyper_params, prog_params)
    elif prog_params['train_language'] is True:
        train_set, test_set = load_language_dataset(hyper_params)
        train_language_rnn(train_set, test_set, hyper_params, prog_params)
    elif prog_params['file'] is not None:
        process_file(audio_processor, hyper_params, prog_params['file'])
    elif prog_params['record'] is True:
        record_and_write(audio_processor, hyper_params)
    elif prog_params['evaluate'] is True:
        evaluate(hyper_params)
    elif prog_params['generate_text'] is True:
        generate_text(hyper_params)


def build_language_training_rnn(sess, hyper_params, prog_params, train_set, test_set):
    model = LanguageModel(hyper_params["num_layers"], hyper_params["hidden_size"], hyper_params["batch_size"],
                          hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                          hyper_params["char_map_length"])

    # Create a Dataset from the train_set and the test_set
    train_dataset = model.build_dataset(train_set, hyper_params["batch_size"], hyper_params["max_input_seq_length"],
                                        hyper_params["char_map"])

    v_iterator = None
    if test_set is []:
        t_iterator = model.add_dataset_input(train_dataset)
        sess.run(t_iterator.initializer)
    else:
        test_dataset = model.build_dataset(test_set, hyper_params["batch_size"], hyper_params["max_input_seq_length"],
                                           hyper_params["char_map"])

        # Build the input stream from the different datasets
        t_iterator, v_iterator = model.add_datasets_input(train_dataset, test_dataset)
        sess.run(t_iterator.initializer)
        sess.run(v_iterator.initializer)

    # Create the model
    model.create_training_rnn(hyper_params["dropout_input_keep_prob"], hyper_params["dropout_output_keep_prob"],
                              hyper_params["grad_clip"], hyper_params["learning_rate"],
                              hyper_params["lr_decay_factor"], use_iterator=True)
    model.add_tensorboard(sess, hyper_params["tensorboard_dir"], prog_params["tb_name"], prog_params["timeline"])
    model.initialize(sess)
    model.restore(sess, hyper_params["checkpoint_dir"] + "/language/")

    # Override the learning rate if given on the command line
    if prog_params["learn_rate"] is not None:
        model.set_learning_rate(sess, prog_params["learn_rate"])

    return model, t_iterator, v_iterator


def build_acoustic_training_rnn(is_mpi,is_chief, hyper_params, prog_params, train_set, test_set):
    model = AcousticModel(hyper_params["num_layers"], hyper_params["hidden_size"], hyper_params["batch_size"],
                          hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                          hyper_params["input_dim"], hyper_params["batch_normalization"],
                          hyper_params["char_map_length"])

    # Create a Dataset from the train_set and the test_set
    train_dataset = model.build_dataset(train_set, hyper_params["batch_size"], hyper_params["max_input_seq_length"],
                                        hyper_params["max_target_seq_length"], hyper_params["signal_processing"],
                                        hyper_params["char_map"])
    #train_dataset = train_dataset.shuffle(10,reshuffle_each_iteration=True)
    v_iterator = None
    if test_set is []:
        t_iterator = model.add_dataset_input(train_dataset)
    else:
        test_dataset = model.build_dataset(test_set, hyper_params["batch_size"], hyper_params["max_input_seq_length"],
                                           hyper_params["max_target_seq_length"], hyper_params["signal_processing"],
                                           hyper_params["char_map"])
        # Build the input stream from the different datasets
        t_iterator, v_iterator = model.add_datasets_input(train_dataset, test_dataset)

    # Create the model
    #tensorboard_dir
    model.create_training_rnn(is_mpi, hyper_params["dropout_input_keep_prob"], hyper_params["dropout_output_keep_prob"],
                              hyper_params["grad_clip"], hyper_params["learning_rate"],
                              hyper_params["lr_decay_factor"], use_iterator=True)

    if is_chief:
        model.add_tensorboard(prog_params["train_dir"], prog_params["timeline"])
    return model, t_iterator, v_iterator


def load_language_dataset(_hyper_params):
    # TODO : write the code...
    train_set = ["the brown lazy fox", "the red quick fox"]
    test_set = ["the white big horse", "the yellow small cat"]
    return train_set, test_set


def configure_tf_session(xla, timeline):
    # Configure tensorflow's session
    config = tf.ConfigProto()
    jit_level = 0
    if xla:
        # Turns on XLA JIT compilation.
        jit_level = tf.OptimizerOptions.ON_1
    config.graph_options.optimizer_options.global_jit_level = jit_level
    run_metadata = tf.RunMetadata()

    # Add timeline data generation options if needed
    if timeline is True:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    else:
        run_options = None
    return config, run_metadata, run_options


def train_language_rnn(train_set, test_set, hyper_params, prog_params):
    # TODO : write the code...
    config, run_metadata, run_options = configure_tf_session(prog_params["XLA"], prog_params["timeline"])

    with tf.Session(config=config) as sess:
        # Initialize the model
        model, t_iterator, v_iterator = build_language_training_rnn(sess, hyper_params, prog_params,
                                                                    train_set, test_set)

    return


def save_acoustic_rnn(dataset, hyper_params, prog_params):
    save_dataset(dataset,prog_params["train_dir"], hyper_params["max_input_seq_length"], hyper_params["signal_processing"],
                                       hyper_params["char_map"])

def train_acoustic_rnn(train_set, test_set, hyper_params, prog_params):
    config, run_metadata, run_options = configure_tf_session(prog_params["XLA"], prog_params["timeline"])
    is_mpi = False
    is_chief = True
    checkpoint_dir = prog_params["train_dir"]
    hooks = None
    if prog_params["is_mpi"] is True:
        is_mpi = True
        hvd.init()
        hooks = [hvd.BroadcastGlobalVariablesHook(0)]
        if hvd.rank() != 0:
            checkpoint_dir = None
            is_chief = False
    # Initialize the model
    model, t_iterator, v_iterator = build_acoustic_training_rnn(is_mpi,is_chief, hyper_params,
                                                                prog_params, train_set, test_set)


    scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),local_init_op=tf.local_variables_initializer(),
                                 summary_op=None)
    scaffold.global_step = model.global_step
    with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                           is_chief=True,
                                           config=config,
                                           log_step_count_steps=3,
                                           hooks=hooks,scaffold=scaffold,save_summaries_steps=None,save_summaries_secs=None) as sess:
        # Override the learning rate if given on the command line
        if prog_params["learn_rate"] is not None:
            model.set_learning_rate(sess, prog_params["learn_rate"])
        if t_iterator is not None:
            sess.run(model.t_iterator_init)
            model.handle_train = sess.run(t_iterator)
        if v_iterator is not None:
            sess.run(model.v_iterator_init)
            model.handle_v = sess.run(v_iterator)
        previous_mean_error_rates = []
        current_step = epoch = 0
        local_step = 0
        while True:
            # Launch training
            mean_error_rate = 0
            for _ in range(hyper_params["steps_per_checkpoint"]):
                logging.info("Start local step  : %d, global step %d", local_step, current_step)
                _step_mean_loss, step_mean_error_rate, current_step, dataset_empty = \
                    model.run_train_step(sess, hyper_params["mini_batch_size"], hyper_params["rnn_state_reset_ratio"],
                                         run_options=run_options, run_metadata=run_metadata)
                logging.info("Stop local step  : %d, global step %d", local_step, current_step)
                local_step += 1
                mean_error_rate += step_mean_error_rate / hyper_params["steps_per_checkpoint"]

                if dataset_empty is True:
                    epoch += 1
                    logging.info("End of epoch number : %d", epoch)
                    if (prog_params["max_epoch"] is not None) and (epoch > prog_params["max_epoch"]):
                        logging.info("Max number of epochs reached, exiting train step")
                        break
                    else:
                        sess.run(model.t_iterator_init)

            # Run an evaluation session
            if (current_step % hyper_params["steps_per_evaluation"] == 0) and (v_iterator is not None):
                model.run_evaluation(sess, run_options=run_options, run_metadata=run_metadata)
                sess.run(model.v_iterator_init)

            # Decay the learning rate if the model is not improving
            if mean_error_rate <= min(previous_mean_error_rates, default=sys.maxsize):
                previous_mean_error_rates.clear()
            previous_mean_error_rates.append(mean_error_rate)
            if len(previous_mean_error_rates) >= 7:
                sess.run(model.learning_rate_decay_op)
                previous_mean_error_rates.clear()
                logging.info("Model is not improving, decaying the learning rate")
                if model.learning_rate_var.eval(session=sess) < 1e-7:
                    logging.info("Learning rate is too low, exiting")
                    break
                logging.info("Overwriting the checkpoint file with the new learning rate")

            if (prog_params["max_epoch"] is not None) and (epoch > prog_params["max_epoch"]):
                logging.info("Max number of epochs reached, exiting training session")
                break
    return

def distributed_train_acoustic_rnn(train_set, test_set, hyper_params, prog_params):
    config, run_metadata, run_options = configure_tf_session(prog_params["XLA"], prog_params["timeline"])
    cluster,server,distributed_device = cluster_meta(prog_params)
    with tf.device(
            tf.train.replica_device_setter(
                worker_device=distributed_device,
                ps_device='/job:ps',
                cluster=cluster)),tf.Session(server.target,config=config) as sess:
        # Initialize the model
        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            device_filters=['/job:ps', distributed_device])

        is_chief = prog_params["is_chief"]
        model, t_iterator, v_iterator = build_acoustic_training_rnn(False, is_chief, hyper_params,prog_params, train_set, test_set)

        # Override the learning rate if given on the command line
        if prog_params["learn_rate"] is not None:
            model.set_learning_rate(sess, prog_params["learn_rate"])

        checkpoint_dir = prog_params["train_dir"]
        scaffold = tf.train.Scaffold(init_op=tf.global_variables_initializer(),local_init_op=tf.local_variables_initializer(),
                                     summary_op=model.train_summaries_op)
        scaffold.global_step = model.global_step
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,checkpoint_dir=checkpoint_dir,
                                               config=config,
                                               scaffold=scaffold) as sess:
            if t_iterator is not None:
               sess.run(model.t_iterator_init)
               model.handle_train = sess.run(t_iterator)


            if v_iterator is not None:
                sess.run(model.v_iterator_init)
                model.handle_v = sess.run(v_iterator)
            previous_mean_error_rates = []
            current_step = epoch = 0
            local_step  = 0
            while True:
                # Launch training
                mean_error_rate = 0
                for _ in range(hyper_params["steps_per_checkpoint"]):
                    logging.info("Start local step  : %d, global step %d", local_step, current_step)
                    _step_mean_loss, step_mean_error_rate, current_step, dataset_empty = \
                        model.run_train_step(sess, hyper_params["mini_batch_size"], hyper_params["rnn_state_reset_ratio"],
                                             run_options=run_options, run_metadata=run_metadata)
                    logging.info("Stop local step  : %d, global step %d", local_step, current_step)
                    mean_error_rate += step_mean_error_rate / hyper_params["steps_per_checkpoint"]
                    local_step += 1
                    if dataset_empty is True:
                        epoch += 1
                        logging.info("End of epoch number : %d", epoch)
                        if (prog_params["max_epoch"] is not None) and (epoch > prog_params["max_epoch"]):
                            logging.info("Max number of epochs reached, exiting train step")
                            break
                        else:
                            if t_iterator is not None:
                                sess.run(model.t_iterator_init)


                # Run an evaluation session
                if is_chief:
                    if (local_step % hyper_params["steps_per_evaluation"] == 0) and (v_iterator is not None):
                        model.run_evaluation(sess, run_options=run_options, run_metadata=run_metadata)
                        # Shuffle
                        sess.run(model.v_iterator_init)


                # Decay the learning rate if the model is not improving
                if mean_error_rate <= min(previous_mean_error_rates, default=sys.maxsize):
                    previous_mean_error_rates.clear()
                previous_mean_error_rates.append(mean_error_rate)
                if len(previous_mean_error_rates) >= 7:
                    sess.run(model.learning_rate_decay_op)
                    previous_mean_error_rates.clear()
                    logging.info("Model is not improving, decaying the learning rate")
                    if model.learning_rate_var.eval() < 1e-7:
                        logging.info("Learning rate is too low, exiting")
                        break
                if (prog_params["max_epoch"] is not None) and (epoch > prog_params["max_epoch"]):
                    logging.info("Max number of epochs reached, exiting training session")
                    break
    return
def start_ps_server(prog_params):
    cluster,server,distributed_device = cluster_meta(prog_params)
    task = prog_params["task"]
    print('Start parameter server %d' % (task))
    server.join()
    return
def cluster_meta(prog_params):
    ps_spec = prog_params["ps_hosts"].split(",")
    worker_spec = prog_params["worker_hosts"].split(",")
    cluster = tf.train.ClusterSpec({
        'ps': ps_spec,
        'worker': worker_spec})
    task = prog_params["task"]
    server = tf.train.Server(
        cluster, job_name=prog_params["role"], task_index=task)
    device = '/job:%s/task:%d' % (prog_params["role"],task)
    return cluster,server,device

def process_file(audio_processor, hyper_params, file):
    feat_vec, original_feat_vec_length = audio_processor.process_audio_file(file)
    if original_feat_vec_length > hyper_params["max_input_seq_length"]:
        logging.warning("File too long")
        return
    elif original_feat_vec_length < hyper_params["max_input_seq_length"]:
        # Pad the feat_vec with zeros
        pad_length = hyper_params["max_input_seq_length"] - original_feat_vec_length
        padding = np.zeros((pad_length, hyper_params["input_dim"]), dtype=np.float)
        feat_vec = np.concatenate((feat_vec, padding), 0)

    with tf.Session() as sess:
        # create model
        model = AcousticModel(hyper_params["num_layers"], hyper_params["hidden_size"], 1,
                              hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                              hyper_params["input_dim"], hyper_params["batch_normalization"],
                              hyper_params["char_map_length"])
        model.create_forward_rnn()
        model.initialize(sess)
        model.restore(sess, hyper_params["checkpoint_dir"] + "/acoustic/")

        (a, b) = feat_vec.shape
        feat_vec = feat_vec.reshape((a, 1, b))
        predictions = model.process_input(sess, feat_vec, [original_feat_vec_length])
        transcribed_text = [dataprocessor.DataProcessor.get_labels_str(hyper_params["char_map"], prediction)
                            for prediction in predictions]
        print(transcribed_text[0])


def generate_text(hyper_params):
    with tf.Session() as sess:
        # Create model
        model = LanguageModel(hyper_params["num_layers"], hyper_params["hidden_size"], 1, 1,
                              hyper_params["max_target_seq_length"], hyper_params["char_map_length"])
        model.create_forward_rnn()
        model.initialize(sess)
        model.restore(sess, hyper_params["checkpoint_dir"] + "/language/")

        # Start with a letter
        text = "O"

        for _ in range(10):
            print(text, end="")
            # Convert to an one-hot encoded vector
            input_vec = dataprocessor.DataProcessor.get_str_to_one_hot_encoded(hyper_params["char_map"], text,
                                                                               add_eos=False)
            feat_vec = np.array(input_vec)
            (a, b) = feat_vec.shape
            feat_vec = feat_vec.reshape((a, 1, b))
            prediction = model.process_input(sess, feat_vec, [1])
            text = dataprocessor.DataProcessor.get_labels_str(hyper_params["char_map"], prediction[0])
        print(text)
        return


def evaluate(hyper_params):
    if hyper_params["test_dataset_dirs"] is None:
        logging.fatal("Setting test_dataset_dirs in config file is mandatory for evaluation mode")
        return

    # Load the test set data
    data_processor = dataprocessor.DataProcessor(hyper_params["test_dataset_dirs"])
    test_set = data_processor.get_dataset()

    logging.info("Using %d size of test set", len(test_set))

    if len(test_set) == 0:
        logging.fatal("No files in test set during an evaluation mode")
        return

    with tf.Session() as sess:
        # create model
        model = AcousticModel(hyper_params["num_layers"], hyper_params["hidden_size"], hyper_params["batch_size"],
                              hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                              hyper_params["input_dim"], hyper_params["batch_normalization"],
                              hyper_params["char_map_length"])

        model.create_forward_rnn()
        model.initialize(sess)
        model.restore(sess, hyper_params["checkpoint_dir"] + "/acoustic/")

        wer, cer = model.evaluate_full(sess, test_set, hyper_params["max_input_seq_length"],
                                       hyper_params["signal_processing"], hyper_params["char_map"])
        print("Resulting WER : {0:.3g} %".format(wer))
        print("Resulting CER : {0:.3g} %".format(cer))
        return


def record_and_write(audio_processor, hyper_params):
    import pyaudio
    _CHUNK = hyper_params["max_input_seq_length"]
    _SR = 22050
    p = pyaudio.PyAudio()

    with tf.Session() as sess:
        # create model
        model = AcousticModel(hyper_params["num_layers"], hyper_params["hidden_size"], 1,
                              hyper_params["max_input_seq_length"], hyper_params["max_target_seq_length"],
                              hyper_params["input_dim"], hyper_params["batch_normalization"],
                              hyper_params["char_map_length"])

        model.create_forward_rnn()
        model.initialize(sess)
        model.restore(sess, hyper_params["checkpoint_dir"] + "/acoustic/")

        # Create stream of listening
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=_SR, input=True, frames_per_buffer=_CHUNK)
        print("NOW RECORDING...")

        while True:
            data = stream.read(_CHUNK)
            data = np.fromstring(data)
            feat_vec, original_feat_vec_length = audio_processor.process_signal(data, _SR)
            (a, b) = feat_vec.shape
            feat_vec = feat_vec.reshape((a, 1, b))
            predictions = model.process_input(sess, feat_vec, [original_feat_vec_length])
            result = [dataprocessor.DataProcessor.get_labels_str(hyper_params["char_map"], prediction)
                      for prediction in predictions]
            print(result, end="")


def parse_args():
    """
    Parses the command line input.

    """
    _DEFAULT_CONFIG_FILE = 'config.ini'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=_DEFAULT_CONFIG_FILE,
                        help='Path to configuration file with hyper-parameters.')
    parser.add_argument('--tb_name', type=str, default=None,
                        help='Tensorboard path name for the run (allow multiples run with the same output path)')
    parser.add_argument('--max_epoch', type=int, default=None,
                        help='Max epoch to train (no limitation if not provided)')
    parser.add_argument('--learn_rate', type=float, default=None,
                        help='Force learning rate to start from this value (overriding checkpoint value)')
    parser.set_defaults(timeline=False)
    parser.add_argument('--timeline', dest='timeline', action='store_true',
                        help='Generate a json file with the timeline (a tensorboard directory'
                             'must be provided in config file)')
    parser.set_defaults(XLA=False)
    parser.add_argument('--XLA', dest='XLA', action='store_true', help='Activate XLA mode in tensorflow')

    parser.add_argument('--train_dir', type=str, default=None,
                        help='Training direcotry')
    parser.add_argument('--task', type=int, default=0,
                        help='Replica index')
    parser.add_argument('--ps_hosts', type=str, default="",
                        help='Parameter servers')
    parser.add_argument('--worker_hosts', type=str, default="",
                        help='Workers servers')

    parser.add_argument('--test_set', type=str, default=None,
                        help='Train set direcotry')
    parser.add_argument('--train_set', type=str, default=None,
                        help='train set direcotry')

    parser.add_argument('--is_mpi', type=int, default=0,
                        help='train set direcotry')

    group = parser.add_mutually_exclusive_group(required=True)
    group.set_defaults(train_acoustic=False)
    group.set_defaults(dtrain_acoustic=False)
    group.set_defaults(save_acoustic=False)
    group.set_defaults(train_language=False)
    group.set_defaults(start_ps=False)
    group.set_defaults(file=None)
    group.set_defaults(record=False)
    group.set_defaults(evaluate=False)
    group.add_argument('--train_acoustic', dest='train_acoustic', action='store_true',
                       help='Train the acoustic network')
    group.add_argument('--dtrain_acoustic', dest='dtrain_acoustic', action='store_true',
                       help='Distributed Train the acoustic network')
    group.add_argument('--save_acoustic', dest='save_acoustic', action='store_true',
                       help='Save the acoustic data')
    group.add_argument('--train_language', dest='train_language', action='store_true',
                       help='Train the language network')
    group.add_argument('--start_ps', dest='start_ps', action='store_true',
                       help='Start parameter server')
    group.add_argument('--file', type=str, help='Path to a wav file to process')
    group.add_argument('--record', dest='record', action='store_true', help='Record and write result on the fly')
    group.add_argument('--evaluate', dest='evaluate', action='store_true', help='Evaluate WER against the test_set')
    group.add_argument('--generate_text', dest='generate_text', action='store_true', help='Generate text from the '
                                                                                          'language model')

    args = parser.parse_args()
    role = 'worker'
    if args.start_ps is True:
        print("PS Node")
        role = 'ps'
    else:
        print("Worker Node")
        if not tf.gfile.Exists(args.train_dir):
            tf.gfile.MakeDirs(args.train_dir)

    is_mpi = None
    if args.is_mpi >0:
        is_mpi = True

    prog_params = {'config_file': args.config, 'tb_name': args.tb_name, 'max_epoch': args.max_epoch,
                   'learn_rate': args.learn_rate, 'timeline': args.timeline, 'train_acoustic': args.train_acoustic,
                   'dtrain_acoustic': args.dtrain_acoustic,'save_acoustic': args.save_acoustic,
                   'train_language': args.train_language, 'file': args.file, 'record': args.record,
                   'evaluate': args.evaluate, 'generate_text': args.generate_text, 'XLA': args.XLA,
                   'worker_hosts':args.worker_hosts, 'ps_hosts':args.ps_hosts, 'task': args.task, 'train_dir': args.train_dir,
                   'role': role, 'start_ps': args.start_ps, 'is_chief': args.task==0,
                   'train_set':args.train_set,'test_set':args.test_set,'is_mpi':is_mpi}
    return prog_params

def save_dataset(input_set,out_dir, max_input_seq_length,
                 signal_processing, char_map):
    def _read_audio_and_transcode_label(filename_label):
        # Need to convert back to string because tf.py_func changed it to a numpy array
        filename = filename_label[0]
        label = filename_label[1]
        audio_processor = audioprocessor.AudioProcessor(max_input_seq_length, signal_processing)
        audio_decoded, audio_length = audio_processor.process_audio_file(filename)
        label_transcoded = dataprocessor.DataProcessor.get_str_labels(char_map, label)
        return np.array(audio_decoded, dtype=np.float32), np.array(audio_length, dtype=np.int32), \
               np.array(label_transcoded, dtype=np.int32)
    i = 0
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    for item in input_set:
        i += 1
        train_filename = "%s/train-%d" % (out_dir,i)
        logging.info('Writing to feature -' + train_filename)
        a,al,labels = _read_audio_and_transcode_label(item)
        writer = tf.python_io.TFRecordWriter(train_filename)
        feature = {'length': _int64_feature(al),
                   'audio': _bytes_feature(tf.compat.as_bytes(a.tostring())),
                   'label': _bytes_feature(tf.compat.as_bytes(labels.tostring()))}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
        writer.close()

if __name__ == "__main__":
    main()

