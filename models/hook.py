from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer

class StepCounterHook(session_run_hook.SessionRunHook):
    """Hook that counts steps per second."""

    def __init__(self,
                 scale=1,
                 every_n_steps=100,
                 every_n_secs=None,
                 output_dir=None,
                 summary_writer=None,
                 summary_op=None):

        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError(
                "exactly one of every_n_steps and every_n_secs should be provided.")
        self._timer = SecondOrStepTimer(every_steps=every_n_steps,
                                        every_secs=every_n_secs)

        self._summary_writer = summary_writer
        self._output_dir = output_dir
        self._last_global_step = None
        self._global_step_check_count = 0
        self._scale = scale
        self._summary_op = summary_op

    def begin(self):
        if self._summary_writer is None and self._output_dir:
            self._summary_writer = SummaryWriterCache.get(self._output_dir)
        self._global_step_tensor = training_util._get_or_create_global_step_read()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use StepCounterHook.")
        self._summary_tag = training_util.get_global_step().op.name + "/sec"

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        _ = run_context

        stale_global_step = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step+1):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                    global_step)
                if elapsed_time is not None:
                    steps_per_sec = elapsed_steps * self._scale / elapsed_time
                    if self._summary_writer is not None:
                        aggregated_summary = run_context.session.run(self._summary_op)
                        self._summary_writer.add_summary(aggregated_summary, global_step)
                        summary = Summary(value=[Summary.Value(
                            tag=self._summary_tag, simple_value=steps_per_sec)])
                        self._summary_writer.add_summary(summary, global_step)
                    logging.info("Speech %s: %g", self._summary_tag, steps_per_sec)

        # Check whether the global step has been increased. Here, we do not use the
        # timer.last_triggered_step as the timer might record a different global
        # step value such that the comparison could be unreliable. For simplicity,
        # we just compare the stale_global_step with previously recorded version.
        if stale_global_step == self._last_global_step:
            # Here, we use a counter to count how many times we have observed that the
            # global step has not been increased. For some Optimizers, the global step
            # is not increased each time by design. For example, SyncReplicaOptimizer
            # doesn't increase the global step in worker's main train step.
            self._global_step_check_count += 1
            if self._global_step_check_count % 20 == 0:
                self._global_step_check_count = 0
                logging.warning(
                    "It seems that global step (tf.train.get_global_step) has not "
                    "been increased. Current value (could be stable): %s vs previous "
                    "value: %s. You could increase the global step by passing "
                    "tf.train.get_global_step() to Optimizer.apply_gradients or "
                    "Optimizer.minimize.", stale_global_step, self._last_global_step)
        else:
            # Whenever we observe the increment, reset the counter.
            self._global_step_check_count = 0

        self._last_global_step = stale_global_step