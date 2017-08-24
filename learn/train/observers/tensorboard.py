import tensorflow as tf

from learn.train.observers.interfaces import TrainingObserver


class TensorBoardLossObserver(TrainingObserver):

    def __init__(self, model, tb_writer, frequency=1):
        self.tb_writer = tb_writer
        super(TensorBoardLossObserver, self).__init__(model, frequency, None, None)

    def _update(self, iteration, iteration_results):
        loss_logs = iteration_results['losses']
        for name, value in loss_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.tb_writer.add_summary(summary, iteration)

        self.tb_writer.flush()

    def finish(self):
        pass
