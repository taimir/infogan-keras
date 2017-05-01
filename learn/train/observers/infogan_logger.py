from learn.train.observers.interfaces import TrainingObserver


class InfoganLogger(TrainingObserver):

    def __init__(self, model, epoch_frequency):
        super(InfoganLogger, self).__init__(model, epoch_frequency, None, None)

    def _update(self, epoch, epoch_results):
        print("epoch {}".format(epoch))
        # TODO: print some logs

    def finish(self):
        print("Training finished")
