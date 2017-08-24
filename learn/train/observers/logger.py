from learn.train.observers.interfaces import TrainingObserver


class Logger(TrainingObserver):

    def __init__(self, model, frequency):
        super(Logger, self).__init__(model, frequency, None, None)

    def _update(self, iteration, iteration_results):
        print("iteration {}: {}".format(iteration, iteration_results))

    def finish(self):
        print("Training finished")
