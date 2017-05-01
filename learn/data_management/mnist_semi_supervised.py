from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from learn.data_management.interfaces import DataProvider


class SemiSupervisedMNISTProvider(DataProvider):

    def __init__(self, batch_size, supervision=0.05):
        self.batch_size = batch_size
        self.supervision_frequency = int(1 / supervision)

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape((-1, 28, 28, 1)) / 255
        self.x_test = x_test.reshape((-1, 28, 28, 1)) / 255
        self.y_test = to_categorical(y_test)
        self.x_train = x_train[1000:]
        self.y_train = to_categorical(y_train[1000:])
        self.x_val = x_train[:1000]
        self.y_val = to_categorical(y_train[:1000])

        self.datagen = ImageDataGenerator(data_format='channels_last')
        self.datagen.fit(x_train)

        self.n_iter = self.x_train.shape[0] // self.batch_size

        # disable shuffling so that we know that the same samples are used for supervision
        self.iterator = self.datagen.flow(self.x_train, self.y_train,
                                          batch_size=self.batch_size,
                                          shuffle=False)

    def iterate_minibatches(self):
        for i in range(self.n_iter):
            x_train, y_train = next(self.iterator)
            minibatch = (x_train, y_train) if i % self.supervision_frequency == 0 \
                else (x_train, None)

            if i == self.n_iter - 1:
                self.iterator.reset()

            yield minibatch

    def training_data(self):
        return self.x_train, self.y_train

    def validation_data(self):
        return self.x_val, self.y_val

    def test_data(self):
        return self.x_test, self.y_test
