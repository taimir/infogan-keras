"""
Example implementation of InfoGAN
"""

from keras.datasets import mnist

from learn.models.infogan import InfoGAN
from learn.models.model_trainer import ModelTrainer

if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    model = InfoGAN()
    model_trainer = ModelTrainer(model)

    model_trainer.train()
