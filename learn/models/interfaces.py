import abc
import six


@six.add_metaclass(abc.ABCMeta)
class Model:

    @abc.abstractmethod
    def train_on_minibatch(self, samples, labels):
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class InfoganPrior:

    def __init__(self,
                 meaningful_dists,
                 noise_dists,
                 prior_params,
                 recurrent_dim,
                 ):
        self.meaningful_dists = meaningful_dists
        self.noise_dists = noise_dists
        self.prior_params = prior_params
        self.recurrent_dim = recurrent_dim

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError

    @abc.abstractmethod
    def assemble_prior_params(self):
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class InfoganGenerator:

    def __init__(self,
                 data_shape,
                 meaningful_dists,
                 noise_dists,
                 data_q_dist,
                 network,
                 recurrent_dim):
        self.data_shape = data_shape
        self.meaningful_dists = meaningful_dists
        self.noise_dists = noise_dists
        self.data_q_dist = data_q_dist
        self.network = network
        self.recurrent_dim = recurrent_dim

    @abc.abstractmethod
    def generate(self, prior_samples):
        raise NotImplementedError

    @abc.abstractmethod
    def get_loss(self, disc_gen_output):
        raise NotImplementedError

    @abc.abstractmethod
    def freeze(self):
        raise NotImplementedError

    @abc.abstractmethod
    def unfreeze(self):
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class InfoganDiscriminator:

    def __init__(self,
                 network):
        self.network = network

    @abc.abstractmethod
    def discriminate(self, samples):
        raise NotImplementedError

    @abc.abstractmethod
    def get_loss(self, gen_samples):
        raise NotImplementedError

    @abc.abstractmethod
    def freeze(self):
        raise NotImplementedError

    @abc.abstractmethod
    def unfreeze(self):
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class InfoganEncoder:

    def __init__(self,
                 batch_size,
                 meaningful_dists,
                 supervised_dist,
                 network,
                 recurrent_dim):
        self.batch_size = batch_size
        self.meaningful_dists = meaningful_dists
        self.supervised_dist = supervised_dist
        self.network = network
        self.recurrent_dim = recurrent_dim

    @abc.abstractmethod
    def encode(self, samples):
        raise NotImplementedError

    @abc.abstractmethod
    def get_supervised_loss(self, real_samples):
        raise NotImplementedError

    @abc.abstractmethod
    def get_mi_loss(self, gen_samples):
        raise NotImplementedError

    @abc.abstractmethod
    def get_labels_input(self):
        raise NotImplementedError

    @abc.abstractmethod
    def freeze(self):
        raise NotImplementedError

    @abc.abstractmethod
    def unfreeze(self):
        raise NotImplementedError
