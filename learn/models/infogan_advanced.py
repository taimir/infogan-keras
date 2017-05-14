import abc
import six

from keras.models import Model as K_Model
from keras.layers import Input
from keras.optimizers import Adam
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate

from learn.models.interfaces import Model


@six.add_metaclass(abc.ABCMeta)
class InfoganPrior:

    def __init__(self,
                 shape,
                 meaningful_dists,
                 noise_dists):
        self.shape = shape
        self.meaningful_dists = meaningful_dists
        self.noise_dists = noise_dists

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class InfoganGenerator:

    def __init__(self,
                 meaningful_dists,
                 noise_dists,
                 network):
        self.meaningful_dists = meaningful_dists
        self.noise_dists = noise_dists
        self.network = network

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
                 meaningful_dists,
                 supervised_dist,
                 network):
        self.meaningful_dists = meaningful_dists
        self.supervised_dist = supervised_dist
        self.network = network

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
    def freeze(self):
        raise NotImplementedError

    @abc.abstractmethod
    def unfreeze(self):
        raise NotImplementedError


class InfoganPriorImpl(InfoganPrior):

    def __init__(self,
                 shape,
                 meaningful_dists,
                 noise_dists):
        super(InfoganPrior, self).__init__(shape, meaningful_dists, noise_dists)

    def sample(self):
        samples = {}
        param_inputs = []
        # all_param_names = []
        # all_param_dist_names = []
        for name, dist in self.noise_dists.items():
            sample, param_inputs = self._sample_latent_input(name, dist)
            samples[name] = sample
            param_inputs += param_inputs
            # all_param_names += param_names
            # all_param_dist_names += [name] * len(param_names)

        for name, dist in self.meaningful_dists.items():
            sample, param_inputs = self._sample_latent_input(name, dist)
            samples[name] = sample
            param_inputs += param_inputs
            # all_param_names += param_names
            # all_param_dist_names += [name] * len(param_names)

        return samples, param_inputs

    def _sample_latent_input(self, dist_name, dist):
        param_names = []
        param_inputs = []
        param_dims = []

        for param_name, (dim, _) in dist.param_info().items():
            param_input = Input(shape=self.shape[:-1] + (dim, ),
                                name="g_input_{}_{}".format(dist_name, param_name))
            param_inputs.append(param_input)
            param_dims.append(dim)
            param_names.append(param_name)

        def sampling_fn(merged_params):
            param_dict = {}
            i = 0
            for j, dim in enumerate(param_dims):
                param = merged_params[:, i:i + dim]
                param_dict[param_names[j]] = param
                i += dim

            return dist.sample(param_dict)

        if len(param_inputs) > 1:
            merged_params = Concatenate(axis=-1,
                                        name="g_params_{}".format(dist_name))(param_inputs)
        else:
            merged_params = param_inputs[0]

        sample = Lambda(function=sampling_fn,
                        name="g_sample_{}".format(dist_name),
                        output_shape=self.shape[:-1] + (dist.sample_size(), ))(merged_params)

        return sample, param_inputs


class InfoGAN2(Model):
    """
    Refactored version of infogan
    """

    def __init__(self,
                 prior,
                 generator,
                 discriminator,
                 encoder):
        self.prior = prior
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder

        # PUTTING IT TOGETHER
        self.sampled_latents, self.prior_param_inputs = self.prior.sample()
        self.generated = self.generator.generate(self.sampled_latents)

        self.real_input = self.discriminator.real_input
        self.real_labels = self.encoder.real_labels

        self.gen_encodings, gen_enc_loss_outputs = self.encoder.encode(self.generated)
        mi_losses = self.encoder.get_mi_loss(self.generated)

        self.real_encodings, real_enc_loss_outputs = self.encoder.encode(self.real_input)
        # this can be empty if the encoder is not supervised
        supervised_losses = self.encoder.get_supervised_loss(self.real_input)

        enc_losses = merge_dicts(mi_losses, supervised_losses)

        disc_gen = self.discriminator.discriminate(self.generated)
        disc_real = self.discriminator.discriminate(self.real_input)
        disc_losses = self.discriminator.get_loss(self.generated)

        gen_losses = self.generator.get_loss(disc_gen)

        # DISCRIMINATOR TRAINING MODEL
        # --------------------------------------------------------------------
        self.generator.freeze()

        disc_train_inputs = [self.real_input,
                             self.real_labels] if self.encoder.supervised else [self.real_input]
        disc_train_inputs += self.prior_param_inputs

        disc_train_outputs = [disc_gen, disc_real] + gen_enc_loss_outputs + real_enc_loss_outputs

        self.disc_train_model = K_Model(inputs=disc_train_inputs,
                                        outputs=disc_train_outputs,
                                        name="disc_train_model")

        disc_train_losses = merge_dicts(disc_losses, enc_losses)
        self.disc_train_model.compile(optimizer=Adam(lr=2e-4, beta_1=0.2),
                                      loss=disc_train_losses)

        # GENERATOR TRAINING MODEL
        # --------------------------------------------------------------------
        self.generator.unfreeze()
        self.discriminator.freeze()
        self.encoder.freeze()

        gen_losses = merge_dicts(gen_losses, mi_losses)
        self.gen_train_model = K_Model(inputs=self.prior_param_inputs,
                                       outputs=[disc_gen] + gen_enc_loss_outputs,
                                       name="gen_train_model")
        self.gen_train_model.compile(optimizer=Adam(lr=1e-3, beta_1=0.2),
                                     loss=gen_losses)


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
