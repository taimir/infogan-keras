import abc
import six
from six.moves import reduce
from operator import mul

import keras.backend as K
from keras.activations import linear
from keras.models import Model as K_Model
from keras.layers import Input, Reshape, Activation
from keras.optimizers import Adam
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate

from learn.models.interfaces import Model


@six.add_metaclass(abc.ABCMeta)
class InfoganPrior:

    def __init__(self,
                 shape_prefix,
                 meaningful_dists,
                 noise_dists):
        self.shape_prefix = shape_prefix
        self.meaningful_dists = meaningful_dists
        self.noise_dists = noise_dists

    @abc.abstractmethod
    def sample(self):
        raise NotImplementedError


@six.add_metaclass(abc.ABCMeta)
class InfoganGenerator:

    def __init__(self,
                 shape_prefix,
                 data_shape,
                 meaningful_dists,
                 noise_dists,
                 data_q_dist,
                 network):
        self.shape_prefix = shape_prefix
        self.data_shape = data_shape
        self.meaningful_dists = meaningful_dists
        self.noise_dists = noise_dists
        self.data_q_dist = data_q_dist
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
                 shape_prefix,
                 meaningful_dists,
                 noise_dists):
        super(InfoganPrior, self).__init__(shape_prefix, meaningful_dists, noise_dists)

    def sample(self):
        samples = {}
        param_inputs = []
        # all_param_names = []
        # all_param_dist_names = []
        for name, dist in self.noise_dists.items():
            sample, param_inputs = self._sample_latent(name, dist)
            samples[name] = sample
            param_inputs += param_inputs
            # all_param_names += param_names
            # all_param_dist_names += [name] * len(param_names)

        for name, dist in self.meaningful_dists.items():
            sample, param_inputs = self._sample_latent(name, dist)
            samples[name] = sample
            param_inputs += param_inputs
            # all_param_names += param_names
            # all_param_dist_names += [name] * len(param_names)

        return samples, param_inputs

    def _sample_latent(self, dist_name, dist):
        param_names = []
        param_inputs = []
        param_dims = []

        for param_name, (dim, _) in dist.param_info().items():
            param_input = Input(shape=self.shape_prefix + (dim, ),
                                name="g_input_{}_{}".format(dist_name, param_name))
            param_inputs.append(param_input)
            param_dims.append(dim)
            param_names.append(param_name)

        def sampling_fn(merged_params):
            param_dict = {}
            i = 0
            for j, dim in enumerate(param_dims):
                param = merged_params[:, i:i + dim]
                flattened_param = K.reshape(param, (-1, dim))
                param_dict[param_names[j]] = flattened_param
                i += dim

            flattened_sample = dist.sample(param_dict)
            return K.reshape(flattened_sample, self.shape_prefix + (dist.sample_size(), ))

        if len(param_inputs) > 1:
            merged_params = Concatenate(axis=-1,
                                        name="g_params_{}".format(dist_name))(param_inputs)
        else:
            merged_params = param_inputs[0]

        sample = Lambda(function=sampling_fn,
                        name="g_sample_{}".format(dist_name),
                        output_shape=self.shape_prefix + (dist.sample_size(), ))(merged_params)

        return sample, param_inputs


class InfoganGeneratorImpl(InfoganGenerator):

    def __init__(self,
                 shape_prefix,
                 data_shape,
                 meaningful_dists,
                 noise_dists,
                 data_q_dist,
                 network):
        super(InfoganGenerator, self).__init__(shape_prefix, data_shape, meaningful_dists,
                                               noise_dists, data_q_dist,
                                               network)

    def generate(self, prior_samples):
        """
        generate - applies the generator to a dictionary of samples from the different
        salient and noise distributions to generate a sample from p_G(x)

        :param prior_samples: dict, keys are dist. names, values are sampled keras tensors
        """
        sampled_latents_flat = list(self.prior_samples.values())
        merged_samples = Concatenate(axis=-1, name="g_concat_prior_samples")(sampled_latents_flat)

        generation_params = self.network.apply(inputs=merged_samples)

        generation_params = Reshape(shape=(-1, reduce(mul, self.data_shape, 1)))(generation_params)

        generated = Lambda(function=self._sample_data,
                           output_shape=self.shape_prefix + self.data_shape,
                           name="g_x_sampling")(generation_params)

        return generated

    def _sample_data(self, params):
        params_dict = {}
        i = 0

        for param_name, (param_dim, param_activ) in self.data_q_dist.param_info().items():
            param = params[:, i:i + param_dim]
            params_dict[param_name] = param_activ(param)

        sampled_image = self.image_dist.sample(params_dict)
        sampled_image = K.reshape(sampled_image, self.shape_prefix + self.data_shape)
        return sampled_image

    @abc.abstractmethod
    def get_loss(self, disc_gen_output):
        # add a dummy activation layer, just to be able to name it properly
        loss_layer_name = "G_gen_loss"
        gen_output = Activation(activation=linear, name=loss_layer_name)(disc_gen_output)

        def gen_loss(targets, preds):
            # NOTE: targets are ignored, cause it's clear those are generated samples
            return -K.log(preds + K.epsilon())

        {loss_layer_name: gen_loss}, [gen_output]

    @abc.abstractmethod
    def freeze(self):
        self.network.freeze()

    @abc.abstractmethod
    def unfreeze(self):
        self.network.unfreeze()


class InfoganDiscriminatorImpl(InfoganDiscriminator):

    def __init__(self,
                 network):
        super(InfoganDiscriminator, self).__init__(network)

    def discriminate(self, samples):
        preactiv = self.network.apply(samples)
        output = Activation(activation=K.sigmoid)(preactiv)
        return output

    def get_loss(self, disc_real_output, disc_gen_output):
        loss_real_name = "D_real_loss"
        loss_gen_name = "D_gen_loss"

        real_output = Activation(activation=linear, name=loss_real_name)(disc_real_output)
        gen_output = Activation(activation=linear, name=loss_gen_name)(disc_gen_output)

        def disc_real_loss(targets, real_preds):
            # NOTE: targets are ignored, cause it's clear those are real samples
            return -K.log(real_preds + K.epsilon()) / 2.0

        def disc_gen_loss(targets, gen_preds):
            # NOTE: targets are ignored, cause it's clear those are real samples
            return -K.log(1 - gen_preds + K.epsilon()) / 2.0

        {loss_real_name: disc_real_loss,
         loss_gen_name: disc_gen_loss}, [real_output, gen_output]

    def freeze(self):
        self.network.freeze()

    def unfreeze(self):
        self.network.unfreeze()


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

        self.gen_encodings = self.encoder.encode(self.generated)
        mi_losses, E_gen_loss_outputs = self.encoder.get_mi_loss(self.sampled_latents,
                                                                 self.gen_encodings)

        self.real_encodings = self.encoder.encode(self.real_input)
        # this can be empty if the encoder is not supervised
        sup_losses, E_real_loss_outputs = self.encoder.get_supervised_loss(self.real_labels,
                                                                           self.real_encodings)

        enc_losses = merge_dicts(mi_losses, sup_losses)

        self.disc_gen = self.discriminator.discriminate(self.generated)
        self.disc_real = self.discriminator.discriminate(self.real_input)
        disc_losses, D_loss_outputs = self.discriminator.get_loss(
            self.disc_real, self.disc_gen)

        gen_losses, G_loss_outputs = self.generator.get_loss(self.disc_gen)

        # DISCRIMINATOR TRAINING MODEL
        self.generator.freeze()

        disc_train_inputs = [self.real_input,
                             self.real_labels] if self.encoder.supervised else [self.real_input]
        disc_train_inputs += self.prior_param_inputs

        disc_train_outputs = D_loss_outputs + E_real_loss_outputs + E_gen_loss_outputs

        self.disc_train_model = K_Model(inputs=disc_train_inputs,
                                        outputs=disc_train_outputs,
                                        name="disc_train_model")

        disc_train_losses = merge_dicts(disc_losses, enc_losses)
        self.disc_train_model.compile(optimizer=Adam(lr=2e-4, beta_1=0.2),
                                      loss=disc_train_losses)

        # GENERATOR TRAINING MODEL
        self.generator.unfreeze()
        self.discriminator.freeze()
        self.encoder.freeze()

        gen_losses = merge_dicts(gen_losses, mi_losses)
        self.gen_train_model = K_Model(inputs=self.prior_param_inputs,
                                       outputs=G_loss_outputs + E_gen_loss_outputs,
                                       name="gen_train_model")
        self.gen_train_model.compile(optimizer=Adam(lr=1e-3, beta_1=0.2),
                                     loss=gen_losses)


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
