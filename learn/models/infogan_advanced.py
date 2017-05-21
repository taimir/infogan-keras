import numpy as np
import keras.backend as K
from keras.activations import linear
from keras.models import Model as K_Model
from keras.layers import Input, Activation, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.layers.core import Lambda
from keras.layers.merge import Concatenate

from learn.models.interfaces import Model, InfoganPrior, InfoganGenerator, InfoganDiscriminator, \
    InfoganEncoder


class InfoGAN2(Model):
    """
    Refactored version of infogan
    """

    def __init__(self,
                 batch_size,
                 data_shape,
                 prior,
                 generator,
                 discriminator,
                 encoder,
                 recurrent_dim):
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.prior = prior
        self.generator = generator
        self.discriminator = discriminator
        self.encoder = encoder
        self.recurrent_dim = recurrent_dim

        if self.recurrent_dim:
            self.shape_prefix = (self.recurrent_dim, )
        else:
            self.shape_prefix = ()

        # PUTTING IT TOGETHER
        self.sampled_latents, self.prior_param_inputs = self.prior.sample()
        self.generated = self.generator.generate(self.sampled_latents)

        self.real_input = Input(shape=self.shape_prefix + self.data_shape,
                                name="real_data_input")
        self.real_labels = encoder.get_labels_input()

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

        # DISCRIMINATOR TRAINING MODEL
        self.generator.freeze()

        disc_train_inputs = [self.real_input]
        if self.encoder.supervised_dist:
            disc_train_inputs.append(self.real_labels)
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

        gen_losses, G_loss_outputs = self.generator.get_loss(self.disc_gen)
        gen_losses = merge_dicts(gen_losses, mi_losses)
        self.gen_train_model = K_Model(inputs=self.prior_param_inputs,
                                       outputs=G_loss_outputs + E_gen_loss_outputs,
                                       name="gen_train_model")
        self.gen_train_model.compile(optimizer=Adam(lr=1e-3, beta_1=0.2),
                                     loss=gen_losses)

    def _train_disc_pass(self, samples_batch, labels_batch=None):
        dummy_targets = [np.ones((self.batch_size, ) + self.shape_prefix, dtype=np.float32)] * \
            len(self.disc_train_model.outputs)
        inputs = [samples_batch]

        if labels_batch is None and self.encoder.supervised_dist:
            dim = self.meaningful_dists[self.supervised_dist_name].sample_size()
            labels_batch = np.zeros((self.batch_size,) + self.shape_prefix + (dim, ))

        if self.encoder.supervised_dist:
            inputs += [labels_batch]

        prior_params = self.prior.assemble_prior_params()
        return self.disc_train_model.train_on_batch(inputs + prior_params,
                                                    dummy_targets)

    def _train_gen_pass(self):
        dummy_targets = [np.ones((self.batch_size,) + self.shape_prefix, dtype=np.float32)] * \
            len(self.gen_train_model.outputs)
        prior_params = self.prior.assemble_prior_params()
        return self.gen_train_model.train_on_batch(prior_params,
                                                   dummy_targets)

    def train_on_minibatch(self, samples, labels=None):
        disc_losses = self._train_disc_pass(samples, labels)
        gen_losses = self._train_gen_pass()

        loss_logs = {}
        for loss, loss_name in zip(gen_losses, self.gen_train_model.metrics_names):
            loss_logs["g_" + loss_name] = loss

        for loss, loss_name in zip(disc_losses, self.disc_train_model.metrics_names):
            loss_logs["d_" + loss_name] = loss

        return {'losses': loss_logs}


class InfoganPriorImpl(InfoganPrior):

    def __init__(self,
                 meaningful_dists,
                 noise_dists,
                 prior_params,
                 recurrent_dim):

        super(InfoganPriorImpl, self).__init__(meaningful_dists,
                                               noise_dists, prior_params, recurrent_dim)
        if self.recurrent_dim:
            self.shape_prefix = (self.recurrent_dim, )
        else:
            self.shape_prefix = ()

    def sample(self):
        samples = {}
        prior_param_inputs = []
        for name, dist in self.noise_dists.items():
            sample, param_inputs = self._sample_latent(name, dist)
            samples[name] = sample
            prior_param_inputs += param_inputs

        for name, dist in self.meaningful_dists.items():
            sample, param_inputs = self._sample_latent(name, dist)
            samples[name] = sample
            prior_param_inputs += param_inputs

        return samples, prior_param_inputs

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
            for param_name, dim in zip(param_names, param_dims):
                if self.recurrent_dim:
                    param = merged_params[:, :, i:i + dim]
                else:
                    param = merged_params[:, i:i + dim]

                param_dict[param_name] = param
                i += dim

            sample = dist.sample(param_dict)
            return sample

        if len(param_inputs) > 1:
            merged_params = Concatenate(axis=-1,
                                        name="g_params_{}".format(dist_name))(param_inputs)
        else:
            merged_params = param_inputs[0]

        sample = Lambda(function=sampling_fn,
                        name="g_sample_{}".format(dist_name),
                        output_shape=self.shape_prefix + (dist.sample_size(), ))(merged_params)

        return sample, param_inputs

    def assemble_prior_params(self):
        params = []
        for dist_name, dist in self.noise_dists.items():
            for param_name in dist.param_info():
                params.append(self.prior_params[dist_name][param_name])
        for dist_name, dist in self.meaningful_dists.items():
            for param_name in dist.param_info():
                params.append(self.prior_params[dist_name][param_name])
        return params


class InfoganGeneratorImpl(InfoganGenerator):

    def __init__(self,
                 data_param_shape,
                 data_shape,
                 meaningful_dists,
                 noise_dists,
                 data_q_dist,
                 network,
                 recurrent_dim):
        super(InfoganGeneratorImpl, self).__init__(data_param_shape, data_shape,
                                                   meaningful_dists, noise_dists, data_q_dist,
                                                   network, recurrent_dim)
        if self.recurrent_dim:
            self.shape_prefix = (self.recurrent_dim, )
        else:
            self.shape_prefix = ()

    def generate(self, prior_samples):
        """
        generate - applies the generator to a dictionary of samples from the different
        salient and noise distributions to generate a sample from p_G(x)

        :param prior_samples: dict, keys are dist. names, values are sampled keras tensors
        """
        sampled_latents_flat = list(prior_samples.values())
        merged_samples = Concatenate(axis=-1, name="g_concat_prior_samples")(sampled_latents_flat)

        generation_params = self.network.apply(inputs=merged_samples)

        # generated = Lambda(function=self._sample_data,
        # output_shape=self.shape_prefix + self.data_shape,
        # name="g_x_sampling")(generation_params)
        generated = generation_params

        return generated

    def _sample_data(self, params):
        params_dict = {}
        i = 0

        for param_name, (param_dim, param_activ) in self.data_q_dist.param_info().items():
            if self.recurrent_dim:
                param = params[:, :, i:i + param_dim]
            else:
                param = params[:, i:i + param_dim]

            params_dict[param_name] = param_activ(param)

        sampled_data = self.data_q_dist.sample(params_dict)

        if self.recurrent_dim:
            sampled_data = sampled_data[:, :, 0]
        else:
            sampled_data = sampled_data[:, 0]

        return sampled_data

    def get_loss(self, disc_gen_output):
        # add a dummy activation layer, just to be able to name it properly
        loss_layer_name = "G_gen_loss"
        gen_output = Activation(activation=linear, name=loss_layer_name)(disc_gen_output)

        def gen_loss(targets, preds):
            # NOTE: targets are ignored, cause it's clear those are generated samples
            return -K.log(preds + K.epsilon())

        return {loss_layer_name: gen_loss}, [gen_output]

    def freeze(self):
        self.network.freeze()

    def unfreeze(self):
        self.network.unfreeze()


class InfoganDiscriminatorImpl(InfoganDiscriminator):

    def __init__(self,
                 network):
        super(InfoganDiscriminatorImpl, self).__init__(network)

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

        return {loss_real_name: disc_real_loss, loss_gen_name: disc_gen_loss}, \
            [real_output, gen_output]

    def freeze(self):
        self.network.freeze()

    def unfreeze(self):
        self.network.unfreeze()


class InfoganEncoderImpl(InfoganEncoder):

    def __init__(self,
                 batch_size,
                 meaningful_dists,
                 supervised_dist,
                 network,
                 recurrent_dim):

        super(InfoganEncoderImpl, self).__init__(batch_size,
                                                 meaningful_dists, supervised_dist,
                                                 network, recurrent_dim)

        if self.recurrent_dim:
            self.shape_prefix = (self.recurrent_dim, )
        else:
            self.shape_prefix = ()

        # Define meaningful dist output layers
        self.dist_output_layers = {}
        for dist_name, dist in self.meaningful_dists.items():
            info = dist.param_info()
            self.dist_output_layers[dist_name] = {}
            for param, (dim, activation) in info.items():
                preact = Dense(dim, name="e_dense_{}_{}".format(dist_name, param))
                if self.recurrent_dim:
                    preact = TimeDistributed(preact, name="e_time_{}_{}".format(dist_name, param))
                act = Activation(activation, name="e_activ_{}_{}".format(dist_name, param))
                self.dist_output_layers[dist_name][param] = [preact, act]

        # define an ordering of params for each dist
        self.orderings = {}
        for dist_name, dist in self.meaningful_dists.items():
            self.orderings[dist_name] = list()
            for param_name, (dim, _) in dist.param_info().items():
                self.orderings[dist_name].append((param_name, dim))

    def encode(self, samples):
        enc_preact = self.network.apply(samples)
        encodings = self._add_enc_outputs(enc_preact)
        return encodings

    def _add_enc_outputs(self, enc_preact):
        posterior_outputs = {}
        for dist_name, dist in self.meaningful_dists.items():
            param_outputs_dict = self._make_enc_outputs(dist_name, dist, enc_preact)
            posterior_outputs[dist_name] = param_outputs_dict

        return posterior_outputs

    def _make_enc_outputs(self, dist_name, dist, layer):
        outputs = {}
        for param, param_layers in self.dist_output_layers[dist_name].items():
            out = layer
            for param_layer in param_layers:
                out = param_layer(out)
            outputs[param] = out
        return outputs

    def _make_loss_output(self, dist_name, param_outputs_dict):
        param_outputs_list = []
        for param_name, _ in self.orderings[dist_name]:
            param_outputs_list.append(param_outputs_dict[param_name])

        if len(param_outputs_list) > 1:
            merged_params = Concatenate(axis=-1)(param_outputs_list)
        else:
            merged_params = param_outputs_list[0]

        return merged_params

    def get_mi_loss(self, gen_samples, gen_encodings):
        loss_outputs = []
        mi_losses = {}
        for dist_name, dist in self.meaningful_dists.items():
            param_outputs_dict = gen_encodings[dist_name]

            loss_output_name = "E_mi_loss_{}".format(dist_name)
            loss_output = self._make_loss_output(dist_name, param_outputs_dict)
            loss_output = Activation(activation=linear,
                                     name=loss_output_name)(loss_output)
            loss_outputs.append(loss_output)

            mi_loss = self._build_loss(gen_samples[dist_name], dist, self.orderings[dist_name])
            mi_losses[loss_output_name] = mi_loss

        return mi_losses, loss_outputs

    def _build_loss(self, samples, dist, param_infos):
        def enc_loss(dummy, param_outputs):
            param_dict = {}
            param_index = 0

            for param_name, dim in param_infos:
                if self.recurrent_dim:
                    param = param_outputs[:, :, param_index:param_index + dim]
                else:
                    param = param_outputs[:, param_index:param_index + dim]

                param_dict[param_name] = param
                param_index += dim

            loss = dist.nll(samples, param_dict)
            return loss

        return enc_loss

    def get_supervised_loss(self, real_labels, real_encodings):
        if not self.supervised_dist:
            return {}, []

        dist = self.meaningful_dists[self.supervised_dist]
        param_outputs_dict = real_encodings[self.supervised_dist]
        loss = self._build_loss(real_labels, dist,
                                self.orderings[self.supervised_dist])

        # since some real instances might not have a label, I assume that
        # this is indicated by all labels in the batch being set to 0 everywhere
        # (which is never the case for discrete labels, and almost impossible for
        # continuous labels)
        def wrapped_loss(targets, preds):
            labels_missing = K.all(K.equal(self.real_labels,
                                           K.zeros_like(self.real_labels)))
            return K.switch(labels_missing,
                            K.zeros((self.batch_size, ) + self.shape_prefix, 1), loss(targets, preds))

        loss_output_name = "E_supervised_loss_{}".format(self.supervised_dist)
        loss_output = self._make_loss_output(self.supervised_dist, param_outputs_dict)
        loss_output = Activation(activation=linear,
                                 name=loss_output_name)(loss_output)

        return {loss_output_name: wrapped_loss}, [loss_output]

    def get_labels_input(self):
        if not self.supervised_dist:
            return None
        dim = self.meaningful_dists[self.supervised_dist].sample_size()
        return Input(shape=self.shape_prefix + (dim, ), name="labels_input")

    def freeze(self):
        for param_layers_dict in self.dist_output_layers.values():
            for param_layers in param_layers_dict.values():
                for layer in param_layers:
                    layer.trainable = False

        self.network.freeze()

    def unfreeze(self):
        for param_layers_dict in self.dist_output_layers.values():
            for param_layers in param_layers_dict.values():
                for layer in param_layers:
                    layer.trainable = True

        self.network.unfreeze()


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
