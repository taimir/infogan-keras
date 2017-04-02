"""
Implementation of the InfoGAN network
"""

import numpy as np
import keras.backend as K
from keras.layers import Input,  Dense, Activation
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras.models import Model
from keras.activations import linear
from keras.optimizers import Adam

from learn.networks.convnets import GeneratorNet, SharedNet, EncoderTop, DiscriminatorTop


class InfoGAN(object):
    """
    Puts together different networks to form the InfoGAN network as per:

    "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial
    Nets" by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel
    """

    def __init__(self, batch_size, image_shape, noise_dists,
                 meaningful_dists, image_dist, prior_params, experiment_id="infogan"):
        """__init__

        :param batch_size - number of real samples passed at each iteration
        :param image_shape - triple (n_chan, img_height, img_width), shape of generated images
        :param noise_dists - dict of {'<name>': Distribution, ...}
        :param meaningful_dists - dict of {'<name>': Distribution, ...}
        :param image_dist - Distribution of the image, for sampling after the generator
        :param prior_params - dict of {'<name>': <param_dict>,...}
        """

        self.batch_size = batch_size
        self.image_shape = image_shape
        self.noise_dists = noise_dists
        self.meaningful_dists = meaningful_dists
        self.image_dist = image_dist
        self.prior_params = prior_params
        self.experiment_id = experiment_id

        # Define meaningful dist output layers
        self.dist_output_layers = {}
        for dist_name, dist in self.meaningful_dists.items():
            info = dist.param_info()
            self.dist_output_layers[dist_name] = {}
            for param, (dim, activation) in info.items():
                dense = Dense(dim, name="e_dense_{}_{}".format(dist_name, param))
                act = Activation(activation, name="e_activ_{}_{}".format(dist_name, param))
                self.dist_output_layers[dist_name][param] = [dense, act]

        # GENERATION BRANCH
        # --------------------------------------------------------------------
        sampled_latents, prior_param_inputs, prior_param_names, prior_param_dist_names = \
            self._sample_latent_inputs()
        self.sampled_latents = sampled_latents
        self.prior_param_inputs = prior_param_inputs
        self.prior_param_names = prior_param_names
        self.prior_param_dist_names = prior_param_dist_names

        sampled_latents_flat = list(self.sampled_latents.values())
        merged_samples = Concatenate(axis=-1, name="g_concat_prior_samples")(sampled_latents_flat)

        gen_net = GeneratorNet(image_shape)
        generation_params = gen_net.apply(inputs=merged_samples)

        generated = Lambda(function=self._sample_image,
                           output_shape=self.image_shape,
                           name="g_x_sampling")(generation_params)
        # used later by tensorboard
        self.tensor_generated = generated

        # shared network for the discriminator & encoder
        shared_net = SharedNet()
        disc_top = DiscriminatorTop()
        encoder_top = EncoderTop()

        # GEN DISC & ENCODER BRANCH
        # --------------------------------------------------------------------
        gen_shared_trunk = shared_net.apply(generated)

        # discriminator
        disc_last_gen = disc_top.apply(gen_shared_trunk)
        # this is a hack around keras, to make the layer name unique
        disc_gen_loss_layer = Activation(linear, name="d_gen_loss_output")
        disc_last_gen = disc_gen_loss_layer(disc_last_gen)

        # encoder
        enc_last_gen = encoder_top.apply(gen_shared_trunk)

        c_post_outputs_gen, mi_losses = self._add_enc_outputs_and_losses(enc_last_gen)
        # user later by tensorboard
        self.c_post_outputs_gen = c_post_outputs_gen

        # REAL DISC & ENCODER BRANCH
        # --------------------------------------------------------------------
        self.real_input = Input(shape=self.image_shape, name="d_input")
        real_shared_trunk = shared_net.apply(self.real_input)

        # discriminator
        disc_last_real = disc_top.apply(real_shared_trunk)
        # this is a hack around keras, to make the layer name unique
        disc_real_loss_layer = Activation(linear, name="d_real_loss_output")
        disc_last_real = disc_real_loss_layer(disc_last_real)

        # encoder
        enc_last_real = encoder_top.apply(real_shared_trunk)

        c_post_outputs_real, _ = self._add_enc_outputs_and_losses(enc_last_real, add_losses=False)
        # user later by tensorboard
        self.c_post_outputs_real = c_post_outputs_real

        # GENERATOR MODEL
        # --------------------------------------------------------------------
        self.gen_model = Model(inputs=prior_param_inputs, outputs=[generated])
        # NOTE: the loss is not used, so it is arbitrary
        self.gen_model.compile(optimizer='adam', loss="mean_squared_error")

        # ENCODER MODEL
        # --------------------------------------------------------------------
        self.encoder_model = Model(inputs=self.real_input,
                                   outputs=c_post_outputs_real,
                                   name="enc_model")
        # NOTE: the loss is not used, so it is arbitrary
        self.encoder_model.compile(optimizer='adam', loss="mean_squared_error")

        # DISCRIMINATOR MODEL
        # --------------------------------------------------------------------
        self.disc_model = Model(inputs=[self.real_input], outputs=[disc_last_real])
        # NOTE: the loss is not used, so it is arbitrary
        self.disc_model.compile(optimizer='adam', loss="mean_squared_error")

        # DISCRIMINATOR TRAINING MODEL
        # --------------------------------------------------------------------
        # freeze the generator layers when training the discriminator
        gen_net.freeze()

        # Define the binary cross entropy (as two losses for convinience)
        # divide by two to compensate for the split
        def disc_real_loss(targets, real_preds):
            # NOTE: targets are ignored, cause it's clear those are real samples
            return -K.log(real_preds + K.epsilon()) / 2.0

        def disc_gen_loss(targets, gen_preds):
            # NOTE: targets are ignored, cause it's clear those are real samples
            return -K.log(1 - gen_preds + K.epsilon()) / 2.0

        self.disc_train_model = Model(inputs=[self.real_input] + prior_param_inputs,
                                      outputs=[disc_last_gen, disc_last_real] + c_post_outputs_gen,
                                      name="disc_train_model")
        disc_losses = {disc_gen_loss_layer.name: disc_gen_loss,
                       disc_real_loss_layer.name: disc_real_loss}
        disc_losses = merge_dicts(disc_losses, mi_losses)
        self.disc_train_model.compile(optimizer=Adam(lr=2e-4, beta_1=0.2),
                                      loss=disc_losses)

        # GENERATOR TRAINING MODEL
        # --------------------------------------------------------------------
        # unfreeze the gen model
        gen_net.unfreeze()
        # freeze the shared net, it's part of the discriminator
        shared_net.freeze()
        # Freeze the discriminator model
        disc_top.freeze()
        # freeze the encoder
        encoder_top.freeze()
        for param_layers_dict in self.dist_output_layers.values():
            for param_layers in param_layers_dict.values():
                for layer in param_layers:
                    layer.trainable = False

        def gen_loss(targets, preds):
            # NOTE: targets are ignored, cause it's clear those are generated samples
            return -K.log(preds + K.epsilon())

        gen_losses = {disc_gen_loss_layer.name: gen_loss}
        gen_losses = merge_dicts(gen_losses, mi_losses)

        self.gen_train_model = Model(inputs=prior_param_inputs,
                                     outputs=[disc_last_gen] + c_post_outputs_gen,
                                     name="gen_train_model")
        self.gen_train_model.compile(optimizer=Adam(lr=1e-3, beta_1=0.2),
                                     loss=gen_losses)

        # FOR DEBUGGING
        self.gen_and_predict = K.function(inputs=[K.learning_phase()] + prior_param_inputs,
                                          outputs=[disc_last_gen, generated])
        self.disc_predict = K.function(inputs=[K.learning_phase(), self.real_input],
                                       outputs=[disc_last_real])

    def _sample_latent_inputs(self):
        samples = {}
        all_param_inputs = []
        all_param_names = []
        all_param_dist_names = []
        for name, dist in self.noise_dists.items():
            sample, param_names, param_inputs = self._sample_latent_input(name, dist)
            samples[name] = sample
            all_param_inputs += param_inputs
            all_param_names += param_names
            all_param_dist_names += [name] * len(param_names)

        for name, dist in self.meaningful_dists.items():
            sample, param_names, param_inputs = self._sample_latent_input(name, dist)
            samples[name] = sample
            all_param_inputs += param_inputs
            all_param_names += param_names
            all_param_dist_names += [name] * len(param_names)

        return samples, all_param_inputs, all_param_names, all_param_dist_names

    def _sample_latent_input(self, dist_name, dist):
        param_names = []
        param_inputs = []
        param_dims = []

        for param_name, (dim, _) in dist.param_info().items():
            param_input = Input(shape=(dim, ),
                                name="g_prior_param_{}_{}".format(dist_name, param_name))
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
                                        name="g_concat_prior_params_{}".format(dist_name))(param_inputs)
        else:
            merged_params = param_inputs[0]

        sample = Lambda(function=sampling_fn,
                        name="g_sample_prior_{}".format(dist_name),
                        output_shape=(dist.sample_size(), ))(merged_params)

        return sample, param_names, param_inputs

    def _get_latent_inputs_shape(self):
        sizes = []
        for dist in self.meaningful_dists.values():
            sizes.append(dist.sample_size())

        for dist in self.noise_dists.values():
            sizes.append(dist.sample_size())

        return (sum(sizes),)

    def _sample_image(self, params):
        params_dict = {'p': params}
        sampled_image = self.image_dist.sample(params_dict)
        return sampled_image

    def _add_enc_outputs_and_losses(self, layer, add_losses=True):
        # add outputs for the parameters of all assumed meaninful distributions
        posterior_outputs = []
        mi_losses = {}
        for dist_name, dist in self.meaningful_dists.items():
            param_outputs_dict = self._add_dist_outputs(dist_name, dist, layer)
            param_outputs_list = []
            param_names_list = []
            param_outputs_dims = []

            for param_name, (dim, _) in dist.param_info().items():
                param_outputs_list.append(param_outputs_dict[param_name])
                param_outputs_dims.append(dim)
                param_names_list.append(param_name)

            loss_output_name = "e_loss_output_{}".format(dist_name)
            if len(param_outputs_list) > 1:
                merged_params = Concatenate(axis=-1,
                                            name=loss_output_name)(param_outputs_list)
            else:
                merged_params = param_outputs_list[0]
                merged_params = Activation(activation=linear,
                                           name=loss_output_name)(merged_params)

            posterior_outputs.append(merged_params)

            # build the mi_loss
            if add_losses:
                samples = self.sampled_latents[dist_name]
                mi_loss = self._build_mi_loss(samples, dist_name, dist, param_names_list,
                                              param_outputs_dims)

                mi_losses[loss_output_name] = mi_loss

        return posterior_outputs, mi_losses

    def _add_dist_outputs(self, dist_name, dist, layer):
        outputs = {}
        for param, param_layers in self.dist_output_layers[dist_name].items():
            out = layer
            for param_layer in param_layers:
                out = param_layer(out)
            outputs[param] = out
        return outputs

    def _build_mi_loss(self, samples, dist_name, dist, param_names_list, param_outputs_dims):
        def mutual_info_loss(targets, preds):
            # ignore the targets
            param_dict = {}
            param_index = 0
            for param_name, dim in zip(param_names_list, param_outputs_dims):
                param_dict[param_name] = preds[:, param_index:param_index + dim]
                param_index += dim

            loss = dist.nll(samples, param_dict)
            return loss

        return mutual_info_loss

    def _assemble_prior_params(self):
        params = []
        for dist_name, param_name in zip(self.prior_param_dist_names, self.prior_param_names):
            params.append(self.prior_params[dist_name][param_name])

        return params

    def sanity_check(self):
        """_sanity_check

        Checks that the gen_train_model uses the same discriminator weights
        as in the disc_model.
        """
        prior_params = self._assemble_prior_params()
        gen_score, samples = self.gen_and_predict([0] + prior_params)
        disc_score1 = self.disc_model.predict(samples)
        disc_score2 = self.disc_predict([0, samples])
        # print("Disc: {}".format(disc_score))
        # print("Gen: {}".format(gen_score))
        assert np.all(np.isclose(gen_score, disc_score1, atol=1.e-2))
        assert np.all(np.equal(gen_score, disc_score2))

    def train_disc_pass(self, samples_batch):
        dummy_targets = [np.ones((self.batch_size,), dtype=np.float32)] * \
            len(self.disc_train_model.outputs)
        prior_params = self._assemble_prior_params()
        return self.disc_train_model.train_on_batch([samples_batch] + prior_params,
                                                    dummy_targets)

    def train_gen_pass(self):
        dummy_targets = [np.ones((self.batch_size,), dtype=np.float32)] * \
            len(self.gen_train_model.outputs)
        prior_params = self._assemble_prior_params()
        return self.gen_train_model.train_on_batch(prior_params,
                                                   dummy_targets)

    def generate(self):
        prior_params = self._assemble_prior_params()
        return self.gen_model.predict(prior_params, batch_size=self.batch_size)

    def encode(self, samples):
        return self.encoder_model.predict(samples, batch_size=self.batch_size)


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
