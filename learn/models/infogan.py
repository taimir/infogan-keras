"""
Implementation of the InfoGAN network
"""

import numpy as np
import keras.backend as K
from keras.layers import Input,  Dense, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.objectives import binary_crossentropy

from learn.networks.convnets import GeneratorNet, SharedNet


class InfoGAN(object):
    """
    Puts together different networks to form the InfoGAN network as per:

    "InfoGAN: Interpretable Representation Learning by Information Maximizing Generative Adversarial
    Nets" by Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, Pieter Abbeel
    """

    def __init__(self, batch_size, image_shape, noise_dists,
                 meaningful_dists, image_dist, prior_params
                 ):
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

        d_input = Input(batch_shape=(2 * self.batch_size,) + self.image_shape, name="d_input")
        sampled_latents, prior_param_inputs, prior_param_names, prior_param_dist_names = self._sample_latent_inputs()
        sampled_latents_flat = list(sampled_latents.values())
        merged_samples = Concatenate(axis=-1, name="g_concat_prior_samples")(sampled_latents_flat)

        self.prior_param_inputs = prior_param_inputs
        self.prior_param_names = prior_param_names
        self.prior_param_dist_names = prior_param_dist_names

        # GENERATOR
        # --------------------------------------------------------------------
        gen_net = GeneratorNet(image_shape)
        generation_params = gen_net.apply(inputs=merged_samples)

        generated = Lambda(function=self._sample_image,
                           output_shape=self.image_shape,
                           name="g_x_sampling")(generation_params)

        self.gen_model = Model(inputs=prior_param_inputs, outputs=[generated])
        # NOTE: the loss here does not matter, it won't be used ...
        # the model is just compiled so that we can generate samples from it with .predict()
        self.gen_model.compile(optimizer='adam', loss='binary_crossentropy')

        # DISCRIMINATOR
        # --------------------------------------------------------------------
        # Disable the generator params while we define the discriminator
        # for layer in self.gen_model.layers:
            # layer.trainable = False

        # now append the real images to the generated ones
        # merged = Concatenate(axis=0, name="d_concat_fake_real")([generated, real_input])

        # the encoder shares the discriminator net
        shared_net = SharedNet()
        shared = shared_net.apply(d_input)

        # binary output for the GAN classification
        disc_last_dense = Dense(1, name="d_classif_layer")
        disc_sigmoid = Activation(activation=K.sigmoid, name="d_classif_activ")

        disc_out = disc_last_dense(shared)
        disc_out = disc_sigmoid(disc_out)

        # def disc_loss(targets, preds):
            # # targets are disregarded, as it's clear what they are
            # # the first batch_size many are zeros (generated)
            # # the second batch_size many are ones (real)
            # targets_generated = K.zeros(shape=(self.batch_size, 1))
            # targets_real = K.ones(shape=(self.batch_size, 1))
            # targets = K.concatenate([targets_generated, targets_real], axis=0)

            # return binary_crossentropy(targets, preds)

        # compile the disc. part only at first
        self.disc_model = Model(inputs=d_input,
                                outputs=disc_out,
                                name="disc_model")
        self.disc_model.compile(optimizer='adam', loss=binary_crossentropy)

        # ENCODER:
        # --------------------------------------------------------------------
        # the encoder produces the statistics of the p(c | x) distribution
        # which is a product of multiple factors (meaningful_dists)

        # unfreeze the generator
        # for layer in self.gen_model.layers:
            # layer.trainable = True

        # TODO: decide whether to freeze the discriminator

        # the encoder shares a common trunk with discriminator
        shared_gen = shared_net.apply(generated)
        encoder_net = Dense(128, name="e_dense_1")(shared_gen)
        encoder_net = BatchNormalization(name="e_dense_bn_1")(encoder_net)
        encoder_net = LeakyReLU(name="e_dense_activ_1")(encoder_net)

        # add outputs for the parameters of all assumed meaninful distributions
        posterior_outputs = []
        mi_losses = []
        for dist_name, dist in meaningful_dists.items():
            param_outputs_dict = self._add_dist_outputs(dist_name, dist, encoder_net)
            param_outputs_list = []
            param_names_list = []
            param_outputs_dims = []

            for param_name, (dim, _) in dist.param_info().items():
                param_outputs_list.append(param_outputs_dict[param_name])
                param_outputs_dims.append(dim)
                param_names_list.append(param_name)

            if len(param_outputs_list) > 1:
                merged_params = Concatenate(axis=-1,
                                            name="e_concat_outputs_{}".format(dist_name))(param_outputs_list)
            else:
                merged_params = param_outputs_list[0]

            posterior_outputs.append(merged_params)

            # build the mi_loss
            samples = sampled_latents[dist_name]
            mi_loss = self._build_mi_loss(samples, dist, param_names_list, param_outputs_dims)

            mi_losses.append(mi_loss)

        self.encoder_model = Model(inputs=prior_param_inputs,
                                   outputs=posterior_outputs,
                                   name="enc_model")
        self.encoder_model.compile(optimizer='adam', loss=mi_losses)

        # ENCODER + GENERATOR are trained together
        def gen_loss(targets, preds):
            # again, targets are not needed - ignore them
            # take only the generated samples
            # preds = preds[:self.batch_size]

            # define the standard disc GAN loss
            loss = K.sum(-K.log(preds))

            return loss


        # Make the discriminator not trainable
        disc_last_dense.trainable = False
        disc_sigmoid.trainable = False

        gen_disc_out =  disc_last_dense(shared_gen)
        gen_disc_out = disc_sigmoid(gen_disc_out)

        self.enc_gen_model = Model(inputs=prior_param_inputs,
                                   outputs=[gen_disc_out] + posterior_outputs,
                                   name="enc_gen_model")
        self.enc_gen_model.compile(optimizer='adam', loss=[gen_loss] + mi_losses)

        # DEBUGGING
        self._layer_functions = {}
        self._layer_names = []
        for layer in self.enc_gen_model.layers:
            self._layer_names.append(layer.name)
            self._layer_functions[layer.name] = K.function(inputs=[K.learning_phase()] + self.enc_gen_model.inputs,
                                                           outputs=[layer.get_output_at(0)])

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
            param_input = Input(batch_shape=(self.batch_size, dim),
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
                        name="g_sample_prior_{}".format(dist_name))(merged_params)

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

    def _add_dist_outputs(self, dist_name, dist, layer):
        info = dist.param_info()
        outputs = {}
        for param, (dim, activation) in info.items():
            out = Dense(dim, name="e_dense_{}_{}".format(dist_name, param))(layer)
            out = Activation(activation, name="e_activ_{}_{}".format(dist_name, param))(out)
            outputs[param] = out

        return outputs

    def _build_mi_loss(self, samples, dist, param_names_list, param_outputs_dims):
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

    def train_disc_pass(self, samples_batch):
        # form the targets
        targets_real = np.ones((self.batch_size,), dtype=np.float32)
        targets_generated = np.zeros((self.batch_size,), dtype=np.float32)
        targets = np.concatenate([targets_real, targets_generated], axis=0)

        # form the real-generated batch
        generated = self.generate()
        batch = np.concatenate([samples_batch, generated], axis=0)

        return self.disc_model.train_on_batch([batch],
                                              [targets])

    def train_gen_pass(self):
        dummy_targets = [np.ones((self.batch_size,), dtype=np.float32)] * \
            len(self.enc_gen_model.outputs)
        prior_params = self._assemble_prior_params()
        return self.enc_gen_model.train_on_batch(prior_params,
                                                 dummy_targets)

    def activation(self, index, samples):
        name = self._layer_names[index]
        prior_params = self._assemble_prior_params()
        return name, self._layer_functions[name]([0] + prior_params)

    def generate(self):
        prior_params = self._assemble_prior_params()
        return self.gen_model.predict(prior_params, batch_size=self.batch_size)
