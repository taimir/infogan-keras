#### Important observations

* BatchNormalization is critical in the discriminator (applied separately to the generated and the real samples).
That way the real & fake distributions are put on the same "level" from the very beginning and the G will not collapse on a single mode.

* Pay attention to the learning rates, add gradient clipping - because the loss of G `-log(preds)` is very steep when preds ~= 0, the learning rates should be relatively small and there should be definitely gradient clipping.

* G and D should be balanced - specifically, if the network D is too complex and G does not have enough capacity, D will always discriminate better and G's loss will keep going up and oscilating for a long time. So pick a medium-powered D and an expressive G model. Setting the learning rate of D somewhat lower than G also helps if D is really dominating G.

* Sine-like shapes in the G (and D) loss are actually normal, since we're doing alternating optimizations.

* It's crucial to freeze all parts of the discriminator during the optimization of G; this includes the shared part with the encoder.

* The amount of noise latent variables that should be used is somewhat questionable ... The way they are sampled as well (uniform vs. white noise?)

* The gradients of the E, G and D models should be balanced. Issues might arise with NLL of continuous distributions, e.g. when the `std` of a Gaussian posterior collapses to ~ 0, the encoder's gradients are too steep and the other models have a hard time converging.

* The prior entropy over the salient latents is absolutely optional, as it is a constant with respect to the network parameters.

* The distributions that the salient latents are sampled from do not coincide with the posterior dist. assumptions:
	- we sample from uniform distributions, just to cover all of the available latent space (e.g. Uniform)
	- the posteriors are often uni-modal, however, as we want to make a prediction for the latent encoding (e.g. Gaussian)

* The network size of the convnets in InfoGAN does not have to be too big - I managed to reproduce the 95 proc. classification accuracy from the paper with a smaller network.
