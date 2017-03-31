#### Important observations

* BatchNormalization is critical in the discriminator (applied separately to the generated and the real samples).
That way the real & fake distributions are put on the same "level" from the very beginning and the G will not collapse on a single mode.

* Pay attention to the learning rates, add gradient clipping - because the loss of G `-log(preds)` is very steep when preds ~= 0, the learning rates should be relatively small and there should be definitely gradient clipping.

* G and D should be balanced - specifically, if the network D is too complex and G does not have enough capacity, D will always discriminate better and G's loss will keep going up and oscilating for a long time. So pick a medium-powered D and an expressive G model. Setting the learning rate of D somewhat lower than G also helps if D is really dominating G.

* Sine-like shapes in the G (and D) loss are actually normal, since we're doing alternating optimizations.

* It's crucial to freeze all parts of the discriminator during the optimization of G; this includes the shared part with the encoder.

* The amount of non-salient noise factors should be sufficiently large.
