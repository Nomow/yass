"""
The :mod:`yass.spikes.augment` module implements functions to generate new
spikes from templates
"""
from yass.spikes.augment.noise import noise_cov
from yass.spikes.augment.util import (make_from_templates,
                                      make_collided, make_noise,
                                      make_spatially_misaligned,
                                      make_temporally_misaligned)

__all__ = ['noise_cov', 'make_from_templates',
           'make_collided', 'make_noise', 'make_spatially_misaligned',
           'make_temporally_misaligned']
