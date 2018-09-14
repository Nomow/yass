"""
The :mod:`yass.explore` module implements functions for data exploration,
this code is not used in any other module
"""


from yass.explore.explorers import SpikeTrainExplorer, RecordingExplorer

__all__ = ['SpikeTrainExplorer', 'RecordingExplorer']

# TODO: check if matplotlib is installed, if not, show error message
