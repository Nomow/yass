"""
Sample data for testing
"""
import shutil
from pathlib import Path

import numpy as np

from yass.batch import RecordingsReader
from yass import geometry as yass_geometry
from yass import pipeline


path_to_data_storage = Path('~', 'data').expanduser()

path_to_neuro_folder = Path(path_to_data_storage, 'neuro')
path_to_neuro_data = Path(path_to_neuro_folder, 'rawDataSample.bin')
path_to_neuro_geom = Path(path_to_neuro_folder, 'channel_positions.npy')


path_to_retina_folder = Path(path_to_data_storage, 'retina')
path_to_retina_config = Path(path_to_retina_folder, 'config.yaml')
path_to_retina_data = Path(path_to_retina_folder, 'ej49_data1_set1.bin')
path_to_retina_geom = Path(path_to_retina_folder, 'ej49_geometry1.txt')


path_to_output_folder = Path(path_to_data_storage, 'yass-testing-data')
path_to_output_folder_neuro = Path(path_to_output_folder, 'neuropixel')
path_to_output_folder_retina = Path(path_to_output_folder, 'retina')

if path_to_output_folder.exists():
    print('Output Folder exists, removing contents....')
    shutil.rmtree(str(path_to_output_folder))


path_to_output_folder.mkdir()
# path_to_output_folder_neuro.mkdir()
path_to_output_folder_retina.mkdir()


# Dataset 1: retina
seconds = 1
channels_total = 49
channels = 49

CONFIG = {
    'data': {
        'root_folder': str(path_to_output_folder_retina),
        'recordings': 'data.bin',
        'geometry': 'geometry.npy'
    },

    'resources': {
        'max_memory': '200MB',
        'processes': 1
    },

    'recordings': {
        'dtype': 'int16',
        'sampling_rate': 20000,
        'n_channels': channels,
        'spatial_radius': 70,
        'spike_size_ms': 1.5,
        'order': 'samples',
    },

    'preprocess': {
        'apply_filter': True,
        'dtype': 'float32'

    },

    'detect': {
        'method': 'threshold',
        'temporal_features': 3

    }
}


observations = CONFIG['recordings']['sampling_rate'] * seconds


# retina, 49ch
retina = RecordingsReader(str(path_to_retina_data),
                          dtype=CONFIG['recordings']['dtype'],
                          n_channels=channels_total,
                          data_order='samples',
                          loader='array').data

sample_data = retina[:observations, :channels]
sample_data.tofile(str(Path(path_to_output_folder_retina, 'data.bin')))

geometry = yass_geometry.parse(str(path_to_retina_geom), channels_total)
sample_geometry = geometry[:channels, :channels]
np.save(str(Path(path_to_output_folder_retina, 'geometry.npy')),
        sample_geometry)

pipeline.run(CONFIG, clean=True,
             output_dir='sample_pipeline_output',
             set_zero_seed=True)


# # Dataset 2: neuropixel
# seconds = 5
# channels = 10
# dtype = 'int16'
# data_order = 'samples'
# sampling_frequency = 30000
# observations = sampling_frequency * seconds

# data = np.fromfile(str(Path(path_to_neuro_data)), dtype=dtype)
# data = data.reshape((385, 1800000)).T
# sample_data = data[:observations, :channels]

# geometry = np.load(str(Path(path_to_neuro_geom)))
# sample_geometry = geometry[:channels, :]

# # save data and geometry
# sample_data.tofile(str(Path(path_to_output_folder_neuro, 'data.bin')))
# np.save(str(Path(path_to_output_folder_neuro, 'geometry.npy')),
#         sample_geometry)


# butterworth(str(Path(path_to_output_folder_neuro, 'data.bin')),
#             dtype=dtype,
#             n_channels=channels, data_order=data_order,
#             order=3, low_frequency=300, high_factor=0.1,
#             sampling_frequency=sampling_frequency, max_memory='1GB',
#             output_path=str(path_to_output_folder_neuro),
#             standarize=True,
#             output_filename='standarized.bin',
#             if_file_exists='overwrite',
#             output_dtype='float32')


# # FIXME: automatically generate config files based on this? also generate
# # a file for conftest data_info()


# # import matplotlib.pyplot as plt
# # standarized = RecordingsReader('tests/data/standarized.bin', loader='array').data
# # plt.plot(standarized[1000:1200])
# # plt.show()
