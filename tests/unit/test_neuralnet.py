"""
neuralnetwork module tests
"""
import os.path as path

import numpy as np
import tensorflow as tf
import yaml

import yass
from yass.batch import BatchProcessor, RecordingsReader
from yass import neuralnetwork
from yass.neuralnetwork import NeuralNetDetector, KerasModel, AutoEncoder
from yass.neuralnetwork.apply import post_processing
from yass.geometry import make_channel_index, n_steps_neigh_channels
from yass.augment import make
from yass.augment.noise import noise_cov


def test_can_train_detector(path_to_config,
                            path_to_sample_pipeline_folder,
                            make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)
    CONFIG = yass.read_config()

    spike_train = np.load(path.join(path_to_sample_pipeline_folder,
                                    'spike_train_post_deconv_post_merge.npy'))
    chosen_templates = np.unique(spike_train[:, 1])
    min_amplitude = 4
    max_amplitude = 60

    templates = make.load_templates(path_to_sample_pipeline_folder,
                                    spike_train, CONFIG, chosen_templates)

    path_to_standarized = path.join(path_to_sample_pipeline_folder,
                                    'preprocess', 'standarized.bin')

    rec = RecordingsReader(path_to_standarized, loader='array').data

    (spatial_sig,
     temporal_sig) = noise_cov(rec, templates.shape[1], templates.shape[1])

    X, y = make.training_data_detect(templates=templates,
                                     minimum_amplitude=min_amplitude,
                                     maximum_amplitude=max_amplitude,
                                     n_clean_per_template=20,
                                     n_collided_per_spike=20,
                                     n_temporally_misaligned_per_spike=0.25,
                                     n_noise=20,
                                     n_spatially_misaliged_per_spike=0,
                                     spatial_SIG=spatial_sig,
                                     temporal_SIG=temporal_sig)

    _, waveform_length, n_neighbors = X.shape

    path_to_model = path.join(make_tmp_folder, 'detect-net.ckpt')

    detector = NeuralNetDetector(path_to_model, [8, 4],
                                 waveform_length, n_neighbors,
                                 threshold=0.5,
                                 channel_index=CONFIG.channel_index,
                                 n_iter=10)

    detector.fit(X, y)


def test_can_reload_detector(path_to_config,
                             path_to_sample_pipeline_folder,
                             make_tmp_folder):
    yass.set_config(path_to_config, make_tmp_folder)
    CONFIG = yass.read_config()

    spike_train = np.load(path.join(path_to_sample_pipeline_folder,
                                    'spike_train_post_deconv_post_merge.npy'))
    chosen_templates = np.unique(spike_train[:, 1])
    min_amplitude = 4
    max_amplitude = 60

    templates = make.load_templates(path_to_sample_pipeline_folder,
                                    spike_train, CONFIG, chosen_templates)

    path_to_standarized = path.join(path_to_sample_pipeline_folder,
                                    'preprocess', 'standarized.bin')

    rec = RecordingsReader(path_to_standarized, loader='array').data

    (spatial_sig,
     temporal_sig) = noise_cov(rec, templates.shape[1], templates.shape[1])

    X, y = make.training_data_detect(templates=templates,
                                     minimum_amplitude=min_amplitude,
                                     maximum_amplitude=max_amplitude,
                                     n_clean_per_template=20,
                                     n_collided_per_spike=20,
                                     n_temporally_misaligned_per_spike=0.25,
                                     n_noise=20,
                                     n_spatially_misaliged_per_spike=0,
                                     spatial_SIG=spatial_sig,
                                     temporal_SIG=temporal_sig)

    _, waveform_length, n_neighbors = X.shape

    path_to_model = path.join(make_tmp_folder, 'detect-net.ckpt')

    detector = NeuralNetDetector(path_to_model, [8, 4],
                                 waveform_length, n_neighbors,
                                 threshold=0.5,
                                 channel_index=CONFIG.channel_index,
                                 n_iter=10)

    detector.fit(X, y)

    NeuralNetDetector.load(path_to_model, threshold=0.5,
                           channel_index=CONFIG.channel_index)


def test_can_use_detector_after_fit(path_to_config,
                                    path_to_sample_pipeline_folder,
                                    make_tmp_folder,
                                    path_to_standarized_data):
    yass.set_config(path_to_config, make_tmp_folder)
    CONFIG = yass.read_config()

    spike_train = np.load(path.join(path_to_sample_pipeline_folder,
                                    'spike_train_post_deconv_post_merge.npy'))
    chosen_templates = np.unique(spike_train[:, 1])
    min_amplitude = 4
    max_amplitude = 60

    templates = make.load_templates(path_to_sample_pipeline_folder,
                                    spike_train, CONFIG, chosen_templates)

    path_to_standarized = path.join(path_to_sample_pipeline_folder,
                                    'preprocess', 'standarized.bin')

    rec = RecordingsReader(path_to_standarized, loader='array').data

    (spatial_sig,
     temporal_sig) = noise_cov(rec, templates.shape[1], templates.shape[1])

    X, y = make.training_data_detect(templates=templates,
                                     minimum_amplitude=min_amplitude,
                                     maximum_amplitude=max_amplitude,
                                     n_clean_per_template=20,
                                     n_collided_per_spike=20,
                                     n_temporally_misaligned_per_spike=0.25,
                                     n_noise=20,
                                     n_spatially_misaliged_per_spike=0,
                                     spatial_SIG=spatial_sig,
                                     temporal_SIG=temporal_sig)

    _, waveform_length, n_neighbors = X.shape

    path_to_model = path.join(make_tmp_folder, 'detect-net.ckpt')

    detector = NeuralNetDetector(path_to_model, [8, 4],
                                 waveform_length, n_neighbors,
                                 threshold=0.5,
                                 channel_index=CONFIG.channel_index,
                                 n_iter=10)

    detector.fit(X, y)

    detector.predict(X)


def test_can_use_neural_networks(path_to_config,
                                 path_to_sample_pipeline_folder,
                                 make_tmp_folder,
                                 path_to_standarized_data):
    yass.set_config(path_to_config, make_tmp_folder)
    CONFIG = yass.read_config()

    PATH_TO_DATA = path_to_standarized_data

    with open(path.join(path_to_sample_pipeline_folder, 'preprocess',
                        'standarized.yaml')) as f:
        PARAMS = yaml.load(f)

    channel_index = make_channel_index(CONFIG.neigh_channels,
                                       CONFIG.geom)

    detection_th = CONFIG.detect.neural_network_detector.threshold_spike
    triage_th = CONFIG.detect.neural_network_triage.threshold_collision
    detection_fname = CONFIG.detect.neural_network_detector.filename
    ae_fname = CONFIG.detect.neural_network_autoencoder.filename
    triage_fname = CONFIG.detect.neural_network_triage.filename

    # instantiate neural networks
    NND = NeuralNetDetector.load(detection_fname, detection_th,
                                 channel_index)
    triage = KerasModel(triage_fname,
                        allow_longer_waveform_length=True,
                        allow_more_channels=True)
    NNAE = AutoEncoder.load(ae_fname, input_tensor=NND.waveform_tf)

    bp = BatchProcessor(PATH_TO_DATA, PARAMS['dtype'], PARAMS['n_channels'],
                        PARAMS['data_order'], '100KB',
                        buffer_size=CONFIG.spike_size)

    out = ('spike_index', 'waveform')
    fn = neuralnetwork.apply.fix_indexes_spike_index

    # detector
    with tf.Session() as sess:
        # get values of above tensors
        NND.restore(sess)

        res = bp.multi_channel_apply(NND.predict_recording,
                                     mode='memory',
                                     sess=sess,
                                     output_names=out,
                                     cleanup_function=fn)

    spike_index_new = np.concatenate([element[0] for element in res], axis=0)
    wfs = np.concatenate([element[1] for element in res], axis=0)

    idx_clean = triage.predict_with_threshold(wfs, triage_th)
    score = NNAE.predict(wfs)
    rot = NNAE.load_rotation()
    neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)

    (score_clear_new,
        spike_index_clear_new) = post_processing(score,
                                                 spike_index_new,
                                                 idx_clean,
                                                 rot,
                                                 neighbors)


def test_splitting_in_batches_does_not_affect(path_to_config,
                                              path_to_sample_pipeline_folder,
                                              make_tmp_folder,
                                              path_to_standarized_data):
    yass.set_config(path_to_config, make_tmp_folder)
    CONFIG = yass.read_config()

    PATH_TO_DATA = path_to_standarized_data

    with open(path.join(path_to_sample_pipeline_folder, 'preprocess',
                        'standarized.yaml')) as f:
        PARAMS = yaml.load(f)

    channel_index = make_channel_index(CONFIG.neigh_channels,
                                       CONFIG.geom)

    detection_th = CONFIG.detect.neural_network_detector.threshold_spike
    triage_th = CONFIG.detect.neural_network_triage.threshold_collision
    detection_fname = CONFIG.detect.neural_network_detector.filename
    ae_fname = CONFIG.detect.neural_network_autoencoder.filename
    triage_fname = CONFIG.detect.neural_network_triage.filename

    # instantiate neural networks
    NND = NeuralNetDetector.load(detection_fname, detection_th,
                                 channel_index)
    triage = KerasModel(triage_fname,
                        allow_longer_waveform_length=True,
                        allow_more_channels=True)
    NNAE = AutoEncoder.load(ae_fname, input_tensor=NND.waveform_tf)

    bp = BatchProcessor(PATH_TO_DATA, PARAMS['dtype'], PARAMS['n_channels'],
                        PARAMS['data_order'], '100KB',
                        buffer_size=CONFIG.spike_size)

    out = ('spike_index', 'waveform')
    fn = neuralnetwork.apply.fix_indexes_spike_index

    # detector
    with tf.Session() as sess:
        # get values of above tensors
        NND.restore(sess)

        res = bp.multi_channel_apply(NND.predict_recording,
                                     mode='memory',
                                     sess=sess,
                                     output_names=out,
                                     cleanup_function=fn)

    spike_index_new = np.concatenate([element[0] for element in res], axis=0)
    wfs = np.concatenate([element[1] for element in res], axis=0)

    idx_clean = triage.predict_with_threshold(wfs, triage_th)
    score = NNAE.predict(wfs)
    rot = NNAE.load_rotation()
    neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)

    (score_clear_new,
        spike_index_clear_new) = post_processing(score,
                                                 spike_index_new,
                                                 idx_clean,
                                                 rot,
                                                 neighbors)

    with tf.Session() as sess:
        # get values of above tensors
        NND.restore(sess)

        res = bp.multi_channel_apply(NND.predict_recording,
                                     mode='memory',
                                     sess=sess,
                                     output_names=('spike_index',
                                                   'waveform'),
                                     cleanup_function=fn)

    spike_index_batch, wfs = zip(*res)

    spike_index_batch = np.concatenate(spike_index_batch, axis=0)
    wfs = np.concatenate(wfs, axis=0)

    idx_clean = triage.predict_with_threshold(x=wfs,
                                              threshold=triage_th)

    score = NNAE.predict(wfs)
    rot = NNAE.load_rotation()
    neighbors = n_steps_neigh_channels(CONFIG.neigh_channels, 2)

    (score_clear_batch,
        spike_index_clear_batch) = post_processing(score,
                                                   spike_index_batch,
                                                   idx_clean,
                                                   rot,
                                                   neighbors)
