import os
import logging
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tqdm

from statsmodels import robust

from yass.deconvolute.util import (svd_shifted_templates,
                                   small_shift_templates,
                                   make_spt_list_parallel, clean_up,
                                   calculate_temp_temp_parallel)
                                   
from yass.deconvolute.deconvolve import (deconvolve_new_allcores_updated,
                                         deconvolve_match_pursuit)
                                         
from yass.deconvolute.match_pursuit import (MatchPursuit_objectiveUpsample, 
                                            MatchPursuitWaveforms)
from yass.cluster.util import (binary_reader, RRR3_noregress_recovery,
                               global_merge_max_dist, PCA, 
                               load_waveforms_from_memory,
                               make_CONFIG2)
from yass import read_config

import multiprocessing as mp
colors = np.asarray(["#000000", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
        "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
        "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
        "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
        "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
        "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
        "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
        "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
        
        "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
        "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
        "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
        "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
        "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
        "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
        "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
        "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94", "#7ED379", "#012C58",
        
        "#7A7BFF", "#D68E01", "#353339", "#78AFA1", "#FEB2C6", "#75797C", "#837393", "#943A4D",
        "#B5F4FF", "#D2DCD5", "#9556BD", "#6A714A", "#001325", "#02525F", "#0AA3F7", "#E98176",
        "#DBD5DD", "#5EBCD1", "#3D4F44", "#7E6405", "#02684E", "#962B75", "#8D8546", "#9695C5",
        "#E773CE", "#D86A78", "#3E89BE", "#CA834E", "#518A87", "#5B113C", "#55813B", "#E704C4",
        "#00005F", "#A97399", "#4B8160", "#59738A", "#FF5DA7", "#F7C9BF", "#643127", "#513A01",
        "#6B94AA", "#51A058", "#A45B02", "#1D1702", "#E20027", "#E7AB63", "#4C6001", "#9C6966",
        "#64547B", "#97979E", "#006A66", "#391406", "#F4D749", "#0045D2", "#006C31", "#DDB6D0",
        "#7C6571", "#9FB2A4", "#00D891", "#15A08A", "#BC65E9", "#FFFFFE", "#C6DC99", "#203B3C",

        "#671190", "#6B3A64", "#F5E1FF", "#FFA0F2", "#CCAA35", "#374527", "#8BB400", "#797868",
        "#C6005A", "#3B000A", "#C86240", "#29607C", "#402334", "#7D5A44", "#CCB87C", "#B88183",
        "#AA5199", "#B5D6C3", "#A38469", "#9F94F0", "#A74571", "#B894A6", "#71BB8C", "#00B433",
        "#789EC9", "#6D80BA", "#953F00", "#5EFF03", "#E4FFFC", "#1BE177", "#BCB1E5", "#76912F",
        "#003109", "#0060CD", "#D20096", "#895563", "#29201D", "#5B3213", "#A76F42", "#89412E",
        "#1A3A2A", "#494B5A", "#A88C85", "#F4ABAA", "#A3F3AB", "#00C6C8", "#EA8B66", "#958A9F",
        "#BDC9D2", "#9FA064", "#BE4700", "#658188", "#83A485", "#453C23", "#47675D", "#3A3F00",
        "#061203", "#DFFB71", "#868E7E", "#98D058", "#6C8F7D", "#D7BFC2", "#3C3E6E", "#D83D66",
        
        "#2F5D9B", "#6C5E46", "#D25B88", "#5B656C", "#00B57F", "#545C46", "#866097", "#365D25",
        "#252F99", "#00CCFF", "#674E60", "#FC009C", "#92896B"])
        
colors = np.concatenate([colors,colors])
def run2(spike_train_cluster,
        templates,
        output_directory='tmp/',
        recordings_filename='standarized.bin'):
    """Deconvolute spikes

    Parameters
    ----------

    spike_index_all: numpy.ndarray (n_data, 3)
        A 2D array for all potential spikes whose first column indicates the
        spike time and the second column the principal channels
        3rd column indicates % confidence of cluster membership
        Note: can now have single events assigned to multiple templates

    templates: numpy.ndarray (n_channels, waveform_size, n_templates)
        A 3D array with the templates

    output_directory: str, optional
        Output directory (relative to CONFIG.data.root_folder) used to load
        the recordings to generate templates, defaults to tmp/

    recordings_filename: str, optional
        Recordings filename (relative to CONFIG.data.root_folder/
        output_directory) used to draw the waveforms from, defaults to
        standarized.bin

    Returns
    -------
    spike_train: numpy.ndarray (n_clear_spikes, 2)
        A 2D array with the spike train, first column indicates the spike
        time and the second column the neuron ID

    Examples
    --------

    .. literalinclude:: ../../examples/pipeline/deconvolute.py
    """

    logger = logging.getLogger(__name__)

    CONFIG = read_config()

    logging.debug('Starting deconvolution. templates.shape: {}, '
                  'spike_index_cluster.shape: {}'.format(templates.shape,
                                                 spike_train_cluster.shape))

    ''' ******************************************
        ************** LOAD PARAMETERS ***********
        ******************************************
    ''' 
    idx_list = compute_idx_list(templates, CONFIG, output_directory,
                                recordings_filename)
    
    print("# of chunks for deconvolution: ", len(idx_list), 
            " verbose mode: ", CONFIG.deconvolution.verbose)
            
    # make deconv directory
    deconv_dir = os.path.join(CONFIG.data.root_folder, 'tmp/deconv')
    if not os.path.isdir(deconv_dir):
        os.makedirs(deconv_dir)
    
    # read recording chunk and share as global variable
    # Cat: TODO: recording_chunk should be a shared variable in 
    #            multiprocessing module;
    buffer_size = 200
    standardized_filename = os.path.join(CONFIG.data.root_folder,
                                        output_directory, recordings_filename)

    ''' ******************************************
        ************** DECONV ********************
        ******************************************
    ''' 
    # compute pairwise convolution filter outside match pursuit
    # Cat: TODO: make sure you don't miss chunks at end
    # Cat: TODO: do we want to do 10sec chunks in deconv?
    initial_chunk = 6 # number of 10 sec chunks to run initially
    chunk_ctr = 0
    max_iter = 5000
    
    # Cat: TODO: read both from CONFIG
    threshold = 20.    
    conv_approx_rank = 5
    
    ''' 
    ***********************************************************
    ****************** LOOP MATCH PURSUIT  ********************
    ***********************************************************
    '''
    # compute pairwise convolution filter outside match pursuit
    # Cat: TODO: make sure you don't miss chunks at end
    # Cat: TODO: do we want to do 10sec chunks in deconv?
    chunk_size = initial_chunk
    
    for chunk_ctr, c in enumerate(range(0, len(idx_list), chunk_size)):
 
        # select segments and chunk to be processed
        #idx_list_local = idx_list[c:c+chunk_size]
        idx_list_local = idx_list[c:c+chunk_size]
        deconv_chunk_dir = os.path.join(CONFIG.data.root_folder,
                          'tmp/deconv/chunk_'+str(chunk_ctr).zfill(6))
        if not os.path.isdir(deconv_chunk_dir):
            os.makedirs(deconv_chunk_dir)
            os.makedirs(deconv_chunk_dir+'/lost_units/')
       
        # temporary test: check same chunk repeatedly:
        idx_list_local = idx_list[:chunk_size]
        
        ''' 
        # *******************************************
        # **** RUN MATCH PURSUIT & RESIDUAL COMP ****
        # *******************************************
        '''
        templates, spike_train_cluster_prev_iteration = match_pursuit_function(
                        CONFIG, 
                        templates, 
                        spike_train_cluster,
                        deconv_chunk_dir,
                        standardized_filename,
                        max_iter,
                        threshold,
                        conv_approx_rank,
                        idx_list_local, 
                        chunk_ctr,
                        buffer_size)
        
        '''
        # *****************************************
        # ************** RECLUSTERING *************
        # *****************************************   
        '''
        templates, spike_train_cluster = reclustering_function(CONFIG,
                                          templates,
                                          deconv_chunk_dir,
                                          spike_train_cluster_prev_iteration,
                                          idx_list_local,
                                          initial_chunk,
                                          output_directory, 
                                          recordings_filename)


    ''' *********************************************************
        **************** POST DECONV CLEAN UP *******************
        *********************************************************
    '''

    # reload all spike trains and concatenate them:
    spike_train = np.zeros((0,2),'int32')
    for chunk_ctr, c in enumerate(range(0, len(idx_list), chunk_size)):

        # make deconv chunk directory
        deconv_chunk_dir = os.path.join(CONFIG.data.root_folder,
                          'tmp/deconv/chunk_'+str(chunk_ctr).zfill(6))

        deconv_results = np.load(deconv_chunk_dir+'/deconv_results.npz')
        temp_train = deconv_results['spike_train']
        idx = np.argsort(temp_train[:,0])
        spike_train = np.vstack((spike_train, temp_train))

    # Cat: TODO: reorder spike train by time
    print ("Final deconv spike train: ", spike_train.shape)
    print ("Final deconv templates: ", templates.shape)

    logger.info('spike_train.shape: {}'.format(spike_train.shape))

    return spike_train, templates

def delete_spikes(templates, spike_train):

    # need to transpose axes for analysis below
    templates = templates.swapaxes(0,1)

    # remove templates < 3SU
    # Cat: TODO: read this threshold and flag from CONFIG
    template_threshold = 3
    
    ptps = templates.ptp(0).max(0)
    idx_remove = np.where(ptps<=template_threshold)[0]
    print ("  deleted spikes from # clusters < 3SU: ", idx_remove.shape[0])
    print (idx_remove)
    # Cat: TODO: speed this up!
    for idx_ in idx_remove:
        temp_idx = np.where(spike_train[:,1]==idx_)[0]
        spike_train = np.delete(spike_train, temp_idx, axis=0)        

    print (spike_train.shape)
    #quit()


    return spike_train


def delete_templates(templates, spike_train_cluster):

    # remove templates < 3SU
    # Cat: TODO: read this threshold and flag from CONFIG
    template_threshold = 3

    # need to transpose axes for analysis below
    templates = templates.swapaxes(0,1)
    
    ptps = templates.ptp(0).max(0)
    idx = np.where(ptps>=template_threshold)[0]
    print ("  deleted # clusters < 3SU: ", templates.shape[2]-idx.shape[0])
    
    templates = templates[:,:,idx]
    
    spike_train_cluster_new = []
    for ctr,k in enumerate(idx):
        temp = np.where(spike_train_cluster[:,1]==k)[0]
        temp_train = spike_train_cluster[temp]
        temp_train[:,1]=ctr
        spike_train_cluster_new.append(temp_train)
        
    spike_train_cluster_new = np.vstack(spike_train_cluster_new)
    
    return templates, spike_train_cluster_new

    
def compute_idx_list(templates, CONFIG, output_directory, 
                                        recordings_filename):
    
    # necessary parameters
    n_channels, n_temporal_big, n_templates = templates.shape
    templates = np.transpose(templates, (2, 1, 0))

    sampling_rate = CONFIG.recordings.sampling_rate
    max_spikes = CONFIG.deconvolution.max_spikes
    n_processors = CONFIG.resources.n_processors
    n_sec_chunk = CONFIG.resources.n_sec_chunk
    
    # Cat: TODO: read from CONFIG file
    buffer_size = 200

    # Grab length of .dat file to compute chunk indexes below
    standardized_filename = os.path.join(CONFIG.data.root_folder, 
                                    output_directory, recordings_filename)
    
    fp = np.memmap(standardized_filename, dtype='float32', mode='r')
    fp_len = fp.shape[0]

    # Generate indexes in chunks (units of seconds); usually 10-60sec
    indexes = np.arange(0, fp_len / n_channels, sampling_rate * n_sec_chunk)
    if indexes[-1] != fp_len / n_channels:
        indexes = np.hstack((indexes, fp_len / n_channels))

    # Make the 4 parameter list to be sent to deconvolution algorithm
    idx_list = []
    for k in range(len(indexes) - 1):
        idx_list.append([
            indexes[k], indexes[k + 1], buffer_size,
            indexes[k + 1] - indexes[k] + buffer_size
        ])

    idx_list = np.int64(np.vstack(idx_list)) #[:2]

    return idx_list

def reclustering_function(CONFIG,
                          templates,
                          deconv_chunk_dir,
                          spike_train_cluster_new,
                          idx_list_local,
                          initial_chunk,
                          output_directory, 
                          recordings_filename):
                              
    idx_chunk = [idx_list_local[0][0], idx_list_local[-1][1], 
                 idx_list_local[0][2], idx_list_local[0][3]]
    
    n_sec_chunk = CONFIG.resources.n_sec_chunk

    # make lists of arguments to be passed to 
    print ("  reclustering initial deconv chunk output...")
    CONFIG2 = make_CONFIG2(CONFIG)
    
    # load spike train and set train to beginning of chunk for indexing
    dec_spike_train_offset = np.load(deconv_chunk_dir+
                                        '/dec_spike_train_offset.npy')
    dec_spike_train_offset[:,0]-=idx_chunk[0]
    print ("  NO. UNIQUE SPIKE IDS: ", 
                        np.unique(dec_spike_train_offset[:,1]).shape)
   
    ''' ************************************************************
        ******************* READ RAW DATA CHUNK ********************
        ************************************************************
    '''
    # read recording chunk and share as global variable
    # Cat: TODO: recording_chunk should be a shared variable in 
    #            multiprocessing module;
    global recording_chunk
    idx = idx_chunk
    data_start = idx[0]
    offset = idx[2]

    buffer_size = 200
    standardized_filename = os.path.join(CONFIG.data.root_folder,
                                         output_directory, 
                                         recordings_filename)
    n_channels = CONFIG.recordings.n_channels
    recording_chunk = binary_reader(idx, buffer_size, 
                standardized_filename, n_channels)
                    

    ''' ************************************************************
        ************** SETUP & RUN RECLUSTERING ********************
        ************************************************************
    '''

    # clean templates
    #clean_templates()


    args_in = []
    units = np.arange(templates.shape[2])
    for unit in units:
        fname_out = (deconv_chunk_dir+
                     "/unit_{}.npz".format(
                     str(unit).zfill(6)))
        if os.path.exists(fname_out)==False:
            args_in.append([unit, 
                            dec_spike_train_offset,
                            spike_train_cluster_new,
                            idx_chunk,
                            templates[:,:,unit],
                            CONFIG2,
                            deconv_chunk_dir,
                            data_start,
                            offset
                            ])

    # run residual-reclustering function
    if len(args_in)>0:
        if CONFIG.resources.multi_processing:
        #if False:
            p = mp.Pool(processes = CONFIG.resources.n_processors)
            res = p.map_async(deconv_residual_recluster, args_in).get(988895)
            p.close()
        else:
            for unit in range(len(args_in)):
                res = deconv_residual_recluster(args_in[unit])
    

    ''' ************************************************************
        ******************* RUN TEMPLATE MERGE  ********************
        ************************************************************
    '''
    # run template merge
    min_spikes = int(max(initial_chunk*n_sec_chunk*0.25,310))
    out_dir = 'deconv'
                            
    spike_train, tmp_loc, templates = global_merge_max_dist(
                                          deconv_chunk_dir, recording_chunk,
                                          CONFIG, min_spikes, out_dir, units)

    #print (templates_first_chunk.shape)
    np.savez(deconv_chunk_dir+"/deconv_results.npz", 
            spike_train=spike_train, 
            tmp_loc=tmp_loc, 
            templates=templates)
    
    
    return templates, spike_train

def compute_residual_function(CONFIG, idx_list_local,
                              buffer_size,
                              standardized_filename,
                              dec_spike_train,
                              sparse_upsampled_templates,
                              deconv_chunk_dir,
                              deconv_id_sparse_temp_map):
                              
    # re-read entire block to get waveforms 
    # get indexes for entire chunk from local chunk list
    idx_chunk = [idx_list_local[0][0], idx_list_local[-1][1], 
                 idx_list_local[0][2], idx_list_local[0][3]]
                 
    # read data block using buffer
    n_channels = CONFIG.recordings.n_channels
    
    #print (standardized_filename)
    recording_chunk = binary_reader(idx_chunk, buffer_size, 
                                    standardized_filename, 
                                    n_channels)
    
    np.save(deconv_chunk_dir+ '/recording_chunk.npy', recording_chunk)
                        
    # compute residual for data chunk and save to disk
    # Cat TODO: parallelize this and also figure out a faster way to 
    #           process this data
    # Note: offset spike train to account for recording_chunk buffer size
    # this also enables working with spikes that are near the edges
    dec_spike_train_offset = dec_spike_train
    dec_spike_train_offset[:,0]+=buffer_size
    
    wf_object = MatchPursuitWaveforms(recording_chunk,
                                      sparse_upsampled_templates,
                                      dec_spike_train_offset,
                                      buffer_size,
                                      n_processors=CONFIG.resources.n_processors)
    
    # compute residual using initial templates obtained above
    # Cat: TODO: parallelize this
    # Note: this uses spike times occuring at beginning of spike
    fname = (deconv_chunk_dir+"/residual.npy")
    
    wf_object.sparse_upsampled_templates = sparse_upsampled_templates
    wf_object.deconv_id_sparse_temp_map = deconv_id_sparse_temp_map

    if os.path.exists(fname)==False:
        wf_object.compute_residual_new(CONFIG)
        np.save(fname, wf_object.data)
    else:
        wf_object.data = np.load(fname)

    dec_spike_train_offset = offset_spike_train(CONFIG, dec_spike_train_offset)
    np.save(deconv_chunk_dir+'/dec_spike_train_offset.npy',dec_spike_train_offset)
    

def offset_spike_train(CONFIG, dec_spike_train_offset):
    # Cat: need to offset spike train as deconv outputs spike times at
    #      beginning of waveform; do this after the residual computation step
    #      which is based on beginning waveform residual computation
    #      So converting back to nn_spike time alignment for clustering steps
    # Cat: TODO read from CONFIG file; make sure this is corrected
    deconv_offset = int(CONFIG.recordings.spike_size_ms*
                                CONFIG.recordings.sampling_rate/1000.)
    print ("  offseting deconv spike train (timesteps): ",deconv_offset)
    dec_spike_train_offset[:,0]+= deconv_offset

    # - colapse unit ids using the expanded templates above
    #   as some templates were duplicated 30 times with shifts
    # - need to reset unit ids back to original templates and collapse
    #   over the spike trains
    # Note: deconv_spike_train does not have data buffer offset in it
    
    # Cat: TODO: read this value from CONFIG or another place; important!!
    upsample_max_val = 32.
    dec_spike_train_offset[:,1] = np.int32(dec_spike_train_offset[:,1]/upsample_max_val)

    return dec_spike_train_offset


def match_pursuit_function(CONFIG, 
                templates, 
                spike_train_cluster,
                deconv_chunk_dir,
                standardized_filename,
                max_iter,
                threshold,
                conv_approx_rank,
                idx_list_local,
                chunk_ctr,
                buffer_size):
                        
    
    print ("")
    print ("Initializing Match Pursuit for chunk: ", chunk_ctr, ", # segments: ", 
            idx_list_local.shape[0], 
            " start: ", idx_list_local[0][0], " end: ", 
            idx_list_local[-1][1], " start(sec): ", 
            round(idx_list_local[0][0]/float(CONFIG.recordings.sampling_rate),1),
            " end(sec): ", 
            round(idx_list_local[-1][1]/float(CONFIG.recordings.sampling_rate),1))
       
    # delete templates below certain treshold
    templates, spike_train_cluster_prev_iteration = delete_templates(templates,
                                                    spike_train_cluster)
    
    # initialize match pursuit
    # Cat: TODO: to read from CONFIG
    # this value sets dynamic upsampling for MP object
    default_upsample_value=0
    mp_object = MatchPursuit_objectiveUpsample(
                              temps=templates,
                              deconv_chunk_dir=deconv_chunk_dir,
                              standardized_filename=standardized_filename,
                              max_iter=max_iter,
                              upsample=default_upsample_value,
                              threshold=threshold,
                              conv_approx_rank=conv_approx_rank,
                              n_processors=CONFIG.resources.n_processors,
                              multi_processing=CONFIG.resources.multi_processing)
    
    print ("  running Match Pursuit...")
    
    # find which sections within current chunk not complete
    args_in = []
    for k in range(len(idx_list_local)):
        fname_out = (deconv_chunk_dir+
                     "/seg_{}_deconv.npz".format(
                     str(k).zfill(6)))
        if os.path.exists(fname_out)==False:
            args_in.append([[idx_list_local[k], k],
                            chunk_ctr,
                            buffer_size])

    if len(args_in)>0: 
        if CONFIG.resources.multi_processing:
            p = mp.Pool(processes = CONFIG.resources.n_processors)
            p.map_async(mp_object.run, args_in).get(988895)
            p.close()
        else:
            for k in range(len(args_in)):
                mp_object.run(args_in[k])
    
    # collect spikes
    res = []
    for k in range(len(idx_list_local)):
        fname_out = (deconv_chunk_dir+
                     "/seg_{}_deconv.npz".format(
                     str(k).zfill(6)))
                     
        data = np.load(fname_out)
        res.append(data['spike_train'])

    print ("  gathering spike trains")
    dec_spike_train = np.vstack(res)
    
    # get corrected spike trains first
    dec_spike_train = mp_object.correct_shift_deconv_spike_train(dec_spike_train)
    print ("  initial deconv spike train: ", dec_spike_train.shape)

    # save original spike ids (before upsampling
    # Cat: TODO: get this value from global/CONFIG
    upsample_max_val = 32.
    temp_spike_train = dec_spike_train.copy()
    temp_spike_train[:,1] = np.int32(dec_spike_train[:,1]/upsample_max_val)
    np.save(deconv_chunk_dir+"/spike_trains_postdeconv_prereclustering.npy", 
                                        temp_spike_train)
    np.save(deconv_chunk_dir+"/templates_postdeconv_prereclustering.npy", 
                                    templates)

    
    '''
    # *****************************************
    # *** COMPUTE RESIDUAL BY DERASTERIZING ***
    # *****************************************
    '''
    # get upsampled templates and mapping for computing residual
    sparse_upsampled_templates, deconv_id_sparse_temp_map = (
                            mp_object.get_sparse_upsampled_templates())
    print ("  sparse_upsampled_templates: ", sparse_upsampled_templates.shape)
    print ("  deconv_id_sparse_temp_map: ", deconv_id_sparse_temp_map)

    # Cat: TODO: this step might be redundant because no new templates are created here
    #       some templates may shrink a bit though...
    if True:
        spike_train = delete_spikes(sparse_upsampled_templates.swapaxes(0,1), 
                                    dec_spike_train)

    compute_residual_function(CONFIG, idx_list_local,
                              buffer_size,
                              standardized_filename,
                              spike_train,
                              sparse_upsampled_templates,
                              deconv_chunk_dir,
                              deconv_id_sparse_temp_map)

    return templates, spike_train_cluster_prev_iteration

def deconv_residual_recluster(data_in): 
    
    unit = data_in[0]
    dec_spike_train_offset = data_in[1]
    spike_train_cluster_new = data_in[2]
    idx_chunk = data_in[3]
    template = data_in[4]
    CONFIG = data_in[5]
    deconv_chunk_dir = data_in[6]
    data_start = data_in[7]
    offset = data_in[8]

    # Cat: TODO: read this from CONFIG
    n_dim_pca_compression = 5
    
    deconv_filename = (deconv_chunk_dir+"/unit_"+str(unit).zfill(6)+'.npz')

    if os.path.exists(deconv_filename)==False:
        
        # select deconv spikes and read waveforms
        unit_sp = dec_spike_train_offset[dec_spike_train_offset[:, 1] == unit, :]

        #print (unit, unit_sp)
        # save all clustered data
        if unit_sp.shape[0]==0: 
            print ("  unit: ", str(unit), " has no spikes...")
            np.savez(deconv_filename, spike_index=[], 
                        templates=[],
                        templates_std=[],
                        weights=[])
            return
               
        if unit_sp.shape[0]!= np.unique(unit_sp[:,0]).shape[0]:
            print ("  unit: ", unit, " non unique spikes found...")
            idx_unique = np.unique(unit_sp[:,0], return_index = True)[1]
            unit_sp = unit_sp[idx_unique]

        # Cat: TODO: load wider waveforms just as in clustering
        # Cat TODO: Need to load from CONFIG; careful as the templates are
        #           now being extended during cluster preamble using flexible val
        n_times = 61
        
        # Cat: TODO read this from disk
        # 
        deconv_max_spikes = 1000
        if unit_sp.shape[0]>deconv_max_spikes:
            idx_deconv = np.random.choice(np.arange(unit_sp.shape[0]),
                                          size=deconv_max_spikes,
                                          replace=False)
            unit_sp = unit_sp[idx_deconv]            

        # Cat: TODO: here we add addtiional offset for buffer inside residual matrix
        # read waveforms by adding templates to residual
        if False:
            wf = get_wfs_from_residual(unit_sp, template, deconv_chunk_dir,
                                   n_times)
        else:    
            # read waveforms from recording chunk in memory
            # load waveforms with some padding then clip them
            # Cat: TODO: spike_padding to be read/fixed in CONFIG
            spike_size = 30
            #print ("  data_start: ", data_start)
            #print ("  offset: ", offset)
            #print ("  unit_sp", unit_sp)
            wf = load_waveforms_from_memory(recording_chunk, data_start, 
                                            0, unit_sp, 
                                            spike_size)
        
            #print ("  wf shape: ", wf.shape)
            
        #np.save(deconv_chunk_dir+'/wf_loaded.npy', wf)
        channel = wf.mean(0).ptp(0).argmax(0)

        # run mfm
        scale = 10 

        triageflag = True
        alignflag = True
        plotting = False
        if unit%10==0:
            plotting = True
            
        n_feat_chans = 5
        n_dim_pca = 3
        wf_start = 0
        wf_end = 40
        mfm_threshold = 0.90
        knn_triage_threshold = 0.90
        upsample_factor = 5
        nshifts = 15
                
        chans = [] 
        gen = 0     #Set default generation for starting clustering stpe
        assignment_global = []
        spike_index = []
        templates = []
        feat_chans_cumulative = []
        
        # plotting parameters
        if plotting:
            #x = np.zeros(100, dtype = int)            
            #fig = plt.figure(figsize =(50,25))
            #grid = plt.GridSpec(10,5,wspace = 0.0,hspace = 0.2)
            #ax_t = fig.add_subplot(grid[13:, 6:])
            
            x = np.zeros(100, dtype = int)
            fig = plt.figure(figsize =(100,100))
            grid = plt.GridSpec(20,20,wspace = 0.0,hspace = 0.2)
            ax_t = fig.add_subplot(grid[13:, 6:])
            
        else:
            fig = []
            grid = []
            ax_t = []
            x = []
            
        deconv_flag = True
        RRR3_noregress_recovery(unit, wf, unit_sp, gen, fig, grid, x,
            ax_t, triageflag, alignflag, plotting, n_feat_chans, 
            n_dim_pca, wf_start, wf_end, mfm_threshold, CONFIG, 
            upsample_factor, nshifts, assignment_global, spike_index, scale,
            knn_triage_threshold, deconv_flag, templates)


        # finish plotting 
        if plotting: 
            #ax_t = fig.add_subplot(grid[13:, 6:])
            for i in range(CONFIG.recordings.n_channels):
                ax_t.text(CONFIG.geom[i,0], CONFIG.geom[i,1], str(i), alpha=0.4, 
                                                                fontsize=30)
                # fill bewteen 2SUs on each channel
                ax_t.fill_between(CONFIG.geom[i,0] + np.arange(-61,0,1)/3.,
                    -scale + CONFIG.geom[i,1], scale + CONFIG.geom[i,1], 
                    color='black', alpha=0.05)
                    
                # plot original templates
                ax_t.plot(CONFIG.geom[:,0]+
                    np.arange(-template.shape[0],0)[:,np.newaxis]/3., 
                    CONFIG.geom[:,1] + template*scale, 'r--', c='red')
                        
            # plot max chan with big red dot                
            ax_t.scatter(CONFIG.geom[channel,0], CONFIG.geom[channel,1], s = 2000, 
                                                    color = 'red')
            
            labels=[]
            if len(spike_index)>0: 
                sic_temp = np.concatenate(spike_index, axis = 0)
                assignment_temp = np.concatenate(assignment_global, axis = 0)
                idx = sic_temp[:,1] == unit
                clusters, sizes = np.unique(assignment_temp[idx], return_counts= True)
                clusters = clusters.astype(int)
                chans.extend(channel*np.ones(clusters.size))
                
                for i, clust in enumerate(clusters):
                    patch_j = mpatches.Patch(color = colors[clust%100], label = "deconv = {}".format(sizes[i]))
                    labels.append(patch_j)
            
            idx3 = np.where(spike_train_cluster_new[:,1]==unit)[0]
            spikes_in_chunk = np.where(np.logical_and(spike_train_cluster_new[idx3][:,0]>idx_chunk[0], 
                                                      spike_train_cluster_new[idx3][:,0]<=idx_chunk[1]))[0]

            patch_original = mpatches.Patch(color = 'red', label = 
                             "cluster in chunk/total: "+ 
                             str(spikes_in_chunk.shape[0])+"/"+
                             str(idx3.shape[0]))
            labels.append(patch_original)
            
            ax_t.legend(handles = labels, fontsize=30)

            # plto title
            fig.suptitle("Unit: "+str(unit), fontsize=25)
            fig.savefig(deconv_chunk_dir+"/unit{}.png".format(unit))
            plt.close(fig)


        # Cat: TODO: note clustering is done on PCA denoised waveforms but
        #            templates are computed on original raw signal
        np.savez(deconv_filename, spike_index=spike_index, 
                        templates=templates)
                        #templates_std=temp_std,
                        #weights=np.asarray([sic.shape[0] for sic in spike_index]))

        print ("**** Unit ", str(unit), ", found # clusters: ", len(spike_index))
        
    else: 
    #    # 
        data = np.load(filename_postclustering, encoding='latin1')
        spike_index = data['spike_index']



    if len(spike_index)==0:
        fname = (deconv_chunk_dir+"/lost_units/unit_"+str(unit).zfill(6)+'.npz')
        np.savez(fname, original_template=template)
    
    # overwrite this variable just in case multiprocessing doesn't destroy it
    wf = None
    
    return channel


def get_wfs_from_residual(unit_sp, template, deconv_chunk_dir, 
                          n_times=61):
                                  
    """Gets clean spikes for a given unit."""
    
    # Note: residual contains buffers
    fname = deconv_chunk_dir+'/residual.npy'
    data = np.load(deconv_chunk_dir+'/residual.npy')

    # Add the spikes of the current unit back to the residual
    x = np.arange(-n_times//2,n_times//2,1)
    temp = data[x + unit_sp[:, :1], :] + template
    
    data = None
    return temp


def visible_chans(temps):
    a = temps.ptp(0) #np.max(temps, axis=0) - np.min(temps, 0)
    vis_chan = a > 1

    return vis_chan

        
def pairwise_filter_conv_local(deconv_chunk_dir, n_time, n_unit, temporal, 
                         singular, spatial, approx_rank, vis_chan, temps):
    
    #print (deconv_chunk_dir+"/parwise_conv.npy")
    if os.path.exists(deconv_chunk_dir+"/pairwise_conv.npy")==False:
        print ("IN LOOP")
        conv_res_len = n_time * 2 - 1
        pairwise_conv = np.zeros([n_unit, n_unit, conv_res_len])
        for unit1 in range(n_unit):
            u, s, vh = temporal[unit1], singular[unit1], spatial[unit1]
            vis_chan_idx = vis_chan[:, unit1]
            for unit2 in range(n_unit):
                for i in range(approx_rank):
                    pairwise_conv[unit2, unit1, :] += np.convolve(
                        np.matmul(temps[:, vis_chan_idx, unit2], vh[i, vis_chan_idx].T),
                        s[i] * u[:, i].flatten(), 'full')

        np.save(deconv_chunk_dir+"/pairwise_conv.npy", pairwise_conv)
    else:
        pairwise_conv = np.load(deconv_chunk_dir+"/pairwise_conv.npy")
        
    return pairwise_conv
    

def fix_spiketrains(up_up_map, spike_train):
    
    # assign unique spike train ids    
    spike_train_fixed = spike_train.copy()
    ctr=0
    #for k in np.unique(spike_train[:,1])[1:]:
    for k in np.arange(1,up_up_map.shape[0],1):

        idx = np.where(spike_train[:,1]==k)[0]
        
        # assign unique template id
        if up_up_map[k]==up_up_map[k-1]:
            spike_train_fixed[idx,1] = ctr
        else:
            ctr+=1
            spike_train_fixed[idx,1] = ctr
        
        #if k%1000==0: 
        print ("  ", k, up_up_map[k], idx.shape, ctr)

    return spike_train_fixed

    
