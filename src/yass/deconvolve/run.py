import os
import logging
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import tqdm

from statsmodels import robust
from scipy.signal import argrelmin

#from yass.deconvolute.util import (svd_shifted_templates,
#                                   small_shift_templates,
#                                   make_spt_list_parallel, clean_up,
#                                   calculate_temp_temp_parallel)
                                   
#from yass.deconvolute.deconvolve import (deconvolve_new_allcores_updated,
#                                         deconvolve_match_pursuit)
                                         
from yass.deconvolve.match_pursuit import (MatchPursuit_objectiveUpsample, 
                                            MatchPursuitWaveforms)
                                            
from yass.cluster.util import (binary_reader, RRR3_noregress_recovery,
                               global_merge_max_dist, PCA, 
                               load_waveforms_from_memory,
                               make_CONFIG2, upsample_parallel, 
                               clean_templates, find_clean_templates)
from yass import read_config

import multiprocessing as mp

            
from matplotlib import colors as mcolors
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
by_hsv = ((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                 for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]


def run(spike_train_cluster,
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

    ''' 
    ******************************************
    ************** LOAD PARAMETERS ***********
    ******************************************
    ''' 
    # compute chunk and segment lists for parallel processing below
    idx_list = compute_idx_list(templates, 
                                CONFIG, 
                                output_directory,
                                recordings_filename)
           
    # make deconv directory
    deconv_dir = os.path.join(CONFIG.data.root_folder, 'tmp/deconv')
    if not os.path.isdir(deconv_dir):
        os.makedirs(deconv_dir)

    # read recording chunk and share as global variable
    # Cat: TODO: recording_chunk should be a shared variable in 
    #            multiprocessing module;
    buffer_size = 200
    standardized_filename = os.path.join(CONFIG.data.root_folder,
                                         output_directory, 
                                         recordings_filename)

    # compute pairwise convolution filter outside match pursuit
    # Cat: TODO: make sure you don't miss chunks at end
    # Cat: TODO: do we want to do 10sec chunks in deconv?
    n_seconds_initial = 60
    initial_chunk = int(n_seconds_initial//CONFIG.resources.n_sec_chunk) 
    
    chunk_ctr = 0
    max_iter = 5000
    
    # Cat: TODO: read both from CONFIG
    threshold = 10.    
    conv_approx_rank = 5
    
    ''' 
    ***********************************************************
    ****************** LOOP MATCH PURSUIT  ********************
    ***********************************************************
    '''
    # compute pairwise convolution filter outside match pursuit
    # Cat: TODO: make sure you don't miss chunks at end
    # Cat: TODO: do we want to do 10sec chunks in deconv?
    
    n_iterations = 0
    if n_iterations>0: 
        if not os.path.isdir(deconv_dir+'/initial/'):
            os.makedirs(deconv_dir+'/initial/')


    # modify templates for steps below
    templates = templates.swapaxes(1,2).swapaxes(0,1)

    # Cat: TODO: fix this from CONFIG
    if templates.shape[0] == 61:
        spike_padding = 0
        pass
    elif templates.shape[0] == 111:
        spike_padding = 25
        templates = templates[spike_padding:-spike_padding]
    else:
        print ("  template array error!")
        quit()

    chunk_size = initial_chunk
    for chunk_ctr, c in enumerate(range(0, len(idx_list), chunk_size)):
 
        print ("  iterative deconv not fully implemented yet...(need to fix abs_max_dist without padding fro deconv templates")
        # decide how many deconv+recluster iterations will do before exiting
        if chunk_ctr==n_iterations:
            break
            
        # select segments and chunk to be processed
        #idx_list_local = idx_list[c:c+chunk_size]
        idx_list_local = idx_list[c:c+chunk_size]
        deconv_chunk_dir = os.path.join(CONFIG.data.root_folder,
                          'tmp/deconv/initial/chunk_'+str(chunk_ctr).zfill(6))
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

    
        (sparse_upsampled_templates, 
         dec_spike_train, 
         deconv_id_sparse_temp_map, 
         spike_train_cluster_prev_iteration) = match_pursuit_function(
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
        # *** COMPUTE RESIDUAL BY DERASTERIZING ***
        # *****************************************
        '''                                                 

        compute_residual_function(CONFIG, 
                                  idx_list_local,
                                  buffer_size,
                                  standardized_filename,
                                  dec_spike_train,
                                  sparse_upsampled_templates,
                                  deconv_chunk_dir,
                                  deconv_id_sparse_temp_map,
                                  chunk_size,
                                  CONFIG.resources.n_sec_chunk)

        
        '''
        # *****************************************
        # ************** RECLUSTERING *************
        # *****************************************   
        '''
        templates, spike_train_cluster = reclustering_function(
                                              CONFIG,
                                              templates,
                                              deconv_chunk_dir,
                                              spike_train_cluster_prev_iteration,
                                              idx_list_local,
                                              initial_chunk,
                                              output_directory, 
                                              recordings_filename)


    ''' 
    ***********************************************************
    *************** RUN MATCH PURSUIT OVER ALL DATA ***********
    ***********************************************************
    '''
    
    # run over rest of data in single chunk run:
    chunk_size = len(idx_list)
    print (templates.shape)
    #templates = templates.swapaxes(1,2)
    for chunk_ctr, c in enumerate(range(0, len(idx_list), chunk_size)):
 
        # select segments and chunk to be processed
        #idx_list_local = idx_list[c:c+chunk_size]
        idx_list_local = idx_list[c:c+chunk_size]
        deconv_chunk_dir = os.path.join(CONFIG.data.root_folder,
                          'tmp/deconv/chunk_'+str(chunk_ctr).zfill(6))
        if not os.path.isdir(deconv_chunk_dir):
            os.makedirs(deconv_chunk_dir)
            os.makedirs(deconv_chunk_dir+'/lost_units/')

        print (" TODO: don't recomp temp_temp for final step if prev. computed!")         
        #templates = templates.swapaxes(1,2)
        match_pursuit_function(
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
    *********************************************************
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

    # Cat: TODO: speed this up!
    for idx_ in idx_remove:
        temp_idx = np.where(spike_train[:,1]==idx_)[0]
        spike_train = np.delete(spike_train, temp_idx, axis=0)        

    return spike_train


    

def align_singletrace_lastchan(wf, CONFIG, upsample_factor = 5, nshifts = 15, 
         ref = None):

    ''' Align all waveforms to the master channel

        wf = selected waveform matrix (# spikes, # samples, # featchans)
        mc = maximum channel from featchans; usually first channle, i.e. 0
    '''

    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1    

    # or loop over every channel and parallelize each channel:
    wf_up = upsample_parallel(wf.T, upsample_factor)

    wlen = wf_up.shape[1]
    wf_start = int(.15 * (wlen-1))
    wf_end = -int(.20 * (wlen-1))
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]

    ref_upsampled = wf_up[-1]
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])

    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2)+1)):
        ref_shifted[:,i] = ref_upsampled[s+wf_start:s+wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis,:], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]
    wf_final = np.zeros([wf.shape[1],wlen_trunc])
    for i,s in enumerate(best_shifts):
        wf_final[i] = wf_up[i,-s+ wf_start: -s+ wf_end]

    return np.float32(wf_final[:,::upsample_factor]), best_shifts
    
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

    residuaL_clustering_flag = False
    if residuaL_clustering_flag:
        print ("  reclustering using residuals ")
    else:
        print ("  reclustering using raw data ")

    buffer_size = 200
    standardized_filename = os.path.join(CONFIG.data.root_folder,
                                         output_directory, 
                                         recordings_filename)
    n_channels = CONFIG.recordings.n_channels
    recording_chunk = binary_reader(idx, 
                                    buffer_size, 
                                    standardized_filename, 
                                    n_channels)
                    

    ''' ************************************************************
        ************** SETUP & RUN RECLUSTERING ********************
        ************************************************************
    '''

    # make argument list
    args_in = []
    #templates=templates.swapaxes(0,1)
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
                            offset,
                            residuaL_clustering_flag
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
        ******* RECOMPUTE TEMPLATES USING NEW SPIKETRAINS  *********
        ************************************************************
    '''
    print ("  TODO:  RECOMPUTE TEMPLATES USING RAW DATA...")
    quit()
    
    
    
    

    ''' ************************************************************
        ******************* RUN TEMPLATE MERGE  ********************
        ************************************************************
    '''
    # run template merge
    out_dir = 'deconv'
    spike_train, templates = global_merge_max_dist(
                                          deconv_chunk_dir, recording_chunk,
                                          CONFIG, out_dir, units)

    #print (templates_first_chunk.shape)
    np.savez(deconv_chunk_dir+"/deconv_results_post_recluster.npz", 
            spike_train=spike_train, 
            templates=templates)
    
    return templates, spike_train

def compute_residual_function(CONFIG, idx_list_local,
                              buffer_size,
                              standardized_filename,
                              dec_spike_train,
                              sparse_upsampled_templates,
                              deconv_chunk_dir,
                              deconv_id_sparse_temp_map,
                              chunk_size,
                              n_sec_chunk):
                              
    # re-read entire block to get waveforms 
    # get indexes for entire chunk from local chunk list
    idx_chunk = [idx_list_local[0][0], idx_list_local[-1][1], 
                 idx_list_local[0][2], idx_list_local[0][3]]
                 
    # read data block using buffer
    n_channels = CONFIG.recordings.n_channels
    
    #print (standardized_filename)
    # Cat: TODO: this is 
    recording_chunk = binary_reader(idx_chunk, 
                                    buffer_size, 
                                    standardized_filename, 
                                    n_channels)

    #np.save('/media/cat/1TB/liam/49channels/data1_allset/tmp/deconv/initial/chunk_000000/raw_total.npy', recording_chunk)
    #data_size = recording_chunk.shape
    
    # compute residual for data chunk and save to disk
    # Cat TODO: parallelize this and also figure out a faster way to 
    #           process this data
    # Note: offset spike train to account for recording_chunk buffer size
    # this also enables working with spikes that are near the edges
    dec_spike_train_offset = dec_spike_train
    dec_spike_train_offset[:,0] += buffer_size
    
    print ("  init residual object")
    wf_object = MatchPursuitWaveforms(sparse_upsampled_templates,
                                      dec_spike_train_offset,
                                      buffer_size,
                                      CONFIG.resources.n_processors,
                                      deconv_chunk_dir,
                                      chunk_size,
                                      n_sec_chunk,
                                      idx_list_local)
    
        
    # compute residual using initial templates obtained above
    # Note: this uses spike times occuring at beginning of spike
    fname = (deconv_chunk_dir+"/residual.npy")
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
                spike_train_cluster_prev_iteration,
                deconv_chunk_dir,
                standardized_filename,
                max_iter,
                threshold,
                conv_approx_rank,
                idx_list_local,
                chunk_ctr,
                buffer_size):
                        
   # global pairwise_conv

    print ("")
    print ("Initializing Match Pursuit for chunk: ", chunk_ctr, ", # segments: ", 
            idx_list_local.shape[0], 
            " start: ", idx_list_local[0][0], " end: ", 
            idx_list_local[-1][1], " start(sec): ", 
            round(idx_list_local[0][0]/float(CONFIG.recordings.sampling_rate),1),
            " end(sec): ", 
            round(idx_list_local[-1][1]/float(CONFIG.recordings.sampling_rate),1))
       
    # delete templates below certain treshold; and collision templates
    #templates, spike_train_cluster_prev_iteration = clean_templates(templates,
    #                                                    spike_train_cluster_prev_iteration,
    #                                                    CONFIG)
    
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
    np.savez(deconv_chunk_dir+"/deconv_results.npz", 
            spike_train=temp_spike_train, 
            templates=templates)
            
    '''
    # ********************************************
    # * LOAD CORRECT TEMPLATES FOR RESIDUAL COMP *
    # ********************************************
    '''
    
    # get upsampled templates and mapping for computing residual
    sparse_upsampled_templates, deconv_id_sparse_temp_map = (
                            mp_object.get_sparse_upsampled_templates())

    return (sparse_upsampled_templates, dec_spike_train, 
            deconv_id_sparse_temp_map, spike_train_cluster_prev_iteration)

    

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
    residuaL_clustering_flag = data_in[9]

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
        spike_size = 111
        #template = template[25:-25,:]
        
        
        # Cat: TODO read this from disk
        deconv_max_spikes = 1000
        if unit_sp.shape[0]>deconv_max_spikes:
            idx_deconv = np.random.choice(np.arange(unit_sp.shape[0]),
                                          size=deconv_max_spikes,
                                          replace=False)
            unit_sp = unit_sp[idx_deconv]         

        # Cat: TODO: here we add addtiional offset for buffer inside residual matrix
        # read waveforms by adding templates to residual
        residuaL_clustering_flag=True
        if residuaL_clustering_flag:
            spike_size = 61
            wf = get_wfs_from_residual(unit_sp, 
                                       template, 
                                       deconv_chunk_dir,
                                       spike_size)
        else:    
            # read waveforms from recording chunk in memory
            # load waveforms with some padding then clip them below
            # Cat: TODO: spike_padding to be read/fixed in CONFIG
            #unit_sp[:,0]+=50
            #spike_size = template.shape[0]//2
            offset = 0
            wf = load_waveforms_from_memory(recording_chunk, 
                                            data_start, 
                                            offset, 
                                            unit_sp, 
                                            spike_size)
        #print (wf.shape)
        
        if wf.shape[1]==111:
            spike_start = 25
            spike_end = -25
        elif wf.shape[1]==61:
            spike_start =0
            spike_end = wf.shape[1]    
        else:
            print ("  spike width irregular fix this...")
            quit()


        #np.save(deconv_chunk_dir+'/wfs_'+str(unit).zfill(6)+'.npy', wf)
        # Cat: TODO: during deconv reclustering may not wish to exclude off-max
        #               channel templates
        channel = wf.mean(0).ptp(0).argmax(0)

        # run mfm
        scale = 10 

        triageflag = False
        alignflag = True
        plotting = False
        if unit%10==0:
            plotting = False
            
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
        RRR3_noregress_recovery(unit, wf[:, spike_start:spike_end], unit_sp, gen, fig, grid, x,
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
                ax_t.fill_between(CONFIG.geom[i,0] + np.arange(-spike_size//2,spike_size//2,1)/3.,
                    -scale + CONFIG.geom[i,1], scale + CONFIG.geom[i,1], 
                    color='black', alpha=0.05)
                    
                # plot original templates
                ax_t.plot(CONFIG.geom[:,0]+
                    np.arange(-template.shape[0]//2,template.shape[0]//2,1)[:,np.newaxis]/3., 
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
                    patch_j = mpatches.Patch(color = sorted_names[clust%100], 
                                             label = "deconv = {}".format(sizes[i]))
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
        #np.savez(deconv_filename, spike_index=spike_index, 
                        #templates=templates)
                        ##templates_std=temp_std,
                        ##weights=np.asarray([sic.shape[0] for sic in spike_index]))

        full_templates = []
        for k in range(len(spike_index)):
            indexes = np.in1d(unit_sp[:,0], spike_index[k][:,0])
            template = wf[indexes].mean(0)
            full_templates.append(template)
        
        np.savez(deconv_filename, 
                        spike_index=spike_index, 
                        templates=full_templates)
                        
        print ("**** Unit ", str(unit), ", found # clusters: ", len(spike_index))
        
    #if len(spike_index)==0:
    #    fname = (deconv_chunk_dir+"/lost_units/unit_"+str(unit).zfill(6)+'.npz')
    #    np.savez(fname, original_template=template)
    
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
        

    