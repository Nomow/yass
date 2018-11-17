# Class to do parallelized clustering

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from scipy import signal
from scipy import stats
from scipy.signal import argrelmax
from scipy.spatial import cKDTree
from copy import deepcopy
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from diptest.diptest import diptest as dp
from sklearn.cluster import AgglomerativeClustering

from yass.explore.explorers import RecordingExplorer
from yass.geometry import n_steps_neigh_channels
from yass import mfm

colors = [
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink',
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink',
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink',
'black','blue','red','green','cyan','magenta','brown','pink',
'orange','firebrick','lawngreen','dodgerblue','crimson','orchid','slateblue',
'darkgreen','darkorange','indianred','darkviolet','deepskyblue','greenyellow',
'peru','cadetblue','forestgreen','slategrey','lightsteelblue','rebeccapurple',
'darkmagenta','yellow','hotpink']

sorted_colors=colors


class Cluster(object):
    """Class for doing clustering."""

    def __init__(self, data_in):
            
        """Sets up the cluster class for each core
        Parameters: ...
              
        """
        
        # load params into correct attributes
        self.load_params(data_in)
        
        # check if clustering already done
        if os.path.exists(self.filename_postclustering): return
            
        # load raw data array
        self.load_raw_data()

        # run generational clustering on channel
        self.cluster(self.starting_indexes, self.starting_gen, 
                     self.triageflag)
                         
        # save clusters and make plots
        self.finish_clustering()
        
        
    def cluster(self, current_indexes, gen, triage_flag):

        ''' Recursive clusteringn function
            channel: current channel being clusterd
            wf = wf_PCA: denoised waveforms (# spikes, # time points, # chans)
            sic = spike_indexes of spikes on current channel
            gen = generation of cluster; increases with each clustering step        
        '''

        # Cat: TODO read from CONFIG File
        self.verbose=True
        
        # Exit if cluster too small
        if current_indexes.shape[0] < self.CONFIG.cluster.min_spikes: return
        
        # Align all chans to max chan - only in gen0
        if gen==0:
            self.align_step(gen)
        wf_align = self.wf_global[current_indexes]

        # Select active channels
        active_chans_flag = True
        if gen == 0:
            if not self.local_clustering:
                self.channel = wf_align.mean(0).ptp(0).argmax()
            #self.active_chans_step(active_chans_flag, wf_align)
            self.active_loc_step(active_chans_flag, wf_align)
        
        # Find feature channels and featurize
        pca_wf, idx_keep_feature, wf_final, feat_chans = self.featurize_step2(
                                                gen, wf_align)

        # KNN Triage and re-PCA step
        pca_wf, idx_keep = self.knn_triage_step(gen, pca_wf, wf_final, triage_flag)

        # Exit if cluster too small
        if idx_keep.shape[0] < self.CONFIG.cluster.min_spikes: return


        # subsample 10,000 spikes if not in deconv and have > 10k spikes
        pca_wf_all = pca_wf.copy() 
        pca_wf = self.subsample_step(pca_wf)

        # mfm cluster step
        vbParam, assignment = self.run_mfm3(gen, pca_wf)
        #vbParam, assignment = self.run_EM(gen, pca_wf_all)
        
        # if we subsampled then recover soft-assignments using above:
        idx_recovered, vbParam2, assignment2 = self.recover_step(gen,
                            pca_wf, vbParam, assignment, pca_wf_all)
        #else:
        #    idx_recovered = np.arange(assignment.shape[0])
        #    assignment2 = assignment
        #    vbParam2 = vbParam
            
        # Exit if cluster too small
        if idx_recovered.shape[0] < self.CONFIG.cluster.min_spikes: return

        # Note: do not compute the spike index or template until done decimating the data
        # template_current = wf_align[idx_keep][idx_recovered].mean(0)
        template_current = wf_align[idx_keep].copy()
        # sic_current = self.sic_global[current_indexes][idx_keep][idx_recovered]
        sic_current = self.sic_global[current_indexes][idx_keep].copy() 

        # make a template that in case the cluster is saved, can be used below
        # Cat: TODO: can take template to be just core of cluster 
        mc = np.argmax(template_current[idx_recovered].mean(0).ptp(0))

        # zero out all wf data arrays as not required below here
        wf = None
        wf_align = None

        '''*************************************************        
           *********** REVIEW AND SAVE RESULTS *************
           *************************************************        
        '''
        # Case #1: single mfm cluster found
        if vbParam.rhat.shape[1] == 1:
            self.single_cluster_step(mc, assignment2, pca_wf_all, vbParam2, 
                            idx_recovered, sic_current, gen, template_current, 
                            feat_chans)
            
        # Case #2: multiple clusters
        else:
            # check if any clusters are stable first
            mask = vbParam2.rhat[idx_recovered]>0
            stability = np.average(mask * vbParam2.rhat[idx_recovered], axis = 0, weights = mask)
            clusters, sizes = np.unique(assignment2[idx_recovered], return_counts = True)
            
            # if at least one stable cluster
            if np.any(stability>self.mfm_threshold):      
                self.multi_cluster_stable(gen, assignment2, idx_keep, 
                            idx_recovered, pca_wf_all, vbParam2, stability, 
                            current_indexes, sizes)

            # if no stable clusters, run spliting algorithm
            else:
                self.multi_cluster_unstable(mc, gen, assignment2, idx_keep, 
                            idx_recovered, pca_wf_all, vbParam2, stability, 
                            current_indexes, sic_current, template_current, 
                            feat_chans)
                            
                                                    
    def load_params(self, data_in):
                
        # CAT: todo read params below from file:
        self.knn_triage_threshold = 0.95 * 100
        self.knn_triage_flag = True
        self.selected_PCA_rank = 5

        # number of spikes used to compute featurization statistics
        self.n_spikes_featurize = 10000
        
        # this indicates channel-wise clustering - NOT postdeconv recluster
        self.deconv_flag = False
            
        # load params passed in
        self.channel = data_in[0]
        
        self.idx_list = data_in[1]
        self.data_start = self.idx_list[0]
        self.data_end = self.idx_list[1]
        self.offset = self.idx_list[2]
        
        self.proc_index = data_in[2]
        self.CONFIG = data_in[3]
        self.spike_indexes_chunk = data_in[4]
        self.n_dim_pca = data_in[5]
        self.n_dim_pca_compression = data_in[6]
        self.wf_start = data_in[7]
        self.wf_end = data_in[8]
        self.n_feat_chans = data_in[9]
        self.out_dir = data_in[10]
        self.mfm_threshold = data_in[11]
        self.upsample_factor = data_in[12]
        self.nshifts = data_in[13]
        self.min_spikes_local = data_in[14]
        self.standardized_filename = data_in[15]
        self.geometry_file = data_in[16]
        self.n_channels = data_in[17]
        self.local_clustering = data_in[18]

        # Check if channel alredy clustered
        self.chunk_dir = (self.CONFIG.data.root_folder+"/tmp/cluster/chunk_"+ \
                                                str(self.proc_index).zfill(6))
        
        if self.local_clustering == True:
            self.filename_postclustering = (self.chunk_dir + "/channel_"+
                                                            str(self.channel)+".npz")
        else:
            self.unit_id = data_in[0]
            self.filename_postclustering = (self.chunk_dir + "/local_unit_"+
                                                            str(self.channel)+".npz")
            self.channel = None
            

        # Cat: TODO: read all these from CONFIG
        self.spike_size = 111
        self.yscale = 1.
        self.xscale = 4.
        self.triageflag = True
        #self.alignflag = True
        self.plotting = True

        self.starting_gen = 0
        self.assignment_global = []
        self.spike_index = []
        self.templates = []
        self.feat_chans_cumulative = []
        self.shifts = []
        self.aligned_wfs_cumulative = []


    def load_raw_data(self):

        # Cat: TO DO: Is this index search expensive for hundreds of chans and many
        #       millions of spikes?  Might want to do once rather than repeat
        if self.local_clustering:
            self.indexes = np.where(self.spike_indexes_chunk[:,1]==self.channel)[0]
        else:
            self.indexes = np.where(self.spike_indexes_chunk[:,1]==self.unit_id)[0]
            
        # limit clustering to at most 50,000 spikes
        if True:
            if self.indexes.shape[0]>50000:
                idx_50k = np.random.choice(np.arange(self.indexes.shape[0]),
                                                  size=50000,
                                                  replace=False)
                self.indexes = self.indexes[idx_50k]

        # check that spkes times not too lcose to edges:
        # first determine length of processing chunk based on lenght of rec
        fp = np.memmap(self.standardized_filename, dtype='float32', mode='r')
        fp_len = fp.shape[0]/self.n_channels

        # limit indexes away from edge of recording
        idx_inbounds = np.where(np.logical_and(self.indexes>=self.spike_size//2,
                                               self.indexes<(fp_len-self.spike_size//2)))[0]
        self.indexes = self.indexes[idx_inbounds]

        # check to see if any duplicate spike times occur
        if np.unique(self.indexes).shape[0] != self.indexes.shape[0]:
            print ("   >>>>>>>>>>>>>>>>>>>>>>>> DUPLICATE SPIKE TIMES <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        self.indexes = np.unique(self.indexes)

        # set spikeindexes from all spikes
        self.sic_global = self.spike_indexes_chunk[self.indexes]

        # sets up initial array of indexes
        self.starting_indexes = np.arange(self.indexes.shape[0])

        # load raw data from disk
        self.wf_global = load_waveforms_from_disk(
                                    self.standardized_filename, 
                                    self.geometry_file,
                                    self.n_channels, 
                                    self.sic_global,
                                    self.spike_size)    

        # make sure no artifacts in data, clip to 1000
        self.wf_global = self.wf_global.clip(min=-1000, max=1000)

        # Cat: TO DO: subsampled indexes outside clustering function is
        # legacy code; remove it
        #self.indexes_subsampled=np.arange(self.wf_global.shape[0])

        # plotting parameters
        if self.plotting:
            # Cat: TO DO: this global x is not necessary, should make it local
            self.x = np.zeros(100, dtype = int)
            self.fig1 = plt.figure(figsize =(60,60))
            self.grid1 = plt.GridSpec(20,20,wspace = 0.0,hspace = 0.2)
            self.ax1 = self.fig1.add_subplot(self.grid1[:,:])

            # setup template plot
            xlim = self.CONFIG.geom[:,0].ptp(0)
            ylim = self.CONFIG.geom[:,1].ptp(0)#/float(xlim)
            self.fig2 = plt.figure(figsize =(100,max(ylim/float(xlim)*100,10)))
            self.ax2 = self.fig2.add_subplot(111)
        else:
            self.fig1 = []
            self.grid2 = []
            self.x = []
            #self.ax_t = []
            

    def align_step(self, gen):
        # align, note: aligning all channels to max chan which is appended to the end
        # note: max chan is first from feat_chans above, ensure order is preserved
        # note: don't want for wf array to be used beyond this function
        # Alignment: upsample max chan only; linear shift other chans
        
        if self.verbose:
            if self.local_clustering:
                print("chan/unit "+str(self.channel)+' gen: '+str(gen)+' # spikes: '+
                      str(self.wf_global.shape[0]))
                print ("chan "+str(self.channel)+' gen: '+str(gen)+" - aligning")
            else:
                print("chan/unit "+str(self.unit_id)+' gen: '+str(gen)+' # spikes: '+
                      str(self.wf_global.shape[0]))
                print ("Unit "+str(self.unit_id)+' gen: '+str(gen)+" - aligning")

        mc = self.wf_global.mean(0).ptp(0).argmax(0)
        best_shifts = align_get_shifts(self.wf_global[:,:,mc], self.CONFIG) 
        self.wf_global = shift_chans(self.wf_global, best_shifts, self.CONFIG)

    def robust_stds(self, wf_align):
        
        stds = np.median(np.abs(wf_align - np.median(wf_align, axis=0, keepdims=True)), axis=0)*1.4826
        return stds

    def active_chans_step(self, active_chans_flag, wf_align):
            
        if active_chans_flag:
            neighbors = n_steps_neigh_channels(self.CONFIG.neigh_channels, 1)
            if self.local_clustering:
                active_chans = np.where(neighbors[self.channel])[0]
            else:
                stds = self.robust_stds(wf_align)
                active_chans = np.where(stds.max(0) > 1.02)[0]

                active_chans = np.sort(np.unique(np.hstack((active_chans, np.where(neighbors[self.channel])[0]))))
                active_chans = np.where(self.connected_channels(active_chans, self.channel, neighbors))[0]
        else:
            active_chans = np.arange(wf_align.shape[2])

        self.active_chans = active_chans
        
    def active_loc_step(self, active_chans_flag, wf_align):
            
        if active_chans_flag:
            active_loc = np.zeros((wf_align.shape[1], wf_align.shape[2]), 'bool')
            max_timepoints = 5
            if self.local_clustering:
                neighbors = n_steps_neigh_channels(self.CONFIG.neigh_channels, 1)
                active_chans = np.where(neighbors[self.channel])[0]
                
            else:
                active_chans = np.arange(wf_align.shape[2])
            
            for chan in active_chans:
                energy = np.median(np.square(wf_align[:, :, chan]), axis=0)
                idx = np.argsort(energy)[::-1][:max_timepoints]
                idx = idx[energy[idx] > 0.5]
                active_loc[idx, chan] = 1


        else:
            active_loc = np.ones((wf_align.shape[1], wf_align.shape[2]), 'bool')
        print('******************************************')
        print(np.sum(active_loc, 0))
        self.active_loc = active_loc

    
    def featurize_step(self, gen, wf_align):
        if self.verbose:
            print("chan/unit "+str(self.channel)+' gen: '+str(gen)+' getting feat chans')
        
        # Cat: TODO: is 10k spikes enough for feat chan selection?
        # Cat: TODO: what do these metrics look like for 100 spikes!?; should we simplify for low spike count?
        feat_chans, mc, robust_stds = self.get_feat_channels_mad(
                                        wf_align[:self.n_spikes_featurize, :, self.active_chans])
        
        # featurize all spikes
        idx_keep_feature, pca_wf, wf_final = self.featurize(
                                            wf_align[:, :, self.active_chans], 
                                            robust_stds, feat_chans, mc)

        if self.verbose:
            if self.local_clustering:
                print("chan "+str(self.channel)+' gen: '+str(gen)+", feat chans: "+
                          str(self.active_chans[feat_chans[:self.n_feat_chans]]) + ", max_chan: "+ str(self.active_chans[mc]))
            else:
                print("local unit "+str(self.unit_id)+' gen: '+str(gen)+", feat chans: "+
                          str(self.active_chans[feat_chans[:self.n_feat_chans]]) + ", max_chan: "+ str(self.active_chans[mc]))
                
        # Cat: todo: we're limiting dimensions to 5; should load this from disk
        n_features_pca = 5
        pca_wf = pca_wf[idx_keep_feature][:,:n_features_pca]
        
        return pca_wf, idx_keep_feature, wf_final, feat_chans
    
    def featurize_step2(self, gen, wf_align):
        if self.verbose:
            print("chan/unit "+str(self.channel)+' gen: '+str(gen)+' getting feat chans')
        
        active_chans = np.where(np.any(self.active_loc, 0))[0]
        stds = np.zeros((wf_align.shape[1], wf_align.shape[2]))
        for chan in active_chans:
            idx = self.active_loc[:, chan] > 0
            stds[idx, chan] = np.std(wf_align[:, idx][:, :, chan], axis=0)
        
        feat_chans = np.argsort(stds.max(0))[::-1][:self.n_feat_chans]
        mc = np.mean(wf_align, 0).ptp(0).argmax()
        if not np.any(feat_chans == mc):
            feat_chans = np.hstack((feat_chans, mc))
        wf_final = np.zeros((wf_align.shape[0], 0))
        for chan in feat_chans:
            idx = self.active_loc[:, chan] > 0
            wf_final = np.concatenate((wf_final, wf_align[:, idx][:, :, chan]), axis=1)

        pca = PCA(n_components=min(self.selected_PCA_rank, wf_final.shape[1]))
        pca.fit(wf_final)
        pca_wf = pca.transform(wf_final)

        # convert boolean to integer indexes
        idx_keep_feature = np.arange(wf_final.shape[0])
        
        return pca_wf, idx_keep_feature, wf_final, feat_chans

        
    def knn_triage_step(self, gen, pca_wf, wf_final, triage_flag):
        
        if triage_flag:
            idx_keep = self.knn_triage(self.knn_triage_threshold, pca_wf)
            idx_keep = np.where(idx_keep==1)[0]
            if self.verbose:
                print("chan "+str(self.channel)+' gen: '+str(gen) + 
                      " triaged, remaining spikes "+ 
                      str(idx_keep.shape[0]))

            if idx_keep.shape[0] < self.CONFIG.cluster.min_spikes: 
                return None, np.array([])

            pca_wf = pca_wf[idx_keep]

            # rerun global compression on residual waveforms
            if True:
                pca = PCA(n_components=min(self.selected_PCA_rank, wf_final[idx_keep].shape[1]))
                pca.fit(wf_final[idx_keep])
                pca_wf = pca.transform(wf_final[idx_keep])
        
        else:
            idx_keep = np.arange(pca_wf.shape[0])
        
        return pca_wf, idx_keep

    def knn_triage(self, th, pca_wf):

        tree = cKDTree(pca_wf)
        dist, ind = tree.query(pca_wf, k=11)
        dist = np.sum(dist, 1)
    
        idx_keep1 = dist < np.percentile(dist, th)
        return idx_keep1
    
    
    def subsample_step(self, pca_wf):
        
        if not self.deconv_flag and (pca_wf.shape[0]> self.CONFIG.cluster.max_n_spikes):
            idx_subsampled = np.random.choice(np.arange(pca_wf.shape[0]),
                             size=self.CONFIG.cluster.max_n_spikes,
                             replace=False)
        
            pca_wf = pca_wf[idx_subsampled]

        return pca_wf
    
    
    def run_mfm3(self, gen, pca_wf):
        
        if self.verbose:
            print("chan "+ str(self.channel)+' gen: '+str(gen)+" - clustering ", 
                                                              pca_wf.shape)
        mask = np.ones((pca_wf.shape[0], 1))
        group = np.arange(pca_wf.shape[0])
        vbParam2 = mfm.spikesort(pca_wf[:,:,np.newaxis],
                                mask,
                                group, self.CONFIG)
        vbParam2.rhat[vbParam2.rhat < 0.1] = 0
        vbParam2.rhat = vbParam2.rhat/np.sum(vbParam2.rhat,
                                             1, keepdims=True)

        assignment2 = np.argmax(vbParam2.rhat, axis=1)
        return vbParam2, assignment2
    
    def run_em2(self, gen, pca_wf):
        max_k = 5
        old_bic_val = np.Inf
        gmm_new = GaussianMixture(n_components=max_k)
        gmm_new.fit(pca_wf)
        new_bic_val = gmm_new.aic(pca_wf)
        while (new_bic_val < old_bic_val) and (gmm_new.n_components > 1):
            gmm_old = gmm_new
            old_bic_val = np.copy(new_bic_val)
            gmm_new = GaussianMixture(n_components=gmm_old.n_components-1)
            gmm_new.fit(pca_wf)
            new_bic_val = gmm_new.aic(pca_wf)
        
        emParam = emPar()
        emParam.rhat = gmm_old.predict_proba(pca_wf)
        emParam.muhat = gmm_old.means_.transpose((1,0))[:,:,np.newaxis]
        emParam.invVhat = (gmm_old.covariances_.transpose((1,2,0)))[:, :, :, np.newaxis]
        emParam.gmm = gmm_old
        
        assignment = np.argmax(emParam.rhat, 1)
        
        return emParam, assignment
            
        
    def recover_step_em(self, gen, pca_wf, emParam, assignment, pca_wf_all):
        
        rhat_all = emParam.gmm.predict_proba(pca_wf_all)
        
        maha_dist = 1
        D = pca_wf_all.shape[1]
        idx_recovered = self.cluster_triage(emParam, pca_wf_all, D*maha_dist)
        emParam.rhat = rhat_all
        assignment2 = emParam.rhat.argmax(1)
        
        return idx_recovered, emParam, assignment2
        
    
    def cluster_triage(self, emParam, pca_wf_all, threshold):
        # k x 5 x 5
        Sig = emParam.gmm.covariances_
        invSig = np.linalg.inv(Sig)

        # k x 5
        mu = emParam.gmm.means_
        
        # n x k x 5
        x_mu = pca_wf_all[:, np.newaxis] - mu[np.newaxis]
        
        # n x k
        maha = np.sqrt(np.matmul(np.matmul(x_mu[:, :, np.newaxis], invSig[np.newaxis]), x_mu[:, :, :, np.newaxis]))[:,:,0,0]
        idx_recovered = np.where(maha.min(1) < threshold)[0]
        
        return idx_recovered
    
    def recover_step(self, gen, pca_wf, vbParam, assignment, pca_wf_all):
        # for post-deconv reclustering, we can safely cluster only 10k spikes or less
        if not self.deconv_flag and (pca_wf.shape[0] <= self.CONFIG.cluster.max_n_spikes):
            vbParam2 = deepcopy(vbParam)
            vbParam2, assignment2 = self.recover_spikes(vbParam2, 
                                                        pca_wf_all)
        else:
            vbParam2, assignment2 = vbParam, assignment

        idx_recovered = np.where(assignment2!=-1)[0]
        if self.verbose:
            print ("chan "+ str(self.channel)+' gen: '+str(gen)+" - recovered ",
                                                str(idx_recovered.shape[0]))
    
        return idx_recovered, vbParam2, assignment2
    
    def recover_spikes(self, vbParam, pca, maha_dist = 1):
    
        N, D = pca.shape
        C = 1
        maskedData = mfm.maskData(pca[:,:,np.newaxis], np.ones([N, C]), np.arange(N))
        
        vbParam.update_local(maskedData)
        assignment = mfm.cluster_triage(vbParam, pca[:,:,np.newaxis], D*maha_dist)
        
        # zero out low assignment vals
        self.recover_threshold = 0.001
        if True:
            vbParam.rhat[vbParam.rhat < self.recover_threshold] = 0
            vbParam.rhat = vbParam.rhat/np.sum(vbParam.rhat,
                                             1, keepdims=True)
            assignment = np.argmax(vbParam.rhat, axis=1)
        
        return vbParam, assignment
        
    
    def single_cluster_step(self, mc, assignment2, pca_wf_all, vbParam2, idx_recovered,
                            sic_current, gen, template_current, feat_chans):
                                
        # exclude units whose maximum channel is not on the current 
        # clustered channel; but only during clustering, not during deconv
        if mc != self.channel and (self.deconv_flag==False): 
            print ("  channel: ", self.channel, " template has maxchan: ", mc, 
                    " skipping ...")
            
            # always plot scatter distributions
            if gen<20:
                split_type = 'mfm non_max-chan'
                end_flag = 'cyan'
                self.plot_clustering_scatter(gen, 
                            assignment2[idx_recovered],
                            assignment2[idx_recovered],
                            pca_wf_all[idx_recovered],
                            vbParam2.rhat[idx_recovered],
                            split_type,
                            end_flag)
            return 
            
        else:         
            N = len(self.assignment_global)
            if self.verbose:
                if self.local_clustering:
                    print("chan "+str(self.channel)+' gen: '+str(gen)+" >>> cluster "+
                        str(N)+" saved, size: "+str(idx_recovered.shape)+"<<<")
                else:
                    print("Unit "+str(self.unit_id)+' gen: '+str(gen)+" >>> cluster "+
                        str(N)+" saved, size: "+str(idx_recovered.shape)+"<<<")
                print ("")
            
            self.assignment_global.append(N * np.ones(assignment2[idx_recovered].shape[0]))
            self.spike_index.append(sic_current[idx_recovered])
            template = np.median(template_current[idx_recovered], 0)
            self.templates.append(template)
            
            # plot template if done
            if self.plotting:
                self.plot_clustering_template(gen, template, 
                                              idx_recovered, feat_chans, N)

                # always plot scatter distributions
                if gen<20:
                    split_type = 'mfm'
                    end_flag = 'red'
                    self.plot_clustering_scatter(gen,  
                            assignment2[idx_recovered],
                            assignment2[idx_recovered],
                            pca_wf_all[idx_recovered],
                            vbParam2.rhat[idx_recovered],
                            split_type,
                            end_flag)
                
                                
    def multi_cluster_stable(self, gen, assignment2, idx_keep, idx_recovered, 
                             pca_wf_all, vbParam2, stability, current_indexes,
                             sizes):
                            
        if self.verbose:
            print("chan "+str(self.channel)+' gen: '+str(gen) + 
                  " multiple clusters, stability " + str(np.round(stability,2)) + 
                  " size: "+str(sizes))

        # always plot scatter distributions
        if self.plotting and gen<20:
            split_type = 'mfm multi split'
            self.plot_clustering_scatter(gen,  
                        assignment2[idx_recovered],
                        assignment2[idx_recovered],
                        pca_wf_all[idx_recovered],
                        vbParam2.rhat[idx_recovered],
                        split_type)
                        

        # remove stable clusters 
        for clust in np.where(stability>self.mfm_threshold)[0]:

            idx = np.where(assignment2[idx_recovered]==clust)[0]
            
            if idx.shape[0]<self.CONFIG.cluster.min_spikes: 
                continue    # cluster too small
            
            if self.verbose:
                print("chan "+str(self.channel)+' gen: '+str(gen)+
                    " reclustering stable cluster"+ 
                    str(idx.shape))
            
            triageflag = True
            #alignflag = True
            self.cluster(current_indexes[idx_keep][idx_recovered][idx], 
                         gen+1, triageflag)

        # run mfm on residual data
        idx = np.in1d(assignment2[idx_recovered], 
                                np.where(stability<=self.mfm_threshold)[0])
        if idx.sum()>self.CONFIG.cluster.min_spikes:
            if self.verbose:
                print("chan "+str(self.channel)+" reclustering residuals "+
                                        str(idx.shape))
                                        
            # recluster
            triageflag = True
            self.cluster(current_indexes[idx_keep][idx_recovered][idx], 
                         gen+1, triageflag)


    def multi_cluster_unstable(self, mc, gen, assignment2, idx_keep, 
                            idx_recovered, pca_wf_all, vbParam2, stability, 
                            current_indexes, sic_current, template_current,
                            feat_chans):
        
        # use EM algorithm for binary split for now
        EM_split = True
        dp_val, assignment3, idx_recovered = self.diptest_step(EM_split, 
                        assignment2, idx_recovered, vbParam2, pca_wf_all)
 
        # exit if clusters were decimated below threshold during diptest
        if idx_recovered.shape[0] < self.CONFIG.cluster.min_spikes: return
                
        # cluster is unimodal, save it
        # don't exit on gen0 split ever
        diptest_thresh = 1.0
        if (dp_val> diptest_thresh and gen!=0):
            self.save_step(dp_val, mc, gen, idx_recovered, pca_wf_all,
                          vbParam2, assignment2, assignment3, sic_current,
                          template_current, feat_chans)
        

        # cluster is not unimodal, split it along binary split and recluster 
        else:
            self.split_step(gen, dp_val, assignment2, assignment3, pca_wf_all, idx_recovered,
                                    vbParam2, idx_keep, current_indexes)

    def diptest_step(self, EM_split, assignment2, idx_recovered, vbParam2, pca_wf_all):
        
        if EM_split: 
            gmm = GaussianMixture(n_components=2)
        
        ctr=0 
        dp_val = 1.0
        idx_temp_keep = np.arange(idx_recovered.shape[0])
        print ("Original vbParam2.muhat.shape: ", vbParam2.muhat.shape
               )
        cluster_idx_keep = np.arange(vbParam2.muhat.shape[1])
        # loop over cluster until at least 3 loops and take lowest dp value
        while True:    
            # use EM algorithm to get binary split
            if EM_split: 
                gmm.fit(pca_wf_all[idx_recovered])
                labels = gmm.predict_proba(pca_wf_all[idx_recovered])

                temp_rhat = labels
                temp_assignment = np.zeros(labels.shape[0], 'int32')
                idx = np.where(labels[:,1]>0.5)[0]
                temp_assignment[idx]=1
            
            # use mfm algorithm to find temp-assignment
            else:
                temp_assignment = self.mfm_binary_split2(
                    vbParam2.muhat[:, cluster_idx_keep],
                    assignment2[idx_recovered],
                    cluster_idx_keep)


                
            # check if any clusters smaller than min spikes
            counts = np.unique(temp_assignment, return_counts=True)[1]

            # update indexes if some clusters too small
            if min(counts)<self.CONFIG.cluster.min_spikes:
                print ("  REMOVING SMALL CLUSTER DURING diptest")
                bigger_cluster_id = np.argmax(counts)
                idx_temp_keep = np.where(temp_assignment==bigger_cluster_id)[0]
                idx_recovered = idx_recovered[idx_temp_keep]

                # This decreases the clusters kept in muhat
                cluster_idx_keep = np.unique(assignment2[idx_recovered])
                
                # exit if cluster gets decimated below threshld
                if idx_recovered.shape[0]<self.CONFIG.cluster.min_spikes:
                    return dp_val, assignment2[idx_recovered], idx_recovered

                # if removed down to a single cluster, recluster it (i.e. send dpval=0.0)
                if cluster_idx_keep.shape[0] < 2:
                    return 0.0, assignment2[idx_recovered], idx_recovered

            # else run the unimodality test
            # Cat: todo: this is not perfect, still misses some bimodal distributions
            else:
                # test EM for unimodality
                dp_new = self.test_unimodality(pca_wf_all[idx_recovered], temp_assignment)
                
                # set initial values
                if ctr==0:
                    assignment3 = temp_assignment
                
                # search for lowest split score (not highest)
                # goal is to find most multimodal split, not unimodal
                if dp_new <dp_val:
                    dp_val= dp_new
                    assignment3 = temp_assignment

                # if at least 3 loops using EM-split, or any loop iteration for mfm
                if ctr>2 or not EM_split:
                    # need to also ensure that we've not deleted any spikes after we
                    #  saved the last lowest-dp avlue assignment
                    if assignment3.shape[0] != temp_assignment.shape[0]:
                        assignment3 = temp_assignment
                    break
                
                ctr+=1
            
        return dp_val, assignment3, idx_recovered

    def mfm_binary_split2(self, muhat, assignment_orig, cluster_index=None):

        centers = muhat[:, :, 0].T
        K, D = centers.shape
        if cluster_index is None:
            cluster_index = np.arange(K)

        label = AgglomerativeClustering(n_clusters=2).fit(centers).labels_
        assignment = np.zeros(len(assignment_orig), 'int16')
        for j in range(2):
            print (j)
            print (np.where(label == j)[0])
            #clusters = cluster_index[np.where(label == j)[0]]
            clusters = cluster_index[np.where(label == j)[0]]
            for k in clusters:
                assignment[assignment_orig == k] = j

        return assignment

    def save_step(self, dp_val, mc, gen, idx_recovered,
                          pca_wf_all, vbParam2, assignment2, assignment3,
                          sic_current, template_current, feat_chans):
                              
        # make sure cluster is on max chan, otherwise omit it
        if mc != self.channel and (self.deconv_flag==False): 
            print ("  channel: ", self.channel, " template has maxchan: ", mc, 
                    " skipping ...")
            
            # always plot scatter distributions
            if gen<20:
                split_type = 'mfm-binary - non max chan'
                end_flag = 'cyan'                       
                self.plot_clustering_scatter(gen,  
                    assignment3,
                    assignment2[idx_recovered],
                    pca_wf_all[idx_recovered],
                    vbParam2.rhat[idx_recovered],
                    split_type,
                    end_flag)
                        
            return 
        
        N= len(self.assignment_global)
        if self.verbose:
            print("chan "+str(self.channel)+' gen: '+str(gen)+" >>> cluster "+
                  str(N)+" saved, size: "+str(idx_recovered.shape)+"<<<")
        
        self.assignment_global.append(N * np.ones(assignment3.shape[0]))
        self.spike_index.append(sic_current[idx_recovered])
        template = np.median(template_current[idx_recovered],0)
        self.templates.append(template)

        # plot template if done
        if self.plotting:
            self.plot_clustering_template(gen, template, idx_recovered, 
                                         feat_chans, N)

            # always plot scatter distributions
            if gen<20:
                # hack to expand the assignments back out to size of original
                # data stream
                assignment3 = np.zeros(pca_wf_all[idx_recovered].shape[0],'int32')
                split_type = 'mfm-binary, dp: '+ str(round(dp_val,5))
                end_flag = 'green'
                self.plot_clustering_scatter(gen,  
                    assignment3,
                    assignment2[idx_recovered],
                    pca_wf_all[idx_recovered],
                    vbParam2.rhat[idx_recovered],
                    split_type,
                    end_flag)     
    
    
    def split_step(self, gen, dp_val, assignment2, assignment3, pca_wf_all,
                           idx_recovered, vbParam2, idx_keep, current_indexes):
                               
        # plot EM labeled data
        if gen<20 and self.plotting:
            split_type = 'mfm-binary, dp: '+ str(round(dp_val,5))
            self.plot_clustering_scatter(gen,  
                    assignment3,
                    assignment2[idx_recovered],
                    pca_wf_all[idx_recovered],
                    vbParam2.rhat[idx_recovered],
                    split_type)
                    

        if self.verbose:
            print("chan "+str(self.channel)+' gen: '+str(gen)+ 
                            " no stable clusters, binary split "+
                            str(idx_recovered.shape))

        # loop over binary split
        for clust in np.unique(assignment3): 
            idx = np.where(assignment3==clust)[0]
            
            if idx.shape[0]<self.CONFIG.cluster.min_spikes: continue 
            
            if self.verbose:
                print("chan "+str(self.channel)+' gen: '+str(gen)+
                    " reclustering cluster"+ str(idx.shape))
            
            # recluster
            triageflag = True
            self.cluster(current_indexes[idx_keep][idx_recovered][idx], 
                         gen+1, triageflag)
                         
    
    def finish_clustering(self,):
        # finish plotting 
        if self.plotting: 

            # finish cluster plots
            if self.local_clustering:
                self.fig1.suptitle("Channel: "+str(self.channel), fontsize=25)
                self.fig1.savefig(self.chunk_dir+"/channel_{}_scatter.png".format(self.channel))
            else:
                self.fig1.suptitle("Local Unit: "+str(self.unit_id), fontsize=25)
                self.fig1.savefig(self.chunk_dir+"/local_unit_{}_scatter.png".format(self.unit_id))
            plt.close(self.fig1)

            # finish template plots
            # plot channel numbers
            for i in range(self.CONFIG.recordings.n_channels):
                self.ax2.text(self.CONFIG.geom[i,0], self.CONFIG.geom[i,1],
                              str(i), alpha=0.4, fontsize=10)
                              
                # fill bewteen 2SUs on each channel
                self.ax2.fill_between(self.CONFIG.geom[i,0] +
                     np.arange(-61,0,1)/self.xscale, -self.yscale +
                     self.CONFIG.geom[i,1], self.yscale + self.CONFIG.geom[i,1],
                     color='black', alpha=0.05)
                
            # plot max chan with big red dot                
            self.ax2.scatter(self.CONFIG.geom[self.channel,0],
                              self.CONFIG.geom[self.channel,1], s = 2000, 
                              color = 'red')
                              
            # if at least 1 cluster is found, plot the template
            if len(self.spike_index)>0: 
                sic_temp = np.concatenate(self.spike_index, axis = 0)
                assignment_temp = np.concatenate(self.assignment_global, axis = 0)
                idx = sic_temp[:,1] == self.channel
                clusters, sizes = np.unique(assignment_temp[idx], return_counts= True)
                clusters = clusters.astype(int)

                labels=[]
                for i, clust in enumerate(clusters):
                    patch_j = mpatches.Patch(color = sorted_colors[clust%100], label = "size = {}".format(sizes[i]))
                    labels.append(patch_j)
                self.ax2.legend(handles = labels, fontsize=30)

            # plto title
            if self.local_clustering:
                self.fig2.suptitle("Channel: " + str(self.channel), fontsize=25)
                self.fig2.savefig(self.chunk_dir + "/channel_{}_template.png".format(self.channel))
            else:
                self.fig2.suptitle("Local Unit: " + str(self.unit_id), fontsize=25)
                self.fig2.savefig(self.chunk_dir + "/local_unit_{}_template.png".format(self.unit_id))
            plt.close(self.fig2)

        # Cat: TODO: note clustering is done on PCA denoised waveforms but
        #            templates are computed on original raw signal
        # recompute templates to contain full width information... 
        
        np.savez(self.filename_postclustering, 
                        spike_index=self.spike_index, 
                        templates=self.templates)
        if self.local_clustering:
            print ("**** Channel ", str(self.channel), " starting spikes: ", 
                self.wf_global.shape[0], ", found # clusters: ", 
                len(self.spike_index))
        else:
            print ("**** Lcoal Unit ", str(self.unit_id), " starting spikes: ", 
                self.wf_global.shape[0], ", found # clusters: ", 
                len(self.spike_index))
            
        # overwrite this variable just in case garbage collector doesn't
        self.wf_global = None
        

    def connected_channels(self, channel_list, ref_channel, neighbors, keep=None):
        if keep is None:
            keep = np.zeros(len(neighbors), 'bool')
        if keep[ref_channel] == 1:
            return keep
        else:
            keep[ref_channel] = 1
            chans = channel_list[neighbors[ref_channel][channel_list]]
            for c in chans:
                keep = self.connected_channels(channel_list, c, neighbors, keep=keep)
            return keep


    def get_feat_channels_mad(self, wf_align):
        '''  Function that uses MAD statistic like robust variance estimator
             to select channels
        '''
        # compute robust stds over units
        stds = self.robust_stds(wf_align)
        
        # max per channel
        std_max = stds.max(0)
        
        # order channels by largest diptest value
        feat_chans = np.argsort(std_max)[::-1]
        #feat_chans = feat_chans[std_max[feat_chans] > 1.2]

        max_chan = wf_align.mean(0).ptp(0).argmax(0)

        return feat_chans, max_chan, stds
    
    def featurize(self, wf, robust_stds, feat_chans, max_chan):
        
        # select argrelmax of mad metric greater than trehsold
        #n_feat_chans = 5

        n_features_per_channel = 2
        wf_final = np.zeros((0,wf.shape[0]), 'float32')
        # select up to 2 features from max amplitude chan;
        trace = robust_stds[:,max_chan]
        idx = argrelmax(trace, axis=0, mode='clip')[0]

        if idx.shape[0]>0:
            idx_sorted = np.argsort(trace[idx])[::-1]
            idx_thresh = idx[idx_sorted[:n_features_per_channel]]
            temp = wf[:,idx_thresh,max_chan]
            wf_final = np.vstack((wf_final, temp.T))
            #wf_final.append(wf[:,idx_thresh,max_chan])
            
        ## loop over all feat chans and select max 2 argrelmax time points as features
        n_feat_chans = np.min((self.n_feat_chans, wf.shape[2]))
        for k in range(n_feat_chans):

            # don't pick max channel again, already picked above
            if feat_chans[k]==max_chan: continue
            
            trace = robust_stds[:,feat_chans[k]]
            idx = argrelmax(trace, axis=0, mode='clip')[0]
            if idx.shape[0]>0:
                idx_sorted = np.argsort(trace[idx])[::-1]
                idx_thresh = idx[idx_sorted[:n_features_per_channel]]
                temp = wf[:,idx_thresh,feat_chans[k]]
                wf_final = np.vstack((wf_final, temp.T))

        # Cat: TODO: this may crash if weird data goes in
        #print (" len wf arra: ", len(wf_final))
        #wf_final = np.array(wf_final)
        #wf_final = wf_final.swapaxes(0,1).reshape(wf.shape[0],-1)
        wf_final = wf_final.T

        # run PCA on argrelmax points;
        # Cat: TODO: read this from config
        pca = PCA(n_components=min(self.selected_PCA_rank, wf_final.shape[1]))
        pca.fit(wf_final)
        pca_wf = pca.transform(wf_final)

        # convert boolean to integer indexes
        idx_keep_feature = np.arange(wf_final.shape[0])

        return idx_keep_feature, pca_wf, wf_final

    def featurize2(self, wf, robust_stds, feat_chans, max_chan):
        
        # select argrelmax of mad metric greater than trehsold
        #n_feat_chans = 5
        
        if not np.any(feat_chans == max_chan):
            feat_chans = np.hstack((feat_chans, max_chan))
        
        #wf_final = wf[:,:,feat_chans].reshape((wf.shape[0], -1))
        wf_final = wf.reshape((wf.shape[0], -1))
        pca = PCA(n_components=min(self.selected_PCA_rank, wf_final.shape[1]))
        pca.fit(wf_final)
        pca_wf = pca.transform(wf_final)

        # convert boolean to integer indexes
        idx_keep_feature = np.arange(wf_final.shape[0])

        return idx_keep_feature, pca_wf, wf_final

    def test_unimodality(self, pca_wf, assignment, max_spikes = 10000):
        
        n_samples = np.max(np.unique(assignment, return_counts=True)[1])

        # compute diptest metric on current assignment+LDA
        #lda = LDA(n_components = 1)
        #trans = lda.fit_transform(pca_wf[:max_spikes], assignment[:max_spikes])
        #diptest = dp(trans.ravel())
        
        ## find indexes of data
        idx1 = np.where(assignment==0)[0]
        idx2 = np.where(assignment==1)[0]
        min_spikes = min(idx1.shape, idx2.shape)[0]

        # limit size difference between clusters to maximum of 5 times
        ratio = 1
        idx1=idx1[:min_spikes*ratio][:max_spikes]
        idx2=idx2[:min_spikes*ratio][:max_spikes]

        idx_total = np.concatenate((idx1,idx2))

        ## run LDA on remaining data
        lda = LDA(n_components = 1)
        #print (pca_wf[idx_total].shape, assignment[idx_total].shape) 
        trans = lda.fit_transform(pca_wf[idx_total], assignment[idx_total])
        diptest = dp(trans.ravel())

        ## also compute gaussanity of distributions
        ## first pick the number of bins; this metric is somewhat sensitive to this
        # Cat: TODO number of bins is dynamically set; need to work on this
        #n_bins = int(np.log(n_samples)*3)
        #y1 = np.histogram(trans, bins = n_bins)
        #normtest = stats.normaltest(y1[0])

        return diptest[1] #, normtest[1]

                                
    def plot_clustering_scatter(self, gen, 
                                assignment2,
                                assignment_original,
                                pca_wf,
                                rhat,
                                split_type,
                                end_point='false'):

        if (self.x[gen]<20) and (gen <20):

            # add generation index
            ax = self.fig1.add_subplot(self.grid1[gen, self.x[gen]])
            self.x[gen] += 1

            # compute cluster memberships
            #mask = rhat>0
            #print (" mask: ", mask.shape)
            #print (" mask: ", mask)
            #print (" rhat: ", rhat.shape)
            #print (" rhat: ", rhat)
            #stability = np.average(mask * rhat, axis = 0, weights = mask)

            clusters, sizes = np.unique(assignment2, return_counts=True)
            # make legend
            labels = []
            for clust,_ in enumerate(clusters):
                patch_j = mpatches.Patch(color = sorted_colors[clust%100], 
                                        label = "size = "+str(int(sizes[clust])))
                                         #+ "stab: "+ str(np.round(stability[clust],2)))
                
                labels.append(patch_j)
            
            # make list of colors; this could be done simpler
            temp_clrs = []
            for k in assignment2:
                temp_clrs.append(sorted_colors[k])

            # make scater plots
            if pca_wf.shape[1]>1:
                ax.scatter(pca_wf[:,0], pca_wf[:,1], 
                    c = temp_clrs, edgecolor = 'k',alpha=0.1)
                
                # add red dot for converged clusters; cyan to off-channel
                if end_point!='false':
                    ax.scatter(pca_wf[:,0].mean(), pca_wf[:,1].mean(), c= end_point, s = 2000,alpha=.5)
            else:
                for clust in clusters:
                    ax.hist(pca_wf[np.where(assignment2==clust)[0]], 100)

            # finish plotting
            ax.legend(handles = labels, fontsize=5)
            ax.set_title("Spikes: "+ str(sizes.sum())+", "+split_type)
           
          
    def plot_clustering_template(self, gen, wf_mean, idx_recovered, feat_chans, N):
        
        # plot template
        #local_scale = min
        self.ax2.plot(self.CONFIG.geom[:,0]+
                  np.arange(-wf_mean.shape[0]//2,wf_mean.shape[0]//2,1)[:,np.newaxis]/self.xscale,
                  self.CONFIG.geom[:,1] + wf_mean[:,:]*self.yscale, c=colors[N%100],
                  alpha=min(max(0.4, idx_recovered.shape[0]/1000.),1))

        # plot feature channels as scatter dots
        #for i in feat_chans:
        #     self.ax_t.scatter(self.CONFIG.geom[i,0]+gen,
        #                       self.CONFIG.geom[i,1]+N,
        #                       s = 30,
        #                       color = colors[N%50],
        #                       alpha=1)




    def run_EM(self, gen, pca_wf):
        ''' Experimental idea of using EM only to do clustering step
        '''
        if self.verbose:
            print("chan "+ str(self.channel)+' gen: '+str(gen)+" - clustering ", 
                                                              pca_wf.shape)

        self.recover_threshold = 0.001

        class vbParam_ojb():
            def __init__(self):
                self.rhat = None

        # test unimodality of cluster
        dp_val = 1.0 
        for k in range(3):
            gmm = GaussianMixture(n_components=2)
            gmm.fit(pca_wf)
            labels = gmm.predict_proba(pca_wf)

            temp_rhat = labels

            # zero out low assignment vals
            temp_rhat[temp_rhat <  self.recover_threshold] = 0
            temp_rhat = temp_rhat/np.sum(temp_rhat, 1, keepdims=True)

            assignment2 = np.argmax(temp_rhat, axis=1)
            
            mask = temp_rhat>0
            stability = np.average(mask * temp_rhat, axis = 0, weights = mask)
            clusters, sizes = np.unique(assignment2, return_counts = True)

            print (" EM: comps: ", 2, "sizes: ", sizes, 
                    "stability: ", stability)
                                                                
            dp_new = self.test_unimodality(pca_wf, assignment2)
            print ("diptest: ", dp_new)
            if dp_new<dp_val:
                dp_val=dp_new
            

        # if distribution unimodal
        if dp_val >0.990: 
            assignment2[:]=0
            temp_rhat=np.ones((assignment2.shape[0],1))
            #print (temp_rhat)
            #quit()
        
        # else
        else:
            components_list = [3,4,5,6]
            for n_components in components_list:
                gmm = GaussianMixture(n_components=n_components)
                
                #ctr=0 
                #dp_val = 1.0
                #idx_temp_keep = np.arange(idx_recovered.shape[0])
                #cluster_idx_keep = np.arange(vbParam2.muhat.shape[0])

                gmm.fit(pca_wf)
                labels = gmm.predict_proba(pca_wf)

                temp_rhat = labels
                temp_assignment = np.zeros(labels.shape[0], 'int32')
                #idx = np.where(labels[:,1]>0.5)[0]
                #temp_assignment[idx]=1

            
                # zero out low assignment vals
                self.recover_threshold = 0.001
                temp_rhat[temp_rhat <  self.recover_threshold] = 0
                temp_rhat = temp_rhat/np.sum(temp_rhat, 1, keepdims=True)

                assignment2 = np.argmax(temp_rhat, axis=1)

                mask = temp_rhat>0
                stability = np.average(mask * temp_rhat, axis = 0, weights = mask)
                clusters, sizes = np.unique(assignment2, return_counts = True)

                print (" EM: comps: ", n_components, "sizes: ", sizes, 
                        "stability: ", stability)
                
                if np.any(stability>0.90):
                    break
                
        vbParam = vbParam_ojb()
        vbParam.rhat = temp_rhat

        return vbParam, assignment2



def align_get_shifts(wf, CONFIG, upsample_factor = 5, nshifts = 15):

    ''' Align all waveforms on a single channel
    
        wf = selected waveform matrix (# spikes, # samples)
        max_channel: is the last channel provided in wf 
        
        Returns: superresolution shifts required to align all waveforms
                 - used downstream for linear interpolation alignment
    '''
    
    # convert nshifts from timesamples to  #of times in upsample_factor
    nshifts = (nshifts*upsample_factor)
    if nshifts%2==0:
        nshifts+=1    
    
    # or loop over every channel and parallelize each channel:
    #wf_up = []
    wf_up = upsample_resample(wf, upsample_factor)

    wlen = wf_up.shape[1]
    wf_start = int(.2 * (wlen-1))
    wf_end = -int(.3 * (wlen-1))
    
    wf_trunc = wf_up[:,wf_start:wf_end]
    wlen_trunc = wf_trunc.shape[1]
    
    # align to last chanenl which is largest amplitude channel appended
    ref_upsampled = wf_up.mean(0)
    
    ref_shifted = np.zeros([wf_trunc.shape[1], nshifts])
    
    for i,s in enumerate(range(-int((nshifts-1)/2), int((nshifts-1)/2+1))):
        ref_shifted[:,i] = ref_upsampled[s+ wf_start: s+ wf_end]

    bs_indices = np.matmul(wf_trunc[:,np.newaxis], ref_shifted).squeeze(1).argmax(1)
    best_shifts = (np.arange(-int((nshifts-1)/2), int((nshifts-1)/2+1)))[bs_indices]

    return best_shifts
    
def upsample_resample(wf, upsample_factor):

    wf = wf.T
    waveform_len, n_spikes = wf.shape
    traces = np.zeros((n_spikes, (waveform_len-1)*upsample_factor+1),'float32')
    for j in range(wf.shape[1]):
        traces[j] = signal.resample(wf[:,j],(waveform_len-1)*upsample_factor+1)
    return traces



def shift_chans(wf, best_shifts, CONFIG):
    # use template feat_channel shifts to interpolate shift of all spikes on all other chans
    # Cat: TODO read this from CNOFIG
    upsample_factor = 5.
    wf_shifted = []
    all_shifts = best_shifts/upsample_factor
    wfs_final=[]
    for k, shift_ in enumerate(all_shifts):
        if int(shift_)==shift_:
            ceil = int(shift_)
            temp = np.roll(wf[k],ceil,axis=0)
        else:
            ceil = int(math.ceil(shift_))
            floor = int(math.floor(shift_))
            temp = np.roll(wf[k],ceil,axis=0)*(shift_-floor)+np.roll(wf[k],floor, axis=0)*(ceil-shift_)
        wfs_final.append(temp)
    wf_shifted = np.array(wfs_final)
    
    return wf_shifted
    
           
           
def load_waveforms_from_disk(standardized_file, 
                             geometry_file,
                             n_channels, 
                             spike_train, 
                             spike_size):


    # initialize rec explorer
    # Cat: TODO is there a faster version of this?!
    # Cat: TODO should parameters by flexible
    re = RecordingExplorer(standardized_file, path_to_geom = geometry_file, 
                           spike_size = 30, neighbor_radius = 100, 
                           dtype = 'float32',n_channels = n_channels, 
                           data_order = 'samples')

    spikes = spike_train[:,0]
    wf_data = re.read_waveforms(spikes)

    return (wf_data)

class emPar:
    """
        Class for all the parameters for the EM inference
    """

    def __init__(self):
        """
            Iniitalizes rhat defined above

            Parameters:
            -----------

            rhat: np.array
                Ngroup x K numpy array defined above

        """

