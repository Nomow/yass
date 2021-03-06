# FIXME: this file needs refactoring
import logging

import numpy as np
import scipy.special as specsci
import math
from numpy.random import dirichlet
import scipy.spatial as ss
from numpy.random import dirichlet
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class maskData:
    """
        Class for creating masked virtual data

        Attributes:
        -----------

        sumY: np.array
            Ngroup x nfeature x nchannel anumpy array which contain the sum of
            the expectation of the
             masked virtual data for each group. Here Ngroup is number of
             unique groups given by the
             coresetting. nfeature is the number of features and nchannel is
             the number of channels.
        sumYSq: np.array
            Ngroup x nfeature x nchannel numpy array which contain the sum of
            the expectation of yy.T
            where y = M * score for each group.
        sumEta: np.array
            Ngroup x nfeature x nchannel numpy array sum of the variance for
            all points in a group.
        weight: np.array
            Ngroup x 1 numpy arraay which contains in the number of points in
            each group
        meanY: np.array
            Ngroup x nfeature x nchannel numpy array which is sumY/weight or
            the empirical mean of sumY
        meanYSq: np.array
            Ngroup x nfeature x nfeature x nchannel numpy array which is
            sumYSq/weight or the empirical
            mean of sumYSq
        meanEta: np.array
            Ngroup x nfeature x nfeature x nchannel numpy array whihc is
            sumEta/weight or the empirical
            mean of sumEta
        groupmask: np.array
            Ngroup x nchannel emprical average of the mask for the datapoints
            for a given group
    """

    def __init__(self, *args):
        """
            Initialization of class attributes. Class method
            calc_maskedData_mfm() is called to actually
            calculate the attributes.

            Parameters
            ----------
            score: np.array
                N x nfeature x nchannel numpy array, where N is the number of
                spikes, nfeature is the
                number of features and nchannel is the number of channels.
                Contains multichannel spike
                data in a low dimensional space.
            mask:  np.array
                N x nchannel numpy array, where N is the number of spikes,
                nchannel is the number of
                channels. Mask for the data.
            group: np.array
                N x 1 numpy array, where N is the number of spikes.
                Coresetting group assignments for
                each spike

        """
        if len(args) > 0:
            self.calc_maskedData_mfm(args[0], args[1], args[2])

    def calc_maskedData_mfm(self, score, mask, group):
        """
            Calculation of class attributes happen here.


            Parameters
            ----------
            score: np.array
                N x nfeature x nchannel numpy array, where N is the number of
                spikes, nfeature is the
                number of features and nchannel is the number of channels.
                Contains multichannel spike
                data in a low dimensional space.
            mask:  np.array
                N x nchannel numpy array, where N is the number of spikes,
                nchannel is the number of
                channels. Mask for the data.
            group: np.array
                N x 1 numpy array, where N is the number of spikes.
                Coresetting group assignments for
                each spike
        """
        N, nfeature, nchannel = score.shape
        uniqueGroup = np.unique(group)
        Ngroup = uniqueGroup.size
        y = mask[:, np.newaxis, :] * score

        y_temp = y[:, :, np.newaxis, :]
        ySq = np.matmul(
            np.transpose(y_temp, [0, 3, 1, 2]),
            np.transpose(y_temp, [0, 3, 2, 1])).transpose((0, 2, 3, 1))

        score_temp = score[:, :, np.newaxis, :]
        scoreSq = np.matmul(
            np.transpose(score_temp, [0, 3, 1, 2]),
            np.transpose(score_temp, [0, 3, 2, 1])).transpose((0, 2, 3, 1))
        z = mask[:, np.newaxis, np.newaxis, :] * scoreSq + \
            (1 - mask)[:, np.newaxis, np.newaxis, :] * \
            (np.eye(nfeature)[np.newaxis, :, :, np.newaxis])
        eta = z - ySq

        if Ngroup == N:
            sumY = y
            sumYSq = ySq
            sumEta = eta
            groupMask = mask
            weight = np.ones(N)

        elif Ngroup < N:

            sumY = np.zeros((Ngroup, nfeature, nchannel))
            sumYSq = np.zeros((Ngroup, nfeature, nfeature, nchannel))
            sumEta = np.zeros((Ngroup, nfeature, nfeature, nchannel))
            groupMask = np.zeros((Ngroup, nchannel))
            weight = np.zeros(Ngroup)
            for n in range(N):
                idx = group[n]
                sumY[idx] += y[n]
                sumYSq[idx] += ySq[n]
                sumEta[idx] += eta[n]
                groupMask[idx] += mask[n]
                weight[idx] += 1

        else:
            raise ValueError(
                "Number of groups is larger than the size of the data")
        # self.y = y
        self.sumY = sumY
        self.sumYSq = sumYSq
        self.sumEta = sumEta
        self.weight = weight
        self.groupMask = groupMask / self.weight[:, np.newaxis]
        self.meanY = self.sumY / self.weight[:, np.newaxis, np.newaxis]
        self.meanYSq = self.sumYSq / \
            self.weight[:, np.newaxis, np.newaxis, np.newaxis]
        self.meanEta = self.sumEta / \
            self.weight[:, np.newaxis, np.newaxis, np.newaxis]


class vbPar:
    """
        Class for all the parameters for the VB inference

        Attributes:
        -----------

        rhat: np.array
            Ngroup x K numpy array containing the probability of each
            representative point of being assigned
            to cluster  0<=k<K. Here K is the number of clusters

        ahat: np.array
            K x 1 numpy array. Posterior dirichlet parameters
        lambdahat, nuhat: np.array
            K x 1 numpy array. Posterior Normal wishart parameters
        muhat, Vhat, invVhat: np.array
            nfeaature x K x nchannel, nfeature x nfeature x K x nchannel,
            nfeature x nfeature x K x nchannel
            respectively. Posterior parameters for the normal wishart
            distribution
    """

    def __init__(self, rhat):
        """
            Iniitalizes rhat defined above

            Parameters:
            -----------

            rhat: np.array
                Ngroup x K numpy array defined above

        """

        self.rhat = rhat

    def update_local(self, maskedData):
        """
            Updates the local parameter rhat for VB inference

            Parameters:
            -----------

            maskedData: maskData object
        """

        pik = dirichlet(self.ahat.ravel())
        Khat = self.ahat.size
        Ngroup, nfeatures, nchannel = maskedData.meanY.shape



        const1 = -nfeatures / 2 * np.log(2 * np.pi)
        prec = self.Vhat.transpose([2, 3, 0, 1]) * self.nuhat[:, np.newaxis, np.newaxis, np.newaxis]
        xmu = (maskedData.meanY[:, :, np.newaxis] - self.muhat).transpose([0, 2, 3, 1])
        maha = -np.squeeze(np.matmul(xmu[:, :, :, np.newaxis], np.matmul(prec, xmu[..., np.newaxis])), axis=(3, 4))/2.0
        const2 = np.linalg.slogdet(prec)[1] / 2.0
        log_rho = np.sum(maha + const1 + const2, axis=-1)
        log_rho += np.log(pik)
        log_rho = log_rho - np.max(log_rho, axis=1)[:, np.newaxis]
        rho = np.exp(log_rho)
        self.rhat = rho / np.sum(rho, axis=1, keepdims=True)

    def update_global(self, suffStat, param):
        """
            Updates the global variables muhat, invVhat, Vhat, lambdahat,
            nuhat, ahat for VB inference

            Parameters:
            ----------

            suffStat: suffStatistics object

            param: Config object (See config.py for details)
        """
        prior = param.cluster.prior
        nfeature, Khat, nchannel = suffStat.sumY.shape
        self.ahat = prior.a + suffStat.Nhat
        self.lambdahat = prior.lambda0 + suffStat.Nhat
        self.muhat = suffStat.sumY / self.lambdahat[:, np.newaxis]
        invV = np.eye(nfeature) / prior.V
        self.invVhat = np.zeros([Khat, nchannel, nfeature, nfeature])
        self.invVhat2 = np.zeros([Khat, nchannel, nfeature, nfeature])
        muhat_temp = np.transpose(self.muhat, [1,2,0])
        sumY_temp = np.transpose(suffStat.sumY, [1,2,0])

        self.invVhat += invV + self.lambdahat[:, np.newaxis, np.newaxis, np.newaxis] * \
                            np.matmul(muhat_temp[..., np.newaxis], muhat_temp[:, :, np.newaxis])
        temp = np.matmul(muhat_temp[..., np.newaxis], sumY_temp[:,:,np.newaxis])
        self.invVhat += - temp - temp.transpose([0,1,3,2])
        self.invVhat += suffStat.sumYSq.transpose([2, 3, 0, 1])
        self.Vhat = np.linalg.solve(self.invVhat, np.eye(nfeature))
        self.invVhat = self.invVhat.transpose([2,3,0,1])
        self.Vhat = self.Vhat.transpose([2,3,0,1])
        self.nuhat = prior.nu + suffStat.Nhat

    def update_global_selected(self, suffStat, param):
        """
            Updates the global variables muhat, invVhat, Vhat, lambdahat,
            nuhat, ahat for VB inference for
            a given cluster (Unused; needs work)

            Parameters:
            ----------

            suffStat: suffStatistics object

            param: Config object (See config.py for details)
        """
        prior = param.cluster.prior
        nfeature, Khat, nchannel = suffStat.sumY.shape
        self.ahat = prior.a + suffStat.Nhat
        self.lambdahat = prior.lambda0 + suffStat.Nhat
        self.muhat = suffStat.sumY / self.lambdahat[:, np.newaxis]
              
        

        invV = np.eye(nfeature) / prior.V
        self.Vhat = np.zeros([nfeature, nfeature, Khat, nchannel])
        self.invVhat = np.zeros([nfeature, nfeature, Khat, nchannel])
        for n in range(nchannel):
            for k in range(Khat):
                self.invVhat[:, :, k, n] = self.invVhat[:, :, k, n] + invV
                self.invVhat[
                    :, :, k, n] = self.invVhat[
                    :, :, k, n] + self.lambdahat[k] * np.dot(
                    self.muhat[:, np.newaxis, k, n],
                    self.muhat[:, np.newaxis, k, n].T)
                temp = np.dot(self.muhat[:, np.newaxis, k, n],
                              suffStat.sumY[:, np.newaxis, k, n].T)
                self.invVhat[
                    :, :, k, n] = self.invVhat[:, :, k, n] - temp - temp.T
                self.invVhat[:, :, k, n] = self.invVhat[
                    :, :, k, n] + suffStat.sumYSq[:, :, k, n]
                self.Vhat[:, :, k, n] = np.linalg.solve(
                    np.squeeze(self.invVhat[:, :, k, n]), np.eye(nfeature))
        self.nuhat = prior.nu + suffStat.Nhat


class suffStatistics:
    """
        Class to calculate precompute sufficient statistics for increased
        efficiency

        Attributes:
        -----------

        Nhat : np.array
            K x 1 numpy array which stores pseudocounts for the number of
            elements in each cluster. K
            is the number of clusters
        sumY : np.array
            nfeature x K x nchannel which stores weighted sum of the
            maskData.sumY weighted by the cluster
            probabilities. (See maskData for more details)
        sumYSq1: np.array
            nfeaature x nfeature x K x nchannel which stores weighted sum of
            the maskData.sumYSq weighted
            by the cluster probabilities. (See maskData for more details)
        sumYSq2: np.array
            nfeaature x nfeature x K x nchannel which stores weighted sum of
            the maskData.sumEta weighted
            by the cluster probabilities. (See maskData for more details)
        sumYSq: np.array
            nfeature x nfeature x K x nchannel. Stores sumYSq1 + sumYSq2.
    """

    def __init__(self, *args):
        """
            Initializes the above attributes and calls calc_suffstat().

            Parameters:
            -----------
            maskedData: maskData object

            vbParam: vbPar object

                or

            suffStat: suffStatistics object
        """

        if len(args) == 2:
            maskedData, vbParam = args
            Khat = vbParam.rhat.shape[1]
            Ngroup, nfeature, nchannel = maskedData.sumY.shape
            self.Nhat = np.sum(
                vbParam.rhat * maskedData.weight[:, np.newaxis], axis=0)
            self.sumY = np.zeros([nfeature, Khat, nchannel])
            self.sumYSq = np.zeros([nfeature, nfeature, Khat, nchannel])
            self.sumYSq1BS = np.zeros(
                [Ngroup, nfeature, nfeature, Khat, nchannel])
            self.sumYSq1 = np.zeros([nfeature, nfeature, Khat, nchannel])
            self.sumYSq2 = np.zeros([nfeature, nfeature, Khat, nchannel])
            self.calc_suffstat(maskedData, vbParam, Ngroup, Khat, nfeature,
                               nchannel)
        elif len(args) == 1:
            self.Nhat = args[0].Nhat.copy()
            self.sumY = args[0].sumY.copy()
            self.sumYSq = args[0].sumYSq.copy()
            self.sumYSq1 = args[0].sumYSq1.copy()
            self.sumYSq2 = args[0].sumYSq2.copy()

    def calc_suffstat(self, maskedData, vbParam, Ngroup, Khat, nfeature,
                      nchannel):
        """
            Calcation of the above attributes happens here. Called by
            __init__().

            Parameters:
            -----------
            maskedData: maskData object

            vbParam: vbPar object

            Ngroup: int
                Number of groups defined by coresetting

            Khat: int
                Number of clusters

            nfeature: int
                Number of features

            nchannel: int
                Number of channels
        """

        # noMask = maskedData.groupMask > 0
        # nnoMask = noMask.sum(0)
        # ind1 = nnoMask == 0
        # self.sumYSq[:,:,:,ind1] = np.eye(nfeature)[:,:,np.newaxis, np.newaxis] * self.Nhat[np.newaxis,np.newaxis,:, np.newaxis]
        # ind2 = nnoMask < Ngroup
        # maskedEta = np.eye(nfeature)
        # unmaskedY = maskedData.sumY[noMask]


        for n in range(nchannel):
            noMask = maskedData.groupMask[:, n] > 0
            nnoMask = np.sum(noMask)
            if nnoMask == 0:
                for k in range(Khat):
                    self.sumYSq[:, :, k, n] = np.eye(nfeature) * self.Nhat[k]

            elif nnoMask < Ngroup:
                maskedEta = np.eye(nfeature)
                unmaskedY = maskedData.sumY[noMask, :, n]
                unmaskedsumYSq = maskedData.sumYSq[noMask, :, :, n]
                unmaskedEta = maskedData.sumEta[noMask, :, :, n]
                unmaskedWeight = maskedData.weight[noMask]

                rhat = vbParam.rhat[noMask, :]
                self.sumY[:, :, n] = np.dot(unmaskedY.T, rhat)

                visibleCluster = np.sum(rhat, axis=0) > 1e-10
                sumMaskedRhat = self.Nhat - \
                    np.sum(rhat * unmaskedWeight[:, np.newaxis],
                           axis=0)
                self.sumYSq2[:, :, :, n] = np.sum(
                    rhat[:, np.newaxis, np.newaxis, :]
                    * unmaskedEta[:, :, :, np.newaxis], axis=0) + \
                    sumMaskedRhat * maskedEta[:, :, np.newaxis]
                self.sumYSq1[:, :, visibleCluster, n] = np.sum(
                    rhat[:, np.newaxis, np.newaxis, visibleCluster] *
                    unmaskedsumYSq[:, :, :, np.newaxis],
                    axis=0)
                self.sumYSq[:, :, :, n] = self.sumYSq1[
                    :, :, :, n] + self.sumYSq2[:, :, :, n]

            elif nnoMask == Ngroup:
                unmaskedY = maskedData.sumY[:, :, n]
                unmaskedsumYSq = maskedData.sumYSq[:, :, :, n]
                unmaskedEta = maskedData.sumEta[:, :, :, n]

                rhat = vbParam.rhat[noMask, :]
                self.sumY[:, :, n] = np.dot(unmaskedY.T, rhat)

                visibleCluster = np.sum(rhat, axis=0) > 1e-10
                sumMaskedRhat = self.Nhat - \
                    np.sum(rhat * maskedData.weight[:, np.newaxis],
                           axis=0)
                self.sumYSq2[:, :, :, n] = np.sum(
                    rhat[:, np.newaxis, np.newaxis, :] *
                    unmaskedEta[:, :, :, np.newaxis],
                    axis=0)
                self.sumYSq1[:, :, visibleCluster, n] = np.sum(
                    rhat[:, np.newaxis, np.newaxis, visibleCluster] *
                    unmaskedsumYSq[:, :, :, np.newaxis],
                    axis=0)
                self.sumYSq[:, :, :, n] = self.sumYSq1[
                    :, :, :, n] + self.sumYSq2[:, :, :, n]


class ELBO_Class:
    """
        Class for calculating the ELBO for VB inference

        Attributes:
        -----------

        percluster: np.array
            K x 1 numpy array containing part of ELBO value that depends on
            each cluster. Can be used to calculate
            value for only a given cluster

        rest_term: float
            Part of ELBO value that has to be calculated regardless of cluster

        total: float
            Total ELBO total = sum(percluster) + rest_term
    """

    def __init__(self, *args):
        """
            Initializes attributes. Calls cal_ELBO_opti()

            Parameters:
            -----------
            maskedData: maskData object

            suffStat: suffStatistics object

            vbParam: vbPar object

            param: Config object (see Config.py)

            K_ind (optional): list
                Cluster indices for which the partial ELBO is being
                calculated. Defaults to all clusters
        """
        if len(args) == 1:
            self.total = args[0].total
            self.percluster = args[0].percluster
            self.rest_term = args[0].rest_term
        else:
            self.calc_ELBO_Opti(args)

    def calc_ELBO_Opti(self, *args):
        """
            Calculates partial (or total) ELBO for the given cluster indices

            Parameters:
            -----------
            maskedData: maskData object

            suffStat: suffStatistics object

            vbParam: vbPar object

            param: Config object (see Config.py)

            K_ind (optional): list
                Cluster indices for which the partial ELBO is being
                calculated. Defaults to all clusters
        """

        if len(args[0]) < 5:
            maskedData, suffStat, vbParam, param = args[0]
            nfeature, Khat, nchannel = vbParam.muhat.shape
            P = nfeature * nchannel
            k_ind = np.arange(Khat)
        else:
            maskedData, suffStat, vbParam, param, k_ind = args[0]
            nfeature, Khat, nchannel = vbParam.muhat.shape
            P = nfeature * nchannel

        prior = param.cluster.prior
        fit_term = np.zeros(Khat)
        bmterm = np.zeros(Khat)
        # entropy_term = np.zeros(Khat)
        rhatp = vbParam.rhat[:, k_ind]
        muhat = np.transpose(vbParam.muhat, [1, 2, 0])
        Vhat = np.transpose(vbParam.Vhat, [2, 3, 0, 1])
        sumY = np.transpose(suffStat.sumY, [1, 2, 0])
        logdetVhat = np.sum(np.linalg.slogdet(Vhat)[1], axis=1, keepdims=False)

        constants = Khat * np.log(prior.beta) - prior.beta - np.log(np.arange(Khat)+1).sum() - specsci.gammaln(vbParam.ahat.sum()) + specsci.gammaln(prior.a * Khat) - Khat * specsci.gammaln(prior.a) - prior.nu * nfeature * Khat * nchannel/2.0 * np.log(prior.V) - Khat * nchannel * specsci.multigammaln(prior.nu/2.0, nfeature) - nfeature * nchannel * np.log(np.pi) /2.0 * rhatp.sum()
        kvarying = specsci.gammaln(vbParam.ahat) + nfeature * nchannel/2.0 * np.log(prior.lambda0/vbParam.lambdahat) + logdetVhat * vbParam.nuhat/2.0 + nchannel * specsci.multigammaln(vbParam.nuhat/2.0, nfeature)
        ikvarying = -rhatp * np.log(rhatp + 1e-200)

        self.total = constants + kvarying.sum() + ikvarying.sum()




def multivariate_normal_logpdf(x, mu, Lam):
    """
        Calculates the gaussian density of the given point(s). returns N x 1
        array which is the density for
        the given cluster (see vbPar.update_local())

        Parameters:
        -----------
        x: np.array
            N x nfeature x nchannel where N is the number of datapoints
            nfeature is the number of features
            and nchannel is the number of channels

        mu: np.array
            nfeature x nchannel numpy array. channelwise mean of the gaussians

        cov: np.array
            nfeature x nfeauter x nchannel numpy array. Channelwise covariance
            of the gaussians

    """

    p, C = mu.shape

    xMinusMu = np.transpose((x - mu), [2, 0, 1])
    maha = -0.5 * np.sum(
        np.matmul(xMinusMu, np.transpose(Lam, [2, 0, 1])) * xMinusMu,
        axis=(0, 2))

    const = -0.5 * p * C * np.log(2 * math.pi)

    logpart = 0

    for c in range(C):
        logpart = logpart + logdet(Lam[:, :, c])
    logpart = logpart * 0.5

    return maha + const + logpart


def logdet(X):
    """
        Calculates log of the determinant of the given symmetric positive
        definite matrix. returns float

        parameters:
        -----------
        X: np.array
            M x M. Symmetric positive definite matrix

    """

    L = np.linalg.cholesky(X)
    return 2 * np.sum(np.log(np.diagonal(L)))


def mult_psi(x, d):
    """
        Calculates the multivariate digamma function. Returns N x 1 array of
        the multivariaate digamma values

        parameters:
        -----------
        x: np.array
            M x 1 array containing the
    """
    v = x - np.asarray([range(d)]) / 2.0
    u = np.sum(specsci.digamma(v), axis=1)
    return u[:, np.newaxis]


def init_param(maskedData, K, param):
    """
        Initializes vbPar object using weighted kmeans++ for initial cluster
        assignment. Calculates sufficient
        statistics. Updates global parameters for the created vbPar object.

        Parameters:
        -----------
        maskedData: maskData object

        K: int
            Number of clusters for weighted kmeans++

        param: Config object (see config.py)

    """
    N, nfeature, nchannel = maskedData.sumY.shape

    data = np.copy(maskedData.meanY.reshape(
        [N, nfeature * nchannel], order='F').T)
    data /= np.std(data, 1)[:, np.newaxis]
    
    # old method using manually written kmeans++
    allocation = weightedKmeansplusplus(
        data,
        maskedData.weight, K)
    #print (allocation.shape)
    
    # new method relies on sklearn kmeans
    #print (np.max(data))
    #kmeans = KMeans(n_clusters=K, random_state=0).fit(data.T)
    #allocation = kmeans.labels_
    
    if N < K:
        rhat = np.zeros([N, N])
    else:
        rhat = np.zeros([N, K])
    rhat[np.arange(N), allocation] = 1
    vbParam = vbPar(rhat)
    suffStat = suffStatistics(maskedData, vbParam)
    vbParam.update_global(suffStat, param)
    # vbParam.update_local(maskedData)
    # suffStat = suffStatistics(maskedData, vbParam)
    return vbParam, suffStat


def weightedKmeansplusplus(X, w, k):
    """
        X = data (n_features, #spikes)
        w = weight (vector of 1s, len=#spikes)
        k = # of clusters requested?

    """
    L = np.asarray([])
    L1 = 0
    p = w ** 2 / np.sum(w ** 2)
    n = X.shape[1]
    ctr_outer = 0
    # L is boolean assignment
    # C is cluster centre (float)
    while np.unique(L).size != k:
        #print ("ctr_outer: ", ctr_outer, L, L1)
        ctr_outer+=1
        ii = np.random.choice(np.arange(n), size=1, replace=True, p=p)
        C = X[:, ii]
        L = np.zeros([1, n]).astype(int)
        for i in range(1, k):
            D = X - C[:, L.ravel()]
            D = np.sum(D * D, axis=0) #L2 dist
            if np.max(D) == 0:
                # C[:, i:k] = X[:, np.ones([1, k - i + 1]).astype(int)]
                return L    # single point
            D = D / np.sum(D)
            ii = np.random.choice(np.arange(n), size=1, replace=True, p=D)
            C = np.concatenate((C, X[:, ii]), axis=1)
            L = np.argmax(
                2 * np.dot(C.T, X) - np.sum(C * C, axis=0)[:, np.newaxis],
                axis=0)
        
        # compute until convergence, i.e. no changes in membership
        ctr_inner=0
        while np.any(L != L1):
        #for m in range(10): #while np.any(L != L1):
            #print ("ctr_inner: ", ctr_inner)
            ctr_inner+=1
            L1 = L
            for i in range(k):
                ll = L == i
                
                # recompute centres
                if np.sum(ll) > 0:
                    C[:, i] = np.dot(X[:, ll], w[ll] / np.sum(w[ll]))
            
            # recompute boolean assignments
            L = np.argmax(
                2 * np.dot(C.T, X) - np.sum(C * C, axis=0)[:, np.newaxis],
                axis=0)
        #if ctr_inner>10: print ("ctr_inner: ", ctr_inner)
    #if ctr_outer>10: print ("ctr_outer: ", ctr_outer)
    return L


def birth_move(maskedData, vbParam, suffStat, param, L):

    Khat = suffStat.sumY.shape[1]
    collectionThreshold = 0.1
    # extraK = param.cluster.n_split
    extraK = 5
    cluster_picked = np.ones(vbParam.rhat.shape[1], dtype = bool)
    
    if np.any(np.sum(vbParam.rhat > collectionThreshold, 0) >= extraK):
        weight = (suffStat.Nhat) * L ** 2
        weight = weight / np.sum(weight)
        idx = np.zeros(1).astype(int)

        # pick until have as many points as 
        birth_move_ctr = 0
        clusters = np.arange(Khat).astype(int)
        while np.sum(idx) < extraK:
            kpicked = np.random.choice(clusters[cluster_picked], p=weight[cluster_picked])
            cluster_picked[kpicked] = False
            if np.sum(cluster_picked)>0:
                weight = weight/np.sum(weight[cluster_picked])
                 
            # kpicked = np.argmax(weight)
            #print(kpicked)
            idx = vbParam.rhat[:, kpicked] > collectionThreshold
            birth_move_ctr+=1
            
        #print ("birth_move_ctr:", birth_move_ctr)

        idx = np.where(idx)[0]
        if idx.size > 10000:
            idx = idx[:10000]
        L = L * 2
        L[kpicked] = 1

        # Creation
        maskedDataPrime = maskData()
        maskedDataPrime.sumY = maskedData.sumY[idx, :, :]
        maskedDataPrime.sumYSq = maskedData.sumYSq[idx, :, :, :]
        maskedDataPrime.sumEta = maskedData.sumEta[idx, :, :, :]
        maskedDataPrime.groupMask = maskedData.groupMask[idx, :]
        maskedDataPrime.weight = maskedData.weight[idx]
        maskedDataPrime.meanY = maskedData.meanY[idx, :, :]
        maskedDataPrime.meanEta = maskedData.meanEta[idx, :, :, :]
        vbParamPrime, suffStatPrime = init_param(maskedDataPrime,
                                                 extraK, param)

        for iter_creation in range(5):
            vbParamPrime.update_local(maskedDataPrime)
            suffStatPrime = suffStatistics(maskedDataPrime, vbParamPrime)
            vbParamPrime.update_global(suffStatPrime, param)

        A = np.ones(Khat, dtype = 'bool')
        A[kpicked] = 0

        vbParam.ahat = np.concatenate(
            (vbParam.ahat[A], vbParamPrime.ahat), axis=0)
        vbParam.lambdahat = np.concatenate(
            (vbParam.lambdahat[A], vbParamPrime.lambdahat), axis=0)
        vbParam.muhat = np.concatenate(
            (vbParam.muhat[:,A], vbParamPrime.muhat), axis=1)
        vbParam.Vhat = np.concatenate(
            (vbParam.Vhat[:,:,A], vbParamPrime.Vhat), axis=2)
        vbParam.invVhat = np.concatenate(
            (vbParam.invVhat[:,:,A], vbParamPrime.invVhat), axis=2)
        vbParam.nuhat = np.concatenate(
            (vbParam.nuhat[A], vbParamPrime.nuhat), axis=0)

        vbParam.update_local(maskedData)
        suffStat = suffStatistics(maskedData, vbParam)
        vbParam.update_global(suffStat, param)
        nbrith = vbParamPrime.rhat.shape[1]
        L = np.concatenate((L[A], np.ones(nbrith)), axis=0)

    return vbParam, suffStat, L


def merge_move(maskedData, vbParam, suffStat, param, L, check_full):
    n_merged = 0
    ELBO = ELBO_Class(maskedData, suffStat, vbParam, param)
    nfeature, K, nchannel = vbParam.muhat.shape

    if K > 1:
        all_checked = 0
    else:
        all_checked = 1
    ctr_merge_move_outer = 0
    while (not all_checked) and (K > 1):
        prec = np.transpose(
            vbParam.Vhat * vbParam.nuhat[
                np.newaxis, np.newaxis, :, np.newaxis],
            axes=[2, 3, 0, 1])
        diff_mhat = (vbParam.muhat[:, np.newaxis] -
                     vbParam.muhat[:, :, np.newaxis]).transpose(1, 2, 3, 0)
        maha = np.sum(np.matmul(np.matmul(
            diff_mhat[:, :, :, np.newaxis, :], prec),
                                diff_mhat[:, :, :, :, np.newaxis]),
                      (2, 3, 4))

        maha[np.arange(K), np.arange(K)] = np.Inf
        merged = 0
        threshold = np.max(np.min(maha, 0))
        ctr_merge_move_inner = 0
        while np.min(maha) < threshold and merged == 0:
            closeset_pair = np.where(maha == np.min(maha))
            ka = closeset_pair[0][0]
            kb = closeset_pair[1][0]
            maha[ka, kb] = np.inf
            if np.argmin(maha[kb, :]).ravel()[0] == ka:
                maha[kb, ka] = np.inf

            vbParam, suffStat, merged, L, ELBO = check_merge(
                maskedData, vbParam, suffStat, ka, kb, param, L, ELBO)
            if merged:
                n_merged += 1
                K -= 1
            
            ctr_merge_move_inner+=1

        if not merged:
            all_checked = 1
        #if ctr_merge_move_inner>10: 
        #    print ("ctr_merge_move_inner: ",ctr_merge_move_inner)
        ctr_merge_move_outer+=1
    
    #if ctr_merge_move_outer>10:
     #   print ("ctr_merge_move_outer: ",ctr_merge_move_outer)

    return vbParam, suffStat, L


def check_merge(maskedData, vbParam, suffStat, ka, kb, param, L, ELBO):
    K = vbParam.rhat.shape[1]
    no_kab = np.ones(K).astype(bool)
    no_kab[[ka, kb]] = False
    ELBO_bmerge = ELBO.total
    
    vbParamTemp = vbPar(
        np.concatenate(
            (vbParam.rhat[:, no_kab],
             np.sum(vbParam.rhat[:, [ka, kb]], axis=1, keepdims=True)),
            axis=1))
    suffStatTemp = suffStatistics()
    suffStatTemp.Nhat = np.append(suffStat.Nhat[no_kab],
                                  np.sum(suffStat.Nhat[[ka, kb]]))
    suffStatTemp.sumY = np.concatenate(
        (suffStat.sumY[:, no_kab, :],
         np.sum(suffStat.sumY[:, (ka, kb), :], axis=1, keepdims=True)),
        axis=1)
    suffStatTemp.sumYSq = np.concatenate(
        (suffStat.sumYSq[:, :, no_kab, :],
         np.sum(suffStat.sumYSq[:, :, (ka, kb), :], axis=2, keepdims=True)),
        axis=2)
    suffStatTemp.sumYSq1 = np.concatenate(
        (suffStat.sumYSq1[:, :, no_kab, :],
         np.sum(suffStat.sumYSq1[:, :, (ka, kb), :], axis=2, keepdims=True)),
        axis=2)
    suffStatTemp.sumYSq2 = np.concatenate(
        (suffStat.sumYSq2[:, :, no_kab, :],
         np.sum(suffStat.sumYSq2[:, :, (ka, kb), :], axis=2, keepdims=True)),
        axis=2)

    vbParamTemp.update_global(suffStatTemp, param)
    # vbParamTemp.update_local(maskedData)
    # suffStatTemp = suffStatistics(maskedData, vbParamTemp)


    ELBO_amerge = ELBO_Class(maskedData, suffStatTemp, vbParamTemp, param)
    # print(ka,kb,ELBO_amerge.total, ELBO_bmerge)
    if ELBO_amerge.total < ELBO_bmerge:
        merged = 0
        return vbParam, suffStat, merged, L, ELBO
    else:
        merged = 1
        d = np.asarray([np.min(L[[ka, kb]])])
        L = np.concatenate((L[no_kab], d), axis=0)

        if L.size == 1:
            L = np.asarray([1])
        return vbParamTemp, suffStatTemp, merged, L, ELBO_amerge


def spikesort(score, mask, group, param):
    maskedData = maskData(score, mask, group)

    vbParam = split_merge(maskedData, param)

    # assignmentTemp = np.argmax(vbParam.rhat, axis=1)

    # assignment = np.zeros(score.shape[0], 'int16')
    # for j in range(score.shape[0]):
    #     assignment[j] = assignmentTemp[group[j]]

    # idx_triage = cluster_triage(vbParam, score, 20)
    # assignment[idx_triage] = -1

    # return assignment, vbParam
    return vbParam


def split_merge(maskedData, param):
    vbParam, suffStat = init_param(maskedData, 1, param)
    iter = 0
    L = np.ones([1])
    n_iter = 10
    extra_iter = 5
    k_max = 1
    
    # Cat: TODO: we limited # iterations; CHECK THIS
    #while iter < n_iter:
    while iter < min(n_iter,1000):
        iter += 1
        vbParam, suffStat, L = birth_move(maskedData, vbParam, suffStat, param,
                                          L)
        # print('birth',vbParam.rhat.shape[1])
        vbParam, suffStat, L = merge_move(maskedData, vbParam, suffStat, param,
                                          L, 0)
        # print('merge',vbParam.rhat.shape[1])

        k_now = vbParam.rhat.shape[1]
        # print(k_now)
        if (k_now > k_max) and (iter + extra_iter > n_iter):
            n_iter = iter + extra_iter
            k_max = k_now

        if iter > 100:
            print ("MFM split_merge reached 100 iterations...<<<<<<<<<< ")

    vbParam, suffStat, L = merge_move(maskedData, vbParam, suffStat, param, L,
                                     1)

    return vbParam


#def cluster_triage(vbParam, score, threshold):

#    maha = calc_mahalonobis(vbParam, score)

#    cluster_id = np.argmin(maha, axis=1)
#    idx = np.all(maha >= threshold, axis=1)
#    cluster_id[idx] = -1
#    return cluster_id


def cluster_triage(vbParam, score, threshold):
	maha = calc_mahalonobis(vbParam, score)
	cluster_id = np.argmax(vbParam.rhat, axis=1)
	idx = np.all(maha >= threshold, axis=1)
	cluster_id[idx] = -1
	return cluster_id
    

def get_core_data(vbParam, score, n_max, threshold):

    n_data = vbParam.rhat.shape[0]
    n_units = int(np.max(vbParam.rhat[:, 1]) + 1)

    idx_keep = np.zeros(n_data, 'bool')
    for k in range(n_units):
        idx_data = np.where(vbParam.rhat[:, 1] == k)[0]

        score_k = score[vbParam.rhat[idx_data, 0].astype('int32')]

        # prec = vbParam.Vhat[:, :, k]*vbParam.nuhat[k]
        # mu = vbParam.muhat[:, k][np.newaxis]
        # score_mu = score_k - mu
        # maha = np.sqrt(np.sum(np.matmul(score_mu, prec)*score_mu, 1))

        prec = np.transpose(
            vbParam.Vhat[:, :, [k]] * vbParam.nuhat[
                np.newaxis, np.newaxis, [k], np.newaxis],
            axes=[2, 3, 0, 1])
        scoremhat = np.transpose(
            score_k[:, :, np.newaxis, :] -
            vbParam.muhat[:, [k]], axes=[0, 2, 3, 1])
        maha = np.sqrt(
            np.sum(
                np.matmul(
                    np.matmul(scoremhat[:, :, :, np.newaxis, :], prec),
                    scoremhat[:, :, :, :, np.newaxis]),
                axis=(3, 4),
                keepdims=False))
        maha = np.squeeze(maha)
        idx_data = idx_data[maha < threshold]

        if idx_data.shape[0] > n_max:
            idx_data = np.random.choice(idx_data, n_max, replace=False)

        idx_keep[idx_data] = 1

    return idx_keep


def calc_mahalonobis(vbParam, score):
    prec = np.transpose(
        vbParam.Vhat * vbParam.nuhat[np.newaxis, np.newaxis, :, np.newaxis],
        axes=[2, 3, 0, 1])
    scoremhat = np.transpose(
        score[:, :, np.newaxis, :] - vbParam.muhat, axes=[0, 2, 3, 1])
    maha = np.sqrt(
        np.sum(
            np.matmul(
                np.matmul(scoremhat[:, :, :, np.newaxis, :], prec),
                scoremhat[:, :, :, :, np.newaxis]),
            axis=(3, 4),
            keepdims=False))

    return maha[:, :, 0]


#def merge_move_quick(maskedData, vbParam, cluster_id, param):

    #n_merged = 0
    #n_features, n_clusters, n_channels = vbParam.muhat.shape

    #n_data = cluster_id.shape[0]
    #cluster_id_list = make_list_by_id(cluster_id)

    #if n_clusters > 1:
        #all_checked = 0
    #else:
        #all_checked = 1

    #while (not all_checked) and (n_clusters > 1):
        #m = np.transpose(vbParam.muhat, [1, 0, 2]).reshape(
            #[n_clusters, n_features * n_channels], order="F")
        #kdist = ss.distance_matrix(m, m)
        #kdist[np.tril_indices(n_clusters)] = np.inf

        #merged = 0
        #k_untested = np.ones(n_clusters)
        #while np.sum(k_untested) > 0 and merged == 0:
            #untested_k = np.argwhere(k_untested)
            #ka = untested_k[np.random.choice(len(untested_k), 1)].ravel()[0]
            #kb = np.argmin(kdist[ka, :]).ravel()[0]
            #k_untested[ka] = 0
            #if np.argmin(kdist[kb, :]).ravel()[0] == ka:
                #k_untested[kb] = 0
            #kdist[min(ka, kb), max(ka, kb)] = np.inf

            #vbParam, cluster_id_list, merged = check_merge_quick(
                #maskedData, vbParam, cluster_id_list, ka, kb, param)
            #if merged:
                #n_merged += 1
                #n_clusters -= 1
        #if not merged:
            #all_checked = 1

    #return vbParam, make_id_array(cluster_id_list, n_data)


def make_list_by_id(cluster_id):

    n_clusters = np.max(cluster_id) + 1

    cluster_id_list = [None]*n_clusters

    for k in range(n_clusters):
        cluster_id_list[k] = np.where(cluster_id == k)[0]

    return cluster_id_list


def make_id_array(cluster_id_list, n_data):

    cluster_id_array = np.zeros(n_data, 'int32')
    n_clusters = len(cluster_id_list)

    for k in range(n_clusters):
        cluster_id_array[cluster_id_list[k]] = k

    return cluster_id_array


def check_merge_quick(maskedData, vbParam, cluster_id, ka, kb, param):

    relevant_data = np.sort(np.hstack([cluster_id[ka], cluster_id[kb]]))
    maskedData_small = maskData()
    maskedData_small.sumY = maskedData.sumY[relevant_data]
    maskedData_small.sumYSq = maskedData.sumYSq[relevant_data]
    maskedData_small.sumEta = maskedData.sumEta[relevant_data]
    maskedData_small.weight = maskedData.weight[relevant_data]
    maskedData_small.groupMask = maskedData.groupMask[relevant_data]
    maskedData_small.meanY = maskedData.meanY[relevant_data]
    maskedData_small.meanYSq = maskedData.meanYSq[relevant_data]
    maskedData_small.meanEta = maskedData.meanEta[relevant_data]

    vbParam_before = vbPar(0)
    vbParam_before.muhat = vbParam.muhat[:, [ka, kb], :]
    vbParam_before.Vhat = vbParam.Vhat[:, :, [ka, kb], :]
    vbParam_before.invVhat = vbParam.invVhat[:, :, [ka, kb], :]
    vbParam_before.lambdahat = vbParam.lambdahat[[ka, kb]]
    vbParam_before.nuhat = vbParam.nuhat[[ka, kb]]
    vbParam_before.ahat = vbParam.ahat[[ka, kb]]

    vbParam_before.update_local(maskedData_small)
    suffStat_before = suffStatistics(maskedData_small, vbParam_before)

    ELBO_bmerge = ELBO_Class(maskedData_small, suffStat_before,
                             vbParam_before, param).total

    suffStat_after = suffStatistics()
    suffStat_after.Nhat = np.sum(suffStat_before.Nhat,
                                 axis=0, keepdims=True)
    suffStat_after.sumY = np.sum(suffStat_before.sumY,
                                 axis=1, keepdims=True)
    suffStat_after.sumYSq = np.sum(suffStat_before.sumYSq,
                                   axis=2, keepdims=True)
    suffStat_after.sumYSq1 = np.sum(suffStat_before.sumYSq1,
                                    axis=2, keepdims=True)
    suffStat_after.sumYSq2 = np.sum(suffStat_before.sumYSq2,
                                    axis=2, keepdims=True)

    vbParam_after = vbPar(np.ones((vbParam_before.rhat.shape[0], 1)))
    vbParam_after.update_global(suffStat_after, param)

    ELBO_amerge = ELBO_Class(maskedData_small, suffStat_after,
                             vbParam_after, param).total
    if ELBO_amerge < ELBO_bmerge:
        merged = 0
        return vbParam, cluster_id, merged

    else:
        merged = 1

        no_kab = np.ones(len(cluster_id), 'bool')
        no_kab[[ka, kb]] = False

        vbParam.muhat = np.concatenate(
            [vbParam.muhat[:, no_kab, :], vbParam_after.muhat], axis=1)
        vbParam.Vhat = np.concatenate(
            [vbParam.Vhat[:, :, no_kab, :], vbParam_after.Vhat], axis=2)
        vbParam.invVhat = np.concatenate(
            [vbParam.invVhat[:, :, no_kab, :], vbParam_after.invVhat],
            axis=2)
        vbParam.lambdahat = np.concatenate(
            [vbParam.lambdahat[no_kab], vbParam_after.lambdahat],
            axis=0)
        vbParam.nuhat = np.concatenate(
            [vbParam.nuhat[no_kab], vbParam_after.nuhat],
            axis=0)
        vbParam.ahat = np.concatenate(
            [vbParam.ahat[no_kab], vbParam_after.ahat],
            axis=0)

        del cluster_id[np.max((ka, kb))]
        del cluster_id[np.min((ka, kb))]
        cluster_id.append(relevant_data)

        return vbParam, cluster_id, merged
