__doc__ = 'help thingy'
__author__ = 'me'


try:
    import lal
    import lalsimulation as LS
except:
    pass
import scipy.interpolate as ip
#import george
import numpy as np
import scipy.optimize as op
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn import linear_model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
    import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared, ConstantKernel, Matern
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.deprecation import deprecated
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import fmin_l_bfgs_b
import warnings
from operator import itemgetter
from scipy.stats import norm

class GaussianProcessRegressor_prior(GaussianProcessRegressor):
    """Same as the sci-kit learn GaussianProcessRegressor, but
       add the method fit_prior which allows fitting with a
       prior on the covariance and length scales"""

    def __init__(self, kernel=None, alpha=1e-10,
                 optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0,
                 normalize_y=False, copy_X_train=True, random_state=None,prior=None):
        GaussianProcessRegressor.__init__(self, kernel=kernel, alpha=alpha,
            optimizer="fmin_l_bfgs_b", n_restarts_optimizer=n_restarts_optimizer,
            normalize_y=False, copy_X_train=True, random_state=None)
        self.prior = prior

    def lognorm_prior(self,val,loc=0.,scale=0.5):
        """Base 10 lognorm prior value at val for lognorm at
        location ln(val)=loc and width(ln(val))~scale
        """
        return np.log(norm.pdf(val*np.log10(np.exp(1)),loc=loc,scale=scale))

    def fit_prior(self, X, y):
        """Fit Gaussian process regression model.
        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data
        y : array-like, shape = (n_samples, [n_output_dims])
            Target values
        Returns
        -------
        self : returns an instance of self.
        """
        if self.kernel is None:  # Use an RBF kernel as default
            self.kernel_ = C(1.0, constant_value_bounds="fixed") \
                * RBF(1.0, length_scale_bounds="fixed")
        else:
            self.kernel_ = clone(self.kernel)

        self._rng = check_random_state(self.random_state)

        X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

        # Normalize target value
        if self.normalize_y:
            self.y_train_mean = np.mean(y, axis=0)
            # demean y
            y = y - self.y_train_mean
        else:
            self.y_train_mean = np.zeros(1)

        if np.iterable(self.alpha) \
           and self.alpha.shape[0] != y.shape[0]:
            if self.alpha.shape[0] == 1:
                self.alpha = self.alpha[0]
            else:
                raise ValueError("alpha must be a scalar or an array"
                                 " with same number of entries as y.(%d != %d)"
                                 % (self.alpha.shape[0], y.shape[0]))

        self.X_train_ = np.copy(X) if self.copy_X_train else X
        self.y_train_ = np.copy(y) if self.copy_X_train else y

        if self.optimizer is not None and self.kernel_.n_dims > 0:
            # Choose hyperparameters based on maximizing the log-marginal
            # likelihood (potentially starting from several initial values)
            def obj_func(theta, eval_gradient=True):
                if eval_gradient:
                    lml, grad = self.log_marginal_likelihood(
                        theta, eval_gradient=True)
                    if self.prior == None:
                        return -lml, -grad
                    else:
                        for i in range(0,len(theta)):
                            if i == 0:
                                val = theta[i]/2.
                            else:
                                val = theta[i]
                            if self.prior==None:
                                pass
                            else:
                                lml = lml + self.lognorm_prior(float(val),self.prior[i][0],self.prior[i][1])
                        return -lml, -grad
                else:
                    lml = self.log_marginal_likelihood(theta)
                    if self.prior == None:
                        return -lml
                    else:
                        for i in range(0,len(theta)):
                            if i == 0:
                                val = theta[i]/2.
                            else:
                                val = theta[i]
                            lml = lml + self.lognorm_prior(float(val),self.prior[i][0],self.prior[i][1])
                        return -lml

            # First optimize starting from theta specified in kernel
            optima = [(self._constrained_optimization(obj_func,
                                                      self.kernel_.theta,
                                                      self.kernel_.bounds))]

            # Additional runs are performed from log-uniform chosen initial
            # theta
            if self.n_restarts_optimizer > 0:
                if not np.isfinite(self.kernel_.bounds).all():
                    raise ValueError(
                        "Multiple optimizer restarts (n_restarts_optimizer>0) "
                        "requires that all bounds are finite.")
                bounds = self.kernel_.bounds
                for iteration in range(self.n_restarts_optimizer):
                    theta_initial = \
                        self._rng.uniform(bounds[:, 0], bounds[:, 1])
                    optima.append(
                        self._constrained_optimization(obj_func, theta_initial,
                                                       bounds))
            # Select result from run with minimal (negative) log-marginal
            # likelihood
            lml_values = list(map(itemgetter(1), optima))
            self.kernel_.theta = optima[np.argmin(lml_values)][0]
            self.log_marginal_likelihood_value_ = -np.min(lml_values)
        else:
            self.log_marginal_likelihood_value_ = \
                self.log_marginal_likelihood(self.kernel_.theta)

        # Precompute quantities required for predictions which are independent
        # of actual query points
        K = self.kernel_(self.X_train_)
        K[np.diag_indices_from(K)] += self.alpha
        self.L_ = cholesky(K, lower=True)  # Line 2
        self.alpha_ = cho_solve((self.L_, True), self.y_train_)  # Line 3

        return self

def make_FDwaveform(M, q, chi1, chi2, approximant_FD, deltaF=0.5, f_min=10, f_max=4096):
    """Generate FD waveform from lal.
    Parameters
    ----------
    M : scalar, mass in solar masses
    q : scalar, mass ratio
    chi1,chi2: aligned spins
    approximant_FD: approximant type
    deltaF: frequency spacing
    f_min: starting freq
    f_max: max freq

    Returns
    -------
    [frequency array, H plus]
    """

    m1 = M*1.0/(1.0+q)
    m2 = M*q/(1.0+q)
    phiRef = 0.0
    m1_SI = m1 * lal.MSUN_SI
    m2_SI = m2 * lal.MSUN_SI
    S1x = 0.0
    S1y = 0.0
    S1z = chi1
    S2x = 0.0
    S2y = 0.0
    S2z = chi2
    f_ref = 0.0
    r = 1e6 * lal.PC_SI
    z = 0.0
    i = 0.0
    lambda1 = 0.0
    lambda2 = 0.0
    waveFlags = None
    nonGRparams = None
    amplitudeO = -1
    phaseO = -1
#    try:
#        # old call signature for lalsimulation waveforms
#        Hp, Hc = LS.SimInspiralChooseFDWaveform(phiRef, deltaF, m1_SI, m2_SI, S1x, S1y, S1z, S2x, S2y, S2z,
#          f_min, f_max, f_ref, r, i, lambda1, lambda2, waveFlags, nonGRparams, amplitudeO, phaseO,
#          approximant_FD)
#        print 'using old lalsimulation waveform signature'
#    except TypeError:
#        # newer call signature for lalsimulation waveforms includes parameters for eccentricity
#        longAscNodes, eccentricity, meanPerAno = 0.0, 0.0, 0.0
#        # If we need extra parameters put them into the Dict; I assume we don't need anything special
#        LALpars = None
#        phaseO = None
#        Hp, Hc = LS.SimInspiralChooseFDWaveform(m1_SI, m2_SI,
#                                                S1x, S1y, S1z, S2x, S2y, S2z, 
#                                                r, i, phiRef, 
#                                                longAscNodes, eccentricity, meanPerAno,
#                                                deltaF, f_min, f_max, f_ref, 
#                                                LALpars, 
#                                                approximant_FD)
#
    Hp, Hc = LS.SimInspiralChooseFDWaveform(phiRef, deltaF, m1_SI, m2_SI, S1x, S1y, S1z, S2x, S2y, S2z,
      f_min, f_max, f_ref, r, i, lambda1, lambda2, waveFlags, nonGRparams, amplitudeO, phaseO,
      approximant_FD)


    f = np.arange(Hp.data.length) * deltaF
    n = len(f)
    return [f, Hp.data.data]

def make_training_list(Mlow,Mhigh,qlow=None,qhigh=None,nM=10,nq=None,oneD=False):
    if (oneD==False):
        n = nM*nq
        Mtrain = np.linspace(Mlow,Mhigh,nM)
        qtrain = np.linspace(qlow,qhigh,nq)
        trainlist = np.empty([n,2])
        for iq,qtr in enumerate(qtrain):
            for iM,Mtr in enumerate(Mtrain):
                trainlist[(iM)*nq+iq,0] = Mtr
                trainlist[(iM)*nq+iq,1] = qtr
    else:
        trainlist = np.linspace(Mlow,Mhigh,nM)
    return trainlist

def make_training_list_dMdq(Mlow,Mhigh,dM,qlow=None,qhigh=None,dq=None,oneD=False):
    # make a training list where one can specify an arbitrary spacing between
    # training points.  dM and dq should be lambda functions which give the 
    # spacing as a function of M and q.
    if (oneD == False):
        print 'NOT IMPLEMENTED YET!!!'
    else:
        Mtrain = [Mlow];
        while (Mtrain[-1]<Mhigh):
            pt = Mtrain[-1] + dM(Mtrain[-1])
            if (pt<=Mhigh):
                Mtrain.append(pt)
            else:
                Mtrain.append(Mhigh)
        Mtrain = np.array(Mtrain)
        return Mtrain

def make_training_list_arb(paramlists,params):
    # use lists of training points in different paramters
    # to create a flattened list that specifies a rectangular
    # grid in the various parameters. Paramlists is a 
    # list in which each element is a list of values for one
    # of the parameters
    from itertools import product
    trainlistlen = 1
    nparams = len(params) 
    for i in range(0,nparams):
        trainlistlen = trainlistlen * len(paramlists[i])
    trainlist = np.empty([nparams,trainlistlen],dtype=object)
    for i,items in enumerate(product(*paramlists)):
        trainlist[:,i] = items
    return trainlist
    #count = 0
    #for i in range(0,numparams):
    #    for val in paramlists[i]


def make_Mf(fHzlow=20, Mlow=2, fmaxR=0.15, DPhi=None):
    # Makes list of M*f used to interpolate waveforms in SVD
    # Dphi specifies how the grid in frequency
    fHzlo = fHzlow/2.
    Mflo = fHzlo * Mlow * lal.MTSUN_SI
    fminR = Mflo;
    fmaxR = 0.15;
    print fmaxR/(Mlow*lal.MTSUN_SI) 
    if DPhi == None:
        # gives a good balance of resolution at low and high frequencies: 
        # a bit worse than greedy points for low masses near q=1; much better at the high q end.
        DeltaPhi = lambda f: 0.3*(f**(4.0/3.0))     
    else:
        DeltaPhi = DPhi
    gPhi = [fminR];
    while (gPhi[-1]<fmaxR):
        pt = gPhi[-1] + DeltaPhi(gPhi[-1]);
        if (pt<=fmaxR):
            gPhi.append(pt);
        else:
            gPhi.append(fmaxR);
    gPhi = np.array(gPhi)
    g = gPhi
    return g

def make_wf_amp_phi(Mtot,q,gA,gPhi,df,chirp=False,fHzlo=22,approximant=LS.IMRPhenomD):
    # choose deltaF, fmin, fmax so they work for the chosen total mass of 20 Msun and producing the wf is cheap
    # gA,gPhi specifies the frequency gridding in Mf for the amplitude and phase respectively.
    
    # allow Mtot to be a tuple for use with the partial function when multiprocessing:
    if type(Mtot) is tuple:
        Mtotal = Mtot[0]
        q_new = Mtot[1]
    else:
        Mtotal = Mtot
        q_new = q
    if chirp == True:
        M = Mtotal*((q_new/(1.+q_new)**2)**(-3./5.))
    else:
        M = Mtotal
    chi1 = 0
    chi2 = 0
    [fHz, hp] = make_FDwaveform(M, q_new, chi1, chi2, approximant, deltaF=df, f_min=fHzlo/2, f_max=0)
    amp = np.abs(hp)
    phi = np.unwrap(np.angle(hp))

    # interpolate onto sparse points
    Mf = fHz * M * lal.MTSUN_SI
    mask = amp > 0
    ampI = ip.InterpolatedUnivariateSpline(Mf[mask], amp[mask])
    phiI = ip.InterpolatedUnivariateSpline(Mf[mask], phi[mask])
    return [ampI(gA), phiI(gPhi)]

def make_wf_amp_phi_dict(inputdict,gA,gPhi,df,chirp=False,fHzlo=22,sameChi=True):
    defaultdict = {'M': 20,
            'q' : 1,
            'chi1' : 0,
            'chi2' : 0,
            'approximant' : LS.IMRPhenomD}
    for key in list(defaultdict.keys()):
        try: 
            defaultdict[key] = inputdict[key]
        except KeyError:
            pass
    # choose whether the mass given is chirp mass or total mass
    if chirp == True:
        defaultdict['M'] = defaultdict['M']*((defaultdict['q']/(1.+defaultdict['q'])**2)**(-3./5.))
    else:
        pass
    # choose whether or not chi1=chi2
    if sameChi == True:
        defaultdict['chi2'] = defaultdict['chi1']
    else:
        pass
    [fHz,hp] = make_FDwaveform(defaultdict['M'],
            defaultdict['q'],
            defaultdict['chi1'],
            defaultdict['chi2'],
            defaultdict['approximant'],
            deltaF=df,
            f_min=fHzlo/2.,
            f_max=0)
    amp = np.abs(hp)
    phi = np.unwrap(np.angle(hp))

    # interpolate onto sparse points
    Mf = fHz * defaultdict['M'] * lal.MTSUN_SI
    mask = amp > 0
    ampI = ip.InterpolatedUnivariateSpline(Mf[mask], amp[mask])
    phiI = ip.InterpolatedUnivariateSpline(Mf[mask], phi[mask])
    return [ampI(gA), phiI(gPhi)]

def make_amps_phis_table(trainlist,gA,gPhi,df,chirp=True,oneD=False):
    if oneD==False:
        wftab = np.array([make_wf_amp_phi(trainlist[i,0],trainlist[i,1],gA,gPhi,df,chirp=chirp) for i in range(0,len(trainlist))])
    else:
        wftab = np.array([make_wf_amp_phi(trainlist[i],1.,gA,gPhi,df,chirp=chirp) for i in range(0,len(trainlist))])
    amps = np.array([line[0] for line in wftab])
    phis = np.array([line[1] for line in wftab])
    #[amps, phis] = np.transpose(wftab, (1,0,2))
    return amps,phis

def make_amps_phis_table_dict(trainlist, paramlist, gA,gPhi,df,chirp=True,ncores=4,sameChi=True):
    # makes the amp/phi table using multiple cores on one node but allows one to
    # choose which parameters are varied for interpolation using paramlist. 
    # Trainlist should have the same number of columns as params in paramlist.

    dictlist=[]
    for line in trainlist.T:
        tempdict = dict()
        for p,param in enumerate(paramlist):
            tempdict[param] = line[p]
        dictlist.append(tempdict)

    from functools import partial
    from multiprocessing import Pool
    pool = Pool(ncores)
    make_wf_amp_phi_parallel = partial(make_wf_amp_phi_dict,gA=gA,gPhi=gPhi,df=df,chirp=chirp,sameChi=sameChi)
    wftab = np.array(pool.map(make_wf_amp_phi_parallel,dictlist))
    pool.close()
    pool.join()
    amps = np.array([line[0] for line in wftab])
    phis = np.array([line[1] for line in wftab])
    return amps,phis

def make_amps_phis_table_parallel(trainlist,gA,gPhi,df,chirp=True,oneD=False,ncores=4):
    # makes the amp/phi table using multiple cores on one node/machine
    from functools import partial
    from multiprocessing import Pool
    pool = Pool(ncores)
    if oneD==False:
        #partial only takes in one argument, so need to send in tuples of (M,q)
        mytrain = [(ex,trainlist[:,1][e]) for e,ex in enumerate(trainlist[:,0])]
        make_wf_amp_phi_parallel = partial(make_wf_amp_phi,q=1.,gA=gA,gPhi=gPhi,df=df,chirp=chirp)
        wftab = np.array(pool.map(make_wf_amp_phi_parallel,mytrain))
        #wftab = np.array([make_wf_amp_phi(trainlist[i,0],trainlist[i,1],g,g,df,chirp=chirp) for i in range(0,len(trainlist))])
    else:
        make_wf_amp_phi_parallel = partial(make_wf_amp_phi,q=1.,gA=gA,gPhi=gPhi,df=df,chirp=chirp)
        wftab = np.array(pool.map(make_wf_amp_phi_parallel,trainlist))
        #wftab = np.array([make_wf_amp_phi(trainlist[i],1.,g,g,df,chirp=chirp) for i in range(0,len(trainlist))])
    pool.close()
    pool.join()
    amps = np.array([line[0] for line in wftab])
    phis = np.array([line[1] for line in wftab])
    print amps.shape, phis.shape
    #[amps, phis] = np.transpose(wftab, (1,0,2))
    #print amps.shape, phis.shape
    return amps,phis

def save_amp_phi(trainmin,trainmax,filename,chirp=True,oneD=False):
    print trainmin,trainmax
    trainlist = np.load('SIM_FILES/trainlist.npy')[trainmin:trainmax]
    gA = np.load('SIM_FILES/gA.npy')
    gPhi = np.load('SIM_FILES/gPhi.npy')
    df = np.load('SIM_FILES/df.npy')
    df=float(df)
    np.save('SIM_FILES/'+filename+'_'+str(trainmin)+'.npy',make_amps_phis_table(trainlist,g,df,chirp=chirp,oneD=True))

def open_amp_phi(trainlist,Nnodes,filename):
    trainlen = len(trainlist)
    Npernode = int(np.ceil(float(trainlen)/Nnodes))
    amps,phis = np.load('SIM_FILES/'+filename+'_0.npy')
    print amps.shape,phis.shape
    for i in range(1,Nnodes):
        trainmin = i*Npernode
        ampnew,phinew = np.load('SIM_FILES/'+filename+'_'+str(trainmin)+'.npy')
        amps = np.append(amps,ampnew,axis=0)
        phis = np.append(phis,phinew,axis=0)
    return amps,phis

def saveall_amp_phi(trainlist,gA,gPhi,df,Nnodes):
    import os
    os.system('rm SIM_LOG/*')
    os.system('rm SIM_FILES/*')
    np.save('SIM_FILES/trainlist.npy',trainlist)
    np.save('SIM_FILES/gA.npy',gA)
    np.save('SIM_FILES/gPhi.npy',gPhi) 
    np.save('SIM_FILES/df.npy',df)
    trainlen = len(trainlist)
    Npernode = int(np.ceil(float(trainlen)/Nnodes))
    for i in range(0,Nnodes):
        trainmin = i*Npernode
        trainmax = (i+1)*Npernode
        if trainmax > trainlen:
            trainmax=trainlen
        lines = open('parallel.sbatch','r').readlines()
        pystr = 'python -c \'from model2D import save_amp_phi; save_amp_phi('+str(trainmin)+','+str(trainmax)+',\"ampsphis\",chirp=True,oneD=True)\'' 
        lines[-1] = pystr
        open('parallel.sbatch','w').writelines(lines)
        os.system('sbatch parallel.sbatch')

def make_cAcP(amps, phis, getErrors=False, ErrorLevel=1e-5,ErrorLevel_Phi=.01, getSigma=False):
    UA, sA, VA = np.linalg.svd(amps, full_matrices=True)
    UP, sP, VP = np.linalg.svd(phis, full_matrices=True)
    VA = VA.T
    VP = VP.T
    cA = np.array(map(lambda A: np.dot(A, VA), amps))
    cP = np.array(map(lambda P: np.dot(P, VP), phis))
    print VA.shape,VP.shape
    if getErrors == True:
        # constant relative error for amplitude
        cA_variance = np.array(map(lambda A: np.diag(np.dot(VA.T,np.dot(np.diag(np.power(A,2.)),VA))),ErrorLevel*amps))
        # constant error of 
        cP_variance = np.array(map(lambda P: np.diag(np.dot(VP.T,np.dot(np.diag(np.power(P,2.)),VP))),ErrorLevel_Phi*np.ones_like(phis)))
        #cA_variance = np.array(map(lambda A: np.diag(VA.T*np.diag(np.power(A,2))*VA),ErrorLevel*amps))
        #cP_variance = np.array(map(lambda P: np.diag(VP.T*np.diag(np.power(P,2))*VP),ErrorLevel*phis))
        if getSigma==False:
            return cA,cP,VA,VP,cA_variance,cP_variance
        else:
            return cA,cP,VA,VP,cA_variance,cP_variance, sA, sP
    else:
        if getSigma==False:
            return cA,cP,VA,VP
        else:
            return cA,cP,VA,VP, sA, sP


def make_interp_list(nM,num_per_Mtrain, Mlow, Mhigh, nq=None,num_per_qtrain=None,qlow=None, qhigh=None):
    if nq==None:
        numM = (nM*num_per_Mtrain)-num_per_Mtrain+1
        z = np.linspace(Mlow,Mhigh,numM)
    else:
        numM = (nM*num_per_Mtrain)-num_per_Mtrain+1
        numq = (nq*num_per_qtrain)-num_per_qtrain+1
        interpM = np.linspace(Mlow,Mhigh,numM)
        interpq = np.linspace(qlow,qhigh,numq)
        z = np.empty([numM*numq,2],dtype=float)
        for inM,interM in enumerate(interpM):
            for inq,interq in enumerate(interpq):
                z[(inM)*numq+inq,0] = interM
                z[(inM)*numq+inq,1] = interq
    return z

class Coef_GP2(object):
    def __init__(self, t, y, kernel=None,normalize=True,err=1e-8,linfit=False,coef_type='amp',prior=None,**kwargs):
        self._linfit = linfit
        self.prior = prior
        try:
            self._t = np.swapaxes(t.reshape(-1,t.shape[1]),0,1)
        except IndexError:
            self._t = t.reshape(-1,1)
        self._normalize=normalize
        if self._normalize:
            if self._linfit:
                self._y = self.whiten_linear(y.reshape(-1,1),whiten_type='std')
            else:
                self._y = self.whiten(y.reshape(-1,1),whiten_type='std')
        else:
            self._y = y.reshape(-1,1)
        try:
            print 'hi!'
            print 'yshape',self._y.shape
            self._err = np.diag(err.reshape(-1,1))/np.square(self._sig)
            print 'errshape',self._err.shape
            print self._err
            print 'done!'
        except AttributeError:
            self._err = err/np.square(self._sig)

        self.kernel(kernel, **kwargs)

    def kernel(self, kernel_type=None,**kwargs):
        if kernel_type == None:
            raise NameError('hi.  please give me a kernel.  kthxbai')
        else:
            kern = kernel_type
            #kern=copy.deepcopy(kernel_type)
        if self._normalize:
            self._gp = GaussianProcessRegressor_prior(kernel=kern, alpha=self._err,normalize_y=False,n_restarts_optimizer=50,prior=self.prior)
        else:
            self._gp = GaussianProcessRegressor_prior(kernel=kern, alpha=self._err,normalize_y=False,n_restarts_optimizer=50,prior=self.prior)
        #self._gp.fit(self._t,self._y)
        self._gp.fit_prior(self._t,self._y)
        #print np.power(10.,np.array(self._gp.kernel_.theta))
        newparams = self._gp.kernel_.get_params()
        print '********************'
        for key in sorted(newparams): print("%s : %s" % (key, newparams[key]))
        print '********************'
        print 'log marginal likelihood:',self._gp.log_marginal_likelihood_value_
    
    def mean(self, x):
        try:
            mean = self._gp.predict(np.swapaxes(x.reshape(-1,x.shape[1]),0,1))
        except IndexError:
            mean = self._gp.predict(x.reshape(-1,1))
        if self._normalize:
            if self._linfit:
                return self.color_linear(mean,x).reshape(-1)
            else:
                return self.color(mean).reshape(-1)
        else:
            return mean.reshape(-1)

    def std(self, x):
        try:
            _, std = self._gp.predict(np.swapaxes(x.reshape(-1,x.shape[1]),0,1),return_std=True)
        except IndexError:
            _, std = self._gp.predict(x.reshape(-1,1),return_std=True)
        if self._normalize:
            return self._sig*std.reshape(-1) 
        else:
            return std.reshape(-1)

    def cov(self,x):
        try:
            _, cov = self._gp.predict(np.swapaxes(x.reshape(-1,x.shape[1]),0,1),return_cov=True)
        except IndexError:
            _, cov = self._gp.predict(x.reshape(-1,1),return_cov=True)
        return cov

    def std_PSD(self,x):
        thecov = self.cov(x)
        #thecov = nearPD(thecov)
        state = False
        count = -300
        while state==False:
            thecov=thecov+np.identity(thecov.shape[0])*np.power(10,count)
            thestd = np.sqrt(np.diag(thecov))
            if ~np.any(thestd<=0):
                state=True
            count=count+0.5
        print count
        print thestd
        if self._normalize:
            thestd = self._sig*thestd
        return thestd

    def whiten(self, data,whiten_type='std'):
        if whiten_type == 'std':
            self._mu, self._sig = np.mean(data), np.std(data)
        elif whiten_type == 'max':
            self._mu, self._sig = np.mean(data), np.amax(data)-np.amin(data)
        return (data - self._mu)/self._sig

    def color(self, data):
        return data * self._sig + self._mu

    def whiten_linear(self,data,whiten_type='std'):
        self._reg = linear_model.LinearRegression()
        # fit all training points
        self._reg.fit(self._t, data)
        # fit just the endpoints
        #tendpoints = np.array([self._t[0,:],self._t[-1,:]]).reshape(-1,1)
        #dataendpoints = np.array([data[0],data[-1]]).reshape(-1,1)
        #self._reg.fit(tendpoints,dataendpoints)
        linear_removed = data - self._reg.predict(self._t)
        return self.whiten(linear_removed,whiten_type=whiten_type)

    def whiten_poly(self,data):
        self._reg = Pipeline([('poly', PolynomialFeatures(degree=3)),
            ('linear', linear_model.LinearRegression(fit_intercept=True))])
        self._reg = self._reg.fit(self._t,data)
        poly_removed = data - self._reg.predict(self._t)
        return self.whiten(poly_removed)

    def color_linear(self,data,x):
        colored = self.color(data)
        return colored + self._reg.predict(np.swapaxes(x.reshape(-1,x.shape[1]),0,1))

    def color_poly(self,data,x):
        colored = self.color(data)
        return colored + self._reg.predict(np.swapaxes(x.reshape(-1,x.shape[1]),0,1))

    def sample(self,x,n_samples):
        try:
            sample = self._gp.sample_y(np.swapaxes(x.reshape(-1,x.shape[1]),0,1),n_samples=n_samples,random_state=np.random.randint(10000))
        except IndexError:
            sample = self._gp.sample_y(x.reshape(-1,1),n_samples=n_samples,random_state=np.random.randint(10000))
        if self._normalize:
            if self._linfit:
                return self.color_linear(sample,x).reshape(-1)
            else:
                return self.color(sample).reshape(-1)
        else:
            return sample.reshape(-1)

def perform_GPR(cA,cP,z,trainlist,oneD=False,kernel=None):
    cAinterp = np.empty([len(z),len(cA.T)])
    cPinterp = np.empty([len(z),len(cP.T)])
    cAinterp_std = np.empty([len(z),len(cA.T)])
    cPinterp_std = np.empty([len(z),len(cP.T)])

    randominit = False
    
    if oneD==False:
        ndim=2
    else:
        ndim=1
    for i in range(0,len(cA.T)):
        #GPamp = Coef_GP(trainlist, cA.T[i], err=1E-5, ndim=ndim,kernel=kernel,randinit=randominit)
        #GPphase = Coef_GP(trainlist, cP.T[i], err=1E-5, ndim=ndim,kernel=kernel,randinit=randominit)
        GPamp = Coef_GP2(trainlist, cA.T[i],kernel=kernel)
        meanamp= GPamp.mean(z)
        stdamp= GPamp.std(z)
        cAinterp[:,i]=meanamp
        cAinterp_std[:,i]=stdamp
    for i in range(0,len(cP.T)):
        GPphase = Coef_GP2(trainlist, cP.T[i],kernel=kernel)
        meanphase= GPphase.mean(z)
        stdphase= GPphase.std(z)
        cPinterp[:,i]=meanphase
        cPinterp_std[:,i]=stdphase
 
    return cAinterp, cPinterp, cAinterp_std,cPinterp_std

def singleGPRnew(i,c,z,trainlist,ndim,kernel,randominit,err,get_cov=False,normalize=True,linfit=False,coef_type='amp',prior=None,jobname=''):
    randominit = False
    print 'coefficient number',i
    #GPamp = Coef_GP(trainlist, cA.T[i], err=err, ndim=ndim,kernel=kernel,randinit=randominit)
    #GPphase = Coef_GP(trainlist, cP.T[i], err=err, ndim=ndim,kernel=kernel,randinit=randominit)
    try:
        error=err.T[i]
    except AttributeError:
        error = err
    GP = Coef_GP2(trainlist, c.T[i],kernel=kernel,err=error,get_cov=get_cov,normalize=normalize,linfit=linfit,coef_type=coef_type,prior=prior)

    joblib.dump(GP, 'SIM_FILES/Coef_GP_'+jobname+'_'+coef_type+'_'+str(i)+'.pkl') 
    #np.save('SIM_FILES/Coef_GP_'+coef_type+'_'+str(i)+'.npy',GP)
    themean= GP.mean(z)
    userealizations = False
    if userealizations==False:
        #thestd = GP.std_PSD(z)
        #thecov = nearPD(thecov)
        #thestd = np.sqrt(np.diag(thecov))
        thestd= GP.std(z)
        if np.any(thestd==0.0):
            print '999 Cov is bad, coef',i
    if get_cov ==True:
        thecov = GP.cov(z)
        return themean,thestd,thecov
    else:
        return themean,thestd
        #cAinterp[:,i]=meanamp
    #cPinterp[:,i]=meanphase
    #cAinterp_std[:,i]=stdamp
    #cPinterp_std[:,i]=stdphase
    return samples

    return meanamp,meanphase,stdamp,stdphase

def perform_GPR_parallel(cA, cP, z, trainlist,oneD=False,kernel=None,kernelPhase=None,ncores=4,err=1e-5,errPhi=None,get_cov=False,prior=None,jobname=''):
    from multiprocessing import Pool
    from functools import partial
    #from pathos.multiprocessing import ProcessingPool as Pool
    try:
        nsamples = z.shape[1]
    except IndexError:
        nsamples = len(z)
    cAinterp = np.empty([nsamples,len(cA.T)])
    cPinterp = np.empty([nsamples,len(cP.T)])
    cAinterp_std = np.empty([nsamples,len(cA.T)])
    cPinterp_std = np.empty([nsamples,len(cP.T)])
    if get_cov:
        cAinterp_cov = np.empty([nsamples,nsamples,len(cA.T)])
        cPinterp_cov = np.empty([nsamples,nsamples,len(cP.T)])
    
    randominit = False
    if oneD==False:
        ndim=2
    else:
        ndim=1
    singleGPRamp = partial(singleGPRnew,c=cA,z=z,trainlist=trainlist,ndim=ndim,kernel=kernel,\
            randominit=randominit,err=err,get_cov=get_cov,normalize=True,linfit=False,coef_type='amp',prior=prior,jobname=jobname)
    pool = Pool(ncores)
    results = pool.map(singleGPRamp,range(0,len(cA.T)))
    pool.close()
    pool.join()
    for i in range(0,len(cA.T)):        
        cAinterp[:,i]=results[i][0]
        cAinterp_std[:,i]=results[i][1]
        if get_cov:
            cAinterp_cov[:,:,i] = results[i][2]
    print "phase\nphase\nphase\nphase\nphase\nphase\nphase\nphase\nphase\nphase\nphase\nphase\nphase\nphase\nphase\n" 
    if kernelPhase == None:
        kernphi = kernel
    else:
        kernphi = kernelPhase
    if errPhi == None:
        error = err
    else:
        error=errPhi
    singleGPRphase = partial(singleGPRnew,c=cP,z=z,trainlist=trainlist,ndim=ndim,kernel=kernphi,\
            randominit=randominit,err=error,get_cov=get_cov,normalize=True,linfit=False,coef_type='phase',prior=prior,jobname=jobname)
    pool = Pool(ncores)
    results = pool.map(singleGPRphase,range(0,len(cP.T)))
    pool.close()
    pool.join()
    for i in range(0,len(cP.T)):
        cPinterp[:,i]=results[i][0]
        cPinterp_std[:,i]=results[i][1]
        if get_cov:
            cPinterp_cov[:,:,i] = results[i][2]

    if get_cov:        
        return cAinterp, cPinterp, cAinterp_std,cPinterp_std,cAinterp_cov,cPinterp_cov
    else:
        return cAinterp, cPinterp, cAinterp_std,cPinterp_std

def quadsum(vect):
    return np.sqrt(np.sum(vect**2))

def get_interp_AmpPhase(cAinterp,cPinterp,VA,VP,oneD=False,chirp=True):
    amp_reconstructed = np.sum(cAinterp * VA, axis=1)
    phase_reconstructed = np.sum(cPinterp * VP, axis=1)
    return amp_reconstructed,phase_reconstructed

def get_interpANDreal_TDWF(M,amp_reconstructed,phase_reconstructed,gA,gPhi,fmax=1310.72,df=0.01,chirp=True):
    Mf = np.arange(0.0,fmax,df) * M * lal.MTSUN_SI
    amp = np.interp(Mf,gA,amp_reconstructed,left=0,right=0) # perhaps I should interpolate log(amp) and log(phi)?
    phase = np.interp(Mf,gPhi,phase_reconstructed,left=0,right=0)
    newfunc = np.multiply(amp,np.exp(1j*phase))
    invfft = np.fft.irfft(newfunc)
    amp_new,phi_new = make_wf_amp_phi(M,1.,gA,gPhi,df,chirp=chirp)
    amp_real = np.interp(Mf,gA,amp_new,left=0,right=0) # perhaps I should interpolate log(amp) and log(phi)?
    phase_real = np.interp(Mf,gPhi,phi_new,left=0,right=0)
    newfunc_real = np.multiply(amp_real,np.exp(1j*phase_real))
    invfft_real = np.fft.irfft(newfunc_real)
    t = np.arange(0.,1./df,1./(2.*fmax))
    return invfft,invfft_real,t

def get_TDWF(M,amp_reconstructed,phase_reconstructed,gA,gPhi,fmax=1310.72,df=0.01,chirp=True):
    Mf = np.arange(0.0,fmax,df) * M * lal.MTSUN_SI
    amp = np.interp(Mf,gA,amp_reconstructed,left=0,right=0) # perhaps I should interpolate log(amp) and log(phi)?
    phase = np.interp(Mf,gPhi,phase_reconstructed,left=0,right=0)
    newfunc = np.multiply(amp,np.exp(1j*phase))
    mylen = len(newfunc)
    num=2
    while num < mylen:
        num = 2*num
    invfft = np.fft.irfft(newfunc,num)
    t = np.linspace(0,1./df,len(invfft))
    return invfft[int(.999*len(invfft)):],t[int(.999*len(invfft)):]

def plot_interp(infile,fraclow,frachigh):
    TDdata = np.load(infile)
    TDint = TDdata[0]
    TDreal= TDdata[1]
    t = TDdata[2]
    print TDint,TDreal
    plt.figure()
    plt.plot(t[int(fraclow*len(TDint)):int(frachigh*len(TDint))-1],TDint[int(fraclow*len(TDint)):int(frachigh*len(TDint))-1],c='b',ls=':',lw=.5)
    plt.plot(t[int(fraclow*len(TDreal)):int(frachigh*len(TDreal))-1],TDreal[int(fraclow*len(TDreal)):int(frachigh*len(TDreal))-1],c='r',ls='-',lw=.5,alpha=0.5)
    plt.savefig('interpWF.png',dpi=400)

def get_draws(cA,cP,z,trainlist,oneD=False,kernel=None,ndraws=20):
    cAinterp_draws = np.empty([ndraws,len(cA.T)])
    cPinterp_draws = np.empty([ndraws,len(cA.T)])

    randominit = False
    
    if oneD==False:
        ndim=2
    else:
        ndim=1
    for i in range(0,len(cA.T)):
        GPamp = Coef_GP(trainlist, cA.T[i], err=1E-5, ndim=ndim,kernel=kernel,randinit=randominit)
        GPphase = Coef_GP(trainlist, cP.T[i], err=1E-5, ndim=ndim,kernel=kernel,randinit=randominit)
        for n in range(0,ndraws):
            cAinterp_draws[n,i] = GPamp.color(GPamp._gp.sample_conditional(GPamp._y, z))
            cPinterp_draws[n,i] = GPphase.color(GPphase._gp.sample_conditional(GPphase._y,z)) 
    return cAinterp_draws, cPinterp_draws


def binCenters(h):
    return (h[:-1]+h[1:])/2.

def getScalarProduct(amp1,amp2,phi1,phi2,freqs,noisecurve=None):
    # computes the scalar product of two freq-domain waveforms
    if noisecurve == None:
        noise = np.ones_like(amp1)
    else:
        noise = noisecurve
    mult = np.multiply
    div = np.true_divide
    newamp = mult(amp1,amp2)
    newphi = phi1-phi2
    return 4.*np.real(np.sum(mult(binCenters(div(mult(newamp,np.exp(1j*newphi)),noise)),np.diff(freqs))))

def multWF(amp1,amp2,phi1,phi2,noisecurve=None):
    # multiplies two complex WFs together given their amplitudes and phases separately
    if noisecurve == None:
        noise = np.ones_like(amp1)
    else:
        noise = noisecurve
    mult = np.multiply
    div = np.true_divide
    newamp = mult(amp1,amp2)
    newphi = phi1-phi2
    newfunc = mult(div(newamp,noise),np.exp(1j*newphi))
    print len(newfunc)
    return newfunc
    #return newfunc[:-1] # make sure that what we give irfft is an even length array, or else it takes forever to compute


def overlap(h1, h2, psd, df=.001, flow=16., fhigh=1024.):
    h1 = np.atleast_2d(h1)
    h2 = np.atleast_2d(h2)
    freqs = np.arange(h1.shape[1])*df
    sel = (freqs > flow) & (freqs < fhigh)
    return 4 * np.sum(h1[:, sel].conj() * h2[:, sel]/psd[sel], axis=1).real * df

def fitting_factor(h1, h2, psd, df=.001, flow=32., fhigh=1024.):
    h1norm = np.sqrt(overlap(h1, h1, psd, df=df, flow=flow, fhigh=fhigh))
    h2norm = np.sqrt(overlap(h2, h2, psd, df=df, flow=flow, fhigh=fhigh))
    return overlap(h1, h2, psd, df=df, flow=flow, fhigh=fhigh) / (h1norm * h2norm)

def computeOverlapNew(M,amp1in,amp2in,phi1in,phi2in,gA,gPhi,noisecurve=None,fmax=1310.72,df=0.01,normed=True):
    f = np.arange(0.0,fmax,df)
    Mf = f * M * lal.MTSUN_SI  
    amp1 = np.interp(Mf,gA,amp1in,left=0,right=0) # perhaps I should interpolate log(amp) and log(phi)?
    amp2 = np.interp(Mf,gA,amp2in,left=0,right=0)
    phi1 = np.interp(Mf,gPhi,phi1in,left=0,right=0)
    phi2 = np.interp(Mf,gPhi,phi2in,left=0,right=0)
    h1 = np.multiply(amp1,np.exp(1j*phi1))
    h2 = np.multiply(amp2,np.exp(1j*phi2))
    if noisecurve == None:
        noise = np.ones_like(amp1)
    else:
        noise = np.interp(f,noisecurve['f'],np.power(noisecurve['asd'],2),left=np.inf,right=np.inf)
    if normed:
        return fitting_factor(h1,h2,noise,df=df,flow=30,fhigh=5000)
    else:
        return overlap(h1,h2,noise,df=df,flow=30,fhigh=5000)


def get_single_overlap_temp(i,cAinterp,cPinterp,VA,VP,z,gA,gPhi,df,chirp,params,oneD=False,noisecurve=None):
        amp_reconstructed = np.sum(cAinterp[i,:] * VA, axis=1)
        phase_reconstructed = np.sum(cPinterp[i,:] * VP, axis=1)
        if oneD==False:
            amp_new,phi_new = make_wf_amp_phi(20.,z[i,1],gA,gPhi,df,chirp=chirp)
            newoverlap = computeOverlapNew(20.,amp_reconstructed,amp_new,phase_reconstructed,phi_new,gA,gPhi,noisecurve=noisecurve,fmax=5000.,df=df)
        else:
            inputdict = dict()
            for p,param in enumerate(params):
                try:
                    inputdict[param] = z[p,i]
                except:
                    inputdict[param] = z[i]
            amp_new,phi_new = make_wf_amp_phi_dict(inputdict,gA,gPhi,df,chirp=chirp,sameChi=True)
            #amp_new,phi_new = make_wf_amp_phi(20.,z[i],gA,gPhi,df,chirp=chirp)
            newoverlap = computeOverlapNew(20.,amp_reconstructed,amp_new,phase_reconstructed,phi_new,gA,gPhi,noisecurve=noisecurve,fmax=5000,df=df)
        return newoverlap



def get_all_overlaps_parallel(cAinterp,cPinterp,VA,VP,z,gA,gPhi,df,chirp,params,oneD=False,ncores=4,noisecurve=None):
    from functools import partial
    from multiprocessing import Pool
    
    overlaps = np.empty(len(z),dtype=float)
    try:
        numinterp = z.shape[1]
    except IndexError:
        numinterp = len(z)
    get_single_overlap_partial=partial(get_single_overlap_temp,cAinterp=cAinterp,cPinterp=cPinterp,VA=VA,VP=VP,z=z,gA=gA,gPhi=gPhi,df=df,chirp=chirp,params=params,oneD=oneD,noisecurve=noisecurve)
    pool = Pool(ncores)
    overlaps = pool.map(get_single_overlap_partial,range(0,numinterp))
    pool.close()
    pool.join()
    return np.array(overlaps)
    
def rearrange_overlaps2(z,overlaps):
    Ms = np.unique(z[0,:])
    qs = np.unique(z[1,:])
    numM = len(Ms)
    numq = len(qs)
    newoverlaps = np.empty([numM,numq],dtype=float)
    newM = np.empty([numM,numq],dtype=float)
    newq = np.empty([numM,numq],dtype=float)

    for l in range(0,z.shape[1]):
        Msel = Ms == z[0,l]
        qsel = qs == z[1,l]
        try:
            newoverlaps[Msel,qsel]=overlaps[l][0]
        except IndexError:
            newoverlaps[Msel,qsel]=overlaps[l]
        newM[Msel,qsel] = z[0,l]
        newq[Msel,qsel] = z[1,l]
    return newoverlaps,newM,newq
