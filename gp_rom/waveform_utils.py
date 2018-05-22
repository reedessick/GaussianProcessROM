__doc__ = """Functions for generating GW waveforms"""
__author__ = """Zoheyr Doctor, zoheyr@gmail.com"""

import lal
import lalsimulation as LS
from argparse import Namespace
import scipy.interpolate as ip
import numpy as np

def make_FDwaveform(paramdict):
    """Generate FD waveform from lal.
    Parameters
    ----------
    paramdict: dictionary
        A dictionary with keys for the different
        parameters that go into ChooseFDWaveform
    M : scalar, mass in solar masses
    q : scalar, mass ratio
    chi1,chi2: aligned spins
    approximant_FD: approximant type
    deltaF: frequency spacing
    f_min: starting freq
    f_max: max freq

    Returns
    -------
    f: array.  Array of frequency points
    Hp: h plus array at frequencies in f
    Hc: h cross array at frequencies in f
    """
    
    # define defaults
    defaultdict = dict()
    defaultdict['m1'] = 10.
    defaultdict['m2'] = 10.
    for key in ['phiRef','S1x','S1y','S1z','S2x','S2y','S2z','f_ref','z','i','lambda1','lambda2','longAscNodes','eccentricity','meanPerAno']:
        defaultdict[key]=0.0
    for key in ['waveFlags','nonGRparams','LALpars']:
        defaultdict[key]=None
    defaultdict['r'] = 1e6*lal.PC_SI
    defaultdict['amplitudeO'] = -1
    defaultdict['phaseO'] = -1
    defaultdict['f_min'] = 40.
    defaultdict['f_max'] = 4098.
    defaultdict['deltaF'] = 0.5
    defaultdict['approximant_FD'] = LS.IMRPhenomD

    # update defaults with what is in paramdict
    for key in paramdict.keys():
        defaultdict[key] = paramdict[key]

    # turn dictionary keys into variable names:
    for k,v in defaultdict.items(): exec(k+'=v') 

    m1_SI = m1*lal.MSUN_SI
    m2_SI = m2*lal.MSUN_SI

    try:
        # old call signature for lalsimulation waveforms
        Hp, Hc = LS.SimInspiralChooseFDWaveform(phiRef, deltaF, m1_SI, m2_SI, S1x, S1y, S1z, S2x, S2y, S2z,
          f_min, f_max, f_ref, r, i, lambda1, lambda2, waveFlags, nonGRparams, amplitudeO, phaseO,
          approximant_FD)
        print 'using old lalsimulation waveform signature'
    except TypeError:
        # newer call signature for lalsimulation waveforms includes parameters for eccentricity
        Hp, Hc = LS.SimInspiralChooseFDWaveform(m1_SI, m2_SI,
                                                S1x, S1y, S1z, S2x, S2y, S2z, 
                                                r, i, phiRef, 
                                                longAscNodes, eccentricity, meanPerAno,
                                                deltaF, f_min, f_max, f_ref, 
                                                LALpars, 
                                                approximant_FD)

    f = np.arange(Hp.data.length) * deltaF
    n = len(f)
    return f, Hp.data.data, Hc.data.data

def get_geometric_freq(g_low=0.004,g_high=0.15,dg=None):
    """
    Create a list of geometric frequencies (M*f) that
    will be used to sparsely interpolate waveforms

    Parameters
    ----------
    g_low: float
        lower geometric frequency bound
    g_high: float
        higher geometric frequency bound
    dg: float or callable (default None)
        if float, linear spacing dg, 
        if callable, callable is iteratively 
            applied to frequency to get next dg
        if none,
            the phase interpolation prescription from Purrer 2014

    Returns
    -------
    g: array of geometric frequencies
    """
        
    if dg == None:
        # gives a good balance of resolution at low and high frequencies: 
        # a bit worse than greedy points for low masses near q=1; much better at the high q end.
        dgfunc = lambda f: 0.3*(f**(4.0/3.0))     
    elif isinstance(dg,float):
        dgfunc = lambda f: dg
    else:
        dgfunc = dg

    g = [g_low];
    while (g[-1]<g_high):
        pt = g[-1] + dgfunc(g[-1]);
        if (pt<=g_high):
            g.append(pt);
        else:
            g.append(g_high);
    return np.array(g)

def interpolate_amp_phase(gA,gPhi,Mf,h):
    """
    Generate waveform and interpolate onto sparse 
    geometric frequency grid

    Parameters
    ----------
    gA: array
        geometric frequency array for amplitudes
    gPhi: array
        geometric frequency array for phases
    Mf: array
        geometric freq array for h
    h: array
        strain array

    Returns
    -------
    amps: list
        list of amplitudes at geometric freqs gA
    phis: list
        list of phases at geometric freqs gPhi
    """
    amp = np.abs(h)
    phi = np.unwrap(np.angle(h))
    mask = amp > 0.
    ampI = ip.InterpolatedUnivariateSpline(Mf[mask], amp[mask])
    phiI = ip.InterpolatedUnivariateSpline(Mf[mask], phi[mask])
    return [ampI(gA), phiI(gPhi)]

def get_sparse_waveform(paramdict,g_min=0.004,g_max=0.3,dgA=0.001,dgP=0.001,**kwargs):
    """
    Given a dictionary of binary parameters, 
    compute the sparse waveform.  

    Parameters
    ---------
    paramdict: dictionary of binary parameters

    Returns
    --------
    amp: list of amplitudes
    phi: list of phases
    """
    f,hp,hc = make_FDwaveform(paramdict)
    Mf = f*(paramdict['m1']+paramdict['m2'])*lal.MTSUN_SI
    gA = get_geometric_freq(g_min,g_max,dgA) 
    gPhi = get_geometric_freq(g_min,g_max,dgP)
    ampp,phip = interpolate_amp_phase(gA,gPhi,Mf,hp) 
    return ampp, phip 





