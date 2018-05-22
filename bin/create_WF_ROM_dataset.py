#!/bin/usr/env python

from matplotlib import pyplot as plt
from gp_rom import waveform_utils as wu
import numpy as np
import lal
import argparse 
from itertools import product

def make_flattened_paramlist_regular(paramlists):
    """
    Make a flattened list of sets of parameters to be 
    used to generate waveforms

    Parameters
    ----------
    paramlists: list
        A list where each element is a list of unique
        parameter values for a parameter such as q

    Returns
    -------
    dataparams: list, shape=(nparams,nwaveforms)
        A flattened list of sets of parameters
    """
    trainlistlen = 1
    nparams = len(paramlists) 
    for i in range(0,nparams):
        trainlistlen = trainlistlen * len(paramlists[i])
    dataparams = np.empty([nparams,trainlistlen],dtype=object)
    for i,items in enumerate(product(*paramlists)):
        dataparams[:,i] = items
    return dataparams

def paramdict_handler(input_paramdict,varname,value, chirp=False,mass=None):
    """
    Handles making the paramdict to be passed
    to makeFDWaveform using input variables
    that may not go into makeFDWaveform like q.

    Parameters
    ----------
    input_paramdict: dictionary
        A dictionary that will be updated with the correct
        value from value and varname
    varname: str
        name of variable
    value: 
        value of the variable
    chirp: bool
        if True, use chirp mass instead of 
        constant total mass
    
    Returns
    -------
    paramdict: dictionary
        A dictionary with updated values
    """
    paramdict = input_paramdict
    if varname=='q':
        if chirp==True:
            total_mass = mass*((value/(1.+value)**2)**(-3./5.)) 
        else:
            total_mass = mass
        paramdict['m1'] = total_mass*value/(1.+value)
        paramdict['m2'] = total_mass - paramdict['m1']
    elif varname=='chi1=chi2':
        paramdict['S1z']=value
        paramdict['S2z']=value
    else:
        paramdict[varname]==value
    return paramdict


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pars', type=str,nargs='+',
            help='parameters over which to generate data')
    parser.add_argument('--lbnd',type=float,nargs='+',
            help='lower bounds on parameters')
    parser.add_argument('--ubnd',type=float,nargs='+',
            help='upper bound on parameters')
    parser.add_argument('--N',type=int,nargs='+',
            help='number parameter values for each parameter')
    parser.add_argument('--mass',default=20.,type=float,
            help='chirp or total mass of the system')
    parser.add_argument('--chirp',type=bool,default=False,
            help='if true, assume mass=chirp mass rather than total mass')
    parser.add_argument('--outfile',type=str,default='WFdata.npy',
            help='name of .npy file where data is saved out to')
    
    options = parser.parse_args()
    pars,lbnd,ubnd,N,mass,chirp,outfile = \
            options.pars,options.lbnd,options.ubnd,options.N,options.mass,options.chirp,options.outfile
    npars = len(pars)
    
    paramlists = []
    for i in range(npars):
        paramlists.append(np.linspace(lbnd[i],ubnd[i],N[i]))
    flat_paramlist = make_flattened_paramlist_regular(paramlists)

    gA = wu.get_geometric_freq(g_low=0.004,g_high=0.15,dg=lambda f: 0.1*f)
    gPhi = wu.get_geometric_freq(g_low=0.004,g_high=0.15)

    amps = []
    phis = []
    fig,ax = plt.subplots(2,sharex=True)

    for i in range(flat_paramlist.shape[1]):
        paramdict = {'deltaF':0.05}
        for p in range(flat_paramlist.shape[0]):
            paramdict = paramdict_handler(paramdict,pars[p],flat_paramlist[p,i],chirp=chirp,mass=mass)  
        f,hp,hc = wu.make_FDwaveform(paramdict)
        Mf = f*(paramdict['m1']+paramdict['m2'])*lal.MTSUN_SI
        amp,phi = wu.interpolate_amp_phase(gA,gPhi,Mf,hp)
        amps.append(amp)
        phis.append(phi)
        ax[0].plot(gA,amp)
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].plot(gPhi,phi)
        ax[1].set_xscale('log')
    fig.savefig('../test/testfig.png')

    np.save(outfile,[gA,gPhi,flat_paramlist,amps,Phis])
