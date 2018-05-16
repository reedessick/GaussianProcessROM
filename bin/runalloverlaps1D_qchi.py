#!/bin/usr/env python

from model2D import *
import matplotlib.pyplot as plt

qlow = 1.
qhigh = 6.
nq = 15
chilow = -1.
chihigh = 1.
nchi = 12
fhzlow = 40.
df = 0.01
ncores = 16
chirp = True
kern_type = 'RBF'

GPR_param = 'q'
useEta = False
if GPR_param =='q':
    if useEta == True:
        etatrain = np.linspace(qlow/((1.+qlow)**2.), qhigh/((1+qhigh)**2.),nq)
        qtrain = np.divide(-(2.*etatrain-1)+np.sqrt(-4.*etatrain+1.),2.*etatrain)
    else:
        qtrain = np.linspace(qlow,qhigh,nq)
    #qlow = min(qtrain)
    #qhigh = max(qtrain)
    #qtrain = np.linspace(qlow,qhigh,nq)
    paramlist = [qtrain]
    params = ['q']
elif GPR_param=='chi':
    chitrain = np.linspace(chilow,chihigh,nchi)
    paramlist = [chitrain]
    params = ['chi1']
else:
    print 'GPR_param has to be either q or chi'
    raise Error
#make the training list:
trainlist = make_training_list_arb(paramlist,params)
print 'trainshape:',trainlist.shape
Mlow=20
gA = make_Mf(fhzlow, Mlow=Mlow, fmaxR=0.15, DPhi=lambda f: 0.1*f)
gPhi = make_Mf(fhzlow, Mlow=Mlow, fmaxR=0.15, DPhi=None)
print 'Making amp/phi table'
amps,phis = make_amps_phis_table_dict(trainlist,params,gA,gPhi,df,chirp=chirp,ncores=ncores)
print amps.shape,phis.shape,gA.shape
for i in np.random.randint(len(amps),size=15):
    plt.plot(gA,amps[i])
    plt.yscale('log')
    plt.xscale('log')
plt.savefig('testfig_A.png',dpi=400)
plt.figure()
for i in np.random.randint(len(phis),size=15):
    plt.plot(gPhi,phis[i])
    plt.xscale('log')
plt.savefig('testfig_P.png',dpi=400)

cA,cP,VA,VP,cA_vars,cP_vars,sA,sP = make_cAcP(amps, phis,getErrors=True,ErrorLevel=1e-4,ErrorLevel_Phi=.01,getSigma=True)

print 'cA min,max:',np.amin(cA),np.amax(cA)
print 'cP min,max:',np.amin(cP),np.amax(cP)
np.save('SIM_FILES/cA.npy',cA)
np.save('SIM_FILES/trainlist.npy',trainlist)
num_per_Mtrain=5
if GPR_param=='chi':
    z_gp = make_interp_list(nchi,num_per_Mtrain, chilow, chihigh)
    z = z_gp
    interplist_gp = make_training_list_arb([z],params)
    trainlist_gp = trainlist
    interplist = interplist_gp
elif GPR_param=='q':
    if useEta ==True:
        z_gp = make_interp_list(nq,num_per_Mtrain,np.amin(etatrain),np.amax(etatrain))
        z = np.divide(-(2.*z_gp-1)+np.sqrt(-4.*z_gp+1.),2.*z_gp)
        interplist = make_training_list_arb([z],params)
        interplist_gp = make_training_list_arb([z_gp],params)
        trainlist_gp = make_training_list_arb([etatrain],params)
    else:
        z_gp = make_interp_list(nq,num_per_Mtrain, qlow, qhigh)
        z = z_gp
        interplist_gp = make_training_list_arb([z_gp],params)
        trainlist_gp = trainlist
        interplist = make_training_list_arb([z],params)
print 'interplist shape:',interplist_gp.shape
#z = z+np.diff(z)[0]/2.
#z = make_interp_list(int((Mhigh-Mlow)/np.diff(trainlist)[0]),num_per_Mtrain, Mlow, Mhigh)
if GPR_param == 'q':
    # try setting a prior on the length scale between
    # the minimum and maximum distance between 
    # neighboring training points
    if kern_type == 'Matern':
        kernel_gpml_amp  = ConstantKernel(1.,(1e-1,1e3))*Matern(length_scale = 2,\
                length_scale_bounds=(0.5*np.amin(np.diff(qtrain)),2.*(np.amax(qtrain)-np.amin(qtrain))),nu=2.5)
        kernel_gpml_phase  = ConstantKernel(1.,(1e-1,1e3))*Matern(length_scale = 2,\
                length_scale_bounds=(0.5*np.amin(np.diff(qtrain)),2.*(np.amax(qtrain)-np.amin(qtrain))),nu=2.5)
    elif kern_type=='RBF':
        kernel_gpml_amp  = ConstantKernel(1.,(1e-1,1e3))*RBF(length_scale = 2,\
                length_scale_bounds=(0.1*np.amin(np.diff(qtrain)),20.*(np.amax(qtrain)-np.amin(qtrain))))
        kernel_gpml_phase  = ConstantKernel(1.,(1e-1,1e3))*RBF(length_scale = 2,\
                length_scale_bounds=(0.1*np.amin(np.diff(qtrain)),20.*(np.amax(qtrain)-np.amin(qtrain))))
    if useEta == True:
        kernel_gpml_amp  = ConstantKernel(1,(2e-2,1e3))*RBF(length_scale = 1,length_scale_bounds=(5e-4,1e5))
        kernel_gpml_phase = ConstantKernel(1,(1e-2,1e3))*RBF(length_scale = 1,length_scale_bounds=(5e-4,1e5))
    # Set the prior on hyperparameters.  the prior needs
    # to a list of tuples where each tuple is the
    # location and width of the lognormal distirbution of
    # each hyperparameter.  The order of the tuples corresponds
    # to the order of the kernel above.  ie. since kernel here is
    # defined as ConstantKernel*RBF, the ConstantKernel hyperparam
    # tuple goes first, then the RBF.
    prior = [(0.,0.5*np.log10(np.exp(1))),(np.log10(0.5*(np.amax(qtrain)-np.amin(qtrain))),1.*np.log10(np.exp(1)))] 

if GPR_param == 'chi':
    if kern_type == 'Matern':
        kernel_gpml_amp  = ConstantKernel(1.,(1e-1,1e3))*Matern(length_scale = .5,\
                length_scale_bounds=(0.5*np.amin(np.diff(chitrain)),2.*(np.amax(chitrain)-np.amin(chitrain))),nu=2.5)
        kernel_gpml_phase  = ConstantKernel(1.,(1e-1,1e3))*Matern(length_scale = .5,\
                length_scale_bounds=(0.5*np.amin(np.diff(chitrain)),2.*(np.amax(chitrain)-np.amin(chitrain))),nu=2.5)
    if kern_type == 'RBF':
        kernel_gpml_amp  = ConstantKernel(1.,(1e-1,1e3))*RBF(length_scale = .5,\
                length_scale_bounds=(0.5*np.amin(np.diff(chitrain)),2.*(np.amax(chitrain)-np.amin(chitrain))))
        kernel_gpml_phase  = ConstantKernel(1.,(1e-1,1e3))*RBF(length_scale = .5,\
                length_scale_bounds=(0.5*np.amin(np.diff(chitrain)),2.*(np.amax(chitrain)-np.amin(chitrain))))
     
    prior = [(0.,0.5*np.log10(np.exp(1))),(np.log10(0.5*(chihigh-chilow)),1.*np.log10(np.exp(1)))]

#cAinterp, cPinterp, cAinterp_std,cPinterp_std = perform_GPR(cA,cP,z,trainlist,oneD=True,kernel=kernel)
print 'doing interpolation....'
#cAinterp, cPinterp, cAinterp_std,cPinterp_std = perform_GPR_parallel(cA,cP,z,trainlist[0],oneD=True,kernel=kernel_gpml_amp,kernelPhase=kernel_gpml_phase,ncores=ncores,err=1e-20)
cAinterp, cPinterp, cAinterp_std,cPinterp_std = \
        perform_GPR_parallel(cA,cP,interplist_gp,trainlist_gp,\
        oneD=True,kernel=kernel_gpml_amp,kernelPhase=kernel_gpml_phase,\
        ncores=ncores,err=cA_vars,errPhi=cP_vars,prior=prior)
#cAinterp, cPinterp, cAinterp_std,cPinterp_std = perform_GPR(cA,cP,z,trainlist,oneD=True,kernel=kernel_gpml)
print 'A interp shape:',cAinterp.shape, '\nP interp shape:',cPinterp.shape,'\nA shape:',cA.shape, '\nP shape:',cP.shape

amps_realinterp,phis_realinterp = make_amps_phis_table_dict(interplist,params,gA,gPhi,df,chirp=chirp,ncores=ncores)
cA_realinterp = np.array(map(lambda A: np.dot(A, VA), amps_realinterp))
cP_realinterp = np.array(map(lambda P: np.dot(P, VP), phis_realinterp))
length_scales = np.empty([2,4])
for cnum,ctype in enumerate(['amp','phase']):
    if ctype=='amp':
        realinterp = cA_realinterp
        interp = cAinterp
        interp_std = cAinterp_std
    elif ctype=='phase':
        realinterp = cP_realinterp
        interp = cPinterp
        interp_std = cPinterp_std
    fig = plt.figure(figsize=(20,8))
    fig2,ax2 = plt.subplots()
    for i in range(0,4):
        GP = joblib.load('SIM_FILES/Coef_GP__'+ctype+'_'+str(i)+'.pkl')
        
        ax2.plot(GP._t,GP._y)
        
        myparams = GP._gp.kernel_.theta
        myparams = GP._gp.kernel_.get_params()
        for key in sorted(myparams): print('mykey',key)
        sigma = myparams['k1__constant_value']
        l = myparams['k2__length_scale']
        #l = myparams['length_scale']
        length_scales[cnum,i] = l
        ax = fig.add_axes([(i*.23)+0.05,.35,.17,.6]) 
        ax.plot(z,realinterp[:,i],color='r',lw=2,label='IMRPhenomD')
        ax.plot(z,interp[:,i],color='k',ls='--',lw=2,label='GPR Mean')
        ax.fill_between(z,interp[:,i]-3*interp_std[:,i],interp[:,i]+3*interp_std[:,i],color='gray',alpha=0.5)
        ax.set_xticklabels([])
        #ax.text(0.7, 0.4, '$l_{%i} = %2.2f$' % (i,l),
        #    transform=ax.transAxes,
        #    color='green', fontsize=15) 
        ax.set_title('$l_{%i} = %2.2f$' % (i,l),fontsize=14)
        frame1 = fig.add_axes([(i*.23)+0.05,.05,.17,.25])
        frame1.plot(z,np.absolute(realinterp[:,i]- interp[:,i]),color='r',lw=1.5)
        frame2 = frame1.twinx()
        frame2.plot(z,interp_std[:,i],color='k',ls='--',lw=1.5)
        frame1.set_yscale('log')
        try:
            frame2.set_yscale('log')
        except:
            pass
        if ctype=='amp':
            frame1.set_ylim([1e-27,1e-21])
            frame2.set_ylim([1e-27,1e-21])
        elif ctype=='phase':
            frame1.set_ylim([1e-5,2])
            frame2.set_ylim([1e-5,2])
        if i==0:
            ax.legend()
            ax.set_ylabel('Projection Coefficient Value',fontsize=14)
            frame1.set_ylabel('|residual|',fontsize=14,color='r')
        if i==3:
            frame2.set_ylabel('GPR Uncertainty',fontsize=14)
        if GPR_param=='chi':
            frame2.set_xlabel(r'$\chi_1 = \chi_2$',fontsize=18)
        elif GPR_param=='q':
            frame2.set_xlabel(r'$q$',fontsize=18)
    #fig.tight_layout()
    fig.savefig('plots/'+ctype+'_coeffs_'+GPR_param+'.png',dpi=400)
    fig2.savefig('plots/regularized_%s_%i.png'%(ctype,i),dpi=400)
 

np.save('clist1D_'+GPR_param+'.npy',[trainlist_gp,interplist_gp,cA,cP,cAinterp, cPinterp, cAinterp_std,cPinterp_std,VA,VP,gA,gPhi,cA_realinterp,cP_realinterp,length_scales])



randindA = np.random.randint(0,len(cA.T),7)
randindP = np.random.randint(0,len(cP.T),7)
errorbars = False

plt.figure()
for i in randindA:
    #print cAinterp_std[:,i]
    #print cPinterp_std[:,i]
    print i,':',cA.T[i],cAinterp[:,i]
    plt.plot(trainlist_gp[0],np.divide((cA.T[i]-np.mean(cAinterp[:,i])),np.std(cAinterp[:,i])),color='r',marker='s',ls='')
    if errorbars == False:
        plt.plot(z_gp,np.divide((cAinterp[:,i]-np.mean(cAinterp[:,i])),np.std(cAinterp[:,i])),color='b')
        #plt.fill_between(z,cAinterp[:,i]-cAinterp_std[:,i],cAinterp[:,i]+cAinterp_std[:,i],color='gray')
    else:
        plt.errorbar(z_gp,cAinterp[:,i],yerr=cAinterp_std[:,i],ls='--',label=str(i))
        print np.divide(cAinterp_std[:,i],cAinterp[:,i])
#plt.yscale('log')
plt.legend()
plt.savefig('plots/cA.png',dpi=400)
plt.close('all')
errorbars = True
plt.figure()
for i in randindP:
    plt.plot(trainlist_gp[0],cP.T[i],color='r',marker='s',ls='')
    if errorbars == False:
        plt.plot(z_gp,cPinterp[:,i],ls='--',color='b')
        plt.fill_between(z_gp,cPinterp[:,i]-cPinterp_std[:,i],cPinterp[:,i]+cPinterp_std[:,i],color='gray')
    else:
        plt.errorbar(z_gp,cPinterp[:,i],yerr=cPinterp_std[:,i],color='b')
#plt.yscale('log')
plt.savefig('plots/cP.png',dpi=400)
plt.legend()
plt.close('all')

#amps,phis = make_amps_phis_table_parallel(z,gA,gPhi,df,chirp=True,oneD=True,ncores=ncores)
amps,phis = make_amps_phis_table_dict(make_training_list_arb([z],params),params,gA,gPhi,df,chirp=chirp,ncores=ncores)
cA_real = np.array(map(lambda A: np.dot(A, VA), amps))
cP_real = np.array(map(lambda P: np.dot(P, VP), phis))
plt.figure()
for i in randindA:
    plt.plot(z_gp,np.divide(np.abs(cA_real[:,i]-cAinterp[:,i]),cA_real[:,i]),label=str(i))
#plt.yscale('log')
plt.legend()
    
#plt.yscale('log')
plt.savefig('plots/cA_err.png',dpi=400)

plt.close('all')

plt.figure()
for i in randindP:
    plt.plot(z_gp,np.divide(np.abs(cP_real[:,i]-cPinterp[:,i]),cP_real[:,i]),color='r')
#plt.yscale('log')

#plt.yscale('log')
plt.savefig('plots/cP_err.png',dpi=400)

plt.close('all')

noisecurve = np.genfromtxt('2015-10-01_H1_O1_Sensitivity_strain_asd.txt',names=['f','asd'],dtype=None)
#overlaps = get_all_overlaps(cAinterp,cPinterp, VA, VP, z, g,df,chirp,oneD=True)
print 'gettin dem overlaps'
overlaps = get_all_overlaps_parallel(cAinterp,cPinterp, VA, VP, z, gA,gPhi,df,chirp,params,oneD=True,ncores=ncores,noisecurve=noisecurve)
np.save('overlaps1D'+GPR_param+'.npy',overlaps)
#overlaps = get_all_overlaps(cAinterp,cPinterp, VA, VP, z, gA,gPhi,df,chirp,oneD=True)
plt.plot(z_gp,1-overlaps,label='GPR')
overlay_errors = False
if overlay_errors:
    def coeffs_to_amp(num):
        coverrors = np.dot(VA,np.dot(np.power(np.diag(cAinterp_std[num,:]),2),VA.T))
        errors = np.sqrt(np.diag(coverrors))
        wfamps =np.sum(cAinterp[num,:]*VA,axis=1) 
        return np.sum(np.divide(errors,wfamps))
    amp_gp_errors = np.array(map(coeffs_to_amp,range(0,cAinterp_std.shape[0])))
    amp_gp_errors = amp_gp_errors/np.amax(amp_gp_errors)
    plt.plot(z_gp,amp_gp_errors,color='k')
    def coeffs_to_phase(num):
        coverrors = np.dot(VP,np.dot(np.power(np.diag(cPinterp_std[num,:]),2),VP.T))
        errors = np.sqrt(np.diag(coverrors))
        wfphase =np.sum(cPinterp[num,:]*VP,axis=1)
        return np.sum(errors)
    phase_gp_errors = np.array(map(coeffs_to_phase,range(0,cPinterp_std.shape[0])))
    phase_gp_errors = phase_gp_errors/np.amax(phase_gp_errors)
    plt.plot(z_gp,phase_gp_errors,color='green')
#plt.plot(z,np.sum(cAinterp_std,axis=1)/np.amax(cAinterp,axis=1),label='GPR fractional error')
for train in trainlist_gp[0]:
        plt.axvline(train,color='r',ls='--')
plt.yscale('log')
if GPR_param=='chi':
    plt.xlabel(r'$\chi_1 = \chi_2$',fontsize=18)
    plt.title('Total Mass 20$M_\odot$, $q=1$',fontsize=18)
elif GPR_param=='q':
    plt.xlabel(r'$q$',fontsize=18)
    plt.title('Total Mass 20$M_\odot$, $\chi_1=\chi_2=0$',fontsize=18)
plt.ylabel('Mismatch',fontsize=16)
#plt.xscale('log')
plt.savefig('oneDoverlaps_'+GPR_param+'.png',dpi=1200)
#plt.savefig('oneDoverlaps.eps',dpi=1200)
