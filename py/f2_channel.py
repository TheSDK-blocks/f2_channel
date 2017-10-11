# f2_channel class 
#Thechannel model in this module is based on 802.11n channel models decribed in
# IEEE 802.11n-03/940r4 TGn Channel Models
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 10.10.2017 18:11
import numpy as np
import scipy.linalg as sli
import tempfile
import subprocess
import shlex
import time

from refptr import *
from thesdk import *

def laplacian_pdf(sigma,theta):
    #power angular spectrum
    Q=1/(1-np.exp(-np.sqrt(2)*(theta[-1]-theta[0])))
    #Q=1
    PAS=Q*np.exp(-np.sqrt(2)*np.abs(theta)/sigma)/(np.sqrt(2)*sigma)
    return PAS

def corrm(d,f,sigma,AoA):
     #Models defined only in the horizontal plane
     # IEEE 802.11n-03/940r4 TGn Channel Models
     #d is a Mx1 distance vector between antenna elements
     #f is the transmission frequency
     #sigma is the angular spread (standard deviation in radians.
     #AoA is th angle of arrival in radians
     c=299792458 #Speed of light, m/s
     lamda=c/f 
     dmat=sli.toeplitz(d,d)
     M=dmat.shape[0] # Number of antennas
     D=2*np.pi*dmat/lamda
     #RXX integ | phi -pi -pi cos(d*np.sin(phi)*laplacian(sigma,phi)dphi
     #RXY integ | phi -pi -pi sin(d*np.sin(phi)*laplacian(sigma,phi)dphi
     #Combine these to matrix
     phirange=np.linspace(-np.pi,np.pi,2**16)+AoA
     dphi=np.diff(phirange)[0]
     #There's an error due to numerical integration. With angle 0 the correlation must be 1
     #calculate that
     Kcorr=1/(np.sum(laplacian_pdf(sigma,phirange-AoA))*dphi)
     laplacianweightmat=np.ones((M,1))*laplacian_pdf(sigma,phirange-AoA)
     R=np.zeros((M,M),dtype='complex')
     for i in range(M): 
         R[i,:]=Kcorr*np.sum(np.exp(1j*D[i,:].reshape((-1,1))*np.sin(phirange))*laplacianweightmat,1)*dphi
     return R

def lambda2meter(distlambda,f):
    c=299792458 
    d=distlambda*c/f
    return d

def gen_dist_mat(d):
    #Missing scipy. fast replacement
    distmat=np.zeros((d.shape[0],d.shape[0]))
    distmat=

#Simple buffer template
class f2_channel(thesdk):

    def __init__(self,*arg): 
        self.proplist = [ 'Rs', 'channeldict' ];    #properties that can be propagated from parent
        self.Rs = 1;                 # sampling frequency
        self.iptr_A = refptr();
        self.model='py';             #can be set externally, but is not propagated
        self.channeldict= { 'model': 'buffer', 'bandwidth':100e6, 'frequency':0, 'distance':0}
        self._Z = refptr();
        self._classfile=__file__
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;

    def init(self):
        pass


    def run(self,*arg):
        if len(arg)>0:
            par=True      #flag for parallel processing
            queue=arg[0]  #multiprocessing.Queue as the first argument
        else:
            par=False

        if self.model=='py':
            if self.channeldict['model'] == 'buffer':
                out=self.buffer()

            if self.channeldict['model'] == 'awgn':
                out=self.awgn()

            if par:
                queue.put(out)
            self._Z.Value=out
        else: 
            print("ERROR: Only Python model currently available")
    
    def buffer(self):
        loss=np.sqrt(self.free_space_path_loss(self.channeldict['frequency'],self.channeldict['distance']))
        print(loss)
        out=np.array(loss*self.iptr_A.Value)
        print(out)
        return out

    def awgn(self):
        kb=1.3806485279e-23
        #noise power density in room temperature, 50 ohm load 
        noise_power_density=4*kb*290*50
        noise_rms_voltage=np.sqrt(noise_power_density*self.channeldict['bandwidth'])
        #complex noise
        noise_voltage=np.sqrt(0.5)*(np.random.normal(0,noise_rms_voltage,self.iptr_A.Value.shape)+1j*np.random.normal(0,noise_rms_voltage,self.iptr_A.Value.shape))
        #Add noise
        loss=np.sqrt(self.free_space_path_loss(self.channeldict['frequency'],self.channeldict['distance']))
        out=np.array(loss*self.iptr_A.Value+noise_voltage)
        return out

    def random_spatial(self):   
        pass

#Helper function

    def free_space_path_loss(self,frequency,distance):
        #The _power_ loss of the free space
        #Distance in meter
        c=299792458 #Speed of light, m/s
        if distance==0:
            loss=1
        else:
            loss=1/(4*np.pi*(distance)*frequency/c)**2
        return loss
