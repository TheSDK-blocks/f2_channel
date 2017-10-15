# f2_channel class 
#Thechannel model in this module is based on 802.11n channel models decribed in
# IEEE 802.11n-03/940r4 TGn Channel Models
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 14.10.2017 20:55
import sys
sys.path.append ('/home/projects/fader/TheSDK/Entities/refptr/py')
sys.path.append ('/home/projects/fader/TheSDK/Entities/thesdk/py')
import numpy as np
from numpy.lib import scimath as scm
import scipy.constants as con
import scipy.linalg as sli
#import tempfile
#import subprocess
#import shlex
#import time

from refptr import *
from thesdk import *


#Order of definition: first the class, then the functions.
#Class should describe the parameters required by the funtions.

class f2_channel(thesdk):
    def __init__(self,*arg): 
        self.proplist = [ 'Rs', 'Rxantennalocations', 'frequency' 'channeldict' ];    #properties that can be propagated from parent
        self.Rs = 100e6; # sampling frequency
        self.frequency=1e9
        self.users=2
        self.Rxantennalocations=np.r_[0]
        #input pointer is a pointer to A matrix indexed as s(user,time, txantenna)
        #Channelis modeled as H(time,rxantenna, txantenna)
        #Thus, Srx is SH^T-> Srx(time,rxantenna)
        self.iptr_A = refptr();
        self.model='py';             #can be set externally, but is not propagated
        self.channeldict= { 'model': 'buffer', 'bandwidth':100e6, 'distance':2 }
        self._Z = refptr();
        self._classfile=__file__
        self.H=np.array([])
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
        #print(loss)
        out=np.array(loss*self.iptr_A.Value)
        #print(out)
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

    def ch802_11n(self):   
        users=self.iptr_A.Value.shape[0]
        Rxantennalocations=self.Rxantennalocations.shape[0]
        txantennas=self.iptr_A.Value.shape[1]
        channelmodel=self.channeldict['model']
        f=self.channeldict['frequency']
        #param_dict={'K':K, 'tau':tau, 'pdb':pdb, 'AS_Tx':AS_Tx, 'AoD':AoD, 'AS_Rx':AS_Rx, 'AoA':AoA}
        param_dict=get_802_11n_channel_params(channelmodel)
        channel_dict={'Rs':self.Rs, 'model':channelmodel, 'frequency':self.frequency, 'Rxantennalocations':self.Rxantennalocations}
        for i in range(self.users):
            t=generate_802_11n_channel(channel_dict)
            if i==0:
               shape=t.shape
               H=np.zeros((self.users,shape[0],shape[1],shape[2]),dtype='complex')
               H[i,:,:,:]=t
            else:
               H[i,:,:,:]=t
        self.H=H
     
    def run(self):
        for i in range(self.users):
            t=channel_propagate(self.iptr_A.Value,self.H[i])
            if i==0:
               shape=t.shape
               srx=np.zeros((self.users,shape[0],shape[1]),dtype='complex')
               srx[i,:,:]=t
            else:
               srx[i,:,:]=t
        self._Z.Value=srx

#Helper functions
def generate_802_11n_channel(*arg): #{'Rs': 'model': 'Rxantennalocations': 'frequency': }
    Rs=arg[0]['Rs']   #Sampling rate
    model=arg[0]['model']
    Rxantennalocations=arg[0]['Rxantennalocations'] 
    frequency=arg[0]['frequency'] 
    antennasrx=Rxantennalocations.size #antennas of the receiver, currently only 1 antenna at the tx
    antennastx=1
    #H matrix structure receiver antennas on rows, tx antennas on columns
    channel_param_dict=get_802_11n_channel_params(model) #param_dict={'K':K, 'tau':tau, 'pdb':pdb, 'AS_Tx':AS_Tx, 'AoD':AoD, 'AS_Rx':AS_Rx, 'AoA':AoA}

    tau=channel_param_dict['tau']
    tauind=np.round(tau*Rs).astype('int')
    chanlen=tauind[-1]+1 #Lenght of the channel in samples at Rs

    #Create a channel fro a single transmitter to a single receiver
    #Currently the tranmitter has 1 antenna, receiver has Multiple antennae
    H=np.zeros((chanlen,antennasrx,antennastx),dtype='complex')
    
    
    #Here we can generate the random Angles of Arrival.
    #Let Angle of departure be o degrees by, and angle of arrival be random
    channel_param_dict['AoD']=np.zeros(channel_param_dict['AoD'].shape)
    shape=channel_param_dict['AoA'].shape
    channel_param_dict['AoA']=np.random.rand(shape[0],shape[1])*360

    #For each channel there are multiple clusters of taps
    for cluster_index in range(channel_param_dict['AoA'].shape[0]):
        #taps inside the cluster
        for tap_index in range(tau.shape[0]):
            tapdict={'Rxantennalocations': Rxantennalocations, 'frequency': frequency, 'K':channel_param_dict['K'][tap_index], 
                    'tau':channel_param_dict['tau'][tap_index], 'pdb':channel_param_dict['pdb'][cluster_index][tap_index], 
                     'AS_Tx':channel_param_dict['AS_Tx'][cluster_index][tap_index], 'AoD':channel_param_dict['AoD'][cluster_index][tap_index], 
                    'AS_Rx':channel_param_dict['AS_Rx'][cluster_index][tap_index], 'AoA':channel_param_dict['AoA'][cluster_index][tap_index] } 
            #Arguments for generate_channel
            #distance array, frequency, {'AS_RX': 'AoA': 'K': 'pdb': }

            #This is really fucked up way. Why on earth th column vector H[:,x] does not remain as a column vector.
            shape=H[tauind[tap_index],:,:].shape
            H[tauind[tap_index],:,:]=H[tauind[tap_index],:,:]+generate_channel_tap(tapdict).reshape(shape)
    return H

def generate_channel_tap(*arg):
    #arg={'Rxantennalocations': , 'frequency':, 'K':, 'tau':, 'pdb':, 'AS_Tx':, 'AoD':, 'AS_Rx':, 'AoA': } 
    #Models defined only in the horizontal plane
    # IEEE 802.11n-03/940r4 TGn Channel Models
    #d is a Mx1 distance vector between antenna elements
    #f is the transmission frequency
    #sigma is the angular spread (standard deviation in radians.
    #AoA is th angle of arrival in degrees
    #Add the receiver array at some point
    K=arg[0]['K']       #Rician factor
    P=10**arg[0]['pdb']
    matdict=arg[0] #we can pass the arguments as is
    X=generate_corr_mat(matdict)
    L=generate_los_mat(matdict)
    channel_tap=np.sqrt(P)*(np.sqrt(K/(K+1))*L+np.sqrt(1/(K+1))*X)
    return channel_tap

def generate_corr_mat(*arg): 
    #arg={'Rxantennalocations': , 'frequency':, 'K':, 'tau':, 'pdb':, 'AS_Tx':, 'AoD':, 'AS_Rx':, 'AoA': } 
    #The same dictionary OK as an argument as for generate channel tap
    #Models defined only in the horizontal plane
    # IEEE 802.11n-03/940r4 TGn Channel Models
    #d is a Mx1 distance vector between antenna elements
    #f is the transmission frequency
    #sigma is the angular spread (standard deviation in radians.
    #AoA is th angle of arrival in radians
    #Add the receiver array at some point
    Rxantennalocations=arg[0]['Rxantennalocations']  # Distance array for the receiver antenna
    Txantennalocations=np.array([0])                 # Only one TX antenna
    frequency=arg[0]['frequency']                            # Frequency
    lamda=con.c/frequency 
    sigmarx=arg[0]['AS_Rx']                          # Angle spread for the receiver
    AoA=arg[0]['AoA']                                #Angle of arrival for received     in degrees
    dmatrx=sli.toeplitz(Rxantennalocations,Rxantennalocations)
    rxantennas=dmatrx.shape[0] # Number of receive antennas
    txantennas=1 #number of transmit antennas
    Drx=2*np.pi*dmatrx/lamda
    #RXX integ | phi -pi -pi cos(d*np.sin(phi)*laplacian(sigma,phi)dphi
    #RXY integ | phi -pi -pi sin(d*np.sin(phi)*laplacian(sigma,phi)dphi

    #Combine these to matrix
    phirangerx=np.linspace(-np.pi,np.pi,2**16)+2*np.pi/360*AoA
    dphirx=np.diff(phirangerx)[0]
    
    #There's an error due to numerical integration. With angle 0 the correlation must be 1
    #calculate that
    Kcorrrx=1/(np.sum(laplacian_pdf(sigma,phirangerx-2*np.pi/360*AoA))*dphirx)
    laplacianweightmatrx=np.ones((rxantennas,1))@laplacian_pdf(sigmarx,phirangerx-2*np.pi/360*AoA)
    Rrx=np.zeros((rxantennas,rxantennas),dtype='complex')
    for i in range(rxantennas): 
        Rrx[i,:]=Kcorrrx*np.sum(np.exp(1j*Drx[i,:].reshape((-1,1))*np.sin(phirangerx))*laplacianweightmatrx,1)*dphirx

    #Would require similar computations if the TX would be modeled
    Rtx=np.diagflat(np.ones((txantennas,1)))

    #Random matrix
    Hiid=1/np.sqrt(2)*(np.random.randn(rxantennas,txantennas)+1j*np.random.rand(rxantennas,txantennas))
    #Correlation matrix 
    X=scm.sqrt(Rrx)@Hiid@scm.sqrt(Rtx)
    return X

def generate_los_mat(*arg): #Distance array, frequency, AoA
    #arg={'Rxantennalocations': , 'frequency':, 'K':, 'tau':, 'pdb':, 'AS_Tx':, 'AoD':, 'AS_Rx':, 'AoA': } 
    #The same dictionary OK as an argument as for generate channel tap
    #Models defined only in the horizontal plane
    # IEEE 802.11n-03/940r4 TGn Channel Models
    #Rxantennalocations is a Mx1 distance vector between antenna elements
    #frequency is the transmission frequency
    #AS_Rx is the angular spread (standard deviation in degrees.
    #AoA is th angle of arrival in degrees
    #Add the receiver array at some point
    Rxantennalocations=arg[0]['Rxantennalocations']  # Distance array for the receiver antenna
    Txantennalocations=np.array([0])                 # Only one TX antenna
    frequency=arg[0]['frequency']                            # Frequency
    lamda=con.c/frequency 
    sigmarx=arg[0]['AS_Rx']                          # Angle spread for the receiver
    AoA=arg[0]['AoA']                                #Angle of arrival for received in degrees
    AoD=np.r_[0]
    lamda=con.c/frequency 
    Drx=2*np.pi*Rxantennalocations/lamda*np.sin(2*np.pi/360*AoA) #Relative phase shift in receiver array
    Dtx=2*np.pi*Txantennalocations/lamda*np.sin(2*np.pi/360*AoD) #Relative phase shift in transmitter array
    LOS_vectorrx=np.exp(-1j*Drx)
    LOS_vectorrx=LOS_vectorrx.reshape((-1,1))
    LOS_vectortx=np.exp(1j*Dtx)
    LOS_vectortx=LOS_vectortx.reshape((-1,1))
    LOS_mat=LOS_vectorrx@LOS_vectortx.transpose()
    return LOS_mat

def generate_pure_channel(*arg):
    model=arg[0]['model']
    d=arg[0]['d'] 
    f=arg[0]['f'] 
    AoA=0
    antennasrx=d.size #antennas of the receiver, currently only 1 antenna at the tx
    antennastx=1
    H=generate_los_mat(d,f,AoA)
    return H

def lambda2meter(distlambda,f):
    d=np.array([distlambda*con.c/f])
    return d

def channel_propagate(signal,H):
    #pass
    #Calculate the convolution of the 3D matrix filter
    #y(n)=SUM s(n-k)@H(k,:,:).T  
    convlen=s.shape[0]+H.shape[0]-1
    srx=np.zeros((convlen,H.shape[1]))
    
    for i in range(H.shape[0]): #0th dim is the "time", k of the filter in
        zt=np.zeros((i,H.shape[1]))
        zt.shape=(-1,H.shape[1])
        zb=np.zeros((H.shape[0]-1,H.shape[1]))
        zb.shape=(-1,H.shape[1])
        s_shift=np.r_['0',np.zeros((i,H.shape[1]),dtype='complex'),s@H[i,:,:].T,np.zeros((H.shape[0]-1-i,H.shape[1]))]
        srx=srx+s_shift
    return srx 



def get_802_11n_channel_params(model):
# This function hard-codes the WLAN 802.11n channel model parameters and
# returns the ones corresponding to the desired channel model. 
#param_dict={'K':K, 'tau':tau, 'pdb':pdb, 'AS_Tx':AS_Tx, 'AoD':AoD, 'AS_Rx':AS_Rx, 'AoA':AoA}

    if model=='A':
        
        tau = np.array([0])
        K=np.zeros(tau.size) #K-factor for Line-of-sight
        pdb = np.array([0])
        AoA = np.array([45])
        AS_Rx = np.array([40])
        AoD = np.array([45])
        AS_Tx = np.array([40])
        
    elif model=='B':
        
        tau = np.array([0,10,20,30,40,50,60,70,80]) * 1e-9 # Path delays, in seconds
        tau.reshape((1,-1))
        K=np.zeros(tau.size) #K-factor for Line-of-sight
        
        # Average path gains of cluster, in dB
        pdb1 = np.array([0,-5.4,-10.8,-16.2,-21.7,-1*np.inf,-1*np.inf,-1*np.inf,-1*np.inf])
        pdb2 = np.array([-1*np.inf,-1*np.inf,-3.2,-6.3,-9.4,-12.5,-15.6,-18.7,-21.8])
        #   Angular spreads
        AS_Tx_C1 = np.array([14.4,14.4,14.4,14.4,14.4,-1*np.inf,-1*np.inf,-1*np.inf,-1*np.inf])
        AS_Tx_C2 = np.array([-1*np.inf,-1*np.inf,25.4,25.4,25.4,25.4,25.4,25.4,25.4])
        #   Mean angles of departure
        AoD_C1 = np.array([225.1,225.1,225.1,225.1,225.1,-1*np.inf,-1*np.inf,-1*np.inf,-1*np.inf])
        AoD_C2 = np.array([-1*np.inf,-1*np.inf,106.5,106.5,106.5,106.5,106.5,106.5,106.5])

        # Spatial parameters on receiver side:
        #   Angular spreads
        AS_Rx_C1 = np.array([14.4,14.4,14.4,14.4,14.4,-1*np.inf,-1*np.inf,-1*np.inf,-1*np.inf])
        AS_Rx_C2 = np.array([-1*np.inf,-1*np.inf,25.2,25.2,25.2,25.2,25.2,25.2,25.2])
        #   Mean angles of arrival
        AoA_C1 = np.array([4.3,4.3,4.3,4.3,4.3,-1*np.inf,-1*np.inf,-1*np.inf,-1*np.inf])
        AoA_C2 = np.array([-1*np.inf,-1*np.inf,118.4,118.4,118.4,118.4,118.4,118.4,118.4])
        
        pdb = np.array([pdb1,pdb2])
        AS_Tx = np.array([AS_Tx_C1, AS_Tx_C2])
        AoD = np.array([AoD_C1, AoD_C2])
        AS_Rx = np.array([AS_Rx_C1, AS_Rx_C2])
        AoA = np.array([AoA_C1, AoA_C2])
        
    elif model=='C':
        
        tau = np.array([0,10,20,30,40,50,60,70,80,90,110,140,170,200]) * 1e-9
        K=np.zeros(tau.size) #K-factor for Line-of-sight
        
        pdb1 = np.array([0,-2.1,-4.3,-6.5,-8.6,-10.8,-13.0,-15.2,-17.3,-19.5],ndmin=2)
        AoA1 = 290.3*np.ones(pdb1.shape)
        AS_Rx1 = 24.6*np.ones(pdb1.shape)
        AoD1 = 13.5*np.ones(pdb1.shape)
        AS_Tx1 = 24.7*np.ones(pdb1.shape)
        
        pdb2 = np.array([-5.0,-7.2,-9.3,-11.5,-13.7,-15.8,-18.0,-20.2],ndmin=2)
        AoA2 = 332.3*np.ones(pdb2.shape)
        AS_Rx2 = 22.4*np.ones(pdb2.shape)
        AoD2 = 56.4*np.ones(pdb2.shape)
        AS_Tx2 = 22.5*np.ones(pdb2.shape)
        
        pdb1 = np.r_['1',pdb1,-1*np.inf*np.ones((1,4))]
        AoA1 = np.r_['1',AoA1,-1*np.inf*np.ones((1,4))]
        AS_Rx1 = np.r_['1',AS_Rx1,-1*np.inf*np.ones((1,4))]
        AoD1 = np.r_['1',AoD1,-1*np.inf*np.ones((1,4))]
        AS_Tx1 = np.r_['1',AS_Tx1,-1*np.inf*np.ones((1,4))]
        
        pdb2 = np.r_['1',-1*np.inf*np.ones((1,6)),pdb2]
        AoA2 = np.r_['1',-1*np.inf*np.ones((1,6)),AoA2]
        AS_Rx2 = np.r_['1',-1*np.inf*np.ones((1,6)),AS_Rx2]
        AoD2 = np.r_['1',-1*np.inf*np.ones((1,6)),AoD2]
        AS_Tx2 = np.r_['1',-1*np.inf*np.ones((1,6)),AS_Tx2]
        
        pdb = np.r_['0',pdb1, pdb2]
        AoA = np.r_['0',AoA1, AoA2]
        AS_Rx = np.r_['0',AS_Rx1, AS_Rx2]
        AoD = np.r_['0',AoD1, AoD2]
        AS_Tx = np.r_['0',AS_Tx1, AS_Tx2]
        
    elif model=='D':
            
        tau = np.array([0,10,20,30,40,50,60,70,80,90,110,140,170,200,240,290,340,390]) * 1e-9
        tau.reshape((1,-1))
        K=np.zeros(tau.size) #K-factor for Line-of-sight
        K[0]=3
        pdb1 = np.array([0,-0.9,-1.7,-2.6,-3.5,-4.3,-5.2,-6.1,-6.9,-7.8,-9.0,-11.1,-13.7,-16.3,-19.3,-23.2])
        pdb2 = np.array([-6.6,-9.5,-12.1,-14.7,-17.4,-21.9,-25.5])
        pdb3 = np.array([-18.8,-23.2,-25.2,-26.7]) # path losses vector
        
        ASt1 = 27.4*np.ones(pdb1.shape)
        ASt2 = 32.1*np.ones(pdb2.shape)
        ASt3 = 36.8*np.ones(pdb3.shape)
        
        ASt1 = np.array([ASt1,-1*np.inf,-1*np.inf])
        ASt2 = np.array([-1*np.inf*np.ones((1,10)),ASt2,-1*np.inf])
        ASt3 = np.array([-1*np.inf*np.ones((1,14)),ASt3])
        AS_Tx = np.array([ASt1, ASt2, ASt3]) # Tx angular spread vector
        
        ASr1 = 27.7*np.ones(pdb1.shape)
        ASr2 = 31.4*np.ones(pdb2.shape)
        ASr3 = 37.4*np.ones(pdb3.shape)
        
        ASr1 = np.array([ASr1,-1*np.inf,-1*np.inf])
        ASr2 = np.array([-1*np.inf*np.ones((1,10)),ASr2,-1*np.inf])
        ASr3 = np.array([-1*np.inf*np.ones((1,14)),ASr3])
        AS_Rx = np.array([ASr1, ASr2, ASr3]) # Rx angular spread vector
        
        AoD1 = 332.1*np.ones(pdb1.shape)
        AoD2 = 49.3*np.ones(pdb2.shape)
        AoD3 = 275.9*np.ones(pdb3.shape)
        
        AoD1 = np.array([AoD1,-1*np.inf,-1*np.inf])
        AoD2 = np.array([-1*np.inf*np.ones((1,10)),AoD2,-1*np.inf])
        AoD3 = np.array([-1*np.inf*np.ones((1,14)),AoD3])
        AoD = np.array([AoD1, AoD2, AoD3]) # Tx angles of departure
        
        AoA1 = 158.9*np.ones(pdb1.shape)
        AoA2 = 320.2*np.ones(pdb2.shape)
        AoA3 = 276.1*np.ones(pdb3.shape)
        
        AoA1 = np.array([AoA1,-1*np.inf,-1*np.inf])
        AoA2 = np.array([-1*np.inf*np.ones((1,10)),AoA2,-1*np.inf])
        AoA3 = np.array([-1*np.inf*np.ones((1,14)),AoA3])
        AoA = np.array([AoA1, AoA2, AoA3]) # Rx angles of arrival
        
        pdb1 = np.array([pdb1,-1*np.inf,-1*np.inf])
        pdb2 = np.array([-1*np.inf*np.ones((1,10)),pdb2,-1*np.inf])
        pdb3 = np.array([-1*np.inf*np.ones((1,14)),pdb3])
        pdb = np.array([pdb1,pdb2,pdb3]) # path loss vector
        
    elif model=='E':
        
        tau = np.array([0,10,20,30,50,80,110,140,180,230,280,330,380,430,490,560,640,730]) * 1e-9
        tau.reshape((1,-1))
        K=np.zeros(tau.size) #K-factor for Line-of-sight
        K[0]=6
        pdb1 = np.array([-2.6,-3.0,-3.5,-3.9,-4.5,-5.6,-6.9,-8.2,-9.8,-11.7,-13.9,-16.1,-18.3,-20.5,-22.9])
        AoA1 = 163.7*np.ones(pdb1.shape)
        AS_Rx1 = 35.8*np.ones(pdb1.shape)
        AoD1 = 105.6*np.ones(pdb1.shape)
        AS_Tx1 = 36.1*np.ones(pdb1.shape)
        
        pdb2 = np.array([-1.8,-3.2,-4.5,-5.8,-7.1,-9.9,-10.3,-14.3,-14.7,-18.7,-19.9,-22.4])
        AoA2 = 251.8*np.ones(pdb2.shape)
        AS_Rx2 = 41.6*np.ones(pdb2.shape)
        AoD2 = 293.1*np.ones(pdb2.shape)
        AS_Tx2 = 42.5*np.ones(pdb2.shape)
        
        pdb3 = np.array([-7.9,-9.6,-14.2,-13.8,-18.6,-18.1,-22.8])
        AoA3 = 80.0*np.ones(pdb3.shape)
        AS_Rx3 = 37.4*np.ones(pdb3.shape)
        AoD3 = 61.9*np.ones(pdb3.shape)
        AS_Tx3 = 38.0*np.ones(pdb3.shape)
        
        pdb4 = np.array([-20.6,-20.5,-20.7,-24.6])
        AoA4 = 182.0*np.ones(pdb4.shape)
        AS_Rx4 = 40.3*np.ones(pdb4.shape)
        AoD4 = 275.7*np.ones(pdb4.shape)
        AS_Tx4 = 38.7*np.ones(pdb4.shape)
        
        pdb1 = np.array([pdb1,-1*np.inf*np.ones((1,3))])
        AoA1 = np.array([AoA1,-1*np.inf*np.ones((1,3))])
        AS_Rx1 = np.array([AS_Rx1,-1*np.inf*np.ones((1,3))])
        AoD1 = np.array([AoD1,-1*np.inf*np.ones((1,3))])
        AS_Tx1 = np.array([AS_Tx1,-1*np.inf*np.ones((1,3))])
        
        pdb2 = np.array([-1*np.inf*np.ones((1,4)),pdb2,-1*np.inf*np.ones((1,2))])
        AoA2 = np.array([-1*np.inf*np.ones((1,4)),AoA2,-1*np.inf*np.ones((1,2))])
        AS_Rx2 = np.array([-1*np.inf*np.ones((1,4)),AS_Rx2,-1*np.inf*np.ones((1,2))])
        AoD2 = np.array([-1*np.inf*np.ones((1,4)),AoD2,-1*np.inf*np.ones((1,2))])
        AS_Tx2 = np.array([-1*np.inf*np.ones((1,4)),AS_Tx2,-1*np.inf*np.ones((1,2))])
        
        pdb3 = np.array([-1*np.inf*np.ones((1,8)),pdb3,-1*np.inf*np.ones((1,3))])
        AoA3 = np.array([-1*np.inf*np.ones((1,8)),AoA3,-1*np.inf*np.ones((1,3))])
        AS_Rx3 = np.array([-1*np.inf*np.ones((1,8)),AS_Rx3,-1*np.inf*np.ones((1,3))])
        AoD3 = np.array([-1*np.inf*np.ones((1,8)),AoD3,-1*np.inf*np.ones((1,3))])
        AS_Tx3 = np.array([-1*np.inf*np.ones((1,8)),AS_Tx3,-1*np.inf*np.ones((1,3))])
        
        pdb4 = np.array([-1*np.inf*np.ones((1,14)),pdb4])
        AoA4 = np.array([-1*np.inf*np.ones((1,14)),AoA4])
        AS_Rx4 = np.array([-1*np.inf*np.ones((1,14)),AS_Rx4])
        AoD4 = np.array([-1*np.inf*np.ones((1,14)),AoD4])
        AS_Tx4 = np.array([-1*np.inf*np.ones((1,14)),AS_Tx4])
        
        pdb = np.array([pdb1, pdb2, pdb3, pdb4])
        AoA = np.array([AoA1, AoA2, AoA3, AoA4])
        AS_Rx = np.array([AS_Rx1, AS_Rx2, AS_Rx3, AS_Rx4])
        AoD = np.array([AoD1, AoD2, AoD3, AoD4])
        AS_Tx = np.array([AS_Tx1, AS_Tx2, AS_Tx3, AS_Tx4])

  #  Still missing model F
  #  elif model=='F':
  #      pass


    param_dict={'K':K, 'tau':tau, 'pdb':pdb, 'AS_Tx':AS_Tx, 'AoD':AoD, 'AS_Rx':AS_Rx, 'AoA':AoA}
    return param_dict

def free_space_path_loss(self,frequency,distance):
    #The _power_ loss of the free space
    #Distance in meter
    c=299792458 #Speed of light, m/s
    if distance==0:
        loss=1
    else:
        loss=1/(4*np.pi*(distance)*frequency/c)**2
    return loss

def laplacian_pdf(sigma,theta):
    #power angular spectrum
    Q=1/(1-np.exp(-np.sqrt(2)*(theta[-1]-theta[0])))
    #Q=1
    PAS=Q*np.exp(-np.sqrt(2)*np.abs(theta)/sigma)/(np.sqrt(2)*sigma)
    return PAS.reshape((1,-1))
    
if __name__=="__main__":
    import sys
    f=1e9
    AoA=2*np.pi*45/360
    sigma=2*np.pi*30/360
    d=lambda2meter(np.array(range(4)),f)
    dictd=get_802_11n_channel_params('D')
    #print(dictd)
    #arg={'Rxantennalocations': , 'frequency':, 'K':, 'tau':, 'pdb':, 'AS_Tx':, 'AoD':, 'AS_Rx':, 'AoA': } 
    argdict={'Rxantennalocations': np.r_[0] , 'frequency': 1e9, 'K':0.5, 'tau':dictd['tau'], 'pdb':dictd['pdb'][0][0], 'AS_Tx':30, 'AoD':0, 'AS_Rx':45, 'AoA':AoA } 
    #print(argdict)
    #print(argdict['Rxantennalocations'])
    X=generate_corr_mat(argdict)
    L=generate_los_mat(argdict)
    H1=generate_channel_tap(argdict)
    chdict={'Rs':100e6, 'model':'C', 'frequency':f, 'Rxantennalocations':d}
    H1=generate_802_11n_channel(chdict)
    #print(X)
    #print(X.shape)
    #print(L)
    #print(L.shape)
    #print(H)
    #print(H.shape)
    #print(np.sum(np.abs(H)**2))
    #print(H1)
    #Channel to first antenna
    s=np.r_[range(100)]; s.shape=(-1,1)
    srx=channel_propagate(s,H1)
    #print(H1)
    #print(H1[1,:])
    #print(H1.shape)
    #print(srx)
    ch=f2_channel()
    ch.iptr_A.Value=s
    ch.channeldict= { 'model': 'C', 'bandwidth':100e6, 'frequency':1e9}
    ch.Rxantennalocations=np.r_[0, 0.3, 0.6]
    ch.ch802_11n()
    ch.run()
    #print(ch.H)
    #print(ch.H.shape)
    #print(ch.iptr_A.Value)
    print(ch._Z.Value)
    print(ch._Z.Value.shape)

