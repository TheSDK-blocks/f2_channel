# f2_channel class 
# The channel model in this module is based on 802.11n channel models decribed in
# IEEE 802.11n-03/940r4 TGn Channel Models
# Last modification by Marko Kosunen, marko.kosunen@aalto.fi, 12.04.2018 20:14
import numpy as np
import scipy.constants as con

from refptr import *
from thesdk import *
from f2_channel.functions import generate_channel_tap
from f2_channel.functions import generate_corr_mat
from f2_channel.functions import generate_los_mat
from f2_channel.functions import generate_lossless_channel
from f2_channel.functions import lambda2meter
from f2_channel.functions import channel_propagate
from f2_channel.functions import get_802_11n_channel_params
from f2_channel.functions import laplacian_pdf

#Order of definition: first the class, then the functions.
#Class should describe the parameters required by the funtions.

class f2_channel(thesdk):
    def __init__(self,*arg): 
        self.proplist = [ 'Rs', 'channeldir', 'Users', 'Rxantennalocations', 'frequency', 'channeldict', 'noisetemp' ];    #properties that can be propagated from parent
        self.Rs = 100e6; # sampling frequency
        self.frequency=1e9
        self.Users=2
        self.Rxantennalocations=np.r_[0]
        self.Channeldir='Uplink'
        self.noisetemp=290
        #Input pointer is a pointer to A matrix indexed as s(user,time, txantenna)
        #Channel is modeled as H(time,rxantenna, txantenna)
        #Thus, Srx is SH^T-> Srx(time,rxantenna)
        self.iptr_A = IO();
        self.model='py';             #can be set externally, but is not propagated
        self.channeldict= { 'model': 'lossless', 'distance':2 }
        self._Z = IO();          #This is an array of refpointers. One pointer for each Rx antenna
        #Thus, _Z.Data[k].Data is Srx(time)
        self._classfile=__file__
        self.H=np.array([])
        self.DEBUG= False
        if len(arg)>=1:
            parent=arg[0]
            self.copy_propval(parent,self.proplist)
            self.parent =parent;
        self.init()
        self.Rxantennas=len(self.Rxantennalocations)
        self._Z.Data=[IO() for i in range(self.Rxantennas)]
    def init(self):
        pass


    def run(self,*arg):
        #Parallel run not tested. Parallel signal generators rarely needed 
        if len(arg)>0:
            par=True      #flag for parallel processing
            queue=arg[0]  #multiprocessing.Queue as the first argument
        else:
            par=False

        #Its is up to transmitter or receiver where the user channels are merged and how
        #How to handel this? 
        #BS receiver->user signals are combines before receiver antenna array
        #Mobile receiver->user signals are combines before the transmitter antenna array
        if self.model=='py':
            self.print_log(type='I', msg="The channel model is %s " %(self.channeldict['model']))
            
            #test for lossless model
            if self.channeldict['model'] == 'lossless':
                self.lossless()
                out=self.propagate()
                for i in range(self.Rxantennas):
                    if par:
                        z=out[0,:,i]
                        z.shape=(-1,1)
                        queue.put(z)
    
                    self._Z.Data[i].Data=out[0,:,i]
                    #self._Z.Data[i].Data.shape=(-1,1)

            #Test for 802_11n models
            if any(map(lambda x: x== self.channeldict['model'],  ['A', 'B', 'C', 'D', 'E', 'F'])):
                if self.Rs < 100e6:
                    self.print_log(type='F', msg="Minimum sample frequency of 100Ms/S required for IEEE 802.11n channel models")
                else:
                    self.ch802_11n()
                    out=self.propagate()
                    for i in range(self.Rxantennas):
                        if par:
                            z=out[0,:,i]
                            z.shape=(-1,1)
                            queue.put(z)
                        self._Z.Data[i].Data=out[0,:,i]
                        #self._Z.Data[i].Data.shape=(-1,1)
        else: 
            print("ERROR: Only Python model currently available")

    def ch802_11n(self):
        Rxantennalocations=self.Rxantennalocations.shape[0]
        txantennas=self.iptr_A.Data.shape[1]
        #param_dict={'K':K, 'tau':tau, 'pdb':pdb, 'AS_Tx':AS_Tx, 'AoD':AoD, 'AS_Rx':AS_Rx, 'AoA':AoA}
        channel_dict={'Rs':self.Rs, 'model':self.channeldict['model'], 'frequency':self.frequency, 'Rxantennalocations':self.Rxantennalocations, 'distance':self.channeldict['distance']}

        #Users are in different locations, every user has a channel
        for i in range(self.Users):
            t=self.generate_802_11n_channel(channel_dict)
            if i==0:
               shape=t.shape
               H=np.zeros((self.Users,shape[0],shape[1],shape[2]),dtype='complex')
               H[i,:,:,:]=t
            else:
               H[i,:,:,:]=t
        self.H=H

    def lossless(self):
        Rxantennalocations=self.Rxantennalocations.shape[0]
        txantennas=self.iptr_A.Data.shape[1]
        channelmodel=self.channeldict['model']
        f=self.channeldict['frequency']
        #param_dict={'K':K, 'tau':tau, 'pdb':pdb, 'AS_Tx':AS_Tx, 'AoD':AoD, 'AS_Rx':AS_Rx, 'AoA':AoA}
        channel_dict={'Rs':self.Rs, 'model':channelmodel, 'frequency':self.frequency, 'Rxantennalocations':self.Rxantennalocations}
        
        #Users are in different locations, every user has a channel
        for i in range(self.Users):
            t=generate_lossless_channel(channel_dict)
            if i==0:
               shape=t.shape
               H=np.zeros((self.Users,shape[0],shape[1],shape[2]),dtype='complex')
               H[i,:,:,:]=t
            else:
               H[i,:,:,:]=t
        
        self.H=H


    def propagate(self):
        #Its is up to transmitter or receiver where the user channels are merged and how
        #How to handle this? 
        #BS receiver->user signals are combined before receiver antenna array
        #Mobile receiver->user signals are combined before the transmitter antenna array
        for i in range(self.Users):  
            t=channel_propagate(self.iptr_A.Data[i],self.H[i])
            if self.Channeldir=='Downlink': #Every user receives a dedicated signal
                if i==0:
                   shape=t.shape
                   srx=np.zeros((self.Users,shape[0],shape[1]),dtype='complex')
                   srx[i,:,:]=t
                else:
                   srx[i,:,:]=t
            if self.Channeldir=='Uplink': #BS Receives the sum of the users with different channels
                if i==0:
                   shape=t.shape
                   srx=np.zeros((1,shape[0],shape[1]),dtype='complex')
                   srx[0,:,:]=t
                else:
                   srx[0,:,:]=srx[0,:,:]+t

        #Add noise
        #    #noise power density in room temperature, 50 ohm load 
        noise_power_density=4*con.k*self.noisetemp*50
        
        #Bandwidth determined by sample frequency
        noise_rms_voltage=np.sqrt(noise_power_density*self.Rs) 
        msg="Adding %f uV RMS  noise corresponding to %f dBm power to 50 ohm resistor over bandwidth of %f MHz" %(noise_rms_voltage/1e-6, 10*np.log10(noise_rms_voltage**2/(50*1e-3)), self.Rs/1e6)
        self.print_log(type='I', msg=msg)
        
        #complex noise
        noise_voltage=np.sqrt(0.5)*(np.random.normal(0,noise_rms_voltage,srx.shape)+1j*np.random.normal(0,noise_rms_voltage,srx.shape))
        return srx+noise_voltage

    def generate_802_11n_channel(self,*arg): #{'Rs': 'model': 'Rxantennalocations': 'frequency': }
        Rs=arg[0]['Rs']   #Sampling rate
        model=arg[0]['model']
        Rxantennalocations=arg[0]['Rxantennalocations'] 
        frequency=arg[0]['frequency'] 
        distance=arg[0]['distance'] 

        antennasrx=Rxantennalocations.shape[0] #antennas of the receiver, currently only 1 antenna at the tx
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
        #Which one of below is correct 
        channel_param_dict['AoA']=np.random.rand(shape[0],shape[1])*360
        #channel_param_dict['AoA']=np.remainder(channel_param_dict['AoA']+np.ones_like(channel_param_dict['AoA'])*np.random.rand(1,1)*360,360)
        #channel_param_dict['AoA']=np.remainder(channel_param_dict['AoA']+np.ones_like(channel_param_dict['AoA'])*90,360)
        self.print_log(type='I', msg="AoA's %s" %(channel_param_dict['AoA']))

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
        Powerloss=self.free_space_path_loss(distance,frequency,channel_param_dict['lossdict'])
        Powerscale=np.sqrt(sum(sum(10**(channel_param_dict['pdb']/10))))
        
        
        normalizer=np.sqrt(sum(sum(np.abs(H)**2)))
        #self.print_log(type='I', msg="Scaling power with %s" %(Powerscale))
        self.print_log(type='I', msg="Scaling power with %s in order to balance received power to transmitted power" %(normalizer))
        self.print_log(type='I', msg="Applying %s dB free-space loss" %(10*np.log10(Powerloss)))
        return H/(np.sqrt(Powerloss)*normalizer)

    #Loss model
    def free_space_path_loss(self,distance,frequency,lossdict):
        #The _power_ loss of the free space
        #Distance in meters
        if distance==0:
            loss=1
        elif distance < lossdict['dbp']:
            loss=(4*np.pi*frequency/con.c*distance**lossdict['s1'])*(10**np.random.normal(0,lossdict['f1'],(1,1))/10)
        elif distance >=lossdict['dbp']:
            loss=(4*np.pi*frequency/con.c*lossdict['dbp']**lossdict['s1'])*(distance/lossdict['dbp'])**lossdict['s2']
            loss=loss*(10**(np.random.normal(0,lossdict['f2'],(1,1))/10))
        self.print_log(type='I', msg="Path loss is %s dB" %( 10*np.log10(loss)) )
        return loss


    
if __name__=="__main__":
    import sys
    f=1e9
    AoA=2*np.pi*45/360
    sigma=2*np.pi*30/360
    d=lambda2meter(np.array(range(4)),f)
    dictd=get_802_11n_channel_params('D')
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
    ch.iptr_A.Data=s
    ch.channeldict= { 'model': 'C' }
    ch.Rxantennalocations=np.r_[0, 0.3, 0.6]
    ch.ch802_11n()
    ch.run()
    print(ch._Z.Data)
    print(ch._Z.Data.shape)
    ch.Rxantennalocations=np.r_[0, 0.3, 0.6]
    ch.channeldict= { 'model': 'lossless' }
    ch.run()
    #print(ch.H)
    #print(ch.H.shape)
    #print(ch.iptr_A.Data)
    print(ch._Z.Data)
    print(ch._Z.Data.shape)

