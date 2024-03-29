#Helper functions
import sys
import numpy as np
from numpy.lib import scimath as scm
import scipy.constants as con
import scipy.linalg as sli

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
    P=10**(arg[0]['pdb']/10.0)
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
    frequency=arg[0]['frequency']                    # Frequency
    lamda=con.c/frequency 
    sigmarx=arg[0]['AS_Rx']*2*np.pi/360              # Angle spread for the receiver in degrees
    AoA=arg[0]['AoA']                                #Angle of arrival for received in degrees
    dmatrx=sli.toeplitz(Rxantennalocations,Rxantennalocations)
    rxantennas=dmatrx.shape[0] # Number of receive antennas
    txantennas=1 #number of transmit antennas
    Drx=2*np.pi*dmatrx/lamda

    if (sigmarx !=float('-inf')):

        #Combine these to matrix
        phirangerx=np.linspace(-np.pi,np.pi,2**16)+2*np.pi/360*AoA
        dphirx=np.diff(phirangerx)[0]
    
        #There's an error due to numerical integration. With angle 0 the correlation must be 1
        #calculate that. Ff the sigmarx =-inf, the is undefined 
        Kcorrrx=1/(np.sum(laplacian_pdf(sigmarx,phirangerx-2*np.pi/360*AoA))*dphirx)
        laplacianweightmatrx=np.ones((rxantennas,1))@laplacian_pdf(sigmarx,phirangerx-2*np.pi/360*AoA)
        Rrx=np.zeros((rxantennas,rxantennas),dtype='complex')
        for i in range(rxantennas): 
            Rrx[i,:]=Kcorrrx*np.sum(np.exp(1j*Drx[i,:].reshape((-1,1))*np.sin(phirangerx))*laplacianweightmatrx,1)*dphirx
    
    else:
        Rrx=np.zeros((rxantennas,rxantennas),dtype='complex')
    
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
    if (sigmarx !=float('-inf')):
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
    else:
        Drx=np.zeros_like(Rxantennalocations)
        Dtx=np.zeros_like(Txantennalocations)
        LOS_vectorrx=Drx
        LOS_vectorrx=LOS_vectorrx.reshape((-1,1))
        LOS_vectortx=Dtx
        LOS_vectortx=LOS_vectortx.reshape((-1,1))
        LOS_mat=LOS_vectorrx@LOS_vectortx.transpose()
    return LOS_mat

def generate_lossless_channel(*arg):
    Rxantennalocations=arg[0]['Rxantennalocations'] 
    antennasrx=Rxantennalocations.shape[0] #antennas of the receiver, currently only 1 antenna at the tx
    antennastx=1
    H=np.ones((1,antennasrx,antennastx))/np.sqrt(antennasrx*antennastx) #No power gain
    H.shape=(1,antennasrx,antennastx)
    return H

def lambda2meter(distlambda,f):
    d=np.array([distlambda*con.c/f])
    return d

def channel_propagate(signal,H):
    #Calculate the convolution of the 3D matrix filter
    #y(n)=SUM s(n-k)@H(k,:,:).T  
    convlen=signal.shape[0]+H.shape[0]-1
    srx=np.zeros((convlen,H.shape[1]))
    for i in range(H.shape[0]): #0th dim is the "time", k of the filter in
        zt=np.zeros((i,H.shape[1]))
        zt.shape=(-1,H.shape[1])
        zb=np.zeros((H.shape[0]-1,H.shape[1]))
        zb.shape=(-1,H.shape[1])
        s_shift=np.r_['0',np.zeros((i,H.shape[1]),dtype='complex'),signal@H[i,:,:].T,np.zeros((H.shape[0]-1-i,H.shape[1]))]
        srx=srx+s_shift
    return srx 



def get_802_11n_channel_params(model):
# See the channel and loss model in IEEE 802.11n-03/940r4 TGn Channel Models
# This function hard-codes the WLAN 802.11n channel model parameters and
# returns the ones corresponding to the desired channel model. 
#param_dict={'K':K, 'tau':tau, 'pdb':pdb, 'AS_Tx':AS_Tx, 'AoD':AoD, 'AS_Rx':AS_Rx, 'AoA':AoA}

#The IS a more clever way of doing these but at least they are done now.
    if model=='A':
        lossdict={'dbp':5,  's1':2, 's2': 3.5, 'f1':3, 'f2':4}
        tau = np.array([0])
        K=np.zeros(tau.size) #K-factor for Line-of-sight
        pdb = np.array([0],ndmin=2)
        AoA = np.array([45],ndmin=2)
        AS_Rx = np.array([40],ndmin=2)
        AoD = np.array([45],ndmin=2)
        AS_Tx = np.array([40],ndmin=2)
        
    elif model=='B':
        lossdict={'dbp':5,  's1':2, 's2': 3.5, 'f1':3, 'f2':4}
        tau = np.array([0,10,20,30,40,50,60,70,80]) * 1e-9 # Path delays, in seconds
        K=np.zeros(tau.size) #K-factor for Line-of-sight
        
        # Average path gains of cluster, in dB
        pdb1 = np.array([0,-5.4,-10.8,-16.2,-21.7],ndmin=2)
        pdb2 = np.array([-3.2,-6.3,-9.4,-12.5,-15.6,-18.7,-21.8],ndmin=2)
        #these must be reshaped last because others refer their dimensions
     
        #   Angular spreads
        AS_Tx_C1 = np.ones(pdb1.shape)*14.4
        AS_Tx_C1 = np.r_['1', AS_Tx_C1, -1*np.inf*np.ones((1,4))]

        AS_Tx_C2 = np.ones(pdb2.shape)*25.4
        AS_Tx_C2 = np.r_['1', -1*np.inf*np.ones((1,2)), AS_Tx_C2 ]
        AS_Tx = np.r_['0', AS_Tx_C1, AS_Tx_C2]

        #   Mean angles of departure
        AoD_C1 = np.ones(pdb1.shape)*225.1
        AoD_C1 = np.r_['1', AoD_C1, -1*np.inf*np.ones((1,4))]

        AoD_C2 = np.ones(pdb2.shape)*106.5
        AoD_C2 = np.r_['1', -1*np.inf*np.ones((1,2)), AoD_C2 ]

        AoD = np.r_['0',AoD_C1, AoD_C2]

        # Spatial parameters on receiver side:
        #   Angular spreads
        AS_Rx_C1 = np.ones(pdb1.shape)*14.4
        AS_Rx_C1 = np.r_['1', AS_Rx_C1, -1*np.inf*np.ones((1,4))]

        AS_Rx_C2 = np.ones(pdb2.shape)*25.4
        AS_Rx_C2 = np.r_['1', -1*np.inf*np.ones((1,2)), AS_Rx_C2 ]

        AS_Rx = np.r_['0', AS_Rx_C1, AS_Rx_C2]

        #   Mean angles of arrival
        AoA_C1 = np.ones(pdb1.shape)*4.3
        AoA_C2 = np.ones(pdb2.shape)*118.4
        
        AoA_C1 = np.r_['1', AoA_C1, -1*np.inf*np.ones((1,4))]
        AoA_C2 = np.r_['1', -1*np.inf*np.ones((1,2)), AoA_C2 ]

        AoA = np.r_['0', AoA_C1, AoA_C2]

        #Reshape pdb's
        pdb1 = np.r_['1', pdb1, -1*np.inf*np.ones((1,4))]
        pdb2 = np.r_['1', -1*np.inf*np.ones((1,2)), pdb2 ]
        pdb = np.r_['0',pdb1,pdb2]
       
    elif model=='C':
        lossdict={'dbp':5,  's1':2, 's2': 3.5, 'f1':3, 'f2':5}
        tau = np.array([0,10,20,30,40,50,60,70,80,90,110,140,170,200]) * 1e-9
        K=np.zeros(tau.size) #K-factor for Line-of-sight
        
        pdb1 = np.array([0,-2.1,-4.3,-6.5,-8.6,-10.8,-13.0,-15.2,-17.3,-19.5],ndmin=2)
        pdb2 = np.array([-5.0,-7.2,-9.3,-11.5,-13.7,-15.8,-18.0,-20.2],ndmin=2)


        AoA1 = 290.3*np.ones(pdb1.shape)
        AoA1 = np.r_['1',AoA1,-1*np.inf*np.ones((1,4))]

        AoA2 = 332.3*np.ones(pdb2.shape)
        AoA2 = np.r_['1',-1*np.inf*np.ones((1,6)),AoA2]

        AoA = np.r_['0',AoA1, AoA2]

        AS_Rx1 = 24.6*np.ones(pdb1.shape)
        AS_Rx1 = np.r_['1',AS_Rx1,-1*np.inf*np.ones((1,4))]
        
        AS_Rx2 = 22.4*np.ones(pdb2.shape)
        AS_Rx2 = np.r_['1',-1*np.inf*np.ones((1,6)),AS_Rx2]

        AS_Rx = np.r_['0',AS_Rx1, AS_Rx2]
        
        AoD1 = 13.5*np.ones(pdb1.shape)
        AoD1 = np.r_['1',AoD1,-1*np.inf*np.ones((1,4))]

        AoD2 = 56.4*np.ones(pdb2.shape)
        AoD2 = np.r_['1',-1*np.inf*np.ones((1,6)),AoD2]

        AoD = np.r_['0',AoD1, AoD2]

        AS_Tx1 = 24.7*np.ones(pdb1.shape)
        AS_Tx1 = np.r_['1',AS_Tx1,-1*np.inf*np.ones((1,4))]
        
        AS_Tx2 = 22.5*np.ones(pdb2.shape)
        AS_Tx2 = np.r_['1',-1*np.inf*np.ones((1,6)),AS_Tx2]
        
        AS_Tx = np.r_['0',AS_Tx1, AS_Tx2]

        #Reshape pdb's
        pdb1 = np.r_['1',pdb1,-1*np.inf*np.ones((1,4))]
        pdb2 = np.r_['1',-1*np.inf*np.ones((1,6)),pdb2]
        pdb = np.r_['0',pdb1, pdb2]
        
    elif model=='D':
        lossdict={'dbp':10, 's1':2, 's2': 3.5, 'f1':3, 'f2':5}
        tau = np.array([0,10,20,30,40,50,60,70,80,90,110,140,170,200,240,290,340,390]) * 1e-9
        K=np.zeros(tau.size) #K-factor for Line-of-sight
        K[0]=3

        pdb1 = np.array([0,-0.9,-1.7,-2.6,-3.5,-4.3,-5.2,-6.1,-6.9,-7.8,-9.0,-11.1,-13.7,-16.3,-19.3,-23.2],ndmin=2)

        pdb2 = np.array([-6.6,-9.5,-12.1,-14.7,-17.4,-21.9,-25.5],ndmin=2)

        pdb3 = np.array([-18.8,-23.2,-25.2,-26.7],ndmin=2) # path losses vector

        ASt1 = 27.4*np.ones(pdb1.shape)
        ASt1  = np.r_['1',ASt1,-1*np.inf*np.ones((1,2))]

        ASt2 = 32.1*np.ones(pdb2.shape)
        ASt2  = np.r_['1', -1*np.inf*np.ones((1,10)), ASt2, -1*np.inf*np.ones((1,1)) ]
        ASt3 = 36.8*np.ones(pdb3.shape)
        ASt3  = np.r_['1',-1*np.inf*np.ones((1,14)),ASt3]

        AS_Tx = np.r_['0',ASt1, ASt2, ASt3] # Tx angular spread vector
        
        ASr1 = 27.7*np.ones(pdb1.shape)
        ASr1 = np.r_['1',ASr1,-1*np.inf*np.ones((1,2))]

        ASr2 = 31.4*np.ones(pdb2.shape)
        ASr2 = np.r_['1',-1*np.inf*np.ones((1,10)),ASr2,-1*np.inf*np.ones((1,1))]
        
        ASr3 = 37.4*np.ones(pdb3.shape)
        ASr3 = np.r_['1',-1*np.inf*np.ones((1,14)),ASr3]

        AS_Rx = np.r_['0',ASr1, ASr2, ASr3] # Rx angular spread vector
        
        AoD1 = 332.1*np.ones(pdb1.shape)
        AoD1 = np.r_['1',AoD1,-1*np.inf*np.ones((1,2))]

        AoD2 = 49.3*np.ones(pdb2.shape)
        AoD2 = np.r_['1',-1*np.inf*np.ones((1,10)),AoD2,-1*np.inf*np.ones((1,1))]

        AoD3 = 275.9*np.ones(pdb3.shape)
        AoD3 = np.r_['1',-1*np.inf*np.ones((1,14)),AoD3]

        AoD = np.r_['0',AoD1, AoD2, AoD3] # Tx angles of departure
        
        AoA1 = 158.9*np.ones(pdb1.shape)
        AoA1 = np.r_['1',AoA1,-1*np.inf*np.ones((1,2))]

        AoA2 = 320.2*np.ones(pdb2.shape)
        AoA2 = np.r_['1',-1*np.inf*np.ones((1,10)),AoA2,-1*np.inf*np.ones((1,1))]

        AoA3 = 276.1*np.ones(pdb3.shape)
        AoA3 = np.r_['1',-1*np.inf*np.ones((1,14)),AoA3]

        AoA = np.r_['0',AoA1, AoA2, AoA3] # Rx angles of arrival

        #Reshape pdb's
        pdb1 = np.r_['1',pdb1,-1*np.inf*np.ones((1,2))]
        pdb2 = np.r_['1',-1*np.inf*np.ones((1,10)),pdb2,-1*np.inf*np.ones((1,1))]
        pdb3 = np.r_['1',-1*np.inf*np.ones((1,14)),pdb3]
        
        pdb = np.r_['0',pdb1,pdb2,pdb3] # path loss vector
        
    elif model=='E':
        lossdict={'dbp':20, 's1':2, 's2': 3.5, 'f1':3, 'f2':6}
        tau = np.array([0,10,20,30,50,80,110,140,180,230,280,330,380,430,490,560,640,730]) * 1e-9
        K=np.zeros(tau.size) #K-factor for Line-of-sight
        K[0]=6

        pdb1 = np.array([-2.6,-3.0,-3.5,-3.9,-4.5,-5.6,-6.9,-8.2,-9.8,-11.7,-13.9,-16.1,-18.3,-20.5,-22.9],ndmin=2)
        
        pdb2 = np.array([-1.8,-3.2,-4.5,-5.8,-7.1,-9.9,-10.3,-14.3,-14.7,-18.7,-19.9,-22.4],ndmin=2)

        pdb3 = np.array([-7.9,-9.6,-14.2,-13.8,-18.6,-18.1,-22.8],ndmin=2)

        pdb4 = np.array([-20.6,-20.5,-20.7,-24.6],ndmin=2)


        AoA1 = 163.7*np.ones(pdb1.shape)
        AoA1 = np.r_['1',AoA1,-1*np.inf*np.ones((1,3))]
        
        AoA2 = 251.8*np.ones(pdb2.shape)
        AoA2 = np.r_['1',-1*np.inf*np.ones((1,4)),AoA2,-1*np.inf*np.ones((1,2))]

        AoA3 = 80.0*np.ones(pdb3.shape)
        AoA3 = np.r_['1',-1*np.inf*np.ones((1,8)),AoA3,-1*np.inf*np.ones((1,3))]

        AoA4 = 182.0*np.ones(pdb4.shape)
        AoA4 = np.r_['1',-1*np.inf*np.ones((1,14)),AoA4]
        
        AoA = np.r_['0',AoA1, AoA2, AoA3, AoA4]

        AS_Rx1 = 35.8*np.ones(pdb1.shape)
        AS_Rx1 = np.r_['1',AS_Rx1,-1*np.inf*np.ones((1,3))]
        
        AS_Rx2 = 41.6*np.ones(pdb2.shape)
        AS_Rx2 = np.r_['1',-1*np.inf*np.ones((1,4)),AS_Rx2,-1*np.inf*np.ones((1,2))]

        AS_Rx3 = 37.4*np.ones(pdb3.shape)
        AS_Rx3 = np.r_['1',-1*np.inf*np.ones((1,8)),AS_Rx3,-1*np.inf*np.ones((1,3))]

        AS_Rx4 = 40.3*np.ones(pdb4.shape)
        AS_Rx4 = np.r_['1',-1*np.inf*np.ones((1,14)),AS_Rx4]

        AS_Rx = np.r_['0',AS_Rx1, AS_Rx2, AS_Rx3, AS_Rx4]

        AoD1 = 105.6*np.ones(pdb1.shape)
        AoD1 = np.r_['1',AoD1,-1*np.inf*np.ones((1,3))]

        AoD2 = 293.1*np.ones(pdb2.shape)
        AoD2 = np.r_['1',-1*np.inf*np.ones((1,4)),AoD2,-1*np.inf*np.ones((1,2))]
        
        AoD3 = 61.9*np.ones(pdb3.shape)
        AoD3 = np.r_['1',-1*np.inf*np.ones((1,8)),AoD3,-1*np.inf*np.ones((1,3))]

        AoD4 = 275.7*np.ones(pdb4.shape)
        AoD4 = np.r_['1',-1*np.inf*np.ones((1,14)),AoD4]

        AoD = np.r_['0',AoD1, AoD2, AoD3, AoD4]

        AS_Tx1 = 36.1*np.ones(pdb1.shape)
        AS_Tx1 = np.r_['1',AS_Tx1,-1*np.inf*np.ones((1,3))]
        
        AS_Tx2 = 42.5*np.ones(pdb2.shape)
        AS_Tx2 = np.r_['1',-1*np.inf*np.ones((1,4)),AS_Tx2,-1*np.inf*np.ones((1,2))]
        
        AS_Tx3 = 38.0*np.ones(pdb3.shape)
        AS_Tx3 = np.r_['1',-1*np.inf*np.ones((1,8)),AS_Tx3,-1*np.inf*np.ones((1,3))]
        
        AS_Tx4 = 38.7*np.ones(pdb4.shape)
        AS_Tx4 = np.r_['1',-1*np.inf*np.ones((1,14)),AS_Tx4]
        
        AS_Tx = np.r_['0',AS_Tx1, AS_Tx2, AS_Tx3, AS_Tx4]

        #Reshape pdb's
        pdb1 = np.r_['1', pdb1,-1*np.inf*np.ones((1,3))]
        pdb2 = np.r_['1',-1*np.inf*np.ones((1,4)),pdb2,-1*np.inf*np.ones((1,2))]
        pdb3 = np.r_['1',-1*np.inf*np.ones((1,8)),pdb3,-1*np.inf*np.ones((1,3))]
        pdb4 = np.r_['1',-1*np.inf*np.ones((1,14)),pdb4]

        pdb = np.r_['0',pdb1, pdb2, pdb3, pdb4]

    elif model=='F':
        lossdict={'dbp':30, 's1':2, 's2': 3.5, 'f1':3, 'f2':6}
        tau = np.array([0,10,20,30,50,80,110,140,180,230,280,330,400,490,600,730,880,1050]) * 1e-9
        K=np.zeros(tau.size) #K-factor for Line-of-sight
        K[0]=6

        pdb1 = np.array([-3.3,-3.6,-3.9,-4.2,-4.6,-5.3,-6.2,-7.1,-8.2,-9.5,-11.0,-12.5,-14.3,-16.7,-19.9],ndmin=2)
        
        pdb2 = np.array([-1.8,-2.8,-3.5,-4.4,-5.3,-7.4,-7.0,-10.3,-10.4,-13.8,-15.7,-19.9],ndmin=2)

        pdb3 = np.array([-5.7,-6.7,-10.4,-9.6,-14.1,-12.7,-18.5],ndmin=2)

        pdb4 = np.array([-8.8,-13.3,-18.7],ndmin=2)

        pdb5 = np.array([-12.9,-14.2],ndmin=2)

        pdb6 = np.array([-16.3,-21.2],ndmin=2)


        AoA1 = 315.1*np.ones(pdb1.shape)
        AoA1 = np.r_['1',AoA1,-1*np.inf*np.ones((1,3))]
        
        AoA2 = 180.4*np.ones(pdb2.shape)
        AoA2 = np.r_['1',-1*np.inf*np.ones((1,4)),AoA2, -1*np.inf*np.ones((1,2))]

        AoA3 = 74.7*np.ones(pdb3.shape)
        AoA3 = np.r_['1',-1*np.inf*np.ones((1,8)),AoA3, -1*np.inf*np.ones((1,3))]

        AoA4 = 251.5*np.ones(pdb4.shape)
        AoA4 = np.r_['1',-1*np.inf*np.ones((1,12)),AoA4,-1*np.inf*np.ones((1,3)) ]
        
        AoA5 = 68.5*np.ones(pdb5.shape)
        AoA5 = np.r_['1',-1*np.inf*np.ones((1,14)),AoA5,-1*np.inf*np.ones((1,2))]

        AoA6 = 246.2*np.ones(pdb6.shape)
        AoA6 = np.r_['1',-1*np.inf*np.ones((1,16)),AoA6]
        
        AoA = np.r_['0',AoA1, AoA2, AoA3, AoA4, AoA5, AoA6]

        AS_Rx1 = 48.0*np.ones(pdb1.shape)
        AS_Rx1 = np.r_['1',AS_Rx1,-1*np.inf*np.ones((1,3))]
        
        AS_Rx2 = 55.0*np.ones(pdb2.shape)
        AS_Rx2 = np.r_['1',-1*np.inf*np.ones((1,4)),AS_Rx2,-1*np.inf*np.ones((1,2))]

        AS_Rx3 = 42.0*np.ones(pdb3.shape)
        AS_Rx3 = np.r_['1',-1*np.inf*np.ones((1,8)),AS_Rx3,-1*np.inf*np.ones((1,3))]

        AS_Rx4 = 28.6*np.ones(pdb4.shape)
        AS_Rx4 = np.r_['1',-1*np.inf*np.ones((1,12)),AS_Rx4,-1*np.inf*np.ones((1,3))]
                                                                                    
        AS_Rx5 = 30.7*np.ones(pdb5.shape)                                           
        AS_Rx5 = np.r_['1',-1*np.inf*np.ones((1,14)),AS_Rx5,-1*np.inf*np.ones((1,2))]

        AS_Rx6 = 38.2*np.ones(pdb6.shape)
        AS_Rx6 = np.r_['1',-1*np.inf*np.ones((1,16)),AS_Rx6]

        AS_Rx = np.r_['0',AS_Rx1, AS_Rx2, AS_Rx3, AS_Rx4, AS_Rx5, AS_Rx6]

        AoD1 = 56.2*np.ones(pdb1.shape)
        AoD1 = np.r_['1',AoD1,-1*np.inf*np.ones((1,3))]

        AoD2 = 183.7*np.ones(pdb2.shape)
        AoD2 = np.r_['1',-1*np.inf*np.ones((1,4)),AoD2,-1*np.inf*np.ones((1,2))]
        
        AoD3 = 153.0*np.ones(pdb3.shape)
        AoD3 = np.r_['1',-1*np.inf*np.ones((1,8)),AoD3,-1*np.inf*np.ones((1,3))]

        AoD4 = 112.5*np.ones(pdb4.shape)
        AoD4 = np.r_['1',-1*np.inf*np.ones((1,12)),AoD4,-1*np.inf*np.ones((1,3))]
                                                                                
        AoD5 = 291.0*np.ones(pdb5.shape)                                        
        AoD5 = np.r_['1',-1*np.inf*np.ones((1,14)),AoD5,-1*np.inf*np.ones((1,2))]
        
        AoD6 = 62.3*np.ones(pdb6.shape)
        AoD6 = np.r_['1',-1*np.inf*np.ones((1,16)),AoD6]

        AoD = np.r_['0',AoD1, AoD2, AoD3, AoD4, AoD5, AoD6]

        AS_Tx1 = 41.6*np.ones(pdb1.shape)
        AS_Tx1 = np.r_['1',AS_Tx1,-1*np.inf*np.ones((1,3))]
        
        AS_Tx2 = 55.2*np.ones(pdb2.shape)
        AS_Tx2 = np.r_['1',-1*np.inf*np.ones((1,4)),AS_Tx2,-1*np.inf*np.ones((1,2))]
        
        AS_Tx3 = 47.4*np.ones(pdb3.shape)
        AS_Tx3 = np.r_['1',-1*np.inf*np.ones((1,8)),AS_Tx3,-1*np.inf*np.ones((1,3))]
        
        AS_Tx4 = 27.2*np.ones(pdb4.shape)
        AS_Tx4 = np.r_['1',-1*np.inf*np.ones((1,12)),AS_Tx4,-1*np.inf*np.ones((1,3))]
                                                                                    
        AS_Tx5 = 33.0*np.ones(pdb5.shape)                                           
        AS_Tx5 = np.r_['1',-1*np.inf*np.ones((1,14)),AS_Tx5,-1*np.inf*np.ones((1,2))]

        AS_Tx6 = 38.0*np.ones(pdb6.shape)
        AS_Tx6 = np.r_['1',-1*np.inf*np.ones((1,16)),AS_Tx6]
        
        AS_Tx = np.r_['0',AS_Tx1, AS_Tx2, AS_Tx3, AS_Tx4,AS_Tx5,AS_Tx6]

        #Reshape pdb's
        pdb1 = np.r_['1', pdb1,-1*np.inf*np.ones((1,3))]
        pdb2 = np.r_['1',-1*np.inf*np.ones((1,4)),pdb2,-1*np.inf*np.ones((1,2))]
        pdb3 = np.r_['1',-1*np.inf*np.ones((1,8)),pdb3,-1*np.inf*np.ones((1,3))]
        pdb4 = np.r_['1',-1*np.inf*np.ones((1,12)),pdb4,-1*np.inf*np.ones((1,3))]
        pdb5 = np.r_['1',-1*np.inf*np.ones((1,14)),pdb5,-1*np.inf*np.ones((1,2))]
        pdb6 = np.r_['1',-1*np.inf*np.ones((1,16)),pdb6]

        pdb = np.r_['0',pdb1, pdb2, pdb3, pdb4,pdb5,pdb6]


    param_dict={'K':K, 'tau':tau, 'pdb':pdb, 'AS_Tx':AS_Tx, 'AoD':AoD, 'AS_Rx':AS_Rx, 'AoA':AoA, 'lossdict':lossdict }
    return param_dict


def laplacian_pdf(sigma,theta):
    #power angular spectrum
    Q=1/(1-np.exp(-np.sqrt(2)*(theta[-1]-theta[0])))
    #Q=1
    PAS=Q*np.exp(-np.sqrt(2)*np.abs(theta)/sigma)/(np.sqrt(2)*sigma)
    return PAS.reshape((1,-1))

