#clear
#reset
"""
Created on Mon May 18 22:19:52 2020
@author: admin
"""
import numpy as np
import math
import matplotlib.pyplot as plt

X = np.array([0, 100, 1e3])        #Start position (x,y,z)
V = np.array([0, -2, 100])         #Start velocity (x,y,z)
w = 0                              #Turn rate
Ts = 1                             #Sampling time
N = 260                            #Trajectory sample length
                         
#Set initial conditions 
s0 =  np.array([[X[0]],[X[1]],[X[2]],[V[0]],[V[1]],[V[2]],[w]])

#Define time events (speed and turn rate at t_cmd instances)
t_cmd = np.array([5,110,180,210,240,260])
speed = np.array([-40,-45,-1,0,0,0])
turn  = np.array([-0.03,-0.02,-0.01,0,0.05,0])
si = s0
n = 0   


def generateTrajectory(t_cmd,speed,turn,n,N,Ts,si):
    trajectory = np.zeros((N+1,len(si)))
    trajectory[0] = si.T
    for i in range(N):
        if i*Ts == t_cmd[n]:
            si[5] = speed[n]
            si[6] = turn[n]
            n = min(len(t_cmd),n+1)
    
        tr = si[6]    
        
        if abs(tr)<1e-5:
            tr = 1e-5    
            
        F = np.array([[1,0,0,math.sin(tr*Ts)/tr,(math.cos(tr*Ts)-1)/tr,0,0],
             [0,1,0,(1-math.cos(tr*Ts))/tr,math.sin(tr*Ts)/tr,0,0],
             [0,0,1,          0,                   0,        Ts,0],
             [0,0,0,math.cos(tr*Ts),       -math.sin(tr*Ts),  0,0],
             [0,0,0,math.sin(tr*Ts),        math.cos(tr*Ts),  0,0],
             [0,0,0,          0,                   0,         1,0],
             [0,0,0,          0,                   0,         0,1]])     

        si = F.dot(si)
        trajectory[i+1] = si.T           

    return trajectory.T

traj = generateTrajectory(t_cmd=t_cmd,speed=speed,turn=turn,n=n,N=N,Ts=Ts,si=si)
#plt.plot(traj[0,:],traj[1,:])
#plt.grid('true')


def generateMeasurements(N,sigmx,sigmy,sigmz,traj):
    xmdat=[]
    ymdat=[]
    zmdat=[]
    vxmdat=[]
    vymdat=[]
    vzmdat=[]
    
    for j in range(N+1):        
    #Generate measurements by adding Gaussian noise to trajectory        
        xmdat.append(j),ymdat.append(j),zmdat.append(j),vxmdat.append(j),vymdat.append(j),vzmdat.append(j)
        xmdat[j] = traj[0,j]+sigmx*np.random.normal(1)
        ymdat[j] = traj[1,j]+sigmy*np.random.normal(1)
        zmdat[j] = traj[2,j]+sigmz*np.random.normal(1)
        vxmdat[j] = traj[3,j]+sigmx*np.random.normal(1)
        vymdat[j] = traj[4,j]+sigmy*np.random.normal(1)
        vzmdat[j] = traj[5,j]+sigmy*np.random.normal(1)
   
    #If Range/azimuth,elevation,Doppler measurements are needed        
    Ra = np.sqrt(np.array(xmdat)**2+np.array(ymdat)**2)
    Az = np.arctan2(np.array(ymdat),np.array(xmdat))
    El = np.arcsin((zmdat)/(Ra))
    Dp = (np.array(xmdat)*np.array(vxmdat)+np.array(ymdat)*np.array(vymdat)+np.array(zmdat)*np.array(vzmdat))/np.array(Ra)
    
    return Ra,Az,xmdat,ymdat,vxmdat,vymdat
            
    
        
[Ra,Az,xmdat,ymdat,vxmdat,vymdat]=generateMeasurements(N=N,sigmx=3,sigmy=3,sigmz=3,traj=traj)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.spines['bottom'].set_color((0.7412,0.3922,0.3176))
ax.spines['top'].set_color((0.7412,0.3922,0.3176))
ax.spines['right'].set_color((0.7412,0.3922,0.3176))
ax.spines['left'].set_color((0.7412,0.3922,0.3176))
ax.spines['bottom'].set_linewidth(1.7)
ax.spines['top'].set_linewidth(1.7)
ax.spines['right'].set_linewidth(1.7)
ax.spines['left'].set_linewidth(1.7)
#ax.xaxis.label.set_color((1,1,1))
ax.tick_params(axis='x', colors=(0.7412,0.3922,0.3176))
ax.tick_params(axis='y', colors=(0.7412,0.3922,0.3176))

plt.plot(xmdat,ymdat,".",color=(0.3176,0.6667,0.7412),linewidth=1.5)
#plt.axis('equal')
plt.grid('true')
plt.xlabel('x-axis [m]', fontsize=13)
plt.ylabel('y-axis [m]', fontsize=13)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.spines['bottom'].set_color((0.7412,0.3922,0.3176))
ax.spines['top'].set_color((0.7412,0.3922,0.3176))
ax.spines['right'].set_color((0.7412,0.3922,0.3176))
ax.spines['left'].set_color((0.7412,0.3922,0.3176))
ax.spines['bottom'].set_linewidth(1.7)
ax.spines['top'].set_linewidth(1.7)
ax.spines['right'].set_linewidth(1.7)
ax.spines['left'].set_linewidth(1.7)
#ax.xaxis.label.set_color((1,1,1))
ax.tick_params(axis='x', colors=(0.7412,0.3922,0.3176))
ax.tick_params(axis='y', colors=(0.7412,0.3922,0.3176))

plt.plot(traj[0,:],traj[1,:],".",color=(0.3176,0.6667,0.7412),linewidth=1.5)
#plt.axis('equal')
plt.grid('true')
plt.xlabel('x-axis [m]', fontsize=13)
plt.ylabel('y-axis [m]', fontsize=13)







    