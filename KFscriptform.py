# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:00:11 2020

@author: admin
"""


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
n = 0                              #Counter


#Store initial conditions 
s0 =  np.array([[X[0]],[X[1]],[X[2]],[V[0]],[V[1]],[V[2]],[w]])

#Define time events (speed and turn rate at t_cmd instances)
t_cmd = np.array([5,110,180,210,240,260])
speed = np.array([-40,-45,-1,0,0,0])
turn  = np.array([-0.03,-0.02,-0.01,0,0.05,0])

t_cmd = np.array([5,110,180,210,240,260])
speed = np.array([-20,-30,-30,-31,-29,-28])
turn  = np.array([-0.03,-0.02,0.06,0,0.05,0])

si = s0
#trajectory = np.insert(s0,0,0)
#test = []



trajectory = np.zeros((N+1,len(si)))
trajectory[0] = si.T
for i in range(N):
    if i*Ts == t_cmd[n]:
        si[5] = speed[n]
        si[6] = turn[n]       
        n = min(len(t_cmd),n+1)
    

    tr = si[6]    
 #   test[i] = n
    if abs(tr)<1e-5:
        tr = 1e-5    
        
    F = np.array([[1,0,0,math.sin(tr*Ts)/tr,(math.cos(tr*Ts)-1)/tr,0,0],
         [0,1,0,(1-math.cos(tr*Ts))/tr,math.sin(tr*Ts)/tr,0,0],
         [0,0,1,          0,                   0,        Ts,0],
         [0,0,0,math.cos(tr*Ts),       -math.sin(tr*Ts),  0,0],
         [0,0,0,math.sin(tr*Ts),        math.cos(tr*Ts),  0,0],
         [0,0,0,          0,                   0,         1,0],
         [0,0,0,          0,                   0,         0,1]])     
   # print(str(n)) 
    si = F.dot(si)
    trajectory[i+1] = si.T           


        

#traj = generateTrajectory(t_cmd=t_cmd,speed=speed,turn=turn,n=n,N=N,Ts=Ts,si=si)
xdat = trajectory[:,0]
ydat = trajectory[:,1]
zdat = trajectory[:,2]
vxdat = trajectory[:,3]
vydat = trajectory[:,4]
vzdat = trajectory[:,5]

#plt.plot(xdat/1000,ydat/1000)
#plt.plot(ydat/1000,xdat/1000)
#plt.plot(vxdat,vydat)
#plt.axis('equal')
#plt.grid('true')


sigmx = 6
sigmy = 6
sigmz = 4

xmdat = []
ymdat = []
zmdat=[]
vxmdat = []
vymdat = []
vzmdat=[]

for j in range(N+1):
  #  xmdat[j] = xdat[j]+sigmx*np.random.normal(1)
    xmdat.append(j),ymdat.append(j),zmdat.append(j),vxmdat.append(j),vymdat.append(j),vzmdat.append(j)
    xmdat[j] = xdat[j]+sigmx*np.random.normal(1)
    ymdat[j] = ydat[j]+sigmy*np.random.normal(1)
    zmdat[j] = zdat[j]+sigmz*np.random.normal(1)
    vxmdat[j] = vxdat[j]+sigmx*np.random.normal(1)
    vymdat[j] = vydat[j]+sigmy*np.random.normal(1)
    vzmdat[j] = vzdat[j]+sigmz*np.random.normal(1)

Ra = np.sqrt(np.array(xmdat)**2+np.array(ymdat)**2)
Az = np.arctan2(np.array(ymdat),np.array(xmdat))
El = np.arcsin(zmdat/Ra)
Dp = (np.array(xmdat)*np.array(vxmdat)+np.array(ymdat)*np.array(vymdat)+np.array(zmdat)*np.array(vzmdat))/np.array(Ra)

#plt.plot(xmdat,ymdat)

#Do Kalman filtering

#Update time, integer multiples of sampling time
Tup = 1
#Motion dynamics
F = np.array([[1,0,Tup,0],[0,1,0,Tup],[0,0,1,0],[0,0,0,1]])
#Observation matrix
H = np.array([[1,0,0,0],[0,1,0,0]])
#Process noise
G = np.array([[Tup**2/2,0],[0,Tup**2/2],[Tup,0],[0,Tup]])

#Covariance init
Qp = np.diag([1e4,1e4,1e4,1e4])
Rk = np.diag([sigmx**2,sigmy**2])

#Save measurements
zm = np.column_stack((xmdat, ymdat))
#Initial state
Sp = [0,100,0,-2]

#Calculate acceleration
accx = np.diff(vxmdat)/np.diff(np.linspace(0,(N-1)*Tup,N+1))
accy = np.diff(vymdat)/np.diff(np.linspace(0,(N-1)*Tup,N+1))

#Get variance
var_ax = np.mean(accx)**2
var_ay = np.mean(accy)**2

#Define process noise
Qv = np.diag([var_ax,var_ay])
Q = G.dot(Qv).dot(G.T)

#Allocate output
zmkf = np.zeros((N+1,len(Sp)))

for i in range(N+1):
    #Update
    y = zm[i,:]-H.dot(Sp) 
    S = H.dot(Qp).dot(H.T)+Rk
    Kk = Qp.dot(H.T).dot(np.linalg.inv(S))
    Sf = Sp+Kk.dot(y)
    Qf = (np.eye(4)-Kk.dot(H))*Qp
    
    #Prediction
    Sp = F.dot(Sf)
    Qp=F.dot(Qf).dot(F.T)+Q
    zmkf[i] = Sf #Output 
    
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(zmkf[:,0],zmkf[:,1],'.-')
plt.plot(xmdat,ymdat,'.')
plt.grid('true')
# ax.spines['bottom'].set_color((0.7412,0.3922,0.3176))
# ax.spines['top'].set_color((0.7412,0.3922,0.3176))
# ax.spines['right'].set_color((0.7412,0.3922,0.3176))
# ax.spines['left'].set_color((0.7412,0.3922,0.3176))
# ax.spines['bottom'].set_linewidth(1.7)
# ax.spines['top'].set_linewidth(1.7)
# ax.spines['right'].set_linewidth(1.7)
# ax.spines['left'].set_linewidth(1.7)
# ax.tick_params(axis='x', colors=(0.7412,0.3922,0.3176))
# ax.tick_params(axis='y', colors=(0.7412,0.3922,0.3176))
# plt.xlabel('x-axis [m]', fontsize=13)
# plt.ylabel('y-axis [m]', fontsize=13)
    