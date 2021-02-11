# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 19:10:40 2021

@author: Vasily
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.special
import scipy.integrate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import time
import sdeint

#%%
def NoramlAsymetricRandomMatrix(n, tau = 0):
    a = np.zeros([n,n]);
    for i1 in range(0,n):
        for i2 in range(i1+1):
            values = np.random.multivariate_normal([0,0], np.array([[1,tau],[tau,1]])/n);
            a[i1, i2] = values[0];
            a[i2, i1] = values[1];
    return a;

def NormalEIRandomMatrix(n, f=0.5, ve=1, vi=1, me=1, mi=-1, meanSubstract = True):
    ne = int(np.round(n*f))
    ni = n - ne;
    
    is_e = np.repeat(False, n);
    is_e[np.random.choice(np.arange(0,n), size = ne, replace=False)] = True;
    
    Je = np.random.randn(n, n)*np.sqrt(ve);#+me;
    Ji = np.random.randn(n, n)*np.sqrt(vi);#+mi;
    
    J = np.multiply(Je, is_e) + np.multiply(Ji, ~is_e)
    Jmean = np.tile(J.mean(1), (n,1)).transpose();
    
    m = np.repeat(me, n);
    m[~is_e] = mi;
    M = np.tile(m, (n,1));
    
    # dw = 100;
    # dx = 15;
    # mask = np.array([np.arange(-i_n + dx,  n - i_n + dx) for i_n in np.arange(n)]);
    # mask2 = np.exp(-(mask)**2/dw**2);
    
    # mask = np.array([np.arange(-i_n - n + dx,  -i_n + dx) for i_n in np.arange(n)]);
    # mask2 = mask2 + np.exp(-(mask)**2/dw**2); 
    
    # mask = np.array([np.arange(-i_n + n + dx,  -i_n + 2*n + dx) for i_n in np.arange(n)]);
    # mask2 = mask2 + np.exp(-(mask)**2/dw**2); 
    
    # M = M*mask2;
    #M = M - np.tile(M.mean(1), (n,1)).transpose();
    
    Jtot = J - Jmean*meanSubstract + M;
    
    return Jtot, is_e;

#%%

f = 0.5;
n = 4000;#2**12;
v = 25;#2.5;#1.5;
a = 1;
# tau = 0.7;

ve = v/n/a; vi = v/n;
me = 15/np.sqrt(n);#np.sqrt(np.max([ve,vi])); 
mi = -f*me/(1-f);# me #such that f*me + (1-f)*mi = 0 - balanced


# Jtot = NoramlAsymetricRandomMatrix(n,tau)*np.sqrt(v)/(1+tau); #np.random.randn(n,n)*np.sqrt(v/n)
# is_e = np.repeat(True, n);

#Jtot = Jtot - np.tile(Jtot.mean(1), (n,1)).transpose();

Jtot,is_e = NormalEIRandomMatrix(n, f, ve, vi, me, mi, meanSubstract = False);
Jtot = Jtot -  Jtot.mean();
diag = np.copy(np.diag(Jtot));
diag[is_e] = diag[is_e] - me;
diag[~is_e] = diag[~is_e] - mi;
#Jtot = Jtot - np.diag(np.diag(Jtot)) + np.diag(diag)
#Jtot = Jtot - np.tile(Jtot.mean(1), (n,1)).transpose()

# s = np.sqrt(0.04);
# w = np.sqrt(2);
# L = np.eye(n);
# R = np.eye(n)*s/np.sqrt(n);
# J = np.random.randn(n,n);
# M = np.concatenate((np.concatenate( (np.zeros((n-1,1)), np.eye(n-1)), axis = 1 ), np.zeros((1,n))));
# M[-1,0] = 1;

# Jtot = np.matmul(L, np.matmul(J,R)) + M*w;


bins = np.linspace(Jtot.min(), Jtot.max(), 200)
plt.hist(Jtot[:,~is_e].flatten(), bins);
plt.hist(Jtot[:,is_e].flatten(),  bins);
plt.show()
#plt.imshow(Jtot)
#%%
eig, ev = np.linalg.eig(Jtot)

fig,ax = plt.subplots()
ax.scatter(eig.real, eig.imag, s=0.3) 
ax.set_aspect(1)

p = matplotlib.patches.Circle((0,0), radius = np.sqrt(1-f+f/a)*np.sqrt(v),
                                   fill=False, linestyle = '--')
ax.add_patch(p);
plt.show()
#%%
nt = 3200;
tmax = 800;
G = np.random.randn(n,nt+10);

def dxdt(x,t,W , noise = 0):
    return -x + np.matmul(W, np.tanh(x));# + noise*G[:, int(t/tmax*nt)]*tmax/nt;


def dxdt_grad(x,t,W, noise = 0):
    return -np.eye(x.shape[0]) + np.multiply(W, 1/(np.cosh(x)**2))
    
x0 =  np.random.randn(n)/10; #2*np.sin((np.arange(n))/75); #np.zeros((n,1)).squeeze(); x0[0] = 1;#
ts = np.linspace(0, tmax, nt);

#%%
start_time = time.time()

x,info = scipy.integrate.odeint(dxdt, y0 = x0, t = ts, 
                                args = (Jtot, 0),  Dfun = dxdt_grad, full_output=True)

# G = np.eye(x0.shape[0]);

# def f(x,t): return dxdt(x,t,Jtot);
# def g(x,t): return 0.1*G;

# x = sdeint.itoint( f=f, G = g, y0 = x0, tspan = ts)

print("--- Integration took %s seconds ---" % (time.time() - start_time))
plt.plot(x[:,0:20], linewidth=0.5);
plt.show()
#%%
xfinal = x[-1,:];

dx = np.repeat(0.001,n);

eigs_final,evs_final = np.linalg.eig( np.matmul(Jtot, np.diag(1/np.cosh(xfinal)**2)) )

fig,ax = plt.subplots()
ax.scatter(eigs_final.real, eigs_final.imag, s=0.3) 
ax.axvline(x=1, color = 'black')

ax.set_aspect(1)
plt.show()

#%%
nt0 = 200;
tmax = 400;
ylim = [-6, 6];

autocorrelate = lambda x: np.correlate(x, x, mode='full') / np.correlate(np.repeat(1, len(x)), np.repeat(1, len(x)), mode='full')

#y = 3*np.array([np.sin(2*50*np.pi*ts/ts[-1]), np.cos(2*50*np.pi*ts/ts[-1])]).transpose()

fig = plt.figure(figsize = (21, 4))
gs = fig.add_gridspec(1, 20)

ax0 = fig.add_subplot(gs[0:4])
ax1 = fig.add_subplot(gs[4:10])
ax2 = fig.add_subplot(gs[10:12])
ax3 = fig.add_subplot(gs[12:17])
ax4 = fig.add_subplot(gs[17:None])

p = matplotlib.patches.Ellipse((0,0), 
                                2*np.sqrt(1-f+f/a)*np.sqrt(v),#2*(1+tau)*np.sqrt(v)/(1+tau),  
                                2*np.sqrt(1-f+f/a)*np.sqrt(v),#2*(1-tau)*np.sqrt(v)/(1+tau), 
                                fill=False, linestyle = '--')
ax0.add_patch(p);
ax0.scatter(eig.real, eig.imag, s=0.3) 
ax0.axvline(x=1, color = 'black')
ax0.set_aspect(1)
ax0.set_xlim([-3, 3])
ax0.set_ylim([-3, 3])
ax0.set_title('$W$ eigenvalues')
ax0.grid(which = 'both'); ax0.set_axisbelow(True);
ax0.set_xlabel('$Re(\lambda)$')
ax0.set_ylabel('$Im(\lambda)$')

ax1.plot(ts, x, color = 'black', alpha = 0.01)
ax1.plot(ts, x.mean(1), color = 'red', linestyle='--', lw = 1)
# ax1.plot(ts, x[:,0:3])
ax1.set_xlim([0, tmax ])
ax1.set_ylim(ylim)
ax1.grid(axis = 'y'); ax1.set_axisbelow(True);

ax1.set_title('Time dependence')
ax1.set_xlabel('time t');
ax1.set_ylabel('x(t)');

ax2.hist(x[nt0:None,:].flatten(), 100, orientation='horizontal', density=True)
ax2.set_ylim(ylim)
ax2.grid(axis = 'y'); ax2.set_axisbelow(True);

ax2.set_title('Distribution')
ax2.set_xlabel('pdf');
ax2.set_yticklabels([]);


amplitudes = np.abs(np.array([np.fft.fft(x[nt0:None, i_n]) for i_n in range(x.shape[1]) ]).transpose())
fs = np.arange( int(amplitudes.shape[0]/2) )/(np.max(ts) - np.min(ts));

amplitudes = amplitudes / np.sqrt(np.sum(amplitudes**2,0));

ax3.plot(fs, ((amplitudes[0:int(amplitudes.shape[0]/2) ])**2), color = 'black', alpha = 0.002)
ax3.plot(fs, ((amplitudes[0:int(amplitudes.shape[0]/2) ])**2).mean(1),
          color = 'red', linestyle = '--', lw = 1)

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_xlim([1e-3, fs.max()])
ax3.set_ylim([1e-8, 1])

ax3.grid(axis = 'x', which = 'both'); ax1.set_axisbelow(True);

ax3.set_title('Spectral density')
ax3.set_xlabel('frequency $f$');
ax3.set_ylabel('$|S(f)|^2$')
#ax3.set_ylabel('$|x(f)|^2$');



cc = np.array([autocorrelate(x[nt0:None,i_n]) 
                for i_n in range(n)])
cc = np.array([cc[i_n,:]/np.max(cc[i_n,:]) for i_n in range(n)])

ax4.plot(ts[nt0:None] - ts[nt0], cc[:,int((cc.shape[1]-1)/2):None].T, alpha = 0.002, color = 'black'); 

corr = cc[:,int((cc.shape[1]-1)/2):None].T.mean(1);

ax4.plot(ts[nt0:None] - ts[nt0], corr/np.max(corr),color = 'red', linestyle = '--', lw = 1);
ax4.set_ylim([-1.02, 1.02])
ax4.set_xlim([0, 200])
ax4.set_title('Autocorrelation function')
ax4.set_xlabel('time lag $\Delta t$');
ax4.set_ylabel('$C(\Delta t$)');
ax4.grid(which = 'both'); ax1.set_axisbelow(True);


plt.tight_layout()
plt.show()
#%%


#%%
# fig,ax= plt.subplots()
# ax.imshow(x.transpose(), cmap='bwr')#, vmin=-2.5, vmax=2.5 );
# ax.set_aspect(0.2)
# plt.show()