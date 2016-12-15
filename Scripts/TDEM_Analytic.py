from SimPEG import Mesh, Utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.sparse import spdiags,csr_matrix, eye,kron,hstack,vstack,eye,diags
import copy
from scipy.constants import mu_0
from SimPEG import SolverLU
from scipy.sparse.linalg import spsolve,splu
from SimPEG.EM import TDEM
from SimPEG.EM.Analytics.TDEM import hzAnalyticDipoleT,hzAnalyticCentLoopT
from scipy.interpolate import interp2d,LinearNDInterpolator
from scipy.special import ellipk,ellipe

def analytic_infinite_wire(obsloc,wireloc,orientation,I=1.):
	"""
	Compute the response of an infinite wire with orientation 'orientation'
	and current I at the obsvervation locations obsloc

	Output:
	B: magnetic field [Bx,By,Bz]
	"""
	
	n,d = obsloc.shape
	t,d = wireloc.shape
	d = np.sqrt(np.dot(obsloc**2.,np.ones([d,t]))+np.dot(np.ones([n,d]),(wireloc.T)**2.) 
	- 2.*np.dot(obsloc,wireloc.T))
	distr = np.amin(d, axis=1, keepdims = True)
	idxmind = d.argmin(axis=1)
	r = obsloc - wireloc[idxmind]

	orient = np.c_[[orientation for i in range(obsloc.shape[0])]]
	B = (mu_0*I)/(2*np.pi*(distr**2.))*np.cross(orientation,r)
	
	return B

def mag_dipole(m,obsloc):
	"""
	Compute the response of an infinitesimal mag dipole at location (0,0,0)
	with orientation X and magnetic moment 'm' 
	at the obsvervation locations obsloc

	Output:
	B: magnetic field [Bx,By,Bz]
	"""
	
	loc = np.r_[[[0.,0.,0.]]]
	n,d = obsloc.shape
	t,d = loc.shape
	d = np.sqrt(np.dot(obsloc**2.,np.ones([d,t]))+np.dot(np.ones([n,d]),(loc.T)**2.) 
	- 2.*np.dot(obsloc,loc.T))
	d = d.flatten()
	ind = np.where(d==0.)
	d[ind] = 1e6
	x = obsloc[:,0]
	y = obsloc[:,1]
	z = obsloc[:,2]
	#orient = np.c_[[orientation for i in range(obsloc.shape[0])]]
	Bz = (mu_0*m)/(4*np.pi*(d**3.))*(3.*((z**2.)/(d**2.))-1.)
	By = (mu_0*m)/(4*np.pi*(d**3.))*(3.*(z*y)/(d**2.))
	Bx = (mu_0*m)/(4*np.pi*(d**3.))*(3.*(x*z)/(d**2.))
	
	B = np.vstack([Bx,By,Bz]).T
	
	return B

def circularloop(a,obsloc,I=1.):
	"""
	From Simpson, Lane, Immer, Youngquist 2001
	Compute the magnetic field B response of a current loop
	of radius 'a' with intensity 'I'.

	input:
	a: radius in m
	obsloc: obsvervation locations

	Output:
	B: magnetic field [Bx,By,Bz]
	"""
	x = np.atleast_2d(obsloc[:,0]).T
	y = np.atleast_2d(obsloc[:,1]).T
	z = np.atleast_2d(obsloc[:,2]).T
		
	r = np.linalg.norm(obsloc,axis=1)
	loc = np.r_[[[0.,0.,0.]]]
	n,d = obsloc.shape
	r2 = x**2.+y**2.+z**2.
	rho2 = x**2.+y**2.
	alpha2 = a**2.+r2-2*a*np.sqrt(rho2)
	beta2 = a**2.+r2+2*a*np.sqrt(rho2)
	k2 = 1-(alpha2/beta2)
	lbda = x**2.-y**2.
	C = mu_0*I/np.pi
		
	Bx = ((C*x*z)/(2*alpha2*np.sqrt(beta2)*rho2))*\
	((a**2.+r2)*ellipe(k2)-alpha2*ellipk(k2))
	Bx[np.isnan(Bx)] = 0.
	
	By = ((C*y*z)/(2*alpha2*np.sqrt(beta2)*rho2))*\
	((a**2.+r2)*ellipe(k2)-alpha2*ellipk(k2))
	By[np.isnan(By)] = 0.
	
	Bz = (C/(2.*alpha2*np.sqrt(beta2)))*\
	((a**2.-r2)*ellipe(k2)+alpha2*ellipk(k2))
	Bz[np.isnan(Bz)] = 0.
	
	#print Bx.shape
	#print By.shape
	#print Bz.shape
	B = np.hstack([Bx,By,Bz])
	
	return B

def vectorPotential_circularloop(a,obsloc,I=1.):
    """
	From Simpson, Lane, Immer, Youngquist 2001
	Compute the vector potential A response of a current loop
	of radius 'a' with intensity 'I'.

	input:
	a: radius in m
	obsloc: obsvervation locations

	Output:
	A:  Vector potential [Ax,Ay,Az]
	"""
    x = obsloc[:,0]
    y = obsloc[:,1]
    z = obsloc[:,2]

	#Conversion to spherical coordinates
    r = np.linalg.norm(obsloc,axis=1)
    theta = np.arccos(z/r)
    phi = np.arctan2(y,x)
    te0 = a**2.+r**2.+2.*a*r*np.sin(theta)
    k2 = (4.*a*r*np.sin(theta))/te0
    C = mu_0*I*a/(np.pi)
    te1 = 1./np.sqrt(te0)
    te2 = ((2.-k2)*ellipk(k2)-2.*ellipe(k2))/k2

    Aphi = C*te1*te2
    Aphi[np.isnan(Aphi)] = 0.
    
    Ax = -np.sin(phi)*Aphi
    Ay = np.cos(phi)*Aphi
    Az = np.zeros_like(Aphi)

    A = np.vstack([Ax,Ay,Az]).T
	
    return A