from TDEM_Utils import *
from TDEM_Analytic import *

"""
Test the 4th order convergence of BDF4
in comparison to the analytic solution for B
for a cicular loop
"""

PlotIt = True

#Cell size
csx, csy, csz = 10.,10.,10.
# Number of core cells in each directiPon s
ncx, ncy, ncz = 15,15,10
# Number of padding cells to add in each direction
npad = 7
# Vectors of cell lengthts in each direction
hx = [(csx,npad, -1.5),(csx,ncx),(csx,npad, 1.5)]
hy = [(csy,npad, -1.5),(csy,ncy),(csy,npad, 1.5)]
hz= [(csz,npad,-1.5),(csz,ncz),(csy,npad, 1.5)]
# Create mesh
mesh = Mesh.TensorMesh([hx, hy, hz],x0="CCC")


#Define the model
sighalf = 1e-1
m = sighalf*np.ones(mesh.nC)
air = mesh.gridCC[:,2]>0.
m[air] = 1e-8

#Loop and Receiver
radius = 50.
loc = np.r_[[[0.,0.,0.]]]
obsloc = np.r_[[[0.,0.,0.]]]

#Initialize B
CURL = mesh.edgeCurl
listF = np.vstack([mesh.gridFx,mesh.gridFy,mesh.gridFz])
obsindex =np.argmin(np.linalg.norm(listF-obsloc,axis=1))
Aloopx = vectorPotential_circularloop(radius,mesh.gridEx)
Aloopy = vectorPotential_circularloop(radius,mesh.gridEy)
Aloopz = vectorPotential_circularloop(radius,mesh.gridEz)
AloopE = np.r_[Aloopx[:,0],Aloopy[:,1],Aloopz[:,2]]
BloopF_t0 = CURL * AloopE
Bt0_analytic = mu_0/(2*radius)
print 'relative error at initialization: ',np.abs(BloopF_t0[obsindex]-Bt0_analytic)/np.abs(Bt0_analytic)

#Initialize List
Bbslist = []
errorlist=[]
plist = []
klist = []

#Final Time and number of time steps to reach it
timetarget = 1e-4
timesteps = range(4,11,2)

#Analytic
Bz =mu_0*hzAnalyticCentLoopT(radius,timetarget,sighalf)
print 'Analytic solution Bz for time %f s: '%(timetarget),Bz

#Matrices Operators
CURL = mesh.edgeCurl
MsigIe = mesh.getEdgeInnerProduct(m,invMat=True)
MsigIf = mesh.getFaceInnerProduct(m,invProp=True)
MmuIf = mesh.getFaceInnerProduct(mu_0*np.ones(mesh.nC),invProp=True)
MmuIe = mesh.getEdgeInnerProduct(mu_0*np.ones(mesh.nC),invProp=True)
Me = mesh.getEdgeInnerProduct()
DIV = mesh.faceDiv
V = Utils.sdiag(mesh.vol)
A = -CURL*MsigIe*CURL.T*MmuIf
Id = eye(A.shape[0],A.shape[1])

#iterate over number of time steps
for i in timesteps:
    k = timetarget/np.float(i)
    time = [(k,i)]
    print 'time discretization: ',time
    klist.append(k)
    blistback = BDF4_linear(BloopF_t0,A,time)
    Bbslist.append(blistback[-1][obsindex])
    print 'Numerical solution: ',Bbslist[-1]

#Compute the relative error to analytic
errorlist = [np.linalg.norm(Bbslist[i]-Bz)/np.linalg.norm(Bz) for i in range(0,len(Bbslist))]

#Estimate the rate of convergence p
plist0 = [np.log(errorlist[i+1]/errorlist[i])/np.log(klist[i+1]/klist[i])
        for i in range(0,len(errorlist)-1)]

#Estimate the rate of convergence p
plist1 = [np.log(errorlist[i+2]/errorlist[i+1])/np.log(errorlist[i+1]/errorlist[i])
        for i in range(0,len(errorlist)-2)]

#print plist0
#print plist1
#print errorlist

if PlotIt:
    fig0 =plt.figure(figsize=(6,3))
    plt.plot(timesteps[1:],.4.*np.ones(len(plist0)),linestyle='dashed',color='k',linewidth =2.)
    plt.plot(timesteps[1:],plist0)
    #plt.plot(timesteps[2:],plist1)
    plt.gca().set_ylim([0.,7.])
    plt.gca().set_title('convergence rate for BDF4')
    plt.gca().set_xlabel('# of time steps to reach 1e-4 secondes')
    plt.gca().set_xticks(timesteps[1:])
    plt.gca().set_ylabel('convergence rate p')
    plt.gca().legend(['expected convergence rate','estimated convergence rate'],loc=3)
    plt.show()

    fig1 =plt.figure(figsize=(6,6))
    plt.loglog(timesteps,errorlist)
    plt.loglog(timesteps,1./np.r_[timesteps]**4.,linestyle='dashed')
    plt.gca().set_xlabel('# of time steps to reach 1e-4 secondes')
    plt.gca().set_title('Relative Error for Bz at 1e-4 sec')
    plt.gca().set_ylabel('relative error to analytic')
    plt.gca().legend(['Relative error to analytic','expected: 4th order convergence'],loc=3)

    plt.show()


