from TDEM_Utils import *
from TDEM_Analytic import *
import sys
path ="../../pymatsolver/" 
#path = "../../../Documents/pymatsolver/"
sys.path.append(path)
from pymatsolver import BicgJacobiSolver, PardisoSolver

"""
Test the linear convergence of Backward Euler
in comparison to the analytic solution for B
for a cicular loop
"""

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
#hz = [(csz,npad, -1.5),(csz,ncz),(csz,npad,1.5)]
# Create mesh
mesh = Mesh.TensorMesh([hx, hy, hz],x0="CCC")
# Map mesh coordinates from local to UTM coordiantes
#mesh.x0[2] = mesh.x0[2]-mesh.vectorCCz[-npad-1]
#mesh.x0[2] = mesh.x0[2]- csz/2.#-np.max(mesh.vectorCCz[npad:-npad]) 

#Define the model
sighalf = 1e-1
m = sighalf*np.ones(mesh.nC)
air = mesh.gridCC[:,2]>0.
m[air] = 1e-8

radius = 50.

CURL = mesh.edgeCurl
listF = np.vstack([mesh.gridFx,mesh.gridFy,mesh.gridFz])
obsindex =np.argmin(np.linalg.norm(listF,axis=1))
Aloopx = vectorPotential_circularloop(radius,mesh.gridEx)
Aloopy = vectorPotential_circularloop(radius,mesh.gridEy)
Aloopz = vectorPotential_circularloop(radius,mesh.gridEz)
AloopE = np.r_[Aloopx[:,0],Aloopy[:,1],Aloopz[:,2]]
BloopF_t0 = CURL * AloopE
Bt0_analytic = mu_0/(2*radius)
print 'relative error at initialization: ',np.abs(BloopF_t0[obsindex]-Bt0_analytic)/np.abs(Bt0_analytic)

PlotIt = True

loc = np.r_[[[0.,0.,0.]]]
obsloc = np.r_[[[0.,0.,0.]]]
Bbslist = []
errorlist=[]
plist = []
klist = []
timetarget = 1e-4
timesteps = range(10,51,5)

Bz =mu_0*hzAnalyticCentLoopT(radius,timetarget,sighalf)
print 'Analytic solution Bz for time %f s: '%(timetarget),Bz

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

for i in timesteps:
    k = timetarget/np.float(i)
    time = [(k,i)]
    print 'time discretization: ',time
    klist.append(k)
    Ainv = PardisoSolver(Id-k*A)
    blistback = Backward_Euler_linear(BloopF_t0,A,time,Ainv = Ainv)
    Bbslist.append(blistback[-1][obsindex])
    print 'Numerical solution: ',Bbslist[-1]

errorlist = [np.linalg.norm(Bbslist[i]-Bz)/np.linalg.norm(Bz) for i in range(0,len(Bbslist))]

plist = [np.log(errorlist[i+1]/errorlist[i])/np.log(klist[i+1]/klist[i])
        for i in range(0,len(errorlist)-1)]

#plist1 = [np.log(errorlist[i+2]/errorlist[i+1])/np.log(errorlist[i+1]/errorlist[i])
#        for i in range(0,len(errorlist)-2)]


#print plist
#print plist1
#print errorlist

if PlotIt:
    fig0 =plt.figure(1,figsize=(6,3))
    plt.plot(timesteps[1:],1.*np.ones(len(plist)),linestyle='dashed',color='k',linewidth =2.)
    plt.plot(timesteps[1:],plist)
    #plt.plot(timesteps[2:],plist1)
    plt.gca().set_ylim([0.,2.])
    plt.gca().set_title('convergence rate for Backward Euler')
    plt.gca().set_xlabel('# of time steps to reach 1e-4 secondes')
    plt.gca().set_xticks(timesteps[1:])
    plt.gca().set_ylabel('convergence rate p')
    plt.gca().legend(['expected convergence rate','estimated convergence rate'],loc=3)
    plt.show()

    fig1 =plt.figure(2,figsize=(6,6))
    plt.loglog(timesteps,errorlist)
    plt.loglog(timesteps,1./np.r_[timesteps],linestyle='dashed')
    plt.gca().set_xlabel('# of time steps to reach 1e-4 secondes')
    plt.gca().set_title('Relative Error for Bz at 1e-4 sec with Backward Euler')
    plt.gca().set_ylabel('relative error to analytic')
    plt.gca().legend(['Relative error to analytic','expected: linear convergence'],loc=3)
    plt.show()

    a = mesh.plotSlice(blistback[-1],vType='F',normal='Y',view='vec',pcolorOpts={'cmap':'Blues'})#clim = [1e-10,1e-9])
    cb = plt.colorbar(a[0])
    cb.set_label('Tesla')
    #plt.gca().set_xlim([-90.,90])
    #plt.gca().set_ylim([-90.,90])
    plt.gca().set_aspect('equal')
    plt.gca().set_title('B-field at t=1e-4 sec \n for a 50m radius loop \n with a 0.1 S/m half-space with BE')
    plt.show()


