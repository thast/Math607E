from TDEM_Utils import *
from TDEM_Analytic import *

"""
Test the behavior of numerical time-stepping methods
to different initializations
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

#Inductive loop
radius = 50.

CURL = mesh.edgeCurl

#Receiver
obsloc = np.r_[[[0.,0.,0.]]]
listF = np.vstack([mesh.gridFx,mesh.gridFy,mesh.gridFz])
obsindex =np.argmin(np.linalg.norm(listF-obsloc,axis=1))


#Initialization with analytic B
Bloopx = circularloop(radius,mesh.gridFx)
Bloopy = circularloop(radius,mesh.gridFy)
Bloopz = circularloop(radius,mesh.gridFz)
BloopF_analytic = np.r_[Bloopx[:,0],Bloopy[:,1],Bloopz[:,2]]

#Initialization with vector potential A
Aloopx = vectorPotential_circularloop(radius,mesh.gridEx)
Aloopy = vectorPotential_circularloop(radius,mesh.gridEy)
Aloopz = vectorPotential_circularloop(radius,mesh.gridEz)
AloopE = np.r_[Aloopx[:,0],Aloopy[:,1],Aloopz[:,2]]
BloopF_fromA = CURL * AloopE

print 'Relative error between the analytic B and CURL*A at the receiver location: ', np.abs(BloopF_fromA[obsindex]-BloopF_analytic[obsindex])/np.abs(BloopF_analytic[obsindex])

#Matrix definition
MsigIe = mesh.getEdgeInnerProduct(m,invMat=True)
MsigIf = mesh.getFaceInnerProduct(m,invProp=True)
MmuIf = mesh.getFaceInnerProduct(mu_0*np.ones(mesh.nC),invProp=True)
MmuIe = mesh.getEdgeInnerProduct(mu_0*np.ones(mesh.nC),invProp=True)
Me = mesh.getEdgeInnerProduct()
DIV = mesh.faceDiv
V = Utils.sdiag(mesh.vol)
A = -CURL*MsigIe*CURL.T*MmuIf
Id = eye(A.shape[0],A.shape[1])

#Time steps
time = [(1e-06, 100), (2e-06, 100)]

#Compute the field at the time steps
BfromA_listBE = Backward_Euler_linear(BloopF_fromA,A,time)
B_from_analytic_listBE = Backward_Euler_linear(BloopF_analytic,A,time)

#Extract Receiver data
bzfromA = np.r_[[BfromA_listBE[i][obsindex] for i in range(len(BfromA_listBE))]]
bzfromA = bzfromA.flatten()
bzfrom_analytic = np.r_[[B_from_analytic_listBE[i][obsindex] for i in range(len(B_from_analytic_listBE))]]
bzfrom_analytic = bzfrom_analytic.flatten()

#Analytic for comparison
hz = hzAnalyticCentLoopT(radius,time_wrapper(time),sighalf)


if PlotIt:
	plt.loglog(time_wrapper(time),mu_0*hz,color='k',linewidth=6.,linestyle='dashed')#,marker = '+')
	plt.loglog(time_wrapper(time),bzfrom_analytic[1:],color='blue',marker = '+')
	plt.loglog(time_wrapper(time),bzfromA[1:],color = 'red',marker = '*')
	plt.gca().legend(['Analytic','Initialization with \n analytic B','Initialization with \n analytic A and B=CURL*A'])
	plt.gca().set_title('TDEM: Importance of initialization')
	plt.gca().set_xlabel('Time (s)')
	plt.gca().set_ylabel('Bz (T)')
	plt.show()
