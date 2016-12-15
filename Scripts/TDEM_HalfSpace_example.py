from TDEM_Utils import *
from TDEM_Analytic import *


"""
Synthetic Example with a circular loop over an half-space
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
#hz = [(csz,npad, -1.5),(csz,ncz),(csz,npad,1.5)]
# Create mesh
mesh = Mesh.TensorMesh([hx, hy, hz],x0="CCC")
# Map mesh coordinates from local to UTM coordiantes
#mesh.x0[2] = mesh.x0[2]-mesh.vectorCCz[-npad-1]
#mesh.x0[2] = mesh.x0[2]- csz/2.#-np.max(mesh.vectorCCz[npad:-npad]) 

#Define the model
sigback = 1e-1
m = sigback*np.ones(mesh.nC)
air = mesh.gridCC[:,2]>0.
m[air] = 1e-8

#Loop radius and receiver
radius = 50.
obsloc = np.r_[[[0.,0.,0.]]]
listF = np.vstack([mesh.gridFx,mesh.gridFy,mesh.gridFz])
obsindex =np.argmin(np.linalg.norm(listF-obsloc,axis=1))

#Initialization with a loop
CURL = mesh.edgeCurl
Aloopx = vectorPotential_circularloop(radius,mesh.gridEx)
Aloopy = vectorPotential_circularloop(radius,mesh.gridEy)
Aloopz = vectorPotential_circularloop(radius,mesh.gridEz)
AloopE = np.r_[Aloopx[:,0],Aloopy[:,1],Aloopz[:,2]]
BloopF_t0 = CURL * AloopE
Bt0_analytic = mu_0/(2*radius)
print 'relative error at initialization: ',np.abs(BloopF_t0[obsindex]-Bt0_analytic)/np.abs(Bt0_analytic)

#Matrices Operators
MsigIe = mesh.getEdgeInnerProduct(m,invMat=True)
MsigIf = mesh.getFaceInnerProduct(m,invProp=True)
MmuIf = mesh.getFaceInnerProduct(mu_0*np.ones(mesh.nC),invProp=True)
MmuIe = mesh.getEdgeInnerProduct(mu_0*np.ones(mesh.nC),invProp=True)
Me = mesh.getEdgeInnerProduct()
DIV = mesh.faceDiv
V = Utils.sdiag(mesh.vol)
A = -CURL*MsigIe*CURL.T*MmuIf

time = [(1e-06, 100), (2e-06, 100), (5e-06, 100),(1e-05, 100), (2e-05, 100)]

#Compute flux at each time step
BlistBE = Backward_Euler_linear(BloopF_t0,A,time)
BlistBDF2 = BDF2_linear(BloopF_t0,A,time)

#Analytic for comparison
hz = hzAnalyticCentLoopT(radius,time_wrapper(time),sigback)

#Extract Receiver
BEobslist = np.r_[[BlistBE[i][obsindex] for i in range(len(BlistBE))]]
BEobslist = BEobslist.flatten()
BDFobslist = np.r_[[BlistBDF2[i][obsindex] for i in range(len(BlistBDF2))]]
BDFobslist = BDFobslist.flatten()

#Relative errors
relerr_BE = np.abs(BEobslist[1:]-mu_0*hz)/np.abs(mu_0*hz)
relerr_BDF = np.abs(BDFobslist[1:]-mu_0*hz)/np.abs(mu_0*hz)


if PlotIt:
    fig = plt.figure(figsize =(6,6))
    plt.loglog(time_wrapper(time),mu_0*hz,color='k',linewidth=2.,linestyle='dashed')#,marker = '+')
    plt.loglog(time_wrapper(time),BEobslist[1:],color='blue',marker = '+')
    plt.loglog(time_wrapper(time),BDFobslist[1:],color='red',marker = '*')
    plt.gca().legend(['Analytic solution, circular loop','Backward Euler','BDF2'],loc=3)
    plt.gca().set_title('TDEM: Synthetic example: Half-Space')
    plt.gca().set_xlabel('Time (s)')
    plt.gca().set_ylabel('Bz (T)')
    plt.show()

    plt.loglog(time_wrapper(time),relerr_BE,color='blue',marker = '+')
    plt.loglog(time_wrapper(time),relerr_BDF,color = 'red',marker = '*')
    plt.gca().set_title('Relative error on Bz response to half-space')
    plt.gca().set_xlabel('Time (s)')
    plt.gca().set_ylabel('Relative error')
    plt.gca().legend(['Backward Euler','BDF2'],loc=3)
    plt.show()

    a = mesh.plotSlice(BlistBE[-1],vType='F',normal='Y',view='vec',pcolorOpts={'cmap':'Blues'})#clim = [1e-10,1e-9])
    cb = plt.colorbar(a[0])
    cb.set_label('Tesla')
    #plt.gca().set_xlim([-90.,90])
    #plt.gca().set_ylim([-90.,90])
    plt.gca().set_aspect('equal')
    plt.gca().set_title('B-flux at t=%.1g sec \n for a 50m circular loop \n over a layered Earth'%(time_wrapper(time).max()))
    plt.show()