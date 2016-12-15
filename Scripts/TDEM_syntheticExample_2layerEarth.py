from TDEM_Utils import *
from TDEM_Analytic import *


"""
Synthetic Example with a square loop and a layered Earth
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
siglayer1 = 1e-2
sigback = 0.05
thickness = 30.
m = siglayer1*np.ones(mesh.nC)
air = mesh.gridCC[:,2]>0.
background = mesh.gridCC[:,2]<-thickness
m[air] = 1e-8
m[background] = sigback

size_square_loop = 55.

radius = 2.*size_square_loop/np.sqrt(np.pi)

loopcorner = np.r_[[[mesh.vectorNx[np.argmin(np.abs(mesh.vectorNx+size_square_loop))],mesh.vectorNy[np.argmin(np.abs(mesh.vectorNy+size_square_loop))],0.],
                        [mesh.vectorNx[np.argmin(np.abs(mesh.vectorNx+size_square_loop))],mesh.vectorNy[np.argmin(np.abs(mesh.vectorNy-size_square_loop))],0.],
                        [mesh.vectorNx[np.argmin(np.abs(mesh.vectorNx-size_square_loop))],mesh.vectorNy[np.argmin(np.abs(mesh.vectorNy-size_square_loop))],0.],
                        [mesh.vectorNx[np.argmin(np.abs(mesh.vectorNx-size_square_loop))],mesh.vectorNy[np.argmin(np.abs(mesh.vectorNy+size_square_loop))],0.]]]

print 'corner of the square loop: \n',loopcorner

Js = rectangular_plane_layout(mesh,loopcorner, closed = True,I=1.)

CURL = mesh.edgeCurl

obsloc = np.r_[[[0.,0.,0.]]]
listF = np.vstack([mesh.gridFx,mesh.gridFy,mesh.gridFz])
obsindex =np.argmin(np.linalg.norm(listF-obsloc,axis=1))

#Aloop = BiotSavart_A(mesh.gridCC,mesh,Js)
Aloopx = BiotSavart_A(mesh.gridEx,mesh,Js)
Aloopy = BiotSavart_A(mesh.gridEy,mesh,Js)
Aloopz = BiotSavart_A(mesh.gridEz,mesh,Js)
AloopE = np.r_[Aloopx[:,0],Aloopy[:,1],Aloopz[:,2]]

BloopF_t0 = CURL * AloopE

MsigIe = mesh.getEdgeInnerProduct(m,invMat=True)
MsigIf = mesh.getFaceInnerProduct(m,invProp=True)
MmuIf = mesh.getFaceInnerProduct(mu_0*np.ones(mesh.nC),invProp=True)
MmuIe = mesh.getEdgeInnerProduct(mu_0*np.ones(mesh.nC),invProp=True)
Me = mesh.getEdgeInnerProduct()
DIV = mesh.faceDiv
V = Utils.sdiag(mesh.vol)

A = -CURL*MsigIe*CURL.T*MmuIf

time = [(1e-06, 100), (2e-06, 100), (5e-06, 100)]#,(1e-05, 100), (2e-05, 100)]

BlistBE = Backward_Euler_linear(BloopF_t0,A,time)
#BlistBDF2 = BDF2_linear(BloopF_t0,A,time)
hz0 = hzAnalyticCentLoopT(radius,time_wrapper(time),siglayer1)
hz1 = hzAnalyticCentLoopT(radius,time_wrapper(time),sigback)

BEobslist = np.r_[[BlistBE[i][obsindex] for i in range(len(BlistBE))]]
BEobslist = BEobslist.flatten()
#BDFobslist = [BlistBDF2[i][obsindex] for i in range(len(BlistBDF2))]

if PlotIt:
    fig = plt.figure(figsize =(6,6))
    plt.loglog(time_wrapper(time),mu_0*hz0,color='k',linewidth=2.,linestyle='dashed')#,marker = '+')
    plt.loglog(time_wrapper(time),mu_0*hz1,color='green',linewidth=2.,linestyle='dashdot')#,marker = '+')
    plt.loglog(time_wrapper(time),BEobslist[1:],color='blue',marker = '+')
    #plt.loglog(time_wrapper(time),BDFobslist[1:],color='red',marker = '*')
    plt.gca().legend(['Analytic solution, circular loop, layer 1','Analytic solution, circular loop, layer 2','rectangular loop, layered Earth, BE','rectangular loop, layered Earth, BDF2'],loc=3)
    plt.gca().set_title('TDEM: Synthetic example: 2 layers Earth')
    plt.gca().set_xlabel('Time (s)')
    plt.gca().set_ylabel('Bz (T)')
    plt.show()

    a = mesh.plotSlice(BlistBE[-1],vType='F',normal='Y',view='vec',pcolorOpts={'cmap':'Blues'})#clim = [1e-10,1e-9])
    cb = plt.colorbar(a[0])
    cb.set_label('Tesla')
    #plt.gca().set_xlim([-90.,90])
    #plt.gca().set_ylim([-90.,90])
    plt.gca().set_aspect('equal')
    plt.gca().set_title('B-flux at t=%.1g sec \n for a 110m square loop \n over a layered Earth'%(time_wrapper(time).max()))
    plt.show()