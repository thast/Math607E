from TDEM_Utils import *
from TDEM_Analytic import *

"""
Test the quadratic convergence of B = CURL A for a circular loop
in comparison to the analytic solution for B for a central receiver Bz
"""

PlotIt = True

radius = 10.
loc = np.r_[[[0.,0.,0.]]]
obsloc = np.r_[[[0.,0.,0.]]]
Bbslist = []

errorlist=[]
plist = []
hlist = []

Bdip = circularloop(radius,obsloc,I=1.)
Bdip = Bdip[0,2]
meshsize = np.linspace(3,6,4)
for i in meshsize:
    csx, csy, csz = 101./(2.**i+1.),101./(2.**i+1.),101./(2.**i)
    hlist.append(csx)
    ncx, ncy, ncz = 2**i+1,2**i+1,2**i
    hx = [(csx,ncx)]
    hy = [(csy,ncy)]
    hz= [(csz,ncz)]
    mesh = Mesh.TensorMesh([hx, hy, hz],x0="CCC")
    #mesh.x0[2] = mesh.x0[2]- csz/2.
    CURL = mesh.edgeCurl
    listF = np.vstack([mesh.gridFx,mesh.gridFy,mesh.gridFz])
    obsindex =np.argmin(np.linalg.norm(listF,axis=1))
    Aloopx = vectorPotential_circularloop(radius,mesh.gridEx)
    Aloopy = vectorPotential_circularloop(radius,mesh.gridEy)
    Aloopz = vectorPotential_circularloop(radius,mesh.gridEz)
    AloopE = np.r_[Aloopx[:,0],Aloopy[:,1],Aloopz[:,2]]
    BloopE = CURL * AloopE
    Bbslist.append(BloopE[obsindex])

errorlist = [np.linalg.norm(Bbslist[i]-Bdip)/np.linalg.norm(Bdip) for i in range(0,len(Bbslist))]

plist = [np.log(errorlist[i+1]/errorlist[i])/np.log(hlist[i+1]/hlist[i])
        for i in range(0,len(errorlist)-1)]


if PlotIt == True:
    fig0 =plt.figure(num =1,figsize=(6,3))
    plt.plot(range(len(plist)),2*np.ones(len(plist)),linestyle='dashed',color='k',linewidth =2.)
    plt.plot(range(len(plist)),plist)
    plt.gca().set_ylim([1.,3.])
    plt.gca().set_title('convergence rate for CURL*A \n for a circular loop of radius 10m')
    plt.gca().set_xlabel('discretization: cell size (m)')
    plt.gca().set_xticks(range(len(plist)))
    plt.gca().set_xticklabels(['%0.2f'%hlist[i] for i in range(1,len(hlist))])
    plt.gca().set_ylabel('convergence rate p')
    plt.gca().legend(['expected convergence rate','estimated convergence rate'],loc=3)
    plt.show()

    fig1 =plt.figure(num =2,figsize=(6,6))
    plt.loglog(hlist,errorlist)
    plt.loglog(hlist,(1e-3)*(np.r_[hlist])**2.,linestyle='dashed')
    plt.gca().set_xlabel('discretization: cell size (m)')
    plt.gca().invert_xaxis()
    plt.gca().set_title('Relative Error for CURL*A \n for a circular loop of radius 10m')
    #plt.gca().set_xticklabels(['%0.2f'%hlist[i] for i in range(0,len(hlist))])
    plt.gca().set_ylabel('relative error to analytic')
    plt.gca().legend(['Relative error to analytic','expected: quadratic convergence'],loc=3)

    a = mesh.plotSlice(BloopE,vType='F',normal='Y',view='vec',pcolorOpts={'cmap':'Blues'})#clim = [1e-10,1e-9])
    cb = plt.colorbar(a[0])
    cb.set_label('Tesla')
    #plt.gca().set_xlim([-90.,90])
    #plt.gca().set_ylim([-90.,90])
    plt.gca().set_aspect('equal')
    plt.gca().set_title('B-field at t=0 sec \n for a 10m radius loop')
    plt.show()

    plt.show()