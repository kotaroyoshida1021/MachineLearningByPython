import numpy as np
from obj_coord import Coordinates
import math
from scipy import integrate
from scipy.optimize import fmin
from scipy.integrate import odeint
import sys
from Muitipiler import Multipiler

def ProjectedVector(a,b):
    return a - a.dot(b)*b
def normalize(v):
    l_2 = np.linalg.norm(v)
    if l_2==0:
        l_2=1
    return v/l_2

global ALPHA
global length,NDIV,obj_W,NCOORD,PHI,NCOND,OMG_ETA,obj_L,xi0_W,eta0_W,zeta0_W,OMG_ETA0,DIFF_S,IDENTIFY,ALPHA_PARAMS,OMG_ETA_PARAMS
length = 1.076991
NDIV = 251
Ds = length/(NDIV-1)
NCOORD = 251+2
ALPHA = np.zeros(NCOORD,dtype = 'float64')
DIFF_S = np.identity(NDIV-2)
OMG_ETA = np.zeros(NCOORD,dtype = 'float64')

def omegaXiL(s):
    return 0.0
def omegaEtaL(s):
    return 2.917010
    #return 1.0
def omegaZetaL(s):
    return 0.0
def kappa(s):
    return np.sqrt(omegaEtaL(s)**2 + omegaXiL(s)**2)
def kappaSdot(s):
    #未実装;アドホックな対応しかしてません
    return 0

obj_W = Coordinates(length,NDIV)

def dividedCoef(a):
    global ALPHA,PHI,OMG_ETA0
    ALPHA = a[0:NCOORD-2]
    PHI = a[-1]
    OMG_ETA0 = a[-2]

def initialize():
    chi = 0.0
    #delta = 0.343004-math.pi/2.0 
    delta = -math.pi/2.0
    Schi = math.sin(chi);
    Cchi = math.cos(chi);
    Sdelta = math.sin(delta);
    Cdelta = math.cos(delta);
    global zeta0_W
    xi0_W = np.array([Cdelta,0,-Sdelta])
    eta0_W = np.array([Schi*Sdelta,Cchi,Schi*Cdelta])
    zeta0_W = np.array([Cchi*Sdelta,-Schi,Cchi*Cdelta])
    obj_W.DetermineAxies(xi0_W,eta0_W,zeta0_W,omegaXiL,omegaEtaL,omegaZetaL)
    global DIFF_S
    for i in range(1,NDIV-2):
        for j in range(1,NDIV-2):
            DIFF_S[i,j] = i*Ds-j*Ds      
    print(DIFF_S.shape)
    DIFF_S = DIFF_S**2
    PARAMS = np.loadtxt("HyperParam.txt")
    global ALPHA_PARAMS,OMG_ETA_PARAMS
    ALPHA_PARAMS = PARAMS[0:3]
    OMG_ETA_PARAMS = PARAMS[3:6]

def LinearInterpolate(s,Arr):
    p = s/Ds
    n = int(p)
    if(n>=Arr.shape[0]):
        #print("Index Error Why:"+ "s =" + str(s))
        n = Arr.shape[0] - 1
    q = p - float(n)
    if q == 0:
        return float(Arr[n])
    elif n==NDIV-1:
        return float(Arr[n])
    else:
        return float(Arr[n]*(1.0-q) + Arr[n+1]*q)
def alpha(s):
    #Linear_Interpolate
    return LinearInterpolate(s,ALPHA)
def alphaSdot(s):
    #数値微分
    h = 1e-4
    if s<h:
        return alpha(s+h)-alpha(s)/h
    if s+h>length:
        return alpha(s)-alpha(s-h)/h
    else:
        return alpha(s+h)-alpha(s-h)/h

def omegaEtaSdot(omgEta,s):
    l = omgEta
    k = kappa(s)
    kd = kappaSdot(s)
    return (k**2 - l**2)*math.tan(alpha(s)) - kd*l/k

def omegaEta(s):
    return LinearInterpolate(s,OMG_ETA)

def omegaXi(s):
    return math.sqrt(kappa(s)**2 - omegaEta(s)**2)

def omegaZeta(s):
    return - omegaXi(s)*math.tan(alpha(s))

obj_L = Coordinates(length,NDIV)

def initializeForCalculation(a):
    dividedCoef(a)
    #zeta->cos phi sin theta, sin phi sin theta, cos theta
    S = np.linspace(0,length,NDIV)
    global OMG_ETA
    OMG_ETA = odeint(omegaEtaSdot,OMG_ETA0,S)
    print(type(OMG_ETA))
    
    Ctheta = zeta0_W[2]
    theta = np.arccos(zeta0_W[2])
    Stheta = np.sin(theta)
    Cphi = zeta0_W[0]/Stheta
    Sphi = zeta0_W[1]/Stheta
    Cpsi = np.cos(PHI)
    Spsi = np.sin(PHI)
    xi0_L = np.array([Cphi*Ctheta*Cpsi - Sphi*Spsi, Cphi*Spsi + Ctheta*Cpsi*Sphi, -Stheta*Cpsi])
    eta0_L = np.array([-Cpsi*Sphi-Ctheta*Cpsi*Sphi,Cphi*Cpsi - Ctheta*Sphi*Spsi, Stheta*Spsi])
    obj_L.DetermineAxies(xi0_L,eta0_L,zeta0_W,omegaXi,omegaEta,omegaZeta)
    
    return obj_L,OMG_ETA



def objective(a):
    obj_L,OMG_ETA = initializeForCalculation(a)
    ZL = obj_L.ZETA
    ZW = obj_W.ZETA
    M = np.dot(ZL,ZW.T)
    Integrand_1 = np.diag(M)
    Integrand_2 = np.diag(np.dot(obj_L.XI,obj_W.ZETASDOT.T)) - np.diag(OMG_ETA)
    S_COORD = np.linspace(0.0,length,NCOORD)
    S_DIV = np.linspace(0.0,length,NDIV)
    return integrate.simps(Integrand_1**2,S_DIV) + integrate.simps(Integrand_2**2,S_DIV) #+ integrate.simps(OMG_ETA,S_COORD)

###calculation of condition###

def RadialBasisFunc(s_i,s_j,PARAMS):
    return PARAMS[0]*np.exp(-(s_i-s_j)**2/2*PARAMS[1]**2)

def RBFKernelMatrix(PARAMS):
    RET1 = -DIFF_S/2*PARAMS[1]**2
    return PARAMS[0]*np.exp(RET1) + PARAMS[2]*np.identity(NDIV-2)

def PartialDiff_RBFKernelMatrix(i,PARAMS):
    RET1 = -DIFF_S/2*PARAMS[1]**2
    if i==0:
        return np.exp(RET1)
    elif i==1:
        return (PARAMS[0]/PARAMS[1]**3)*np.multiply(DIFF_S,np.exp(-DIFF_S/2*PARAMS[1]**2))
    elif i==2:
        return np.identity(NDIV-2)
    else:
        sys.exit("Error in func: " + __func__ + " due to Partial Differential Error: Incorrect Integer")


def PartialDiff_RBFKernelMatrixDeterminant(i,PARAMS):
    Kdot = PartialDiff_RBFKernelMatrix(i,PARAMS)
    K = RBFKernelMatrix(PARAMS)
    Kdet = np.linalg.det(K)
    if(np.linalg.det(K)==0.0):
        return 0.0
    else:
        Kinv = np.linalg.inv(K)
        return Kdet * np.trace(np.dot(Kinv,Kdot))

#影響されないものを事前にメモ化する
global Ka_DeterminantDiff,Ka_InvDiff,Ke_DeterminantDiff,Ke_InvDiff
def CalculateAndMemorizeKernelMatrix(PARAMS):
    #Ka_DeterminantDiff = np.array([PartialDiff_RBFKernelMatrixDeterminant(0,PARAMS),PartialDiff_RBFKernelMatrixDeterminant(1,PARAMS),PartialDiff_RBFKernelMatrixDeterminant(2,PARAMS)])
    #PartDiffList=[for i in range(3)]
    DeterminantDiff = [PartialDiff_RBFKernelMatrixDeterminant(i,PARAMS) for i in range(3)]
    K = RBFKernelMatrix(PARAMS)
    Kinv = np.linalg.inv(K)
    InvDiff = [Kinv*PartialDiff_RBFKernelMatrix(i,PARAMS)*Kinv.T for i in range(3)]
    return DeterminantDiff,InvDiff

def CalculateCondition(a):
    initializeForCalculation(a)
    a = [Ka_DeterminantDiff[i] - np.dot(np.dot(ALPHA[1:NDIV-1].T,Ka_InvDiff[i]),ALPHA[1:NDIV-1]) for i in range(3)]
    b = [Ke_DeterminantDiff[i] - np.dot(np.dot(OMG_ETA[1:NDIV-1].T,Ke_InvDiff[i]),OMG_ETA[1:NDIV-1]) for i in range(3)]
    LikelihoodConds = a + b
    print(LikelihoodConds)
    return np.array(LikelihoodConds)
def CalculationIneq(a):
    return np.array([])
def cbf():
    f = objective(a)
    print("\r f = %f"%(f))
def main():
    initialize()
    global Ka_DeterminantDiff,Ka_InvDiff,Ke_DeterminantDiff,Ke_InvDiff
    Ka_DeterminantDiff,Ka_InvDiff = CalculateAndMemorizeKernelMatrix(ALPHA_PARAMS)
    Ke_DeterminantDiff,Ke_InvDiff = CalculateAndMemorizeKernelMatrix(OMG_ETA_PARAMS)
    x0 = np.zeros(NDIV)
    multi = Multipiler(objective,x0,"Nelder-Mead",CalculateCondition,CalculationIneq,6,0,cbf)
    multi.LaunchOptimize(1.0e-5)

if __name__ == '__main__':
    main()
