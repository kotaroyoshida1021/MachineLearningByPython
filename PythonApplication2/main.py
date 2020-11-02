import numpy as np
from Coordinates import Coordinates
import math
from scipy import integrate
from scipy.optimize import fmin
import sys

def ProjectedVector(a,b):
    return a - a.dot(b)*b
def normalize(v):
    l_2 = np.linalg.norm(v)
    if l_2==0:
        l_2=1
    return v/l_2

global ALPHA
global length,NDIV,obj_W,NCOORD,PHI,NCOND,OMG_ETA,obj_L,xi0_W,eta0_W,zeta0_W,OMG_ETA0,DIFF_S,IDENTIFY
length = 1.076991
NDIV = 251
Ds = length/(NDIV-1)
NCOORD = 250+1
ALPHA = np.zeros(NCOORD)
DIFF_S = np.identity((NDIV-2,NDIV-2))

def omegaXiL(s):
    return 0.0
def omegaEtaL(s):
    return 2.917010
    #return 1.0
def omegaZetaL(s):
    return 0.0
def kappa(s):
    return sqrt(omegaEtaL(s)**2 + omegaXiL(s)**2)
def kappaSdot(s):
    #未実装;アドホックな対応しかしてません
    return 0

obj_W = Coordinates(omegaXiL,omegaEtaL,omegaZetaL,length,NDIV)

def dividedCoef(a):
    ALPHA = a[0:NCOORD-1]
    PHI = a[-1]
    OMG_ETA0 = a[NCOORD-1]

def initialize():
    chi = 0.0
    #delta = 0.343004-math.pi/2.0 
    delta = -math.pi/2.0
    Schi = math.sin(chi);
    Cchi = math.cos(chi);
    Sdelta = math.sin(delta);
    Cdelta = math.cos(delta);

    xi0_W = np.array([Cdelta,0,-Sdelta])
    eta0_W = np.array([Schi*Sdelta,Cchi,Schi*Cdelta])
    zeta0_W = np.array([Cchi*Sdelta,-Schi,Cchi*Cdelta])
    obj_L.DetermineAxies(xi0,eta0,zeta0)
    for i in range(1,NDIV-1):
        for j in range(1,NDIV-1):
            DIFF_S[i,j] = i*Ds-j*Ds
    DIFF_S = DIFF_S**2

def LinearInterpolate(s,Arr):
    p = s/self.Ds
    n = int(p)
    q = p - float(n)
    if q == 0:
        return Arr[i]
    else:
        return Arr[i]*(1.0-q) + Arr[i+1]*q
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

def omegaEtaSdot(s,omgEta):
    l = omgEta
    k = kappa(s)
    kd = kappaSdot(s)
    return (k**2 - l**2)*np.tan(alpha(s)) - kd*l/k

def omegaEta(s):
    return LinearInterpolate(s,OMG_ETA)

def omegaXi(s):
    return sqrt(kappa(s)**2 - omegaEta(s)**2)

def omegaZeta(s):
    return - omegaXi(s)*np.tan(alpha(s))

obj_L = Coordinates(omegaXi,omegaEta,omegaZeta,length,NDIV)

def initializeForCalculation(a):
    dividedCoef(a)
    #zeta->cos phi sin theta, sin phi sin theta, cos theta
    S = np.linspace(0.0,length,NCOORD)
    OMG_ETA = odeint(omegaEtaSdot,OMG_ETA0,S)

    Ctheta = zeta0_W(2)
    theta = np.arccos(zeta0_W(2))
    Stheta = np.sin(theta)
    Cphi = zeta0_W(0)/Stheta
    Sphi = zeta0_W(1)/Stheta
    Cpsi = np.cos(PHI)
    Spsi = np.sin(PHI)
    xi0_L = np.array([Cphi*Ctheta*Cpsi - Sphi*Spsi, Cphi*Spsi + Ctheta*Cpsi*Sphi, -Stheta*Cpsi])
    eta0_L = np.array([-Cpsi*Sphi-Ctheta*Cpsi*Sphi,Cphi*Cpsi - Ctheta*Sphi*Spsi, Stheta*Spsi])
    obj_L.DetermineAxies(xi0_L,eta0_L,zeta0_W)
    
def objective(a):
    initializeForCalculation(a)
    Integrand_1 = np.diag(np.dot(obj_L.ZETA,obj_W.ZETA.T))
    Integrand_2 = np.diag(np.dot(obj_L.XI,obj_W.ZETASDOT.T)) - np.diag(OMG_ETA)
    S_COORD = np.linspace(0.0,length,NCOORD)
    S_DIV = np.linspace(0.0,length,NDIV)
    return integrate.simps(Integrand_1**2,S_DIV) + integrate.simps(Integrand_2,S_DIV) #+ integrate.simps(OMG_ETA,S_COORD)

###calculation of condition###

def RadialBasisFunc(s_i,s_j,PARAMS):
    return PARAMS[0]*np.exp(-(s_i-s_j)**2/2*PARAMS[1]**2)

def RBFKernelMatrix(PARAMS):
    RET1 = -DIFF_S/2*PARAMS[1]**2
    return PARAMS[0]*np.exp(RET1) + PARAMS[2]*np.identity((NDIV-2,NDIV-2))

def PartialDiff_RBFKernelMatrix(i,PARAMS):
    if i==0:
        return np.exp(RET1)
    elif i==1:
        return (PARAMS[0]/PARAMS[1]**3)*np.multiply(DIFF_S,np.exp(-DIFF_S/2*PARAMS[1]**2))
    elif i==2:
        return np.identity((NDIV-2,NDIV-2))
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
global Ka_DeterminantDiff,Ka_InvDiff
def CalculateAndMemorizeKernelMatrix(PARAMS):
    #Ka_DeterminantDiff = np.array([PartialDiff_RBFKernelMatrixDeterminant(0,PARAMS),PartialDiff_RBFKernelMatrixDeterminant(1,PARAMS),PartialDiff_RBFKernelMatrixDeterminant(2,PARAMS)])
    #PartDiffList=[for i in range(3)]
    Ka_DeterminantDiff = [PartialDiff_RBFKernelMatrixDeterminant(i,PARAMS) for i in range(3)]
    Kinv = np.linalg.inv(K)
    Ka_InvDiff = [Kinv*PartialDiff_RBFKernelMatrix(i,PARAMS)*Kinv.T for i in range(3)]

def CalculateCondition(a):
    initializeForCalculation(a)
    LikelihoodConds = [Ka_DeterminantDiff[i] - np.dot(ALPHA[1:NDIV-1]*Ka_InvDiff[i],ALPHA[1:NDIV-1]) for i in range(3)]
    return np.array(LikelihoodConds)
def CalculationIneq(a):
    return np.array([])

def main():
    initialize()