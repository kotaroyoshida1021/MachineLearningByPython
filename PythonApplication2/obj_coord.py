import numpy as np
import math
import scipy.integrate
from scipy import integrate
from scipy.integrate import odeint
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax

class Coordinates(object):
    def __init__(self,omgXi,omgEta,omgZeta,length,MAX):
        self.NDIV = MAX
        self.length = length
        self.omegaXi = omgXi
        self.omegaEta = omgEta
        self.omegaZeta = omgZeta
        self.Ds = self.length/float(self.NDIV-1);
        self.pos_x = np.zeros(self.NDIV)
        self.pos_y = np.zeros(self.NDIV)
        self.pos_z = np.zeros(self.NDIV)
        #print('Coordinates emerge\n')

    def Sdot(self,X,s):
        Y = np.reshape(X,[3,3]).copy();
        OmgMat = np.array([[0.0,self.omegaZeta(s),-self.omegaEta(s)],[-self.omegaZeta(s),0.0,self.omegaXi(s)],[self.omegaEta(s),-self.omegaXi(s),0.0]])
        dXds = np.dot(OmgMat,Y)
        return dXds.flatten();
        return dXds

    def DetermineAxies(self,xi0,eta0,zeta0):
        
        tmp = np.array([xi0,eta0,zeta0])
        X0 = tmp.flatten()
        #print(X0)
        S = np.linspace(0.0,self.length,self.NDIV)
        
        self.X1 = odeint(self.Sdot,X0,S)
        self.XI = self.X1[:,0:3]
        self.ETA = self.X1[:,3:6]
        self.ZETA = self.X1[:,6:9]
        self.POS = np.zeros((self.NDIV,3))
        self.ZETASDOT = np.zeros((self.NDIV,3))
        #線形補間によって近似した関数の積分を行う
        pre_s = 0
        pre_POS = np.zeros(3)
        for i in range(self.NDIV):
            if i==0:
                POSITION = np.zeros(3)
            else:
                POSITION = pre_POS + self.ZETA[i-1]*self.Ds + (self.ZETA[i]-self.ZETA[i-1])*(pre_s + self.Ds*0.5 - (i-1)*self.Ds)
            self.POS[i] = POSITION
            pre_POS = POSITION
            pre_s = i*self.Ds
            self.ZETASDOT[i] = self.XI[i]*self.omegaEta(S[i]) - self.ETA[i]*self.omegaXi(S[i])
    def zeta(self,s):
        p = s/self.Ds;
        n = int(p);
        q = p - float(n);
        if q == 0:
            ret = np.array([self.X1[n,6],self.X1[n,7],self.X1[n,8]])
            return ret
        else :
            ret_n = np.array([self.X1[n,6],self.X1[n,7],self.X1[n,8]])
            ret_n1 = np.array([self.X1[n+1,6],self.X1[n+1,7],self.X1[n+1,8]])
            return (1.0-q)*ret_n + q*ret_n1
    def xi(self,s):
        
        p = s/self.Ds;
        n = int(p);
        q = p - float(n);
        if q == 0:
            ret = np.array([self.X1[n,0],self.X1[n,1],self.X1[n,2]])
            return ret
        else :
            ret_n = np.array([self.X1[n,0],self.X1[n,1],self.X1[n,2]])
            ret_n1 = np.array([self.X1[n+1,0],self.X1[n+1,1],self.X1[n+1,2]])
            return (1.0-q)*ret_n + q*ret_n1

    def eta(self,s):
        
        p = s/self.Ds;
        n = int(p);
        q = p - float(n);
        if q == 0:
            ret = np.array([self.X1[n,3],self.X1[n,4],self.X1[n,5]])
            return ret
        else :
            ret_n = np.array([self.X1[n,3],self.X1[n,4],self.X1[n,5]])
            ret_n1 = np.array([self.X1[n+1,3],self.X1[n+1,4],self.X1[n+1,5]])
            return (1.0-q)*ret_n + q*ret_n1

    def zetaSdot(self,s):
        return self.omegaEta(s)*self.xi(s) - self.omegaXi(s)*self.eta(s)
    
def main():
    length = np.pi
    NDIV = 501
    def omegaXiL(s):
        return 0.0
    def omegaEtaL(s):
        #return 2.917010
        return 1.0
    def omegaZetaL(s):
        return 0.0
    obj_L = Coordinates(omegaXiL,omegaEtaL,omegaZetaL,length,NDIV)
    chi = 0.0
    #delta = 0.343004-math.pi/2.0 
    delta = -math.pi/2.0
    Schi = math.sin(chi);
    Cchi = math.cos(chi);
    Sdelta = math.sin(delta);
    Cdelta = math.cos(delta);
    xi0 = np.array([Cdelta,0,-Sdelta])
    eta0 = np.array([Schi*Sdelta,Cchi,Schi*Cdelta])
    zeta0 = np.array([Cchi*Sdelta,-Schi,Cchi*Cdelta])
    obj_L.DetermineAxies(xi0,eta0,zeta0)
    print("size = ")
    print(np.shape(obj_L.X1))
    print(np.shape(obj_L.XI))
    print(np.shape(obj_L.ETA))
    print(np.shape(obj_L.ZETA))
if __name__ == '__main__':
    main()
    """description of class"""

