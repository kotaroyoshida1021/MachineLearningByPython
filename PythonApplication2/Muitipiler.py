import numpy as np
from scipy.integrate import odeint
from scipy import integrate
from scipy import optimize
import sys
import OptimizeClass
MAX = 30
class Multipiler():
    """description of class"""
    def __init__(self,obj,x0,method,CondFunc,IneqFunc,NCOND,NINEQ,cbf):
        self.obj = obj
        self.x0 = x0
        self.method = method
        self.CondFunc = CondFunc
        self.IneqFunc = IneqFunc
        self.NCOND = NCOND
        self.NINEQ = NINEQ
        self.cbf = cbf
        ####calc####
        self.Lambda = np.zeros(self.NCOND)
        self.Mu = np.zeros(self.NINEQ)
        self.r = np.ones(self.NCOND)*10
        self.s = np.ones(self.NINEQ)*10
        self.c = sys.float_info.max
        self.alpha = 10
        self.beta = 0.25
    def Lagrangian(self,a):
        if(self.NCOND!=0):
            COND = self.CondFunc(a)
            PenaltyFunc_ForCond = np.dot(COND,self.Lambda)+np.dot(COND**2,self.r)
        else:
            PenaltyFunc_ForCond = 0.0
        if(self.NINEQ!=0):
            INEQ = self.IneqFunc(a)
            TMP = self.Mu + self.s*INEQ
            def INEQ_FUNC(i):
                if TMP[i]<0:
                    return -0.5*self.Mu[i]*self.Mu[i]/self.s[i]
                else:
                    return self.Mu[i]*INEQ[i]+0.5*self.s[i]*INEQ[i]**2
            FORCALC = [INEQ_FUNC(i) for i in range(self.NINEQ)]
            PenaltyFunc_ForIneq = np.sum(np.array(FORCALC))
        else:
            PenaltyFunc_ForIneq = 0.0
        return self.obj(a) + PenaltyFunc_ForCond + PenaltyFunc_ForIneq
    
    def LaunchOptimize(self,eps):
        for i in range(MAX):
            print("================",i,"-th iteration================")
            if(self.method=="Powell"):
                a = optimize.minimize(fun=self.Lagrangian,x0=self.x0,method = self.method,callback=self.cbf)
            elif self.method == "Nelder-Mead":
                a = optimize.minimize(fun=self.Lagrangian,x0=self.x0,method = self.method,callback=self.cbf,options={'adaptive': True})
            #a = optimize.minimize(self.Lagrangian,method = self.method)
            #準ニュートンおよびCGには未対応
            NCOND = self.NCOND
            NINEQ = self.NINEQ
            
            print("==================params=========================")
            print(a)
            G_max = 0
            H_max = 0
            print("=================================================")
            if(NCOND!=0):
                print("=================CONDITION=======================")
                COND = self.CondFunc(a.x)
                print(COND)
            if(NINEQ!=0):
                print("=================INEQUALITIES=======================")
                INEQ = self.IneqFunc(a.x)
                print(NINEQ)
            if(NCOND!=0):
                G = np.abs(COND)
                try:
                    G_max = np.amax(G)
                except ValueError:
                    G_max = 0
            if(NINEQ!=0):
                H = np.maximum(H,-self.Mu/self.s)
                try:
                    H_max = np.amax(H)
                except ValueError:
                    H_max = -1
            c = max(G_max,H_max)
            if c<eps:
                break
            else:
                if(NCOND!=0):
                    self.Lambda = self.Lambda + self.r*COND
                    for i in range(self.NCOND):
                        if G[i]<self.beta*c:
                            self.r[i] *= self.alpha
                if(NINEQ!=0):
                    Z = np.zeros(self.NINEQ)
                    self.Mu = np.maximum(Z,self.Mu+self.s*INEQ)
                
                    for i in range(self.NINEQ):
                        if H[i]<self.beta*c:
                            self.s[i] *= self.alpha
      
        return a
    #Debug
    
#def objective(a):
#    return a[0]*a[0]+a[1]*a[1]

#def CondFunc(a):
 #   ret = np.sum(a)-1
#    lt = [ret]
#    return np.array(lt)

#def IneqFunc(a):
#    return np.array([])

#def cbf(a):
##    f = objective(a)
#    print("\r f = %f"%(f))

#def main():
#    x0 = np.array([0.0,0.0])
#    multi = Multipiler(objective,x0,"Nelder-Mead",CondFunc,IneqFunc,1,0,cbf)
#    multi.LaunchOptimize(1.0e-5)


#if __name__ == '__main__':
#    main()


