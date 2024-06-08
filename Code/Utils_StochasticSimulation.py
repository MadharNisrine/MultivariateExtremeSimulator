import numpy as np
from copulas.univariate import GaussianKDE
from copulas.bivariate import Clayton
from scipy.stats import t
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
import pandas as pd
import random
from scipy.stats import genpareto
from scipy.stats import expon,poisson,pareto
import pathlib
from scipy import integrate
 
 
#########################################################################
#
#----------------------TRM Theoretical values--------------
#
#########################################################################
 
def CGumbStudentK3(Fx,Fy,Fz,theta):
    return np.exp(-((-np.log(Fx))**(theta) + (-np.log(Fy))**(theta)+ (-np.log(Fz))**(theta))**(1/theta))
def xdxCGumbStudentK3(x,theta,nu,Fy,Fz):
    Fx= t.cdf(x,df=nu)
    fx = t.pdf(x,df=nu)
    return x*fx/Fx * (-np.log(Fx))**(theta-1) * ((-np.log(Fx))**(theta) + (-np.log(Fy))**(theta)+ (-np.log(Fz))**(theta))**(1/theta-1) * np.exp(-((-np.log(Fx))**(theta) + (-np.log(Fy))**(theta)+ (-np.log(Fz))**(theta))**(1/theta))
 
 
def getRiskMeasuresStudentGumbelK3(alpha,nuVect=[2,3,2.5],theta=0.7):
    nu1,nu2,nu3 = nuVect
    " Return VaR, ES, DCTE for a given risk factor in Data at level alpha"
    VaRTheoX = t.ppf(alpha,df=nu1)
    ES = (nu1+t.ppf(alpha,df=nu1)**2)/((nu1-1)*(1-alpha)) * t.pdf(t.ppf(alpha,df=nu1),df=nu1)
   
    NormalizingConstant = (1 - 3*alpha + 3*CGumbStudentK3(alpha,alpha,1,theta) -   CGumbStudentK3(alpha,alpha,alpha,theta) )**(-1)
    VaRTheoY = t.ppf(alpha,df=nu2)
    Fy = t.cdf(VaRTheoY,df=nu2)
   
    VaRTheoZ = t.ppf(alpha,df=nu3)
    Fz = t.cdf(VaRTheoZ,df=nu3)
 
 
    #IC = integrate.quad(xdxCGumbStudentK3,VaRTheoX,np.inf,args=(theta,nu1,Fy,Fz))[0]
   
    
    IC1 = integrate.quad(xdxCGumbStudentK3,VaRTheoX,np.inf,args=(theta,nu1,Fy,1))[0]
    IC2 = integrate.quad(xdxCGumbStudentK3,VaRTheoX,np.inf,args=(theta,nu1,1,Fz))[0]
    IC3 = integrate.quad(xdxCGumbStudentK3,VaRTheoX,np.inf,args=(theta,nu1,Fy,Fz))[0]
    IC = IC1 + IC2 - IC3
 
    DCTE = NormalizingConstant *((1-alpha)*ES- IC)
   
    ## MES
    
    NormalizingConstantMES = (1 - 2*alpha + CGumbStudentK3(1,alpha,alpha,theta) )**(-1)
   
    ICMES1 = integrate.quad(xdxCGumbStudentK3,-np.inf,np.inf,args=(theta,nu1,Fy,1))[0]
    ICMES2 = integrate.quad(xdxCGumbStudentK3,-np.inf,np.inf,args=(theta,nu1,1,Fz))[0]
    ICMES3 = integrate.quad(xdxCGumbStudentK3,-np.inf,np.inf,args=(theta,nu1,Fy,Fz))[0]
    ICMES = - ICMES1 - ICMES2 + ICMES3
    MES = ICMES * NormalizingConstantMES
    return VaRTheoX, ES, DCTE, MES
 
 
 
#########################################################################
#
#------------------------ Empricial TRMs Estimation --------------------
#
#########################################################################
 
 
 
def getRiskMeasuresEmpiricalMES(data,idxRF,alpha,ExcdTheo=False,VaRRF= [0,0]):
    " Return Empirical estimation of VaR, ES, DCTE,MES for a given risk factor in Data at level alpha if ExcdTheo then ES,DCTE,MES are computed when exceeding theoretical VaR level"
    X = data.iloc[:,idxRF].values
    n,d = data.shape
    if ExcdTheo:
        VaR = VaRRF
       
    else :
        VaR = np.quantile(data,alpha,axis=0)
       
    ES = np.mean(X[X>VaR[idxRF]])
    ## DCTE Computation
    T_Emp = set(list((np.arange(n)[data.iloc[:,0]>VaR[0]]).reshape(-1,)))
    for i in range(d):
        T_Emp=set(T_Emp&set(list((np.arange(n)[data.iloc[:,i]>VaR[i]]).reshape(-1,))))
    T_Emp = list(T_Emp)
    DCTE = np.mean(X[T_Emp])
    Nb_ExcDep = len(T_Emp)
 
    ## MES Computation
    idxList = list(np.arange(d)) #with all index
    idxList.remove(idxRF) #without index of risk factor of interest
    T_Emp = set(list((np.arange(n)[data.iloc[:,idxList[0]]>VaR[idxList[0]]]).reshape(-1,)))
    for i in idxList[1:]:
        T_Emp=set(T_Emp&set(list((np.arange(n)[data.iloc[:,i]>VaR[i]]).reshape(-1,))))
    T_Emp = list(T_Emp)
    MES = np.mean(X[T_Emp])
    Nb_ExcMES = len(T_Emp)
   
    return VaR,ES,DCTE,Nb_ExcDep,MES,Nb_ExcMES
 
 
 
#########################################################################
#
#----------------Joint Simulation algorithm with fixed seed--------------
#
#########################################################################
 
 
def prodInd(j,deltaMatrix):
    vectDelta_j = list(deltaMatrix[:,j])
    vectDelta_j.remove(vectDelta_j[j])
    return np.prod(np.array(vectDelta_j)<0)
def ComputeWk(k,deltaMatrix):
    K = deltaMatrix.shape[0]
    Delta = list(deltaMatrix[k,:])
    Delta.remove(Delta[k])
    Ind = [prodInd(j,deltaMatrix)  for j in range(K) if j != k]
    WwoE = np.dot(np.array(Delta),np.array(Ind))
    return WwoE
 
def DeltaMatrix(i,DELTA,K):
    deltaMatrix_i = np.zeros((K,K))
    deltaMatrix_i[0,1:] = DELTA.iloc[i,:]
    deltaMatrix_i[:,0] = - deltaMatrix_i[0,:]
    for k in range(1,K):
        for l in range(k,K):
            if k != l:
                deltaMatrix_i[k,l] = deltaMatrix_i[0,l] - deltaMatrix_i[0,k]
                deltaMatrix_i[l,k] = -deltaMatrix_i[k,l]
 
    return deltaMatrix_i
 
def simulation(sample,M,replacing=False,seed=20231229):
    np.random.seed(seed)
    N,K = sample.shape
    DELTA = pd.DataFrame(columns=np.arange(K-1),index=np.arange(N))
    newSample = pd.DataFrame(columns=np.arange(K),index=np.arange(M))
    for k in range(0,K-1):
        DELTA.iloc[:,k] = sample.iloc[:,0] - sample.iloc[:,k+1]
    if replacing :
        Msample = DELTA.iloc[np.random.choice(list(np.arange(N)),M),:]
    else :
        Msample = DELTA.iloc[random.sample(list(np.arange(N)),M),:]
    E = expon.rvs(size=M)
 
    for m in range(M):
        deltaMatrix_m = DeltaMatrix(m,Msample,K)
        for k in range(K):
            newSample.iloc[m,k] = E[m] +  ComputeWk(k,deltaMatrix_m)
    return newSample
 
 
 
#########################################################################
#
#---------------- ES Estimation with joint Simulation --------------
#
#########################################################################
 
def StabilityNewSamples(ExcessStand,X,ParamsOriginScale,VaR,M=1000,K=100,seed=20231228):
    # ParamsOriginScale = ['gamma','sigma','uX']
    nu,uX0 = ParamsOriginScale
    ESSimu = []
    ESExt = []
    Nb_VaR = [] # Nb of observations in new sample beyon the VaR level
    for k in range(K):
        NewExcessStandard = simulation(pd.DataFrame(ExcessStand),M,True,seed)
        NewExcess = np.zeros(NewExcessStandard.shape[0])
        NewExcess = t.ppf((1-np.exp(-np.array(NewExcessStandard.iloc[:,0].values+uX0,dtype=float))),df=nu)
 
        #NewExcess = s*(np.exp(np.array(NewExcessStandard.iloc[:,0].values,dtype=float)*g) - 1)/g
        Nb_VaR.append(sum(NewExcess>VaR))
        ESSimu.append(np.mean(NewExcess[NewExcess>VaR] ))
 
        DataExtended = np.concatenate([NewExcess[NewExcess>VaR],X[X>VaR]])
        ESExt.append(np.mean(DataExtended))
    return ESExt,ESSimu,Nb_VaR
 
 
#########################################################################
#
#----------------DCTE Estimation with joint Simulation --------------
#
#########################################################################
 
 
def StabilityonlyDCTE_NewSamples(ExcessStand,Excess,ParamsOriginScale,VaR,M=1000,K=100,seed=20231229):
    # ParamsOriginScale = ['gamma','sigma','uX']
    nu1,uX0,nu2,uY0,nu3,uZ0 = ParamsOriginScale
    VaRX,VaRY,VaRZ = VaR
   
    DCTE_SimuX = []
    DCTE_ExtX = []
   
 
   
    DCTE_SimuY = []
    DCTE_ExtY = []
   
    DCTE_SimuZ = []
    DCTE_ExtZ = []
   
    Nb_DCTE = []
    for k in range(K):
        NewExcessStandard = simulation(pd.DataFrame(ExcessStand),M,True,seed)
       
        NewExcessX = np.zeros(NewExcessStandard.shape[0])
        #NewExcessX = s1*(np.exp(np.array(NewExcessStandard.iloc[:,0].values,dtype=float)*g1) - 1)/g1
        NewExcessX = t.ppf((1-np.exp(-np.array(NewExcessStandard.iloc[:,0].values+uX0,dtype=float))),df=nu1)
       
        NewExcessY = np.zeros(NewExcessStandard.shape[0])
        NewExcessY = t.ppf((1-np.exp(-np.array(NewExcessStandard.iloc[:,1].values+uY0,dtype=float))),df=nu2)
        #NewExcessY = s2*(np.exp(np.array(NewExcessStandard.iloc[:,1].values,dtype=float)*g2) - 1)/g2
       
        NewExcessZ = np.zeros(NewExcessStandard.shape[0])
        NewExcessZ = t.ppf((1-np.exp(-np.array(NewExcessStandard.iloc[:,2].values+uZ0,dtype=float))),df=nu3)
        #NewExcessZ = s3*(np.exp(np.array(NewExcessStandard.iloc[:,2].values,dtype=float)*g3) - 1)/g3
        
 
       
        DataExtended = np.concatenate([(NewExcessX).reshape(-1,1),(NewExcessY).reshape(-1,1),(NewExcessZ).reshape(-1,1)],axis=1)
        DataExtended = np.concatenate([DataExtended,Excess])
       
                        
        ## DCTE Estimation
        
        
        Nexc = len(NewExcessX)
        TXsupa = set(list((np.arange(Nexc)[NewExcessX>VaRX]).reshape(-1,)))
        TY = set((np.arange(Nexc)[NewExcessY>VaRY]).reshape(-1,))
        TZ = set((np.arange(Nexc)[NewExcessZ>VaRZ]).reshape(-1,))
 
        T_Sim = list(set(list(TZ&TY&TXsupa)))
        Nb_DCTE.append(len(T_Sim))
        DCTE_SimuX.append(np.mean(NewExcessX[T_Sim]))
       
        Nexc = DataExtended.shape[0]
        TXsupa = set(list((np.arange(Nexc)[DataExtended[:,0]>VaRX]).reshape(-1,)))
        TY = set((np.arange(Nexc)[DataExtended[:,1]>VaRY]).reshape(-1,))
        TZ = set((np.arange(Nexc)[DataExtended[:,2]>VaRZ]).reshape(-1,))
 
        T_Ext = list(set(list(TZ&TY&TXsupa)))
 
        DCTE_ExtX.append(np.mean(DataExtended[T_Ext,0]))
 
 
 
        DCTE_SimuY.append(np.mean(NewExcessY[T_Sim]))
        DCTE_ExtY.append(np.mean(DataExtended[T_Ext,1]))
       
        DCTE_SimuZ.append(np.mean(NewExcessZ[T_Sim]))
        DCTE_ExtZ.append(np.mean(DataExtended[T_Ext,2]))
       
    return DCTE_SimuX,DCTE_ExtX,DCTE_SimuY,DCTE_ExtY,DCTE_SimuZ,DCTE_ExtZ,Nb_DCTE
 
 
#########################################################################
#
#---------------- MMES Estimation with joint Simulation --------------
#
#########################################################################
 
 
def StabilityonlyMES_NewSamples(ExcessStand,Excess,ParamsOriginScale,VaR,M=1000,K=100,seed=20231229):
    # ParamsOriginScale = ['gamma','sigma','uX']
    nu1,uX0,nu2,uY0,nu3,uZ0 = ParamsOriginScale
    VaRX,VaRY,VaRZ = VaR
   
    MES_SimuX = []
    MES_ExtX = []
   
 
   
    MES_SimuY = []
    MES_ExtY = []
   
    MES_SimuZ = []
    MES_ExtZ = []
   
    Nb_MES = []
    for k in range(K):
        NewExcessStandard = simulation(pd.DataFrame(ExcessStand),M,True,seed)
       
        NewExcessX = np.zeros(NewExcessStandard.shape[0])
        #NewExcessX = s1*(np.exp(np.array(NewExcessStandard.iloc[:,0].values,dtype=float)*g1) - 1)/g1
        NewExcessX = t.ppf((1-np.exp(-np.array(NewExcessStandard.iloc[:,0].values+uX0,dtype=float))),df=nu1)
       
        NewExcessY = np.zeros(NewExcessStandard.shape[0])
        NewExcessY = t.ppf((1-np.exp(-np.array(NewExcessStandard.iloc[:,1].values+uY0,dtype=float))),df=nu2)
        #NewExcessY = s2*(np.exp(np.array(NewExcessStandard.iloc[:,1].values,dtype=float)*g2) - 1)/g2
       
        NewExcessZ = np.zeros(NewExcessStandard.shape[0])
        NewExcessZ = t.ppf((1-np.exp(-np.array(NewExcessStandard.iloc[:,2].values+uZ0,dtype=float))),df=nu3)
        #NewExcessZ = s3*(np.exp(np.array(NewExcessStandard.iloc[:,2].values,dtype=float)*g3) - 1)/g3
       
 
       
        DataExtended = np.concatenate([(NewExcessX).reshape(-1,1),(NewExcessY).reshape(-1,1),(NewExcessZ).reshape(-1,1)],axis=1)
        DataExtended = np.concatenate([DataExtended,Excess])
       
                        
        ## MES Estimation
        
        
        Nexc = len(NewExcessX)
        #TXsupa = set(list((np.arange(Nexc)[NewExcessX+uX>VaRX]).reshape(-1,)))
        TY = set((np.arange(Nexc)[NewExcessY>VaRY]).reshape(-1,))
        TZ = set((np.arange(Nexc)[NewExcessZ>VaRZ]).reshape(-1,))
 
        T_Sim = list(set(list(TZ&TY)))
        Nb_MES.append(len(T_Sim))
        MES_SimuX.append(np.mean(NewExcessX[T_Sim]))
       
        Nexc = DataExtended.shape[0]
        #TXsupa = set(list((np.arange(Nexc)[DataExtended[:,0]>VaRX]).reshape(-1,)))
        TY = set((np.arange(Nexc)[DataExtended[:,1]>VaRY]).reshape(-1,))
        TZ = set((np.arange(Nexc)[DataExtended[:,2]>VaRZ]).reshape(-1,))
 
        T_Ext = list(set(list(TZ&TY)))
 
        MES_ExtX.append(np.mean(DataExtended[T_Ext,0]))
 
 
 
        MES_SimuY.append(np.mean(NewExcessY[T_Sim]))
        MES_ExtY.append(np.mean(DataExtended[T_Ext,1]))
       
        MES_SimuZ.append(np.mean(NewExcessZ[T_Sim]))
        MES_ExtZ.append(np.mean(DataExtended[T_Ext,2]))
       
    return MES_SimuX,MES_ExtX,MES_SimuY,MES_ExtY,MES_SimuZ,MES_ExtZ,Nb_MES
 
def SummaryErrorEmpSimuExt(Emp,Sim,Ext,Theo,NbVaREmp,NbVaR,TRMname):
    print('-'*10+TRMname+'-'*10)
    print(f"Relative error on empirical estimation %.2f "%((Emp-Theo)/Theo))
    print(f"Empirical estimation %.2f, Theoretical value %.2f "%(Emp,Theo))
    print(f"Nb. of Observations above VaR Level %.0f on original sample"%(NbVaREmp))
 
    print(f"\nRelative error of empirical estimation on simulated sample %.3f "%((Sim[0]-Theo)/Theo))
    print(f"Empirical estimation on simulated sample %.2f, Theoretical value %.2f "%(Sim[0],Theo))
    print(f"Nb. of Observations above VaR Level %.0f on simulated sample"%(NbVaR[0]))
 
    print(f"\nRelative error of empirical estimation on extended sample %.3f "%((Ext[0]-Theo)/Theo))
    print(f"Empirical estimation on extended sample %.2f, Theoretical value %.2f "%(Ext[0],Theo))
