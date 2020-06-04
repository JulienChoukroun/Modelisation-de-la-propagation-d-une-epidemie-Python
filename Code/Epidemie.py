
# -*- coding: utf-8 -*-
"""
@author: Choukroun Julien
         Gourdon Jessica
         Sagnes Luc
"""


import numpy as np
from numpy.linalg import *
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import copy

t0 = 0
tf = 100
deltaT = 1/24 # Le pas est d'un jour.
t = np.arange(t0,tf,deltaT)
n = t.shape[0]
N = 67000000 # La population de la France.

# Constantes
alpha = 0.3 # Représente la quantité d’individu sélectionnés pour être traités par unité de temps.
beta = 1 # Représente le nombre de personnes rencontrées par un individu par unité de temps.
gamma = 0.02 # Représente le taux de guérison par unité de temps, c’est-à-dire le taux déterminant le passage du compartiment I au compartiment R. Il concerne donc les personnes guérissant du virus ainsi que celles décédées.
delta = 0.8 # Réduit l’infectivité d’un individu d'un facteur delta.
eta = 0.4 # Représente le taux déterminant le passage du compartiment T au R. Il concerne donc les personnes traitées qui deviennent immunisés ou décèdent du virus, devenant donc des « rétablis ».

S0 = N-1 # Ensemble de la population N auquel on soustrait le patient zéro.
I0 = 1 # Initialement, il n’y a qu’une personne infectieuse : le patient zéro.
T0 = 0
R0 = 0

eps = 1e-12
Nmax = 100

X0 = np.zeros(4)
X0[0] = S0
X0[1] = I0
      
MatriceNewton = np.zeros((4,n))
MatriceNewton[:,0] = X0
MatricePointFixe = np.zeros((4,n))
MatricePointFixe[:,0] = X0
MatriceJacobi = np.zeros((4,n))
MatriceJacobi[:,0] = X0
MatriceGS = np.zeros((4,n))
MatriceGS[:,0] = X0
MatriceSOR = np.zeros((4,n))
MatriceSOR[:,0] = X0

X = X0

ValeurXNewton = [X0]
ValeurXPointFixe = [X0]
ValeurJacobi = [X0]
ValeurGS = [X0]
ValeurSOR = [X0]

def fonctionS(X):
        Res = -(beta/N)*(X[1]+delta*X[2])*X[0]
        return Res 
    
def fonctionI(X):
        Res = (beta/N)*X[0]*(X[1]+delta*X[2])-(alpha+gamma)*X[1]
        return Res
    
def fonctionT(X):
        Res = alpha*X[1]-eta*X[2]
        return Res

def fonctionR(X):
        Res = gamma*X[1]+eta*X[2]
        return Res
    
def f(X):
    fX = np.zeros(4)
    fX[0] = fonctionS(X)
    fX[1] = fonctionI(X)
    fX[2] = fonctionT(X)
    fX[3] = fonctionR(X)
    return fX

# Cette fonction sera notre fonction f dans la méthode de Newton.
def g(X):
    Res = X-ValeurXNewton[-1]-deltaT*f(X)
    return Res

# Cette fonction sera notre fonction F dans la méthode du point fixe.
def F(X):
    F = np.zeros(4)
    F[0] = ValeurXPointFixe[-1][0]-deltaT*(beta/N)*(X[1]+delta*X[2])*X[0]
    F[1] = ValeurXPointFixe[-1][1]+deltaT*((beta/N)*X[0]*(X[1]+delta*X[2])-(alpha+gamma)*X[1])
    F[2] = ValeurXPointFixe[-1][2]+deltaT*(alpha*X[1]-eta*X[2])
    F[3] = ValeurXPointFixe[-1][3]+deltaT*(gamma*X[1]+eta*X[2])
    return F

# Le but ici est de créer une fonction qui inverse une matrice. 
# Donc nous créons une fonction qui calcule le déterminant et une fonction qui calcule la comatrice.
def Determinant(M):
    # On initialise la valeur du déterminant à 0.
    det = 0
    # On vérifie au préalable que la matrice en question est carrée.
    if (np.shape(M)[0] != np.shape(M)[1]):
        print("La matrice doit être carrée")
    # Si la matrice est de taille 1*1.
    elif (np.shape(M)[0] == 1):
        det = M[0,0]
    # Si la matrice est de taille 2*2, on calcule le déterminant.
    elif (np.shape(M)[0] == 2):
        det = M[0,0] * M[1,1] - M[0,1] * M[1,0]
    # Si la matrice est de taille n*n, on calcule le déterminant par rapport à la 1ère colonne.
    else:
        for i in range(np.shape(M)[0]):
            m1 = copy.deepcopy(M) # Matrice tampon.
            m1 = np.delete(m1,0,1) # On supprime la 1ère colonne de la 1ère matrice tampon.
            m2 = copy.deepcopy(m1) # 2ème matrice tampon.
            m2 = np.delete(m2,i,0) # On supprime la ième ligne de la 2ème matrice tampon.
            det = det+M[i,0]*((-1)**(i))*Determinant(m2)  # Calcul du déterminant.
    return det

def Comatrice(M):
    # On vérifie au préalable que la matrice en question est carrée.
    if (np.shape(M)[0] != np.shape(M)[1]):
        print("La matrice doit être carrée")
    else:
        com = np.zeros(np.shape(M))
        for i in range(np.shape(M)[0]):
            for j in range(np.shape(M)[1]):
                m1 = copy.deepcopy(M) # Matrice tampon.
                m1 = np.delete(m1,i,0) # On supprime la ième ligne de la 1ère matrice tampon.
                m2 = copy.deepcopy(m1) # 2ème matrice tampon.
                m2 = np.delete(m2,j,1) # On supprime la jème colonne de la 2ème matrice tampon.
                com[i,j] = (-1)**(i+j)*Determinant(m2)
    return com

def Inverse(M):
    Minverse = np.zeros(np.shape(M))
    if (Determinant(M) == 0):
        print("La matrice n'est pas inversible")
    else:
        Mcom = Comatrice(M)
        Minverse = (1/Determinant(M))*Mcom.T
    return Minverse

# Jacobienne de x -> x -x[n] -deltat*f(x)
def Jacobienne(X):
    Jacobienne = np.zeros((4,4))
    Jacobienne[0,0] = 1+deltaT*(beta/N)*(X[1]+delta*X[2])
    Jacobienne[0,1] = deltaT*(beta/N)*X[0]
    Jacobienne[0,2] = deltaT*(beta/N)*X[0]*delta
    Jacobienne[1,0] = -deltaT*(beta/N)*(X[1]+delta*X[2])
    Jacobienne[1,1] = 1-deltaT*(beta/N)*X[0]+deltaT*(alpha+gamma)
    Jacobienne[1,2] = -deltaT*(beta/N)*X[0]*delta
    Jacobienne[2,1] = -deltaT*alpha
    Jacobienne[2,2] = 1+eta*deltaT
    Jacobienne[3,1] = -deltaT*gamma
    Jacobienne[3,2] = -eta*deltaT
    Jacobienne[3,3] = 1
    return Jacobienne

## Méthode d’Euler implicite pour une résolution d’un système d’équations non-linéaires.
# Methode de Newton.
def NewtonNd(f, df, x0, eps, Nmax):
    x = x0
    k = 0
    erreur = 1.0
    Erreur = []
    while(erreur>eps and k<Nmax):
          xold = copy.deepcopy(x)
          x = x-np.dot(Inverse(df(x)),f(x))
          k = k+1
          erreur = np.linalg.norm(x-xold)/np.linalg.norm(x)
          Erreur.append(erreur)
    return (x, Erreur)

def afficheNewton():
    ErreurNewton = []
    for i in range(1,n):
        Xdebut = ValeurXNewton[-1]
        (Xsuiv, Err) = NewtonNd(g, Jacobienne, Xdebut, eps, Nmax)
        MatriceNewton[:,i] = Xsuiv
        ValeurXNewton.append(Xsuiv)
        ErreurNewton = ErreurNewton+Err
    SN = MatriceNewton[0]
    IN = MatriceNewton[1]
    TN = MatriceNewton[2]
    RN = MatriceNewton[3]
    plt.figure()
    plt.plot(t,SN,label="Susceptibles")
    plt.plot(t,IN,label="Infectieux")
    plt.plot(t,TN,label="Traités")
    plt.plot(t,RN,label="Rétablis")
    plt.title("Méthode de Newton")
    plt.xlabel("Jours")
    plt.ylabel("Population")
    plt.legend()
    plt.show()
    return ErreurNewton

# Methode du point fixe.
def PointFixe(F, x0, eps, Nmax):
    x = x0
    k = 0
    erreur = np.linalg.norm(F(x)-x)
    Erreur = []
    Erreur.append(erreur)
    while(erreur>eps and k<Nmax):
          x = F(x)
          k = k+1
          erreur = np.linalg.norm(F(x)-x)
          Erreur.append(erreur)
    return (x, Erreur)

def affichePointFixe():
    ErreurPointFixe = []
    for i in range(1,n):
        Xdebut = ValeurXPointFixe[-1]
        (Xsuiv, Err) = PointFixe(F, Xdebut, eps, Nmax)
        MatricePointFixe[:,i] = Xsuiv
        ValeurXPointFixe.append(Xsuiv)
        ErreurPointFixe = ErreurPointFixe+Err
    SPF = MatricePointFixe[0]
    IPF = MatricePointFixe[1]
    TPF = MatricePointFixe[2]
    RPF = MatricePointFixe[3]
    plt.figure()
    plt.plot(t,SPF,label="Susceptibles")
    plt.plot(t,IPF,label="Infectieux")
    plt.plot(t,TPF,label="Traités")
    plt.plot(t,RPF,label="Rétablis")
    plt.title("Méthode du point fixe")
    plt.xlabel("Jours")
    plt.ylabel("Population")
    plt.legend()
    plt.show()       
    return ErreurPointFixe

## Méthode d’Euler explicite pour une résolution d’un système d’équations linéaires.
def créationA(X):
    M = np.zeros((4,4))
    M[0,0] = 1
    M[0,1] = -deltaT*(beta/N)*X[0]
    M[0,2] = -deltaT*(beta/N)*X[0]*delta
    M[1,1] = 1+deltaT*(beta/N)*X[0]-deltaT*(alpha+gamma)
    M[1,2] = deltaT*(beta/N)*X[0]*delta
    M[2,1] = deltaT*alpha
    M[2,2] = 1-deltaT*eta
    M[3,1] = deltaT*gamma
    M[3,2] = deltaT*eta
    M[3,3] = 1
    A = Inverse(M)
    return A

# Méthode de Jacobi
def Jacobi(A, b, x0, eps, Nmax):
    k = 0
    x = x0
    N = -np.triu(A,1)-np.tril(A,-1)
    invD = np.diag(1./np.diag(A))
    # np.diag(A) renvoie un vecteur contenant les elements diagonaux de A car A est une matrice.
    # np.diag() de ce vecteur là crée une matrice diagonale avec les éléments du vecteurs.
    normb = linalg.norm(b)
    residu = []
    residu.append(linalg.norm(np.dot(A,x)-b)/normb)
    while(linalg.norm(np.dot(A,x)-b)/normb>eps and k<Nmax):
        x = np.dot(invD,np.dot(N,x)+b)
        residu.append(linalg.norm(np.dot(A,x)-b)/normb)
        k = k+1
    return (x, k, residu)

def afficheJacobi():
    ErreurJacobi = []
    for i in range(1,n):
        Xdebut = ValeurJacobi[-1]
        A = créationA(Xdebut)
        (Xsuiv, k, Err) = Jacobi(A, Xdebut, Xdebut, eps, Nmax)
        MatriceJacobi[:,i] = Xsuiv
        ValeurJacobi.append(Xsuiv)
        ErreurJacobi = ErreurJacobi+Err
    SJ = MatriceJacobi[0]
    IJ = MatriceJacobi[1]
    TJ = MatriceJacobi[2]
    RJ = MatriceJacobi[3]
    plt.figure()
    plt.plot(t,SJ,label="Susceptibles")
    plt.plot(t,IJ,label="Infectieux")
    plt.plot(t,TJ,label="Traités")
    plt.plot(t,RJ,label="Rétablis")
    plt.title("Méthode de Jacobi")
    plt.xlabel("Jours")
    plt.ylabel("Population")
    plt.legend()
    plt.show()
    return ErreurJacobi

# Cette fonction est utilisée dans la méthode de Gauss-Seidel et dans la méthode SOR.
def Descente(A, b):
    n = A.shape[0]
    x = np.zeros(n)
    x[0] = b[0]/A[0,0]
    for i in range(1,n):
        # De la 2ème ligne à la derniere ligne.
        x[i] = (b[i]-np.dot(A[i,0:i],x[0:i]))/A[i,i]
    return x

# Méthode de Gauss-Seidel.
def GaussSeidel(A, b, x0, eps, Nmax):
    k = 0
    x = x0
    N = -np.triu(A,1)
    M = np.tril(A)
    normb = linalg.norm(b)
    residu = []
    residu.append(linalg.norm(np.dot(A,x)-b)/normb)
    while(linalg.norm(np.dot(A,x)-b)/normb>eps and k<Nmax):
        x = Descente(M,(np.dot(N,x)+b))  
        # M = D-E triangulaire inférieure.
        residu.append(linalg.norm(np.dot(A,x)-b)/normb)
        k = k+1
    return (x, k, residu)

def afficheGaussSeidel():
    ErreurGS = []
    for i in range(1,n):
        Xdebut = ValeurGS[-1]
        A = créationA(Xdebut)
        (Xsuiv, k, Err) = GaussSeidel(A, Xdebut, Xdebut, eps, Nmax)
        MatriceGS[:,i] = Xsuiv
        ValeurGS.append(Xsuiv)
        ErreurGS = ErreurGS+Err
    SGS = MatriceGS[0]
    IGS = MatriceGS[1]
    TGS = MatriceGS[2]
    RGS = MatriceGS[3]
    plt.figure()
    plt.plot(t,SGS,label="Susceptibles")
    plt.plot(t,IGS,label="Infectieux")
    plt.plot(t,TGS,label="Traités")
    plt.plot(t,RGS,label="Rétablis")
    plt.title("Méthode de Gauss-Seidel")
    plt.xlabel("Jours")
    plt.ylabel("Population")
    plt.legend()
    plt.show()    
    return ErreurGS 

# Méthode SOR.
def SOR(A,b,x0,w,eps,Nmax):
    k = 0
    x = x0
    N = -np.triu(A,1)+(1.0-w)/w*np.diag(np.diag(A))
    M = np.diag(np.diag(A))/w+np.tril(A,-1)
    normb = linalg.norm(b)
    residu = []
    residu.append(linalg.norm(np.dot(A,x)-b)/normb)
    while(linalg.norm(np.dot(A,x)-b)/normb>eps and k<Nmax):
        x = Descente(M,(np.dot(N,x)+b))
        residu.append(linalg.norm(np.dot(A,x)-b)/normb)
        k = k+1
    return (x, k, residu)

def afficheSOR():
    ErreurSOR=[]
    for i in range(1,n):
        Xdebut = ValeurSOR[-1]
        A = créationA(Xdebut)
        (Xsuiv, k, Err) = SOR(A, Xdebut, Xdebut, 0.5, eps, Nmax)
        #0<w<2 pour convergence
        MatriceSOR[:,i] = Xsuiv
        ValeurSOR.append(Xsuiv)
        ErreurSOR = ErreurSOR+Err
    SSOR = MatriceSOR[0]
    ISOR = MatriceSOR[1]
    TSOR = MatriceSOR[2]
    RSOR = MatriceSOR[3]
    plt.figure()
    plt.plot(t,SSOR,label="Susceptibles")
    plt.plot(t,ISOR,label="Infectieux")
    plt.plot(t,TSOR,label="Traités")
    plt.plot(t,RSOR,label="Rétablis")
    plt.title("Méthode SOR")
    plt.xlabel("Jours")
    plt.ylabel("Population")
    plt.legend()
    plt.show()     
    return ErreurSOR

# On peut comparer plusieurs méthodes entres elles en regardant le nombre d'erreurs que l'on fait pour chaque méthode.
def ComparateurConvergence(ErreurNewton, ErreurPointFixe, ErreurJacobi, ErreurGS, ErreurSOR):
    cptNewton = len(ErreurNewton)
    cptPointFixe = len(ErreurPointFixe)
    cptJacobi = len(ErreurJacobi)
    cptGS = len(ErreurGS)
    cptSOR = len(ErreurSOR)
    print("Nombres d'erreurs de la méthode de Newton :")
    print(cptNewton)
    print("Nombres d'erreurs de la méthode du point fixe :")
    print(cptPointFixe)
    print("Nombres d'erreurs de la méthode de Jacobi :")
    print(cptJacobi)
    print("Nombres d'erreurs de la méthode de Gauss-Seidel :")
    print(cptGS)
    print("Nombres d'erreurs de la méthode SOR :")
    print(cptSOR)
    if(cptNewton>cptPointFixe and cptJacobi>cptPointFixe and cptGS>cptPointFixe and cptSOR>cptPointFixe):
        print("La méthode du point fixe converge plus rapidement.")
    elif(cptPointFixe>cptNewton and cptJacobi>cptNewton and cptGS>cptNewton and cptSOR>cptNewton):
        print("La méthode de Newton converge plus rapidement.")
    elif (cptNewton>cptJacobi and cptGS>cptJacobi and cptPointFixe>cptJacobi and cptSOR>cptJacobi):
        print("La méthode de Jacobi converge plus rapidement.")
    elif (cptNewton>cptSOR and cptPointFixe>cptSOR and cptJacobi>cptSOR and cptGS>cptSOR):
        print ("La méthode SOR converge plus rapidement.")
    else :
        print ("La méthode de Gauss-Seidel converge plus rapidement.")


ErreurNewton = afficheNewton()
ErreurPointFixe = affichePointFixe()
ErreurJacobi = afficheJacobi()
ErreurGS = afficheGaussSeidel()
ErreurSOR = afficheSOR()
ComparateurConvergence(ErreurNewton, ErreurPointFixe, ErreurJacobi, ErreurGS, ErreurSOR)