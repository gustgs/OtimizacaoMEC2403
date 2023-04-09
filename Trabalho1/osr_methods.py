import numpy as np
import numdifftools as nd
import line_search_methods as lsm
from timeit import default_timer as timer

def univariante(P0, f, maxiter, tol, line_step):
    start = timer()
    dimens = P0.size
    Pmin = P0.copy()
    grad = nd.Gradient(f)
    passos = 0
    ciclos = 0   
    dirs = np.zeros(shape=(dimens,dimens))
    for i in np.arange(dimens):
        ek = np.zeros(dimens)
        ek[i] = 1
        dirs[i] = ek.copy()
      
    #listPmin = np.array([Pmin])
    listPmin = []
    listPmin.append(Pmin)   
    while(not np.allclose(grad(Pmin), np.zeros(dimens), atol=tol)):
        ciclos = ciclos + 1
        for i in np.arange(dimens):
            dir = dirs[i]
            P_pss_cte, intervalo, sentido = lsm.passo_cte(dir, Pmin, f, step=line_step)
            Pmin = lsm.secao_aurea(intervalo, sentido, Pmin, f)[0]
            passos = passos + 1
            #listPmin = np.append(listPmin, np.array([Pmin]), axis=0)
            listPmin.append(Pmin)
            if(np.allclose(grad(Pmin), np.zeros(dimens), atol = tol)):
                break
            if(passos == maxiter):
                print('Nao convergiu !')
                end = timer()
                return listPmin, passos, end-start
    end = timer()
    return listPmin, ciclos , passos, end-start

def powell(P0, f, maxiter, tol, line_step):
    start = timer()
    dimens = P0.size
    Pmin = P0.copy()
    grad = nd.Gradient(f)
    passos = 0
    ciclos = 0
    dirs = np.zeros(shape=(dimens+1,dimens))
    for j in np.arange(dimens):
        ek = np.zeros(dimens)
        ek[j] = 1
        dirs[j] = ek
        
    listPmin = []
    listPmin.append(Pmin)    
    while(not np.allclose(grad(Pmin), np.zeros(dimens), atol = tol)):
        P0 = Pmin
        ciclos = ciclos + 1
        for i in np.arange(dimens+1):
            if(i == dimens):
                dirs[i] = Pmin - P0
                
            dir = dirs[i].copy()
            P_pss_cte, intervalo, sentido = lsm.passo_cte(dir, Pmin, f, step=line_step)
            Pmin = lsm.secao_aurea(intervalo, sentido, Pmin, f)[0]
            listPmin.append(Pmin) 
            passos = passos + 1
        
            if( i > 0):
                dirs[i-1] = dirs[i].copy()
        
            if(np.allclose(grad(Pmin), np.zeros(dimens), atol = tol)):
                break
            if(passos == maxiter):
                print('Nao convergiu !')
                end = timer()
                return listPmin, passos, end-start
    end = timer()
    return listPmin, ciclos, passos, end-start

def newtonRaphson(P0, f, maxiter, tol, line_step):
    start = timer()
    dimens = P0.size
    Pmin = P0.copy()
    grad = nd.Gradient(f)
    H = nd.Hessian(f)
    passos = 0
    
    listPmin = []
    listPmin.append(Pmin)     
    while(not np.allclose(grad(Pmin), np.zeros(dimens), atol = tol)):
        if(passos == maxiter):
            print('Nao convergiu !')
            end = timer()
            return listPmin, passos, end-start
        dir = -np.linalg.inv(H(Pmin)).dot(grad(Pmin))
        P_pss_cte, intervalo, sentido = lsm.passo_cte(dir, Pmin, f, step=line_step)
        Pmin = lsm.secao_aurea(intervalo, sentido, Pmin, f)[0]
        listPmin.append(Pmin) 
        passos = passos + 1
    end = timer()
    return listPmin, passos, end-start

def steepestDescent(P0, f, maxiter, tol, line_step):
    start = timer()
    dimens = P0.size
    Pmin = P0.copy()
    grad = nd.Gradient(f)
    passos = 0
    
    listPmin = []
    listPmin.append(Pmin)     
    while(not np.allclose(grad(Pmin), np.zeros(dimens), atol = tol)):
        if(passos == maxiter):
            print('Nao convergiu !')
            end = timer()
            return listPmin, passos, end-start
        dir = -grad(Pmin)
        P_pss_cte, intervalo, sentido = lsm.passo_cte(dir, Pmin, f, step=line_step)
        Pmin = lsm.secao_aurea(intervalo, sentido, Pmin, f)[0]
        listPmin.append(Pmin) 
        passos = passos + 1
    end = timer()             
    return listPmin, passos, end-start

def fletcherReeves(P0, f, maxiter, tol, line_step):
    start = timer()
    dimens = P0.size
    Pmin = P0.copy()
    grad = nd.Gradient(f)
    passos = 0
    
    listPmin = []
    listPmin.append(Pmin)
    
    #primeiro passo
    grad_k = grad(P0)
    dir = -grad_k
    dir_k = np.copy(dir)
    
    if(np.allclose(grad(Pmin), np.zeros(dimens), atol = tol)):
        end = timer()
        return listPmin, passos, end-start
    
    P_pss_cte, intervalo, sentido = lsm.passo_cte(dir, Pmin, f)
    Pmin = lsm.secao_aurea(intervalo, sentido, Pmin, f)[0]
    listPmin.append(Pmin) 
    passos = passos + 1
    grad_knext = grad(Pmin)
         
    while(not np.allclose(grad(Pmin), np.zeros(dimens), atol = tol)):
        if(passos == maxiter):
            print('Nao convergiu !')
            end = timer()
            return listPmin, passos, end-start
        beta = (np.transpose(grad_knext)).dot(grad_knext)/((np.transpose(grad_k)).dot(grad_k))
        dir = -grad_knext + beta*dir_k
        P_pss_cte, intervalo, sentido = lsm.passo_cte(dir, Pmin, f, step=line_step)
        Pmin = lsm.secao_aurea(intervalo, sentido, Pmin, f)[0]
        listPmin.append(Pmin) 
        
        grad_k = np.copy(grad_knext)
        dir_k = np.copy(dir)
        grad_knext = grad(Pmin)
        passos = passos + 1
    end = timer()
    return listPmin, passos, end-start