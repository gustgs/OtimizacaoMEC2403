import numpy as np
import line_search_methods as lsm
from timeit import default_timer as timer

def univariante(P0, f, grad_f, maxiter, tol, line_step):
    start = timer()
    dimens = P0.size
    Pmin = P0.copy()
    passos = 0
    ciclos = 0   
    dirs = np.zeros(shape=(dimens,dimens))
    for i in np.arange(dimens):
        ek = np.zeros(dimens)
        ek[i] = 1
        dirs[i] = ek.copy()
      
    listPmin = []
    listPmin.append(Pmin)   
    while(np.linalg.norm(grad_f(Pmin)) >tol):
        ciclos = ciclos + 1
        for i in np.arange(dimens):
            dir = dirs[i]
            P_pss_cte, intervalo, sentido, alpha_pc = lsm.passo_cte(dir, Pmin, f, step=line_step)
            Pmin, alpha_sau = lsm.secao_aurea(intervalo, dir, sentido, Pmin, f)
            passos = passos + 1
            listPmin.append(Pmin)
            if(np.linalg.norm(grad_f(Pmin)) <= tol):
                break
            if(passos == maxiter):
                print('Nao convergiu !')
                end = timer()
                return listPmin, passos, end-start
    end = timer()
    return listPmin, ciclos , passos, end-start

def powellv2(P0, f, grad_f, maxiter, tol, line_step):
    
    grad = grad_f(P0)
    norm_grad = np.linalg.norm(grad)
    Min = P0.copy()
    
    e1 = np.array([1,0])
    e2 = np.array([0,1])
    canonicas = np.array([e1, e2], dtype=float)
    passos = 0
    ciclos = 0
    direcoes = canonicas.copy()
    while(norm_grad > tol):
        ciclos = ciclos + 1
        P1 = Min
        if (ciclos%4 == 0):
            direcoes = canonicas.copy()
        
        for i in np.arange(3):
            if (i == 2):
                d = Min - P1
                direcoes[1] = d                
            else:
                d = direcoes[i].copy()
              
            P_pss_cte, intervalo, sentido, alpha = lsm.passo_cte(d, Min, f, step=line_step)
            Min = lsm.secao_aurea(intervalo, sentido, Min, f)[0]
            grad = grad_f(Min)
            norm_grad = np.linalg.norm(grad)
            passos = passos + 1
            
            if (i==1):
                direcoes[0] = direcoes[1].copy()
            if (passos == maxiter):
                print('Nao convergiu')
                break
            print(f'Ciclo={ciclos}, Passo={passos}, d={d}, P={Min}, Grad={grad}, Norm_grad={norm_grad}')
        
        if(passos == maxiter):
            break
    return Min, ciclos, passos                  

def powell(P0, f, grad_f, maxiter, tol, line_step):
    start = timer()
    dimens = P0.size
    Pmin = P0.copy()
    #grad = nd.Gradient(f)
    passos = 0
    ciclos = 0
    dirs_canonicas = np.zeros(shape=(dimens+1,dimens))
    for j in np.arange(dimens):
        ek = np.zeros(dimens)
        ek[j] = 1
        dirs_canonicas[j] = ek
    
    dirs = dirs_canonicas.copy()    
    listPmin = []
    listPmin.append(Pmin)    
    while(np.linalg.norm(grad_f(Pmin)) > tol):
        P1 = Pmin
        ciclos = ciclos + 1
        
        if (ciclos%(dimens+2) == 0):
            dirs = dirs_canonicas.copy()
            
        for i in np.arange(dimens+1):
            if(i == dimens):
                dirs[i] = Pmin - P1
                
            dir = dirs[i].copy()
            P_pss_cte, intervalo, sentido, alpha = lsm.passo_cte(dir, Pmin, f, step=line_step)
            Pmin = lsm.secao_aurea(intervalo, dir, sentido, Pmin, f)[0]
            listPmin.append(Pmin) 
            passos = passos + 1
            print(f'Ciclo={ciclos}, Passso {passos}: dir={dir}, grad={grad_f(Pmin)}, norm_grad={np.linalg.norm(grad_f(Pmin))} P={Pmin} ')
        
            if( i > 0):
                dirs[i-1] = dirs[i].copy()
        
            if(np.linalg.norm(grad_f(Pmin)) < tol):
                break
            if(passos == maxiter):
                print('Nao convergiu !')
                end = timer()
                return listPmin, passos, end-start
    end = timer()
    return listPmin, ciclos, passos, end-start

def newtonRaphson(P0, f, grad_f, hessian_f, maxiter, tol, line_step):
    start = timer()
    dimens = P0.size
    Pmin = P0.copy()
    passos = 0
    
    listPmin = []
    listPmin.append(Pmin)
    grad = grad_f(Pmin)
    norm_grad = np.linalg.norm(grad)
    while(norm_grad > tol):
        if(passos == maxiter):
            print('Nao convergiu !')
            end = timer()
            return listPmin, passos, end-start
        hess = hessian_f(Pmin)
        dir = -np.linalg.inv(hess).dot(grad)
        P_pss_cte, intervalo, sentido, alpha = lsm.passo_cte(dir.copy(), Pmin.copy(), f, step=line_step)
        Pmin, alpha_min = lsm.secao_aurea(intervalo, dir.copy(), sentido, Pmin.copy(), f)
        listPmin.append(Pmin.copy()) 
        grad = grad_f(Pmin)
        norm_grad = np.linalg.norm(grad)
        
        passos = passos + 1
        #print(f'Passo{passos}, P={Pmin}, alpha={alpha_min}, d={dir}, norm_grad={norm_grad}, grad={grad}, hessian={hess}')
    end = timer()
    return listPmin, passos, end-start

def steepestDescent(P0, f, grad_f, maxiter, tol, line_step):
    start = timer()
    dimens = P0.size
    Pmin = P0.copy()
    grad = grad_f(Pmin)
    norm_grad = np.linalg.norm(grad)
    passos = 0
    
    listPmin = []
    listPmin.append(Pmin)     
    while(norm_grad > tol):
        if(passos == maxiter):
            print('Nao convergiu !')
            end = timer()
            return listPmin, passos, end-start
        dir = -grad
        P_pss_cte, intervalo, sentido, alpha = lsm.passo_cte(dir, Pmin, f, line_step)
        Pmin, alpha_min = lsm.secao_aurea(intervalo, dir, sentido, Pmin, f)
        listPmin.append(Pmin) 
        passos = passos + 1
        grad = grad_f(Pmin)
        norm_grad = np.linalg.norm(grad)
        print(f'Passo{passos}, P={Pmin}, alpha={alpha_min}, d={dir}, norm_grad={norm_grad}, grad={grad}')
    end = timer()             
    return listPmin, passos, end-start

def fletcherReeves(P0, f, grad_f, maxiter, tol, line_step):
    start = timer()
    dimens = P0.size
    Pmin = P0.copy()
    grad = grad_f(Pmin)
    passos = 0
    
    listPmin = []
    listPmin.append(Pmin)
    
    #primeiro passo
    grad_k = grad_f(P0)
    norm_grad = np.linalg.norm(grad_k)
    
    dir = -grad_k
    dir_k = dir.copy()
    
    if(norm_grad < tol):
        end = timer()
        return listPmin, passos, end-start
    
    P_pss_cte, intervalo, sentido, alpha = lsm.passo_cte(dir, Pmin, f)
    Pmin = lsm.secao_aurea(intervalo, dir, sentido, Pmin, f)[0]
    listPmin.append(Pmin) 
    passos = passos + 1
    grad_knext = grad_f(Pmin)
    norm_grad = np.linalg.norm(grad_knext)
    print(f'Passso {passos}: dir={dir}, grad_k={grad_k}, grad_k+1={grad_knext}, P={Pmin} ')
         
    while(norm_grad > tol):
        if(passos == maxiter):
            print('Nao convergiu !')
            end = timer()
            return listPmin, passos, end-start
        #beta = (np.transpose(grad_knext)).dot(grad_knext)/((np.transpose(grad_k)).dot(grad_k))
        beta = (np.linalg.norm(grad_knext)/np.linalg.norm(grad_k))**2
        dir = -grad_knext + beta*dir_k
        P_pss_cte, intervalo, sentido, alpha = lsm.passo_cte(dir/(np.linalg.norm(dir)), Pmin, f, step=line_step)
        Pmin = lsm.secao_aurea(intervalo, dir/(np.linalg.norm(dir)), sentido, Pmin, f)[0]
        listPmin.append(Pmin) 
        
        grad_k = grad_knext.copy()
        dir_k = dir.copy()
        grad_knext = grad_f(Pmin)
        norm_grad = np.linalg.norm(grad_knext)
        passos = passos + 1
        print(f'Passso {passos}: dir={dir}, grad_k={grad_k}, grad_k+1={grad_knext}, P={Pmin}')
    end = timer()
    
    return listPmin, passos, end-start