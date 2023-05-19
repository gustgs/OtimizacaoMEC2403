import numpy as np

#Metodo da Penalidade
def p_penal(x, params):
    #leitura dos parametros
    hk_list = params[3]
    cl_list = params[6]
    cl_mont = params[9]
    
    p = 0
    for hk in hk_list:
        p = p + (hk(x))**2
    
    for i in np.arange(len(cl_list)):
        p = p + cl_mont[i]*cl_list[i](x)**2
        
    return  p

def phi_penal(x, params, r):
    #leitura dos parametros
    f = params[0]
    hk_list = params[3]
    cl_list = params[6]
    cl_mont = params[9]
    
    p = 0
    for hk in hk_list:
        p = p + (hk(x))**2
    
    for i in np.arange(len(cl_list)):
        p = p + cl_mont[i]*cl_list[i](x)**2
        
    return f(x) + (1/2)*r*p

def grad_phi_penal(x, params, r):
    #leitura dos parametros
    grad_f = params[1]
    hk_list = params[3]
    grad_hk_list = params[4]
    cl_list = params[6]
    grad_cl_list = params[7]
    cl_mont = params[9]
    
    dimens = x.size
    grad_p = np.zeros(dimens, dtype=float)
    
    for i in np.arange(len(hk_list)):
        grad_p = grad_p + 2*hk_list[i](x)*grad_hk_list[i](x)
    for j in np.arange(len(cl_list)):
        grad_p = grad_p + 2*cl_mont[j]*cl_list[j](x)*grad_cl_list[j](x)
        
    return grad_f(x) + (1/2)*r*grad_p

def hess_phi_penal(x, params, r):
    #leitura dos parametros
    hess_f = params[2]
    hk_list = params[3]
    grad_hk_list = params[4]
    hess_hk_list = params[5]
    cl_list = params[6]
    grad_cl_list = params[7]
    hess_cl_list = params[8]
    cl_mont = params[9]    
    
    dimens = x.size    
    hessian_p = np.zeros((dimens, dimens), dtype=float)
    
    for i in np.arange(dimens):    
        for j in np.arange(dimens):
            for k in np.arange(len(grad_hk_list)):
                hessian_p[i,j] = hessian_p[i,j] + 2*grad_hk_list[k](x)[i]*grad_hk_list[k](x)[j]
            for l in np.arange(len(grad_cl_list)):
                hessian_p[i,j] = hessian_p[i,j] + 2*cl_mont[l]*grad_cl_list[l](x)[j]*grad_cl_list[l](x)[i]
    
    for k in np.arange(len(hk_list)):
        hessian_p = hessian_p + hk_list[k](x)*hess_hk_list[k](x)
    
    for k in np.arange(len(cl_list)):
        hessian_p = hessian_p + cl_mont[k]*cl_list[k](x)*hess_cl_list[k](x)
     
    return hess_f(x) + (1/2)*r*hessian_p

#### Metodo da Barreira 
def phi_bar(x, params, r):
    #leitura dos parametros
    f = params[0]
    cl_list = params[6]
    
    b = 0
    for cl in cl_list:
        b = b - 1/cl(x)
            
    return f(x) + r*b

def b_bar(x, params):
    #leitura dos parametros
    cl_list = params[6]
    
    b = 0
    for cl in cl_list:
        b = b - 1/cl(x)
            
    return b

def grad_phi_bar(x, params, r):
    #leitura dos parametros
    grad_f = params[1]
    cl_list = params[6]
    grad_cl_list = params[7]
    
    dimens = x.size
    grad_b = np.zeros(dimens, dtype=float)
    
    for i in np.arange(len(cl_list)):
        grad_b = grad_b + (cl_list[i](x))**(-2)*grad_cl_list[i](x)
            
    return grad_f(x) + r*grad_b

def hess_phi_bar(x, params, r):
    #leitura dos parametros
    hess_f = params[2]
    cl_list = params[6]
    grad_cl_list = params[7]
    hess_cl_list = params[8]
    
    dimens = x.size    
    hessian_b = np.zeros((dimens, dimens), dtype=float)
    
    for i in np.arange(dimens):    
        for j in np.arange(dimens):
            for k in np.arange(len(cl_list)):
                hessian_b[i,j] = hessian_b[i,j] - 2*((cl_list[k](x))**(-3))*grad_cl_list[k](x)[i]*grad_cl_list[k](x)[j]
    
    for k in np.arange(len(cl_list)):
        hessian_b = hessian_b + ((cl_list[k](x))**(-2))*hess_cl_list[k](x)
    
    return hess_f(x) + r*hessian_b