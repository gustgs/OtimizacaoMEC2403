import numpy as np
import math

def passo_cte(dir_unit, P1, f, step=0.01, mod_dir=1000):
        #recebe o vetor direcao unitario, P1, funcao
        #pode receber tambem o modulo do vetor direcao e o step
        for alpha in np.arange(0, mod_dir, step):
                Pmin = P1 + alpha*dir_unit
                alpha_min = alpha
                if (f(P1 + alpha*dir_unit) < f(P1 + (alpha+step)*dir_unit)):
                        break
        
        #retorna o P(x1,x2) minimo e o intervalo de busca(alpha minimo e o alpha seguinte)                 
        return Pmin, np.array([alpha_min, alpha_min+step])
    
def bissecao(intervalo, dir_unit, P1, f, tol=0.00001):
    
    alpha_upper = intervalo[1]
    alpha_lower = intervalo[0]
    alpha_med = (alpha_lower + alpha_upper)/2
    Pmin = P1 + alpha_med*dir_unit    
    
    if (alpha_upper - alpha_lower) <= tol:
        return Pmin, alpha_med
    else:
        f1 = f(P1 + (alpha_med - tol)*dir_unit)
        f2 = f(P1 + (alpha_med + tol)*dir_unit)
              
        if (f1 > f2):
            alpha_lower = alpha_med
            intervalo[0] = alpha_lower
            return  bissecao(intervalo, dir_unit, P1, f, tol)
        else:
            alpha_upper = alpha_med
            intervalo[1] = alpha_upper            
            return  bissecao(intervalo, dir_unit, P1, f, tol)

def secao_aurea(intervalo, dir_unit, P1, f, tol=0.00001):
    alpha_upper = intervalo[1]
    alpha_lower = intervalo[0]
    beta = alpha_upper - alpha_lower
    Ra = (math.sqrt(5)-1)/2
    
    alpha_e = alpha_lower + (1-Ra)*beta
    alpha_d = alpha_lower + Ra*beta 
    f1 = f(P1 + alpha_e*dir_unit)
    f2 = f(P1 + alpha_d*dir_unit)
    
    while (beta > tol):
        if (f1 > f2):
            alpha_lower = alpha_e
            f1 = f2
            beta = alpha_upper - alpha_lower
            alpha_e = alpha_lower + (1-Ra)*beta
            alpha_d = alpha_lower + Ra*beta 
            f2 = f(P1 + alpha_d*dir_unit)
        else:
            alpha_upper = alpha_d
            f2 = f1
            beta = alpha_upper - alpha_lower
            alpha_e = alpha_lower + (1-Ra)*beta
            alpha_d = alpha_lower + Ra*beta 
            f1 = f(P1 + alpha_e*dir_unit)
            
    alpha_med = (alpha_lower + alpha_upper)/2
    Pmin = P1 + alpha_med*dir_unit
    
    return Pmin, alpha_med

def secao_aurea_recursiva(intervalo, dir_unit, P1, f, tol=0.00001):
    #nao é a melhor implementacao pois chama funcao f 2vezes em cada iteração
    alpha_upper = intervalo[1]
    alpha_lower = intervalo[0]
    alpha_med = (alpha_lower + alpha_upper)/2
    Pmin = P1 + alpha_med*dir_unit
    
    beta = alpha_upper - alpha_lower
    Ra = (math.sqrt(5)-1)/2
    
    if beta <= tol :
        return Pmin, alpha_med
    else:
        alpha_e = alpha_lower + (1-Ra)*beta
        alpha_d = alpha_lower + Ra*beta
        f1 = f(P1 + alpha_e*dir_unit)
        f2 = f(P1 + alpha_d*dir_unit)
        if (f1 > f2):
            alpha_lower = alpha_e
            intervalo[0] = alpha_lower            
        else:
            alpha_upper = alpha_d
            intervalo[1] = alpha_upper
    return secao_aurea(intervalo, dir_unit, P1, f, tol)