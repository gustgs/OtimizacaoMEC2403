import numpy as np

def univariante(passo, dimens):
    indice = passo%dimens - 1
    if (indice == -1) :
        indice = dimens - 1
    ek = np.zeros(dimens)
    ek[indice] = 1
    
    return ek
    
def powell(P, P1, direcoes, passos, ciclos, dimens):
    indice = passos%(dimens + 1) - 1
    if (indice == -1):
        dir = P - P1
        direcoes[dimens - 1] = dir        
    elif (indice == 0):
        ciclos = ciclos + 1
        if (ciclos%(dimens+2) == 0):
            direcoes = np.eye(dimens, dtype=float)
        P1 = P.copy()
        dir = direcoes[indice].copy()
    else:
        dir = direcoes[indice].copy()
        direcoes[indice-1] = dir
  
    return dir, direcoes, P1, ciclos            

def newtonRaphson(P, grad_P, hessian_f):
    return -np.linalg.inv(hessian_f(P)).dot(grad_P)

def steepestDescent(grad):
    return -grad

def fletcherReeves(dir_last, grad, grad_last, passo):
    if passo == 1:
        grad_last = grad.copy()
        return -grad, grad_last
    else:
        beta = (np.linalg.norm(grad)/np.linalg.norm(grad_last))**2
        grad_last = grad.copy()
        return -grad + beta*dir_last, grad_last
    
def bfgs(P, P_last, grad, grad_last, S_last, passo, dimens):
    if (passo == 1):
        dir = -S_last.dot(grad)
    else:
        delta_x_k = P - P_last
        delta_g_k = grad - grad_last
        
        #para o numpy, vetor 1-D linha e vetor coluna sao a mesma coisa (nao e necessrio transpor)
        #matrizes
        A = np.outer(delta_x_k, np.transpose(delta_x_k))
        B = S_last.dot(np.outer(delta_g_k, np.transpose(delta_x_k)))
        C = np.outer(delta_x_k, np.transpose(S_last.dot(delta_g_k)))
        
        #Escalares        
        d = np.transpose(delta_x_k).dot(delta_g_k)
        e = np.transpose(delta_g_k).dot(S_last.dot(delta_g_k))
                
        S = S_last + (d + e)*A/(d**2) - (B + C)/d
        dir = -S.dot(grad)
        S_last = S.copy()
    P_last = P
    grad_last = grad
    return dir, P_last, grad_last, S_last