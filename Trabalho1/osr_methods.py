import numpy as np

def univariante(passo, dimens):
    #indice do vetor = (resto da divisao do passo pela dimensao) - 1
    #primeira posicao do vetor no python tem indice 0
    indice = passo%dimens - 1
    
    if (indice == -1) :
        #indice = -1 indica que se trata da ultima posicao do array
        #no pyton esse indice eh o tamanho do vetor - 1
        indice = dimens - 1
        
    #define a direcao canonica a ser utilizada
    ek = np.zeros(dimens)
    ek[indice] = 1
    
    return ek
    
def powell(P, P0, direcoes, passos, ciclos, dimens):
    #indice do vetor = (resto da divisao do passo pela dimensao) - 1
    #primeira posicao do vetor no python tem indice 0
    indice = passos%(dimens + 1) - 1
    
    if (indice == -1):
        #indice = -1 indica que se trata da ultima posicao do array
        #no pyton esse indice eh o tamanho do vetor - 1
        #direcao n + 1 do ciclo = Patual - P0
        dir = P - P0
        direcoes[dimens - 1] = dir        
    elif (indice == 0):
        #indice = 0 significa que vamos usar a primeira direcao do conjunto
        #representa o inicio de um novo ciclo
        ciclos = ciclos + 1

        if (ciclos%(dimens+2) == 0):
            #se ciclo for multipl de dimens + 2, conjunto de direcoes = canonicas
            direcoes = np.eye(dimens, dtype=float)
        P0 = P.copy()
        dir = direcoes[indice].copy()
        
    else:
        dir = direcoes[indice].copy()
        direcoes[indice-1] = dir
  
    return dir, direcoes, P0, ciclos            

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