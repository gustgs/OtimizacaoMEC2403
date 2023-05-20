import numpy as np
import osr_methods as osr
import line_search_methods as lsm
import ocr_methods as ocr
import numdifftools as nd
import matplotlib.pyplot as plt
from timeit import default_timer as timer

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

def newtonRaphson(grad_P, hessian_f):
    return -np.linalg.inv(hessian_f).dot(grad_P)

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

def osr_ctrl(P0, params, r, ctrl_num, metodo_ocr, metodo_osr):
    #controle numerico
    maxiter = ctrl_num[0]
    tol_conv = ctrl_num[1]
    tol_search = ctrl_num[2]
    line_step = ctrl_num[3]
    eps = ctrl_num[4]
    
    metodo = metodo_osr
        
    #inicializacoes auxiliares dos metodos de OSR
    passos = 0
    dimens = P0.size
    Pmin = P0.copy()
    listPmin = []
    listPmin.append(Pmin)
    
    if metodo_ocr == 1:
        grad = ocr.grad_phi_penal(Pmin, params, r)
    elif metodo_ocr == 2:
        grad = ocr.grad_phi_bar(Pmin, params, r)
    
    norm_grad = np.linalg.norm(grad)
    flag_conv = True

    if (metodo == 2):
        direcoes = np.eye(dimens, dtype=float)
        ciclos = 0
        P1 = P0.copy()
    elif (metodo == 5):
        #o metodo recebe a direcao anterior 
        #inicializo a direcao com um vetor de zeros mas que nunca e usado
        #uso apenas para enviar como parametro na primeira iteracao do metodo, o qual atualiza o valor de dir para a iteracao seguinte
        dir = np.zeros((1, dimens))
        grad_last = grad.copy()
    elif(metodo == 6):
        S_last = np.eye(dimens)
        grad_last = grad.copy()
        P_last = P0.copy()
    
    #calculo do Pmin
    start = timer()
    while (norm_grad > tol_conv):
        if (passos == maxiter):
            flag_conv = False
            break
        passos = passos + 1
        if (metodo == 1):
            dir = osr.univariante(passos, dimens)
        elif (metodo == 2):
            dir, direcoes, P1, ciclos = osr.powell(Pmin, P1, direcoes,passos, ciclos, dimens)
        elif (metodo == 3):
            dir = osr.steepestDescent(grad)
        elif (metodo == 4):
            if metodo_ocr == 1:
                hess = ocr.hess_phi_penal(Pmin, params, r)
            elif metodo_ocr == 2:
                hess = ocr.hess_phi_bar(Pmin, params,r)
            dir = osr.newtonRaphson(grad, hess)
        elif (metodo == 5):
            dir, grad_last = osr.fletcherReeves(dir, grad, grad_last, passos)
        elif (metodo == 6):
            dir, P_last, grad_last, S_last = osr.bfgs(Pmin, P_last, grad, grad_last, S_last, passos, dimens)
        
        intervalo = lsm.passo_cte(dir, Pmin, params, r, metodo_ocr, eps, line_step)
        alpha = lsm.secao_aurea(intervalo, dir, Pmin, params, r, metodo_ocr, tol_search)
        Pmin = Pmin + alpha*dir
        listPmin.append(Pmin)
        
        if metodo_ocr == 1:
            grad = ocr.grad_phi_penal(Pmin, params, r)
        elif metodo_ocr == 2:
            grad = ocr.grad_phi_bar(Pmin, params, r)
            
        norm_grad = np.linalg.norm(grad)
        
    end = timer()
    tempoExec = end - start    
        
    return listPmin, passos, flag_conv, tempoExec