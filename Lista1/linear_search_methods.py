import numpy as np
import math

def dirUnit(direcao):
    #calcula o vetor unitario na direcao informada
    return direcao/np.linalg.norm(direcao)

def passo_cte(direcao, P1, f, step = 0.01, modulo_direcao = 1000):
    #linear search pelo metodo do passo constante
    
    #calcula o vetor unitario na direcao de busca solicitada
    direcao_unitaria = dirUnit(direcao)
    
    #epsilon
    eps=0.00000001
    
    #define o sentido unitario correto de busca
    if (f(P1 - eps*direcao_unitaria) > f(P1 + eps*direcao_unitaria)):
        sentido_busca = direcao_unitaria
    else:
        sentido_busca = -direcao_unitaria
        
    #iteracao em alpha variando de 0 ate o modulo informado da direcao(opcional) ou ate 1000 (default)
    for alpha in np.arange(0, modulo_direcao, step):
        # atualiza os valores de Pmin e alpha min com os valores de cada step
        Pmin = P1 + alpha*sentido_busca
        alpha_min = alpha
                
        #verifica se a funcao fica ascendente no proximo step
        #caso positivo, a busca unidimensional termina e os valores de Pmin e alpha min ja estao guardados
        if (f(Pmin) < f(P1 + (alpha+step)*sentido_busca)):
            break
        
        #retorna o Pmin = [x1,x2]  e o intervalo de busca = [alpha min, alpha min + step]                 
    return Pmin, np.array([alpha_min, alpha_min+step]), sentido_busca
    
def bissecao(intervalo, sentido_busca, P1, f, tol=0.00001):
    #linear search pelo metodo da bissecao
    #funcao estruturada de forma recursiva
    
    #epsilon
    eps = 0.00000001
    
    #atribui os limites superior e inferior da busca a variaveis internas do metodo
    alpha_upper = intervalo[1]
    alpha_lower = intervalo[0]
    
    #atualiza o valor de alpha min (igual ao último alpha med) e  Pmin em cada chamada do metodo
    alpha_med = (alpha_lower + alpha_upper)/2
    alpha_min = alpha_med
    Pmin = P1 + alpha_min*sentido_busca    
    
    #condicao de convergencia
    if (alpha_upper - alpha_lower) <= tol:
        #caso positivo, a busca termina e os valores de Pmin e alpha min ja estao guardados
         return Pmin, alpha_min
    else:
        #caso negativo, verifica se o lado a esquerda ou a direita do alpha med deve ser descartado
        # e chama, recursivamente, o metodo da bissecao com o intervalo restante
        f1 = f(P1 + (alpha_med - eps)*sentido_busca)
        f2 = f(P1 + (alpha_med + eps)*sentido_busca)
        
        #    verifica se o valor de f a esquerda e maior do que o valor de f a direita do alpha med
        if (f1 > f2):
            #caso positivo, define novo intervalo de busca como sendo do alpha med atual ate o alpha upper atual
            alpha_lower = alpha_med
            intervalo[0] = alpha_lower
            
            #chama novamente o metodo com o novo intervalo de busca
            return  bissecao(intervalo, sentido_busca, P1, f, tol)
        else:
            #caso negativo, define novo intervalo de busca como sendo do alpha lower atual ate o alpha med atual
            alpha_upper = alpha_med
            intervalo[1] = alpha_upper 
            #chama novamente o metodo com o novo intervalo de busca           
            return  bissecao(intervalo, sentido_busca, P1, f, tol)

def secao_aurea(intervalo, sentido_busca, P1, f, tol=0.00001):
    #linear search pelo metodo da secao aurea
    
    #atribui os limites superior e inferior da busca a variaveis internas do metodo
    alpha_upper = intervalo[1]
    alpha_lower = intervalo[0]
    beta = alpha_upper - alpha_lower
    
    #razao aurea
    Ra = (math.sqrt(5)-1)/2
    
    # define os pontos de analise de f com base na razao aurea
    alpha_e = alpha_lower + (1-Ra)*beta
    alpha_d = alpha_lower + Ra*beta 
    
    #primeira iteracao avalia f nos 2 pontos selecionados pela razao aurea
    f1 = f(P1 + alpha_e*sentido_busca)
    f2 = f(P1 + alpha_d*sentido_busca)
    
    #loop enquanto a convergência nao for obtida
    while (beta > tol):
        if (f1 > f2):
            #caso positivo, define novo intervalo variando de alpha_e ate alpha_upper
            # e aproveita os valores anteriores de alpha_d e f2 como novos alpha_e e f1
            alpha_lower = alpha_e
            f1 = f2
            alpha_e = alpha_d            
            
            #calcula novo alpha_d e f2=f(alpha_d)
            beta = alpha_upper - alpha_lower
            #alpha_e = alpha_lower + (1-Ra)*beta
            alpha_d = alpha_lower + Ra*beta 
            f2 = f(P1 + alpha_d*sentido_busca)
        else:
            #caso negativo, define novo intervalo variando de alpha_lower ate alpha_d
            # e aproveita os valores anteriores de alpha_e e f1 como novos alpha_d e f2
            alpha_upper = alpha_d
            f2 = f1
            alpha_d = alpha_e
            
            #calcula novo alpha_e e f1=f(alpha_e)
            beta = alpha_upper - alpha_lower
            alpha_e = alpha_lower + (1-Ra)*beta
            #alpha_d = alpha_lower + Ra*beta 
            f1 = f(P1 + alpha_e*sentido_busca)
            
    # calcula Pmin e alpha min apos convergência
    alpha_med = (alpha_lower + alpha_upper)/2
    alpha_min = alpha_med
    Pmin = P1 + alpha_min*sentido_busca
    
    return Pmin, alpha_min