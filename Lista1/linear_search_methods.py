import numpy as np
import math

def passo_cte(direcao_unitaria, P1, f, step = 0.01, modulo_direcao = 1000):
    #linear search pelo método do passo constante
        
    #iteracao em alpha variando de 0 até o modulo informado da direção(opcional) ou até 1000 (default)
    for alpha in np.arange(0, modulo_direcao, step):
        # atualiza os valores de Pmin e alpha min com os valores de cada step
        Pmin = P1 + alpha*direcao_unitaria
        alpha_min = alpha
                
        #verifica se a funcao fica ascendente no proximo step
        #caso positivo, a busca unidimensional termina e os valores de Pmin e alpha min já estão guardados
        if (f(P1 + alpha*direcao_unitaria) < f(P1 + (alpha+step)*direcao_unitaria)):
            break
        
        #retorna o Pmin = [x1,x2]  e o intervalo de busca = [alpha min, alpha min + step]                 
        return Pmin, np.array([alpha_min, alpha_min+step])
    
def bissecao(intervalo, direcao_unitaria, P1, f, tol=0.00001):
    #linear search pelo método da bisseção
    #função estruturada de forma recursiva
    
    #atribui os limites superior e inferior da busca à variáveis internas do método
    alpha_upper = intervalo[1]
    alpha_lower = intervalo[0]
    
    #atualiza o valor de alpha min (igual ao último alpha med) e  Pmin em cada chamada do método
    alpha_med = (alpha_lower + alpha_upper)/2
    alpha_min = alpha_med
    Pmin = P1 + alpha_min*direcao_unitaria    
    
    #condicao de convergencia
    if (alpha_upper - alpha_lower) <= tol:
        #caso positivo, a busca termina e os valores de Pmin e alpha min já estão guardados
         return Pmin, alpha_min
    else:
        #caso negativo, verifica se o lado à esquerda ou à direita do alpha med deve ser descartado
        # e chama, recursivamente, o metodo da bisseção com o intervalo restante
        f1 = f(P1 + (alpha_med - tol)*direcao_unitaria)
        f2 = f(P1 + (alpha_med + tol)*direcao_unitaria)
        
        #    verifica se o valor de f à esquerda é maior do que o valor de f à direita do alpha med
        if (f1 > f2):
            #caso positivo, define novo intervalo de busca como sendo do alpha med atual até o alpha upper atual
            alpha_lower = alpha_med
            intervalo[0] = alpha_lower
            
            #chama novamente o método com o novo intervalo de busca
            return  bissecao(intervalo, direcao_unitaria, P1, f, tol)
        else:
            #caso negativo, define novo intervalo de busca como sendo do alpha lower atual até o alpha med atual
            alpha_upper = alpha_med
            intervalo[1] = alpha_upper 
            #chama novamente o método com o novo intervalo de busca           
            return  bissecao(intervalo, direcao_unitaria, P1, f, tol)

def secao_aurea(intervalo, dir_unit, P1, f, tol=0.00001):
    #linear search pelo método da seção áurea
    
    #atribui os limites superior e inferior da busca à variáveis internas do método
    alpha_upper = intervalo[1]
    alpha_lower = intervalo[0]
    beta = alpha_upper - alpha_lower
    
    #razão áurea
    Ra = (math.sqrt(5)-1)/2
    
    # define os pontos de analise de f com base na razao aurea
    alpha_e = alpha_lower + (1-Ra)*beta
    alpha_d = alpha_lower + Ra*beta 
    
    #primeira iteração avalia f nos 2 pontos selecionados pela razao aurea
    f1 = f(P1 + alpha_e*dir_unit)
    f2 = f(P1 + alpha_d*dir_unit)
    
    #loop enquanto a convergência não for obtida
    while (beta > tol):
        if (f1 > f2):
            #caso positivo, define novo intervalo variando de alpha_e até alpha_upper
            # e aproveita os valores anteriores de alpha_d e f2 como novos alpha_e e f1
            alpha_lower = alpha_e
            f1 = f2
            alpha_e = alpha_d            
            
            #calcula novo alpha_d e f2=f(alpha_d)
            beta = alpha_upper - alpha_lower
            #alpha_e = alpha_lower + (1-Ra)*beta
            alpha_d = alpha_lower + Ra*beta 
            f2 = f(P1 + alpha_d*dir_unit)
        else:
            #caso negativo, define novo intervalo variando de alpha_lower até alpha_d
            # e aproveita os valores anteriores de alpha_e e f1 como novos alpha_d e f2
            alpha_upper = alpha_d
            f2 = f1
            alpha_d = alpha_e
            
            #calcula novo alpha_e e f1=f(alpha_e)
            beta = alpha_upper - alpha_lower
            alpha_e = alpha_lower + (1-Ra)*beta
            #alpha_d = alpha_lower + Ra*beta 
            f1 = f(P1 + alpha_e*dir_unit)
            
    # calcula Pmin e alpha min após convergência
    alpha_med = (alpha_lower + alpha_upper)/2
    alpha_min = alpha_med
    Pmin = P1 + alpha_min*dir_unit
    
    return Pmin, alpha_min

#metodo nao utilizado por nao ser otimizado nas chamadas de f
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