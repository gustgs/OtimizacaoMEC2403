import numpy as np

def passo_cte(direcao, P0, f, step = 0.01):
    #line search pelo metodo do passo constante
    
    #epsilon
    eps=0.00000001
   
    #define o sentido unitario correto de busca
    if (f(P0 - eps*direcao) >= f(P0 + eps*direcao)):
        sentido_busca = direcao.copy()
        flag = 0
    else:
        sentido_busca = -direcao.copy()
        flag = 1
        
    P = P0.copy()
    P_next = P + step*sentido_busca
    alpha = 0
    while (f(P) > f(P_next)):           
        alpha = alpha + step
        P = P0 + alpha*sentido_busca
        P_next = P0 + (alpha+step)*sentido_busca
    
    intervalo = np.array([alpha, alpha + step])
    
    if(flag == 1):
        intervalo = -intervalo
        
    #retorna o intervalo de busca = [alpha min, alpha min + step]                 
    return intervalo
    
def secao_aurea(intervalo, direcao, P0, f, tol=0.00000001):
    #line search pelo metodo da secao aurea
    
    #verifica o sentido da busca
    if(intervalo[1] < 0):
        intervalo = -intervalo
        sentido_busca = -direcao.copy()
        flag = 1
    else:
        sentido_busca = direcao.copy()
        flag = 0
    
    #atribui os limites superior e inferior da busca a variaveis internas do metodo
    alpha_upper = intervalo[1]
    alpha_lower = intervalo[0]
    beta = alpha_upper - alpha_lower
    
    #razao aurea
    Ra = (np.sqrt(5)-1)/2
    
    # define os pontos de analise de f com base na razao aurea
    alpha_e = alpha_lower + (1-Ra)*beta
    alpha_d = alpha_lower + Ra*beta 
    
    #primeira iteracao avalia f nos 2 pontos selecionados pela razao aurea
    f1 = f(P0 + alpha_e*sentido_busca)
    f2 = f(P0 + alpha_d*sentido_busca)
    
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
            f2 = f(P0 + alpha_d*sentido_busca)
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
            f1 = f(P0 + alpha_e*sentido_busca)
            
    # calcula Pmin e alpha min apos convergência
    alpha_med = (alpha_lower + alpha_upper)/2
    alpha_min = alpha_med
    
    if (flag == 1):
        alpha_min = -alpha_min
    
    return alpha_min