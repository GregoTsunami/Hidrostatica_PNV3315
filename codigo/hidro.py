import pandas as pd
import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import pyplot, transforms


#lê a planilha do excel
ct = pd.read_excel('Dunkerque.xlsx', sheet_name='BxWL')
#pega a coluna das balizas
vetX = pd.read_excel('Dunkerque.xlsx', sheet_name='BxWL', usecols="A", skiprows=0) 
#pega as balizas e joga os valores num vetor
x = vetX.values.flatten()

#pega todas as colunas de todas WL
for k in range(0, 9):
    wl = k #linha d'agua utilizada
    y = ct[k].values #une os dados para cada WL num vetor

    def splcubic(x,y):
        #lê o tamanho do vetor X
        n = len(x)
    
        #pega o dicionário para ler a posição e o valor dela na matriz
        #ak = yk
        a = {k: v for k, v in enumerate(y)}
        #hk = xk+1 - xk (distância entre X1 e X2, X2 e X3...)
        h = {k: x[k+1] - x[k] for k in range(n - 1)}
    
    
        #[A]*[C] = [B]
        #A=
        #   | 1         0               0               0   |               | c |               |                     0                     |
        #   | h0    2(hk-1 + hk)          hk            0   |       *       | c |       =       |    3/hk (ak+1 - ak) - 3/hk-1 (ak - ak-1)  |
        #   | 0         h1          2(hk + hk+1)      hk+1  |               | c |               | 3/hk+! (ak+2 - ak+1) - 3/hk+1 (ak+1 - ak) |
        #   | 0         0               0               1   |               | c |               |                     0                     |

        A = [ [1] + [0]*(n-1)]
        for k in range(1, (n-1)):
            linha = [0] * n
            linha[k-1] = h[k-1]
            linha[k] = 2*(h[k-1] + h[k])
            linha[k+1] = h[k]
            A.append(linha)
        A.append([0]*(n-1) + [1])
    
        B = [0]
        for k in range(1, (n-1)):
            linha = ((3/h[k])*(a[k+1] - a[k])) - ((3/h[k-1])*(a[k] - a[k-1]))
            B.append(linha)
        B.append(0)
        
        #Resolvendo para [C]:
        C = np.linalg.solve(A, B)
        c = {k: v for k, v in enumerate(C)} #faz um dicionário para ver a posição e o valor de c0, c1, c2...
        print(c)
        
        
        #Calculando bk = ((1/hk)*(ak+1 - ak)) - ((hk/3)*(2*ck + ck+1)):
        b = {} #cria um dicionário para b
        for k in range(n-1):
            b[k] = ((1/h[k])*(a[k+1] - a[k])) - ((h[k]/3) *(2*c[k] + c[k+1]))
            
        #Calculando dk = (ck+1 - ck)/3*hk
        d = {} #cria um dicionário para d
        for k in range(n-1):
            d[k] = (c[k+1] - c[k])/(3*h[k])
        
        
        #Obtendo as equações para S(x)
        S = {} #cria um dicionário para S
        for k in range(n-1):
            eq = f'{a[k]}{b[k]:+}*(x-{x[k]}){c[k]:+}*(x-{x[k]})**2{d[k]:+}*(x-{x[k]})**3'
            S[k] = {'eq': eq, 'domain':[x[k], x[k+1]]}

            # S[k] = a[k] + (b[k]*(x-x[k])) + (c[k]*(x-x[k])**2) + (d[k]*(x-x[k])**3)
            for key, value in S.items():
                def p(x):
                    q = eval(value['eq'])
                    return q
            t = np.linspace(*value['domain'], 100)
            plt.plot(t, p(t), label=f"$S_{key}(x)$")
        
        return S
    print("Para linha d'agua: ")
    print(wl)
    print(x)
    print(y)
    result = splcubic(x,y)
    plt.scatter(x,y)
    plt.show()