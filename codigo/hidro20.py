import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import pyplot, transforms

#lê a planilha do excel
ct = pd.read_excel('Dunkerque.xlsx', sheet_name='BxWL')

#BALIZAS#
#####################
vetX = pd.read_excel('Dunkerque.xlsx', sheet_name='BxWL', usecols="A", skiprows=0) #pega a coluna das balizas
x = vetX.values.flatten() #pega as balizas e joga os valores num vetor

#WL#
#####################
#pega a linha das WL
vetZ = pd.read_excel('Dunkerque.xlsx', sheet_name='BxWL', header=None)
wl = vetZ.iloc[0].drop(vetZ.columns[0]).values.flatten()
######################
pontos = [1]
x_spl = []
z_spl = []
#####################################################
#interpola os pontos de Z de acordo com o que o úsuario digita
def spline_z(splines):
    zi = []
    if splines == 0:
        zi = wl
    else:
        for i in range (0,(8*splines)+1):
            zi.append(i*((wl[2]-wl[1])/splines))
    return zi

#####################################################
#interpola os pontos de X de acordo com o que o úsuario digita
def spline_x(splines):
    xi = []
    if splines == 0:
        xi = x
    else:
        for i in range (0, (20*splines)+1):
            xi.append(i*((x[2]-x[1])/splines))
    return xi

######################################################
def grafico_X_Y():
    for k in range(0,len(wl)):
        y = ct[k].values #une os dados num vetor)
        y_spl = splcubic(x,y, x_spl)
        plot_wlxbal(x_spl,y_spl, k)
def plot_wlxbal(x,y, k):
    plt.scatter(x,y)
    plt.plot(x,y)
    if wl[k] == wl[-1]:
        plt.ylim(-1, 17)
        plt.gca().set_aspect(20/8, adjustable='box')
        plt.grid()
        plt.xlabel('Baliza')
        plt.ylabel('Meia-boca')
        plt.show()
########################################
def grafico_Z_Y():
    for k in range(0,len(x)):
        y = ct.iloc[k].drop(ct.columns[0]).values.flatten()
        y_spl = splcubic(wl,y, z_spl)
        plot_calxwl(z_spl,y_spl, k)
def plot_calxwl(z,y, k):
    if x[k] > (x[-1])/2:
        for i in range (0,len(y)):
            y[i] = y[i]*-1
        plt.scatter(y,z)
        plt.plot(y,z)
    elif x[k] == (x[-1])/2:
        plt.scatter(y,z, color='#edca7f')
        plt.plot(y,z, color='#edca7f')
        for i in range (0,len(y)):
            y[i] = y[i]*-1
        plt.scatter(y,z, color='#edca7f')
        plt.plot(y,z, color='#edca7f')
    else:
        plt.scatter(y,z)
        plt.plot(y,z)
    if x[k] == x[-1]:
        plt.xlim(-16, 16)
        #plt.gca().set_aspect('auto')
        plt.xlabel('Meia-boca')
        plt.ylabel('Calado')
        plt.grid()
        plt.show()
###########################################
#################################################
##################################################
#converte o sistema de coordenadas
def conversao(pontos):
    for i in range (0, len(pontos)):
        pontos[i][0] = pontos [i][0] - (x_spl[-1]/2)
        #pontos[i][2] = pontos [i][2] - (z_spl[-1])
###################################################
def paineis(pontos):
    paineis = []
    guarda2 = guarda1 = guarda3 = guarda4 = [0,0,0]
    for i in range(0,3):
        for j in range (0, len(pontos)):
            if guarda2 [i] > pontos[j][i]:
                guarda2 = pontos[j]
    for k in range (0, len(pontos)):
        if pontos[k][0] == guarda2 [0] and pontos[k][2] == (guarda2[2]+z_spl[1]):
            guarda1 = pontos[k]
    for k in range (0, len(pontos)):
        if pontos[k][0] == (guarda1 [0]+x_spl[1]) and pontos[k][2] == (guarda1[2]):
            guarda4 = pontos[k]
    for k in range (0, len(pontos)):
        if pontos[k][0] == (guarda4 [0]) and pontos[k][2] == (guarda2[2]):
            guarda3 = pontos[k]
    paineis = [guarda1, guarda2, guarda3, guarda4]

    return paineis
###################################################
def organiza(pontos):
    bckp_pontos = pontos.copy()
    pontos.clear()
    n = len(x_spl)*len(wl)
    while len(pontos) != len(bckp_pontos):
        guarda = bckp_pontos[n]
        for i in range (0, len(bckp_pontos)):
            if guarda[2] < bckp_pontos[i][2]:
                pontos.append(guarda)
                n += 1
                print('inclui - z')
            else:
                pontos.append(bckp_pontos[i])
                print('inclui - x')
    """
        for i in range (0, len(x_spl)):
            for j in range (0, len(z_spl)):
                if guarda[2] <= bckp_pontos [j][2] or bckp_pontos[j][2] == z_spl[len(z_spl)]:
                    pontos.append(bckp_pontos[j])
                    print('inclui - z')
            if bckp_pontos[i][0] <= bckp_pontos [i+1][0] or bckp_pontos[i][0] == x_spl[len(x_spl)]:
                pontos.append(bckp_pontos[i])
                print('inclui - x')
     """
    print('organizado')
    return pontos
#####################################################
# Calcula o valor de Y de um determinado x:
def spline_y(x, y, x_spl, n, a, b, c, d):
    vetS = []
    for k in range(n-1):
        eq = f'{a[k]}{b[k]:+}*(x-{x[k]}){c[k]:+}*(x-{x[k]})**2{d[k]:+}*(x-{x[k]})**3'
        vetS.append(eq)


    interval = None
    for k in range(n-1):
        if x[k] <= x_spl <= x[k+1]:
            interval = k
            break

    if interval is None:
        return None  # Valor de x fora do intervalo de dados

    # Calculando o valor de Y para x_spl
    sol = f'{a[interval]}{b[interval]:+}*(x_spl-{x[interval]}){c[interval]:+}*(x_spl-{x[interval]})**2{d[interval]:+}*(x_spl-{x[interval]})**3'
    y_val = eval(sol)

    return y_val
#####################################################
#caso eu queria ler um valor especifico de x ou z:
def splcubic(x,y,x_spl):
    y_spl = []
    #lê o tamanho do vetor X
    n = len(x)

    #pega o dicionário para ler a posição e o valor dela na matriz
    #ak = yk
    a= {k: v for k, v in enumerate(y)}
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

        #S[k] = a[k] + (b[k]*(x-x[k])) + (c[k]*(x-x[k])**2) + (d[k]*(x-x[k])**3)
        """
        for key, value in S.items():
            def p(x):
                q = eval(value['eq'])
                return q
        t = np.linspace(*value['domain'], 100)
        plt.plot(t, p(t), label=f"$S_{key}(x)$")
        """

    for i in range(0, len(x_spl)):
        if spline_y(x, y, x_spl[i], n, a, b, c, d) < 0: #Ignora valores de Y menores que 0
            y_spl.append(0.0)
        else:
            y_spl.append(spline_y(x, y, x_spl[i], n, a, b, c, d))

    return y_spl

##############################################
def areaPainel(pain):
    print('seu painel ta ai: ', pain)
    v1 = pain[0]
    v2 = pain[1]
    v3 = pain[2]
    v4 = pain[3]
    #faz a multiplicação vetorial
    mv1 = np.cross(v1, v2)
    mv2 = np.cross(v3, v4)
    #calculo do vetor área
    area = (mv1 + mv2)/2
    #tira o módulo da área
    modArea = np.linalg.norm(area)
    print('Area do painel: ', area, 'Módulo da Area: ', modArea)
    return area, modArea

#####################################################
splines = int(input("digite o valor de pontos entre balizas desejado: "))
print("\nx:", x)
x_spl = np.array(spline_x(splines))
z_spl = np.array(spline_z(splines))
n = 0
m = 0
#pega todas as meia-bocas de todas Balizas interpoladas
for k in np.arange(0, z_spl[-1]+(z_spl[1]/2), z_spl[1]):
    y = ct[n].values #une os dados num vetor
    print("\ny:", y)
    print("\nPara linha d'agua: ", k)
    y_spl = splcubic(x,y, x_spl)
    print('\ny_spl', y_spl)
    for i in range(0, len(x_spl)):
        if [x_spl[i],y_spl[i],z_spl[m]] != pontos[i]:
            pontos.append([x_spl[i],y_spl[i],z_spl[m]])
    n += 1
    m += 1
    if n == len(wl):
        n = 0
    if m ==len(z_spl):
        m = 0
print('\nx_spl', x_spl)

pontos.pop(0)
grafico_X_Y()
grafico_Z_Y()
conversao(pontos)
pain = paineis(pontos)
# print('paineis:', pain)
areaPainel(pain)
