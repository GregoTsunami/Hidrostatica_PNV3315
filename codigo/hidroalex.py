import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot, transforms

def pop (lista): # Funcao que ira tirar os nan, que vem junto com os indicadores de paineis
    for i in range (0,len(lista[0])-388):
        if type(lista [0][i][0]) == str:
            lista[0].pop(i)
            i = 0
    n = lista[0][4][3]
    for i in range (0,len(lista[0])):
        if lista [0][i][1] is n:
            lista[0][i].pop()
            lista[0][i].pop()
            lista[0][i].pop()
    return lista

def invert(lista): #Funcao que ira transpor os pontos de um lado para o outro,
    for i in range (0,len(lista)): # negativando o valor da meia-boca e trocando
        if (len(lista[i])) == 1: # a posicao dos pontos 2 e 4.
            lista[i][0] = (lista[i][0]) * (-1)
            i += 1
        lista[i][2] = (lista[i][2])*(-1)
    trans = []
    for i in range (0,len(lista)-2):
        if lista[i][0] == 2 and lista[i+2][0] == 4:
            lista [i][0] = 4
            lista [i+2][0] = 2
            trans = lista[i].copy()
            lista[i] = lista[i+2]
            lista [i+2] = trans
            i += 2
    return lista

def orienta(lista, mult): # muda o sistema de referencia para a linha d'agua projetada = 8,5344m = 28ft
    for i in range (0,len(lista)):
        if len(lista[i]) > 1:
            lista[i][3] = lista[i][3]-(1.2192 * mult)
        
    return lista

def area (lista):
    area = []
    for i in range (0,len(lista), 5):
        if lista[i][3] <= 0 and lista[i+1][3] <= 0 and lista[i+2][3] <= 0 and lista[i+3][3] <= 0:
            v1 = [(lista[i+1][1] + (-lista[i][1])), (lista[i+1][2] + (-lista[i][2])), (lista[i+1][3] + (-lista[i][3]))]
            v2 = [(lista[i+3][1] + (-lista[i][1])), (lista[i+3][2] + (-lista[i][2])), (lista[i+3][3] + (-lista[i][3]))]
            v3 = [(lista[i+3][1] + (-lista[i+2][1])), (lista[i+3][2] + (-lista[i+2][2])), (lista[i+3][3] + (-lista[i+2][3]))]
            v4 = [(lista[i+1][1] + (-lista[i+2][1])), (lista[i+1][2] + (-lista[i+2][2])), (lista[i+1][3] + (-lista[i+2][3]))]
            v12 = [((v1[1]*v2[2]) + (-(v1[2]*v2[1]))), ((v1[2]*v2[0]) + (-(v1[0]*v2[2]))), ((v1[0]*v2[1]) + (-(v1[1]*v2[0])))]
            v34 = [((v3[1]*v4[2]) + (-(v3[2]*v4[1]))), ((v3[2]*v4[0]) + (-(v3[0]*v4[2]))), ((v3[0]*v4[1]) + (-(v3[1]*v4[0])))]
            v1234 = [(v12[0] + v34[0])/2, (v12[1] + v34[1])/2, (v12[2] + v34[2])/2]
            area.append(v1234)
            if i+4 != len(lista): 
                area.append(lista[i+4])
        #\left(u_2v_3-u_3v_2,\:u_3v_1-u_1v_3,\:u_1v_2-u_2v_1\right)
    #print(area)
    return area 

def graficos_1 (lista):
    x = []
    y = []
    z = []
    plt.figure(facecolor=".5", figsize = (60,7))
    plt.style.use('ggplot')
    for i in range (0, len(lista)):
        if len(lista[i]) !=1:
            x.append(lista[i][1])
            y.append(lista[i][2])
    plt.plot(x, y, color = 'blue')
    plt.title("Balizas x Meia-bocas")
    plt.xlabel('Balizas')
    plt.ylabel('Meia-bocas')
    plt.xlim(max(x), min(x)) 
    plt.show()
    
    y = []
    plt.figure(facecolor=".5", figsize = (15,20))
    plt.style.use('ggplot')
    for i in range (0, len(lista)):
        if len(lista[i]) != 1:
            if lista[i][1] > 10 :
                y.append(-(lista[i][2]))
                z.append(lista[i][3])
            else:
                y.append(lista[i][2])
                z.append(lista[i][3])
    plt.plot(y,z, color = 'blue')
    plt.title("Meia-bocas x Linhas D'agua")
    plt.xlabel('Meia-bocas')
    plt.ylabel("Linhas D'agua")
    plt.show()
    
    plt.figure(facecolor=".5", figsize = (60,7))
    plt.style.use('ggplot')
    plt.plot(x, z, color = 'blue')
    plt.title("Balizas x Linhas D'agua")
    plt.xlabel('Balizas')
    plt.ylabel("Linhas D'agua")
    plt.xlim(max(x), min(x)) 
    plt.show()
    
    return 

def Sw(lista):
    y = []
    x = []
    temp = []
    for i in range(3,10):
        temp = lista.copy()
        Sw = 0
        temp = (orienta(temp, i)) #Mult [3, 10]
        area_temp = area(temp)
        y.append(i * 1.2192)
        for j in range (0,len(area_temp)):
            if len(area_temp[j]) != 1:
                Sw += np.sqrt(area_temp[j][0] * 2 + area_temp[j][1] * 2 + area_temp[j][2] ** 2)
        x.append(Sw)
    plt.figure(facecolor=".5")
    plt.style.use('ggplot')
    plt.plot(x, y, color = 'blue')
    plt.title("Superficie Molhada")
    plt.xlabel('Superficie molhada')
    plt.ylabel("calado")
    plt.show()
    
    return x

def Wl(lista):#A é o A(x,y,z) e n o número de paineis
    x = []
    y = []
    soma = 0
    temp = []
    for i in range(3,10):
        temp = lista.copy()
        temp = (orienta(temp, i)) #Mult [3, 10]
        area_temp = area(temp)
        y.append(i * 1.2192)
        for j in range(len(area_temp)):
            if len(area_temp[j])>1:
                soma += area_temp[j][2]#*(-1)
        x.append(soma)
    plt.figure(facecolor=".5")
    plt.style.use('ggplot')
    plt.plot(x, y, color = 'blue')
    plt.title("Linha d'agua")
    plt.xlabel("Linha d'agua")
    plt.ylabel("Calado")
    plt.show()
    
    return x

def c(p1,p2,p3,p4, i):
    c = (p1[i] + p2[i] + p3[i] + p4[i])/4
    return c

def lcf(lista):#P é lista dos painéis p(x,y,z) e A lista de A(x,y,z)
    y = []
    x = []
    somatoriaA = div = 0
    temp = []
    for i in range(3,10):
        temp = lista.copy()
        temp = (orienta(temp, i)) #Mult [3, 10]
        area_temp = area(temp)
        y.append(i * 1.2192)
        for j in range(0, len(area_temp), 5):
            if len(area_temp[j]) != 1:
                cx = c(temp[j],temp[j+1],temp[j+2],temp[j+3], 0)
                for k in range(0, len(area_temp)):
                    if len(area_temp[k]) != 1:
                        somatoriaA += area_temp[k][2] * (-1)
                        div += area_temp[k][2] * cx *(-1)
        result = div/somatoriaA
        x.append(result)
    plt.figure(facecolor=".5")
    plt.style.use('ggplot')
    plt.plot(x, y, color = 'blue')
    plt.title("Longitudinal Center of Floatation")
    plt.xlabel("LCF")
    plt.ylabel("Calado")
    plt.show()
    
    return x

def tcf(lista):
    x = []
    y = []
    temp = []
    soma1 = soma2 = 0
    for i in range(3,10):
        temp = lista.copy()
        temp = (orienta(temp, i)) #Mult [3, 10]
        area_temp = area(temp)
        y.append(i * 1.2192)
        for j in range(0, len(area_temp), 5):
            if len(area_temp[j]) != 1:
                cy = c(temp[j],temp[j+1],temp[j+2],temp[j+3], 1)
                for k in range(0, len(area_temp)):
                    if len(area_temp[k]) != 1:
                        soma1 += (-area_temp[k][2]) *cy
                        soma2 += (-area_temp[k][2])
        x.append(soma1/soma2)
    plt.figure(facecolor=".5")
    plt.style.use('ggplot')
    plt.plot(x, y, color = 'blue')
    plt.title("tranversal Center of Floatation")
    plt.xlabel("TCF")
    plt.ylabel("Calado")
    plt.show()
    
    return x


def Il(lista, tcf):
    x = []
    y = []
    temp = []
    soma1 = 0
    for i in range(3,10):
        temp = lista.copy()
        temp = (orienta(temp, i)) #Mult [3, 10]
        area_temp = area(temp)
        y.append(i * 1.2192)
        for j in range(0, len(area_temp)-1, 5):
            if len(area_temp[j]) != 1:
                cx = c(temp[j],temp[j+1],temp[j+2],temp[j+3], 0)
                for k in range(0, len(area_temp)-1):
                    if len(area_temp[k]) != 1:
                        soma1 += ((-area_temp[k][2]) ((cx - tcf[0])*2))# + Il
        x.append(soma1)
    plt.figure(facecolor=".5")
    plt.style.use('ggplot')
    plt.plot(x, y, color = 'blue')
    plt.title("Inertia longitudinal")
    plt.xlabel("Il")
    plt.ylabel("Calado")
    plt.show()
    
    return x

def It(lista, lcf):
    x = []
    y = []
    temp = []
    soma1 = 0
    for i in range(3,10):
        temp = lista.copy()
        temp = (orienta(temp, i)) #Mult [3, 10]
        area_temp = area(temp)
        y.append(i * 1.2192)
        for j in range(0, len(area_temp)-1, 5):
            if len(area_temp[j]) != 1:
                cy = c(temp[j],temp[j+1],temp[j+2],temp[j+3], 1)
                for k in range(0, len(area_temp)-1):
                    if len(area_temp[k]) != 1:
                        soma1 += ((-area_temp[k][2]) ((cy - lcf[0])*2))#+ It 
        x.append(soma1)
    plt.figure(facecolor=".5")
    plt.style.use('ggplot')
    plt.plot(x, y, color = 'blue')
    plt.title("Inertia transversal")
    plt.xlabel("It")
    plt.ylabel("Calado")
    plt.show()
    
    return x

def lcb(lista):
    x = []
    y = []
    temp = []
    soma1 = 0
    for i in range(3,10):
        temp = lista.copy()
        temp = (orienta(temp, i)) #Mult [3, 10]
        area_temp = area(temp)
        y.append(i * 1.2192)
        for j in range(0, len(area_temp)-1, 5):
            if len(area_temp[j]) != 1:
                cx = c(temp[j],temp[j+1],temp[j+2],temp[j+3], 0)
                for k in range(0, len(area_temp)-1):
                    if len(area_temp[k]) != 1:
                        soma1 += ((area_temp[k][0]* cx )*(cx/2))#/volume
        x.append(soma1)
    plt.figure(facecolor=".5")
    plt.style.use('ggplot')
    plt.plot(x, y, color = 'blue')
    plt.title("Longitudinal Center of Bouyancy")
    plt.xlabel("LCB")
    plt.ylabel("Calado")
    plt.show()
    
    return x

def tcb(lista):
    x = []
    y = []
    temp = []
    soma1 = 0
    for i in range(3,10):
        temp = lista.copy()
        temp = (orienta(temp, i)) #Mult [3, 10]
        area_temp = area(temp)
        y.append(i * 1.2192)
        for j in range(0, len(area_temp)-1, 5):
            if len(area_temp[j]) != 1:
                cy = c(temp[j],temp[j+1],temp[j+2],temp[j+3], 1)
                for k in range(0, len(area_temp)-1):
                    if len(area_temp[k]) != 1:
                        soma1 += ((area_temp[k][0]* cy ) * (cy/2))#/volume
        x.append(soma1)
    plt.figure(facecolor=".5")
    plt.style.use('ggplot')
    plt.plot(x, y, color = 'blue')
    plt.title("Transversal Center of Bouyancy")
    plt.xlabel("TCB")
    plt.ylabel("Calado")
    plt.show()
    
    return x

def kb(lista):
    x = []
    y = []
    temp = []
    soma1 = 0
    for i in range(3,10):
        temp = lista.copy()
        temp = (orienta(temp, i)) #Mult [3, 10]
        area_temp = area(temp)
        y.append(i * 1.2192)
        for j in range(0, len(area_temp)-1, 5):
            if len(area_temp[j]) != 1:
                cz = c(temp[j],temp[j+1],temp[j+2],temp[j+3], 2)
                for k in range(0, len(area_temp)-1):
                    if len(area_temp[k]) != 1:
                        soma1 += ((area_temp[k][0]* cz )*(cz/2))#/volume
        x.append(soma1)
    plt.figure(facecolor=".5")
    plt.style.use('ggplot')
    plt.plot(x, y, color = 'blue')
    plt.title("KB")
    plt.xlabel("KB")
    plt.ylabel("Calado")
    plt.show()
    
    return x

def pp_hidro(Sw, Wl, lcf, tcf, Il, It, lcb, tcb, kb):
    y = []
    for i in range(3,10):
        y.append(i * 1.2192)
    
    plt.figure(facecolor=".5", figsize = (20,7))
    plt.style.use('ggplot')
    plt.plot(Sw, y, color = 'blue')
    plt.plot(Wl, y, color = 'cyan')
    plt.plot(lcf, y, color = 'yellow')
    plt.plot(tcf, y, color = 'green')
    plt.plot(Il, y, color = 'black')
    plt.plot(It, y, color = 'grey')
    plt.plot(lcb, y, color = 'red')
    plt.plot(tcb, y, color = 'navy')
    plt.plot(kb, y, color = 'purple')
    plt.title("Propriedades Hidrostaticas")
    plt.ylabel("Calado")
    plt.show()
    
    return

hidro = []
transit = pd.read_excel(r'C:\Users\WINDOWS\Downloads\Dunkerque.xlsx') #Transforma a tabela em uma variavel
pontos_a = []
pontos_b = []
pontos_a.append(transit.values.tolist()) #transforma o DataFrame em uma Lista
pontos_a = (pop(pontos_a))[0] #Chama a funcao pop
pontos_b.append(transit.values.tolist())
pontos_b = (pop(pontos_b))[0]
pontos_b = (invert(pontos_b)) # Inverte os pontos da tabela para completar o outro lado do navio
paineis = pontos_a + [[0]] + pontos_b

graficos_1(pontos_a)

Sw = Sw(paineis)
Wl = Wl(paineis)
lcf = lcf(paineis)
tcf = tcf(paineis)
Il = Il (paineis, tcf)
It = It (paineis, lcf)
lcb = lcb(paineis)
tcb = tcb(paineis)
kb = kb(paineis)

pp_hidro(Sw, Wl, lcf, tcf, Il, It, lcb, tcb, kb)