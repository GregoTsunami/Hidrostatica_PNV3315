import pandas as pd
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import pyplot, transforms

def splinecubica(x, y):
    i = max(len(x), len(y))
    
    # h e dy com intervalos (i-1)
    h = np.zeros(i - 1)
    dy = np.zeros(i - 1)
    for k in range(i - 1):
        h[k] = x[k+1] - x[k]  # distancia entre 2 pontos de x
        dy[k] = y[k+1] - y[k]  # distancia entre 2 pontos de y
    
    # matriz a tem tamanho i2 x i
    a = np.zeros((i - 2, i))
    for l in range(i - 2):
        for c in range(i):
            if l == c:
                a[l, c] = h[l]
            elif c == l+1:
                a[l, c] = 2 * (h[l] + h[c])
            elif c == l+2:
                a[l, c] = h[c-1]
            else:
                a[l, c] = 0
    
    # tirando a primeira e segunda coluna de A, temos uma matriz quadrada
    A = np.zeros((i - 2, i - 2))
    for l in range(i - 2):
        for c in range(i - 2):
            A[l, c] = a[l, c+1]
    
    # matriz B
    B = np.zeros(i - 2)
    for l in range(i - 2):
        B[l] = 6 * (dy[l+1] / h[l+1] - dy[l] / h[l])
    
    # g de 2 a i-1
    r = np.linalg.solve(A, B)
    
    # calculo de g
    g = np.zeros(i)
    g[0] = 0
    g[-1] = 0
    for k in range(1, i-1):
        g[k] = r[k-1]
    
    # ak, bk, ck. dk em ordem das colunas
    S = np.zeros((i-1, 4))
    for k in range(i-1):
        S[k, 0] = (g[k+1] - g[k]) / (6 * h[k])  # ak
        S[k, 1] = g[k+1] / 2  # bk
        S[k, 2] = dy[k] / h[k] + (2 * h[k] * g[k+1] + g[k] * h[k]) / 6  # ck
        S[k, 3] = y[k+1]  # dk
    
    
    plt.plot(S)
    plt.show()
    return S

def limites(nS, nWL, B):
    """
    Saves the initial (bow) and final (stern) positions of the vessel for each waterline in a vector of size [nWL, 2].
    
    Args:
        nS (int): Number of stations.
        nWL (int): Number of waterlines.
        B (numpy.ndarray): Array of beam values.
        
    Returns:
        numpy.ndarray: Array of initial and final positions for each waterline.
    """
    lim = np.zeros((nWL, 2), dtype=int)
    
    for i in range(nWL):
        k = (nS - nS % 2) // 2  # Start from the midship
        
        # Find the first point to be considered in the calculations
        while B[k, i] > 0:
            lim[i, 0] = k - 1
            k -= 1
            if k == 0:
                break
        
        k = (nS - nS % 2) // 2  # Start from the midship
        
        # Find the last point to be considered
        while B[k, i] > 0:
            lim[i, 1] = k + 1
            k += 1
            if k == nS:
                break
    
    return lim

def propriedades(S, WL, B, lim):
    dS = S[1] - S[0]  # distância relativa entre cada baliza
    dWL = WL[1] - WL[0]  # distância relativa entre cada linha d'água
    nS = len(S)  # pois S é um vetor coluna
    nWL = len(WL)  # pois S é um vetor coluna

    # vamos cálcular a área para a chapa na quilha AK através da integral por Simpson
    NS = nS
    if NS % 2 == 0:
        NS += 1
        S = np.append(S, 0)
        B = np.vstack((B, np.zeros((1, B.shape[1]))))

    # calculando os coeficientes de simpson
    K = np.ones(NS)
    K[1:-1:2] = 4  # pares recebem 4
    K[2:-1:2] = 2  # ímpares recebem 2 exceto naos extremos

    AK = np.sum(B[:, 0] * K) * (2 * dS / 3)
    # A posição da chapa na quilha CK é (0,0,0)

    # cálculo do volume submerso para cada calado
    V = np.zeros(nWL)

    # cálculo da área molhada
    SW = np.zeros(nWL)
    SW[0] = AK

    # cálculo da área de linha d'água
    AWL = np.zeros(nWL)
    AWL[0] = AK

    # cálculo da posição longitudinal para o centro de flutuação
    E = S - (S[-1] - S[0]) / 2
    SDE = np.sum(B[:, 0] * E)
    SD = np.sum(B[:, 0])
    LCF = np.zeros(nWL)
    LCF[0] = dS * SDE / SD

    # cálculo da posição transversal para o centro de flutuação
    TCF = np.zeros(nWL)

    # cálculo da posição longitudinal para o centro de carena
    LCB = np.zeros(nWL)

    # cálculo da posição transversal para o centro de carena
    TCB = np.zeros(nWL)

    # cálculo da altura do centro de carena
    KB = np.zeros(nWL)

    # cálculo da posição longitudinal do BM
    BML = np.zeros(nWL)

    # cálculo da posição transversal do BM
    BMT = np.zeros(nWL)

    for t in range(1, nWL):
        # vamos guardar em uma matriz tridimensional A os vetores área das chapas
        A = np.zeros((nS - 1, t, 3))
        for l in range(nS - 1):
            for c in range(t):
                A[l, c, 0] = dWL * (B[l, c + 1] + B[l, c] - B[l + 1, c + 1] - B[l + 1, c])
                A[l, c, 1] = 2 * dWL * dS
                A[l, c, 2] = dS * (B[l, c] + B[l + 1, c] - B[l, c + 1] - B[l + 1, c + 1])
        A *= 0.5  # A área A é a media entre a área calculada por v1xv2 e por v3xv4 para aumentarmos a precisão

        # vamos guardar em uma matriz tridimensional C os vetores posição das chapas em relação a origem do sistema
        C = np.zeros((nS - 1, t, 3))
        for l in range(nS - 1):
            for c in range(t):
                C[l, c, 0] = S[l] + S[l + 1]
                C[l, c, 1] = 0.5 * (B[l, c] + B[l + 1, c] + B[l, c + 1] + B[l + 1, c + 1])
                C[l, c, 2] = WL[c] + WL[c + 1]
        C *= 0.5  # a posição de cada chapa é a media aritmetica da posição de cada ponto
        C[:, :, 0] -= (S[-1] - S[0]) / 2  # como a origem do sistema se encontrava na proa mudamos a posição para a meia nau

        # zerando as áreas e posições das chapas lc que não pertencem a embarcação
        for c in range(t):
            # zerando as áreas e posições iniciais
            A[:lim[c, 0] - 1, c, :] = 0
            C[:lim[c, 0] - 1, c, :] = 0
            # zerando as áreas e posições finais
            A[lim[c, 1]:, c, :] = 0
            C[lim[c, 1]:, c, :] = 0

        # a partir de agora podemos calcular as propriedades desejadas

        # cálculo do volume submerso para cada calado
        V[t] = np.sum(A[:, :t, 0] * C[:, :t, 0] + A[:, :t, 1] * C[:, :t, 1] + A[:, :t, 2] * C[:, :t, 2]) * (2 / 3)

        # cálculo da área molhada
        SW[t] = AK + np.sum(2 * np.sqrt(np.sum(A[:, :t, :]**2, axis=2)))

        # cálculo da área de linha d'água
        AWL[t] = AK - 2 * np.sum(A[:, :t, 2])

        # cálculo da posição longitudinal para o centro de flutuação
        LCF[t] = (LCF[0] * AK - 2 * np.sum(A[:, :t, 2] * C[:, :t, 0])) / AWL[t]

        # cálculo da posição transversal para o centro de flutuação
        TCF[t] = 0  # embarcação simetrica

        # cálculo da posição longitudinal para o centro de carena
        LCB[t] = np.sum(C[:, :t, 0] * A[:, :t, 0] * C[:, :t, 0]) / V[t]

        # cálculo da posição transversal para o centro de carena
        TCB[t] = 0  # embarcação simetrica

        # cálculo da altura do centro de carena
        KB[t] = -np.sum(C[:, :t, 2] * A[:, :t, 2] * C[:, :t, 2]) / V[t]

        # cálculo da posição longitudinal do BM
        IL = np.sum((dS * dWL**3) / 12 - A[:, :t, 2] * C[:, :t, 0]**2)
        BML[t] = IL / V[t]

        # cálculo da posição transversal do BM
        IT = np.sum((dWL * dS**3) / 12 - A[:, :t, 2] * (C[:, :t, 1] - LCF[t])**2)
        BMT[t] = IT / V[t]

    # vamos guardar as propriedades numa matriz que retorna seus valores
    P = np.vstack((V, SW, AWL, LCF, TCF, LCB, TCB, KB, BML, BMT))

    return P

def principal(S, WL, B):
    e = 0.01  # 1% error tolerance
    erro = 1  # initial error to enter the loop
    dS = S[1] - S[0]
    dWL = WL[1] - WL[0]
    nS = len(S)
    nWL = len(WL)

    lim = limites(nS, nWL, B)
    P1 = propriedades(S, WL, B, lim)

    while erro > e:
        # Refining the number of points by creating new stations for each waterline
        b = np.zeros((2 * nS - 1, nWL))
        for c in range(nWL):
            S3 = splinecubica(S, B[:, c])
            b[-1, c] = B[-1, c]
            for l in range(nS - 1):
                b[2 * l, c] = B[l, c]
                b[2 * l + 1, c] = (S3[l, 0] * (dS / 2)**3 + S3[l, 1] * (dS / 2)**2 +
                                   S3[l, 2] * (dS / 2) + S3[l, 3])

        # Creating new station values
        s = np.zeros(2 * nS - 1)
        s[-1] = S[-1]
        for l in range(nS - 1):
            s[2 * l] = S[l]
            s[2 * l + 1] = S[l] + dS / 2
        S = s
        dS = S[1] - S[0]
        nS = len(S)
        B = b

        # Creating new waterlines for each station
        b = np.zeros((nS, 2 * nWL - 1))
        for l in range(nS):
            S3 = splinecubica(WL, B[l, :])
            b[l, -1] = B[l, -1]
            for c in range(nWL - 1):
                b[l, 2 * c] = B[l, c]
                b[l, 2 * c + 1] = (S3[c, 0] * (dWL / 2)**3 + S3[c, 1] * (dWL / 2)**2 +
                                   S3[c, 2] * (dWL / 2) + S3[c, 3])

        # Creating new waterline values
        wl = np.zeros(2 * nWL - 1)
        wl[-1] = WL[-1]
        for c in range(nWL - 1):
            wl[2 * c] = WL[c]
            wl[2 * c + 1] = WL[c] + dWL / 2
        WL = wl
        dWL = WL[1] - WL[0]
        nWL = len(WL)
        B = b

        lim = limites(nS, nWL, B)
        P2 = propriedades(S, WL, B, lim)

        # Calculating error for this iteration
        erro = np.abs(1 - P2[:, -1] / P1[:, (nWL + 1) // 2 - 1])
        erro = np.max(erro)

        P1 = P2

    plotando(WL, P1)

def plotando(WL, P):
    plt.plot(WL, P[7, :])
    plt.show()
