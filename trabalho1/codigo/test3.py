# teste 1
# tentativa de correção das props hidrostaticas 3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union
import numpy.typing as npt

def main():
    # Carrega dados do Excel
    data = pd.ExcelFile('super_tanker.xlsx')
    ct = pd.read_excel(data, sheet_name='BxWLxDunkerque', header=None)
    
    # Balizas originais (x) e linhas d'água (wl)
    x = ct.iloc[1:, 0].values.astype(float).flatten()
    wl = ct.iloc[0, 1:].values.astype(float).flatten()
    
    # Input do usuário
    splines = int(input("Pontos entre balizas (0 para originais): "))
    draft = float(input("Calado de operação (m): "))
    
    # Gera pontos interpolados
    x_spl = spline_x(x, splines)
    z_spl = spline_z(wl, splines)
    
    # Processa geometria do casco
    pontos = gerar_pontos_casco(x, wl, ct, x_spl, z_spl)
    paineis = criar_paineis(x_spl, z_spl, pontos)
    
    # Cálculos hidrostáticos
    resultados = calcular_propriedades(paineis, x_spl, z_spl, pontos, draft)
    
    # Exibe resultados
    print("\n=== RESULTADOS HIDROSTÁTICOS ===")
    for key, value in resultados.items():
        print(f"{key}: {value}")

def spline_x(x: npt.NDArray[np.float64], splines: int) -> npt.NDArray[np.float64]:
    """Interpola balizas com espaçamento uniforme entre pontos"""
    if splines == 0:
        return x.copy()
    
    new_x = []
    for i in range(len(x)-1):
        segment = np.linspace(x[i], x[i+1], splines+2)
        new_x.extend(segment[:-1])
    new_x.append(x[-1])
    return np.array(new_x)

def spline_z(wl: npt.NDArray[np.float64], splines: int) -> npt.NDArray[np.float64]:
    """Interpola linhas d'água com espaçamento uniforme"""
    if splines == 0:
        return wl.copy()
    
    new_z = []
    for i in range(len(wl)-1):
        segment = np.linspace(wl[i], wl[i+1], splines+2)
        new_z.extend(segment[:-1])
    new_z.append(wl[-1])
    return np.array(new_z)

def gerar_pontos_casco(x, wl, ct, x_spl, z_spl):
    """Gera a malha completa de pontos do casco"""
    pontos = []
    for z_idx, z in enumerate(z_spl):
        # Spline para cada linha d'água
        y_values = ct.iloc[1:, z_idx+1].values.astype(float).flatten()
        y_spl = cubic_spline(x, y_values, x_spl)
        
        for x_idx, x_val in enumerate(x_spl):
            pontos.append([x_val, y_spl[x_idx], z])
    
    # Ajuste de coordenadas
    for p in pontos:
        p[0] -= x_spl[-1]/2  # Centraliza longitudinalmente
    return pontos

def cubic_spline(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], 
                x_new: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Implementação robusta de spline cúbica natural"""
    n = len(x)
    h = np.diff(x)
    a = y.copy()
    
    # Sistema tridiagonal
    A = np.zeros((n, n))
    B = np.zeros(n)
    
    # Condições de contorno naturais
    A[0, 0] = 1
    A[-1, -1] = 1
    
    for i in range(1, n-1):
        A[i, i-1] = h[i-1]
        A[i, i] = 2*(h[i-1] + h[i])
        A[i, i+1] = h[i]
        B[i] = 3*((a[i+1] - a[i])/h[i] - (a[i] - a[i-1])/h[i-1])
    
    c = np.linalg.solve(A, B)
    
    # Coeficientes
    b = np.zeros(n-1)
    d = np.zeros(n-1)
    for i in range(n-1):
        b[i] = (a[i+1] - a[i])/h[i] - h[i]*(2*c[i] + c[i+1])/3
        d[i] = (c[i+1] - c[i])/(3*h[i])
    
    # Avaliação
    y_new = np.zeros_like(x_new)
    for i, xi in enumerate(x_new):
        idx = min(np.searchsorted(x, xi) - 1, n-2)
        dx = xi - x[idx]
        y_new[i] = a[idx] + b[idx]*dx + c[idx]*dx**2 + d[idx]*dx**3
    return y_new

def criar_paineis(x_spl, z_spl, pontos):
    """Cria painéis retangulares entre pontos adjacentes"""
    n_x = len(x_spl)
    n_z = len(z_spl)
    paineis = []
    
    for z in range(n_z-1):
        for x in range(n_x-1):
            idx = z*n_x + x
            painel = [
                pontos[idx],          # Inferior esquerdo
                pontos[idx + 1],      # Inferior direito
                pontos[idx + n_x + 1],# Superior direito
                pontos[idx + n_x]     # Superior esquerdo
            ]
            paineis.append(painel)
    return paineis

def calcular_propriedades(paineis, x_spl, z_spl, pontos, draft):
    """Calcula todas as propriedades hidrostáticas"""
    # Dimensões principais
    Lpp = x_spl[-1] - x_spl[0]
    B = max(p[1] for p in pontos)*2
    T = draft
    
    # Inicialização
    resultados = {
        'Volume moldado': 0,
        'Área flutuação': 0,
        'LCF': 0,
        'TCF': 0,
        'KB': 0,
        'It': 0,
        'IL': 0,
        'Cb': 0,
        'KMt': 0,
        'KMl': 0
    }
    
    # Cálculo do centro de flutuação
    lcf, tcf = 0, 0
    area_total = 0
    for p in pontos:
        if abs(p[2] - draft) < 1e-3:
            resultados['Área flutuação'] += p[1]*2*(x_spl[1]-x_spl[0])
            lcf += p[0] * p[1]*2*(x_spl[1]-x_spl[0])
            tcf += p[1] * p[1]*(x_spl[1]-x_spl[0])
            area_total += p[1]*2*(x_spl[1]-x_spl[0])
    
    if area_total > 0:
        resultados['LCF'] = lcf / area_total
        resultados['TCF'] = tcf / area_total
    
    # Cálculo de volume e momentos
    for painel in paineis:
        p1, p2, p3, p4 = painel
        dz = abs(p4[2] - p1[2])
        y_avg = (p1[1] + p2[1] + p3[1] + p4[1])/4
        dx = x_spl[1] - x_spl[0]
        
        volume = y_avg * dx * dz
        resultados['Volume moldado'] += volume
        
        # Momentos
        z_centro = (p1[2] + p4[2])/2
        resultados['KB'] += volume * z_centro
        
        # Inércias
        resultados['It'] += (dx * dz**3)/12 + dx*dz*(y_avg - resultados['TCF'])**2
        resultados['IL'] += (dz * dx**3)/12 + dz*dx*(p1[0] - resultados['LCF'])**2
    
    # Coeficientes finais
    if resultados['Volume moldado'] > 0:
        resultados['KB'] /= resultados['Volume moldado']
        resultados['Cb'] = resultados['Volume moldado'] / (Lpp * B * T)
        resultados['KMt'] = resultados['KB'] + resultados['It']/resultados['Volume moldado']
        resultados['KMl'] = resultados['KB'] + resultados['IL']/resultados['Volume moldado']
    
    return resultados

if __name__ == "__main__":
    main()