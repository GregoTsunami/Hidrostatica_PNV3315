# TEMOS ALGO


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
    
    # Gera os gráficos solicitados
    gerar_graficos(x, wl, ct)
    
    # Cálculo para todos os calados interpolados
    all_drafts = z_spl
    all_results = []
    for d in all_drafts:
        res = calcular_propriedades(paineis, x_spl, z_spl, pontos, d)
        all_results.append(res)
    
    # Gera os novos gráficos hidrostáticos (um por um)
    plot_hidrostatic_properties(all_drafts, all_results)

def gerar_graficos(x, wl, ct):
    """Gera os gráficos de baliza X meia-boca e calado X meia-boca"""
    plt.figure(figsize=(14, 6))
    
    # Gráfico 1: Baliza X Meia-boca para todas as linhas d'água
    plt.subplot(1, 2, 1)
    for z_idx in range(len(wl)):
        y_values = ct.iloc[1:, z_idx+1].values.astype(float).flatten()
        plt.plot(x, y_values, label=f'WL = {wl[z_idx]}m')
    
    plt.title('Curvas de Baliza X Meia-boca')
    plt.xlabel('Baliza (m)')
    plt.ylabel('Meia-boca (m)')
    plt.grid(True)
    plt.legend()
    
    # Gráfico 2: Calado X Meia-boca para todas as balizas
    plt.subplot(1, 2, 2)
    for x_idx in range(len(x)):
        y_values = ct.iloc[x_idx+1, 1:].values.astype(float).flatten()
        plt.plot(y_values, wl, label=f'Baliza = {x[x_idx]}m')
    
    plt.title('Curvas de Calado X Meia-boca')
    plt.xlabel('Meia-boca (m)')
    plt.ylabel('Calado (m)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

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
    """Gera a malha completa de pontos do casco com interpolação segura"""
    pontos = []
    
    # Primeiro, criamos um mapeamento de z para y_values
    z_to_y = {}
    for z_idx, z in enumerate(wl):
        y_values = ct.iloc[1:, z_idx+1].values.astype(float).flatten()
        z_to_y[z] = y_values
    
    # Agora para cada z interpolado, encontramos os dois z mais próximos para interpolar
    for z in z_spl:
        # Encontra os dois wls mais próximos
        if z in z_to_y:  # Se for um valor original, usa diretamente
            y_spl = cubic_spline(x, z_to_y[z], x_spl)
        else:
            # Encontra os wls adjacentes
            lower_wl = None
            upper_wl = None
            for wz in sorted(z_to_y.keys()):
                if wz < z:
                    lower_wl = wz
                elif wz > z:
                    upper_wl = wz
                    break
            
            if lower_wl is not None and upper_wl is not None:
                # Interpola entre os dois wls
                alpha = (z - lower_wl) / (upper_wl - lower_wl)
                y_lower = cubic_spline(x, z_to_y[lower_wl], x_spl)
                y_upper = cubic_spline(x, z_to_y[upper_wl], x_spl)
                y_spl = y_lower * (1 - alpha) + y_upper * alpha
            elif lower_wl is not None:  # Extrapola usando o último
                y_spl = cubic_spline(x, z_to_y[lower_wl], x_spl)
            elif upper_wl is not None:  # Extrapola usando o primeiro
                y_spl = cubic_spline(x, z_to_y[upper_wl], x_spl)
            else:
                raise ValueError("Não foi possível interpolar as linhas d'água")
        
        # Adiciona os pontos para esta linha d'água
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
                pontos[idx + n_x + 1], # Superior direito
                pontos[idx + n_x]      # Superior esquerdo
            ]
            paineis.append(painel)
    return paineis

def calcular_propriedades(paineis, x_spl, z_spl, pontos, draft):
    """Calcula todas as propriedades hidrostáticas com proteção contra divisão por zero"""
    # Dimensões principais
    Lpp = x_spl[-1] - x_spl[0]
    B = max(p[1] for p in pontos)*2
    T = draft
    
    # Inicialização com valores padrão
    resultados = {
        'Volume moldado': 0.0,
        'Área flutuação': 0.0,
        'LCF': 0.0,
        'TCF': 0.0,
        'KB': 0.0,
        'It': 0.0,
        'IL': 0.0,
        'Cb': 0.0,
        'KMt': 0.0,
        'KMl': 0.0
    }
    
    # Cálculo do centro de flutuação
    lcf, tcf = 0.0, 0.0
    area_total = 0.0
    for p in pontos:
        if abs(p[2] - draft) < 1e-3:
            area_elemento = p[1]*2*(x_spl[1]-x_spl[0])
            resultados['Área flutuação'] += area_elemento
            lcf += p[0] * area_elemento
            tcf += p[1] * area_elemento/2  # Correção no cálculo do TCF
            area_total += area_elemento
    
    if area_total > 0:
        resultados['LCF'] = lcf / area_total
        resultados['TCF'] = tcf / area_total
    
    # Cálculo de volume e momentos
    volume_total = 0.0
    momento_z_total = 0.0
    
    for painel in paineis:
        p1, p2, p3, p4 = painel
        dz = abs(p4[2] - p1[2])
        y_avg = (p1[1] + p2[1] + p3[1] + p4[1])/4
        dx = x_spl[1] - x_spl[0]
        
        # Apenas painéis abaixo do calado
        if p1[2] <= draft and p4[2] <= draft:
            volume = y_avg * dx * dz
            volume_total += volume
            
            # Momentos
            z_centro = (p1[2] + p4[2])/2
            momento_z_total += volume * z_centro
            
            # Inércias
            resultados['It'] += (dx * dz**3)/12 + dx*dz*(y_avg - resultados['TCF'])**2
            resultados['IL'] += (dz * dx**3)/12 + dz*dx*(p1[0] - resultados['LCF'])**2
    
    resultados['Volume moldado'] = volume_total
    
    if volume_total > 0:
        resultados['KB'] = momento_z_total / volume_total
        resultados['KMt'] = resultados['KB'] + resultados['It']/volume_total
        resultados['KMl'] = resultados['KB'] + resultados['IL']/volume_total
    
    # Cálculo do Cb com proteção contra divisão por zero
    if T > 0 and Lpp > 0 and B > 0:
        resultados['Cb'] = volume_total / (Lpp * B * T)
    else:
        resultados['Cb'] = 0.0
    
    return resultados

def plot_hidrostatic_properties(drafts, results):
    """Gera gráficos individuais das propriedades hidrostáticas com tratamento de erros"""
    propriedades = list(results[0].keys())
    
    for prop in propriedades:
        plt.figure(figsize=(10, 6))
        valores = [res[prop] for res in results]
        
        # Filtra valores válidos
        drafts_validos = []
        valores_validos = []
        for d, v in zip(drafts, valores):
            if not (np.isnan(v) or np.isinf(v)):
                drafts_validos.append(d)
                valores_validos.append(v)
        
        if len(drafts_validos) == 0:
            print(f"Não há dados válidos para {prop}")
            plt.close()
            continue
        
        plt.plot(drafts_validos, valores_validos, 'b-', marker='o', markersize=6, linewidth=2)
        plt.title(f'Variação de {prop} com o Calado', fontsize=14)
        plt.xlabel('Calado (m)', fontsize=12)
        plt.ylabel(prop, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Ajuste seguro dos limites do eixo Y
        y_min, y_max = min(valores_validos), max(valores_validos)
        y_range = y_max - y_min
        
        if y_range > 0:
            plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        elif y_min == 0:
            plt.ylim(-0.1, 0.1)  # Caso todos os valores sejam zero
        else:
            plt.ylim(y_min*0.9, y_max*1.1)  # Caso todos os valores sejam iguais
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()