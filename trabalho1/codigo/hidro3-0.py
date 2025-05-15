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
        print(f"{key}: {value:.3f}")
    
    # Gera os gráficos originais
    gerar_graficos(x, wl, ct)
    
    # Cálculo para todos os calados interpolados
    all_drafts = z_spl
    all_results = []
    for d in all_drafts:
        res = calcular_propriedades(paineis, x_spl, z_spl, pontos, d)
        all_results.append(res)
    
    # Gera os gráficos hidrostáticos
    plot_hidrostatic_properties(all_drafts, all_results)
    
    plot_all_hidrostatic_curves(all_drafts, all_results)

def gerar_graficos(x, wl, ct):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    for z_idx in range(len(wl)):
        y_values = ct.iloc[1:, z_idx+1].values.astype(float).flatten()
        plt.plot(x, y_values, label=f'WL = {wl[z_idx]}m')
    
    plt.title('Curvas de Baliza X Meia-boca')
    plt.xlabel('Baliza (m)')
    plt.ylabel('Meia-boca (m)')
    plt.grid(True)
    plt.legend()
    
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
    if splines == 0:
        return x.copy()
    
    new_x = []
    for i in range(len(x)-1):
        segment = np.linspace(x[i], x[i+1], splines+2)
        new_x.extend(segment[:-1])
    new_x.append(x[-1])
    return np.array(new_x)

def spline_z(wl: npt.NDArray[np.float64], splines: int) -> npt.NDArray[np.float64]:
    if splines == 0:
        return wl.copy()
    
    new_z = []
    for i in range(len(wl)-1):
        segment = np.linspace(wl[i], wl[i+1], splines+2)
        new_z.extend(segment[:-1])
    new_z.append(wl[-1])
    return np.array(new_z)

def gerar_pontos_casco(x, wl, ct, x_spl, z_spl):
    pontos = []
    y_matrix = ct.iloc[1:, 1:].values.astype(float).T
    
    for z in z_spl:
        z_idx = np.searchsorted(wl, z) - 1
        z_idx = max(0, min(z_idx, len(wl)-2))
        z0, z1 = wl[z_idx], wl[z_idx+1]
        alpha_z = (z - z0)/(z1 - z0) if z1 != z0 else 0
        
        for x_val in x_spl:
            x_idx = np.searchsorted(x, x_val) - 1
            x_idx = max(0, min(x_idx, len(x)-2))
            x0, x1 = x[x_idx], x[x_idx+1]
            alpha_x = (x_val - x0)/(x1 - x0) if x1 != x0 else 0
            
            y = (1 - alpha_x)*(1 - alpha_z)*y_matrix[z_idx, x_idx] + \
                alpha_x*(1 - alpha_z)*y_matrix[z_idx, x_idx+1] + \
                (1 - alpha_x)*alpha_z*y_matrix[z_idx+1, x_idx] + \
                alpha_x*alpha_z*y_matrix[z_idx+1, x_idx+1]
            
            pontos.append([x_val, y, z])
    
    return pontos

def criar_paineis(x_spl, z_spl, pontos):
    n_x = len(x_spl)
    n_z = len(z_spl)
    paineis = []
    
    for z in range(n_z-1):
        for x in range(n_x-1):
            idx = z*n_x + x
            painel = [
                pontos[idx],
                pontos[idx + 1],
                pontos[idx + n_x + 1],
                pontos[idx + n_x]
            ]
            paineis.append(painel)
    return paineis

def calcular_propriedades(paineis, x_spl, z_spl, pontos, draft):
    Lpp = x_spl[-1] - x_spl[0]
    B = max(p[1]*2 for p in pontos)
    T = draft
    
    resultados = {
        'Volume moldado': 0.0,
        'Área flutuação': 0.0,
        'LCF': 0.0,
        'TCF': 0.0,
        'LCB': 0.0,
        'TCB': 0.0,
        'KB': 0.0,
        'It': 0.0,
        'IL': 0.0,
        'BMt': 0.0,
        'BMl': 0.0,
        'KMt': 0.0,
        'KMl': 0.0
    }

    volume_total = 0.0
    mom_x = 0.0
    mom_z = 0.0
    area_total = 0.0
    mom_lcf = 0.0
    It = 0.0
    Il = 0.0

    # Cálculo do Volume e KB
    for i in range(len(x_spl)-1):
        dx = x_spl[i+1] - x_spl[i]
        x_centro = (x_spl[i] + x_spl[i+1])/2
        
        for j in range(len(z_spl)-1):
            if z_spl[j+1] > draft:
                continue
            
            dz = z_spl[j+1] - z_spl[j]
            z_centro = (z_spl[j] + z_spl[j+1])/2
            
            y1 = next(p[1] for p in pontos if p[0] == x_spl[i] and p[2] == z_spl[j])
            y2 = next(p[1] for p in pontos if p[0] == x_spl[i+1] and p[2] == z_spl[j])
            y_avg = (y1 + y2)/2
            
            volume_painel = y_avg * dx * dz * 2  # Volume para ambos os lados
            
            volume_total += volume_painel
            mom_x += volume_painel * x_centro
            mom_z += volume_painel * z_centro

    # Cálculo da Área de Flutuação e LCF
    for i in range(len(x_spl)-1):
        dx = x_spl[i+1] - x_spl[i]
        x_centro = (x_spl[i] + x_spl[i+1])/2
        
        try:
            y1 = next(p[1] for p in pontos if p[0] == x_spl[i] and abs(p[2] - draft) < 1e-3)
            y2 = next(p[1] for p in pontos if p[0] == x_spl[i+1] and abs(p[2] - draft) < 1e-3)
        except StopIteration:
            continue
        
        y_avg = (y1 + y2)/2
        area_segment = y_avg * dx * 2  # Área total (dois lados)
        area_total += area_segment
        mom_lcf += x_centro * area_segment
        
        # Cálculo de It (Momento de Inércia Transversal)
        It += (y_avg**3) * dx * 2 / 3  # Integral de y³ dx * 2 lados

    # Atualiza LCF para uso no cálculo do IL
    if area_total > 0:
        resultados['Área flutuação'] = area_total
        resultados['LCF'] = mom_lcf / area_total

    # Cálculo do IL (Momento de Inércia Longitudinal)
    lcf = resultados['LCF']
    for i in range(len(x_spl)-1):
        dx = x_spl[i+1] - x_spl[i]
        x_centro = (x_spl[i] + x_spl[i+1])/2
        
        try:
            y1 = next(p[1] for p in pontos if p[0] == x_spl[i] and abs(p[2] - draft) < 1e-3)
            y2 = next(p[1] for p in pontos if p[0] == x_spl[i+1] and abs(p[2] - draft) < 1e-3)
        except StopIteration:
            continue
        
        y_avg = (y1 + y2)/2
        area_segment = y_avg * dx * 2  # Área total (dois lados)
        
        # Integral de (x - LCF)² * dA
        Il += (x_centro - lcf)**2 * area_segment

    # Atribuição final dos resultados
    if volume_total > 0:
        resultados['Volume moldado'] = volume_total
        resultados['LCB'] = mom_x / volume_total
        resultados['KB'] = mom_z / volume_total
        
        resultados['It'] = It
        resultados['IL'] = Il
        resultados['BMt'] = It / volume_total
        resultados['BMl'] = Il / volume_total
        resultados['KMt'] = resultados['KB'] + resultados['BMt']
        resultados['KMl'] = resultados['KB'] + resultados['BMl']
    
    return resultados

def plot_hidrostatic_properties(drafts, results):
    propriedades = list(results[0].keys())
    
    for prop in propriedades:
        plt.figure(figsize=(10, 6))
        valores = [res[prop] for res in results]
        
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
        
        y_min, y_max = min(valores_validos), max(valores_validos)
        y_range = y_max - y_min
        if y_range > 0:
            plt.ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
        elif y_min == 0:
            plt.ylim(-0.1, 0.1)
        else:
            plt.ylim(y_min*0.9, y_max*1.1)
        
        plt.tight_layout()
        plt.show()

def plot_all_hidrostatic_curves(drafts: List[float], all_results: List[Dict[str, float]]):
    propriedades = list(all_results[0].keys())
    cores = plt.cm.tab10(np.linspace(0, 1, len(propriedades)))  # Paleta de cores
    
    plt.figure(figsize=(12, 8))
    
    for i, prop in enumerate(propriedades):
        valores = [res[prop] for res in all_results]
        
        # Filtrar valores válidos
        drafts_validos = []
        valores_validos = []
        for d, v in zip(drafts, valores):
            if not (np.isnan(v) or np.isinf(v)) and v != 0:
                drafts_validos.append(d)
                valores_validos.append(v)
        
        if len(drafts_validos) > 0:
            plt.plot(drafts_validos, valores_validos, 
                     label=prop, color=cores[i], linewidth=2)

    plt.title('Curvas Hidrostáticas', fontsize=14)
    plt.xlabel('Calado (m)', fontsize=12)
    plt.ylabel('Valores', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legenda fora do gráfico
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
