import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


#import os
#import pyperclip

#filename = os.listdir()[3]

def dibujar_grafico_ipc(fechas, serie):
    """
    Dibuja un gráfico estilizado y limpio para la variación logarítmica del IPC.
    Esconde todos los detalles estéticos dentro de la función.
    """
    # 1. Configuración del tamaño y resolución del lienzo
    plt.figure(figsize=(10, 5), dpi=100)

    # 2. Estilo de la línea y un sutil sombreado inferior para darle profundidad
    plt.plot(fechas, serie, color='#1e3d59', linewidth=1.8, label='Variación logarítmica')
    plt.fill_between(fechas, serie, color='#1e3d59', alpha=0.08)

    # 3. Títulos y etiquetas con colores suaves (evitando el negro puro) y tipografía clara
    plt.title('Variación Mensual del IPC', fontsize=14, fontweight='bold', pad=15, color='#17252a')
    plt.xlabel('Fecha', fontsize=11, labelpad=10, color='#2b7a78')
    plt.ylabel('Variacion Mensual', fontsize=11, labelpad=10, color='#2b7a78')

    # 4. Limpieza de los bordes del gráfico (Spines) para un look moderno
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#def2f1')
    ax.spines['bottom'].set_color('#def2f1')

    # 5. Cuadrícula de fondo muy tenue para guiar la vista sin ensuciar
    ax.grid(True, linestyle='--', alpha=0.5, color='#def2f1')

    # 6. Rotación y alineación de las fechas para que no se superpongan
    plt.xticks(rotation=30, ha='right', color='#17252a')
    plt.yticks(color='#17252a')

    # 7. Ajuste automático de márgenes para evitar cortes en las etiquetas
    plt.tight_layout()
    plt.show()


ipc_data = pd.read_excel("PEM_VAR_IPC_2023.xlsx")
fechas = ipc_data.iloc[1, 3:]
ipc_data.columns = ipc_data.iloc[1, :]
ipc_data = ipc_data.iloc[20, 2:]
ipc_data = pd.to_numeric(ipc_data, errors='coerce')
ipc_laged = ipc_data.shift(1)
ipc_laged = np.array(ipc_laged);
ipc_data = np.array(ipc_data)
ipc_data = ipc_data[1:];
ipc_laged = ipc_laged[1:]

serie = np.log(ipc_data / ipc_laged)*100

dibujar_grafico_ipc(fechas, serie)

#B
#Graficar correlaciones parciales
# Autocorrelacion Rezago K
# COV(yt, yt-k) / Var(yt)
mu = np.mean(serie)
varianza = np.mean(np.square(serie - mu))

autocov = {}
for i in range(1, len(serie)):
    autocov[i] = np.mean((serie[i:] - mu) * (serie[:-i] - mu))

plt.plot(autocov.keys(), autocov.values())
plt.show()

j = 1
autocorr = {}

for element in autocov.values():
    autocorr[j] = element/ varianza
    j += 1
#Autocorrelograma
plt.plot(autocorr.keys(), autocorr.values())
plt.show()

sorted(autocorr.items(), key=lambda x: x[1], reverse=True)
#Autocorrelaciones parciales
np_autocorr = np.array([i for i in autocorr.values()])


rho = np.insert(np_autocorr, 0, 1)
pacf = np.zeros(len(rho))
pacf[0] = 1
pacf[1] = np_autocorr[0]


for k in range(2, len(rho)):
    R = np.array([[rho[abs(i-j)] for j in range(k)] for i in range(k)])
    r = rho[1:k+1]
    phi = np.linalg.solve(R, r)
    pacf[k] = phi[-1]

#C)
def residuos_calc(y, c, phi, theta):
    n = len(y)
    p = len(phi)
    q = len(theta)
    eps = np.zeros(n)

    for t in range(n):
        ar = 0
        for i in range(1, p+1):
            if t - i >= 0:
                ar += phi[i-1] * y[t - i]
        ma = 0
        for l in range(1, q+1):
            if t - l >= 0:
                ma += theta[l-1] * eps[t - l]
        eps[t] = y[t] - c - ar - ma
    return eps

def log_likelihood(params, y, p, q):
    c = params[0]
    phi = params[1:1+p]
    theta = params[1+p:p+q+1]
    sigma2 = params[-1]

    eps = residuos_calc(y, c, phi, theta)
    n = len(y)

    return -0.5 * n * np.log(2 * np.pi * sigma2) - 0.5 * np.sum(eps**2) / sigma2

def neg_log_likelihood(params, y, p, q):
    return -log_likelihood(params, y, p, q)


p, q = 1, 1
init_params = np.zeros(2 + p + q)
init_params[0] = mu
init_params[1:1+p] = 0.5
init_params[1+p:p+q+1] = 0
init_params[-1] = varianza

result = minimize(neg_log_likelihood, init_params, args=(serie, p, q), method='L-BFGS-B')
#Resultado es un ARMA(1,1)

modelos = {}
for p in range(1,7):
    for q in range(1,7):
        init_params = np.zeros(2+p+q)
        init_params[0] = mu
        init_params[1:1+p] = 0.5
        init_params[1+p:p+q+1] = 0
        init_params[-1] = varianza

        modelos[(p,q)] = minimize(neg_log_likelihood, init_params, args=(serie, p, q), method='L-BFGS-B')

n = len(serie)
criterios = {}

for (p, q), res in modelos.items():
    k = 2 + p + q
    ll = -res.fun

    criterios[(p, q)] = {
        'AIC': -2 * ll / n + 2 * k / n,
        'BIC': -2 * ll / n + np.log(n) * k / n,
        'HQC': -2 * ll / n + 2 * np.log(np.log(n)) * k / n
    }

