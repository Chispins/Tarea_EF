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


def plot_autocorrelograma(autocov, varianza):
    """
    Dibuja el autocorrelograma a partir de las autocovarianzas.

    Parámetros:
    autocov (dict): Diccionario con rezagos como llaves y autocovarianzas como valores
    varianza (float): Varianza de la serie (para normalizar)
    """
    autocorr = {}
    j = 0

    for element in autocov.values():
        autocorr[j] = element / varianza
        j += 1

    # Autocorrelograma
    plt.plot(autocorr.keys(), autocorr.values())
    plt.xlabel('Rezago')
    plt.ylabel('Autocorrelación')
    plt.title('Autocorrelograma')
    plt.axhline(y=0, color='black', linewidth=0.5)
    plt.show()


def plot_autocorrelograma_parcial(valores_pacf):
    """
    Función exclusiva para graficar la autocorrelación parcial (PACF).
    Recibe un array o lista con los valores ya calculados.
    """
    plt.figure(figsize=(10, 5))

    # Grafica las líneas verticales típicas de la PACF
    plt.stem(range(len(valores_pacf)), valores_pacf)

    # Línea de referencia en cero
    plt.axhline(0, color='red', linestyle='--', linewidth=1)

    # Detalles estéticos
    plt.title(f"Autocorrelograma Parcial (PACF) - Hasta Rezago {len(valores_pacf) - 1}")
    plt.xlabel("Rezago")
    plt.ylabel("Autocorrelación Parcial")
    plt.grid(True, alpha=0.3)
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

plot_autocorrelograma(autocov, varianza)
j = 1
autocorr = {}

for element in autocov.values():
    autocorr[j] = element/ varianza
    j += 1
#Autocorrelograma
plt.plot(autocorr.keys(), autocorr.values())
plt.show()

#sorted(autocorr.items(), key=lambda x: x[1], reverse=True)
#Autocorrelaciones parciales
np_autocorr = np.array([i for i in autocorr.values()])


rho = np.insert(np_autocorr, 0, 1)
max_lags = 40
pacf = np.zeros(max_lags+1)
pacf[0] = 1
pacf[1] = np_autocorr[0]

plot_autocorrelograma_parcial(pacf)

for k in range(2, max_lags + 1):
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

print(f"{'Modelo':<12} {'AIC':<10} {'BIC':<10} {'HQC':<10}")
print("-" * 42)
for q in range(1, 7):
    if (1, q) in criterios:
        print(f"ARMA(1,{q})   {criterios[(1,q)]['AIC']:<10.4f} {criterios[(1,q)]['BIC']:<10.4f} {criterios[(1,q)]['HQC']:<10.4f}")

# Modelo escogido es (1,5)
#D
res_15 = modelos[(1,5)]
c_est   = res_15.x[0]
phi_est = res_15.x[1]
theta_est = res_15.x[2:7]
sigma2_est = res_15.x[7]

m = max(1, 5+1)
F = np.zeros((m, m))
F[0, 0] = phi_est
for i in range(1, m):
    F[i-1, i] = 1

G = np.zeros((m, m))
G[0, 0] = 1
for i in range(1, m):
    G[i, 0] = theta_est[i-1]

eigenvalues, V = np.linalg.eig(F)
D = np.diag(eigenvalues)



#IRF 20 periodos
H = 20
irf = np.zeros(H)
alpha = G.copy()
for h in range(H):
    irf[h] = alpha[0,0]
    alpha = F @ alpha

#g
phi_arr = np.array([phi_est]) if np.isscalar(phi_est) else phi_est
theta_arr = theta_est.flatten() if theta_est.ndim > 1 else theta_est

eps_last = residuos_calc(serie, c_est, phi_arr, theta_arr)
n = len(serie) #DRY

alpha_t = np.zeros((m, 1))
alpha_t[0, 0] = serie[-1]
for i in range(1, m):
    alpha_t[i, 0] = theta_arr[i-1] * eps_last[-1]
H = 8
proyecciones = np.zeros(H)
alpha = alpha_t.copy()
for h in range(H):
    alpha = c_est * G + F @ alpha
    proyecciones[h] = alpha[0, 0]

mayo  = proyecciones[0]
junio = proyecciones[1]
dic   = proyecciones[7]
