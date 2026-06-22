import pandas as pd
import numpy as np
from scipy.stats import f
from matplotlib import pyplot as plt

filename = r"bbdd_Tarea_4.xlsx"
data = pd.read_excel(filename)
# Nota: Las tablas de los resultados que se utilizaron al final del desarollo y en el informe fueron realizados con IA

def estimar_chow_activo(data, columna, umbral=0):
    activo = data.iloc[1:, columna]
    rezagos = data.iloc[:-1, columna]
    activo = np.array(activo)
    rezagos = np.array(rezagos)
    if umbral == 0:
        y_chistosa = rezagos > 0
    else:
        y_chistosa = rezagos > np.quantile(rezagos, umbral)

    estado_1_activo = activo[y_chistosa]
    estado_1_rezagos = rezagos[y_chistosa]
    estado_2_activo = activo[~y_chistosa]
    estado_2_rezagos = rezagos[~y_chistosa]
    X1 = np.array([np.ones(shape=estado_1_rezagos.shape), estado_1_rezagos])
    X2 = np.array([np.ones(shape=estado_2_rezagos.shape), estado_2_rezagos])
    X1 = X1.T
    X2 = X2.T

    Betas_s1 = np.linalg.inv(X1.T @ X1) @ X1.T @ estado_1_activo
    Betas_s2 = np.linalg.inv(X2.T @ X2) @ X2.T @ estado_2_activo

    Xu = np.array([np.ones(shape=rezagos.shape), rezagos]).T
    Bu = np.linalg.inv(Xu.T @ Xu) @ Xu.T @ activo

    yr_pred = Xu @ Bu
    SSRr = np.sum((yr_pred - activo)**2)

    y1_pred = X1 @ Betas_s1
    y2_pred = X2 @ Betas_s2

    SSRu = [
        np.sum((y1_pred - estado_1_activo)**2),
        np.sum((y2_pred - estado_2_activo)**2),
    ]
    SSRu = np.sum(SSRu)

    k = 2
    n = rezagos.size
    n1 = estado_1_rezagos.size
    n2 = estado_2_rezagos.size
    F_statistic = ((SSRr - SSRu) / k) / (SSRu / (n1 + n2 - 2 * k))
    p = f.sf(F_statistic, k, n1 + n2 - 2*k)

    return F_statistic, p, Betas_s1, Betas_s2

def calculo_percentiles():
    umbral_list = [i / 10 for i in range(1, 10)]
    results = {}
    for i in [1, 2]:
        for element in umbral_list:
            F_statistic, p, b1, b2 = estimar_chow_activo(data, columna=i, umbral=element)
            results["activo_" + str(i) + "_quantile_" + str(element)] = {
                "F": F_statistic,
                "p": p,
                "betas_reg1": b1,
                "betas_reg2": b2
            }
    return results

def table_pregunta_1():
    """
    (i) Fije gamma = 0, estime cada régimen por mínimos cuadrados y compare si φ1 y φ2 difieren.
    """
    print("\n=== RESPUESTA PREGUNTA (i): ESTIMACIÓN CON γ = 0 ===")
    for activo in [1, 2]:
        F, p, b1, b2 = estimar_chow_activo(data, columna=activo, umbral=0)

        print(f"Activo {activo}:")
        print(f"  Régimen ganancias (q_t > 0):  α = {b1[0]:.4f},  φ = {b1[1]:.4f}")
        print(f"  Régimen pérdidas  (q_t <= 0): α = {b2[0]:.4f},  φ = {b2[1]:.4f}")
        print(f"  Diferencia descriptiva (φ1 - φ2) = {b1[1] - b2[1]:.4f}")
        print()
def table_pregunta_2():
    """
    (ii) Contraste formal de igualdad sobre los dos regímenes iniciales (γ = 0). Reporte p-value.
    """
    print("\n=== RESPUESTA PREGUNTA (ii): TEST DE CHOW PARA γ = 0 ===")
    for activo in [1, 2]:
        F, p, b1, b2 = estimar_chow_activo(data, columna=activo, umbral=0)

        print(f"Activo {activo}:")
        print(f"  Estadístico F = {F:.4f}")
        print(f"  p-value       = {p:.4f}")
        # Conclusión rápida según el p-value obtenido
        if p > 0.05:
            print("  Conclusión: No difiere la dinámica (p-value > 0.05). Regímenes estadísticamente iguales.")
        else:
            print("  Conclusión: Sí difiere la dinámica (p-value <= 0.05). Existe cambio estructural.")
        print()
def table_pregunta_3():
    """
    (iii) Repita usando otros umbrales (percentiles del 0.1 al 0.9) y la Mediana (P-0.5)
          para observar cuánto cambian los coeficientes y el ajuste.
    """
    umbrales = [i / 10 for i in range(1, 10)]

    tabla_p3 = {
        'Umbral (q)': [f"Percentil {u}" for u in umbrales],
        'Activo 1 φ1 (Ganancia)': [], 'Activo 1 φ2 (Pérdida)': [], 'Activo 1 Δφ': [],
        'Activo 2 φ1 (Ganancia)': [], 'Activo 2 φ2 (Pérdida)': [], 'Activo 2 Δφ': []
    }

    for activo in [1, 2]:
        for u in umbrales:
            F, p, b1, b2 = estimar_chow_activo(data, columna=activo, umbral=u)
            # b1 = Régimen 1 (ganancias), b2 = Régimen 2 (pérdidas)
            tabla_p3[f'Activo {activo} φ1 (Ganancia)'].append(b1[1])
            tabla_p3[f'Activo {activo} φ2 (Pérdida)'].append(b2[1])
            tabla_p3[f'Activo {activo} Δφ'].append(b1[1] - b2[1])  # Ganancia - Pérdida

    df_p3 = pd.DataFrame(tabla_p3)
    print("\n=== RESPUESTA PREGUNTA (iii): ANÁLISIS DE COEFICIENTES EN DISTINTOS UMBRALES ===")
    print(df_p3.to_string(index=False, formatters={
        'Activo 1 φ1 (Ganancia)': '{:.4f}'.format, 'Activo 1 φ2 (Pérdida)': '{:.4f}'.format,
        'Activo 1 Δφ': '{:.4f}'.format,
        'Activo 2 φ1 (Ganancia)': '{:.4f}'.format, 'Activo 2 φ2 (Pérdida)': '{:.4f}'.format,
        'Activo 2 Δφ': '{:.4f}'.format
    }))
    print("\n*Nota: El 'Percentil 0.5' corresponde exactamente a usar la Mediana.")
    print()

results = calculo_percentiles()