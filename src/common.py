"""Parámetros del experimento y carga de datos.
Valores tomados del código oneDkhEstimator (Tomanek y Stutts, Zenodo 10.5281/zenodo.5683312),
caso aluminio Al6061-T6, grupo "Temperature" (viento constante de 5 m/s)."""
import numpy as np, pandas as pd

MATERIALS = {
    "al": dict(X=[0, 0.01, 0.02, 0.03, 0.04, 0.05], rho=2767.99, c=896.0,
               D=0.003175, L=0.150, Tinf=20.5, v=5.0),
}

def derived(p):
    A = np.pi * p["D"] ** 2 / 4      # área transversal [m^2]
    s = np.pi * p["D"]               # perímetro [m]
    return A, s

def load_csv(path):
    """Devuelve t (n,), TT (n,6) con la temperatura en exceso θ = T - T∞."""
    df = pd.read_csv(path)
    t = df["Time"].to_numpy(float)
    TT = df[[f"TC{i}" for i in range(1, 7)]].to_numpy(float)
    return t, TT

def h_churchill_bernstein(v, D):
    """Estimación inicial de h (ecuación de Churchill-Bernstein, aire)."""
    rhoair, muair, Pr, kair = 1.23, 1.789e-5, 0.71, 0.02602
    Re = rhoair * v * D / muair
    return kair / D * (0.3 + 0.62 * Re**0.5 * Pr**(1/3) / (1 + (0.4/Pr)**(2/3))**0.25
                       * (1 + (Re/282000)**(5/8))**(-4/5))
