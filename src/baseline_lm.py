"""Baseline: reproducción del estimador de Tomanek y Stutts (mínimos cuadrados
Levenberg-Marquardt sobre la solución analítica). Estima k, h y Pss.
Uso: python src/baseline_lm.py data/50al1.csv"""
import sys, json
import numpy as np
from scipy.optimize import least_squares
from common import MATERIALS, derived, load_csv, h_churchill_bernstein

N_TERMS = 100

def fit_boundary(t, TT, X):
    """Ajusta α y τ de la rampa de calentamiento (como en el código original)."""
    B = (3*TT[:, 0] - 4*TT[:, 1] + TT[:, 2]) / (2*X[1])
    f0 = lambda V: V[0]*(1 - np.exp(-V[1]/10000*(t + V[2]/10))) - B
    out = least_squares(f0, [3000, 700, 30], method="lm")
    return out.x[1]/10000, out.x[2]/10          # alpha, tau

def theta_model(V, t, x, ap, tau, p):
    """Solución analítica θ(x,t). V = [h, k, Pss]."""
    A, s = derived(p); rho, c, L = p["rho"], p["c"], p["L"]
    nu = V[0]*s/(rho*A*c); K = V[1]/(rho*c); P = V[2]
    tot = np.zeros_like(t, dtype=float)
    for n in range(1, N_TERMS+1):
        b = n*np.pi*np.sqrt(K)/L
        tot += 2*K*P*((-b**2-nu)*np.exp((-b**2-nu)*t-ap*tau)
              + (b**2-ap+nu)*np.exp(-t*(b**2+nu)) + (b**2+nu)*np.exp(-ap*(t+tau))
              - b**2 + ap - nu) / (V[1]*L*A*(-b**2+ap-nu)*(b**2+nu)) \
              * np.cos(b/np.sqrt(K)*x)
    return (-(np.exp(-ap*tau-nu*t)*nu - np.exp(-ap*(t+tau))*nu + (ap-nu)*(np.exp(-nu*t)-1))
            * P*K/(V[1]*L*A*nu*(ap-nu)) + tot)

def estimate(csv, mat="al"):
    p = MATERIALS[mat]; X = np.array(p["X"]); D, L = p["D"], p["L"]
    A, s = derived(p)
    t, TT = load_csv(csv)
    ap, tau = fit_boundary(t, TT, X)
    # modelo estacionario -> valores iniciales
    Tss = TT[-20:, :].T.reshape(-1); xss = np.repeat(X, 20)
    def f01(V):
        m = np.sqrt(4*V[1]/D)
        return V[0]/m*np.cosh(m*(L-xss))/np.sinh(m*L) - Tss
    o1 = least_squares(f01, [50, 1], method="lm")
    h0 = h_churchill_bernstein(p["v"], D); k0 = h0/o1.x[1]; P0 = o1.x[0]*k0*A
    # modelo transitorio
    tt = np.tile(t, 6); xx = np.repeat(X, len(t)); Tv = TT.T.reshape(-1)
    res = lambda V: theta_model(V, tt, xx, ap, tau, p) - Tv
    out = least_squares(res, [h0, k0, P0], method="lm")
    rmse = float(np.sqrt(np.mean(out.fun**2)))
    return dict(archivo=csv, alpha=ap, tau=tau, h=out.x[0], k=out.x[1], Pss=out.x[2],
                rmse_C=rmse, k_inicial=k0)

if __name__ == "__main__":
    r = estimate(sys.argv[1]); print(json.dumps(r, indent=2))
