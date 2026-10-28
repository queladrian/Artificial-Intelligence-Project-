"""Verifica que la formulación EDP+condiciones de frontera que usará la PINN
reproduce la solución analítica del baseline (solver de diferencias finitas)."""
import numpy as np
from common import MATERIALS, derived
from baseline_lm import theta_model

def fd_solution(k, h, P, ap, tau, p, T_end=300.0, nx=31, dt=0.1):
    A, s = derived(p); rho, c, L = p["rho"], p["c"], p["L"]
    kap, nu = k/(rho*c), h*s/(rho*A*c)
    x = np.linspace(0, L, nx); dx = x[1]-x[0]
    th = np.zeros(nx); out = {}
    nt = int(round(T_end/dt))
    assert kap*dt/dx**2 < 0.5
    hist = [th.copy()]
    for n in range(nt):
        t = n*dt
        F = P*(1 - np.exp(-ap*(t+tau)))        # potencia que entra [W]
        g = np.empty(nx+2); g[1:-1] = th
        g[0] = th[1] + 2*dx*F/(k*A)                           # -kA θx(0) = F
        g[-1] = th[-2]                                        # extremo adiabático
        th = th + dt*(kap*(g[:-2]-2*g[1:-1]+g[2:])/dx**2 - nu*g[1:-1])
        hist.append(th.copy())
    return x, np.array(hist), dt

if __name__ == "__main__":
    p = MATERIALS["al"]; k, h, P, ap, tau = 146.47, 141.49, 1.4157, 0.09905, 2.014
    x, H, dt = fd_solution(k, h, P, ap, tau, p)
    for xi in p["X"]:
        i = int(round(xi/(x[1]-x[0])))
        ts = np.array([50., 150., 300.]); idx = (ts/dt).astype(int)
        an = theta_model([h, k, P], ts, np.full(3, xi), ap, tau, p)
        print(f"x={xi:.2f} m  FD={np.round(H[idx, i],2)}  analítica={np.round(an,2)}")
