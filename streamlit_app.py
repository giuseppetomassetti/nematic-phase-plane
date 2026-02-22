"""
Streamlit version: Interactive phase-plane viewer for the nematic film ODE
  (1+p²)[(ε+u²)p²+u²] = u·u″·[u²p²+ε(2-p²)+u²]

Axes: u > 0, p = u' > 0 (first quadrant only, exploiting reflection symmetry).

Controls (sidebar sliders):
  ε   – elastic-to-surface-tension ratio
  λ   – aspect ratio (boundary condition)
  u_max, p_max – viewport bounds
  density – streamline density

Numba JIT accelerates the vector-field evaluation over the grid.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numba import njit, prange
from scipy.optimize import brentq
import streamlit as st

# ── Streamlit config (must be first Streamlit command) ──────────────────────
st.set_page_config(page_title="Nematic Film Phase Plane", layout="wide")

# ── Numba-accelerated vector field ──────────────────────────────────────────

@njit(parallel=True, cache=True)
def vector_field(U, P, eps, du, dp):
    """
    Fill du = p  and  dp = u″  arrays in-place over a 2-D grid.
    Singular denominator → NaN (masked later).
    """
    rows, cols = U.shape
    for i in prange(rows):
        for j in range(cols):
            u = U[i, j]
            p = P[i, j]
            num = (1.0 + p * p) * ((eps + u * u) * p * p + u * u)
            den = u * (u * u * p * p + eps * (2.0 - p * p) + u * u)
            du[i, j] = p
            dp[i, j] = num / den if abs(den) > 1e-10 else np.nan


@njit(cache=True)
def _upp(uu, pp, eps):
    """Evaluate u'' from the ODE."""
    if uu <= 0.0:
        return np.nan
    num = (1.0 + pp * pp) * ((eps + uu * uu) * pp * pp + uu * uu)
    den = uu * (uu * uu * pp * pp + eps * (2.0 - pp * pp) + uu * uu)
    return num / den if abs(den) > 1e-10 else np.nan


@njit(cache=True)
def shoot_u_at_1(A, eps, n_steps=6000):
    """
    Integrate from (u=A, p=0) at xi=0 forward to xi=1.
    Returns u(1).  Uses the symmetry condition u'(0)=0.
    """
    h = 1.0 / n_steps
    u, p = A, 0.0
    for _ in range(n_steps):
        k1u = p;              k1p = _upp(u,             p,             eps)
        k2u = p + h/2*k1p;   k2p = _upp(u + h/2*k1u,  p + h/2*k1p,  eps)
        k3u = p + h/2*k2p;   k3p = _upp(u + h/2*k2u,  p + h/2*k2p,  eps)
        k4u = p + h  *k3p;   k4p = _upp(u + h  *k3u,  p + h  *k3p,  eps)
        u = u + h/6*(k1u + 2*k2u + 2*k3u + k4u)
        p = p + h/6*(k1p + 2*k2p + 2*k3p + k4p)
        if np.isnan(u) or np.isnan(p) or u <= 0.0:
            return np.nan
    return u


@njit(cache=True)
def integrate_trajectory(u0, p0, eps, xi_max=2.0, n_steps=4000):
    """
    Forward-integrate a single trajectory with 4th-order Runge-Kutta.
    Returns (u_arr, p_arr) arrays.
    """
    h = xi_max / n_steps
    u = np.empty(n_steps + 1)
    p = np.empty(n_steps + 1)
    u[0], p[0] = u0, p0

    for k in range(n_steps):
        u_k, p_k = u[k], p[k]

        k1u, k1p = p_k,               _upp(u_k,             p_k,  eps)
        k2u, k2p = p_k + h/2*k1p,     _upp(u_k + h/2*k1u,  p_k + h/2*k1p, eps)
        k3u, k3p = p_k + h/2*k2p,     _upp(u_k + h/2*k2u,  p_k + h/2*k2p, eps)
        k4u, k4p = p_k + h  *k3p,     _upp(u_k + h  *k3u,  p_k + h  *k3p, eps)

        u[k+1] = u_k + h/6*(k1u + 2*k2u + 2*k3u + k4u)
        p[k+1] = p_k + h/6*(k1p + 2*k2p + 2*k3p + k4p)

        if np.isnan(p[k+1]) or np.isnan(u[k+1]) or u[k+1] <= 0.0:
            u[k+1:] = np.nan
            p[k+1:] = np.nan
            break

    return u, p


@njit(parallel=True, cache=True)
def beltrami_H(U, P, eps, H_out):
    """
    H(u,p) = u/sqrt(1+p²) - ε·p²/(u·(1+p²)^(3/2))
    This is the Beltrami first integral of the ODE (conserved along trajectories).
    For ε=0 it reduces to u/sqrt(1+p²) = a, i.e. the catenary hyperbolas.
    """
    rows, cols = U.shape
    for i in prange(rows):
        for j in range(cols):
            u = U[i, j]
            p = P[i, j]
            s = (1.0 + p * p) ** 0.5
            H_out[i, j] = u / s - eps * p * p / (u * s * s * s) if u > 1e-10 else np.nan


# ── Numba warmup (JIT compilation on first render) ──────────────────────────
@st.cache_resource
def _warmup_numba():
    """Trigger JIT compilation once per session."""
    _U0 = np.ones((4, 4), np.float64)
    _P0 = np.ones((4, 4), np.float64)
    _du = np.empty_like(_U0)
    _dp = np.empty_like(_P0)
    vector_field(_U0, _P0, 0.1, _du, _dp)
    integrate_trajectory(1.0, 0.5, 0.1, 1.0, 10)
    _H0 = np.empty_like(_U0)
    beltrami_H(_U0, _P0, 0.1, _H0)
    shoot_u_at_1(0.5, 0.1, 10)
    return "Numba JIT compilation done."

_warmup_numba()


GRID_N   = 60          # resolution of the background vector grid
LAMBDA_C = 1.5089      # classical Goldschmidt critical aspect ratio


def _upp_py(uu, pp, eps):
    """Pure-Python version of _upp for trajectory reconstruction."""
    if uu <= 0:
        return np.nan
    num = (1 + pp*pp) * ((eps + uu*uu)*pp*pp + uu*uu)
    den = uu * (uu*uu*pp*pp + eps*(2 - pp*pp) + uu*uu)
    return num/den if abs(den) > 1e-10 else np.nan


def find_bvp_solution(eps, lam, n_steps=8000):
    """
    Shooting method: bisect on the neck radius A in (0, lam) to find
    the solution satisfying u'(0)=0 and u(1)=lam.
    Returns (u_traj, p_traj) on [0,1], or (None, None) if not found.
    Also looks for a second (thin) branch if the first one exists.
    """
    results = []
    # Sweep A values to find sign-change brackets
    A_lo, A_hi = 0.02, lam - 1e-3
    if A_lo >= A_hi:
        return results

    def f(A):
        val = shoot_u_at_1(A, eps, n_steps)
        return (val - lam) if not np.isnan(val) else np.nan

    # Scan for sign changes
    n_scan = 200
    A_scan = np.linspace(A_lo, A_hi, n_scan)
    f_scan = np.array([f(a) for a in A_scan])

    for i in range(n_scan - 1):
        fa, fb = f_scan[i], f_scan[i + 1]
        if np.isnan(fa) or np.isnan(fb):
            continue
        if fa * fb < 0:
            try:
                A_sol = brentq(f, A_scan[i], A_scan[i + 1], xtol=1e-9)
            except Exception:
                continue
            # Reconstruct full trajectory u(ξ), p(ξ) for ξ in [0,1]
            h = 1.0 / n_steps
            u_t = np.empty(n_steps + 1)
            p_t = np.empty(n_steps + 1)
            u_t[0], p_t[0] = A_sol, 0.0
            ok = True
            for k in range(n_steps):
                u_k, p_k = u_t[k], p_t[k]
                k1u = p_k;              k1p = _upp_py(u_k,            p_k,            eps)
                k2u = p_k+h/2*k1p;    k2p = _upp_py(u_k+h/2*k1u,   p_k+h/2*k1p,   eps)
                k3u = p_k+h/2*k2p;    k3p = _upp_py(u_k+h/2*k2u,   p_k+h/2*k2p,   eps)
                k4u = p_k+h  *k3p;    k4p = _upp_py(u_k+h  *k3u,   p_k+h  *k3p,   eps)
                u_t[k+1] = u_k + h/6*(k1u+2*k2u+2*k3u+k4u)
                p_t[k+1] = p_k + h/6*(k1p+2*k2p+2*k3p+k4p)
                if np.isnan(u_t[k+1]) or u_t[k+1] <= 0:
                    ok = False; break
            if ok:
                results.append((A_sol, u_t, p_t))
    return results


def draw(eps, lam, u_max, p_max, density):
    """Draw the phase plane."""
    fig = plt.figure(figsize=(11, 8))
    fig.patch.set_facecolor('#1a1a2e')
    
    ax = fig.add_subplot(111)
    ax.set_facecolor('#0f0f23')
    ax.set_xlim(0.02, u_max)
    ax.set_ylim(0.0,  p_max)
    ax.set_xlabel(r'$u$',     color='white', fontsize=13)
    ax.set_ylabel(r"$p=u'$",  color='white', fontsize=13)
    ax.tick_params(colors='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#444466')
    ax.set_title(
        rf'Phase plane  $(u > 0,\; p > 0)$   $\varepsilon = {eps:.3f}$,  $\lambda = {lam:.2f}$',
        color='white', fontsize=13, pad=8
    )

    # ── Vector field (dense grid for H computation, coarse for arrows) ───
    u_vals = np.linspace(0.05, u_max, GRID_N)
    p_vals = np.linspace(0.02, p_max, GRID_N)
    U, P   = np.meshgrid(u_vals, p_vals)
    dU = np.empty_like(U)
    dP = np.empty_like(P)
    vector_field(U, P, eps, dU, dP)

    bad = np.isnan(dP) | (np.abs(dP) > 500)
    dU[bad] = np.nan
    dP[bad] = np.nan

    speed = np.hypot(dU, dP)
    spd_clip = np.nanpercentile(speed[~bad], 94) if np.any(~bad) else 1.0
    speed_norm = np.clip(speed, 0, spd_clip)

    # ── Beltrami first integral H(u,p) = const  — exact invariant curves ─
    # Trajectories of the ODE ARE the level curves of H (Beltrami identity).
    H_grid = np.empty_like(U)
    beltrami_H(U, P, eps, H_grid)
    H_finite = H_grid[np.isfinite(H_grid)]
    if H_finite.size > 0:
        h_min = max(np.percentile(H_finite, 2),  0.05)
        h_max = np.percentile(H_finite, 97)
        levels = np.linspace(h_min, h_max, 22)
        # background colour fill
        ax.contourf(U, P, H_grid, levels=levels,
                    cmap='twilight_shifted', alpha=0.35, zorder=0)
        # exact trajectory curves — bold & primary
        cs = ax.contour(U, P, H_grid, levels=levels,
                        colors='white', linewidths=0.9, alpha=0.75, zorder=1)
        ax.clabel(cs, levels[::4], inline=True, fontsize=6,
                  fmt=r'$H\!=\!%.2f$', colors='#ccccff')

        # ── Direction arrowheads on H curves ───
        n_per_curve = max(2, int(density * 3))
        sample_pts = []
        for path in cs.get_paths():
            verts = path.vertices
            if len(verts) < 4:
                continue
            idx = np.round(np.linspace(1, len(verts) - 2, n_per_curve)).astype(int)
            for i in idx:
                xu, xpv = verts[i]
                if 0 < xu <= u_max and 0 < xpv <= p_max:
                    sample_pts.append((xu, xpv))
        if sample_pts:
            sp = np.array(sample_pts)
            Us_b = sp[:, 0].reshape(1, -1)
            Ps_b = sp[:, 1].reshape(1, -1)
            dUs_b = np.empty_like(Us_b)
            dPs_b = np.empty_like(Ps_b)
            vector_field(Us_b, Ps_b, eps, dUs_b, dPs_b)
            du_b = dUs_b[0]
            dp_b = dPs_b[0]
            spd_b = np.hypot(du_b, dp_b)
            delta = 1e-9
            for k, (xu, xpv) in enumerate(sample_pts):
                sk = spd_b[k]
                if not (np.isfinite(sk) and sk > 1e-12):
                    continue
                dx = delta * du_b[k] / sk
                dy = delta * dp_b[k] / sk
                ax.annotate('',
                            xy=(xu + dx, xpv + dy), xytext=(xu, xpv),
                            arrowprops=dict(arrowstyle='->', color='white',
                                            lw=0.8, mutation_scale=8),
                            annotation_clip=True, zorder=3)

    # ── Slowest-manifold locus ────────────────────────────────────────────
    ax.contour(U, P, speed_norm, levels=[spd_clip * 0.04],
               colors=['#00ffe0'], linewidths=0.8, linestyles='--', alpha=0.5)

    # ── Vertical line at u = λ (boundary condition) ───────────────────────
    ax.axvline(lam, color='#ffffff', lw=0.8, ls=':', alpha=0.5,
               label=fr'$u=\lambda={lam:.2f}$')
    # mark the critical Goldschmidt threshold
    ax.axvline(LAMBDA_C, color='#ff4444', lw=0.9, ls='--', alpha=0.6,
               label=fr'$\lambda_c={LAMBDA_C}$')

    # ── BVP solution trajectory ───────────────────────────────────────────
    solutions = find_bvp_solution(eps, lam)
    bvp_colors = ['#00ff88', '#ff9900']   # green = wide branch, orange = narrow
    for idx, (A_sol, u_t, p_t) in enumerate(solutions):
        col = bvp_colors[idx % len(bvp_colors)]
        mask = (u_t < u_max) & (p_t < p_max) & (p_t >= 0)
        ax.plot(u_t[mask], p_t[mask], color=col, lw=2.4,
                zorder=7, alpha=0.95,
                label=fr'BVP sol. $A={A_sol:.3f}$')
        ax.plot(A_sol, 0, 's', color=col, ms=7, zorder=8,
                markeredgecolor='white', markeredgewidth=0.5)
        # mark endpoint at u=λ
        if mask.any():
            ax.plot(u_t[mask][-1], p_t[mask][-1], '^', color=col,
                    ms=7, zorder=8, markeredgecolor='white', markeredgewidth=0.5)
    if not solutions:
        ax.text(0.5, 0.5, 'No BVP solution found\nin this range',
                transform=ax.transAxes, color='#ff6666',
                ha='center', va='center', fontsize=10,
                bbox=dict(facecolor='#1a1a3a', edgecolor='#ff6666',
                          alpha=0.8, boxstyle='round'))

    # ── Legend ────────────────────────────────────────────────────────────
    ax.legend(
        title=r'Exact invariant curves $H(u,p)=H_0$', title_fontsize=8,
        fontsize=7, loc='upper right',
        facecolor='#1a1a3a', edgecolor='#555577', labelcolor='white',
        ncol=1, framealpha=0.8
    )

    fig.suptitle(
        r'$(1+p^2)[(\varepsilon+u^2)p^2+u^2] = u\,u^{\prime\prime}[u^2p^2+\varepsilon(2-p^2)+u^2]$',
        color='#ccccff', fontsize=10, y=0.98
    )

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("Nematic Film Phase Plane Explorer")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    eps = st.slider(r"ε", 0.0, 1.0, 0.1, 0.005,
                    help="Elastic-to-surface-tension ratio")

with col2:
    lam = st.slider(r"λ", 0.3, 3.0, 1.2, 0.01,
                    help="Aspect ratio (boundary condition)")

with col3:
    u_max = st.slider(r"u_max", 0.5, 2 * LAMBDA_C, 2 * LAMBDA_C, 0.05,
                      help="Viewport u-range upper bound")

with col4:
    p_max = st.slider(r"p_max", 0.1, 10.0, 1.0, 0.1,
                      help="Viewport p-range upper bound")

with col5:
    density = st.slider("density", 0.5, 3.0, 1.4, 0.1,
                       help="Streamline/arrow density")

# Draw and display
fig = draw(eps, lam, u_max, p_max, density)
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# Info section
with st.expander("📖 About this visualization"):
    st.markdown(r"""
    This interactive explorer visualizes the phase plane of the nematic film equation:
    
    $$(1+p^2)[(\varepsilon+u^2)p^2+u^2] = u \cdot u''[u^2p^2+\varepsilon(2-p^2)+u^2]$$
    
    **Key features:**
    - **White contours**: Exact invariant curves from the **Beltrami identity** $H(u,p) = \text{const}$
    - **Green trajectory**: Wide branch BVP solution (neck profile)
    - **Orange trajectory**: Narrow branch BVP solution (if it exists)
    - **Red dashed line**: Goldschmidt critical threshold $\lambda_c \approx 1.51$
    
    **Parameters:**
    - **ε** controls the balance between elastic bending and surface tension
    - **λ** is the aspect ratio (boundary condition: $u(1) = \lambda$)
    - **u_max, p_max** adjust the viewport
    - **density** controls the number of direction arrows
    
    For ε=0, the system reduces to **catenaries** (hanging chains), and $H = u/\sqrt{1+p^2} = a$.
    """)
