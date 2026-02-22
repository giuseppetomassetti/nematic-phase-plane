# Nematic Film Phase Plane Explorer

Interactive visualization of the nematic liquid crystal film ODE:

$$(1+p^2)[(\varepsilon+u^2)p^2+u^2] = u \cdot u''[u^2p^2+\varepsilon(2-p^2)+u^2]$$

## Three-Layer Fix Summary

### 1. ✅ Missing `requirements.txt` (Immediate Fix)
**Problem**: Streamlit Cloud had no package manifest for installing dependencies.

**Solution**: Created `requirements.txt` with:
```
numpy
matplotlib
scipy
numba
llvmlite
streamlit
```

This ensures all dependencies are installed when deploying to Streamlit Cloud.

---

### 2. ✅ Double-Backslash Bug in Desktop App (Bug Fix)
**Problem**: The `interactive_phase_plane.py` had escaped backslashes in slider key lookups that didn't match the slider definitions:
```python
# ❌ BEFORE (broken):
sliders[r'$\\varepsilon$'].val    # double backslash (escaped)
# vs.
slider_specs = [(..., r'$\varepsilon$', ...)]   # single backslash (in label)
```

**Solution**: Fixed all references in `update()`, `on_click()`, and initial `draw()`:
```python
# ✅ AFTER (fixed):
sliders[r'$\varepsilon$'].val     # single backslash (matches label)
sliders[r'$\lambda$'].val
sliders[r'$u_{\max}$'].val
sliders[r"$p_{\max}$"].val
```

**Note**: The desktop app will now work locally if you have matplotlib widgets.

---

### 3. ✅ Streamlit Incompatibility (Architectural Rewrite)
**Problem**: The code is a **desktop matplotlib application** with GUI widgets (`matplotlib.widgets.Slider`, `Button`, `TextBox`). Streamlit is a **web framework** that doesn't support these.

| Feature | Desktop Matplotlib | Streamlit |
|---------|-------------------|-----------|
| Show plot | `plt.show()` | `st.pyplot(fig)` |
| Sliders | `matplotlib.widgets.Slider` | `st.slider()` |
| Buttons | `matplotlib.widgets.Button` | `st.button()` |
| Text input | `matplotlib.widgets.TextBox` | `st.text_input()` |
| Click events | `fig.canvas.mpl_connect()` | ❌ Not supported |

**Solution**: Created `streamlit_app.py` with:
- ✅ All **Numba-compiled functions preserved** (vector_field, integrate_trajectory, beltrami_H, etc.)
- ✅ Replaced matplotlib widgets with **Streamlit sidebar sliders** (`st.slider()`)
- ✅ Use `st.pyplot(fig)` instead of `plt.show()`
- ✅ Added `@st.cache_resource` decorator to warm up Numba JIT compilation once per session
- ✅ Proper memory management with `plt.close(fig)` after rendering

**Key optimizations for Streamlit Cloud**:
1. **Numba JIT caching**: `@st.cache_resource` ensures the JIT-compiled warm-up only runs once per session
2. **Figure cleanup**: `plt.close(fig)` prevents memory leaks across reruns
3. **Stateless computation**: No persistent matplotlib state; figures are recreated on each parameter change

---

## Usage

### Local Desktop (matplotlib GUI)
```bash
cd /Users/joe/Documents/Repos/nematic-phase-plane/NematicFilmsOIST
python interactive_phase_plane.py
```
- Drag sliders to adjust parameters in real-time
- Click on the phase plane to trace custom trajectories

### Local Streamlit (web app)
```bash
cd /Users/joe/Documents/Repos/nematic-phase-plane
pip install -r requirements.txt
streamlit run streamlit_app.py
```
- Opens in browser at `http://localhost:8501`
- Adjust parameters via sidebar sliders

### Streamlit Cloud (deployed)
1. Push to GitHub and link repository to [Streamlit Cloud](https://streamlit.io/cloud)
2. Deploy `streamlit_app.py` 
3. Live at `app URL`

---

## Files Modified

- **`requirements.txt`** (new) — Package manifest for cloud deployment
- **`streamlit_app.py`** (new) — Streamlit-compatible web version
- **`interactive_phase_plane.py`** (fixed) — Desktop app backslash bugs corrected
- **`.streamlit/config.toml`** (new) — Streamlit theme & logging config

---

## Technical Notes

### Why Numba Works on Streamlit Cloud
- JIT compilation fires **once** on first render
- Streamlit's caching mechanism (`@st.cache_resource`) prevents recompilation on widget interactions
- `llvmlite` must be in `requirements.txt` for LLVM backend

### BVP Solution Performance
- `find_bvp_solution()` uses pure-Python integration (`_upp_py`) to solve shooting boundary-value problems
- For faster performance, consider wrapping the RK4 loop in `@njit` if it becomes a bottleneck

### Invariant Curves (Beltrami Identity)
- White contours are **exact trajectories** of the ODE (conserved quantity $H(u,p) = \text{const}$)
- For $\varepsilon=0$, reduces to catenary: $H = u/\sqrt{1+p^2} = a$
- Direction arrows show flow direction along these curves

---

## References

- Belivaqua et al.: "A variational analysis of nematic axisymmetric films — the covariant derivative case"
- Goldschmidt problem: Minimal axisymmetric surfaces ($\lambda_c \approx 1.5089$)
