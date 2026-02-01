import sys
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, medfilt
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from dierssen_qwip_functions import *

from ipywidgets import IntSlider, Button, HBox, VBox, Output, Label
from IPython.display import display

# --- wavelength axis config (adjust if needed) ---
WL_START = 400   # nm for spectral_bin_1
WL_STEP  = 1     # nm per bin

# ---------- general helpers ----------
def spectral_cols(df):
    return [c for c in df.columns if c != "timestamp"]

def wl_axis_from_bins(df, wl_start=WL_START, wl_step=WL_STEP):
    n = len(spectral_cols(df))
    return np.arange(wl_start, wl_start + n*wl_step, wl_step, dtype=float)

def summarize_nans(df):
    cols = spectral_cols(df)
    nan_map = df[cols].isna()
    row_nan_count = nan_map.sum(axis=1)
    total_cells = len(df)*len(cols)
    total_nans  = int(nan_map.values.sum())
    full_nan_rows = (row_nan_count == len(cols))

    summary = pd.DataFrame({
        "timestamp": df["timestamp"].values,
        "nan_count": row_nan_count.values,
        "nan_pct": (row_nan_count.values / len(cols))*100,
        "full_nan_row": full_nan_rows.values
    })
    print(f"[NaN Summary] Total cells: {total_cells:,} | Total NaNs: {total_nans:,} ({100*total_nans/total_cells:.2f}%)")
    print(f"Rows with ALL NaNs: {full_nan_rows.sum()} / {len(df)}")
    return summary

def plot_overlay_row(before_df, after_df, row_idx=0, title="Overlay"):
    cols = spectral_cols(before_df)
    x = wl_axis_from_bins(before_df)
    y0 = before_df[cols].iloc[row_idx].values.astype(float)
    y1 = after_df[cols].iloc[row_idx].values.astype(float)
    ts = after_df["timestamp"].iloc[row_idx]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y0, mode='markers', name='Before', opacity=0.5))
    fig.add_trace(go.Scatter(x=x, y=y1, mode='markers', name='After'))
    fig.update_layout(
        title=f"{title} — row {row_idx} @ {ts}",
        xaxis_title="Wavelength (nm)", yaxis_title="Reflectance",
        width=900, height=500
    )
    fig.show()
    

def plot_reflectance_heatmap(df, spectral_prefix="", 
                             wl_start=400, cmap="RdYlGn",
                             vmin=-0.075, vmax=0.275,
                             title="Reflectance Variation Over Time",
                             figsize=(10,7)):
    """
    Plot time × wavelength reflectance heatmap.
    df must contain a 'timestamp' column and spectral_bin_* columns.
    """

    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain a 'timestamp' column.")

    # Spectral columns
    spec_cols = [c for c in df.columns if c != "timestamp"]
    if len(spec_cols) == 0:
        raise ValueError(f"No columns starting with '{spectral_prefix}' found.")

    # Convert values → matrix
    data = df[spec_cols].to_numpy()

    # Wavelength axis
    wl_end = wl_start + len(spec_cols) - 1
    wavelengths = np.arange(wl_start, wl_end + 1)

    # Date labels
    dates = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")

    # Plot
    plt.figure(figsize=figsize)

    plt.imshow(
        data,
        aspect='auto',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[wavelengths[0], wavelengths[-1], len(df), 0],
        origin='upper'      # time runs from top to bottom
    )

    plt.colorbar(label="Reflectance")

    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Date")
    plt.title(title, fontsize=14)

    # Y-ticks → readable dates
    yticks = np.linspace(0, len(df)-1, 10).astype(int)
    plt.yticks(yticks, dates.iloc[yticks], rotation=0)

    plt.tight_layout()
    plt.show()

def plot_waterfall_surface(df, title="Waterfall", clip_min=None, clip_max=None):
    cols = spectral_cols(df)
    M = df.set_index("timestamp")[cols]
    if clip_min is not None and clip_max is not None:
        M_plot = M.where((M > clip_min) & (M < clip_max))
    else:
        M_plot = M.copy()

    x = wl_axis_from_bins(df)
    y = M_plot.index.to_list()
    Z = M_plot.values

    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y, colorbar=dict(title="Rrs"))])
    fig.update_layout(
        title=title, autosize=False, width=900, height=800,
        scene=dict(xaxis_title='Spectral Bins (nm)', yaxis_title='Time', zaxis_title='Reflectance'),
        margin=dict(l=65, r=50, b=65, t=90)
    )
    fig.show()

# ---------- robust spectral/temporal filters ----------
def mad(x, c=1.4826):
    m = np.nanmedian(x)
    return c * np.nanmedian(np.abs(x - m))

def hampel_1d(x, window=5, n_sigmas=3.0):
    x = np.asarray(x, dtype=float)
    y = x.copy()
    n = len(x)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = x[lo:hi]
        med = np.nanmedian(seg)
        s = mad(seg)
        if s == 0 or np.isnan(s): 
            continue
        if np.abs(x[i] - med) > n_sigmas * s:
            y[i] = med
    return y

def hampel_mask_1d(x, window=9, n_sigmas=3.0):
    x = np.asarray(x, dtype=float)
    n = x.size
    k = window // 2
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i-k); hi = min(n, i+k+1)
        seg = x[lo:hi]
        med = np.nanmedian(seg)
        s = 1.4826 * np.nanmedian(np.abs(seg - med))  # MAD
        if np.isfinite(x[i]) and np.isfinite(s) and s > 0:
            mask[i] = np.abs(x[i] - med) > n_sigmas * s
    return mask

def rolling_median_1d(x, window=5):
    w = max(3, window if window % 2 == 1 else window + 1)
    try:
        return medfilt(x, kernel_size=w)
    except Exception:
        return x

def savgol_safe(x, suggested=11, poly=3):
    n = len(x)
    w = min(suggested, n if n % 2 == 1 else n - 1)
    w = max(3, w)
    p = min(poly, w - 1)
    try:
        return savgol_filter(x, window_length=w, polyorder=p, mode="interp")
    except ValueError:
        return x

# ===============================
# QWIP Helpers
# ===============================

# ---- Tunables ----
REQ_WLS = [492, 665]
VIS_RANGE = (400, 700)         # inclusive
MIN_VALID_RATIO = 0.90
QWIP_THRESH = 0.2
REPAIR_MAX_GAP = 2             # AFTER only
POLY_P = np.array([-8.399884740300151e-09, 1.715532100780679e-05,
                   -1.301670056641901e-02, 4.357837742180596e+00,
                   -5.449532021524279e+02])

# ---- Helpers ----
def map_bins_to_wls(df, wl_start, wl_step):
    cols = [c for c in df.columns if c.startswith("spectral_bin_")]
    wls = (wl_start + np.arange(len(cols)) * wl_step).astype(int)
    return df.rename(columns=dict(zip(cols, map(str, wls)))), wls

def extract_vis(df_wl_renamed, all_wls):
    vis = np.arange(VIS_RANGE[0], VIS_RANGE[1] + 1, 1)  # 400..700
    keep = [str(w) for w in vis if str(w) in df_wl_renamed.columns]
    return df_wl_renamed.loc[:, keep], vis

def minimal_repair_short_gaps(A, max_gap):
    """Linear-interp fill only for short (<= max_gap) contiguous NaN runs."""
    A = np.array(A, float, copy=True)
    n_rows, n_cols = A.shape
    for r in range(n_rows):
        x = A[r]
        isn = np.isnan(x)
        if not isn.any(): 
            continue
        i = 0
        while i < n_cols:
            if not isn[i]:
                i += 1
                continue
            j = i
            while j + 1 < n_cols and isn[j + 1]:
                j += 1
            run_len = j - i + 1
            L, R = i - 1, j + 1
            if run_len <= max_gap and L >= 0 and R < n_cols and np.isfinite(x[L]) and np.isfinite(x[R]):
                x[i:j + 1] = np.interp(np.arange(i, j + 1), [L, R], [x[L], x[R]])
            i = j + 1
    return A

def find_indices(targets, wls_vis):
    return [int(np.argmin(np.abs(wls_vis - t))) for t in targets]

def build_evaluable_mask(M, wls_vis, min_valid_ratio, req_wls=REQ_WLS):
    i_req = find_indices(req_wls, wls_vis)
    valid_counts = np.isfinite(M).sum(axis=1)
    min_needed = int(np.ceil(min_valid_ratio * M.shape[1]))
    ok_vis = valid_counts >= min_needed
    have_all = np.logical_and.reduce([np.isfinite(M[:, i]) for i in i_req])
    return ok_vis & have_all, i_req

def compute_avw_ndi_qwip(M, wls_vis, evaluable_mask, poly_p=POLY_P):
    w = wls_vis.astype(float).reshape(1, -1)
    avw = np.full(M.shape[0], np.nan, float)
    ndi  = np.full(M.shape[0], np.nan, float)

    idx492, idx665 = find_indices(REQ_WLS, wls_vis)
    E = evaluable_mask

    # AVW = sum(R) / sum(R / λ)
    num = np.nansum(M[E], axis=1)
    den = np.nansum(M[E] / w, axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        avw[E] = num / den

    # NDI = (R665 - R492) / (R665 + R492)
    num = M[E, idx665] - M[E, idx492]
    den = M[E, idx665] + M[E, idx492]
    ndi[E] = np.divide(num, den, out=np.full_like(num, np.nan), where=den != 0)

    qwip = np.polyval(poly_p, avw) - ndi
    return avw, ndi, qwip

def split_flags(qwip, evaluable_mask, thresh=QWIP_THRESH):
    flags_missing  = ~evaluable_mask
    flags_spectral = np.zeros_like(evaluable_mask, dtype=bool)
    flags_spectral[evaluable_mask] = np.abs(qwip[evaluable_mask]) >= thresh
    status = np.where(flags_missing, "missing_data",
             np.where(flags_spectral, "spectral_fail", "pass"))
    return flags_missing, flags_spectral, status

def summarize(status, flags_missing, flags_spectral, label):
    total = len(status)
    n_eval = int((~flags_missing).sum())
    n_miss = int(flags_missing.sum())
    n_spec = int(flags_spectral.sum())
    eval_cov = 100.0 * n_eval / total if total else 0.0
    spec_fail = 100.0 * n_spec / n_eval if n_eval else 0.0
    miss_rate = 100.0 * n_miss / total if total else 0.0
    print(f"{label} — preview:")
    print(pd.Series(status).head(10).to_string(index=False))
    print(f"\n{label} — meaningful metrics")
    print(f"- Evaluable coverage: {n_eval}/{total} = {eval_cov:.1f}%")
    print(f"- Spectral-fail rate (among evaluable): {n_spec}/{n_eval} = {spec_fail:.1f}%")
    print(f"- Missing-data rate: {n_miss}/{total} = {miss_rate:.1f}%")

def make_report(df_src, wl_start, wl_step, label, repair_max_gap=None):
    # 1) Map to wavelength columns and extract VIS
    df_wl, all_wls = map_bins_to_wls(df_src, wl_start, wl_step)
    rrs_vis, wls_vis = extract_vis(df_wl, all_wls)

    # 2) Optional minimal repair (AFTER), else raw (BEFORE)
    M = rrs_vis.to_numpy(float)
    if repair_max_gap is not None:
        M = minimal_repair_short_gaps(M, repair_max_gap)

    # 3) Evaluable mask + metrics
    evaluable_mask, _ = build_evaluable_mask(M, wls_vis, MIN_VALID_RATIO, REQ_WLS)
    avw, ndi, qwip = compute_avw_ndi_qwip(M, wls_vis, evaluable_mask)
    flags_missing, flags_spectral, status = split_flags(qwip, evaluable_mask, QWIP_THRESH)

    report = pd.DataFrame({
        "timestamp": df_src["timestamp"].values,
        "AVW": avw, "NDI": ndi, "QWIP": qwip,
        "status": status
    })
    summarize(status, flags_missing, flags_spectral, label)
    return report

# ===============================
# wavelength grid + helpers
# ===============================

WL = np.arange(400, 779, 1)          # 379 bins
WL_ix = {w:i for i,w in enumerate(WL)}

def to_arr(x):
    # handle already-numpy, list-of-floats, or stringified lists
    if isinstance(x, np.ndarray): return x
    if isinstance(x, list): return np.array(x, dtype=float)
    if isinstance(x, str):
        x = x.strip().replace('[','').replace(']','')
        if not x: return np.full(WL.shape, np.nan, dtype=float)
        return np.array([float(v) for v in x.split(',')], dtype=float)
    return np.full(WL.shape, np.nan, dtype=float)

def band(w):          # nearest wavelength index
    w = int(round(w))
    return WL_ix.get(w, None)

def band_mean(arr, w0, w1):
    i0, i1 = band(w0), band(w1)
    if i0 is None or i1 is None: return np.nan
    if i0 > i1: i0, i1 = i1, i0
    seg = arr[i0:i1+1]
    if seg.size == 0: return np.nan
    m = np.nanmean(seg)
    return float(m)

def make_overlay_viewer(df_before, df_after, title="Temporal Despike (no imputation)"):
    # --- helpers ---
    def _ts_label(df, idx):
        if 'timestamp' in df.columns:
            try:
                return str(df.loc[df.index[idx], 'timestamp'])
            except Exception:
                return f"row {idx}"
        return f"row {idx}"

    # spectral columns: assume same cols, but be defensive
    cols_before = spectral_cols(df_before)
    cols_after  = spectral_cols(df_after)
    cols = [c for c in cols_before if c in cols_after]

    n_rows = min(len(df_before), len(df_after))

    out = Output()
    slider = IntSlider(
        min=0, max=n_rows-1, step=1, value=0,
        description='row_idx', continuous_update=False
    )

    btn_first = Button(description='⏮ First', tooltip='Go to first row (0)')
    btn_prev  = Button(description='◀ Prev',  tooltip='Previous row')
    btn_next  = Button(description='Next ▶',  tooltip='Next row')
    btn_last  = Button(description=f'Last ⏭', tooltip=f'Go to last row ({n_rows-1})')

    pos_label = Label(value=f"{slider.value} / {n_rows-1}")

    def _render(idx):
        out.clear_output(wait=True)
        with out:
            # y-values
            y_before = df_before.iloc[idx][cols].values.astype(float)
            y_after  = df_after.iloc[idx][cols].values.astype(float)

            x = np.arange(len(cols))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=y_before,
                mode='markers',
                name='Before'
            ))
            fig.add_trace(go.Scatter(
                x=x, y=y_after,
                mode='markers',
                name='After'
            ))

            ts = _ts_label(df_before, idx)
            fig.update_layout(
                title=f"{title} — {ts}",
                xaxis_title="Wavelength (nm)",
                yaxis_title="Reflectance",
                width=900,
                height=500
            )

            # IMPORTANT: use display(fig), NOT fig.show()
            display(fig)

        pos_label.value = f"{idx} / {n_rows-1}"

    def _on_slider_change(change):
        if change['name'] == 'value':
            _render(change['new'])

    def _clamp_set(val):
        slider.value = max(slider.min, min(slider.max, val))

    def _go_first(_): _clamp_set(slider.min)
    def _go_prev(_):  _clamp_set(slider.value - 1)
    def _go_next(_):  _clamp_set(slider.value + 1)
    def _go_last(_):  _clamp_set(slider.max)

    slider.observe(_on_slider_change, names='value')
    btn_first.on_click(_go_first)
    btn_prev.on_click(_go_prev)
    btn_next.on_click(_go_next)
    btn_last.on_click(_go_last)

    # initial draw
    _render(slider.value)
    controls = HBox([btn_first, btn_prev, slider, btn_next, btn_last, pos_label])
    display(VBox([controls, out]))