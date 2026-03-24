# =========================
# File: app_ecsr.py
# =========================
from __future__ import annotations

import re
import tempfile
import textwrap
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from ecsr_core import (
    Config,
    EcsrInterpResult,
    InterpQuickResult,
    build_longform_fuel_table,
    compute_doc_curve_pchip,
    compute_ecsr_band_interpolated,
    compute_quick_metrics_interpolated,
    default_config,
    interpolate_curve_knn_from_scenarios,
    run_pipeline,
    write_excel_results,
)

# ---- time sweep in app ----
SWEEP_MIN_APP = 100.0
SWEEP_MAX_APP = 5000.0
SWEEP_STEP_APP = 10.0

# ---- fuel sweep in app ----
FUEL_SWEEP_MIN_APP = 0.20
FUEL_SWEEP_MAX_APP = 3.00
FUEL_SWEEP_STEP_APP = 0.02

_BREAKPOINT_SPEED_TOL_KT = 1.0
_BREAKPOINT_DISTANCE_NM = 100.0

_GROUP_META: Dict[str, Tuple[str, str]] = {
    "ISA_C": ("ISA", "°C"),
    "WIND_kt": ("Vėjas", "kt"),
    "WEIGHT_kg": ("Masė", "kg"),
    "ZP_ft": ("Aukštis", "ft"),
}

# Breakpoint graphs meta
_BP_GRAPHS: Dict[str, Dict[str, Any]] = {
    "g4": {"x_col": "WIND_kt", "x_name_lt": "vėjo", "x_label": "Vėjo greitis skrydžio kryptimi (kt)"},
    "g5": {"x_col": "WEIGHT_kg", "x_name_lt": "masės", "x_label": "Masė (kg)"},
    "g6": {"x_col": "ZP_ft", "x_name_lt": "skrydžio aukščio", "x_label": "Skrydžio aukštis (ft)"},
    "g7": {"x_col": "ISA_C", "x_name_lt": "ISA nuokrypio", "x_label": "ISA nuokrypis (°C)"},
}
_BP_OTHER_COLS = ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt"]

BUILTIN_SCENARIOS_DIR = Path("data/scenarios")

# ------------------------- Upload helper (NEW) -------------------------


def _save_uploads_to_tempdir(uploads: List[Any]) -> Tuple[tempfile.TemporaryDirectory, Path, List[str]]:
    """
    Save Streamlit UploadedFile list into a temporary folder.
    Returns: (tmp_handle, tmp_path, saved_names)

    Keep tmp_handle alive while you need the files.
    """
    tmp = tempfile.TemporaryDirectory()
    tmp_path = Path(tmp.name)

    saved_names: List[str] = []
    for uf in uploads or []:
        name = Path(getattr(uf, "name", "")).name
        if not name:
            continue
        (tmp_path / name).write_bytes(uf.getbuffer())
        saved_names.append(name)

    return tmp, tmp_path, saved_names


# ------------------------- UI helpers -------------------------

def _text_with_stepper(label: str, *, key: str, step: float, placeholder: str = "") -> Optional[float]:
    """
    Text input that supports +/- step buttons safely.
    - Uses a separate internal widget key (key__txt)
    - Uses on_click callbacks to update state (safe for rapid clicks)
    Returns: None if empty, else float
    """
    txt_key = f"{key}__txt"

    def _parse(s: str) -> Optional[float]:
        s = (s or "").strip()
        if s == "":
            return None
        try:
            return float(s.replace(",", "."))
        except ValueError:
            return None

    def _bump(delta: float) -> None:
        cur = _parse(st.session_state.get(txt_key, ""))
        v = 0.0 if cur is None else float(cur)
        st.session_state[txt_key] = f"{v + delta:g}"

    c1, c2, c3 = st.columns([8, 1, 1], gap="small")

    with c1:
        st.text_input(label, key=txt_key, placeholder=placeholder)

    with c2:
        st.button("−", key=f"{key}_minus", on_click=_bump, args=(-float(step),))

    with c3:
        st.button("+", key=f"{key}_plus", on_click=_bump, args=(+float(step),))

    return _parse(st.session_state.get(txt_key, ""))

def _opt_float_text(label: str, *, key: str, placeholder: str = "", help: str = "") -> Optional[float]:
    """
    Text input that can be empty on first load.
    Returns None if empty, else float (supports 0, comma decimals).
    """
    raw = st.text_input(label, key=key, placeholder=placeholder, help=help).strip()
    if raw == "":
        return None
    try:
        return float(raw.replace(",", "."))
    except ValueError:
        st.error(f"Neteisinga reikšmė: „{label}“")
        return None

def _scenario_sort_key(name: str) -> tuple[int, str]:
    m = re.search(r"(\d+)", str(name))
    num = int(m.group(1)) if m else 10**12
    return (num, str(name))


def _scenario_lookup(scenarios: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for sc in scenarios:
        if sc.get("scenarioName") == name:
            return sc
    return None


def _show_fig(fig) -> None:
    import matplotlib.pyplot as plt

    st.pyplot(fig, clear_figure=False)
    plt.close(fig)


def _black_note(text: str) -> None:
    if not text:
        return
    st.markdown(
        f"<div style='color:inherit; font-size:16px; margin-top:6px;'>{text}</div>",
        unsafe_allow_html=True,
    )


def _fmt_num(v: float) -> str:
    if not np.isfinite(v):
        return "—"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:g}"

def _fmt_eur(v: float, *, decimals: int = 1) -> str:
    """
    Format EUR with comma decimals for LT UI.
    Example: 986.5 -> "986,5"
    """
    if not np.isfinite(v):
        return ""
    return f"{float(v):.{int(decimals)}f}".replace(".", ",")


def _group_label(group_col: str, group_val: float) -> str:
    name, unit = _GROUP_META.get(group_col, (group_col, ""))
    unit_txt = f" {unit}" if unit else ""
    return f"{name}={_fmt_num(float(group_val))}{unit_txt}"


def _conditions_sentence_from_row(row: pd.Series) -> str:
    parts: List[str] = []
    for col in ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt"]:
        if col not in row:
            continue
        v = pd.to_numeric(row.get(col), errors="coerce")
        if np.isfinite(v):
            name, unit = _GROUP_META.get(col, (col, ""))
            parts.append(f"{name} = {_fmt_num(float(v))} {unit}".strip())
    return "Pradinės sąlygos: " + (", ".join(parts) if parts else "—")


def _conditions_sentence_from_row_with_costs(row: pd.Series, cfg: Config) -> str:
    base = _conditions_sentence_from_row(row)

    parts: List[str] = []
    fp = float(getattr(cfg, "fuel_price_eur_per_kg", float("nan")))
    if np.isfinite(fp):
        parts.append(f"Degalų kaina = {fp:.2f} €/kg")

    tc = float(getattr(cfg, "time_cost_operational", float("nan")))
    if np.isfinite(tc):
        parts.append(f"Laiko sąnaudos = {tc:.0f} €/h.")

    mode = str(getattr(cfg, "breakpoint_saving_mode", "default")).strip().lower()
    if mode == "custom":
        thr = float(getattr(cfg, "breakpoint_saving_eur", float("nan")))
        if np.isfinite(thr):
            parts.append(f"Sutaupymas≥{thr:.0f} €/100NM")

    if not parts:
        return base
    return f"{base}. " + ", ".join(parts)


def _conditions_sentence_from_filters(
    fixed: Dict[str, Optional[float]],
    *,
    x_col: str,
    grouped_by: Optional[str],
) -> str:
    fixed_parts: List[str] = []
    for col in ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt"]:
        if col == x_col:
            continue
        val = fixed.get(col, None)
        if val is None:
            continue
        name, unit = _GROUP_META.get(col, (col, ""))
        fixed_parts.append(f"{name} = {_fmt_num(float(val))} {unit}".strip())

    base = "Pradinės sąlygos: "
    base += ", ".join(fixed_parts) if fixed_parts else "—"

    if grouped_by:
        gname, _ = _GROUP_META.get(grouped_by, (grouped_by, ""))
        base += f". Grafikas sugrupuotas pagal: {gname}."
    return base


# ------------------------- validation helpers -------------------------


def _numeric_bounds(df: pd.DataFrame, col: str) -> Optional[Tuple[float, float]]:
    if df is None or df.empty or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    return float(np.nanmin(vals)), float(np.nanmax(vals))


def _fmt_bounds(lo: float, hi: float, unit: str) -> str:
    unit_txt = f" {unit}".strip()
    if abs(lo - round(lo)) < 1e-9 and abs(hi - round(hi)) < 1e-9:
        return f"{int(round(lo))}–{int(round(hi))} {unit_txt}".strip()
    return f"{lo:g}–{hi:g} {unit_txt}".strip()


def _validate_value_in_bounds(
    *,
    label: str,
    value: float,
    bounds: Optional[Tuple[float, float]],
    unit: str = "",
) -> Optional[str]:
    if bounds is None or not np.isfinite(value):
        return "Nepakanka duomenų interpolacijai."
    lo, hi = bounds
    if value < lo or value > hi:
        return (
            f"Neteisinga įvestis: {label} = {_fmt_num(float(value))} {unit}".strip()
            + f". Galima riba: {_fmt_bounds(lo, hi, unit)}."
        )
    return None


def _validate_interp_inputs(
    ref_df: pd.DataFrame,
    *,
    fl_ft: float,
    weight_kg: float,
    isa_c: float,
    wind_kt: float,
) -> Optional[str]:
    checks = [
        ("Aukštis", fl_ft, _numeric_bounds(ref_df, "ZP_ft"), "ft"),
        ("Masė", weight_kg, _numeric_bounds(ref_df, "WEIGHT_kg"), "kg"),
        ("ISA", isa_c, _numeric_bounds(ref_df, "ISA_C"), "°C"),
        ("Vėjas", wind_kt, _numeric_bounds(ref_df, "WIND_kt"), "kt"),
    ]
    for label, value, bounds, unit in checks:
        msg = _validate_value_in_bounds(label=label, value=float(value), bounds=bounds, unit=unit)
        if msg:
            return msg
    return None


def _validate_sweep_input(value: float, *, lo: float, hi: float, label: str, unit: str) -> Optional[str]:
    if not np.isfinite(value):
        return f"Neteisinga įvestis: {label}."
    if value < lo or value > hi:
        return (
            f"Neteisinga įvestis: {label} = {_fmt_num(float(value))} {unit}. "
            f"Galima riba: {_fmt_bounds(float(lo), float(hi), unit)}."
        )
    return None


def _normalize_ui_error(exc: Exception) -> str:
    msg = str(exc).strip()
    if msg:
        return msg
    return "Nepakanka duomenų interpolacijai arba įvestis už leistinų ribų."


def _validate_breakpoint_fixed_inputs(
    summary_tbl: pd.DataFrame,
    *,
    fixed: Dict[str, Optional[float]],
    x_col: str,
) -> Optional[str]:
    for col in ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt"]:
        if col == x_col:
            continue
        val = fixed.get(col, None)
        if val is None:
            continue
        label, unit = _GROUP_META.get(col, (col, ""))
        msg = _validate_value_in_bounds(
            label=label,
            value=float(val),
            bounds=_numeric_bounds(summary_tbl, col),
            unit=unit,
        )
        if msg:
            return msg
    return None


# ------------------------- economical scenarios table -------------------------


def _scenario_trial_label(name: str) -> str:
    m = re.search(r"(\d+)", str(name))
    if m:
        return f"{int(m.group(1))} bandymas"
    return str(name)


def _build_economical_scenarios_table(summary_tbl: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if summary_tbl is None or summary_tbl.empty:
        return pd.DataFrame()

    need = [
        "ScenarioName",
        "V_ECSR_kt",
        "ECSR_low_kt",
        "ECSR_high_kt",
        "V_notch_kt",
        "DOCmin_EurPerNM",
        "DOCnotch_EurPerNM",
    ]
    if not set(need).issubset(summary_tbl.columns):
        return pd.DataFrame()

    df = summary_tbl[need].copy()
    for col in need[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    tol_kt = float(getattr(cfg, "breakpoint_speed_tol_kt", _BREAKPOINT_SPEED_TOL_KT))
    df = df.loc[np.isfinite(df["V_ECSR_kt"]) & np.isfinite(df["V_notch_kt"])].copy()
    if df.empty:
        return pd.DataFrame()

    df["DeltaV_kt"] = df["V_notch_kt"] - df["V_ECSR_kt"]
    df["DocDiff_EurPerNM"] = df["DOCnotch_EurPerNM"] - df["DOCmin_EurPerNM"]

    df = df.loc[df["DeltaV_kt"] >= tol_kt].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(by="ScenarioName", key=lambda s: s.map(_scenario_sort_key)).reset_index(drop=True)

    out = pd.DataFrame(
        {
            "Bandymas": df["ScenarioName"].map(_scenario_trial_label),
            "ECON (kt)": df["V_ECSR_kt"].round(0).astype(int),
            "ECSR": [
                f"{int(round(min(lo, hi)))}"
                if int(round(min(lo, hi))) == int(round(max(lo, hi)))
                else f"{int(round(min(lo, hi)))}–{int(round(max(lo, hi)))}"
                for lo, hi in zip(df["ECSR_low_kt"].tolist(), df["ECSR_high_kt"].tolist())
            ],
            "IASnotch (kt)": df["V_notch_kt"].round(0).astype(int),
            "ΔV (kt)": df["DeltaV_kt"].round(0).astype(int),
            "DOC ECON (EUR/NM)": df["DOCmin_EurPerNM"].map(lambda v: f"{float(v):.3f}" if np.isfinite(v) else "—"),
            "DOC IASnotch (EUR/NM)": df["DOCnotch_EurPerNM"].map(lambda v: f"{float(v):.3f}" if np.isfinite(v) else "—"),
            "DOC skirtumas (EUR/NM)": df["DocDiff_EurPerNM"].map(lambda v: f"{float(v):.3f}" if np.isfinite(v) else "—"),
        }
    )
    return out


# ------------------------- Result card -------------------------

def _result_card_html(value: str, unit: str, caption: str, *, max_width_px: int = 170, box_height_px: int = 42) -> str:
    safe_value = (value or "").strip()
    value_html = safe_value if safe_value else "&nbsp;"
    safe_unit = (unit or "").strip()

    unit_html = (
        f"<span class='cc-unit' style='margin-left:6px;line-height:1;'>{safe_unit}</span>"
        if safe_unit
        else ""
    )

    html = f"""
<style>
  /* Light mode defaults (iframe-local) */
  .cc-card {{
    border: 1px solid rgba(0,0,0,0.18);
    background: rgba(0,0,0,0.04);
    color: rgba(0,0,0,0.95);
  }}
  .cc-caption {{
    color: rgba(0,0,0,0.70);
  }}
  .cc-unit {{
    font-size: 12px;
    font-weight: 800;
    color: inherit;
    opacity: 0.85;
  }}
  .cc-value {{
    font-size: 22px;
    font-weight: 900;
    line-height: 1;
    color: inherit;
  }}

  /* System dark mode (works inside iframe) */
  @media (prefers-color-scheme: dark) {{
    .cc-card {{
      border: 1px solid rgba(255,255,255,0.22);
      background: rgba(255,255,255,0.08);
      color: rgba(255,255,255,0.96);
    }}
    .cc-caption {{
      color: rgba(255,255,255,0.78);
    }}
  }}
</style>

<div style="width:100%;display:flex;justify-content:center;">
  <div style="width:100%;max-width:{int(max_width_px)}px;">
    <div class="cc-card" style="
        border-radius:10px;
        padding:4px 8px;
        height:{int(box_height_px)}px;
        display:flex;
        justify-content:center;
        align-items:center;
        gap:4px;
    ">
      <span class="cc-value">{value_html}</span>
      {unit_html}
    </div>
    <div class="cc-caption" style="text-align:center;font-size:15px;margin-top:8px;">
      {caption}
    </div>
  </div>
</div>
"""
    return textwrap.dedent(html).strip()


def _render_result_card(value: str, unit: str, caption: str, *, max_width_px: int = 170, box_height_px: int = 42) -> None:
    components.html(
        _result_card_html(value, unit, caption, max_width_px=max_width_px, box_height_px=box_height_px),
        height=box_height_px + 52,
        scrolling=False,
    )
# ------------------------- plotting helpers -------------------------


def _mpl_academic_fig(figsize: Tuple[float, float] = (8.6, 5.2)):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=260)
    ax = fig.add_subplot(1, 1, 1)
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    return fig, ax


def _add_axis_arrows(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.annotate(
        "",
        xy=(1.02, 0.0),
        xytext=(0.0, 0.0),
        xycoords=("axes fraction", "axes fraction"),
        textcoords=("axes fraction", "axes fraction"),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "black"},
        clip_on=False,
    )
    ax.annotate(
        "",
        xy=(0.0, 1.02),
        xytext=(0.0, 0.0),
        xycoords=("axes fraction", "axes fraction"),
        textcoords=("axes fraction", "axes fraction"),
        arrowprops={"arrowstyle": "->", "linewidth": 1.2, "color": "black"},
        clip_on=False,
    )


def _annotate_tiny_above(
    ax,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    dx_pts: int = 0,
    dy_pts: int = 34,
) -> None:
    ha = "center" if dx_pts == 0 else ("left" if dx_pts > 0 else "right")
    va = "bottom" if dy_pts >= 0 else "top"
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx_pts, dy_pts),
        textcoords="offset points",
        ha=ha,
        va=va,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.8,
            "color": color,
            "shrinkA": 0,
            "shrinkB": 0,
            "connectionstyle": "arc3,rad=0.0",
        },
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
        color=color,
        clip_on=False,
    )


def _place_econ_annotation_inside(
    ax,
    *,
    x: float,
    y: float,
    text: str,
    prefer: str,
) -> None:
    if prefer not in {"left", "right"}:
        raise ValueError("prefer must be 'left' or 'right'")

    bbox = {"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.88}

    if prefer == "right":
        ax.annotate(
            text,
            xy=(x, y),
            xycoords="data",
            xytext=(60, -22),
            textcoords="offset points",
            ha="left",
            va="top",
            arrowprops={
                "arrowstyle": "->",
                "linewidth": 1.8,
                "color": "black",
                "shrinkA": 0,
                "shrinkB": 10,
                "connectionstyle": "arc3,rad=0.0",
            },
            bbox=bbox,
            color="black",
            clip_on=True,
            zorder=30,
        )
        return

    ax.annotate(
        "",
        xy=(x, y),
        xycoords="data",
        xytext=(-26, -34),
        textcoords="offset points",
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.8,
            "color": "black",
            "shrinkA": 0,
            "shrinkB": 10,
            "connectionstyle": "arc3,rad=0.0",
        },
        clip_on=True,
        zorder=29,
    )

    ax.annotate(
        text,
        xy=(x, y),
        xycoords="data",
        xytext=(-30, -34),
        textcoords="offset points",
        ha="right",
        va="top",
        bbox=bbox,
        color="black",
        clip_on=True,
        zorder=30,
    )


# ------------------------- Graphs 1-3 -------------------------


def _plot_doc_vs_ias(sc: Dict[str, Any], cfg: Config, time_cost_eur_per_hr: float, *, distance_nm: float = 1.0):
    cur = compute_doc_curve_pchip(sc, float(time_cost_eur_per_hr), cfg, ngrid=700)

    doc_grid_eur = cur["DOC_grid_per_nm"] * float(distance_nm)
    doc_raw_eur = cur["DOC_raw_per_nm"] * float(distance_nm)
    doc_opt_eur = float(cur["DOC_opt_per_nm"]) * float(distance_nm)
    doc_notch_eur = float(cur["DOC_notch_per_nm"]) * float(distance_nm)

    econ_kt = int(round(float(cur["IAS_opt"])))
    notch_kt = int(round(float(cur["IAS_notch"])))

    # IMPORTANT: if ECON is >= IASnotch - 1 kt, we treat it as "same" (single black marker)
    same = (float(cur["IAS_opt"]) >= (float(cur["IAS_notch"]) - 1.0))

    if same:
        econ_color = "black"
        notch_color = "black"
    else:
        econ_color = "orange"
        notch_color = "dodgerblue"

    fig, ax = _mpl_academic_fig()
    ax.plot(cur["IAS_grid"], doc_grid_eur, linewidth=2.2, color="darkred", label="DOC kreivė")
    ax.scatter(cur["IAS_raw"], doc_raw_eur, s=22, marker="o", color="darkred", label="_nolegend_")

    ax.relim()
    ax.autoscale_view()

    if same:
        # Show ONE point at IASnotch (black), like in scenario mode requirement
        x_same = float(cur["IAS_notch"])
        y_same = doc_notch_eur

        ax.scatter([x_same], [y_same], s=95, marker="x", color="black",
                   label=f"ECON / IASnotch ({notch_kt} kt)")
        _annotate_tiny_above(
            ax,
            x_same,
            y_same,
            f"ECON = IASnotch = {notch_kt} kt",
            color="black",
        )
    else:
        ax.scatter([float(cur["IAS_opt"])], [doc_opt_eur], s=95, marker="x", color=econ_color,
                   label=f"ECON ({econ_kt} kt)")
        _annotate_tiny_above(
            ax,
            float(cur["IAS_opt"]),
            doc_opt_eur,
            f"ECON {econ_kt} kt",
            color=econ_color,
            dx_pts=-60,
            dy_pts=40,
        )

        ax.scatter([float(cur["IAS_notch"])], [doc_notch_eur], s=95, marker="x", color=notch_color,
                   label=f"IASnotch ({notch_kt} kt)")
        _annotate_tiny_above(
            ax,
            float(cur["IAS_notch"]),
            doc_notch_eur,
            f"IASnotch {notch_kt} kt",
            color=notch_color,
            dx_pts=14,
        )

    ax.set_title(f"DOC priklausomybė nuo IAS — {sc['scenarioName']}")
    ax.set_xlabel("IAS (kt)")
    ax.set_ylabel("DOC (EUR)")
    _add_axis_arrows(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig

# ------------------------- Graph 2/3 helpers (unchanged) -------------------------


def _plot_econ_vs_time_cost(longform_tbl: pd.DataFrame, summary_tbl: pd.DataFrame, scenario_name: str, *, tc_operational: float):
    fig, ax = _mpl_academic_fig(figsize=(9.0, 4.8))

    lf = longform_tbl.loc[longform_tbl["ScenarioName"] == scenario_name].copy()
    if lf.empty:
        raise ValueError("Nėra longform duomenų pasirinktam scenarijui.")

    lf["TIME_COST"] = pd.to_numeric(lf["TIME_COST"], errors="coerce")
    lf["IASopt"] = pd.to_numeric(lf["IASopt"], errors="coerce")
    lf = lf.loc[np.isfinite(lf["TIME_COST"]) & np.isfinite(lf["IASopt"])].sort_values("TIME_COST")
    if lf.shape[0] < 2:
        raise ValueError("Per mažai taškų ECON kreivei (reikia bent 2).")

    x = lf["TIME_COST"].to_numpy(float)
    y = lf["IASopt"].to_numpy(float)

    c_curve = "darkred"
    c_be = "purple"
    c_in = "orange"

    ax.plot(x, y, linewidth=2.4, color=c_curve, label="ECON kreivė", zorder=2)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    ax.set_xlim(x_min, x_max)
    pad = 0.06 * (y_max - y_min) if y_max > y_min else 2.0
    ax.set_ylim(y_min - pad, y_max + pad)
    y_axis_bottom = float(ax.get_ylim()[0])

    row = summary_tbl.loc[summary_tbl["ScenarioName"] == scenario_name]
    be = float("nan")
    if not row.empty:
        be = float(pd.to_numeric(row["BreakEven_TIME_COST_EurPerHr"], errors="coerce").iloc[0])

    def _econ_at(ct: float) -> float:
        return float(np.interp(float(ct), x, y))

    be_ok = np.isfinite(be) and (x_min <= be <= x_max)
    in_ok = np.isfinite(tc_operational)

    if be_ok and in_ok and abs(float(be) - float(tc_operational)) < 1e-9:
        x0 = float(be)
        y0 = _econ_at(x0)
        ax.plot(
            [x0, x0],
            [y_axis_bottom, y0],
            linewidth=2.0,
            linestyle="--",
            color=c_in,
            label=f"Laiko sąnaudų įvestis / lūžio taškas ({x0:.0f} €/h)",
            zorder=1,
        )
        ax.scatter([x0], [y0], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{x0:.0f} €/h ({int(round(y0))} kt)", zorder=20)
        _place_econ_annotation_inside(ax, x=x0, y=y0, text=f"{int(round(y0))} kt", prefer="right")
    else:
        if be_ok:
            y_at_be = _econ_at(float(be))
            ax.plot([be, be], [y_axis_bottom, y_at_be], linewidth=2.0, linestyle="--", color=c_be, label=f"Laiko sąnaudų lūžio taškas ({float(be):.0f} €/h)", zorder=1)
            ax.scatter([be], [y_at_be], s=52, marker="o", color=c_be, label="_nolegend_", zorder=18)

        if in_ok:
            tc_operational = float(tc_operational)
            econ_y = _econ_at(tc_operational)
            ax.plot([tc_operational, tc_operational], [y_axis_bottom, econ_y], linewidth=2.0, linestyle="--", color=c_in, label=f"Laiko sąnaudų įvestis ({tc_operational:.0f} €/h)", zorder=1)
            ax.scatter([tc_operational], [econ_y], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{tc_operational:.0f} €/h ({int(round(econ_y))} kt)", zorder=20)
            _place_econ_annotation_inside(ax, x=tc_operational, y=econ_y, text=f"{int(round(econ_y))} kt", prefer="right")

    ax.set_title(f"ECON priklausomybė nuo laiko sąnaudų — {scenario_name}", pad=22)
    ax.set_xlabel("Laiko sąnaudos (€/h)")
    ax.set_ylabel("ECON (kt)")
    _add_axis_arrows(ax)
    ax.legend(loc="best")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def _plot_econ_vs_fuel_price(
    scenarios: List[Dict[str, Any]],
    summary_tbl: pd.DataFrame,
    scenario_name: str,
    *,
    fuel_price_operational: float,
):
    fig, ax = _mpl_academic_fig(figsize=(9.0, 4.8))

    sc = _scenario_lookup(scenarios, scenario_name)
    if not sc:
        raise ValueError("Nerastas scenarijus fuel sweep duomenims.")

    fp = np.asarray(sc.get("fuelPriceVec", []), float).reshape(-1)
    econ = np.asarray(sc.get("IAS_opt_kt_fp", []), float).reshape(-1)
    ok = np.isfinite(fp) & np.isfinite(econ)
    fp = fp[ok]
    econ = econ[ok]
    if fp.size < 2:
        raise ValueError("Per mažai taškų ECON kreivei pagal degalų kainą (reikia bent 2).")

    order = np.argsort(fp, kind="mergesort")
    fp = fp[order]
    econ = econ[order]

    ax.plot(fp, econ, linewidth=2.4, color="darkred", label="ECON kreivė", zorder=2)

    x_min, x_max = float(np.nanmin(fp)), float(np.nanmax(fp))
    y_min, y_max = float(np.nanmin(econ)), float(np.nanmax(econ))
    ax.set_xlim(x_min, x_max)
    pad = 0.06 * (y_max - y_min) if y_max > y_min else 2.0
    ax.set_ylim(y_min - pad, y_max + pad)
    y_axis_bottom = float(ax.get_ylim()[0])

    row = summary_tbl.loc[summary_tbl["ScenarioName"] == scenario_name]
    be = float("nan")
    if not row.empty:
        be = float(pd.to_numeric(row["BreakEven_FUEL_PRICE_EurPerKg"], errors="coerce").iloc[0])

    def _econ_at(price: float) -> float:
        return float(np.interp(float(price), fp, econ))

    c_be = "purple"
    c_in = "orange"

    be_ok = np.isfinite(be) and (x_min <= be <= x_max)
    in_ok = np.isfinite(fuel_price_operational)

    if be_ok and in_ok and abs(float(be) - float(fuel_price_operational)) < 1e-12:
        x0 = float(be)
        y0 = _econ_at(x0)
        ax.plot([x0, x0], [y_axis_bottom, y0], linewidth=2.0, linestyle="--", color=c_in, label=f"Degalų įvestis / lūžio taškas ({x0:.2f} €/kg)", zorder=1)
        ax.scatter([x0], [y0], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{x0:.2f} €/kg ({int(round(y0))} kt)", zorder=20)
        _place_econ_annotation_inside(ax, x=x0, y=y0, text=f"{int(round(y0))} kt", prefer="left")
    else:
        if be_ok:
            y_at_be = _econ_at(float(be))
            ax.plot([be, be], [y_axis_bottom, y_at_be], linewidth=2.0, linestyle="--", color=c_be, label=f"Degalų lūžio taškas ({float(be):.2f} €/kg)", zorder=1)
            ax.scatter([be], [y_at_be], s=52, marker="o", color=c_be, label="_nolegend_", zorder=18)

        if in_ok:
            fp_in = float(fuel_price_operational)
            econ_y = _econ_at(fp_in)
            ax.plot([fp_in, fp_in], [y_axis_bottom, econ_y], linewidth=2.0, linestyle="--", color=c_in, label=f"Degalų įvestis ({fp_in:.2f} €/kg)", zorder=1)
            ax.scatter([fp_in], [econ_y], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{fp_in:.2f} €/kg ({int(round(econ_y))} kt)", zorder=20)
            _place_econ_annotation_inside(ax, x=fp_in, y=econ_y, text=f"{int(round(econ_y))} kt", prefer="left")

    ax.set_title(f"ECON priklausomybė nuo degalų kainos — {scenario_name}", pad=22)
    ax.set_xlabel("Degalų kaina (€/kg)")
    ax.set_ylabel("ECON (kt)")
    _add_axis_arrows(ax)
    ax.legend(loc="best")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig

def _plot_doc_vs_ias_input_knn(
    scenarios: List[Dict[str, Any]],
    summary_tbl: pd.DataFrame,
    *,
    fl_ft: float,
    wt_kg: float,
    isa_c: float,
    wind_kt: float,
    cfg: Config,
):
    # Validate 4D inputs exactly like other input-graphs
    msg = _validate_interp_inputs(summary_tbl, fl_ft=fl_ft, weight_kg=wt_kg, isa_c=isa_c, wind_kt=wind_kt)
    if msg:
        raise ValueError(msg)

    # Use precomputed per-scenario DOC(IAS) vectors (computed right after run_pipeline)
    usable = [
        sc for sc in scenarios
        if isinstance(sc.get("docIASVec"), np.ndarray) and isinstance(sc.get("docVec"), np.ndarray)
        and sc.get("docIASVec").size >= 2 and sc.get("docVec").size >= 2
    ]
    if len(usable) < 8:
        raise ValueError("Nepakanka scenarijų DOC(KNN) interpolacijai (reikia bent 8 su DOC vektoriais).")

    # Common IAS grid (no extrapolation outside available IAS ranges)
    ias_lo = min(float(np.nanmin(sc["docIASVec"])) for sc in usable)
    ias_hi = max(float(np.nanmax(sc["docIASVec"])) for sc in usable)
    x_grid = np.linspace(ias_lo, ias_hi, 500, dtype=float)

    # Same neighbor philosophy as Graph 2/3: nearest scenarios in (FL, WT, ISA, WIND)
    k_use = min(30, len(usable))
    min_nb = min(8, max(3, len(usable) // 2))

    y_grid, _diag = interpolate_curve_knn_from_scenarios(
        usable,
        fl_ft=float(fl_ft),
        weight_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
        x_grid=x_grid,
        x_vec_key="docIASVec",
        y_vec_key="docVec",
        k=k_use,
        power=2.0,
        min_neighbors=min_nb,
    )

    x = np.asarray(x_grid, float)
    y = np.asarray(y_grid, float)

    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < 50:
        raise ValueError("Nepakanka duomenų DOC kreivei nubraižyti (KNN).")

    x = x[ok]
    y = y[ok]

    # ECON from DOC minimum on interpolated curve
    j = int(np.nanargmin(y))
    v_opt = float(x[j])
    doc_opt = float(y[j])

    # IASnotch from interpolated quick metrics (same source as other input results)
    qres = compute_quick_metrics_interpolated(
        summary_tbl,
        fl_ft=float(fl_ft),
        weight_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
    )
    v_notch = float(qres.v_notch_kt)

    # Clamp display logic: if ECON is within 1 kt of IASnotch OR exceeds it -> show ONE black marker at IASnotch
    same = (v_opt >= (v_notch - 1.0))

    fig, ax = _mpl_academic_fig()
    ax.plot(x, y, linewidth=2.2, color="darkred", label="DOC kreivė (KNN)")

    econ_kt = int(round(v_opt))
    notch_kt = int(round(v_notch))

    if same:
        # One black marker at IASnotch
        y_notch = float(np.interp(v_notch, x, y))
        ax.scatter([v_notch], [y_notch], s=95, marker="x", color="black",
                   label=f"ECON / IASnotch ({notch_kt} kt)")
        _annotate_tiny_above(ax, v_notch, y_notch,
                             f"ECON = IASnotch = {notch_kt} kt",
                             color="black")
    else:
        # Two markers (ECON orange, IASnotch blue)
        ax.scatter([v_opt], [doc_opt], s=95, marker="x", color="orange",
                   label=f"ECON ({econ_kt} kt)")
        _annotate_tiny_above(ax, v_opt, doc_opt, f"ECON {econ_kt} kt",
                             color="orange", dx_pts=-60, dy_pts=40)

        y_notch = float(np.interp(v_notch, x, y))
        ax.scatter([v_notch], [y_notch], s=95, marker="x", color="dodgerblue",
                   label=f"IASnotch ({notch_kt} kt)")
        _annotate_tiny_above(ax, v_notch, y_notch, f"IASnotch {notch_kt} kt",
                             color="dodgerblue", dx_pts=14)

    ax.set_title("DOC priklausomybė nuo IAS — Įvestis (KNN)")
    ax.set_xlabel("IAS (kt)")
    ax.set_ylabel("DOC (EUR/NM)")
    _add_axis_arrows(ax)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _plot_econ_vs_time_cost_input_knn(
    scenarios: List[Dict[str, Any]],
    summary_tbl: pd.DataFrame,
    *,
    fl_ft: float,
    wt_kg: float,
    isa_c: float,
    wind_kt: float,
    cfg: Config,
):
    msg = _validate_interp_inputs(summary_tbl, fl_ft=fl_ft, weight_kg=wt_kg, isa_c=isa_c, wind_kt=wind_kt)
    if msg:
        raise ValueError(msg)

    tc_msg = _validate_sweep_input(
        float(cfg.time_cost_operational),
        lo=SWEEP_MIN_APP,
        hi=SWEEP_MAX_APP,
        label="Laiko sąnaudos",
        unit="€/h",
    )
    if tc_msg:
        raise ValueError(tc_msg)

    qres = compute_quick_metrics_interpolated(
        summary_tbl,
        fl_ft=float(fl_ft),
        weight_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
    )

    fig, ax = _mpl_academic_fig(figsize=(9.0, 4.8))
    x_grid = np.arange(SWEEP_MIN_APP, SWEEP_MAX_APP + 1e-9, SWEEP_STEP_APP, dtype=float)

    y_grid, _diag = interpolate_curve_knn_from_scenarios(
        scenarios,
        fl_ft=float(fl_ft),
        weight_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
        x_grid=x_grid,
        x_vec_key="timeCostVec",
        y_vec_key="IAS_opt_kt",
        k=min(30, len(scenarios)),
        min_neighbors=min(8, max(3, len(scenarios) // 2)),
        power=2.0,
    )

    x = x_grid
    y = np.asarray(y_grid, float)

    if y.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("Nepakanka duomenų ECON kreivei nubraižyti.")

    c_curve = "darkred"
    c_be = "purple"
    c_in = "orange"

    ax.plot(x, y, linewidth=2.4, color=c_curve, label="ECON kreivė", zorder=2)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    ax.set_xlim(x_min, x_max)
    pad = 0.06 * (y_max - y_min) if y_max > y_min else 2.0
    ax.set_ylim(y_min - pad, y_max + pad)
    y_axis_bottom = float(ax.get_ylim()[0])

    def _econ_at(ct: float) -> float:
        return float(np.interp(float(ct), x, y))

    be = float(qres.be_time_cost_eur_per_hr)
    tc_in = float(cfg.time_cost_operational)

    be_ok = np.isfinite(be) and (x_min <= be <= x_max)
    in_ok = np.isfinite(tc_in) and (x_min <= tc_in <= x_max)

    if be_ok and in_ok and abs(be - tc_in) < 1e-9:
        y0 = _econ_at(tc_in)
        ax.plot([tc_in, tc_in], [y_axis_bottom, y0], linewidth=2.0, linestyle="--", color=c_in, label=f"Laiko sąnaudų įvestis / lūžio taškas ({tc_in:.0f} €/h)", zorder=1)
        ax.scatter([tc_in], [y0], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{tc_in:.0f} €/h ({int(round(y0))} kt)", zorder=20)
        _place_econ_annotation_inside(ax, x=tc_in, y=y0, text=f"{int(round(y0))} kt", prefer="right")
    else:
        if be_ok:
            y_be = _econ_at(be)
            ax.plot([be, be], [y_axis_bottom, y_be], linewidth=2.0, linestyle="--", color=c_be, label=f"Laiko sąnaudų lūžio taškas ({be:.0f} €/h)", zorder=1)
            ax.scatter([be], [y_be], s=52, marker="o", color=c_be, label="_nolegend_", zorder=18)

        if in_ok:
            econ_y = _econ_at(tc_in)
            ax.plot([tc_in, tc_in], [y_axis_bottom, econ_y], linewidth=2.0, linestyle="--", color=c_in, label=f"Laiko sąnaudų įvestis ({tc_in:.0f} €/h)", zorder=1)
            ax.scatter([tc_in], [econ_y], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{tc_in:.0f} €/h ({int(round(econ_y))} kt)", zorder=20)
            _place_econ_annotation_inside(ax, x=tc_in, y=econ_y, text=f"{int(round(econ_y))} kt", prefer="right")

    ax.set_title("ECON priklausomybė nuo laiko sąnaudų", pad=22)
    ax.set_xlabel("Laiko sąnaudos (€/h)")
    ax.set_ylabel("ECON (kt)")
    _add_axis_arrows(ax)
    ax.legend(loc="best")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def _plot_econ_vs_fuel_price_input_knn(
    scenarios: List[Dict[str, Any]],
    summary_tbl: pd.DataFrame,
    *,
    fl_ft: float,
    wt_kg: float,
    isa_c: float,
    wind_kt: float,
    cfg: Config,
):
    msg = _validate_interp_inputs(summary_tbl, fl_ft=fl_ft, weight_kg=wt_kg, isa_c=isa_c, wind_kt=wind_kt)
    if msg:
        raise ValueError(msg)

    fp_msg = _validate_sweep_input(
        float(cfg.fuel_price_eur_per_kg),
        lo=FUEL_SWEEP_MIN_APP,
        hi=FUEL_SWEEP_MAX_APP,
        label="Degalų kaina",
        unit="€/kg",
    )
    if fp_msg:
        raise ValueError(fp_msg)

    qres = compute_quick_metrics_interpolated(
        summary_tbl,
        fl_ft=float(fl_ft),
        weight_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
    )

    fig, ax = _mpl_academic_fig(figsize=(9.0, 4.8))
    x_grid = np.arange(FUEL_SWEEP_MIN_APP, FUEL_SWEEP_MAX_APP + 1e-12, FUEL_SWEEP_STEP_APP, dtype=float)

    y_grid, _diag = interpolate_curve_knn_from_scenarios(
        scenarios,
        fl_ft=float(fl_ft),
        weight_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
        x_grid=x_grid,
        x_vec_key="fuelPriceVec",
        y_vec_key="IAS_opt_kt_fp",
        k=min(30, len(scenarios)),
        min_neighbors=min(8, max(3, len(scenarios) // 2)),
        power=2.0,
    )

    x = x_grid
    y = np.asarray(y_grid, float)

    if y.size < 2 or not np.all(np.isfinite(y)):
        raise ValueError("Nepakanka duomenų ECON kreivei nubraižyti.")

    c_curve = "darkred"
    c_be = "purple"
    c_in = "orange"

    ax.plot(x, y, linewidth=2.4, color=c_curve, label="ECON kreivė", zorder=2)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))
    ax.set_xlim(x_min, x_max)
    pad = 0.06 * (y_max - y_min) if y_max > y_min else 2.0
    ax.set_ylim(y_min - pad, y_max + pad)
    y_axis_bottom = float(ax.get_ylim()[0])

    def _econ_at(price: float) -> float:
        return float(np.interp(float(price), x, y))

    be = float(qres.be_fuel_price_eur_per_kg)
    fp_in = float(cfg.fuel_price_eur_per_kg)

    be_ok = np.isfinite(be) and (x_min <= be <= x_max)
    in_ok = np.isfinite(fp_in) and (x_min <= fp_in <= x_max)

    if be_ok and in_ok and abs(be - fp_in) < 1e-12:
        y0 = _econ_at(fp_in)
        ax.plot([fp_in, fp_in], [y_axis_bottom, y0], linewidth=2.0, linestyle="--", color=c_in, label=f"Degalų įvestis / lūžio taškas ({fp_in:.2f} €/kg)", zorder=1)
        ax.scatter([fp_in], [y0], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{fp_in:.2f} €/kg ({int(round(y0))} kt)", zorder=20)
        _place_econ_annotation_inside(ax, x=fp_in, y=y0, text=f"{int(round(y0))} kt", prefer="left")
    else:
        if be_ok:
            y_be = _econ_at(be)
            ax.plot([be, be], [y_axis_bottom, y_be], linewidth=2.0, linestyle="--", color=c_be, label=f"Degalų lūžio taškas ({be:.2f} €/kg)", zorder=1)
            ax.scatter([be], [y_be], s=52, marker="o", color=c_be, label="_nolegend_", zorder=18)

        if in_ok:
            econ_y = _econ_at(fp_in)
            ax.plot([fp_in, fp_in], [y_axis_bottom, econ_y], linewidth=2.0, linestyle="--", color=c_in, label=f"Degalų įvestis ({fp_in:.2f} €/kg)", zorder=1)
            ax.scatter([fp_in], [econ_y], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{fp_in:.2f} €/kg ({int(round(econ_y))} kt)", zorder=20)
            _place_econ_annotation_inside(ax, x=fp_in, y=econ_y, text=f"{int(round(econ_y))} kt", prefer="left")

    ax.set_title("ECON priklausomybė nuo degalų kainos", pad=22)
    ax.set_xlabel("Degalų kaina (€/kg)")
    ax.set_ylabel("ECON (kt)")
    _add_axis_arrows(ax)
    ax.legend(loc="best")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


# ------------------------- Breakpoint graphs helpers -------------------------


def _unique_sorted(series: pd.Series) -> List[float]:
    x = pd.to_numeric(series, errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    return sorted(set(float(v) for v in x))


def _nuniq(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    v = pd.to_numeric(df[col], errors="coerce")
    v = v[np.isfinite(v)]
    return int(v.nunique())


def _varying_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    out: List[str] = []
    for c in cols:
        if c not in df.columns:
            continue
        if _nuniq(df, c) > 1:
            out.append(c)
    return out


def _filter_summary_by_constants(
    summary_tbl: pd.DataFrame,
    *,
    zp_ft: Optional[float] = None,
    weight_kg: Optional[float] = None,
    isa_c: Optional[float] = None,
    wind_kt: Optional[float] = None,
    tol: float = 1e-6,
) -> pd.DataFrame:
    df = summary_tbl.copy()
    for col, val in [("ZP_ft", zp_ft), ("WEIGHT_kg", weight_kg), ("ISA_C", isa_c), ("WIND_kt", wind_kt)]:
        if val is None:
            continue
        if col not in df.columns:
            return df.iloc[0:0].copy()
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        df = df.loc[np.isfinite(v) & np.isclose(v, float(val), rtol=1e-6, atol=tol)]
    return df


def _validate_breakpoint_request(filtered: pd.DataFrame, *, x_col: str, candidates: List[str]) -> Tuple[bool, Optional[str], str]:
    others = [c for c in candidates if c != x_col]
    varying = _varying_cols(filtered, others)

    if len(varying) == 0:
        return True, None, ""
    if len(varying) == 1:
        return True, varying[0], ""

    pretty = ", ".join(_GROUP_META.get(c, (c, ""))[0] for c in varying)
    msg = f"Šiuo metu yra nefiksuoti keli parametrai: {pretty}. Prašome palikti nefiksuotą tik vieną papildomą parametrą."
    return False, None, msg


def _label_points_with_overlap_avoidance(
    ax,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    fmt: str,
    y_offset_pts: int = 6,
    fontsize: int = 7,
) -> None:
    """
    Simple/old style:
    Put every label above its dot with a short connector line.
    No overlap-avoidance logic (stable, predictable).
    """
    xs = np.asarray(xs, float).reshape(-1)
    ys = np.asarray(ys, float).reshape(-1)
    if xs.size == 0:
        return

    for x0, y0 in zip(xs.tolist(), ys.tolist()):
        if not (np.isfinite(x0) and np.isfinite(y0)):
            continue

        ax.annotate(
            fmt.format(y0),
            xy=(x0, y0),
            xycoords="data",
            xytext=(0, int(y_offset_pts)),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=int(fontsize),
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.6,
                "color": "black",
                "shrinkA": 6,
                "shrinkB": 6,
            },
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
            },
            clip_on=False,
            zorder=50,
        )


def _bbox_overlap_frac(a, b) -> float:
    """Return overlap area / min(area(a), area(b)) in display (pixel) coordinates."""
    x0 = max(a.x0, b.x0)
    y0 = max(a.y0, b.y0)
    x1 = min(a.x1, b.x1)
    y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    inter = (x1 - x0) * (y1 - y0)
    area_a = max((a.x1 - a.x0) * (a.y1 - a.y0), 1e-9)
    area_b = max((b.x1 - b.x0) * (b.y1 - b.y0), 1e-9)
    return float(inter / min(area_a, area_b))


def _label_points_global_dedup(
    ax,
    candidates: List[Tuple[float, float, str]],
    *,
    overlap_frac: float = 0.80,
    y_offset_pts: int = 6,
    fontsize: int = 7,
) -> None:
    """
    Place labels globally (across all plotted lines) and skip labels that would overlap
    already-placed labels by >= overlap_frac.
    """
    if not candidates:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    kept_bboxes = []

    # Choose who "wins" when labels clash:
    # current policy = higher y first. If you want "first group wins", we can change this.
    candidates_sorted = sorted(
        [(float(x), float(y), str(t)) for x, y, t in candidates if np.isfinite(x) and np.isfinite(y)],
        key=lambda r: r[1],
        reverse=True,
    )

    for x0, y0, txt in candidates_sorted:
        ann = ax.annotate(
            txt,
            xy=(x0, y0),
            xycoords="data",
            xytext=(0, int(y_offset_pts)),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=int(fontsize),
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.6,
                "color": "black",
                "shrinkA": 6,
                "shrinkB": 6,
            },
            bbox={
                "boxstyle": "round,pad=0.08",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
            },
            clip_on=False,
            zorder=50,
        )

        fig.canvas.draw()
        bb = ann.get_window_extent(renderer=renderer)

        too_close = any(_bbox_overlap_frac(bb, bb2) >= float(overlap_frac) for bb2 in kept_bboxes)
        if too_close:
            ann.remove()
        else:
            kept_bboxes.append(bb)


def _plot_breakpoint_vs_grouped(
    summary_tbl: pd.DataFrame,
    *,
    y_col: str,
    y_label: str,
    x_col: str,
    title: str,
    x_label: str,
    group_col: Optional[str],
    fmt: str,
) -> Any:
    fig, ax = _mpl_academic_fig()

    df = summary_tbl.copy()
    df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    keep = np.isfinite(df[y_col]) & np.isfinite(df[x_col])
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError("Nėra duomenų po filtravimo.")

    used_group = None
    if group_col and group_col in df.columns:
        df[group_col] = pd.to_numeric(df[group_col], errors="coerce")
        df = df.loc[np.isfinite(df[group_col])].copy()
        if not df.empty and int(df[group_col].nunique()) > 1:
            used_group = group_col

    if used_group:
        g = df.groupby([used_group, x_col], as_index=False)[y_col].median().rename(columns={y_col: "BE"})

        # Plot all groups first + collect all label candidates
        label_candidates: List[Tuple[float, float, str]] = []

        for grp_val, sub in g.groupby(used_group, sort=True):
            sub = sub.sort_values(x_col)
            xs = sub[x_col].to_numpy(float)
            ys = sub["BE"].to_numpy(float)

            ax.plot(xs, ys, linewidth=2.2, marker="o", markersize=4.8, label=_group_label(used_group, float(grp_val)))

            for x0, y0 in zip(xs.tolist(), ys.tolist()):
                if np.isfinite(x0) and np.isfinite(y0):
                    label_candidates.append((float(x0), float(y0), fmt.format(float(y0))))

        # GLOBAL label placement with overlap dedup across groups
        _label_points_global_dedup(ax, label_candidates, overlap_frac=0.80, y_offset_pts=6, fontsize=7)

        ax.legend(loc="best")
        y_for_limits = g["BE"].to_numpy(float)

    else:
        df = df.sort_values(x_col)
        xs = df[x_col].to_numpy(float)
        ys = df[y_col].to_numpy(float)

        ax.plot(xs, ys, linewidth=2.2, color="darkred")
        ax.scatter(xs, ys, s=26, color="darkred")

        # Single series: old labeling is fine (or you can also use global_dedup)
        _label_points_with_overlap_avoidance(ax, xs, ys, fmt=fmt, y_offset_pts=6, fontsize=7)

        y_for_limits = ys

    y_min = float(np.nanmin(y_for_limits))
    y_max = float(np.nanmax(y_for_limits))
    rng = y_max - y_min

    if rng <= 0 or not np.isfinite(rng):
        pad = max(0.15 * max(abs(y_min), 1.0), 0.20)
        ax.set_ylim(y_min - pad, y_max + pad)
    else:
        top_pad = max(0.25 * rng, 0.12 * max(abs(y_max), 1.0), 0.20)
        bot_pad = max(0.15 * rng, 0.06 * max(abs(y_min), 1.0), 0.10)
        ax.set_ylim(y_min - bot_pad, y_max + top_pad)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    _add_axis_arrows(ax)
    fig.tight_layout()
    return fig


def _nearest_available(summary_tbl: pd.DataFrame, col: str, target: float) -> float:
    vals = pd.to_numeric(summary_tbl.get(col, pd.Series([], dtype=float)), errors="coerce").to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError(f"Nėra reikšmių '{col}' stulpelyje.")
    uniq = np.array(sorted(set(float(v) for v in vals)), dtype=float)
    j = int(np.nanargmin(np.abs(uniq - float(target))))
    return float(uniq[j])


def _filter_summary_snapped(
    summary_tbl: pd.DataFrame,
    *,
    fixed: Dict[str, Optional[float]],
    x_col: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    msg = _validate_breakpoint_fixed_inputs(summary_tbl, fixed=fixed, x_col=x_col)
    if msg:
        raise ValueError(msg)

    snapped: Dict[str, float] = {}
    for col in ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt"]:
        if col == x_col:
            continue
        v = fixed.get(col, None)
        if v is None:
            continue
        snapped[col] = _nearest_available(summary_tbl, col, float(v))

    df = _filter_summary_by_constants(
        summary_tbl,
        zp_ft=snapped.get("ZP_ft"),
        weight_kg=snapped.get("WEIGHT_kg"),
        isa_c=snapped.get("ISA_C"),
        wind_kt=snapped.get("WIND_kt"),
    )
    return df, snapped


# ------------------------- Break-even formatting (UI) -------------------------


def _format_break_even_for_app_time(summary_tbl: pd.DataFrame, *, sweep_min: float, sweep_max: float) -> pd.Series:
    if summary_tbl.empty:
        return pd.Series(dtype="object")

    col = "BreakEven_TIME_COST_EurPerHr"
    s = pd.to_numeric(summary_tbl.get(col, pd.Series([np.nan] * len(summary_tbl))), errors="coerce").to_numpy(float)

    out: List[str] = []
    for v in s.tolist():
        if np.isfinite(v) and float(sweep_min) - 1e-9 <= float(v) <= float(sweep_max) + 1e-9:
            out.append(f"{int(round(float(v)))}")
        else:
            out.append("Nėra lūžio taško")
    return pd.Series(out, index=summary_tbl.index, dtype="object")


def _format_break_even_for_app_fuel(summary_tbl: pd.DataFrame, *, ceiling: float) -> pd.Series:
    if summary_tbl.empty:
        return pd.Series(dtype="object")

    col = "BreakEven_FUEL_PRICE_EurPerKg"
    s = pd.to_numeric(summary_tbl.get(col, pd.Series([np.nan] * len(summary_tbl))), errors="coerce").to_numpy(float)

    out: List[str] = []
    for v in s.tolist():
        if np.isfinite(v) and float(v) <= float(ceiling) + 1e-12:
            out.append(f"{float(v):.2f}")
        else:
            out.append("Nėra lūžio taško")
    return pd.Series(out, index=summary_tbl.index, dtype="object")


# ------------------------- state -------------------------


def _clear_excel_download_artifacts() -> None:
    """Clear any previously generated Excel download payload from session state."""
    st.session_state.pop("excel_bytes", None)
    st.session_state.pop("excel_name", None)


def _init_state() -> None:
    for k in ["fig_g1", "fig_g2", "fig_g3"]:
        st.session_state.setdefault(k, None)
    for k in ["cap_g1", "cap_g2", "cap_g3"]:
        st.session_state.setdefault(k, "")
    for k in ["open_g1", "open_g2", "open_g3"]:
        st.session_state.setdefault(k, False)
    for k in ["err_g1", "err_g2", "err_g3"]:
        st.session_state.setdefault(k, "")

    for gid in _BP_GRAPHS.keys():
        st.session_state.setdefault(f"fig_{gid}_time", None)
        st.session_state.setdefault(f"fig_{gid}_fuel", None)
        st.session_state.setdefault(f"cap_{gid}", "")
        st.session_state.setdefault(f"open_{gid}", False)
        st.session_state.setdefault(f"err_{gid}", "")

    st.session_state.setdefault("show_glossary", False)
    st.session_state.setdefault("excel_written_msg", "")

    st.session_state.setdefault("saving_custom_enabled", False)
    st.session_state.setdefault("saving_custom_value", 2.0)

    st.session_state.setdefault("ecsr_calc_fl_txt", "")
    st.session_state.setdefault("ecsr_calc_wt_txt", "")
    st.session_state.setdefault("ecsr_calc_isa_txt", "")
    st.session_state.setdefault("ecsr_calc_wind_txt", "")
    st.session_state.setdefault("ecsr_calc_last", None)
    st.session_state.setdefault("ecsr_calc_err", "")
    st.session_state.setdefault("ecsr_calc_fl", 0.0)
    st.session_state.setdefault("ecsr_calc_wt", 0.0)
    st.session_state.setdefault("ecsr_calc_isa", 0.0)
    st.session_state.setdefault("ecsr_calc_wind", 0.0)

    st.session_state.setdefault("in_fl_txt", "")
    st.session_state.setdefault("in_wt_txt", "")
    st.session_state.setdefault("in_isa_txt", "")
    st.session_state.setdefault("in_wind_txt", "")
    st.session_state.setdefault("in_fl", 0.0)
    st.session_state.setdefault("in_wt", 0.0)
    st.session_state.setdefault("in_isa", 0.0)
    st.session_state.setdefault("in_wind", 0.0)
    st.session_state.setdefault("in_metric", "Pasirinkite...")
    st.session_state.setdefault("in_dist_nm", 100.0)
    st.session_state.setdefault("quick_doc_dist_nm", 100.0)
    st.session_state.setdefault("mode", "Scenarijus") 
    st.session_state.setdefault("in_last_res", None)
    st.session_state.setdefault("in_err", "")

    st.session_state.setdefault("outliers_tbl", pd.DataFrame())

    # NEW: show uploaded filenames
    st.session_state.setdefault("uploaded_names", [])
    st.session_state.setdefault("input_root_label", "uploaded_files")


# ========================= UI =========================
st.set_page_config(layout="wide")

cfg0 = default_config()
_init_state()

# ========================= UI (SIDEBAR) =========================
with st.sidebar:
    st.header("Įvestys")

    data_source = st.radio(
        "Duomenų šaltinis",
        options=["Scenarijai, jau esantys sistemoje", "Įkelti naujus scenarijus"],
        index=0,
        help="Jeigu pasirinksite jau sistemoje esančius scenarijus, naudotojui nereikės įkelti naujų ir bus galima iškart matyti rezultatus.",
        key="data_source",
    )

    uploads: List[Any] = []
    if data_source == "Įkelti naujus scenarijus":
        st.markdown("Įkelkite scenarijų failus (CSV arba TXT). Galite įkelti kelis failus iš karto.")
        uploads = st.file_uploader(
            "Scenarijų failai",
            type=["csv", "txt"],
            accept_multiple_files=True,
            help="Pasirinkite *.csv / *.txt failus. Galite pažymėti kelis failus iš karto.",
            key="uploader_files",
        )

    mode = st.radio(
        "Peržiūros režimas",
        options=["Scenarijus", "Įvestis"],
        index=0,
        key="mode",  # keep selection across reruns
    )

    saving_custom_enabled = st.checkbox(
        "Taikyti sutaupymo vertę (€/100NM)",
        key="saving_custom_enabled",
    )

    if not bool(st.session_state.get("saving_custom_enabled", False)):
        st.session_state["saving_custom_value"] = float(st.session_state.get("saving_custom_value", 2.0) or 2.0)

    # ---- NO FORM HERE (buttons inside forms cause crashes) ----
    fuel_price = st.number_input(
        "Degalų kaina (€/kg)",
        min_value=0.0,
        step=0.05,
        format="%.2f",
        key="fuel_price_val",
    )

    tc_op = st.number_input(
        "Laiko sąnaudos (€/h)",
        min_value=0.0,
        step=100.0,
        format="%.0f",
        key="time_cost_val",
    )

    epsilon_pct = st.number_input(
        "ECSR epsilon (%)",
        min_value=0.0,
        step=0.1,
        format="%.1f",
        key="epsilon_pct_val",
    )

    saving_custom = float(st.session_state.get("saving_custom_value", 2.0))
    if bool(st.session_state.get("saving_custom_enabled", False)):
        saving_custom = st.number_input(
            "Sutaupymas (€/100NM)",
            min_value=0.0,
            step=1.0,
            key="saving_custom_value",
        )

    run_btn = st.button("Generuoti", type="primary", use_container_width=True)
        
if run_btn:
    st.session_state["excel_written_msg"] = ""
    _clear_excel_download_artifacts()
    st.session_state["outliers_tbl"] = pd.DataFrame()

    if float(fuel_price) <= 0.0 or float(tc_op) <= 0.0:
        st.error("Prašome įvesti teigiamas reikšmes: 'Degalų kaina' ir 'Laiko sąnaudos'.")
        st.stop()

    if float(epsilon_pct) < 0.0:
        st.error("ECSR epsilon negali būti neigiamas.")
        st.stop()

    cfg = replace(
        cfg0,
        fuel_price_eur_per_kg=float(fuel_price),
        time_cost_operational=float(tc_op),
        epsilon_break_even=float(float(epsilon_pct) / 100.0),
        time_cost_min=SWEEP_MIN_APP,
        time_cost_max=SWEEP_MAX_APP,
        time_cost_step=SWEEP_STEP_APP,
        fuel_price_min=FUEL_SWEEP_MIN_APP,
        fuel_price_max=FUEL_SWEEP_MAX_APP,
        fuel_price_step=FUEL_SWEEP_STEP_APP,
        breakpoint_speed_tol_kt=_BREAKPOINT_SPEED_TOL_KT,
        breakpoint_distance_nm=_BREAKPOINT_DISTANCE_NM,
        breakpoint_saving_mode=("custom" if saving_custom_enabled else "default"),
        breakpoint_saving_eur=float(saving_custom),
    )

    with st.spinner("Skaičiuojama..."):
        # -------------------------
        # Load scenarios
        # -------------------------
        if data_source == "Scenarijai, jau esantys sistemoje":
            root_dir = BUILTIN_SCENARIOS_DIR
            if not root_dir.exists() or not root_dir.is_dir():
                st.error(
                    "Rinkmenoje scenarijų nerasta. "
                    "Įsitikinkite, kad sistemoje yra įkeltų scenarijų."
                )
                st.stop()

            out_dir, _xlsx_path_unused, summary_tbl, longform_tbl, outliers_tbl, logs, scenarios = run_pipeline(
                root_dir=root_dir,
                cfg=cfg,
                return_scenarios=True,
            )
            st.session_state["uploaded_names"] = []
            st.session_state["input_root_label"] = str(root_dir)

        else:
            if not uploads:
                st.error("Prašome įkelti bent vieną scenarijaus failą (*.csv / *.txt).")
                st.stop()

            tmp, tmp_path, saved_names = _save_uploads_to_tempdir(list(uploads))
            try:
                if not saved_names:
                    st.error("Nepavyko įrašyti įkeltų failų. Bandykite dar kartą.")
                    st.stop()

                out_dir, _xlsx_path_unused, summary_tbl, longform_tbl, outliers_tbl, logs, scenarios = run_pipeline(
                    root_dir=tmp_path,
                    cfg=cfg,
                    return_scenarios=True,
                )
            finally:
                tmp.cleanup()

            st.session_state["uploaded_names"] = saved_names
            st.session_state["input_root_label"] = "uploaded_files"

        # -------------------------
        # Post-processing (common)
        # -------------------------
        fuel_longform_tbl = build_longform_fuel_table(scenarios)

        # Precompute per-scenario DOC(IAS) vectors for Graph 1 (Įvestis) kNN
        for sc in scenarios:
            try:
                cur = compute_doc_curve_pchip(sc, float(cfg.time_cost_operational), cfg, ngrid=400)
                sc["docIASVec"] = np.asarray(cur["IAS_grid"], float)
                sc["docVec"] = np.asarray(cur["DOC_grid_per_nm"], float)
            except Exception:
                sc.pop("docIASVec", None)
                sc.pop("docVec", None)

    # -------------------------
    # Save state (outside spinner)
    # -------------------------
    st.session_state["last_cfg"] = cfg
    st.session_state["summary_tbl"] = summary_tbl
    st.session_state["longform_tbl"] = longform_tbl
    st.session_state["fuel_longform_tbl"] = fuel_longform_tbl
    st.session_state["scenarios"] = scenarios
    st.session_state["generated_out_dir"] = str(out_dir)
    st.session_state["excel_written_msg"] = ""
    st.session_state["outliers_tbl"] = outliers_tbl if isinstance(outliers_tbl, pd.DataFrame) else pd.DataFrame()

    for k in ["fig_g1", "fig_g2", "fig_g3"]:
        st.session_state[k] = None
    for k in ["cap_g1", "cap_g2", "cap_g3"]:
        st.session_state[k] = ""
    for k in ["open_g1", "open_g2", "open_g3"]:
        st.session_state[k] = False
    for k in ["err_g1", "err_g2", "err_g3"]:
        st.session_state[k] = ""

    for gid in _BP_GRAPHS.keys():
        st.session_state[f"open_{gid}"] = False
        st.session_state[f"fig_{gid}_time"] = None
        st.session_state[f"fig_{gid}_fuel"] = None
        st.session_state[f"cap_{gid}"] = ""
        st.session_state[f"err_{gid}"] = ""

    st.session_state["ecsr_calc_last"] = None
    st.session_state["ecsr_calc_err"] = ""
    st.session_state["in_last_res"] = None
    st.session_state["in_err"] = ""

# ========================= MAIN VIEW =========================

if "summary_tbl" not in st.session_state:
    st.info("Pasirinkite duomenų šaltinį ir spauskite „Generuoti“.")
    st.stop()

summary_tbl: pd.DataFrame = st.session_state["summary_tbl"]
longform_tbl: pd.DataFrame = st.session_state["longform_tbl"]
scenarios: List[Dict[str, Any]] = st.session_state.get("scenarios", [])
cfg: Config = st.session_state.get("last_cfg", cfg0)

scenario_names = sorted([sc.get("scenarioName", "") for sc in scenarios if sc.get("scenarioName")], key=_scenario_sort_key)
if not scenario_names:
    st.warning("Nerasta scenarijų šiame įkeltų failų rinkinyje.")
    st.stop()

# ========================= ECSR skaičiuoklė (SCENARIUS MODE ONLY) =========================
if mode == "Scenarijus":
    st.header("ECSR skaičiuoklė")

    c1, c2, c3, c4, c5 = st.columns([1.2, 1.2, 1.2, 1.2, 1.0], gap="medium")
    with c1:
        fl_ft = st.number_input("Aukštis (ft)", step=500.0, key="ecsr_calc_fl")
    with c2:
        wt_kg = st.number_input("Masė (kg)", step=500.0, key="ecsr_calc_wt")
    with c3:
        isa_c = st.number_input("ISA nuokrypis (°C)", step=1.0, key="ecsr_calc_isa")
    with c4:
        wind_kt = st.number_input("Vėjo greitis skrydžio kryptimi (kt)", step=1.0, key="ecsr_calc_wind")
    with c5:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        calc_btn = st.button("Skaičiuoti", use_container_width=True, key="btn_ecsr_calc")

    if calc_btn:
        try:
            msg = _validate_interp_inputs(
                summary_tbl,
                fl_ft=float(fl_ft),
                weight_kg=float(wt_kg),
                isa_c=float(isa_c),
                wind_kt=float(wind_kt),
            )
            if msg:
                raise ValueError(msg)

            res = compute_ecsr_band_interpolated(
                summary_tbl,
                fl_ft=float(fl_ft),
                weight_kg=float(wt_kg),
                isa_c=float(isa_c),
                wind_kt=float(wind_kt),
            )

            st.session_state["ecsr_calc_last"] = res
            st.session_state["ecsr_calc_err"] = ""
        except Exception as e:
            st.session_state["ecsr_calc_last"] = None
            st.session_state["ecsr_calc_err"] = _normalize_ui_error(e)

    err = str(st.session_state.get("ecsr_calc_err", "")).strip()
    if err:
        st.error(err)

    res = st.session_state.get("ecsr_calc_last", None)
    if isinstance(res, EcsrInterpResult):
        lo_i = int(round(res.ecsr_low_kt))
        hi_i = int(round(res.ecsr_high_kt))
        rng_txt = f"{lo_i}" if lo_i == hi_i else f"{min(lo_i, hi_i)}–{max(lo_i, hi_i)}"
        _, mid, _ = st.columns([2.2, 1.2, 2.2], gap="large")
        with mid:
            _render_result_card(rng_txt, "kt", "ECSR")

    st.divider()

# ========================= Greita peržiūra =========================
st.header("Greita peržiūra")

metric_items: List[Tuple[str, str]] = [
    ("Pasirinkite...", "__NONE__"),
    ("ECON (kt)", "V_ECSR_kt"),
    ("IASnotch (kt)", "V_notch_kt"),
    ("ECSR (kt)", "__ECSR_RANGE__"),
    ("DOCmin_perNM (EUR/NM)", "DOCmin_EurPerNM"),
    ("DOCnotch_perNM (EUR/NM)", "DOCnotch_EurPerNM"),
    ("DOCmin_perX (EUR)", "__DOCMIN_PER_X__"),
    ("Laiko sąnaudų lūžio taškas (€/h)", "__BREAK_TIME__"),
    ("Degalų sąnaudų lūžio taškas (€/kg)", "__BREAK_FUEL__"),
]
metric_map: Dict[str, str] = {k: v for k, v in metric_items}


def _ecsr_range_str(lo: float, hi: float) -> str:
    if np.isfinite(lo) and np.isfinite(hi):
        lo_i = int(round(float(lo)))
        hi_i = int(round(float(hi)))
        if lo_i == hi_i:
            return f"{lo_i}"
        return f"{min(lo_i, hi_i)}–{max(lo_i, hi_i)}"
    return ""


def _show_result_card(value: str, unit: str) -> None:
    _, mid, _ = st.columns([2.2, 1.2, 2.2], gap="large")
    with mid:
        _render_result_card(value, unit, "Rezultatas")


if mode == "Scenarijus":
    top_l, top_r = st.columns(2, gap="large")
    with top_l:
        pick_scn = st.selectbox("Pasirinkite scenarijų", ["Pasirinkite..."] + scenario_names, index=0, key="quick_scn")
    with top_r:
        pick_metric_label = st.selectbox("Pasirinkite rodiklį", list(metric_map.keys()), index=0, key="quick_metric")

    show_placeholder = (pick_scn == "Pasirinkite..." or metric_map[pick_metric_label] == "__NONE__")

    distance_nm = 100.0
    is_docmin_per_x = (not show_placeholder) and (metric_map[pick_metric_label] == "__DOCMIN_PER_X__")
    if is_docmin_per_x:
        st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)
        _, mid, _ = st.columns([2.2, 1.2, 2.2], gap="large")
        with mid:
            distance_nm = st.number_input(
                "Atstumas (NM)",
                min_value=0.0,
                step=10.0,
                key="quick_doc_dist_nm",
            )

    shown_value = ""
    shown_unit = ""
    docmin_total = float("nan")
    docnotch_total = float("nan")
    diff_total = float("nan")
    quick_caption = ""

    if not show_placeholder:
        col_key = metric_map[pick_metric_label]
        row_df = summary_tbl.loc[summary_tbl["ScenarioName"] == pick_scn]
        if not row_df.empty:
            row = row_df.iloc[0]
            quick_caption = _conditions_sentence_from_row_with_costs(row, cfg)

            if col_key == "__ECSR_RANGE__":
                shown_value = _ecsr_range_str(
                    float(pd.to_numeric(row.get("ECSR_low_kt", np.nan), errors="coerce")),
                    float(pd.to_numeric(row.get("ECSR_high_kt", np.nan), errors="coerce")),
                )
                shown_unit = "kt" if shown_value else ""

            elif col_key == "__BREAK_TIME__":
                be_str = _format_break_even_for_app_time(summary_tbl=row_df, sweep_min=SWEEP_MIN_APP, sweep_max=SWEEP_MAX_APP)
                shown_value = str(be_str.iloc[0]) if len(be_str) else ""
                shown_unit = "€/h" if (shown_value and shown_value != "Nėra lūžio taško") else ""

            elif col_key == "__BREAK_FUEL__":
                be_str = _format_break_even_for_app_fuel(summary_tbl=row_df, ceiling=float(cfg.break_search.fuel_ceiling_eur_per_kg))
                shown_value = str(be_str.iloc[0]) if len(be_str) else ""
                shown_unit = "€/kg" if (shown_value and shown_value != "Nėra lūžio taško") else ""

            elif col_key == "__DOCMIN_PER_X__":
                v_min_per_nm = float(pd.to_numeric(row.get("DOCmin_EurPerNM", np.nan), errors="coerce"))
                v_notch_per_nm = float(pd.to_numeric(row.get("DOCnotch_EurPerNM", np.nan), errors="coerce"))
                dist = float(distance_nm)
                if np.isfinite(v_min_per_nm) and np.isfinite(dist):
                    docmin_total = float(v_min_per_nm) * dist
                if np.isfinite(v_notch_per_nm) and np.isfinite(dist):
                    docnotch_total = float(v_notch_per_nm) * dist
                if np.isfinite(docmin_total) and np.isfinite(docnotch_total):
                    diff_total = float(docnotch_total) - float(docmin_total)
                    diff_total = round(diff_total, 1)  
            else:
                val = float(pd.to_numeric(row.get(col_key, np.nan), errors="coerce"))
                if np.isfinite(val):
                    if "kt" in pick_metric_label:
                        shown_value = f"{int(round(val))}"
                        shown_unit = "kt"
                    elif "€/h" in pick_metric_label:
                        shown_value = f"{int(round(val))}"
                        shown_unit = "€/h"
                    elif "€/kg" in pick_metric_label:
                        shown_value = f"{val:.2f}"
                        shown_unit = "€/kg"
                    elif "EUR/NM" in pick_metric_label:
                        shown_value = f"{val:.3f}"   
                        shown_unit = "EUR/NM"
                    elif "EUR" in pick_metric_label:
                        shown_value = f"{val:.0f}"
                        shown_unit = "EUR"
                    else:
                        shown_value = f"{val:.3f}"

    st.markdown("<div style='height: 26px'></div>", unsafe_allow_html=True)

    if show_placeholder:
        _show_result_card("", "")
    else:
        if is_docmin_per_x:
            r1, r2, r3 = st.columns(3, gap="large")
            with r1:
                v = _fmt_eur(docmin_total, decimals=1) if np.isfinite(docmin_total) else ""
                _render_result_card(v, "EUR" if v else "", "DOCmin rezultatas", max_width_px=190)
            with r2:
                v = _fmt_eur(docnotch_total, decimals=1) if np.isfinite(docnotch_total) else ""
                _render_result_card(v, "EUR" if v else "", "DOCnotch rezultatas", max_width_px=190)
            with r3:
                v = _fmt_eur(diff_total, decimals=1) if np.isfinite(diff_total) else ""
                _render_result_card(v, "EUR" if v else "", "Skirtumas (DOCnotch − DOCmin)", max_width_px=220)
        else:
            _show_result_card(shown_value, shown_unit)

    st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
    _black_note(quick_caption)

else:
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        in_fl = st.number_input("Aukštis (ft)", step=500.0, key="in_fl")
    with c2:
        in_wt = st.number_input("Masė (kg)", step=500.0, key="in_wt")
    with c3:
        in_isa = st.number_input("ISA nuokrypis (°C)", step=1.0, key="in_isa")
    with c4:
        in_wind = st.number_input("Vėjo greitis skrydžio kryptimi (kt)", step=1.0, key="in_wind")

    pick_metric_label = st.selectbox("Pasirinkite rodiklį", list(metric_map.keys()), index=0, key="in_metric")
    col_key = metric_map[pick_metric_label]
    show_placeholder = (col_key == "__NONE__")

    distance_nm = float(st.session_state.get("in_dist_nm", 100.0))
    if not show_placeholder and col_key == "__DOCMIN_PER_X__":
        distance_nm = st.number_input(
            "Atstumas (NM)",
            min_value=0.0,
            step=10.0,
            key="in_dist_nm",
        )

    st.markdown("<div style='height: 6px'></div>", unsafe_allow_html=True)
    btn = st.button("Skaičiuoti", type="primary")

    if btn and not show_placeholder:
        try:
            msg = _validate_interp_inputs(summary_tbl, fl_ft=float(in_fl), weight_kg=float(in_wt), isa_c=float(in_isa), wind_kt=float(in_wind))
            if msg:
                raise ValueError(msg)

            res_in: InterpQuickResult = compute_quick_metrics_interpolated(
                summary_tbl,
                fl_ft=float(in_fl),
                weight_kg=float(in_wt),
                isa_c=float(in_isa),
                wind_kt=float(in_wind),
            )
            st.session_state["in_last_res"] = res_in
            st.session_state["in_err"] = ""
        except Exception as e:
            st.session_state["in_last_res"] = None
            st.session_state["in_err"] = _normalize_ui_error(e)

    err = str(st.session_state.get("in_err", "")).strip()
    if err:
        st.error(err)

    shown_value = ""
    shown_unit = ""
    docmin_total = float("nan")
    docnotch_total = float("nan")
    diff_total = float("nan")

    if not show_placeholder:
        res_in = st.session_state.get("in_last_res", None)
        if isinstance(res_in, InterpQuickResult):
            if col_key == "__ECSR_RANGE__":
                shown_value = _ecsr_range_str(res_in.ecsr_low_kt, res_in.ecsr_high_kt)
                shown_unit = "kt" if shown_value else ""
            elif col_key == "__BREAK_TIME__":
                v = float(res_in.be_time_cost_eur_per_hr)
                if np.isfinite(v) and (SWEEP_MIN_APP - 1e-9 <= v <= SWEEP_MAX_APP + 1e-9):
                    shown_value = f"{int(round(v))}"
                    shown_unit = "€/h"
                else:
                    shown_value = "Nėra lūžio taško"
            elif col_key == "__BREAK_FUEL__":
                v = float(res_in.be_fuel_price_eur_per_kg)
                if np.isfinite(v) and v <= float(cfg.break_search.fuel_ceiling_eur_per_kg) + 1e-12:
                    shown_value = f"{v:.2f}"
                    shown_unit = "€/kg"
                else:
                    shown_value = "Nėra lūžio taško"
            elif col_key == "__DOCMIN_PER_X__":
                dist = float(distance_nm)
                docmin_total = float(res_in.docmin_eur_per_nm) * dist if np.isfinite(res_in.docmin_eur_per_nm) else float("nan")
                docnotch_total = float(res_in.docnotch_eur_per_nm) * dist if np.isfinite(res_in.docnotch_eur_per_nm) else float("nan")
                if np.isfinite(docmin_total) and np.isfinite(docnotch_total):
                    diff_total = float(docnotch_total) - float(docmin_total)
                    diff_total = round(diff_total, 1)
                else:
                    diff_total = float("nan")
            else:
                mapping = {
                    "V_ECSR_kt": (res_in.v_ecsr_kt, "kt"),
                    "V_notch_kt": (res_in.v_notch_kt, "kt"),
                    "DOCmin_EurPerNM": (res_in.docmin_eur_per_nm, "EUR/NM"),
                    "DOCnotch_EurPerNM": (res_in.docnotch_eur_per_nm, "EUR/NM"),
                }
                if col_key in mapping:
                    val, unit = mapping[col_key]
                    if np.isfinite(val):
                        if unit == "kt":
                            shown_value = f"{int(round(float(val)))}"
                        elif unit == "EUR/NM":
                            shown_value = f"{float(val):.3f}"
                        else:
                            shown_value = f"{float(val):g}"
                        shown_unit = unit

    st.markdown("<div style='height: 26px'></div>", unsafe_allow_html=True)

    if show_placeholder:
        _show_result_card("", "")
    else:
        if col_key == "__DOCMIN_PER_X__":
            r1, r2, r3 = st.columns(3, gap="large")
            with r1:
                v = _fmt_eur(docmin_total, decimals=1) if np.isfinite(docmin_total) else ""
                _render_result_card(v, "EUR" if v else "", "DOCmin rezultatas", max_width_px=190)
            with r2:
                v = _fmt_eur(docnotch_total, decimals=1) if np.isfinite(docnotch_total) else ""
                _render_result_card(v, "EUR" if v else "", "DOCnotch rezultatas", max_width_px=190)
            with r3:
                v = _fmt_eur(diff_total, decimals=1) if np.isfinite(diff_total) else ""
                _render_result_card(v, "EUR" if v else "", "Skirtumas (DOCnotch − DOCmin)", max_width_px=220)
        else:
            _show_result_card(shown_value, shown_unit)

# ========================= Grafikai =========================
st.divider()
st.header("Grafikai")

if mode == "Scenarijus":
    with st.expander("Grafikas 1 — DOC (eur/nm) vs IAS (kt)", expanded=st.session_state["open_g1"]):
        col1, col2 = st.columns([3, 1], gap="large")
        with col1:
            g1_scn = st.selectbox("Scenarijus", scenario_names, key="g1_scn")
        with col2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.button("Generuoti grafiką", key="btn_g1s", use_container_width=True):
                st.session_state["open_g1"] = True
                try:
                    sc = _scenario_lookup(scenarios, g1_scn)
                    if sc is None:
                        raise ValueError("Nerastas pasirinktas scenarijus.")
                    st.session_state["fig_g1"] = _plot_doc_vs_ias(sc, cfg, float(cfg.time_cost_operational), distance_nm=1.0)
                    row0 = summary_tbl.loc[summary_tbl["ScenarioName"] == g1_scn]
                    st.session_state["cap_g1"] = _conditions_sentence_from_row_with_costs(row0.iloc[0], cfg) if not row0.empty else ""
                    st.session_state["err_g1"] = ""
                except Exception as e:
                    st.session_state["fig_g1"] = None
                    st.session_state["cap_g1"] = ""
                    st.session_state["err_g1"] = _normalize_ui_error(e)
                st.rerun()
        if st.session_state.get("err_g1"):
            st.error(st.session_state["err_g1"])
        if st.session_state["fig_g1"] is not None:
            _show_fig(st.session_state["fig_g1"])
            _black_note(st.session_state.get("cap_g1", ""))

    with st.expander("Grafikas 2 — ECON (kt) vs Laiko sąnaudos (eur/h)", expanded=st.session_state["open_g2"]):
        col1, col2 = st.columns([3, 1], gap="large")
        with col1:
            g2_scn = st.selectbox("Scenarijus", scenario_names, key="g2_scn")
        with col2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.button("Generuoti grafiką", key="btn_g2s", use_container_width=True):
                st.session_state["open_g2"] = True
                try:
                    st.session_state["fig_g2"] = _plot_econ_vs_time_cost(longform_tbl, summary_tbl, g2_scn, tc_operational=float(cfg.time_cost_operational))
                    row0 = summary_tbl.loc[summary_tbl["ScenarioName"] == g2_scn]
                    st.session_state["cap_g2"] = _conditions_sentence_from_row_with_costs(row0.iloc[0], cfg) if not row0.empty else ""
                    st.session_state["err_g2"] = ""
                except Exception as e:
                    st.session_state["fig_g2"] = None
                    st.session_state["cap_g2"] = ""
                    st.session_state["err_g2"] = _normalize_ui_error(e)
                st.rerun()
        if st.session_state.get("err_g2"):
            st.error(st.session_state["err_g2"])
        if st.session_state["fig_g2"] is not None:
            _show_fig(st.session_state["fig_g2"])
            _black_note(st.session_state.get("cap_g2", ""))

    with st.expander("Grafikas 3 — ECON (kt) vs Degalų kaina (eur/kg)", expanded=st.session_state["open_g3"]):
        col1, col2 = st.columns([3, 1], gap="large")
        with col1:
            g3_scn = st.selectbox("Scenarijus", scenario_names, key="g3_scn")
        with col2:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.button("Generuoti grafiką", key="btn_g3s", use_container_width=True):
                st.session_state["open_g3"] = True
                try:
                    st.session_state["fig_g3"] = _plot_econ_vs_fuel_price(scenarios, summary_tbl, g3_scn, fuel_price_operational=float(cfg.fuel_price_eur_per_kg))
                    row0 = summary_tbl.loc[summary_tbl["ScenarioName"] == g3_scn]
                    st.session_state["cap_g3"] = _conditions_sentence_from_row_with_costs(row0.iloc[0], cfg) if not row0.empty else ""
                    st.session_state["err_g3"] = ""
                except Exception as e:
                    st.session_state["fig_g3"] = None
                    st.session_state["cap_g3"] = ""
                    st.session_state["err_g3"] = _normalize_ui_error(e)
                st.rerun()
        if st.session_state.get("err_g3"):
            st.error(st.session_state["err_g3"])
        if st.session_state["fig_g3"] is not None:
            _show_fig(st.session_state["fig_g3"])
            _black_note(st.session_state.get("cap_g3", ""))

else:
    with st.expander("Grafikas 1 — DOC (eur/nm) vs IAS (kt)", expanded=st.session_state["open_g1"]):
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.0], gap="small")
        with c1:
            fl = st.number_input("Aukštis (ft)", value=float(st.session_state["in_fl"]), step=500.0, key="g1i_fl")
        with c2:
            wt = st.number_input("Masė (kg)", value=float(st.session_state["in_wt"]), step=500.0, key="g1i_wt")
        with c3:
            isa = st.number_input("ISA nuokrypis (°C)", value=float(st.session_state["in_isa"]), step=1.0, key="g1i_isa")
        with c4:
            wind = st.number_input("Vėjo greitis skrydžio kryptimi (kt)", value=float(st.session_state["in_wind"]), step=1.0, key="g1i_wind")
        with c5:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.button("Generuoti grafiką", key="btn_g1i", use_container_width=True):
                st.session_state["open_g1"] = True
                try:
                    st.session_state["fig_g1"] = _plot_doc_vs_ias_input_knn(
                        scenarios,
                        summary_tbl,
                        fl_ft=fl,
                        wt_kg=wt,
                        isa_c=isa,
                        wind_kt=wind,
                        cfg=cfg,
                    )
                    st.session_state["err_g1"] = ""
                except Exception as e:
                    st.session_state["fig_g1"] = None
                    st.session_state["err_g1"] = _normalize_ui_error(e)
                st.rerun()
        if st.session_state.get("err_g1"):
            st.error(st.session_state["err_g1"])
        if st.session_state["fig_g1"] is not None:
            _show_fig(st.session_state["fig_g1"])

    with st.expander("Grafikas 2 — ECON (kt) vs Laiko sąnaudos (eur/h)", expanded=st.session_state["open_g2"]):
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.0], gap="small")
        with c1:
            fl = st.number_input("Aukštis (ft)", value=float(st.session_state["in_fl"]), step=500.0, key="g2i_fl")
        with c2:
            wt = st.number_input("Masė (kg)", value=float(st.session_state["in_wt"]), step=500.0, key="g2i_wt")
        with c3:
            isa = st.number_input("ISA nuokrypis (°C)", value=float(st.session_state["in_isa"]), step=1.0, key="g2i_isa")
        with c4:
            wind = st.number_input("Vėjo greitis skrydžio kryptimi (kt)", value=float(st.session_state["in_wind"]), step=1.0, key="g2i_wind")
        with c5:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.button("Generuoti grafiką", key="btn_g2i", use_container_width=True):
                st.session_state["open_g2"] = True
                try:
                    st.session_state["fig_g2"] = _plot_econ_vs_time_cost_input_knn(
                        scenarios,
                        summary_tbl,
                        fl_ft=fl,
                        wt_kg=wt,
                        isa_c=isa,
                        wind_kt=wind,
                        cfg=cfg,
                    )
                    st.session_state["err_g2"] = ""
                except Exception as e:
                    st.session_state["fig_g2"] = None
                    st.session_state["err_g2"] = _normalize_ui_error(e)
                st.rerun()
        if st.session_state.get("err_g2"):
            st.error(st.session_state["err_g2"])
        if st.session_state["fig_g2"] is not None:
            _show_fig(st.session_state["fig_g2"])

    with st.expander("Grafikas 3 — ECON (kt) vs Degalų kaina (eur/kg)", expanded=st.session_state["open_g3"]):
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.0], gap="small")
        with c1:
            fl = st.number_input("Aukštis (ft)", value=float(st.session_state["in_fl"]), step=500.0, key="g3i_fl")
        with c2:
            wt = st.number_input("Masė (kg)", value=float(st.session_state["in_wt"]), step=500.0, key="g3i_wt")
        with c3:
            isa = st.number_input("ISA nuokrypis (°C)", value=float(st.session_state["in_isa"]), step=1.0, key="g3i_isa")
        with c4:
            wind = st.number_input("Vėjo greitis skrydžio kryptimi (kt)", value=float(st.session_state["in_wind"]), step=1.0, key="g3i_wind")
        with c5:
            st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
            if st.button("Generuoti grafiką", key="btn_g3i", use_container_width=True):
                st.session_state["open_g3"] = True
                try:
                    st.session_state["fig_g3"] = _plot_econ_vs_fuel_price_input_knn(
                        scenarios,
                        summary_tbl,
                        fl_ft=fl,
                        wt_kg=wt,
                        isa_c=isa,
                        wind_kt=wind,
                        cfg=cfg,
                    )
                    st.session_state["err_g3"] = ""
                except Exception as e:
                    st.session_state["fig_g3"] = None
                    st.session_state["err_g3"] = _normalize_ui_error(e)
                st.rerun()
        if st.session_state.get("err_g3"):
            st.error(st.session_state["err_g3"])
        if st.session_state["fig_g3"] is not None:
            _show_fig(st.session_state["fig_g3"])


# --- Breakpoint graphs 4-7 ---


def _bp_filter_ui_scenario(graph_id: str, *, summary_tbl0: pd.DataFrame) -> Dict[str, Optional[float]]:
    meta = _BP_GRAPHS[graph_id]
    x_col = meta["x_col"]
    fixed: Dict[str, Optional[float]] = {}

    cols = st.columns(3, gap="medium")
    other_cols = [c for c in _BP_OTHER_COLS if c != x_col]

    for i, col in enumerate(other_cols):
        label, unit = _GROUP_META.get(col, (col, ""))
        values = _unique_sorted(summary_tbl0[col]) if col in summary_tbl0.columns else []
        options = ["nefiksuoti"] + [f"{v:.0f}" for v in values] if values else ["nefiksuoti"]
        with cols[i % 3]:
            pick = st.selectbox(
                f"Fiksuoti {label} ({unit})" if unit else f"Fiksuoti {label}",
                options,
                key=f"{graph_id}_flt_{col}",
            )
        fixed[col] = None if pick == "nefiksuoti" else float(pick)

    fixed[x_col] = None
    return fixed


def _bp_filter_ui_input(graph_id: str) -> Dict[str, Optional[float]]:
    meta = _BP_GRAPHS[graph_id]
    x_col = meta["x_col"]
    fixed: Dict[str, Optional[float]] = {x_col: None}

    cols = st.columns(3, gap="medium")
    other_cols = [c for c in _BP_OTHER_COLS if c != x_col]
    defaults = {
        "ZP_ft": float(st.session_state.get("in_fl", 0.0)),
        "WEIGHT_kg": float(st.session_state.get("in_wt", 0.0)),
        "ISA_C": float(st.session_state.get("in_isa", 0.0)),
        "WIND_kt": float(st.session_state.get("in_wind", 0.0)),
    }

    for i, col in enumerate(other_cols):
        label, unit = _GROUP_META.get(col, (col, ""))
        with cols[i % 3]:
            unfixed = st.checkbox(
                f"Pažymėkite, jeigu norite „{label}“ vertės nefiksuoti",
                value=False,
                key=f"{graph_id}_unfix_{col}",
            )
            if unfixed:
                fixed[col] = None
            else:
                value = st.number_input(
                    f"{label} ({unit})" if unit else f"{label}",
                    value=float(defaults.get(col, 0.0)),
                    step=1.0 if col in {"ISA_C", "WIND_kt"} else 500.0,
                    key=f"{graph_id}_in_{col}",
                )
                fixed[col] = float(value)

    return fixed


for gid, meta in _BP_GRAPHS.items():
    x_col = meta["x_col"]
    x_label = meta["x_label"]
    x_name_lt = meta["x_name_lt"]

    exp_title = f"Grafikas {gid[-1]} — Lūžio taškai vs {x_label}"
    open_key = f"open_{gid}"
    fig_key_time = f"fig_{gid}_time"
    fig_key_fuel = f"fig_{gid}_fuel"
    cap_key = f"cap_{gid}"
    err_key = f"err_{gid}"

    with st.expander(exp_title, expanded=st.session_state.get(open_key, False)):
        if mode == "Scenarijus":
            fixed = _bp_filter_ui_scenario(gid, summary_tbl0=summary_tbl)
            filtered_local = _filter_summary_by_constants(
                summary_tbl,
                zp_ft=fixed.get("ZP_ft"),
                weight_kg=fixed.get("WEIGHT_kg"),
                isa_c=fixed.get("ISA_C"),
                wind_kt=fixed.get("WIND_kt"),
            )

            candidates = [c for c in _BP_OTHER_COLS if c != x_col]
            ok, group_col, msg = _validate_breakpoint_request(filtered_local, x_col=x_col, candidates=candidates)
            if not ok:
                st.error(msg)

            if st.button("Generuoti grafikus", key=f"btn_{gid}", disabled=not ok):
                st.session_state[open_key] = True
                try:
                    st.session_state[fig_key_time] = _plot_breakpoint_vs_grouped(
                        filtered_local,
                        y_col="BreakEven_TIME_COST_EurPerHr",
                        y_label="Laiko sąnaudos (€/h)",
                        x_col=x_col,
                        title=f"Laiko lūžio taško priklausomybė nuo {x_name_lt}",
                        x_label=x_label,
                        group_col=group_col,
                        fmt="{:.0f} €/h",
                    )

                    st.session_state[fig_key_fuel] = _plot_breakpoint_vs_grouped(
                        filtered_local,
                        y_col="BreakEven_FUEL_PRICE_EurPerKg",
                        y_label="Degalų kaina (€/kg)",
                        x_col=x_col,
                        title=f"Degalų lūžio taško priklausomybė nuo {x_name_lt}",
                        x_label=x_label,
                        group_col=group_col,
                        fmt="{:.2f} €/kg",
                    )

                    st.session_state[cap_key] = _conditions_sentence_from_filters(fixed, x_col=x_col, grouped_by=group_col)
                    st.session_state[err_key] = ""
                except Exception as e:
                    st.session_state[fig_key_time] = None
                    st.session_state[fig_key_fuel] = None
                    st.session_state[cap_key] = ""
                    st.session_state[err_key] = _normalize_ui_error(e)
                st.rerun()

            if st.session_state.get(err_key):
                st.error(st.session_state[err_key])
            if st.session_state.get(fig_key_time) is not None:
                _show_fig(st.session_state[fig_key_time])
            if st.session_state.get(fig_key_fuel) is not None:
                _show_fig(st.session_state[fig_key_fuel])
            if st.session_state.get(fig_key_time) is not None or st.session_state.get(fig_key_fuel) is not None:
                _black_note(st.session_state.get(cap_key, ""))

        else:
            fixed_in = _bp_filter_ui_input(gid)

            candidates = [c for c in _BP_OTHER_COLS if c != x_col]
            ok = True
            group_col = None
            msg = ""
            snapped: Dict[str, float] = {}

            try:
                filtered_local, snapped = _filter_summary_snapped(summary_tbl, fixed=fixed_in, x_col=x_col)
                if filtered_local.empty:
                    raise ValueError("Nėra duomenų pagal pasirinktas fiksuotas reikšmes.")
                ok, group_col, msg = _validate_breakpoint_request(filtered_local, x_col=x_col, candidates=candidates)
            except Exception as e:
                ok = False
                msg = _normalize_ui_error(e)
                filtered_local = summary_tbl.iloc[0:0].copy()

            if not ok and msg:
                st.error(msg)

            if st.button("Generuoti grafikus", key=f"btn_{gid}_input", disabled=not ok):
                st.session_state[open_key] = True
                try:
                    st.session_state[fig_key_time] = _plot_breakpoint_vs_grouped(
                        filtered_local,
                        y_col="BreakEven_TIME_COST_EurPerHr",
                        y_label="Laiko sąnaudos (€/h)",
                        x_col=x_col,
                        title=f"Laiko lūžio taško priklausomybė nuo {x_name_lt}",
                        x_label=x_label,
                        group_col=group_col,
                        fmt="{:.0f} €/h",
                    )
                    st.session_state[fig_key_fuel] = _plot_breakpoint_vs_grouped(
                        filtered_local,
                        y_col="BreakEven_FUEL_PRICE_EurPerKg",
                        y_label="Degalų kaina (€/kg)",
                        x_col=x_col,
                        title=f"Degalų lūžio taško priklausomybė nuo {x_name_lt}",
                        x_label=x_label,
                        group_col=group_col,
                        fmt="{:.2f} €/kg",
                    )
                    st.session_state[cap_key] = ""
                    st.session_state[err_key] = ""
                except Exception as e:
                    st.session_state[fig_key_time] = None
                    st.session_state[fig_key_fuel] = None
                    st.session_state[cap_key] = ""
                    st.session_state[err_key] = _normalize_ui_error(e)
                st.rerun()

            if st.session_state.get(err_key):
                st.error(st.session_state[err_key])
            if st.session_state.get(fig_key_time) is not None:
                _show_fig(st.session_state[fig_key_time])
            if st.session_state.get(fig_key_fuel) is not None:
                _show_fig(st.session_state[fig_key_fuel])

# ========================= Summary tables + download =========================
st.divider()
st.header("Ekonomiški scenarijai")

econ_tbl = _build_economical_scenarios_table(summary_tbl, cfg)
if econ_tbl.empty:
    st.info("Su pasirinktais parametrais, ekonomiškų scenarijų neegzistuoja (ECON ~= IASnotch).")
else:
    st.dataframe(econ_tbl, use_container_width=True)

st.divider()
st.header("Bendra rezultatų lentelė")

display_tbl = summary_tbl.copy()
display_tbl = display_tbl.sort_values(by="ScenarioName", key=lambda s: s.map(_scenario_sort_key)).reset_index(drop=True)
display_tbl["ScenarioName"] = display_tbl["ScenarioName"].map(_scenario_trial_label)
display_tbl = display_tbl.rename(columns={"ScenarioName": "Bandymas"})

display_tbl["Laiko sąnaudų lūžio taškas (€/h)"] = _format_break_even_for_app_time(
    summary_tbl=summary_tbl,
    sweep_min=SWEEP_MIN_APP,
    sweep_max=SWEEP_MAX_APP,
)
display_tbl["Degalų sąnaudų lūžio taškas (€/kg)"] = _format_break_even_for_app_fuel(
    summary_tbl=summary_tbl,
    ceiling=float(cfg.break_search.fuel_ceiling_eur_per_kg),
)

display_tbl = display_tbl.drop(
    columns=["BreakEven_TIME_COST_EurPerHr", "BreakEven_FUEL_PRICE_EurPerKg"],
    errors="ignore",
)

st.dataframe(display_tbl, use_container_width=True)

col_spacer, col_btn, col_gloss = st.columns([6, 2, 2], gap="large")

with col_btn:
    download_name = "ECSR_results.xlsx"

    if st.button("Parsisiųsti Excel", use_container_width=True):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)

                run_info = {
                    "input_root_dir": str(st.session_state.get("input_root_label", "uploaded_files")),
                    "output_dir": str(tmp_path),
                    "n_scenarios_ok": len(scenarios),
                    "fuel_price_eur_per_kg": cfg.fuel_price_eur_per_kg,
                    "time_cost_operational": cfg.time_cost_operational,
                }

                outlier_rows = st.session_state.get("outliers_tbl", pd.DataFrame())
                if not isinstance(outlier_rows, pd.DataFrame):
                    outlier_rows = pd.DataFrame()

                temp_xlsx_path = write_excel_results(
                    out_dir=tmp_path,
                    summary_tbl=summary_tbl,
                    longform_tbl=longform_tbl,
                    outlier_rows=outlier_rows,
                    cfg=cfg,
                    run_info=run_info,
                )

                with open(temp_xlsx_path, "rb") as f:
                    st.session_state["excel_bytes"] = f.read()
                st.session_state["excel_name"] = temp_xlsx_path.name

        except Exception as e:
            st.error(f"Klaida generuojant Excel: {e}")

    if "excel_bytes" in st.session_state:
        st.download_button(
            "Parsisiųsti Excel",
            data=st.session_state["excel_bytes"],
            file_name=st.session_state.get("excel_name", download_name),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

with col_gloss:
    if st.button("Trumpinių paaiškinimai", use_container_width=True):
        st.session_state["show_glossary"] = not st.session_state["show_glossary"]

if st.session_state["show_glossary"]:
    with st.expander("Trumpinių paaiškinimai", expanded=True):
        st.markdown(
            """
**IAS** – nurodomasis oro greitis, **kt**  
**TAS** – tikrasis oro greitis, **kt**  
**WIND** – vėjo greitis skrydžio kryptimi (teigiamas – pavėjis, neigiamas – priešinis vėjas), **kt**

**DOC** – tiesioginės eksploatacinės sąnaudos, **EUR** / Gali būti matuojamos ir 1 jūrmylei - tuomet **EUR/NM**  
**DOCmin_perNM** – minimalios DOC sąnaudos per **1 jūrmylią**, **EUR/NM**  
**DOCnotch_perNM** – DOC sąnaudos per **1 jūrmylią**, kai galios svirtis yra fiksuotoje padėtyje (angl. notch) arba kai lėktuvas skrenda **IASnotch** greičiu, **EUR/NM**  
**DOCmin_perX** – minimalios DOC sąnaudos per **X jūrmylių**, **EUR**

**EPSILON** - nedidelė procentinė tolerancija, kurios dydis priklauso nuo oro linijų prioritetų ir kuri nusako, kokiu mastu sąnaudos gali padidėti virš **DOCmin**, **%**

**ECON** – ekonominis greitis, **kt**  
**IASnotch** – maksimalus kreiserinis greitis arba greitis prie tam tikrų sąlygų, kai galios svirtis yra fiksuotoje padėtyje, **kt**  
**ECSR** – ekonominis greičio diapazonas, gautas pritaikius epsilon įvertį **kt**
            """
        )
