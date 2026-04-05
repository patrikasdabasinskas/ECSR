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

_GROUP_META: Dict[str, Tuple[str, str]] = {
    "ISA_C": ("ISA", "°C"),
    "WIND_kt": ("Vėjas", "kt"),
    "WEIGHT_kg": ("Masė", "kg"),
    "ZP_ft": ("Aukštis", "ft"),
}

# Breakpoint graphs meta
_BP_GRAPHS: Dict[str, Dict[str, Any]] = {
    "g4": {"x_col": "WIND_kt", "x_name_lt": "vėjo", "x_label": "Vėjo dedamoji (kt)"},
    "g5": {"x_col": "WEIGHT_kg", "x_name_lt": "masės", "x_label": "Masė (kg)"},
    "g6": {"x_col": "ZP_ft", "x_name_lt": "skrydžio aukščio", "x_label": "Skrydžio aukštis (ft)"},
    "g7": {"x_col": "ISA_C", "x_name_lt": "ISA nuokrypio", "x_label": "ISA nuokrypis (°C)"},
}

_DOC_GRAPHS: Dict[str, Dict[str, Any]] = {
    "d1": {"x_col": "WIND_kt", "x_name_lt": "vėjo", "x_label": "Vėjo dedamoji (kt)"},
    "d2": {"x_col": "WEIGHT_kg", "x_name_lt": "masės", "x_label": "Masė (kg)"},
    "d3": {"x_col": "ZP_ft", "x_name_lt": "skrydžio aukščio", "x_label": "Skrydžio aukštis (ft)"},
    "d4": {"x_col": "ISA_C", "x_name_lt": "ISA nuokrypio", "x_label": "ISA nuokrypis (°C)"},
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

def _econ_from_docmin_with_notch_rule(v_docmin: float, v_notch: float, *, min_gap_kt: float = 1.0) -> float:
    """
    ECON rule:
    - start from true DOC-min speed
    - if DOC-min speed is less than 1 kt below IASnotch, treat ECON as IASnotch
    - otherwise keep DOC-min speed
    """
    if not np.isfinite(v_docmin):
        return float("nan")
    if not np.isfinite(v_notch):
        return float(v_docmin)

    if float(v_docmin) >= float(v_notch) - float(min_gap_kt):
        return float(v_notch)

    return float(v_docmin)

def _disp_econ_kt(v: float) -> Optional[int]:
    if not np.isfinite(v):
        return None
    return int(np.ceil(float(v)))


def _disp_notch_kt(v: float) -> Optional[int]:
    if not np.isfinite(v):
        return None
    return int(np.floor(float(v)))


def _disp_gap_kt(v_notch: float, v_econ: float) -> Optional[int]:
    notch_i = _disp_notch_kt(v_notch)
    econ_i = _disp_econ_kt(v_econ)
    if notch_i is None or econ_i is None:
        return None
    return int(notch_i - econ_i)


def _disp_speeds_differ(v_notch: float, v_econ: float, min_gap_kt: int = 1) -> bool:
    gap = _disp_gap_kt(v_notch, v_econ)
    return gap is not None and gap >= int(min_gap_kt)


def _raw_speeds_differ(v_notch: float, v_econ: float, min_gap_kt: float = 1.0) -> bool:
    if not (np.isfinite(v_notch) and np.isfinite(v_econ)):
        return False
    return (float(v_notch) - float(v_econ)) >= float(min_gap_kt)


def _fmt_speed_econ(v: float) -> str:
    x = _disp_econ_kt(v)
    return "" if x is None else str(x)


def _fmt_speed_notch(v: float) -> str:
    x = _disp_notch_kt(v)
    return "" if x is None else str(x)

def _safe_display_econ_kt(v_econ: float, v_notch: float) -> Optional[int]:
    econ_i = _disp_econ_kt(v_econ)
    notch_i = _disp_notch_kt(v_notch)
    if econ_i is None or notch_i is None:
        return None
    return min(econ_i, notch_i)


def _fmt_speed_econ_safe(v_econ: float, v_notch: float) -> str:
    x = _safe_display_econ_kt(v_econ, v_notch)
    return "" if x is None else str(x)

def _ecsr_range_str(
    lo: float,
    hi: float,
    v_notch: Optional[float] = None,
    v_econ: Optional[float] = None,
) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return ""

    lo_raw = float(min(lo, hi))
    hi_raw = float(max(lo, hi))

    lo_i = _disp_econ_kt(lo_raw)
    hi_i = _disp_notch_kt(hi_raw)

    if lo_i is None or hi_i is None:
        return ""

    notch_i = None
    if v_notch is not None and np.isfinite(v_notch):
        notch_i = _disp_notch_kt(float(v_notch))
        if notch_i is not None:
            hi_i = min(hi_i, notch_i)

    if v_econ is not None and np.isfinite(v_econ):
        econ_i = _disp_econ_kt(float(v_econ))
        if econ_i is not None:
            lo_i = min(lo_i, econ_i)
            hi_i = max(hi_i, econ_i)

    if notch_i is not None:
        hi_i = min(hi_i, notch_i)
        lo_i = min(lo_i, hi_i)

    if lo_i > hi_i:
        lo_i = hi_i

    return f"{lo_i}" if lo_i == hi_i else f"{lo_i}–{hi_i}"


def _scenario_docmin_econ_kt(
    sc: Dict[str, Any],
    cfg: Config,
) -> float:
    cur = compute_doc_curve_pchip(sc, float(cfg.time_cost_operational), cfg, ngrid=700)
    v_docmin = float(cur["IAS_opt"])
    v_notch = float(cur["IAS_notch"])
    return _econ_from_docmin_with_notch_rule(
        v_docmin,
        v_notch,
        min_gap_kt=float(cfg.breakpoint_speed_tol_kt),
    )


def _input_docmin_econ_kt(
    scenarios: List[Dict[str, Any]],
    summary_tbl: pd.DataFrame,
    cfg: Config,
    *,
    fl_ft: float,
    wt_kg: float,
    isa_c: float,
    wind_kt: float,
    fallback_v_econ: float,
    fallback_v_notch: float,
) -> float:
    try:
        doc_curve_res = _compute_input_doc_curve_knn(
            scenarios,
            summary_tbl,
            cfg,
            fl_ft=float(fl_ft),
            wt_kg=float(wt_kg),
            isa_c=float(isa_c),
            wind_kt=float(wind_kt),
        )
        v_docmin = float(doc_curve_res["IAS_opt"])
    except Exception:
        v_docmin = float(fallback_v_econ)

    return _econ_from_docmin_with_notch_rule(
        v_docmin,
        float(fallback_v_notch),
        min_gap_kt=float(cfg.breakpoint_speed_tol_kt),
    )


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
        parts.append(f"Laiko sąnaudos = {tc:.0f} €/h")

    mode = str(getattr(cfg, "breakpoint_saving_mode", "default")).strip().lower()

    if mode == "per_nm":
        thr = float(getattr(cfg, "breakpoint_saving_eur_per_nm", float("nan")))
        if np.isfinite(thr):
            parts.append(f"Sutaupymas ≥ {thr:.2f} €/NM")
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



# ------------------------- economical scenarios table -------------------------


def _scenario_trial_label(name: str) -> str:
    m = re.search(r"(\d+)", str(name))
    if m:
        return f"{int(m.group(1))} bandymas"
    return str(name)


def _build_economical_scenarios_table(
    summary_tbl: pd.DataFrame,
    cfg: Config,
    scenarios: List[Dict[str, Any]],
) -> pd.DataFrame:

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

    dist_cols: List[int] = [int(d) for d in getattr(cfg, "distances_nm", ()) if int(d) == 100]
    for d in dist_cols:
        need.extend(
            [
                f"DOCmin_{d}NM_EUR",
                f"DOCnotch_{d}NM_EUR",
            ]
        )

    need = [c for c in need if c in summary_tbl.columns]

    df = summary_tbl[need].copy()
    for col in need[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.loc[np.isfinite(df["V_ECSR_kt"]) & np.isfinite(df["V_notch_kt"])].copy()
    if df.empty:
        return pd.DataFrame()

    df["IASnotch_disp_kt"] = df["V_notch_kt"].map(_disp_notch_kt)

    econ_docmin_vals: List[Optional[int]] = []
    for _, row in df.iterrows():
        sc = _scenario_lookup(scenarios, str(row["ScenarioName"])) if "ScenarioName" in row else None
        if sc is not None:
            try:
                v_econ_docmin = _scenario_docmin_econ_kt(sc, cfg)
                v_notch = float(pd.to_numeric(row.get("V_notch_kt", np.nan), errors="coerce"))
                econ_docmin_vals.append(_safe_display_econ_kt(v_econ_docmin, v_notch))
            except Exception:
                econ_docmin_vals.append(None)
        else:
            econ_docmin_vals.append(None)

    df["ECON_safe_disp_kt"] = econ_docmin_vals
    df["DeltaV_disp_kt"] = df["IASnotch_disp_kt"] - df["ECON_safe_disp_kt"]
    df["DeltaV_raw_kt"] = pd.to_numeric(df["V_notch_kt"], errors="coerce") - pd.to_numeric(df["V_ECSR_kt"], errors="coerce")
    df["DocDiff_EurPerNM"] = df["DOCnotch_EurPerNM"] - df["DOCmin_EurPerNM"]

    for d in dist_cols:
        min_col = f"DOCmin_{d}NM_EUR"
        notch_col = f"DOCnotch_{d}NM_EUR"
        if min_col in df.columns and notch_col in df.columns:
            df[f"DocDiff_{d}NM_EUR"] = df[notch_col] - df[min_col]

    df = df.loc[(df["DeltaV_raw_kt"] >= float(cfg.breakpoint_speed_tol_kt)) & (df["DocDiff_EurPerNM"] > 0.0)].copy()
    if df.empty:
        return pd.DataFrame()

    df = df.sort_values(by="ScenarioName", key=lambda s: s.map(_scenario_sort_key)).reset_index(drop=True)

    out_data: Dict[str, Any] = {
        "Bandymas": df["ScenarioName"].map(_scenario_trial_label),
        "ECON (kt)": df["ECON_safe_disp_kt"],
        "ECSR": [
            _ecsr_range_str(lo, hi, vn, ve_docmin if ve_docmin is not None else np.nan)
            for lo, hi, vn, ve_docmin in zip(
                df["ECSR_low_kt"].tolist(),
                df["ECSR_high_kt"].tolist(),
                df["V_notch_kt"].tolist(),
                [
                    _scenario_docmin_econ_kt(_scenario_lookup(scenarios, str(sc_name)), cfg)
                    if _scenario_lookup(scenarios, str(sc_name)) is not None else np.nan
                    for sc_name in df["ScenarioName"].tolist()
                ],
            )
        ],
        "IASnotch (kt)": df["IASnotch_disp_kt"],
        "ΔV (kt)": df["DeltaV_raw_kt"].map(lambda v: f"{float(v):.1f}" if np.isfinite(v) else ""),
        "DOC ECON (EUR/NM)": df["DOCmin_EurPerNM"].map(lambda v: f"{float(v):.3f}" if np.isfinite(v) else ""),
        "DOC IASnotch (EUR/NM)": df["DOCnotch_EurPerNM"].map(lambda v: f"{float(v):.3f}" if np.isfinite(v) else ""),
        "DOC skirtumas (EUR/NM)": df["DocDiff_EurPerNM"].map(lambda v: f"{float(v):.3f}" if np.isfinite(v) else ""),
    }

    for d in dist_cols:
        min_col = f"DOCmin_{d}NM_EUR"
        notch_col = f"DOCnotch_{d}NM_EUR"
        diff_col = f"DocDiff_{d}NM_EUR"

        if min_col in df.columns:
            out_data[f"DOC ECON {d}NM (EUR)"] = df[min_col].map(lambda v: f"{float(v):.1f}" if np.isfinite(v) else "")
        if notch_col in df.columns:
            out_data[f"DOC IASnotch {d}NM (EUR)"] = df[notch_col].map(lambda v: f"{float(v):.1f}" if np.isfinite(v) else "")
        if diff_col in df.columns:
            out_data[f"DOC skirtumas {d}NM (EUR)"] = df[diff_col].map(lambda v: f"{float(v):.1f}" if np.isfinite(v) else "")

    return pd.DataFrame(out_data)


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
  :root {{
    --cc-border: rgba(0,0,0,0.18);
    --cc-bg: rgba(0,0,0,0.04);
    --cc-text: rgba(0,0,0,0.95);
    --cc-caption: rgba(0,0,0,0.70);
  }}

  .cc-card {{
    border: 1px solid var(--cc-border);
    background: var(--cc-bg);
    color: var(--cc-text);
  }}
  .cc-caption {{
    color: var(--cc-caption);
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
</style>

<script>
(function() {{
  function parseRgb(s) {{
    // "rgb(r,g,b)" or "rgba(r,g,b,a)"
    const m = (s || "").match(/rgba?\\((\\d+),\\s*(\\d+),\\s*(\\d+)/i);
    if (!m) return null;
    return [parseInt(m[1],10), parseInt(m[2],10), parseInt(m[3],10)];
  }}

  function luminance(rgb) {{
    const [r,g,b] = rgb.map(v => v/255);
    // perceived luminance
    return 0.2126*r + 0.7152*g + 0.0722*b;
  }}

  function pickTheme() {{
    try {{
      const pdoc = window.parent && window.parent.document ? window.parent.document : document;
      const body = pdoc.body;
      const bg = pdoc.defaultView.getComputedStyle(body).backgroundColor;
      const rgb = parseRgb(bg);
      if (!rgb) return;

      const lum = luminance(rgb);
      const isDark = lum < 0.5;

      const root = document.documentElement;
      if (isDark) {{
        root.style.setProperty("--cc-border", "rgba(255,255,255,0.22)");
        root.style.setProperty("--cc-bg", "rgba(255,255,255,0.08)");
        root.style.setProperty("--cc-text", "rgba(255,255,255,0.96)");
        root.style.setProperty("--cc-caption", "rgba(255,255,255,0.78)");
      }} else {{
        root.style.setProperty("--cc-border", "rgba(0,0,0,0.18)");
        root.style.setProperty("--cc-bg", "rgba(0,0,0,0.04)");
        root.style.setProperty("--cc-text", "rgba(0,0,0,0.95)");
        root.style.setProperty("--cc-caption", "rgba(0,0,0,0.70)");
      }}
    }} catch(e) {{}}
  }}

  pickTheme();
  // In case Streamlit toggles theme without full reload:
  setTimeout(pickTheme, 50);
  setTimeout(pickTheme, 250);
}})();
</script>

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

    econ_raw = float(cur["IAS_opt"])
    notch_raw = float(cur["IAS_notch"])

    econ_kt = _safe_display_econ_kt(econ_raw, notch_raw)
    notch_kt = _disp_notch_kt(notch_raw)

    same = not _disp_speeds_differ(notch_raw, econ_raw, min_gap_kt=int(round(float(cfg.breakpoint_speed_tol_kt))))

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
    ax.set_ylabel("DOC (EUR/NM)")
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
    v_notch_scn = float("nan")
    if not row.empty:
        be = float(pd.to_numeric(row["BreakEven_TIME_COST_EurPerHr"], errors="coerce").iloc[0])
        v_notch_scn = float(pd.to_numeric(row["V_notch_kt"], errors="coerce").iloc[0])

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
        ax.scatter([x0], [y0], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{x0:.0f} €/h ({_safe_display_econ_kt(y0, v_notch_scn)} kt)", zorder=20)
        _place_econ_annotation_inside(ax, x=x0, y=y0, text=f"{_safe_display_econ_kt(y0, v_notch_scn)} kt", prefer="right")
    else:
        if be_ok:
            y_at_be = _econ_at(float(be))
            ax.plot([be, be], [y_axis_bottom, y_at_be], linewidth=2.0, linestyle="--", color=c_be, label=f"Laiko sąnaudų lūžio taškas ({float(be):.0f} €/h)", zorder=1)
            ax.scatter([be], [y_at_be], s=52, marker="o", color=c_be, label="_nolegend_", zorder=18)

        if in_ok:
            tc_operational = float(tc_operational)
            econ_y = _econ_at(tc_operational)
            ax.plot([tc_operational, tc_operational], [y_axis_bottom, econ_y], linewidth=2.0, linestyle="--", color=c_in, label=f"Laiko sąnaudų įvestis ({tc_operational:.0f} €/h)", zorder=1)
            ax.scatter([tc_operational], [econ_y], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{tc_operational:.0f} €/h ({_safe_display_econ_kt(econ_y, v_notch_scn)} kt)", zorder=20)
            _place_econ_annotation_inside(ax, x=tc_operational, y=econ_y, text=f"{_safe_display_econ_kt(econ_y, v_notch_scn)} kt", prefer="right")

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
    v_notch_scn = float("nan")
    if not row.empty:
        be = float(pd.to_numeric(row["BreakEven_FUEL_PRICE_EurPerKg"], errors="coerce").iloc[0])
        v_notch_scn = float(pd.to_numeric(row["V_notch_kt"], errors="coerce").iloc[0])


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
        ax.scatter([x0], [y0], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{x0:.2f} €/kg ({_safe_display_econ_kt(y0, v_notch_scn)} kt)", zorder=20)
        _place_econ_annotation_inside(ax, x=x0, y=y0, text=f"{_safe_display_econ_kt(y0, v_notch_scn)} kt", prefer="left")
    else:
        if be_ok:
            y_at_be = _econ_at(float(be))
            ax.plot([be, be], [y_axis_bottom, y_at_be], linewidth=2.0, linestyle="--", color=c_be, label=f"Degalų lūžio taškas ({float(be):.2f} €/kg)", zorder=1)
            ax.scatter([be], [y_at_be], s=52, marker="o", color=c_be, label="_nolegend_", zorder=18)

        if in_ok:
            fp_in = float(fuel_price_operational)
            econ_y = _econ_at(fp_in)
            ax.plot([fp_in, fp_in], [y_axis_bottom, econ_y], linewidth=2.0, linestyle="--", color=c_in, label=f"Degalų įvestis ({fp_in:.2f} €/kg)", zorder=1)
            ax.scatter([fp_in], [econ_y], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{fp_in:.2f} €/kg ({_safe_display_econ_kt(econ_y, v_notch_scn)} kt)", zorder=20)
            _place_econ_annotation_inside(ax, x=fp_in, y=econ_y, text=f"{_safe_display_econ_kt(econ_y, v_notch_scn)} kt", prefer="left")

    ax.set_title(f"ECON priklausomybė nuo degalų kainos — {scenario_name}", pad=22)
    ax.set_xlabel("Degalų kaina (€/kg)")
    ax.set_ylabel("ECON (kt)")
    _add_axis_arrows(ax)
    ax.legend(loc="best")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig

def _compute_input_doc_curve_knn(
    scenarios: List[Dict[str, Any]],
    summary_tbl: pd.DataFrame,
    cfg: Config,
    *,
    fl_ft: float,
    wt_kg: float,
    isa_c: float,
    wind_kt: float,
) -> Dict[str, Any]:
    msg = _validate_interp_inputs(
        summary_tbl,
        fl_ft=float(fl_ft),
        weight_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
    )
    if msg:
        raise ValueError(msg)

    usable = [
        sc for sc in scenarios
        if isinstance(sc.get("docIASVec"), np.ndarray)
        and isinstance(sc.get("docVec"), np.ndarray)
        and sc.get("docIASVec").size >= 2
        and sc.get("docVec").size >= 2
    ]
    if len(usable) < 8:
        raise ValueError("Nepakanka scenarijų DOC interpolacijai (reikia bent 8 su DOC vektoriais).")

    ias_lo = min(float(np.nanmin(sc["docIASVec"])) for sc in usable)
    ias_hi = max(float(np.nanmax(sc["docIASVec"])) for sc in usable)
    if not np.isfinite(ias_lo) or not np.isfinite(ias_hi) or ias_hi <= ias_lo:
        raise ValueError("Nepavyko nustatyti galiojančio IAS intervalo DOC kreivei.")

    k_use = min(30, len(usable))
    min_nb = min(8, max(3, len(usable) // 2))

    x_full = np.linspace(ias_lo, ias_hi, 500, dtype=float)
    y_full, _diag1 = interpolate_curve_knn_from_scenarios(
        usable,
        fl_ft=float(fl_ft),
        weight_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
        x_grid=x_full,
        x_vec_key="docIASVec",
        y_vec_key="docVec",
        k=k_use,
        power=2.0,
        min_neighbors=min_nb,
    )

    x_full = np.asarray(x_full, float)
    y_full = np.asarray(y_full, float)

    ok = np.isfinite(x_full) & np.isfinite(y_full)
    if int(ok.sum()) < 80:
        raise ValueError("Nepakanka duomenų DOC kreivei nubraižyti.")

    x_full = x_full[ok]
    y_full = y_full[ok]

    j0 = int(np.nanargmin(y_full))

    left_i = max(0, j0 - 3)
    right_i = min(len(x_full) - 1, j0 + 3)

    x_left = float(x_full[left_i])
    x_right = float(x_full[right_i])

    if not np.isfinite(x_left) or not np.isfinite(x_right) or x_right <= x_left:
        x_left = float(max(ias_lo, x_full[j0] - 2.0))
        x_right = float(min(ias_hi, x_full[j0] + 2.0))

    x_fine = np.linspace(x_left, x_right, 800, dtype=float)
    y_fine, _diag2 = interpolate_curve_knn_from_scenarios(
        usable,
        fl_ft=float(fl_ft),
        weight_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
        x_grid=x_fine,
        x_vec_key="docIASVec",
        y_vec_key="docVec",
        k=k_use,
        power=2.0,
        min_neighbors=min_nb,
    )

    x_fine = np.asarray(x_fine, float)
    y_fine = np.asarray(y_fine, float)

    ok2 = np.isfinite(x_fine) & np.isfinite(y_fine)

    if int(ok2.sum()) >= 80:
        x_min_search = x_fine[ok2]
        y_min_search = y_fine[ok2]
    else:
        x_min_search = x_full
        y_min_search = y_full

    j = int(np.nanargmin(y_min_search))
    v_opt_raw = float(x_min_search[j])

    qres = compute_quick_metrics_interpolated(
        summary_tbl,
        fl_ft=float(fl_ft),
        weight_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
    )
    v_notch = float(qres.v_notch_kt)

    v_opt = _econ_from_docmin_with_notch_rule(
        v_opt_raw,
        v_notch,
        min_gap_kt=float(cfg.breakpoint_speed_tol_kt),
    )
    doc_opt = float(np.interp(v_opt, x_full, y_full))

    return {
        "IAS_grid": x_full,
        "DOC_grid_per_nm": y_full,
        "IAS_opt": v_opt,
        "DOC_opt_per_nm": doc_opt,
    }

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
    cur = _compute_input_doc_curve_knn(
        scenarios,
        summary_tbl,
        cfg,
        fl_ft=float(fl_ft),
        wt_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
    )

    x = np.asarray(cur["IAS_grid"], float)
    y = np.asarray(cur["DOC_grid_per_nm"], float)
    v_opt = float(cur["IAS_opt"])
    doc_opt = float(cur["DOC_opt_per_nm"])

    qres = compute_quick_metrics_interpolated(
        summary_tbl,
        fl_ft=float(fl_ft),
        weight_kg=float(wt_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
    )
    v_notch = float(qres.v_notch_kt)

    same = not _disp_speeds_differ(v_notch, v_opt, min_gap_kt=int(round(float(cfg.breakpoint_speed_tol_kt))))

    fig, ax = _mpl_academic_fig()
    ax.plot(x, y, linewidth=2.2, color="darkred", label="DOC kreivė")

    econ_kt = _safe_display_econ_kt(v_opt, v_notch)
    notch_kt = _disp_notch_kt(v_notch)

    if same:
        y_notch = float(np.interp(v_notch, x, y))
        ax.scatter(
            [v_notch],
            [y_notch],
            s=95,
            marker="x",
            color="black",
            label=f"ECON / IASnotch ({notch_kt} kt)",
        )
        _annotate_tiny_above(
            ax,
            v_notch,
            y_notch,
            f"ECON = IASnotch = {notch_kt} kt",
            color="black",
        )
    else:
        ax.scatter(
            [v_opt],
            [doc_opt],
            s=95,
            marker="x",
            color="orange",
            label=f"ECON ({econ_kt} kt)",
        )
        _annotate_tiny_above(
            ax,
            v_opt,
            doc_opt,
            f"ECON {econ_kt} kt",
            color="orange",
            dx_pts=-60,
            dy_pts=40,
        )

        y_notch = float(np.interp(v_notch, x, y))
        ax.scatter(
            [v_notch],
            [y_notch],
            s=95,
            marker="x",
            color="dodgerblue",
            label=f"IASnotch ({notch_kt} kt)",
        )
        _annotate_tiny_above(
            ax,
            v_notch,
            y_notch,
            f"IASnotch {notch_kt} kt",
            color="dodgerblue",
            dx_pts=14,
        )

    ax.set_title("DOC priklausomybė nuo IAS")
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
        lo=float(cfg.time_cost_min),
        hi=float(cfg.time_cost_max),
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
    x_grid = np.arange(
        float(cfg.time_cost_min),
        float(cfg.time_cost_max) + 1e-9,
        float(cfg.time_cost_step),
        dtype=float,
    )

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

    v_notch_input = float(qres.v_notch_kt)
    if np.isfinite(v_notch_input):
        y = np.minimum(y, v_notch_input)

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
        ax.scatter([tc_in], [y0], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{tc_in:.0f} €/h ({_safe_display_econ_kt(y0, v_notch_input)} kt)", zorder=20)
        _place_econ_annotation_inside(ax, x=tc_in, y=y0, text=f"{_safe_display_econ_kt(y0, v_notch_input)} kt", prefer="right")
    else:
        if be_ok:
            y_be = _econ_at(be)
            ax.plot([be, be], [y_axis_bottom, y_be], linewidth=2.0, linestyle="--", color=c_be, label=f"Laiko sąnaudų lūžio taškas ({be:.0f} €/h)", zorder=1)
            ax.scatter([be], [y_be], s=52, marker="o", color=c_be, label="_nolegend_", zorder=18)

        if in_ok:
            econ_y = _econ_at(tc_in)
            ax.plot([tc_in, tc_in], [y_axis_bottom, econ_y], linewidth=2.0, linestyle="--", color=c_in, label=f"Laiko sąnaudų įvestis ({tc_in:.0f} €/h)", zorder=1)
            ax.scatter([tc_in], [econ_y], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{tc_in:.0f} €/h ({_safe_display_econ_kt(econ_y, v_notch_input)} kt)", zorder=20)
            _place_econ_annotation_inside(ax, x=tc_in, y=econ_y, text=f"{_safe_display_econ_kt(econ_y, v_notch_input)} kt", prefer="right")

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
        lo=float(cfg.fuel_price_min),
        hi=float(cfg.fuel_price_max),
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
    x_grid = np.arange(
        float(cfg.fuel_price_min),
        float(cfg.fuel_price_max) + 1e-12,
        float(cfg.fuel_price_step),
        dtype=float,
    )

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

    v_notch_input = float(qres.v_notch_kt)
    if np.isfinite(v_notch_input):
        y = np.minimum(y, v_notch_input)

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
        ax.scatter([fp_in], [y0], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{fp_in:.2f} €/kg ({_safe_display_econ_kt(y0, v_notch_input)} kt)", zorder=20)
        _place_econ_annotation_inside(ax, x=fp_in, y=y0, text=f"{_safe_display_econ_kt(y0, v_notch_input)} kt", prefer="left")
    else:
        if be_ok:
            y_be = _econ_at(be)
            ax.plot([be, be], [y_axis_bottom, y_be], linewidth=2.0, linestyle="--", color=c_be, label=f"Degalų lūžio taškas ({be:.2f} €/kg)", zorder=1)
            ax.scatter([be], [y_be], s=52, marker="o", color=c_be, label="_nolegend_", zorder=18)

        if in_ok:
            econ_y = _econ_at(fp_in)
            ax.plot([fp_in, fp_in], [y_axis_bottom, econ_y], linewidth=2.0, linestyle="--", color=c_in, label=f"Degalų įvestis ({fp_in:.2f} €/kg)", zorder=1)
            ax.scatter([fp_in], [econ_y], s=95, marker="x", linewidths=2.2, color="black", label=f"ECON@{fp_in:.2f} €/kg ({_safe_display_econ_kt(econ_y, v_notch_input)} kt)", zorder=20)
            _place_econ_annotation_inside(ax, x=fp_in, y=econ_y, text=f"{_safe_display_econ_kt(econ_y, v_notch_input)} kt", prefer="left")

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
    color: str = "black",
) -> None:
    xs = np.asarray(xs, float).reshape(-1)
    ys = np.asarray(ys, float).reshape(-1)
    if xs.size == 0:
        return

    va = "bottom" if int(y_offset_pts) >= 0 else "top"

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
            va=va,
            fontsize=int(fontsize),
            color=color,
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.8,
                "color": color,
                "shrinkA": 1,
                "shrinkB": 0,
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

def _filter_label_candidates_by_min_delta(
    candidates: List[Tuple[float, float, str, str]],
    *,
    min_delta: float,
) -> List[Tuple[float, float, str, str]]:
    """
    Keep labels only if values are sufficiently separated inside each group/series
    across neighboring x-points.

    This function does NOT compare different groups against each other.
    Cross-group suppression is handled later in _label_points_global_dedup().
    """
    if not candidates or not np.isfinite(min_delta) or float(min_delta) <= 0.0:
        return candidates

    thr = float(min_delta)

    groups: Dict[str, List[Tuple[float, float, str, str]]] = {}
    for row in candidates:
        groups.setdefault(str(row[3]), []).append(row)

    kept: List[Tuple[float, float, str, str]] = []

    for rows in groups.values():
        rows = sorted(rows, key=lambda r: float(r[0]))
        if len(rows) == 1:
            kept.extend(rows)
            continue

        ys = [float(r[1]) for r in rows]
        diffs = [abs(ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]

        if not any(d >= thr for d in diffs):
            continue

        keep_mask = [False] * len(rows)
        for i in range(len(rows)):
            left_ok = i > 0 and abs(ys[i] - ys[i - 1]) >= thr
            right_ok = i < len(rows) - 1 and abs(ys[i + 1] - ys[i]) >= thr
            if left_ok or right_ok:
                keep_mask[i] = True

        kept.extend([row for row, keep in zip(rows, keep_mask) if keep])

    return kept

def _label_points_global_dedup(
    ax,
    candidates: List[Tuple[float, float, str, str]],
    *,
    overlap_frac: float = 0.80,
    y_offset_pts: int = 6,
    fontsize: int = 7,
    same_x_tol: float = 1e-9,
    close_y_delta: float = 0.1,
    color: str = "black",
) -> None:
    """
    Deduplicate labels across groups at the same x-position.

    If two groups have at least one shared x where their y-values differ by
    <= close_y_delta, they are considered part of the same family. Only one
    representative group from that family keeps labels.
    """
    if not candidates:
        return

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    rows = [
        (float(x), float(y), str(txt), str(grp))
        for x, y, txt, grp in candidates
        if np.isfinite(x) and np.isfinite(y)
    ]
    if not rows:
        return

    group_points: Dict[str, List[Tuple[float, float, str]]] = {}
    for x, y, txt, grp in rows:
        group_points.setdefault(grp, []).append((x, y, txt))

    for grp in group_points:
        group_points[grp] = sorted(group_points[grp], key=lambda r: r[0])

    groups = list(group_points.keys())

    def _groups_are_close(grp_a: str, grp_b: str) -> bool:
        pts_a = group_points[grp_a]
        pts_b = group_points[grp_b]

        for x1, y1, _ in pts_a:
            for x2, y2, _ in pts_b:
                if abs(float(x1) - float(x2)) <= float(same_x_tol):
                    if abs(float(y1) - float(y2)) <= float(close_y_delta):
                        return True
        return False

    families: List[List[str]] = []
    visited: set[str] = set()

    for grp in groups:
        if grp in visited:
            continue

        stack = [grp]
        family: List[str] = []

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            family.append(cur)

            for other in groups:
                if other in visited:
                    continue
                if _groups_are_close(cur, other):
                    stack.append(other)

        families.append(family)

    representative_groups: set[str] = set()
    for family in families:
        best_grp = max(
            family,
            key=lambda g: float(np.mean([y for _, y, _ in group_points[g]])),
        )
        representative_groups.add(best_grp)

    kept_bboxes: List[Any] = []
    label_rows: List[Tuple[float, float, str, str]] = []

    for grp in sorted(representative_groups):
        for x, y, txt in group_points[grp]:
            label_rows.append((x, y, txt, grp))

    label_rows = sorted(label_rows, key=lambda r: (r[0], -r[1], r[3]))
    va = "bottom" if int(y_offset_pts) >= 0 else "top"

    for x0, y0, txt, _grp in label_rows:
        ann = ax.annotate(
            txt,
            xy=(x0, y0),
            xycoords="data",
            xytext=(0, int(y_offset_pts)),
            textcoords="offset points",
            ha="center",
            va=va,
            fontsize=int(fontsize),
            color=color,
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.8,
                "color": color,
                "shrinkA": 1,
                "shrinkB": 0,
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

        too_close_bbox = any(_bbox_overlap_frac(bb, bb2) >= float(overlap_frac) for bb2 in kept_bboxes)
        if too_close_bbox:
            ann.remove()
            continue

        kept_bboxes.append(bb)

def _plot_saving_vs_grouped(
    summary_tbl: pd.DataFrame,
    *,
    x_col: str,
    title: str,
    x_label: str,
    group_col: Optional[str],
    cfg: Config,
    show_point_labels: bool = False,
    distance_nm: float = 1.0,
) -> Any:
    fig, ax = _mpl_academic_fig()

    df = summary_tbl.copy()
    df[x_col] = pd.to_numeric(df[x_col], errors="coerce")
    df["DOCmin_EurPerNM"] = pd.to_numeric(df["DOCmin_EurPerNM"], errors="coerce")
    df["DOCnotch_EurPerNM"] = pd.to_numeric(df["DOCnotch_EurPerNM"], errors="coerce")

    keep = (
        np.isfinite(df[x_col])
        & np.isfinite(df["DOCmin_EurPerNM"])
        & np.isfinite(df["DOCnotch_EurPerNM"])
    )
    df = df.loc[keep].copy()
    if df.empty:
        raise ValueError("Nėra duomenų po filtravimo.")

    dist_nm = max(float(distance_nm), 0.0)
    v_econ = pd.to_numeric(df["V_ECSR_kt"], errors="coerce")
    v_notch = pd.to_numeric(df["V_notch_kt"], errors="coerce")

    speed_gap_ok = []
    for ve, vn in zip(v_econ.tolist(), v_notch.tolist()):
        speed_gap_ok.append(_raw_speeds_differ(float(vn), float(ve), min_gap_kt=float(cfg.breakpoint_speed_tol_kt)))

    speed_gap_ok = pd.Series(speed_gap_ok, index=df.index)

    raw_saving = (df["DOCnotch_EurPerNM"] - df["DOCmin_EurPerNM"]).clip(lower=0.0)
    df["Saving_Eur"] = np.where(speed_gap_ok, raw_saving * dist_nm, 0.0)

    used_group = None
    if group_col and group_col in df.columns:
        df[group_col] = pd.to_numeric(df[group_col], errors="coerce")
        df = df.loc[np.isfinite(df[group_col])].copy()
        if not df.empty and int(df[group_col].nunique()) > 1:
            used_group = group_col

    y_all: List[float] = []

    if used_group:
        g = (
            df.groupby([used_group, x_col], as_index=False)["Saving_Eur"]
            .median()
            .rename(columns={"Saving_Eur": "Saving"})
        )

        label_candidates: List[Tuple[float, float, str, str]] = []

        for grp_val, sub in g.groupby(used_group, sort=True):
            sub = sub.sort_values(x_col)
            xs = sub[x_col].to_numpy(float)
            ys = sub["Saving"].to_numpy(float)

            ax.plot(
                xs,
                ys,
                linewidth=2.2,
                marker="o",
                markersize=4.8,
                linestyle="-",
                color="darkred",
                label=_group_label(used_group, float(grp_val)),
            )

            y_all.extend(ys.tolist())

            if show_point_labels:
                for x0, y0 in zip(xs.tolist(), ys.tolist()):
                    if np.isfinite(x0) and np.isfinite(y0):
                        label_candidates.append(
                            (float(x0), float(y0), f"{float(y0):.3f}", str(grp_val))
                        )

        if show_point_labels and label_candidates:
            _label_points_global_dedup(
                ax,
                label_candidates,
                overlap_frac=0.80,
                y_offset_pts=6,
                fontsize=7,
                color="black",
            )

        ax.legend(loc="best")

    else:
        g = (
            df.groupby(x_col, as_index=False)["Saving_Eur"]
            .median()
            .rename(columns={"Saving_Eur": "Saving"})
            .sort_values(x_col)
        )

        xs = g[x_col].to_numpy(float)
        ys = g["Saving"].to_numpy(float)

        ax.plot(
            xs,
            ys,
            linewidth=2.2,
            marker="o",
            color="darkred",
        )
        ax.scatter(xs, ys, s=26, color="darkred")
        y_all.extend(ys.tolist())

        if show_point_labels:
            _label_points_with_overlap_avoidance(
                ax,
                xs,
                ys,
                fmt="{:.3f}",
                y_offset_pts=6,
                fontsize=7,
                color="black",
            )

    y_for_limits = np.asarray(y_all, float)
    y_for_limits = y_for_limits[np.isfinite(y_for_limits)]
    if y_for_limits.size:
        y_min = max(0.0, float(np.nanmin(y_for_limits)))
        y_max = max(0.0, float(np.nanmax(y_for_limits)))
        rng = y_max - y_min

        if rng <= 0 or not np.isfinite(rng):
            pad = max(0.15 * max(abs(y_max), 1.0), 0.05)
            ax.set_ylim(0.0, y_max + pad)
        else:
            top_pad = max(0.25 * rng, 0.12 * max(abs(y_max), 1.0), 0.05)
            ax.set_ylim(0.0, y_max + top_pad)

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Sutaupymas (Eur) per {int(round(dist_nm))} NM" if dist_nm > 0 else "Sutaupymas (Eur)")
    _add_axis_arrows(ax)
    fig.tight_layout()
    return fig

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
    show_point_labels: bool = False,
    min_label_delta: float = 0.0,
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
        label_candidates: List[Tuple[float, float, str, str]] = []

        for grp_val, sub in g.groupby(used_group, sort=True):
            sub = sub.sort_values(x_col)
            xs = sub[x_col].to_numpy(float)
            ys = sub["BE"].to_numpy(float)

            ax.plot(xs, ys, linewidth=2.2, marker="o", markersize=4.8, label=_group_label(used_group, float(grp_val)))

            for x0, y0 in zip(xs.tolist(), ys.tolist()):
                if np.isfinite(x0) and np.isfinite(y0):
                    label_candidates.append(
                        (float(x0), float(y0), fmt.format(float(y0)), str(grp_val))
                    )

        # GLOBAL label placement with overlap dedup across groups
        if show_point_labels:
            label_candidates = _filter_label_candidates_by_min_delta(
                label_candidates,
                min_delta=float(min_label_delta),
            )
            _label_points_global_dedup(
                ax,
                label_candidates,
                overlap_frac=0.80,
                y_offset_pts=6,
                fontsize=7,
                close_y_delta=float(min_label_delta),
            )

        ax.legend(loc="best")
        y_for_limits = g["BE"].to_numpy(float)

    else:
        df = df.sort_values(x_col)
        xs = df[x_col].to_numpy(float)
        ys = df[y_col].to_numpy(float)

        ax.plot(xs, ys, linewidth=2.2, color="darkred")
        ax.scatter(xs, ys, s=26, color="darkred")

        # Single series: old labeling is fine (or you can also use global_dedup)
        if show_point_labels:
            single_candidates = [
                (float(x0), float(y0), fmt.format(float(y0)), "__single__")
                for x0, y0 in zip(xs.tolist(), ys.tolist())
                if np.isfinite(x0) and np.isfinite(y0)
            ]
            single_candidates = _filter_label_candidates_by_min_delta(
                single_candidates,
                min_delta=float(min_label_delta),
            )
            _label_points_global_dedup(
                ax,
                single_candidates,
                overlap_frac=0.80,
                y_offset_pts=6,
                fontsize=7,
                close_y_delta=float(min_label_delta),
            )

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


def _find_valid_breakpoint_from_curve(
    x_vals: np.ndarray,
    econ_vals: np.ndarray,
    v_notch: float,
    docmin_per_nm: float,
    docnotch_per_nm: float,
    cfg: Config,
) -> float:
    x = np.asarray(x_vals, float)
    y = np.asarray(econ_vals, float)

    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    if x.size == 0 or not np.isfinite(v_notch):
        return float("nan")

    if not (
        np.isfinite(docmin_per_nm)
        and np.isfinite(docnotch_per_nm)
        and float(docnotch_per_nm) > float(docmin_per_nm)
    ):
        return float("nan")

    valid = np.array(
        [_raw_speeds_differ(float(v_notch), float(v), min_gap_kt=float(cfg.breakpoint_speed_tol_kt)) for v in y],
        dtype=bool,
    )

    if not np.any(valid):
        return float("nan")

    return float(x[np.argmax(valid)])

def _build_interpolated_sweep_table(
    summary_tbl: pd.DataFrame,
    scenarios: List[Dict[str, Any]],
    *,
    x_col: str,
    fixed: Dict[str, Optional[float]],
    include_x_value: Optional[float] = None,
    cfg: Optional[Config] = None,
) -> pd.DataFrame:
    if summary_tbl is None or summary_tbl.empty:
        return pd.DataFrame()
    if cfg is None:
        raise ValueError("cfg is required.")

    x_vals = _unique_sorted(summary_tbl[x_col])
    if include_x_value is not None and np.isfinite(float(include_x_value)):
        x_vals = sorted(set([*x_vals, float(include_x_value)]))

    if not x_vals:
        return pd.DataFrame()

    rows: List[Dict[str, float]] = []

    for x_val in x_vals:
        zp_ft = float(x_val) if x_col == "ZP_ft" else float(fixed["ZP_ft"])
        weight_kg = float(x_val) if x_col == "WEIGHT_kg" else float(fixed["WEIGHT_kg"])
        isa_c = float(x_val) if x_col == "ISA_C" else float(fixed["ISA_C"])
        wind_kt = float(x_val) if x_col == "WIND_kt" else float(fixed["WIND_kt"])

        try:
            qres = compute_quick_metrics_interpolated(
                summary_tbl,
                fl_ft=zp_ft,
                weight_kg=weight_kg,
                isa_c=isa_c,
                wind_kt=wind_kt,
            )

            v_notch = float(qres.v_notch_kt)

            try:
                doc_curve_res = _compute_input_doc_curve_knn(
                    scenarios,
                    summary_tbl,
                    cfg,
                    fl_ft=zp_ft,
                    wt_kg=weight_kg,
                    isa_c=isa_c,
                    wind_kt=wind_kt,
                )
                v_econ = float(doc_curve_res["IAS_opt"])
            except Exception:
                v_econ = float(qres.v_ecsr_kt)

            if np.isfinite(v_econ) and np.isfinite(v_notch):
                v_econ = min(v_econ, v_notch)

            time_x = np.arange(
                float(cfg.time_cost_min),
                float(cfg.time_cost_max) + 1e-9,
                float(cfg.time_cost_step),
                dtype=float,
            )
            time_y, _ = interpolate_curve_knn_from_scenarios(
                scenarios,
                fl_ft=zp_ft,
                weight_kg=weight_kg,
                isa_c=isa_c,
                wind_kt=wind_kt,
                x_grid=time_x,
                x_vec_key="timeCostVec",
                y_vec_key="IAS_opt_kt",
                k=min(30, len(scenarios)),
                min_neighbors=min(8, max(3, len(scenarios) // 2)),
                power=2.0,
            )
            time_y = np.asarray(time_y, float)
            if np.isfinite(v_notch):
                time_y = np.minimum(time_y, v_notch)

            fuel_x = np.arange(
                float(cfg.fuel_price_min),
                float(cfg.fuel_price_max) + 1e-12,
                float(cfg.fuel_price_step),
                dtype=float,
            )
            fuel_y, _ = interpolate_curve_knn_from_scenarios(
                scenarios,
                fl_ft=zp_ft,
                weight_kg=weight_kg,
                isa_c=isa_c,
                wind_kt=wind_kt,
                x_grid=fuel_x,
                x_vec_key="fuelPriceVec",
                y_vec_key="IAS_opt_kt_fp",
                k=min(30, len(scenarios)),
                min_neighbors=min(8, max(3, len(scenarios) // 2)),
                power=2.0,
            )
            fuel_y = np.asarray(fuel_y, float)
            if np.isfinite(v_notch):
                fuel_y = np.minimum(fuel_y, v_notch)

            be_time = _find_valid_breakpoint_from_curve(
                time_x,
                time_y,
                v_notch,
                float(qres.docmin_eur_per_nm),
                float(qres.docnotch_eur_per_nm),
                cfg,
            )

            be_fuel = _find_valid_breakpoint_from_curve(
                fuel_x,
                fuel_y,
                v_notch,
                float(qres.docmin_eur_per_nm),
                float(qres.docnotch_eur_per_nm),
                cfg,
            )

            rows.append(
                {
                    "ZP_ft": zp_ft,
                    "WEIGHT_kg": weight_kg,
                    "ISA_C": isa_c,
                    "WIND_kt": wind_kt,
                    "V_ECSR_kt": float(v_econ),
                    "V_notch_kt": float(v_notch),
                    "DOCmin_EurPerNM": float(qres.docmin_eur_per_nm),
                    "DOCnotch_EurPerNM": float(qres.docnotch_eur_per_nm),
                    "BreakEven_TIME_COST_EurPerHr": be_time,
                    "BreakEven_FUEL_PRICE_EurPerKg": be_fuel,
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)

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

    for gid in _DOC_GRAPHS.keys():
        st.session_state.setdefault(f"fig_{gid}", None)
        st.session_state.setdefault(f"cap_{gid}", "")
        st.session_state.setdefault(f"open_{gid}", False)
        st.session_state.setdefault(f"err_{gid}", "")

    st.session_state.setdefault("show_glossary", False)
    st.session_state.setdefault("excel_written_msg", "")

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
    st.session_state.setdefault("in_dist_nm", 0.0)
    st.session_state.setdefault("quick_doc_dist_nm", 0.0)
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
        key="mode",
        help=(
            "„Scenarijus“ režime peržiūrimi konkretūs įkelti scenarijai. "
            "„Įvestis“ režime rezultatai ir grafikai apskaičiuojami pagal jūsų įvestas sąlygas "
            "naudojant scenarijų interpoliaciją."
        ),
    )

    saving_mode_nm = st.checkbox(
        "Taikyti sutaupymo vertę (€/NM)",
        key="saving_mode_nm",
    )

    saving_custom_nm = None

    if saving_mode_nm:
        saving_custom_nm = st.number_input(
            "Sutaupymas (€/NM)",
            min_value=0.0,
            step=0.1,
            key="saving_custom_nm",
        )

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

    if saving_mode_nm:
        saving_mode_value = "per_nm"
    else:
        saving_mode_value = "default"

    cfg = replace(
        cfg0,
        fuel_price_eur_per_kg=float(fuel_price),
        time_cost_operational=float(tc_op),
        epsilon_break_even=float(float(epsilon_pct) / 100.0),
        breakpoint_saving_mode=saving_mode_value,
        breakpoint_saving_eur_per_nm=float(saving_custom_nm or 0.0),
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

    for gid in _DOC_GRAPHS.keys():
        st.session_state[f"open_{gid}"] = False
        st.session_state[f"fig_{gid}"] = None
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
fuel_ceiling = float(getattr(getattr(cfg, "break_search", None), "fuel_ceiling_eur_per_kg", float("inf")))

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
        wind_kt = st.number_input("Vėjo komponentė skrydžio kryptimi (kt)", step=1.0, key="ecsr_calc_wind")
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
        qres = compute_quick_metrics_interpolated(
            summary_tbl,
            fl_ft=float(fl_ft),
            weight_kg=float(wt_kg),
            isa_c=float(isa_c),
            wind_kt=float(wind_kt),
        )

        v_econ_docmin = _input_docmin_econ_kt(
            scenarios,
            summary_tbl,
            cfg,
            fl_ft=float(fl_ft),
            wt_kg=float(wt_kg),
            isa_c=float(isa_c),
            wind_kt=float(wind_kt),
            fallback_v_econ=float(res.v_ecsr_kt),
            fallback_v_notch=float(qres.v_notch_kt),
        )

        rng_txt = _ecsr_range_str(
            res.ecsr_low_kt,
            res.ecsr_high_kt,
            float(qres.v_notch_kt),
            v_econ_docmin,
        )
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
    ("Sutaupymas_perX (EUR)", "__SAVING_PER_X__"),
    ("Laiko sąnaudų lūžio taškas (€/h)", "__BREAK_TIME__"),
    ("Degalų sąnaudų lūžio taškas (€/kg)", "__BREAK_FUEL__"),
]
metric_map: Dict[str, str] = {k: v for k, v in metric_items}

def _show_result_card(value: str, unit: str) -> None:
    _, mid, _ = st.columns([2.2, 1.2, 2.2], gap="large")
    with mid:
        _render_result_card(value, unit, "Rezultatas")


if mode == "Scenarijus":
    top_l, top_r = st.columns(2, gap="large")
    with top_l:
        pick_scn = st.selectbox(
            "Pasirinkite scenarijų",
            ["Pasirinkite..."] + scenario_names,
            index=0,
            key="quick_scn",
        )
    with top_r:
        pick_metric_label = st.selectbox(
            "Pasirinkite rodiklį",
            list(metric_map.keys()),
            index=0,
            key="quick_metric",
        )

    show_placeholder = (pick_scn == "Pasirinkite..." or metric_map[pick_metric_label] == "__NONE__")

    distance_nm = 0.0
    is_saving_per_x = (not show_placeholder) and (metric_map[pick_metric_label] == "__SAVING_PER_X__")

    if is_saving_per_x:
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
    diff_total = float("nan")
    quick_caption = ""

    if not show_placeholder:
        col_key = metric_map[pick_metric_label]
        row_df = summary_tbl.loc[summary_tbl["ScenarioName"] == pick_scn]
        if not row_df.empty:
            row = row_df.iloc[0]
            quick_caption = _conditions_sentence_from_row_with_costs(row, cfg)

            if col_key == "__ECSR_RANGE__":
                sc = _scenario_lookup(scenarios, pick_scn)
                v_econ_docmin = float("nan")
                if sc is not None:
                    v_econ_docmin = _scenario_docmin_econ_kt(sc, cfg)

                shown_value = _ecsr_range_str(
                    float(pd.to_numeric(row.get("ECSR_low_kt", np.nan), errors="coerce")),
                    float(pd.to_numeric(row.get("ECSR_high_kt", np.nan), errors="coerce")),
                    float(pd.to_numeric(row.get("V_notch_kt", np.nan), errors="coerce")),
                    v_econ_docmin,
                )
                shown_unit = "kt" if shown_value else ""

            elif col_key == "__BREAK_TIME__":
                be_str = _format_break_even_for_app_time(
                    summary_tbl=row_df,
                    sweep_min=float(cfg.time_cost_min),
                    sweep_max=float(cfg.time_cost_max),
                )
                shown_value = str(be_str.iloc[0]) if len(be_str) else ""
                shown_unit = "€/h" if (shown_value and shown_value != "Nėra lūžio taško") else ""

            elif col_key == "__BREAK_FUEL__":
                be_str = _format_break_even_for_app_fuel(summary_tbl=row_df, ceiling=fuel_ceiling)
                shown_value = str(be_str.iloc[0]) if len(be_str) else ""
                shown_unit = "€/kg" if (shown_value and shown_value != "Nėra lūžio taško") else ""

            elif col_key == "__SAVING_PER_X__":
                v_min_per_nm = float(pd.to_numeric(row.get("DOCmin_EurPerNM", np.nan), errors="coerce"))
                v_notch_per_nm = float(pd.to_numeric(row.get("DOCnotch_EurPerNM", np.nan), errors="coerce"))
                v_notch_raw = float(pd.to_numeric(row.get("V_notch_kt", np.nan), errors="coerce"))
                v_econ_raw = float(pd.to_numeric(row.get("V_ECSR_kt", np.nan), errors="coerce"))
                dist = float(distance_nm)

                if _raw_speeds_differ(v_notch_raw, v_econ_raw, min_gap_kt=float(cfg.breakpoint_speed_tol_kt)):
                    if np.isfinite(v_min_per_nm) and np.isfinite(v_notch_per_nm) and np.isfinite(dist):
                        diff_total = round(float((v_notch_per_nm - v_min_per_nm) * dist), 1)
                    else:
                        diff_total = float("nan")
                else:
                    diff_total = 0.0
            else:
                val = float(pd.to_numeric(row.get(col_key, np.nan), errors="coerce"))
                if np.isfinite(val):
                    if col_key == "V_ECSR_kt":
                        sc = _scenario_lookup(scenarios, pick_scn)
                        if sc is not None:
                            v_econ_docmin = _scenario_docmin_econ_kt(sc, cfg)
                            v_notch_ui = float(pd.to_numeric(row.get("V_notch_kt", np.nan), errors="coerce"))
                            shown_value = _fmt_speed_econ_safe(v_econ_docmin, v_notch_ui)
                            shown_unit = "kt" if shown_value else ""
                    elif col_key == "V_notch_kt":
                        shown_value = _fmt_speed_notch(val)
                        shown_unit = "kt" if shown_value else ""
                    elif "EUR/h" in pick_metric_label:
                        shown_value = f"{val:.1f}"
                        shown_unit = "EUR/h"
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
        if is_saving_per_x:
            v = _fmt_eur(diff_total, decimals=1) if np.isfinite(diff_total) else "0,0"
            _show_result_card(v, "EUR")
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
        in_wind = st.number_input("Vėjo dedamoji skrydžio kryptimi (kt)", step=1.0, key="in_wind")

    pick_metric_label = st.selectbox("Pasirinkite rodiklį", list(metric_map.keys()), index=0, key="in_metric")
    col_key = metric_map[pick_metric_label]
    show_placeholder = (col_key == "__NONE__")

    distance_nm = float(st.session_state.get("in_dist_nm", 0.0))

    if not show_placeholder and col_key == "__SAVING_PER_X__":
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
    diff_total = float("nan")

    if not show_placeholder:
        res_in = st.session_state.get("in_last_res", None)
        if isinstance(res_in, InterpQuickResult):
            if col_key == "__ECSR_RANGE__":
                v_econ_docmin = _input_docmin_econ_kt(
                    scenarios,
                    summary_tbl,
                    cfg,
                    fl_ft=float(in_fl),
                    wt_kg=float(in_wt),
                    isa_c=float(in_isa),
                    wind_kt=float(in_wind),
                    fallback_v_econ=float(res_in.v_ecsr_kt),
                    fallback_v_notch=float(res_in.v_notch_kt),
                )
                shown_value = _ecsr_range_str(
                    res_in.ecsr_low_kt,
                    res_in.ecsr_high_kt,
                    res_in.v_notch_kt,
                    v_econ_docmin,
                )
                shown_unit = "kt" if shown_value else ""
            elif col_key == "__BREAK_TIME__":
                v = float(res_in.be_time_cost_eur_per_hr)
                valid_break = (
                    np.isfinite(v)
                    and (float(cfg.time_cost_min) - 1e-9 <= v <= float(cfg.time_cost_max) + 1e-9)
                )
                
                if valid_break:
                    shown_value = f"{v:.2f}"
                    shown_unit = "€/kg"
                else:
                    shown_value = "Nėra lūžio taško"
                    shown_unit = ""
            elif col_key == "__BREAK_FUEL__":
                v = float(res_in.be_fuel_price_eur_per_kg)
                valid_break = (
                    np.isfinite(v)
                    and v <= float(fuel_ceiling) + 1e-12
                ) 

                if valid_break:
                    shown_value = f"{v:.2f}"
                    shown_unit = "€/kg"
                else:
                    shown_value = "Nėra lūžio taško"
                    shown_unit = ""
            elif col_key == "__SAVING_PER_X__":
                dist = float(distance_nm)
                v_notch_ui = float(res_in.v_notch_kt)
                v_econ_ui = float(res_in.v_ecsr_kt)

                if _raw_speeds_differ(v_notch_ui, v_econ_ui, min_gap_kt=float(cfg.breakpoint_speed_tol_kt)):
                    if np.isfinite(res_in.docmin_eur_per_nm) and np.isfinite(res_in.docnotch_eur_per_nm):
                        diff_total = round(float((res_in.docnotch_eur_per_nm - res_in.docmin_eur_per_nm) * dist), 1)
                    else:
                        diff_total = float("nan")
                else:
                    diff_total = 0.0
            else:
                if col_key == "V_ECSR_kt":
                    econ_val = _input_docmin_econ_kt(
                        scenarios,
                        summary_tbl,
                        cfg,
                        fl_ft=float(in_fl),
                        wt_kg=float(in_wt),
                        isa_c=float(in_isa),
                        wind_kt=float(in_wind),
                        fallback_v_econ=float(res_in.v_ecsr_kt),
                        fallback_v_notch=float(res_in.v_notch_kt),
                    )
                    if np.isfinite(econ_val):
                        shown_value = _fmt_speed_econ_safe(econ_val, float(res_in.v_notch_kt))
                        shown_unit = "kt" if shown_value else ""

                else:
                    mapping = {
                        "V_notch_kt": (res_in.v_notch_kt, "kt"),
                        "DOCmin_EurPerNM": (res_in.docmin_eur_per_nm, "EUR/NM"),
                        "DOCnotch_EurPerNM": (res_in.docnotch_eur_per_nm, "EUR/NM"),
                    }
                    if col_key in mapping:
                        val, unit = mapping[col_key]
                        if np.isfinite(val):
                            if col_key == "V_notch_kt":
                                shown_value = _fmt_speed_notch(float(val))
                            elif unit == "kt":
                                shown_value = _fmt_speed_econ(float(val))
                            elif unit == "EUR/NM":
                                shown_value = f"{float(val):.3f}"
                            elif unit == "EUR/h":
                                shown_value = f"{float(val):.1f}"
                            else:
                                shown_value = f"{float(val):g}"
                            shown_unit = unit
    st.markdown("<div style='height: 26px'></div>", unsafe_allow_html=True)

    if show_placeholder:
        _show_result_card("", "")
    else:
        if col_key == "__SAVING_PER_X__":
            v = _fmt_eur(diff_total, decimals=1) if np.isfinite(diff_total) else "0,0"
            _show_result_card(v, "EUR")
        else:
            _show_result_card(shown_value, shown_unit)


def _bp_filter_ui_scenario(
    graph_id: str,
    *,
    summary_tbl0: pd.DataFrame,
    allow_unfixed: bool = True,
) -> Dict[str, Optional[float]]:
    meta = _BP_GRAPHS.get(graph_id) or _DOC_GRAPHS[graph_id]
    x_col = meta["x_col"]
    fixed: Dict[str, Optional[float]] = {}

    cols = st.columns(3, gap="medium")
    other_cols = [c for c in _BP_OTHER_COLS if c != x_col]

    for i, col in enumerate(other_cols):
        label, unit = _GROUP_META.get(col, (col, ""))
        values = _unique_sorted(summary_tbl0[col]) if col in summary_tbl0.columns else []

        if allow_unfixed:
            options = ["nefiksuoti"] + [_fmt_num(v) for v in values] if values else ["nefiksuoti"]
        else:
            options = [_fmt_num(v) for v in values] if values else []

        with cols[i % 3]:
            if not options:
                st.error(f"Nėra galimų reikšmių parametrui: {label}")
                fixed[col] = None
                continue

            pick = st.selectbox(
                f"Fiksuoti {label} ({unit})" if unit else f"Fiksuoti {label}",
                options,
                key=f"{graph_id}_flt_{col}_{'u' if allow_unfixed else 'f'}",
            )

        if allow_unfixed and pick == "nefiksuoti":
            fixed[col] = None
        else:
            fixed[col] = float(pick)

    fixed[x_col] = None
    return fixed

def _bp_filter_ui_input(graph_id: str) -> Dict[str, Optional[float]]:
    meta = _BP_GRAPHS.get(graph_id) or _DOC_GRAPHS[graph_id]
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
        input_key = f"inputmode_{graph_id}_in_{col}"

        if input_key not in st.session_state:
            st.session_state[input_key] = float(defaults.get(col, 0.0))

        with cols[i % 3]:
            value = st.number_input(
                f"{label} ({unit})" if unit else f"{label}",
                step=1.0 if col in {"ISA_C", "WIND_kt"} else 500.0,
                key=input_key,
            )
            fixed[col] = float(value)

    return fixed

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

    for gid, meta in _DOC_GRAPHS.items():
        x_col = meta["x_col"]
        x_label = meta["x_label"]
        x_name_lt = meta["x_name_lt"]

        doc_graph_titles = {
            "d1": "Grafikas 2 — Sutaupymas (Eur) vs Vėjo komponentė (kt)",
            "d2": "Grafikas 3 — Sutaupymas (Eur) vs Masė (kg)",
            "d3": "Grafikas 4 — Sutaupymas (Eur) vs Skrydžio aukštis (ft)",
            "d4": "Grafikas 5 — Sutaupymas (Eur) vs ISA nuokrypis (°C)",
        }
        exp_title = doc_graph_titles[gid]
        open_key = f"open_{gid}"
        fig_key = f"fig_{gid}"
        cap_key = f"cap_{gid}"
        err_key = f"err_{gid}"

        with st.expander(exp_title, expanded=st.session_state.get(open_key, False)):
            fixed = _bp_filter_ui_scenario(gid, summary_tbl0=summary_tbl, allow_unfixed=False)
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

            cc1, cc2 = st.columns([2.2, 1.0], gap="small")
            with cc1:
                use_distance = st.checkbox(
                    "Atstumas sutaupymui",
                    key=f"{gid}_use_distance_scn",
                )
                saving_distance_nm = 1.0
                if use_distance:
                    saving_distance_nm = st.number_input(
                        "Atstumas (NM)",
                        min_value=0.0,
                        step=10.0,
                        key=f"{gid}_distance_scn",
                    )
            with cc2:
                run_graph = st.button(
                    "Generuoti grafiką",
                    key=f"btn_{gid}_doc",
                    disabled=not ok,
                    use_container_width=True,
                )

            if run_graph:
                st.session_state[open_key] = True
                try:
                    st.session_state[fig_key] = _plot_saving_vs_grouped(
                        filtered_local,
                        x_col=x_col,
                        title=f"Sutaupymo priklausomybė nuo {x_name_lt}",
                        x_label=x_label,
                        group_col=group_col,
                        cfg=cfg,
                        show_point_labels=True,
                        distance_nm=float(saving_distance_nm),
                    )
                    st.session_state[cap_key] = _conditions_sentence_from_filters(fixed, x_col=x_col, grouped_by=group_col)
                    st.session_state[err_key] = ""
                except Exception as e:
                    st.session_state[fig_key] = None
                    st.session_state[cap_key] = ""
                    st.session_state[err_key] = _normalize_ui_error(e)
                st.rerun()

            if st.session_state.get(err_key):
                st.error(st.session_state[err_key])
            if st.session_state.get(fig_key) is not None:
                _show_fig(st.session_state[fig_key])
                _black_note(st.session_state.get(cap_key, ""))

    with st.expander("Grafikas 6 — ECON (kt) vs Laiko sąnaudos (eur/h)", expanded=st.session_state["open_g2"]):
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

    with st.expander("Grafikas 7 — ECON (kt) vs Degalų kaina (eur/kg)", expanded=st.session_state["open_g3"]):
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
            wind = st.number_input("Vėjo dedamoji skrydžio kryptimi (kt)", value=float(st.session_state["in_wind"]), step=1.0, key="g1i_wind")
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

    for gid, meta in _DOC_GRAPHS.items():
        x_col = meta["x_col"]
        x_label = meta["x_label"]
        x_name_lt = meta["x_name_lt"]

        doc_graph_titles = {
            "d1": "Grafikas 2 — Sutaupymas (Eur) vs Vėjo komponentė (kt)",
            "d2": "Grafikas 3 — Sutaupymas (Eur) vs Masė (kg)",
            "d3": "Grafikas 4 — Sutaupymas (Eur) vs Skrydžio aukštis (ft)",
            "d4": "Grafikas 5 — Sutaupymas (Eur) vs ISA nuokrypis (°C)",
        }
        exp_title = doc_graph_titles[gid]
        open_key = f"open_{gid}"
        fig_key = f"fig_{gid}"
        cap_key = f"cap_{gid}"
        err_key = f"err_{gid}"

        with st.expander(exp_title, expanded=st.session_state.get(open_key, False)):
            fixed_in = _bp_filter_ui_input(gid)

            ok = True
            group_col = None
            msg = ""

            try:
                current_x_value = {
                    "ZP_ft": float(st.session_state.get("inputmode_" + gid + "_in_ZP_ft", st.session_state.get("in_fl", 0.0))),
                    "WEIGHT_kg": float(st.session_state.get("inputmode_" + gid + "_in_WEIGHT_kg", st.session_state.get("in_wt", 0.0))),
                    "ISA_C": float(st.session_state.get("inputmode_" + gid + "_in_ISA_C", st.session_state.get("in_isa", 0.0))),
                    "WIND_kt": float(st.session_state.get("inputmode_" + gid + "_in_WIND_kt", st.session_state.get("in_wind", 0.0))),
                }[x_col]

                filtered_local = _build_interpolated_sweep_table(
                    summary_tbl,
                    scenarios,
                    x_col=x_col,
                    fixed=fixed_in,
                    include_x_value=current_x_value,
                    cfg=cfg,
                )

                if filtered_local.empty:
                    raise ValueError("Nepakanka duomenų interpoliuotam grafikui sudaryti.")

                ok = True
                group_col = None
                msg = ""
            except Exception as e:
                ok = False
                msg = _normalize_ui_error(e)
                filtered_local = summary_tbl.iloc[0:0].copy()

            if not ok and msg:
                st.error(msg)

            cc1, cc2 = st.columns([2.2, 1.0], gap="small")
            with cc1:
                use_distance = st.checkbox(
                    "Atstumas sutaupymui",
                    key=f"{gid}_use_distance_input",
                )
                saving_distance_nm = 1.0
                if use_distance:
                    saving_distance_nm = st.number_input(
                        "Atstumas (NM)",
                        min_value=0.0,
                        step=10.0,
                        key=f"{gid}_distance_input",
                    )
            with cc2:
                run_graph = st.button(
                    "Generuoti grafiką",
                    key=f"btn_{gid}_doc_input",
                    disabled=not ok,
                    use_container_width=True,
                )

            if run_graph:
                st.session_state[open_key] = True
                try:
                    st.session_state[fig_key] = _plot_saving_vs_grouped(
                        filtered_local,
                        x_col=x_col,
                        title=f"Sutaupymo priklausomybė nuo {x_name_lt}",
                        x_label=x_label,
                        group_col=group_col,
                        cfg=cfg,
                        show_point_labels=True,
                        distance_nm=float(saving_distance_nm),
                    )
                    st.session_state[cap_key] = _conditions_sentence_from_filters(
                        fixed_in,
                        x_col=x_col,
                        grouped_by=None,
                    )
                    st.session_state[err_key] = ""
                except Exception as e:
                    st.session_state[fig_key] = None
                    st.session_state[cap_key] = ""
                    st.session_state[err_key] = _normalize_ui_error(e)
                st.rerun()

            if st.session_state.get(err_key):
                st.error(st.session_state[err_key])
            if st.session_state.get(fig_key) is not None:
                _show_fig(st.session_state[fig_key])
                _black_note(st.session_state.get(cap_key, ""))

    with st.expander("Grafikas 6 — ECON (kt) vs Laiko sąnaudos (eur/h)", expanded=st.session_state["open_g2"]):
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.0], gap="small")
        with c1:
            fl = st.number_input("Aukštis (ft)", value=float(st.session_state["in_fl"]), step=500.0, key="g2i_fl")
        with c2:
            wt = st.number_input("Masė (kg)", value=float(st.session_state["in_wt"]), step=500.0, key="g2i_wt")
        with c3:
            isa = st.number_input("ISA nuokrypis (°C)", value=float(st.session_state["in_isa"]), step=1.0, key="g2i_isa")
        with c4:
            wind = st.number_input("Vėjo dedamoji skrydžio kryptimi (kt)", value=float(st.session_state["in_wind"]), step=1.0, key="g2i_wind")
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

    with st.expander("Grafikas 7 — ECON (kt) vs Degalų kaina (eur/kg)", expanded=st.session_state["open_g3"]):
        c1, c2, c3, c4, c5 = st.columns([1.1, 1.1, 1.1, 1.1, 1.0], gap="small")
        with c1:
            fl = st.number_input("Aukštis (ft)", value=float(st.session_state["in_fl"]), step=500.0, key="g3i_fl")
        with c2:
            wt = st.number_input("Masė (kg)", value=float(st.session_state["in_wt"]), step=500.0, key="g3i_wt")
        with c3:
            isa = st.number_input("ISA nuokrypis (°C)", value=float(st.session_state["in_isa"]), step=1.0, key="g3i_isa")
        with c4:
            wind = st.number_input("Vėjo dedamoji skrydžio kryptimi (kt)", value=float(st.session_state["in_wind"]), step=1.0, key="g3i_wind")
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

for gid, meta in _BP_GRAPHS.items():
    x_col = meta["x_col"]
    x_label = meta["x_label"]
    x_name_lt = meta["x_name_lt"]

    _bp_title_map = {
        "g4": "Grafikas 8 — Lūžio taškai vs Vėjo komponentė (kt)",
        "g5": "Grafikas 9 — Lūžio taškai vs Masė (kg)",
        "g6": "Grafikas 10 — Lūžio taškai vs Skrydžio aukštis (ft)",
        "g7": "Grafikas 11 — Lūžio taškai vs ISA nuokrypis (°C)",
    }
    exp_title = _bp_title_map[gid]
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
                        show_point_labels=True,
                        min_label_delta=40.0,
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
                        show_point_labels=True,
                        min_label_delta=0.05,
                    )

                    st.session_state[cap_key] = _conditions_sentence_from_filters(
                        fixed,
                        x_col=x_col,
                        grouped_by=group_col,
                    )
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

            ok = True
            group_col = None
            msg = ""

            try:
                include_x_value = None

                main_input_defaults = {
                    "ZP_ft": st.session_state.get("in_fl", None),
                    "WEIGHT_kg": st.session_state.get("in_wt", None),
                    "ISA_C": st.session_state.get("in_isa", None),
                    "WIND_kt": st.session_state.get("in_wind", None),
                }

                candidate_x = main_input_defaults.get(x_col, None)

                if candidate_x is not None:
                    candidate_x = float(candidate_x)
                    bounds = _numeric_bounds(summary_tbl, x_col)
                    if bounds is not None:
                        lo, hi = bounds
                        if np.isfinite(candidate_x) and lo <= candidate_x <= hi:
                            include_x_value = candidate_x

                filtered_local = _build_interpolated_sweep_table(
                    summary_tbl,
                    scenarios,
                    x_col=x_col,
                    fixed=fixed_in,
                    include_x_value=include_x_value,
                    cfg=cfg,
                )

                if filtered_local.empty:
                    raise ValueError("Nepakanka duomenų interpoliuotam grafikui sudaryti.")

                ok = True
                group_col = None
                msg = ""
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
                    show_point_labels=True,
                    min_label_delta=40.0,
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
                        show_point_labels=True,
                        min_label_delta=0.05,
                    )

                    st.session_state[cap_key] = _conditions_sentence_from_filters(
                        fixed_in,
                        x_col=x_col,
                        grouped_by=None,
                    )
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

# ========================= Summary tables + download =========================
st.divider()
st.header("Ekonomiški scenarijai")

econ_tbl = _build_economical_scenarios_table(summary_tbl, cfg, scenarios)
if econ_tbl.empty:
    st.info("Su pasirinktais parametrais, ekonomiškų scenarijų neegzistuoja (ECON ~= IASnotch).")
else:
    st.dataframe(econ_tbl, use_container_width=True)

st.divider()
st.header("Bendra rezultatų lentelė")

# ------------------------- Bendra rezultatų lentelė (renamed + reordered) -------------------------

display_tbl = summary_tbl.copy()

# 1) Sort + scenario label
display_tbl = display_tbl.sort_values(by="ScenarioName", key=lambda s: s.map(_scenario_sort_key)).reset_index(drop=True)
display_tbl["_ScenarioName_raw"] = display_tbl["ScenarioName"]
display_tbl["ScenarioName"] = display_tbl["ScenarioName"].map(_scenario_trial_label)

# 2) Create ECSR interval column (IASlow–IAShigh)
# 2) Create ECSR interval column (IASlow–IAShigh)
e_lo = pd.to_numeric(display_tbl.get("ECSR_low_kt", np.nan), errors="coerce")
e_hi = pd.to_numeric(display_tbl.get("ECSR_high_kt", np.nan), errors="coerce")

def _fmt_interval(
    lo: float,
    hi: float,
    v_notch: Optional[float] = None,
    v_econ: Optional[float] = None,
) -> str:
    return _ecsr_range_str(lo, hi, v_notch, v_econ)

v_notch_tbl = pd.to_numeric(display_tbl.get("V_notch_kt", np.nan), errors="coerce")
v_econ_tbl = pd.to_numeric(display_tbl.get("V_ECSR_kt", np.nan), errors="coerce")

econ_docmin_tbl: List[float] = []
for _, row in display_tbl.iterrows():
    sc_name = str(row.get("_ScenarioName_raw", ""))
    sc = _scenario_lookup(scenarios, sc_name) if sc_name else None

    if sc is not None:
        try:
            econ_docmin_tbl.append(_scenario_docmin_econ_kt(sc, cfg))
        except Exception:
            econ_docmin_tbl.append(float("nan"))
    else:
        econ_docmin_tbl.append(float("nan"))

display_tbl["ECSR, kt"] = [
    _fmt_interval(lo, hi, vn, ve)
    for lo, hi, vn, ve in zip(
        e_lo.tolist(),
        e_hi.tolist(),
        v_notch_tbl.tolist(),
        econ_docmin_tbl,
    )
]

# 3) Add UI-friendly break-even strings
display_tbl["Laiko sąnaudų lūžio taškas, eur/h"] = _format_break_even_for_app_time(
    summary_tbl=summary_tbl,
    sweep_min=float(cfg.time_cost_min),
    sweep_max=float(cfg.time_cost_max),
)

display_tbl["Degalų sąnaudų lūžio taškas, eur/kg"] = _format_break_even_for_app_fuel(
    summary_tbl=summary_tbl,
    ceiling=fuel_ceiling,
)

# 4) Rename columns
rename_map = {
    "ScenarioName": "Bandymas",

    "ZP_ft": "Aukštis, ft",
    "WEIGHT_kg": "Masė, kg",
    "ISA_C": "ISA, °C",
    "WIND_kt": "Vėjo komponentė, kt",

    "FuelPrice_EurPerKg": "Degalų kaina, eur/kg",
    "TIME_COST_Operational_EurPerHr": "Laiko sąnaudos, eur/h",

    "V_notch_kt": "IASnotch, kt",
    "V_ECSR_kt": "ECON, kt",
    "ECSR_low_kt": "IASlow, kt",
    "ECSR_high_kt": "IAShigh, kt",

    "DOCmin_EurPerNM": "DOCmin, eur/nm",
    "DOCnotch_EurPerNM": "DOCnotch, eur/nm",
}

# Rename distance DOC columns (DOCmin_100NM_EUR -> DOCmin_100nm, eur)
for d in cfg.distances_nm:
    rename_map[f"DOCmin_{d}NM_EUR"] = f"DOCmin_{d}nm, eur"
    rename_map[f"DOCnotch_{d}NM_EUR"] = f"DOCnotch_{d}nm, eur"

display_tbl = display_tbl.rename(columns=rename_map)

for col in [
    "Aukštis, ft",
    "Masė, kg",
]:
    if col in display_tbl.columns:
        vals = pd.to_numeric(display_tbl[col], errors="coerce")
        display_tbl[col] = vals.map(lambda v: f"{int(round(v))}" if np.isfinite(v) else "")

if "IASnotch, kt" in display_tbl.columns:
    vals = pd.to_numeric(display_tbl["IASnotch, kt"], errors="coerce")
    display_tbl["IASnotch, kt"] = vals.map(lambda v: _fmt_speed_notch(v))

if "ECON, kt" in display_tbl.columns and "IASnotch, kt" in display_tbl.columns:
    econ_display_vals: List[str] = []

    for _, row in display_tbl.iterrows():
        sc_name = str(row.get("_ScenarioName_raw", ""))
        sc = _scenario_lookup(scenarios, sc_name) if sc_name else None
        notch_val = float(pd.to_numeric(row.get("IASnotch, kt", np.nan), errors="coerce"))

        if sc is not None:
            try:
                econ_docmin = _scenario_docmin_econ_kt(sc, cfg)
                econ_display_vals.append(_fmt_speed_econ_safe(econ_docmin, notch_val))
            except Exception:
                econ_display_vals.append("")
        else:
            econ_display_vals.append("")

    display_tbl["ECON, kt"] = econ_display_vals

if "IASlow, kt" in display_tbl.columns:
    vals = pd.to_numeric(display_tbl["IASlow, kt"], errors="coerce")
    display_tbl["IASlow, kt"] = vals.map(lambda v: _fmt_speed_econ(v))

if "IAShigh, kt" in display_tbl.columns:
    vals = pd.to_numeric(display_tbl["IAShigh, kt"], errors="coerce")
    display_tbl["IAShigh, kt"] = vals.map(lambda v: _fmt_speed_notch(v))

display_tbl = display_tbl.drop(
    columns=["BreakEven_TIME_COST_EurPerHr", "BreakEven_FUEL_PRICE_EurPerKg"],
    errors="ignore",
)

# 6) Reorder columns: meta -> costs -> speeds/ECSR -> break-evens -> DOC per NM -> DOC per distance
ordered_cols = [
    "Bandymas",
    "Aukštis, ft",
    "Masė, kg",
    "ISA, °C",
    "Vėjo komponentė, kt",

    "Degalų kaina, eur/kg",
    "Laiko sąnaudos, eur/h",

    "IASnotch, kt",
    "ECON, kt",
    "IASlow, kt",
    "IAShigh, kt",
    "ECSR, kt",

    "Laiko sąnaudų lūžio taškas, eur/h",
    "Degalų sąnaudų lūžio taškas, eur/kg",

    "DOCmin, eur/nm",
    "DOCnotch, eur/nm",
]

# Add distance columns next
for d in cfg.distances_nm:
    if int(d) == 100:
        ordered_cols.append(f"DOCmin_{d}nm, eur")
        ordered_cols.append(f"DOCnotch_{d}nm, eur")

# Keep only columns that exist (safe if config changes)

display_tbl = display_tbl.drop(
    columns=["DOCmin_EurPerHr", "DOCnotch_EurPerHr", "DOCmin, eur/h", "DOCnotch, eur/h"],
    errors="ignore",
)

ordered_cols = [c for c in ordered_cols if c in display_tbl.columns]

# Append any remaining columns at end (optional: keep nothing extra)
remaining = [c for c in display_tbl.columns if c not in ordered_cols]
display_tbl = display_tbl[ordered_cols + remaining]

display_tbl = display_tbl.drop(columns=["_ScenarioName_raw"], errors="ignore")

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
**WIND** – vėjo dedamoji skrydžio kryptimi (teigiamas – pavejui, neigiamas – priešinis vėjas), **kt**

**DOC** – tiesioginės eksploatacinės sąnaudos, **EUR** / Gali būti matuojamos ir 1 jūrmylei - tuomet **EUR/NM**  
**DOCmin_perNM** – minimalios DOC sąnaudos per **1 jūrmylią**, **EUR/NM**  
**DOCnotch_perNM** – DOC sąnaudos per **1 jūrmylią**, kai galios svirtis yra fiksuotoje padėtyje (angl. notch) arba kai lėktuvas skrenda **IASnotch** greičiu, **EUR/NM**  
**Sutaupymas_perX** – sutaupymas pasirinktame maršruto ilgyje X, **EUR**


**EPSILON** - nedidelė procentinė tolerancija, kurios dydis priklauso nuo oro linijų prioritetų ir kuri nusako, kokiu mastu sąnaudos gali padidėti virš **DOCmin**, **%**

**ECON** – ekonominis greitis, **kt**  
**IASnotch** – maksimalus kreiserinis greitis arba greitis prie tam tikrų sąlygų, kai galios svirtis yra fiksuotoje padėtyje, **kt**  
**ECSR** – ekonominis greičio diapazonas, gautas pritaikius epsilon įvertį **kt**
            """
        )
