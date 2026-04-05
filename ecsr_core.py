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
if __package__:
    from .ecsr_core import (
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
else:
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

@@ -191,75 +206,90 @@ def _econ_from_docmin_with_notch_rule(v_docmin: float, v_notch: float, *, min_ga
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


def _fmt_speed_econ(v: float) -> str:
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
def _fmt_speed_econ_safe(v_econ: float, v_notch: float, *, min_gap_kt: float = 1.0) -> str:
    econ_i = _safe_display_econ_kt(v_econ, v_notch)
    notch_i = _disp_notch_kt(v_notch)
    if econ_i is None:
        return ""

    # Avoid a UI paradox where savings are positive (raw gap >= min_gap_kt),
    # but rounded ECON and IASnotch appear identical.
    if notch_i is not None and int(econ_i) == int(notch_i) and _raw_speeds_differ(v_notch, v_econ, min_gap_kt=float(min_gap_kt)):
        return f"{float(v_econ):.1f}"

    return str(econ_i)

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

@@ -511,83 +541,84 @@ def _build_economical_scenarios_table(

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
    df["DeltaV_kt"] = df["IASnotch_disp_kt"] - df["ECON_safe_disp_kt"]
    df["DeltaV_disp_kt"] = df["IASnotch_disp_kt"] - df["ECON_safe_disp_kt"]
    df["DeltaV_raw_kt"] = pd.to_numeric(df["V_notch_kt"], errors="coerce") - pd.to_numeric(df["V_ECSR_kt"], errors="coerce")
    df["DocDiff_EurPerNM"] = df["DOCnotch_EurPerNM"] - df["DOCmin_EurPerNM"]

    for d in dist_cols:
        min_col = f"DOCmin_{d}NM_EUR"
        notch_col = f"DOCnotch_{d}NM_EUR"
        if min_col in df.columns and notch_col in df.columns:
            df[f"DocDiff_{d}NM_EUR"] = df[notch_col] - df[min_col]

    df = df.loc[(df["DeltaV_kt"] >= 1) & (df["DocDiff_EurPerNM"] > 0.0)].copy()
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
        "ΔV (kt)": df["DeltaV_kt"].round(0).astype(int),
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
@@ -1856,51 +1887,51 @@ def _plot_saving_vs_grouped(
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
        speed_gap_ok.append(_disp_speeds_differ(float(vn), float(ve), min_gap_kt=int(round(float(cfg.breakpoint_speed_tol_kt)))))
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
@@ -2114,51 +2145,51 @@ def _find_valid_breakpoint_from_curve(
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
        [_disp_speeds_differ(float(v_notch), float(v), min_gap_kt=int(round(float(cfg.breakpoint_speed_tol_kt)))) for v in y],
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
@@ -2766,77 +2797,76 @@ if mode == "Scenarijus":
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
                dist = float(distance_nm)

                sc = _scenario_lookup(scenarios, pick_scn)
                v_econ_docmin = float("nan")
                if sc is not None:
                    v_econ_docmin = _scenario_docmin_econ_kt(sc, cfg)

                if _disp_speeds_differ(v_notch_raw, v_econ_docmin, min_gap_kt=int(round(float(cfg.breakpoint_speed_tol_kt)))):
                    if np.isfinite(v_min_per_nm) and np.isfinite(v_notch_per_nm) and np.isfinite(dist):
                        diff_total = round(float((v_notch_per_nm - v_min_per_nm) * dist), 1)
                    else:
                        diff_total = float("nan")
                else:
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
                    if col_key == "V_ECSR_kt":
                        v_econ_docmin = float(pd.to_numeric(row.get("V_ECSR_kt", np.nan), errors="coerce"))
                        v_notch_ui = float(pd.to_numeric(row.get("V_notch_kt", np.nan), errors="coerce"))
                        if np.isfinite(v_econ_docmin) and np.isfinite(v_notch_ui):
                            shown_value = _fmt_speed_econ_safe(
                                v_econ_docmin,
                                v_notch_ui,
                                min_gap_kt=float(cfg.breakpoint_speed_tol_kt),
                            )
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
@@ -2907,160 +2937,98 @@ else:
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
                v_notch_ui = float(res_in.v_notch_kt)

                try:
                    doc_curve_res = _compute_input_doc_curve_knn(
                        scenarios,
                        summary_tbl,
                        cfg,
                        fl_ft=float(in_fl),
                        wt_kg=float(in_wt),
                        isa_c=float(in_isa),
                        wind_kt=float(in_wind),
                    )
                    econ_for_ui = float(doc_curve_res["IAS_opt"])
                except Exception:
                    econ_for_ui = float(res_in.v_ecsr_kt)

                if np.isfinite(econ_for_ui) and np.isfinite(v_notch_ui):
                    econ_for_ui = min(econ_for_ui, v_notch_ui)

                v = float(res_in.be_time_cost_eur_per_hr)
                valid_break = (
                    np.isfinite(v)
                    and (float(cfg.time_cost_min) - 1e-9 <= v <= float(cfg.time_cost_max) + 1e-9)
                    and _disp_speeds_differ(v_notch_ui, econ_for_ui, min_gap_kt=int(round(float(cfg.breakpoint_speed_tol_kt))))
                    and np.isfinite(res_in.docmin_eur_per_nm)
                    and np.isfinite(res_in.docnotch_eur_per_nm)
                    and float(res_in.docnotch_eur_per_nm) > float(res_in.docmin_eur_per_nm)
                )
                v = float(res_in.be_time_cost_eur_per_hr)
                valid_break = (
                    np.isfinite(v)
                    and (float(cfg.time_cost_min) - 1e-9 <= v <= float(cfg.time_cost_max) + 1e-9)
                )

                if valid_break:
                    shown_value = f"{int(round(v))}"
                    shown_unit = "€/h"
                else:
                    shown_value = "Nėra lūžio taško"
                    shown_unit = ""
            elif col_key == "__BREAK_FUEL__":
                v_notch_ui = float(res_in.v_notch_kt)

                try:
                    doc_curve_res = _compute_input_doc_curve_knn(
                        scenarios,
                        summary_tbl,
                        cfg,
                        fl_ft=float(in_fl),
                        wt_kg=float(in_wt),
                        isa_c=float(in_isa),
                        wind_kt=float(in_wind),
                    )
                    econ_for_ui = float(doc_curve_res["IAS_opt"])
                except Exception:
                    econ_for_ui = float(res_in.v_ecsr_kt)

                if np.isfinite(econ_for_ui) and np.isfinite(v_notch_ui):
                    econ_for_ui = min(econ_for_ui, v_notch_ui)

                v = float(res_in.be_fuel_price_eur_per_kg)
                valid_break = (
                    np.isfinite(v)
                    and v <= float(fuel_ceiling) + 1e-12
                    and _disp_speeds_differ(v_notch_ui, econ_for_ui, min_gap_kt=int(round(float(cfg.breakpoint_speed_tol_kt))))
                    and np.isfinite(res_in.docmin_eur_per_nm)
                    and np.isfinite(res_in.docnotch_eur_per_nm)
                    and float(res_in.docnotch_eur_per_nm) > float(res_in.docmin_eur_per_nm)
                )
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

                econ_for_ui = _input_docmin_econ_kt(
                    scenarios,
                    summary_tbl,
                    cfg,
                    fl_ft=float(in_fl),
                    wt_kg=float(in_wt),
                    isa_c=float(in_isa),
                    wind_kt=float(in_wind),
                    fallback_v_econ=float(res_in.v_ecsr_kt),
                    fallback_v_notch=float(v_notch_ui),
                )

                if _disp_speeds_differ(v_notch_ui, econ_for_ui, min_gap_kt=int(round(float(cfg.breakpoint_speed_tol_kt)))):
                    if np.isfinite(res_in.docmin_eur_per_nm) and np.isfinite(res_in.docnotch_eur_per_nm):
                        diff_total = round(float((res_in.docnotch_eur_per_nm - res_in.docmin_eur_per_nm) * dist), 1)
                    else:
                        diff_total = float("nan")
                else:
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
                if col_key == "V_ECSR_kt":
                    econ_val = float(res_in.v_ecsr_kt)
                    notch_val = float(res_in.v_notch_kt)
                    if np.isfinite(econ_val) and np.isfinite(notch_val):
                        shown_value = _fmt_speed_econ_safe(
                            econ_val,
                            notch_val,
                            min_gap_kt=float(cfg.breakpoint_speed_tol_kt),
                        )
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
@@ -3805,51 +3773,57 @@ for d in cfg.distances_nm:
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
                econ_display_vals.append(
                    _fmt_speed_econ_safe(
                        econ_docmin,
                        notch_val,
                        min_gap_kt=float(cfg.breakpoint_speed_tol_kt),
                    )
                )
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
ecsr_core.py
ecsr_core.py
+23
-14

@@ -598,59 +598,68 @@ def _econ_from_docmin_with_notch_rule(
    if not np.isfinite(v_notch):
        return float(v_docmin)

    if float(v_docmin) >= float(v_notch) - float(min_gap_kt):
        return float(v_notch)

    return float(v_docmin)


def _disp_econ_kt(v: Any) -> np.ndarray:
    x = np.asarray(v, float)
    out = np.full(x.shape, np.nan, dtype=float)
    ok = np.isfinite(x)
    out[ok] = np.ceil(x[ok])
    return out


def _disp_notch_kt(v: Any) -> np.ndarray:
    x = np.asarray(v, float)
    out = np.full(x.shape, np.nan, dtype=float)
    ok = np.isfinite(x)
    out[ok] = np.floor(x[ok])
    return out

def _display_speed_gap_ok(v_notch: Any, v_econ: Any, min_gap_kt: float) -> np.ndarray:
    """
    Display-only speed-gap check:
    IASnotch display = floor(v_notch)
    ECON display = ceil(v_econ)
    """
    vn = _disp_notch_kt(v_notch)
    ve = _disp_econ_kt(v_econ)
    return np.isfinite(vn) & np.isfinite(ve) & ((vn - ve) >= float(min_gap_kt))


def _continuous_speed_gap_ok(v_notch: Any, v_econ: Any, min_gap_kt: float) -> np.ndarray:
    """
    Economic/computational speed-gap rule on raw values (no display rounding).
    """
    vn = np.asarray(v_notch, float)
    ve = np.asarray(v_econ, float)
    return np.isfinite(vn) & np.isfinite(ve) & ((vn - ve) >= float(min_gap_kt))


def unique_x(x: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x).reshape(-1)
    a1 = np.asarray(a1).reshape(-1)
    a2 = np.asarray(a2).reshape(-1)
    m = np.isfinite(x) & np.isfinite(a1) & np.isfinite(a2)
    x = x[m]
    a1 = a1[m]
    a2 = a2[m]

    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    a1s = a1[order]
    a2s = a2[order]

    df = pd.DataFrame({"x": xs, "a1": a1s, "a2": a2s})
    g = df.groupby("x", sort=False, as_index=False).median(numeric_only=True)
    return g["x"].to_numpy(float), g["a1"].to_numpy(float), g["a2"].to_numpy(float)


def _pchip_fn(x: np.ndarray, y: np.ndarray) -> PchipInterpolator:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2:
@@ -1482,55 +1491,55 @@ def interpolate_curve_knn_from_scenarios(
        used += 1
        used_names.append(name)

    if used < int(min_neighbors):
        raise ValueError("Per silpna interpolacija: per mažai artimų scenarijų (kNN).")

    diag: Dict[str, Any] = {
        "k_requested": int(k),
        "neighbors_used": int(used),
        "neighbor_names": used_names,
        "neighbor_distances": di.astype(float).tolist(),
        "neighbor_weights": w.astype(float).tolist(),
        "bounds": bounds,
        "x_vec_key": str(x_vec_key),
        "y_vec_key": str(y_vec_key),
    }
    return y_acc, diag


# ========================= BREAKPOINTS + SUMMARY TABLES + PIPELINE =========================


def _use_money_gate(cfg: Config) -> bool:
    return str(cfg.breakpoint_saving_mode).strip().lower() in {"per_nm", "per_hour_trip"}

def _raw_speed_gap_ok(v_notch: Any, v_econ: Any, min_gap_kt: float) -> np.ndarray:
    """
    Computational gate for break-even/saving logic (raw speeds, no UI rounding).
    """
    return _continuous_speed_gap_ok(v_notch, v_econ, min_gap_kt)


def _doc_advantage_ok(doc_notch: Any, doc_econ: Any, *, atol: float = 1e-12) -> np.ndarray:
    dn = np.asarray(doc_notch, float)
    de = np.asarray(doc_econ, float)
    return np.isfinite(dn) & np.isfinite(de) & ((dn - de) > float(atol))

def _delta_doc_trip_avg_per_hour(
    doc_notch_per_nm: np.ndarray,
    doc_min_per_nm: np.ndarray,
    gs_notch_kt: np.ndarray,
    gs_econ_kt: np.ndarray,
    trip_distance_nm: float,
) -> np.ndarray:
    dist = float(trip_distance_nm)

    doc_notch_total = np.asarray(doc_notch_per_nm, float) * dist
    doc_econ_total = np.asarray(doc_min_per_nm, float) * dist

    time_econ_h = dist / np.asarray(gs_econ_kt, float)

    saving_total_trip = doc_notch_total - doc_econ_total
    return saving_total_trip / time_econ_h
