# =========================
# File: ecsr_core.py
# =========================
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, PchipInterpolator

# ========================= CONFIG =========================


@dataclass(frozen=True)
class KeyCfg:
    round_zp_ft: float = 10
    round_weight_kg: float = 10
    round_isa_c: float = 1
    round_wind_kt: float = 1
    hash_len: int = 10


@dataclass(frozen=True)
class ParseCfg:
    use_magnitude_hints: bool = True
    weight_typical_min: float = 12000
    weight_typical_max: float = 30000
    zp_typical_min: float = 0
    zp_typical_max: float = 30000


@dataclass(frozen=True)
class ExportCfg:
    write_excel: bool = True
    excel_filename: str = "ECSR_results.xlsx"
    write_csv: bool = False
    include_audit_cols: bool = False


@dataclass(frozen=True)
class BreakSearchCfg:
    """
    Auto break-even search settings (used for fuel break-even when grid sweep doesn't find it).
    """

    fuel_ceiling_eur_per_kg: float = 50.0
    fuel_expand_factor: float = 2.0
    fuel_bisect_abs_tol: float = 0.01
    fuel_bisect_max_iter: int = 40


@dataclass(frozen=True)
class Config:
    fuel_price_eur_per_kg: float = 0.575

    time_cost_min = 100
    time_cost_max: float = 5000
    time_cost_step: float = 10

    fuel_price_min: float = 0.20
    fuel_price_max: float = 3.00
    fuel_price_step: float = 0.02

    time_cost_operational: float = 1300.0
    epsilon_break_even: float = 0.01

    breakpoint_speed_tol_kt: float = 1.0
    breakpoint_saving_mode: str = "default"  # "default" | "per_nm" | "per_hour_trip"
    breakpoint_saving_eur_per_nm: float = 0.0
    breakpoint_saving_eur_per_hour: float = 0.0
    breakpoint_trip_distance_nm: float = 0.0

    delim_scan_lines: int = 25
    delim_candidates: Tuple[str, ...] = (",", ";", "\t")
    delim_trial_max_rows: int = 40

    required_columns: Tuple[str, ...] = (
        "ZP",
        "ISA",
        "WEIGHT",
        "IAS",
        "TAS",
        "WIND",
        "WFE",
        "TQ",
        "SHP",
        "SR",
        "SPEED_CODE",
    )
    delim_numeric_key_cols: Tuple[str, ...] = ("IAS", "TAS", "WIND", "WFE", "SPEED_CODE")

    min_gs_kt: float = 50
    max_gs_kt: float = 450
    min_fuel_kg_per_nm: float = 0.05
    max_fuel_kg_per_nm: float = 15
    min_rows: int = 5

    speed_code_bad_values: Tuple[float, ...] = (1.0, 0.99, 999.0)
    speed_code_tol: float = 1e-6

    distances_nm: Tuple[int, ...] = (100, 150, 200, 250, 300)

    key: KeyCfg = KeyCfg()
    parse: ParseCfg = ParseCfg()
    export: ExportCfg = ExportCfg()
    break_search: BreakSearchCfg = BreakSearchCfg()


def default_config() -> Config:
    return Config()

def _dbg(msg: str) -> None:
    print(f"[ECSR DEBUG] {msg}", flush=True)

# ========================= SORT =========================


def _scenario_sort_key(name: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)", str(name))
    num = int(m.group(1)) if m else 10**12
    return (num, str(name))


def _sort_tables_for_export(summary_tbl: pd.DataFrame, longform_tbl: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "ScenarioName" in summary_tbl.columns:
        summary_tbl = summary_tbl.sort_values(by="ScenarioName", key=lambda s: s.map(_scenario_sort_key)).reset_index(drop=True)

    if "ScenarioName" in longform_tbl.columns:
        lf = longform_tbl.copy()
        lf["_TIME_COST_NUM"] = pd.to_numeric(lf.get("TIME_COST", np.nan), errors="coerce")
        lf = (
            lf.sort_values(
                by=["ScenarioName", "_TIME_COST_NUM"],
                key=lambda s: s.map(_scenario_sort_key) if s.name == "ScenarioName" else s,
            )
            .drop(columns=["_TIME_COST_NUM"])
            .reset_index(drop=True)
        )
        longform_tbl = lf

    return summary_tbl, longform_tbl


# ========================= FILE DISCOVERY =========================


def list_scenario_files(root_dir: Path) -> List[Dict[str, str]]:
    patterns = ("*.csv", "*.CSV", "*.txt", "*.TXT", "*.csv.txt", "*.CSV.TXT")
    found: List[Path] = []
    for pat in patterns:
        found.extend(root_dir.glob(pat))

    found = [p for p in found if p.is_file()]
    unique_paths: List[Path] = []
    seen = set()
    for p in found:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique_paths.append(p)

    return [{"name": p.name, "fullpath": str(p)} for p in unique_paths]


# ========================= DELIMITER DETECTION =========================


def detect_delimiter_heuristic_fgetl(file_path: Path, scan_lines: int) -> str:
    lines: List[str] = []
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for _ in range(scan_lines):
            line = f.readline()
            if not line:
                break
            line = line.replace("\ufeff", "").strip()
            if line:
                lines.append(line)

    if not lines:
        raise ValueError("Empty file.")

    candidates = [",", ";", "\t"]
    best = ","
    best_score = -1e18
    for d in candidates:
        counts = np.array([ln.count(d) + 1 for ln in lines], dtype=float)
        score = float(counts.mean() - 0.25 * counts.std(ddof=0))
        if score > best_score:
            best_score = score
            best = d
    return best


def _trial_read_table(file_path: Path, delim: str, max_rows: int) -> pd.DataFrame:
    return pd.read_csv(
        file_path,
        sep=delim,
        header=0,
        skiprows=[1],
        nrows=max_rows,
        engine="python",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        skipinitialspace=True,
    )


def normalize_variable_names(df: pd.DataFrame) -> pd.DataFrame:
    vn = [str(c).upper() for c in df.columns]
    vn = [re.sub(r"[^A-Z0-9]+", "_", c) for c in vn]
    vn = [re.sub(r"_+", "_", c) for c in vn]
    vn = [re.sub(r"^_|_$", "", c) for c in vn]

    mapping = {
        "G_WT": "WEIGHT",
        "GW": "WEIGHT",
        "WEI_GHT": "WEIGHT",
        "WEIGHT_KG": "WEIGHT",
        "ZP_FT": "ZP",
        "ALT": "ZP",
        "ALTITUDE": "ZP",
        "IAS_KT": "IAS",
        "TAS_KT": "TAS",
        "WIND_KT": "WIND",
        "FF": "WFE",
        "FUEL_FLOW": "WFE",
        "TORQUE": "TQ",
        "SHAFT_HP": "SHP",
        "SPEEDCODE": "SPEED_CODE",
        "SPD_CODE": "SPEED_CODE",
        "SPEED_CODE_": "SPEED_CODE",
        "SPEED_CODE__": "SPEED_CODE",
        "SPDCD": "SPEED_CODE",
    }
    vn = [mapping.get(c, c) for c in vn]
    out = df.copy()
    out.columns = vn
    return out


def should_treat_as_decimal(val_thousands: float, col_name: str, cfg: Config) -> bool:
    if not np.isfinite(val_thousands):
        return False
    cn = str(col_name).upper()
    if cn == "WEIGHT":
        return bool(val_thousands < cfg.parse.weight_typical_min or val_thousands > cfg.parse.weight_typical_max)
    if cn == "ZP":
        return bool(val_thousands < cfg.parse.zp_typical_min or val_thousands > cfg.parse.zp_typical_max)
    return False


def _safe_float(s: str) -> float:
    try:
        return float(s)
    except Exception:
        return float("nan")


def to_numeric_smart(v: Any, col_name: str, cfg: Config) -> np.ndarray:
    if isinstance(v, (np.ndarray, list, tuple)) and len(v) == 0:
        return np.array([], dtype=float)

    if isinstance(v, (pd.Series, pd.Index)):
        s = v.astype(str)
    elif isinstance(v, (list, tuple, np.ndarray)):
        s = pd.Series(v).astype(str)
    else:
        try:
            return np.array([float(v)], dtype=float)
        except Exception:
            s = pd.Series([str(v)])

    s = s.str.strip().str.replace("\xa0", "", regex=False).str.replace(" ", "", regex=False)

    x = np.full(len(s), np.nan, dtype=float)
    is_thousands_candidate_col = str(col_name).upper() in {"WEIGHT", "ZP"}

    for i, si in enumerate(s.tolist()):
        if si in {"", "-", "NA", "N/A", "na", "n/a"}:
            continue

        has_comma = "," in si
        has_dot = "." in si

        if has_comma and has_dot:
            last_comma = si.rfind(",")
            last_dot = si.rfind(".")
            si2 = si.replace(",", "") if last_dot > last_comma else si.replace(".", "").replace(",", ".")
            x[i] = _safe_float(si2)
            continue

        if has_comma and not has_dot:
            parts = si.split(",")
            if len(parts) == 2:
                tail_len = len(parts[1])
                if tail_len == 3:
                    si_th = si.replace(",", "")
                    val_th = _safe_float(si_th)
                    if cfg.parse.use_magnitude_hints and should_treat_as_decimal(val_th, col_name, cfg):
                        x[i] = _safe_float(si.replace(",", "."))
                    else:
                        x[i] = val_th
                elif tail_len <= 2:
                    x[i] = _safe_float(si.replace(",", "."))
                else:
                    x[i] = _safe_float(si.replace(",", ""))
            else:
                x[i] = _safe_float(si.replace(",", ""))
            continue

        if has_dot and not has_comma:
            parts = si.split(".")
            if is_thousands_candidate_col and len(parts) >= 2 and all(len(p) == 3 for p in parts[1:]):
                x[i] = _safe_float(si.replace(".", ""))
            else:
                x[i] = _safe_float(si)
            continue

        x[i] = _safe_float(si)

    return x


def sanitize_name(s: str) -> str:
    s2 = re.sub(r"\s+", "_", str(s))
    s2 = re.sub(r"[^A-Za-z0-9_\-]+", "_", s2)
    s2 = re.sub(r"_+", "_", s2)
    s2 = re.sub(r"^_|_$", "", s2)
    return s2


def numeric_success_rate(T0: pd.DataFrame, key_cols: Sequence[str], cfg: Config) -> float:
    rates = []
    for c in key_cols:
        if c not in T0.columns:
            rates.append(0.0)
            continue
        x = to_numeric_smart(T0[c], c, cfg)
        rates.append(float(np.isfinite(x).mean()))
    return float(np.mean(rates)) if rates else 0.0


def detect_delimiter_robust_trial_import(file_path: Path, cfg: Config) -> Tuple[str, Dict[str, Any]]:
    heur = detect_delimiter_heuristic_fgetl(file_path, cfg.delim_scan_lines)
    cands = [heur] + [d for d in cfg.delim_candidates if d != heur]

    need = list(cfg.required_columns)
    best = ""
    best_score = -1e18
    best_detail: Dict[str, Any] = {}

    for d in cands:
        try:
            T0 = _trial_read_table(file_path, d, cfg.delim_trial_max_rows)
            if T0.empty:
                score = -1e18
                detail = {"ok": False, "why": "empty", "width": int(T0.shape[1]), "rows": int(T0.shape[0])}
            else:
                T0 = normalize_variable_names(T0)
                vn = list(T0.columns)
                missing = [c for c in need if c not in vn]
                miss_count = len(missing)

                if miss_count > 0:
                    score = 1000 - 200 * miss_count + T0.shape[1]
                    detail = {"ok": False, "missing": missing, "width": int(T0.shape[1]), "rows": int(T0.shape[0])}
                else:
                    ok_rate = numeric_success_rate(T0, cfg.delim_numeric_key_cols, cfg)
                    score = 2000 + 500 * ok_rate + 0.5 * T0.shape[1] + 0.1 * T0.shape[0]
                    detail = {"ok": True, "width": int(T0.shape[1]), "rows": int(T0.shape[0]), "numericOkRate": float(ok_rate)}
        except Exception as e:
            score = -1e18
            detail = {"ok": False, "why": "read_failed", "msg": str(e)}

        if score > best_score:
            best_score = score
            best = d
            best_detail = detail

    if not best:
        raise ValueError(f"Delimiter detection failed for file: {file_path}")

    return best, {"candidates": cands, "chosen": best, "detail": best_detail}


# ========================= CLEAN + DERIVED =========================


def stable_median(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


def compute_derived_masked_fixed_wind(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tas = df["TAS"].to_numpy(dtype=float)
    wind = df["WIND"].to_numpy(dtype=float)
    wfe = df["WFE"].to_numpy(dtype=float)

    gs = tas + wind
    fuel = np.full_like(gs, np.nan, dtype=float)
    time = np.full_like(gs, np.nan, dtype=float)

    ok = np.isfinite(gs) & (gs > 0) & np.isfinite(wfe)
    fuel[ok] = wfe[ok] / gs[ok]
    time[ok] = 1.0 / gs[ok]
    return gs, fuel, time


def sanity_mask(gs: np.ndarray, fuel: np.ndarray, time: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    n = gs.size
    valid = np.ones(n, dtype=bool)
    reasons = np.array(["OK"] * n, dtype=object)

    idx = ~np.isfinite(gs) | (gs < cfg.min_gs_kt) | (gs > cfg.max_gs_kt)
    valid[idx] = False
    reasons[idx] = "BadGS"

    idx = ~np.isfinite(fuel) | (fuel < cfg.min_fuel_kg_per_nm) | (fuel > cfg.max_fuel_kg_per_nm)
    valid[idx] = False
    reasons[idx] = "BadFuelKgPerNM"

    idx = ~np.isfinite(time) | (time <= 0)
    valid[idx] = False
    reasons[idx] = "BadTimeHrPerNM"

    return valid, reasons


def make_empty_outlier_table() -> pd.DataFrame:
    cols = [
        "ScenarioName",
        "FileName",
        "Reason",
        "Meta_ZP",
        "Meta_WEIGHT",
        "Meta_ISA",
        "Meta_WIND",
        "ZP",
        "WEIGHT",
        "ISA",
        "SPEED_CODE",
        "IAS",
        "TAS",
        "WIND",
        "WFE",
        "TQ",
        "SHP",
        "SR",
    ]
    return pd.DataFrame({c: pd.Series(dtype="object") for c in cols})


def build_outliers_table(
    tbad: pd.DataFrame,
    file_path: Path,
    scenario_name: str,
    reasons: np.ndarray,
    zp_med: float,
    w_med: float,
    isa_med: float,
    wind_med: float,
) -> pd.DataFrame:
    if tbad is None or tbad.empty:
        return make_empty_outlier_table()

    out = pd.DataFrame(
        {
            "ScenarioName": [scenario_name] * len(tbad),
            "FileName": [file_path.name] * len(tbad),
            "Reason": list(reasons.astype(str)),
            "Meta_ZP": [zp_med] * len(tbad),
            "Meta_WEIGHT": [w_med] * len(tbad),
            "Meta_ISA": [isa_med] * len(tbad),
            "Meta_WIND": [wind_med] * len(tbad),
        }
    )
    for c in ["ZP", "WEIGHT", "ISA", "SPEED_CODE", "IAS", "TAS", "WIND", "WFE", "TQ", "SHP", "SR"]:
        out[c] = tbad[c].to_numpy()
    return out


def process_scenario_file(file_path: Path, cfg: Config) -> Tuple[Dict[str, Any], pd.DataFrame]:
    delim, delim_debug = detect_delimiter_robust_trial_import(file_path, cfg)

    df = pd.read_csv(
        file_path,
        sep=delim,
        header=0,
        skiprows=[1],
        engine="python",
        dtype=str,
        keep_default_na=False,
        na_values=[],
        skipinitialspace=True,
    )
    if df.empty or len(df) < 3:
        raise ValueError("No usable data rows in file.")

    audit: Dict[str, Any] = {"nRawRows": int(len(df)), "delimiter": delim, "delimDebug": delim_debug}

    df = normalize_variable_names(df)

    missing = [c for c in cfg.required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {', '.join(missing)}")

    for c in cfg.required_columns:
        df[c] = to_numeric_smart(df[c], c, cfg)

    scode = df["SPEED_CODE"].to_numpy(dtype=float)
    bad = np.zeros(len(df), dtype=bool)
    for v in cfg.speed_code_bad_values:
        bad |= np.isfinite(scode) & (np.abs(scode - v) <= cfg.speed_code_tol)
    audit["nRemoved_speedCode"] = int(bad.sum())
    df = df.loc[~bad].copy()

    miss = ~np.isfinite(df["IAS"]) | ~np.isfinite(df["TAS"]) | ~np.isfinite(df["WIND"]) | ~np.isfinite(df["WFE"])
    audit["nRemoved_nanKey"] = int(miss.sum())
    df = df.loc[~miss].copy()

    if len(df) < cfg.min_rows:
        raise ValueError("Too few valid rows after SPEED_CODE + NaN filtering.")

    df = df.sort_values("IAS", kind="mergesort").reset_index(drop=True)

    gs, fuel, time = compute_derived_masked_fixed_wind(df)
    valid, reasons = sanity_mask(gs, fuel, time, cfg)
    audit["nRemoved_sanity"] = int((~valid).sum())

    zp_med = stable_median(df["ZP"].to_numpy(float))
    w_med = stable_median(df["WEIGHT"].to_numpy(float))
    isa_med = stable_median(df["ISA"].to_numpy(float))
    wind_med = stable_median(df["WIND"].to_numpy(float))

    base = file_path.stem + "".join(file_path.suffixes[1:])
    scenario_name = sanitize_name(base)

    outliers = build_outliers_table(
        df.loc[~valid].copy(),
        file_path,
        scenario_name,
        reasons[~valid],
        zp_med,
        w_med,
        isa_med,
        wind_med,
    )

    df = df.loc[valid].copy()
    if len(df) < cfg.min_rows:
        raise ValueError("Too few rows after sanity filtering.")

    gs, fuel, time = compute_derived_masked_fixed_wind(df)
    audit["nFinalRows"] = int(len(df))

    v_notch = float(np.nanmax(df["IAS"].to_numpy(float)))

    return (
        {
            "fileName": file_path.name,
            "scenarioName": scenario_name,
            "ZP_ft": float(zp_med),
            "WEIGHT_kg": float(w_med),
            "ISA_C": float(isa_med),
            "WIND_kt": float(wind_med),
            "pointTable": df,
            "Fuel_kg_per_NM": fuel,
            "Time_h_per_NM": time,
            "V_notch": v_notch,
            "audit": audit,
        },
        outliers,
    )


# ========================= CORE ECONOMICS =========================

def _econ_from_docmin_with_notch_rule(
    v_docmin: float,
    v_notch: float,
    *,
    min_gap_kt: float = 1.0,
) -> float:
    """
    ECON rule used by UI and core:
    - start from true DOC-min speed
    - if DOC-min is less than 1 kt below IASnotch, treat ECON as IASnotch
    - otherwise keep DOC-min speed
    """
    if not np.isfinite(v_docmin):
        return float("nan")
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
    Breakpoint/saving rule must follow displayed speeds:
    IASnotch display = floor(v_notch)
    ECON display = ceil(v_econ)
    """
    vn = _disp_notch_kt(v_notch)
    ve = _disp_econ_kt(v_econ)
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
        raise ValueError("Need at least 2 points for interpolation.")
    return PchipInterpolator(x, y, extrapolate=True)


def _build_fuel_time_interpolators(sc: Dict[str, Any]) -> Tuple[PchipInterpolator, PchipInterpolator, np.ndarray]:
    cache = sc.get("_cache")
    if isinstance(cache, dict):
        fuel_itp = cache.get("fuel_itp")
        time_itp = cache.get("time_itp")
        iasu = cache.get("iasu")
        if fuel_itp is not None and time_itp is not None and iasu is not None:
            return fuel_itp, time_itp, iasu

    df: pd.DataFrame = sc["pointTable"]
    ias = df["IAS"].to_numpy(float)
    fuel = sc["Fuel_kg_per_NM"]
    time = sc["Time_h_per_NM"]
    iasu, fuelu, timeu = unique_x(ias, fuel, time)
    fuel_itp = _pchip_fn(iasu, fuelu)
    time_itp = _pchip_fn(iasu, timeu)
    return fuel_itp, time_itp, iasu

def _prepare_scenario_runtime_cache(sc: Dict[str, Any]) -> Dict[str, Any]:
    fuel_itp, time_itp, iasu = _build_fuel_time_interpolators(sc)
    v_notch = float(sc["V_notch"])

    dense_grid = np.linspace(float(iasu.min()), float(iasu.max()), max(400, int(iasu.size * 40)))
    sweep_grid = np.linspace(float(iasu.min()), float(iasu.max()), max(120, int(iasu.size * 10)))

    sc["_cache"] = {
        "fuel_itp": fuel_itp,
        "time_itp": time_itp,
        "iasu": iasu,
        "v_notch": v_notch,
        "dense_grid": dense_grid,
        "sweep_grid": sweep_grid,
    }
    return sc


def compute_doc_curve_pchip(sc: Dict[str, Any], tc: float, cfg: Config, *, ngrid: int = 700) -> Dict[str, Any]:
    df: pd.DataFrame = sc["pointTable"]
    if df is None or not hasattr(df, "columns") or "IAS" not in df.columns:
        raise ValueError("Scenarijuje nėra 'pointTable' lentelės arba trūksta 'IAS' stulpelio.")

    ias_raw = pd.to_numeric(df["IAS"], errors="coerce").to_numpy(float)
    fuel_raw = np.asarray(sc.get("Fuel_kg_per_NM", []), float).reshape(-1)
    time_raw = np.asarray(sc.get("Time_h_per_NM", []), float).reshape(-1)
    n = min(ias_raw.size, fuel_raw.size, time_raw.size)
    if n < 2:
        raise ValueError("Per mažai taškų kreivei sudaryti (reikia bent 2).")

    ias_raw = ias_raw[:n]
    fuel_raw = fuel_raw[:n]
    time_raw = time_raw[:n]

    ok = np.isfinite(ias_raw) & np.isfinite(fuel_raw) & np.isfinite(time_raw)
    ias_raw = ias_raw[ok]
    fuel_raw = fuel_raw[ok]
    time_raw = time_raw[ok]
    if ias_raw.size < 2:
        raise ValueError("Po filtravimo liko per mažai galiojančių taškų.")

    order = np.argsort(ias_raw, kind="mergesort")
    ias_raw = ias_raw[order]
    fuel_raw = fuel_raw[order]
    time_raw = time_raw[order]

    fuel_itp, time_itp, iasu = _build_fuel_time_interpolators(sc)

    x_min = float(np.nanmin(iasu))
    x_max = float(np.nanmax(iasu))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        raise ValueError("Neteisingas IAS intervalas (min/max).")

    ngrid = int(max(180, min(320, ngrid)))
    ias_grid = np.linspace(x_min, x_max, ngrid)

    doc_grid = cfg.fuel_price_eur_per_kg * fuel_itp(ias_grid) + float(tc) * time_itp(ias_grid)
    j = int(np.nanargmin(doc_grid))
    v_opt = float(ias_grid[j])

    v_notch = float(sc.get("V_notch", np.nan))
    if not np.isfinite(v_notch):
        raise ValueError("Scenarijuje nėra korektiško 'V_notch'.")

    doc_opt = float(doc_grid[j])

    doc_notch = float(cfg.fuel_price_eur_per_kg * fuel_itp(np.array([v_notch]))[0] + float(tc) * time_itp(np.array([v_notch]))[0])
    doc_raw = cfg.fuel_price_eur_per_kg * fuel_raw + float(tc) * time_raw

    return {
        "IAS_grid": ias_grid,
        "DOC_grid_per_nm": np.asarray(doc_grid, float),
        "IAS_opt": v_opt,
        "DOC_opt_per_nm": doc_opt,
        "IAS_notch": v_notch,
        "DOC_notch_per_nm": doc_notch,
        "IAS_raw": ias_raw,
        "DOC_raw_per_nm": np.asarray(doc_raw, float),
    }


def compute_optimum_at_time_cost(sc: Dict[str, Any], tc: float, cfg: Config) -> Dict[str, float]:
    fuel_itp, time_itp, iasu = _build_fuel_time_interpolators(sc)
    v_notch = float(sc["V_notch"])

    def doc_at(x_ias: np.ndarray) -> np.ndarray:
        return cfg.fuel_price_eur_per_kg * fuel_itp(x_ias) + float(tc) * time_itp(x_ias)

    ngrid = max(400, int(iasu.size * 40))
    ias_grid = np.linspace(float(iasu.min()), float(iasu.max()), ngrid)
    doc_grid = doc_at(ias_grid)

    j = int(np.nanargmin(doc_grid))
    v_opt = float(ias_grid[j])
    doc_min = float(doc_grid[j])
    doc_notch = float(doc_at(np.array([v_notch]))[0])

    return {"IAS_opt_kt": v_opt, "DOC_min_EurPerNM": doc_min, "DOC_notch_EurPerNM": doc_notch}

def run_parametric_sweep(sc: Dict[str, Any], time_cost_vec: np.ndarray, cfg: Config) -> Dict[str, Any]:
    time_cost_vec = np.asarray(time_cost_vec, float).reshape(-1)
    sc["timeCostVec"] = time_cost_vec

    cache = sc.get("_cache", {})
    fuel_itp = cache.get("fuel_itp")
    time_itp = cache.get("time_itp")
    v_notch = cache.get("v_notch")
    ias_grid = cache.get("sweep_grid")

    if fuel_itp is None or time_itp is None or v_notch is None or ias_grid is None:
        v_notch = float(sc["V_notch"])
        fuel_itp, time_itp, iasu = _build_fuel_time_interpolators(sc)
        ngrid = max(120, int(iasu.size * 10))
        ias_grid = np.linspace(float(iasu.min()), float(iasu.max()), ngrid)

    fuel_grid = fuel_itp(ias_grid)
    time_grid = time_itp(ias_grid)
    fuel_notch = float(fuel_itp(np.array([v_notch]))[0])
    time_notch = float(time_itp(np.array([v_notch]))[0])

    doc_min = np.full(time_cost_vec.size, np.nan, dtype=float)
    ias_opt = np.full(time_cost_vec.size, np.nan, dtype=float)
    doc_notch = np.full(time_cost_vec.size, np.nan, dtype=float)

    for i, tc in enumerate(time_cost_vec):
        doc_grid = cfg.fuel_price_eur_per_kg * fuel_grid + float(tc) * time_grid
        j = int(np.nanargmin(doc_grid))
        doc_min[i] = float(doc_grid[j])
        ias_opt[i] = float(ias_grid[j])
        doc_notch[i] = float(cfg.fuel_price_eur_per_kg * fuel_notch + float(tc) * time_notch)

    sc["DOC_min_EurPerNM"] = doc_min
    sc["IAS_opt_kt"] = ias_opt
    sc["DOC_notch_EurPerNM"] = doc_notch
    return sc

def run_fuel_price_sweep(sc: Dict[str, Any], fuel_price_vec: np.ndarray, cfg: Config) -> Dict[str, Any]:
    if not isinstance(sc, dict):
        raise ValueError("Invalid scenario object in run_fuel_price_sweep (expected dict).")

    fuel_price_vec = np.asarray(fuel_price_vec, float).reshape(-1)
    sc["fuelPriceVec"] = fuel_price_vec

    tc_op = float(cfg.time_cost_operational)

    cache = sc.get("_cache", {})
    fuel_itp = cache.get("fuel_itp")
    time_itp = cache.get("time_itp")
    v_notch = cache.get("v_notch")
    ias_grid = cache.get("sweep_grid")

    if fuel_itp is None or time_itp is None or v_notch is None or ias_grid is None:
        v_notch = float(sc["V_notch"])
        fuel_itp, time_itp, iasu = _build_fuel_time_interpolators(sc)
        ngrid = max(120, int(iasu.size * 10))
        ias_grid = np.linspace(float(iasu.min()), float(iasu.max()), ngrid)

    fuel_grid = fuel_itp(ias_grid)
    time_grid = time_itp(ias_grid)
    fuel_notch = float(fuel_itp(np.array([v_notch]))[0])
    time_notch = float(time_itp(np.array([v_notch]))[0])

    doc_grid_2d = fuel_price_vec.reshape(-1, 1) * fuel_grid.reshape(1, -1) + float(tc_op) * time_grid.reshape(1, -1)
    j = np.nanargmin(doc_grid_2d, axis=1).astype(int)
    row_idx = np.arange(fuel_price_vec.size, dtype=int)

    doc_min = doc_grid_2d[row_idx, j].astype(float)
    ias_opt = ias_grid[j].astype(float)
    doc_notch = (fuel_price_vec * fuel_notch + float(tc_op) * time_notch).astype(float)

    sc["DOC_min_EurPerNM_fp"] = doc_min
    sc["IAS_opt_kt_fp"] = ias_opt
    sc["DOC_notch_EurPerNM_fp"] = doc_notch
    return sc

def compute_ecsr_band(sc: Dict[str, Any], tc: float, cfg: Config) -> Dict[str, float]:
    df: pd.DataFrame = sc["pointTable"]
    ias = df["IAS"].to_numpy(float)
    fuel = sc["Fuel_kg_per_NM"]
    time = sc["Time_h_per_NM"]
    iasu, fuelu, timeu = unique_x(ias, fuel, time)

    ngrid = max(160, int(iasu.size * 12))
    ias_grid = np.linspace(float(iasu.min()), float(iasu.max()), ngrid)

    fuel_grid = _pchip_fn(iasu, fuelu)(ias_grid)
    time_grid = _pchip_fn(iasu, timeu)(ias_grid)
    doc_grid = cfg.fuel_price_eur_per_kg * fuel_grid + float(tc) * time_grid

    i_min = int(np.nanargmin(doc_grid))
    doc_min = float(doc_grid[i_min])
    thr = doc_min * (1.0 + cfg.epsilon_break_even)

    ok = doc_grid <= thr
    if not np.any(ok):
        low = high = float(ias_grid[i_min])
    else:
        left = np.where(ok & (ias_grid <= ias_grid[i_min]))[0]
        right = np.where(ok & (ias_grid >= ias_grid[i_min]))[0]
        i_low = int(left[0]) if left.size else i_min
        i_high = int(right[-1]) if right.size else i_min
        low = float(ias_grid[i_low])
        high = float(ias_grid[i_high])

    return {"ECSR_low_kt": low, "ECSR_high_kt": high, "DOC_min_EurPerNM": doc_min}


# ========================= SUMMARY INTERPOLATION (ECSR calculator + input quick view) =========================


@dataclass(frozen=True)
class EcsrInterpResult:
    fl_ft: float
    weight_kg: float
    isa_c: float
    wind_kt: float
    v_ecsr_kt: float
    ecsr_low_kt: float
    ecsr_high_kt: float


@dataclass(frozen=True)
class InterpQuickResult:
    fl_ft: float
    weight_kg: float
    isa_c: float
    wind_kt: float
    v_ecsr_kt: float
    v_notch_kt: float
    ecsr_low_kt: float
    ecsr_high_kt: float
    docmin_eur_per_nm: float
    docnotch_eur_per_nm: float
    docmin_eur_per_h: float
    docnotch_eur_per_h: float
    be_time_cost_eur_per_hr: float
    be_fuel_price_eur_per_kg: float


def _require_columns(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Trūksta stulpelių interpolacijai: {', '.join(missing)}")


def _numeric_col(df: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def _validate_interp_dataset(summary_tbl: pd.DataFrame) -> pd.DataFrame:
    need = ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt"]
    _require_columns(summary_tbl, need)

    df = summary_tbl.copy()
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need)

    if df.shape[0] < 8:
        raise ValueError("Nepakanka scenarijų interpolacijai (reikia bent 8).")

    for c in need:
        if int(df[c].nunique()) < 2:
            raise ValueError(f"Nepakanka įvairovės interpolacijai: '{c}' turi mažiau nei 2 unikalias reikšmes.")

    return df


def _range_check(val: float, x: np.ndarray, label: str, unit: str) -> None:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Nepakanka duomenų interpolacijai.")

    lo = float(np.nanmin(x))
    hi = float(np.nanmax(x))

    if not (lo <= float(val) <= hi):
        def _fmt(v: float) -> str:
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            return f"{v:g}"

        raise ValueError(
            f"Neteisinga įvestis: {label} = {_fmt(float(val))} {unit}. "
            f"Galima riba: {_fmt(lo)}–{_fmt(hi)} {unit}."
        )


def compute_ecsr_band_interpolated(
    summary_tbl: pd.DataFrame,
    *,
    fl_ft: float,
    weight_kg: float,
    isa_c: float,
    wind_kt: float,
    min_points_required: int = 12,
) -> EcsrInterpResult:
    df = _validate_interp_dataset(summary_tbl)
    if df.shape[0] < int(min_points_required):
        raise ValueError(f"Nepakanka scenarijų stabiliai interpolacijai (reikia bent {int(min_points_required)}).")

    zp = _numeric_col(df, "ZP_ft")
    wt = _numeric_col(df, "WEIGHT_kg")
    isa = _numeric_col(df, "ISA_C")
    wnd = _numeric_col(df, "WIND_kt")

    _range_check(fl_ft, zp, "Aukštis", "ft")
    _range_check(weight_kg, wt, "Masė", "kg")
    _range_check(isa_c, isa, "ISA", "°C")
    _range_check(wind_kt, wnd, "Vėjas", "kt")

    pts = np.column_stack([zp, wt, isa, wnd]).astype(float)

    need = ["V_ECSR_kt", "V_notch_kt", "ECSR_low_kt", "ECSR_high_kt"]
    _require_columns(df, need)
    v_ecsr = _numeric_col(df, "V_ECSR_kt")
    v_notch = _numeric_col(df, "V_notch_kt")
    e_lo = _numeric_col(df, "ECSR_low_kt")
    e_hi = _numeric_col(df, "ECSR_high_kt")

    itp_v = LinearNDInterpolator(pts, v_ecsr, fill_value=np.nan)
    itp_notch = LinearNDInterpolator(pts, v_notch, fill_value=np.nan)
    itp_lo = LinearNDInterpolator(pts, e_lo, fill_value=np.nan)
    itp_hi = LinearNDInterpolator(pts, e_hi, fill_value=np.nan)

    q = np.array([[float(fl_ft), float(weight_kg), float(isa_c), float(wind_kt)]], dtype=float)

    v = float(itp_v(q)[0])
    v_notch_q = float(itp_notch(q)[0])
    lo = float(itp_lo(q)[0])
    hi = float(itp_hi(q)[0])

    if not (np.isfinite(v) and np.isfinite(v_notch_q) and np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("Negalima interpoliuoti šioms sąlygoms: trūksta duomenų.")
    
    v = min(v, v_notch_q)
    hi = min(hi, v_notch_q)
    lo = min(lo, hi)

    return EcsrInterpResult(
        fl_ft=float(fl_ft),
        weight_kg=float(weight_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
        v_ecsr_kt=float(v),
        ecsr_low_kt=float(min(lo, hi)),
        ecsr_high_kt=float(max(lo, hi)),
    )

def compute_econ_vs_time_cost_interpolated(
    longform_tbl: pd.DataFrame,
    *,
    fl_ft: float,
    weight_kg: float,
    isa_c: float,
    wind_kt: float,
    min_points_required: int = 8,
) -> pd.DataFrame:
    need = ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt", "TIME_COST", "IASopt"]
    _require_columns(longform_tbl, need)

    df = longform_tbl.copy()
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need)

    if df.empty:
        raise ValueError("Negalima atlikti interpoliacijos.")

    for col, label, unit, val in [
        ("ZP_ft", "Aukštis", "ft", fl_ft),
        ("WEIGHT_kg", "Masė", "kg", weight_kg),
        ("ISA_C", "ISA", "°C", isa_c),
        ("WIND_kt", "Vėjas", "kt", wind_kt),
    ]:
        _range_check(float(val), _numeric_col(df, col), label, unit)

    rows: List[Dict[str, float]] = []
    grouped = df.groupby("TIME_COST", sort=True)

    q = np.array([[float(fl_ft), float(weight_kg), float(isa_c), float(wind_kt)]], dtype=float)

    for tc, sub in grouped:
        if sub.shape[0] < int(min_points_required):
            continue

        pts = np.column_stack(
            [
                _numeric_col(sub, "ZP_ft"),
                _numeric_col(sub, "WEIGHT_kg"),
                _numeric_col(sub, "ISA_C"),
                _numeric_col(sub, "WIND_kt"),
            ]
        ).astype(float)
        y = _numeric_col(sub, "IASopt")

        ok = np.all(np.isfinite(pts), axis=1) & np.isfinite(y)
        pts = pts[ok]
        y = y[ok]
        if pts.shape[0] < int(min_points_required):
            continue

        val = float(LinearNDInterpolator(pts, y, fill_value=np.nan)(q)[0])
        if np.isfinite(val):
            rows.append({"TIME_COST": float(tc), "IASopt": float(val)})

    out = pd.DataFrame(rows)
    if out.shape[0] < 2:
        raise ValueError("Nepakanka taškų ECON priklausomybei nuo laiko sąnaudų sudaryti.")
    return out.sort_values("TIME_COST").reset_index(drop=True)

def compute_econ_vs_fuel_price_interpolated(
    fuel_longform_tbl: pd.DataFrame,
    *,
    fl_ft: float,
    weight_kg: float,
    isa_c: float,
    wind_kt: float,
    min_points_required: int = 8,
) -> pd.DataFrame:
    need = ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt", "FUEL_PRICE", "IASopt"]
    _require_columns(fuel_longform_tbl, need)

    df = fuel_longform_tbl.copy()
    for c in need:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=need)

    if df.empty:
        raise ValueError("Negalima atlikti interpoliacijos.")

    for col, label, unit, val in [
        ("ZP_ft", "Aukštis", "ft", fl_ft),
        ("WEIGHT_kg", "Masė", "kg", weight_kg),
        ("ISA_C", "ISA", "°C", isa_c),
        ("WIND_kt", "Vėjas", "kt", wind_kt),
    ]:
        _range_check(float(val), _numeric_col(df, col), label, unit)

    rows: List[Dict[str, float]] = []
    grouped = df.groupby("FUEL_PRICE", sort=True)

    q = np.array([[float(fl_ft), float(weight_kg), float(isa_c), float(wind_kt)]], dtype=float)

    for fp, sub in grouped:
        if sub.shape[0] < int(min_points_required):
            continue

        pts = np.column_stack(
            [
                _numeric_col(sub, "ZP_ft"),
                _numeric_col(sub, "WEIGHT_kg"),
                _numeric_col(sub, "ISA_C"),
                _numeric_col(sub, "WIND_kt"),
            ]
        ).astype(float)
        y = _numeric_col(sub, "IASopt")

        ok = np.all(np.isfinite(pts), axis=1) & np.isfinite(y)
        pts = pts[ok]
        y = y[ok]
        if pts.shape[0] < int(min_points_required):
            continue

        val = float(LinearNDInterpolator(pts, y, fill_value=np.nan)(q)[0])
        if np.isfinite(val):
            rows.append({"FUEL_PRICE": float(fp), "IASopt": float(val)})

    out = pd.DataFrame(rows)
    if out.shape[0] < 2:
        raise ValueError("Nepakanka taškų ECON priklausomybei nuo degalų kainos sudaryti.")
    return out.sort_values("FUEL_PRICE").reset_index(drop=True)

def _interp_scalar_4d(pts: np.ndarray, q: np.ndarray, y: np.ndarray, *, name: str) -> float:
    itp = LinearNDInterpolator(pts, y, fill_value=np.nan)
    v = float(itp(q)[0])
    if not np.isfinite(v):
        raise ValueError(f"Negalima interpoliuoti šioms sąlygoms: trūksta duomenų.")
    return v

@dataclass(frozen=True)
class SummaryInterpolators4D:
    pts: np.ndarray
    itp: Dict[str, LinearNDInterpolator]


def build_summary_interpolators_4d(
    summary_tbl: pd.DataFrame,
    *,
    min_points_required: int = 20,
) -> SummaryInterpolators4D:
    df = _validate_interp_dataset(summary_tbl)
    if df.shape[0] < int(min_points_required):
        raise ValueError(f"Nepakanka scenarijų stabiliai interpolacijai (reikia bent {int(min_points_required)}).")

    zp = _numeric_col(df, "ZP_ft")
    wt = _numeric_col(df, "WEIGHT_kg")
    isa = _numeric_col(df, "ISA_C")
    wnd = _numeric_col(df, "WIND_kt")

    pts = np.column_stack([zp, wt, isa, wnd]).astype(float)

    need = [
        "V_ECSR_kt",
        "V_notch_kt",
        "ECSR_low_kt",
        "ECSR_high_kt",
        "DOCmin_EurPerNM",
        "DOCnotch_EurPerNM",
        "DOCmin_EurPerHr",
        "DOCnotch_EurPerHr",
    ]
    _require_columns(df, need)

    interp_map: Dict[str, LinearNDInterpolator] = {}
    for col in need:
        y = _numeric_col(df, col)
        interp_map[col] = LinearNDInterpolator(pts, y, fill_value=np.nan)

    return SummaryInterpolators4D(pts=pts, itp=interp_map)

def compute_quick_metrics_interpolated_from_prebuilt(
    summary_tbl: pd.DataFrame,
    prebuilt: SummaryInterpolators4D,
    *,
    fl_ft: float,
    weight_kg: float,
    isa_c: float,
    wind_kt: float,
) -> InterpQuickResult:
    df = _validate_interp_dataset(summary_tbl)

    zp = _numeric_col(df, "ZP_ft")
    wt = _numeric_col(df, "WEIGHT_kg")
    isa = _numeric_col(df, "ISA_C")
    wnd = _numeric_col(df, "WIND_kt")

    _range_check(fl_ft, zp, "Aukštis", "ft")
    _range_check(weight_kg, wt, "Masė", "kg")
    _range_check(isa_c, isa, "ISA", "°C")
    _range_check(wind_kt, wnd, "Vėjas", "kt")

    q = np.array([[float(fl_ft), float(weight_kg), float(isa_c), float(wind_kt)]], dtype=float)

    def _eval(name: str) -> float:
        v = float(prebuilt.itp[name](q)[0])
        if not np.isfinite(v):
            raise ValueError("Negalima interpoliuoti šioms sąlygoms: trūksta duomenų.")
        return v

    lo_v = _eval("ECSR_low_kt")
    hi_v = _eval("ECSR_high_kt")
    v_ecsr_v = _eval("V_ECSR_kt")
    v_notch_v = _eval("V_notch_kt")
    v_ecsr_v = min(v_ecsr_v, v_notch_v)

    docmin_nm_v = _eval("DOCmin_EurPerNM")
    docnotch_nm_v = _eval("DOCnotch_EurPerNM")
    docmin_h_v = _eval("DOCmin_EurPerHr")
    docnotch_h_v = _eval("DOCnotch_EurPerHr")

    return InterpQuickResult(
        fl_ft=float(fl_ft),
        weight_kg=float(weight_kg),
        isa_c=float(isa_c),
        wind_kt=float(wind_kt),
        v_ecsr_kt=v_ecsr_v,
        v_notch_kt=v_notch_v,
        ecsr_low_kt=float(min(lo_v, hi_v)),
        ecsr_high_kt=float(max(lo_v, hi_v)),
        docmin_eur_per_nm=docmin_nm_v,
        docnotch_eur_per_nm=docnotch_nm_v,
        docmin_eur_per_h=docmin_h_v,
        docnotch_eur_per_h=docnotch_h_v,
        be_time_cost_eur_per_hr=float("nan"),
        be_fuel_price_eur_per_kg=float("nan"),
    )


def compute_quick_metrics_interpolated(
    summary_tbl: pd.DataFrame,
    *,
    fl_ft: float,
    weight_kg: float,
    isa_c: float,
    wind_kt: float,
    min_points_required: int = 20,
) -> InterpQuickResult:
    prebuilt = build_summary_interpolators_4d(
        summary_tbl,
        min_points_required=min_points_required,
    )
    return compute_quick_metrics_interpolated_from_prebuilt(
        summary_tbl,
        prebuilt,
        fl_ft=fl_ft,
        weight_kg=weight_kg,
        isa_c=isa_c,
        wind_kt=wind_kt,
    )


# ========================= GLOBAL POINT CLOUD (for Įvestis Graph 1) =========================


def build_global_point_cloud(scenarios: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for sc in scenarios:
        df: pd.DataFrame = sc.get("pointTable")
        if df is None or df.empty or "IAS" not in df.columns:
            continue

        ias = pd.to_numeric(df["IAS"], errors="coerce").to_numpy(float)
        fuel = np.asarray(sc.get("Fuel_kg_per_NM", []), float).reshape(-1)
        time = np.asarray(sc.get("Time_h_per_NM", []), float).reshape(-1)
        n = min(ias.size, fuel.size, time.size)
        if n < 2:
            continue

        ias = ias[:n]
        fuel = fuel[:n]
        time = time[:n]
        ok = np.isfinite(ias) & np.isfinite(fuel) & np.isfinite(time)
        ias, fuel, time = ias[ok], fuel[ok], time[ok]
        if ias.size < 2:
            continue

        for x, f, t in zip(ias.tolist(), fuel.tolist(), time.tolist()):
            rows.append(
                {
                    "ZP_ft": float(sc["ZP_ft"]),
                    "WEIGHT_kg": float(sc["WEIGHT_kg"]),
                    "ISA_C": float(sc["ISA_C"]),
                    "WIND_kt": float(sc["WIND_kt"]),
                    "IAS": float(x),
                    "Fuel_kg_per_NM": float(f),
                    "Time_h_per_NM": float(t),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("Nepavyko sukurti global point-cloud (tuščia).")

    need = ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt", "IAS", "Fuel_kg_per_NM", "Time_h_per_NM"]
    _require_columns(out, need)
    out = out.dropna(subset=need)
    if out.shape[0] < 100:
        raise ValueError("Per mažai taškų globaliame point-cloud (reikia bent ~100).")
    return out


def compute_doc_curve_interpolated_from_cloud(
    cloud: pd.DataFrame,
    *,
    fl_ft: float,
    weight_kg: float,
    isa_c: float,
    wind_kt: float,
    time_cost_eur_per_hr: float,
    fuel_price_eur_per_kg: float,
    ngrid: int = 700,
    min_valid_grid_points: int = 60,
) -> Dict[str, Any]:
    need = ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt", "IAS", "Fuel_kg_per_NM", "Time_h_per_NM"]
    _require_columns(cloud, need)

    for col, label, unit, val in [
        ("ZP_ft", "Aukštis", "ft", fl_ft),
        ("WEIGHT_kg", "Masė", "kg", weight_kg),
        ("ISA_C", "ISA", "°C", isa_c),
        ("WIND_kt", "Vėjas", "kt", wind_kt),
    ]:
        x = pd.to_numeric(cloud[col], errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError(f"Nėra galiojančių reikšmių '{col}' rėžiui nustatyti.")
        _range_check(float(val), x, label, unit)

    ias_all = pd.to_numeric(cloud["IAS"], errors="coerce").to_numpy(float)
    ias_all = ias_all[np.isfinite(ias_all)]
    if ias_all.size < 10:
        raise ValueError("Per mažai IAS taškų globaliame cloud.")
    ias_min = float(np.nanmin(ias_all))
    ias_max = float(np.nanmax(ias_all))
    if not (np.isfinite(ias_min) and np.isfinite(ias_max) and ias_max > ias_min):
        raise ValueError("Neteisingas globalus IAS intervalas.")

    pts = np.column_stack(
        [
            pd.to_numeric(cloud["ZP_ft"], errors="coerce").to_numpy(float),
            pd.to_numeric(cloud["WEIGHT_kg"], errors="coerce").to_numpy(float),
            pd.to_numeric(cloud["ISA_C"], errors="coerce").to_numpy(float),
            pd.to_numeric(cloud["WIND_kt"], errors="coerce").to_numpy(float),
            pd.to_numeric(cloud["IAS"], errors="coerce").to_numpy(float),
        ]
    ).astype(float)
    y_fuel = pd.to_numeric(cloud["Fuel_kg_per_NM"], errors="coerce").to_numpy(float)
    y_time = pd.to_numeric(cloud["Time_h_per_NM"], errors="coerce").to_numpy(float)

    ok = np.all(np.isfinite(pts), axis=1) & np.isfinite(y_fuel) & np.isfinite(y_time)
    pts = pts[ok]
    y_fuel = y_fuel[ok]
    y_time = y_time[ok]
    if pts.shape[0] < 100:
        raise ValueError("Per mažai taškų 5D interpolacijai (Fuel/Time).")

    itp_fuel = LinearNDInterpolator(pts, y_fuel, fill_value=np.nan)
    itp_time = LinearNDInterpolator(pts, y_time, fill_value=np.nan)

    ngrid = int(max(300, ngrid))
    ias_grid = np.linspace(ias_min, ias_max, ngrid)

    q = np.column_stack(
        [
            np.full_like(ias_grid, float(fl_ft)),
            np.full_like(ias_grid, float(weight_kg)),
            np.full_like(ias_grid, float(isa_c)),
            np.full_like(ias_grid, float(wind_kt)),
            ias_grid,
        ]
    ).astype(float)

    fuel_grid = np.asarray(itp_fuel(q), float).reshape(-1)
    time_grid = np.asarray(itp_time(q), float).reshape(-1)
    good = np.isfinite(fuel_grid) & np.isfinite(time_grid)

    if int(good.sum()) < int(min_valid_grid_points):
        raise ValueError("Negalima sudaryti DOC kreivės: per silpna 5D interpolacija (per mažai galiojančių taškų).")

    ias_grid2 = ias_grid[good]
    fuel_grid2 = fuel_grid[good]
    time_grid2 = time_grid[good]

    doc_grid = float(fuel_price_eur_per_kg) * fuel_grid2 + float(time_cost_eur_per_hr) * time_grid2
    j = int(np.nanargmin(doc_grid))
    v_opt_raw = float(ias_grid2[j])

    v_notch_candidates = pd.to_numeric(cloud["IAS"], errors="coerce").to_numpy(float)
    v_notch_candidates = v_notch_candidates[np.isfinite(v_notch_candidates)]
    v_notch = float(np.nanmax(v_notch_candidates)) if v_notch_candidates.size else float("nan")

    v_opt = _econ_from_docmin_with_notch_rule(
        v_opt_raw,
        v_notch,
        min_gap_kt=1.0,
    )
    doc_opt = float(
        float(fuel_price_eur_per_kg) * itp_fuel(
            np.array([[float(fl_ft), float(weight_kg), float(isa_c), float(wind_kt), v_opt]], dtype=float)
        )[0]
        + float(time_cost_eur_per_hr) * itp_time(
            np.array([[float(fl_ft), float(weight_kg), float(isa_c), float(wind_kt), v_opt]], dtype=float)
        )[0]
    )

    return {
        "IAS_grid": ias_grid2,
        "DOC_grid_per_nm": np.asarray(doc_grid, float),
        "IAS_opt": v_opt,
        "DOC_opt_per_nm": doc_opt,
        "IAS_raw": np.array([], dtype=float),
        "DOC_raw_per_nm": np.array([], dtype=float),
    }


# ========================= LONGFORM FUEL TABLE (optional in app) =========================


def build_longform_fuel_table(scenarios: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        fp = np.asarray(sc.get("fuelPriceVec", []), float).reshape(-1)
        iasopt = np.asarray(sc.get("IAS_opt_kt_fp", []), float).reshape(-1)
        docmin = np.asarray(sc.get("DOC_min_EurPerNM_fp", []), float).reshape(-1)
        docnotch = np.asarray(sc.get("DOC_notch_EurPerNM_fp", []), float).reshape(-1)

        n = min(fp.size, iasopt.size, docmin.size, docnotch.size)
        if n < 2:
            continue

        fp, iasopt, docmin, docnotch = fp[:n], iasopt[:n], docmin[:n], docnotch[:n]
        ok = np.isfinite(fp) & np.isfinite(iasopt) & np.isfinite(docmin) & np.isfinite(docnotch)
        fp, iasopt, docmin, docnotch = fp[ok], iasopt[ok], docmin[ok], docnotch[ok]
        if fp.size < 2:
            continue

        for x, y, d1, d2 in zip(fp.tolist(), iasopt.tolist(), docmin.tolist(), docnotch.tolist()):
            rows.append(
                {
                    "ZP_ft": float(sc["ZP_ft"]),
                    "WEIGHT_kg": float(sc["WEIGHT_kg"]),
                    "ISA_C": float(sc["ISA_C"]),
                    "WIND_kt": float(sc["WIND_kt"]),
                    "FUEL_PRICE": float(x),
                    "IASopt": float(y),
                    "DOCmin": float(d1),
                    "DOCnotch": float(d2),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "ZP_ft",
                "WEIGHT_kg",
                "ISA_C",
                "WIND_kt",
                "FUEL_PRICE",
                "IASopt",
                "DOCmin",
                "DOCnotch",
            ]
        )

    return pd.DataFrame(rows)


# ========================= FAST kNN INTERPOLATION (Įvestis Graph 2/3) =========================


def _scenario_meta_frame(scenarios: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        rows.append(
            {
                "ScenarioName": str(sc.get("scenarioName", "")),
                "ZP_ft": float(sc.get("ZP_ft", np.nan)),
                "WEIGHT_kg": float(sc.get("WEIGHT_kg", np.nan)),
                "ISA_C": float(sc.get("ISA_C", np.nan)),
                "WIND_kt": float(sc.get("WIND_kt", np.nan)),
            }
        )
    df = pd.DataFrame(rows)
    for c in ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt"]).reset_index(drop=True)
    if df.shape[0] < 8:
        raise ValueError("Nepakanka scenarijų kNN interpolacijai (reikia bent 8).")
    return df


def _bounds_from_meta(meta: pd.DataFrame) -> Dict[str, Tuple[float, float]]:
    b: Dict[str, Tuple[float, float]] = {}
    for c in ["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt"]:
        x = meta[c].to_numpy(float)
        b[c] = (float(np.nanmin(x)), float(np.nanmax(x)))
    return b


def _check_inputs_in_bounds(fl_ft: float, weight_kg: float, isa_c: float, wind_kt: float, bounds: Dict[str, Tuple[float, float]]) -> None:
    for name, val in [("ZP_ft", fl_ft), ("WEIGHT_kg", weight_kg), ("ISA_C", isa_c), ("WIND_kt", wind_kt)]:
        lo, hi = bounds[name]
        if not (lo <= float(val) <= hi):
            label = {"ZP_ft": "Aukštis", "WEIGHT_kg": "Masė", "ISA_C": "ISA", "WIND_kt": "Vėjas"}[name]
            unit = {"ZP_ft": "ft", "WEIGHT_kg": "kg", "ISA_C": "°C", "WIND_kt": "kt"}[name]

            def _fmt(v: float) -> str:
                if abs(v - round(v)) < 1e-9:
                    return str(int(round(v)))
                return f"{v:g}"

            raise ValueError(
                f"Neteisinga įvestis: {label} = {_fmt(float(val))} {unit}. "
                f"Galima riba: {_fmt(lo)}–{_fmt(hi)} {unit}."
            )


def _normalized_distances(
    meta: pd.DataFrame,
    *,
    fl_ft: float,
    weight_kg: float,
    isa_c: float,
    wind_kt: float,
    bounds: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    eps = 1e-12
    q = np.array([fl_ft, weight_kg, isa_c, wind_kt], dtype=float)
    X = meta[["ZP_ft", "WEIGHT_kg", "ISA_C", "WIND_kt"]].to_numpy(float)

    scales = np.array(
        [
            max(bounds["ZP_ft"][1] - bounds["ZP_ft"][0], eps),
            max(bounds["WEIGHT_kg"][1] - bounds["WEIGHT_kg"][0], eps),
            max(bounds["ISA_C"][1] - bounds["ISA_C"][0], eps),
            max(bounds["WIND_kt"][1] - bounds["WIND_kt"][0], eps),
        ],
        dtype=float,
    )
    d = (X - q.reshape(1, 4)) / scales.reshape(1, 4)
    return np.sqrt(np.sum(d * d, axis=1))


def interpolate_curve_knn_from_scenarios(
    scenarios: List[Dict[str, Any]],
    *,
    fl_ft: float,
    weight_kg: float,
    isa_c: float,
    wind_kt: float,
    x_grid: np.ndarray,
    x_vec_key: str,
    y_vec_key: str,
    k: int = 30,
    power: float = 2.0,
    min_neighbors: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Local weighted interpolation using only nearest scenarios.

    Returns (y_grid, diag) so app can do:
      y_grid, _diag = interpolate_curve_knn_from_scenarios(...)
    """
    meta = _scenario_meta_frame(scenarios)
    bounds = _bounds_from_meta(meta)
    _check_inputs_in_bounds(fl_ft, weight_kg, isa_c, wind_kt, bounds)

    d = _normalized_distances(meta, fl_ft=float(fl_ft), weight_kg=float(weight_kg), isa_c=float(isa_c), wind_kt=float(wind_kt), bounds=bounds)
    order = np.argsort(d, kind="mergesort")

    k = int(max(1, min(int(k), int(meta.shape[0]))))
    idx = order[:k]
    di = d[idx]

    eps = 1e-12
    if float(np.nanmin(di)) <= eps:
        w = np.zeros_like(di, dtype=float)
        w[int(np.argmin(di))] = 1.0
    else:
        w = 1.0 / np.power(di + eps, float(power))
    w_sum = float(np.sum(w))
    if not np.isfinite(w_sum) or w_sum <= 0:
        raise ValueError("Per silpna interpolacija: nepavyko sudaryti svorių (kNN).")
    w = w / w_sum

    xg = np.asarray(x_grid, float).reshape(-1)
    y_acc = np.zeros_like(xg, dtype=float)

    used = 0
    used_names: List[str] = []

    sc_map: Dict[str, Dict[str, Any]] = {str(s.get("scenarioName", "")): s for s in scenarios}

    for jj, scen_row_idx in enumerate(idx.tolist()):
        name = str(meta.loc[scen_row_idx, "ScenarioName"])
        sc = sc_map.get(name)
        if sc is None:
            continue

        xv = np.asarray(sc.get(x_vec_key, []), float).reshape(-1)
        yv = np.asarray(sc.get(y_vec_key, []), float).reshape(-1)
        n = min(xv.size, yv.size)
        if n < 2:
            continue

        xv = xv[:n]
        yv = yv[:n]
        ok = np.isfinite(xv) & np.isfinite(yv)
        xv = xv[ok]
        yv = yv[ok]
        if xv.size < 2:
            continue

        o2 = np.argsort(xv, kind="mergesort")
        xv = xv[o2]
        yv = yv[o2]

        y_acc += float(w[jj]) * np.interp(xg, xv, yv)
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
    Kept for backward compatibility, but now follows displayed-speed logic.
    """
    return _display_speed_gap_ok(v_notch, v_econ, min_gap_kt)


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


def _first_true_x(x: np.ndarray, cond: np.ndarray) -> float:
    x = np.asarray(x, float).reshape(-1)
    cond = np.asarray(cond, bool).reshape(-1)

    ok = np.isfinite(x)
    x = x[ok]
    cond = cond[ok]
    if x.size < 2:
        return float("nan")

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    cond = cond[order]

    if not np.any(cond):
        return float("nan")
    k = int(np.where(cond)[0][0])
    return float(x[k])


def _last_true_x(x: np.ndarray, cond: np.ndarray) -> float:
    x = np.asarray(x, float).reshape(-1)
    cond = np.asarray(cond, bool).reshape(-1)

    ok = np.isfinite(x)
    x = x[ok]
    cond = cond[ok]
    if x.size < 2:
        return float("nan")

    order = np.argsort(x, kind="mergesort")
    x = x[order]
    cond = cond[order]

    if not np.any(cond):
        return float("nan")
    if np.all(cond):
        return float("inf")
    k = int(np.where(cond)[0][-1])
    return float(x[k])


def _econ_exists_at_time_cost(sc: Dict[str, Any], tc: float, cfg: Config) -> bool:
    cur = current_operating_point_result(
        sc,
        fuel_price_eur_per_kg=float(cfg.fuel_price_eur_per_kg),
        time_cost_eur_per_hr=float(tc),
        cfg=cfg,
    )
    return bool(cur["econ_exists"])


def _econ_exists_at_fuel_price(sc: Dict[str, Any], fp: float, cfg: Config) -> bool:
    cur = current_operating_point_result(
        sc,
        fuel_price_eur_per_kg=float(fp),
        time_cost_eur_per_hr=float(cfg.time_cost_operational),
        cfg=cfg,
    )
    return bool(cur["econ_exists"])


def _optimum_at_fuel_price(sc: Dict[str, Any], fp: float, cfg: Config) -> Tuple[float, float, float]:
    tc_op = float(cfg.time_cost_operational)
    cur = current_operating_point_result(
        sc,
        fuel_price_eur_per_kg=float(fp),
        time_cost_eur_per_hr=tc_op,
        cfg=cfg,
    )
    return float(cur["v_econ"]), float(cur["doc_econ_per_nm"]), float(cur["doc_notch_per_nm"])


def _worth_at_fuel_price(sc: Dict[str, Any], fp: float, cfg: Config) -> bool:
    cur = current_operating_point_result(
        sc,
        fuel_price_eur_per_kg=float(fp),
        time_cost_eur_per_hr=float(cfg.time_cost_operational),
        cfg=cfg,
    )
    return bool(int(round(float(cur["econ_exists"]))))


def current_operating_point_result(
    sc: Dict[str, Any],
    *,
    fuel_price_eur_per_kg: float,
    time_cost_eur_per_hr: float,
    cfg: Config,
) -> Dict[str, float]:
    """
    Evaluate current operating point consistently.

    Returns final current-point result after applying:
    - DOC-min search
    - ECON/notch rule
    - displayed-speed gap rule
    - DOC advantage rule
    - optional money gate

    If ECON does not truly exist at current operating point, result is collapsed to IASnotch.
    """
    cache = sc.get("_cache", {})
    fuel_itp = cache.get("fuel_itp")
    time_itp = cache.get("time_itp")
    iasu = cache.get("iasu")
    v_notch = cache.get("v_notch")
    ias_grid = cache.get("dense_grid")

    if fuel_itp is None or time_itp is None or iasu is None or v_notch is None or ias_grid is None:
        v_notch = float(sc.get("V_notch", np.nan))
        if not np.isfinite(v_notch):
            return {
                "econ_exists": 0.0,
                "v_docmin_raw": float("nan"),
                "v_econ": float("nan"),
                "v_notch": float("nan"),
                "doc_econ_per_nm": float("nan"),
                "doc_notch_per_nm": float("nan"),
                "gs_econ_kt": float("nan"),
                "gs_notch_kt": float("nan"),
            }

        fuel_itp, time_itp, iasu = _build_fuel_time_interpolators(sc)
        ngrid = max(400, int(iasu.size * 40))
        ias_grid = np.linspace(float(iasu.min()), float(iasu.max()), ngrid)

    doc_grid = (
        float(fuel_price_eur_per_kg) * fuel_itp(ias_grid)
        + float(time_cost_eur_per_hr) * time_itp(ias_grid)
    )
    j = int(np.nanargmin(doc_grid))
    v_docmin_raw = float(ias_grid[j])

    v_econ_raw = float(v_docmin_raw)
    doc_econ_raw = float(
        float(fuel_price_eur_per_kg) * fuel_itp(np.array([v_econ_raw], dtype=float))[0]
        + float(time_cost_eur_per_hr) * time_itp(np.array([v_econ_raw], dtype=float))[0]
    )
    gs_econ_raw = float(1.0 / time_itp(np.array([v_econ_raw], dtype=float))[0])

    v_econ = _econ_from_docmin_with_notch_rule(
        v_docmin_raw,
        v_notch,
        min_gap_kt=float(cfg.breakpoint_speed_tol_kt),
    )

    doc_econ = float(
        float(fuel_price_eur_per_kg) * fuel_itp(np.array([v_econ], dtype=float))[0]
        + float(time_cost_eur_per_hr) * time_itp(np.array([v_econ], dtype=float))[0]
    )
    doc_notch = float(
        float(fuel_price_eur_per_kg) * fuel_itp(np.array([v_notch], dtype=float))[0]
        + float(time_cost_eur_per_hr) * time_itp(np.array([v_notch], dtype=float))[0]
    )

    gs_econ = float(1.0 / time_itp(np.array([v_econ], dtype=float))[0])
    gs_notch = float(1.0 / time_itp(np.array([v_notch], dtype=float))[0])

    speed_ok = bool(
        _display_speed_gap_ok(
            np.array([v_notch], dtype=float),
            np.array([v_econ], dtype=float),
            float(cfg.breakpoint_speed_tol_kt),
        )[0]
    )
    econ_ok = bool(
        _doc_advantage_ok(
            np.array([doc_notch], dtype=float),
            np.array([doc_econ], dtype=float),
        )[0]
    )

    money_ok = True
    if _use_money_gate(cfg):
        mode = str(cfg.breakpoint_saving_mode).strip().lower()

        if mode == "per_nm":
            money_ok = bool((doc_notch - doc_econ) >= float(cfg.breakpoint_saving_eur_per_nm))

        elif mode == "per_hour_trip":
            delta_val = float(
                _delta_doc_trip_avg_per_hour(
                    doc_notch_per_nm=np.array([doc_notch], dtype=float),
                    doc_min_per_nm=np.array([doc_econ], dtype=float),
                    gs_notch_kt=np.array([gs_notch], dtype=float),
                    gs_econ_kt=np.array([gs_econ], dtype=float),
                    trip_distance_nm=float(cfg.breakpoint_trip_distance_nm),
                )[0]
            )
            money_ok = bool(delta_val >= float(cfg.breakpoint_saving_eur_per_hour))

    econ_exists = bool(speed_ok and econ_ok and money_ok)

    if not econ_exists:
        v_econ = float(v_notch)
        doc_econ = float(doc_notch)
        gs_econ = float(gs_notch)

    return {
        "econ_exists": float(econ_exists),
        "v_docmin_raw": float(v_docmin_raw),
        "v_econ": float(v_econ),
        "v_notch": float(v_notch),
        "doc_econ_per_nm": float(doc_econ),
        "doc_notch_per_nm": float(doc_notch),
        "gs_econ_kt": float(gs_econ),
        "gs_notch_kt": float(gs_notch),
        "v_econ_raw": float(v_econ_raw),
        "doc_econ_raw_per_nm": float(doc_econ_raw),
        "gs_econ_raw_kt": float(gs_econ_raw),
        "positive_saving_raw": float(doc_notch > doc_econ_raw),
    }

def _find_fuel_break_even_auto(sc: Dict[str, Any], cfg: Config) -> float:
    start = float(cfg.fuel_price_min)
    ceiling = float(cfg.break_search.fuel_ceiling_eur_per_kg)
    factor = float(cfg.break_search.fuel_expand_factor)
    tol = float(cfg.break_search.fuel_bisect_abs_tol)
    max_iter = int(cfg.break_search.fuel_bisect_max_iter)

    if start <= 0 or ceiling <= start or factor <= 1.0:
        return float("inf")

    if _worth_at_fuel_price(sc, start, cfg):
        return start

    lo = start
    hi = max(start * factor, start + tol)

    while hi <= ceiling and not _worth_at_fuel_price(sc, hi, cfg):
        lo = hi
        hi = hi * factor
        if not np.isfinite(hi):
            return float("inf")

    if hi > ceiling:
        if not _worth_at_fuel_price(sc, ceiling, cfg):
            return float("inf")
        hi = ceiling

    for _ in range(max_iter):
        if (hi - lo) <= tol:
            return hi
        mid = 0.5 * (lo + hi)
        if _worth_at_fuel_price(sc, mid, cfg):
            hi = mid
        else:
            lo = mid

    return hi

def _positive_saving_at_time_cost(sc: Dict[str, Any], tc: float, cfg: Config) -> bool:
    cur = current_operating_point_result(
        sc,
        fuel_price_eur_per_kg=float(cfg.fuel_price_eur_per_kg),
        time_cost_eur_per_hr=float(tc),
        cfg=cfg,
    )
    saving = float(cur["doc_notch_per_nm"] - cur["doc_econ_per_nm"])
    return bool(np.isfinite(saving) and saving > 0.0)


def _positive_saving_at_fuel_price(sc: Dict[str, Any], fp: float, cfg: Config) -> bool:
    cur = current_operating_point_result(
        sc,
        fuel_price_eur_per_kg=float(fp),
        time_cost_eur_per_hr=float(cfg.time_cost_operational),
        cfg=cfg,
    )
    saving = float(cur["doc_notch_per_nm"] - cur["doc_econ_per_nm"])
    return bool(np.isfinite(saving) and saving > 0.0)

def compute_threshold_x_time_break(sc: Dict[str, Any], cfg: Config) -> Dict[str, float]:
    """
    Time breakpoint:
    - below breakpoint -> ECON exists (ECON < IASnotch, saving exists)
    - at/above breakpoint -> ECON collapses to IASnotch (no saving)

    Returns the first time cost where ECON no longer exists.
    """
    tc = np.asarray(sc.get("timeCostVec", []), float).reshape(-1)
    tc = tc[np.isfinite(tc)]
    if tc.size < 2:
        return {"X_timeCost_EurPerHr": float("nan")}

    tc = np.sort(tc, kind="mergesort")

    econ_exists = np.array(
        [_econ_exists_at_time_cost(sc, float(t), cfg) for t in tc.tolist()],
        dtype=bool,
    )

    if not np.any(econ_exists):
        return {"X_timeCost_EurPerHr": float(tc[0])}

    if np.all(econ_exists):
        return {"X_timeCost_EurPerHr": float("inf")}

    had_econ = False
    for t, ok in zip(tc.tolist(), econ_exists.tolist()):
        if ok:
            had_econ = True
            continue
        if had_econ:
            return {"X_timeCost_EurPerHr": float(t)}

    return {"X_timeCost_EurPerHr": float("inf")}


def compute_threshold_x_fuel_break(sc: Dict[str, Any], cfg: Config) -> Dict[str, float]:
    """
    Fuel breakpoint:
    - below breakpoint -> ECON does not exist (ECON = IASnotch, no saving)
    - at/above breakpoint -> ECON exists (ECON < IASnotch, saving exists)

    Returns the first fuel price where ECON starts to exist.
    """
    fp = np.asarray(sc.get("fuelPriceVec", []), float).reshape(-1)
    fp = fp[np.isfinite(fp)]
    if fp.size < 2:
        return {"X_fuelPrice_EurPerKg": float("nan")}

    fp = np.sort(fp, kind="mergesort")

    econ_exists = np.array(
        [_econ_exists_at_fuel_price(sc, float(p), cfg) for p in fp.tolist()],
        dtype=bool,
    )

    if np.all(econ_exists):
        return {"X_fuelPrice_EurPerKg": float(fp[0])}

    if not np.any(econ_exists):
        return {"X_fuelPrice_EurPerKg": float("inf")}

    for p, ok in zip(fp.tolist(), econ_exists.tolist()):
        if ok:
            return {"X_fuelPrice_EurPerKg": float(p)}

    return {"X_fuelPrice_EurPerKg": float("inf")}


def compute_break_even_time_cost_rounded(sc: Dict[str, Any], x_continuous: float, cfg: Config) -> float:
    if np.isnan(float(x_continuous)):
        return float("nan")
    if np.isinf(float(x_continuous)):
        return float("inf")
    return float(x_continuous)


def compute_break_even_fuel_price_rounded(sc: Dict[str, Any], x_continuous: float, cfg: Config) -> float:
    if np.isnan(float(x_continuous)):
        return float("nan")
    if np.isinf(float(x_continuous)):
        return float("inf")
    return float(x_continuous)


def _positive_saving_mask(doc_min: np.ndarray, doc_notch: np.ndarray) -> np.ndarray:
    doc_min = np.asarray(doc_min, float)
    doc_notch = np.asarray(doc_notch, float)
    return np.isfinite(doc_min) & np.isfinite(doc_notch) & ((doc_notch - doc_min) > 0.0)


def build_summary_table(scenarios: List[Dict[str, Any]], cfg: Config) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    tc_op = float(cfg.time_cost_operational)

    for sc in scenarios:
        th_t = compute_threshold_x_time_break(sc, cfg)
        tc_be = compute_break_even_time_cost_rounded(sc, float(th_t["X_timeCost_EurPerHr"]), cfg)

        th_f = compute_threshold_x_fuel_break(sc, cfg)
        fp_be = compute_break_even_fuel_price_rounded(sc, float(th_f["X_fuelPrice_EurPerKg"]), cfg)

        cur_now = current_operating_point_result(
            sc,
            fuel_price_eur_per_kg=float(cfg.fuel_price_eur_per_kg),
            time_cost_eur_per_hr=float(tc_op),
            cfg=cfg,
        )

        v_notch = float(cur_now["v_notch"])

        v_ecsr = float(cur_now["v_econ"])
        docmin = float(cur_now["doc_econ_per_nm"])
        gs_ecsr = float(cur_now["gs_econ_kt"])

        docnotch = float(cur_now["doc_notch_per_nm"])
        gs_notch = float(cur_now["gs_notch_kt"])

        docmin_per_h = float(docmin * gs_ecsr) if np.isfinite(docmin) and np.isfinite(gs_ecsr) else float("nan")
        docnotch_per_h = float(docnotch * gs_notch) if np.isfinite(docnotch) and np.isfinite(gs_notch) else float("nan")

        ecsr = compute_ecsr_band(sc, tc_op, cfg)
        low = float(ecsr["ECSR_low_kt"])
        high = float(ecsr["ECSR_high_kt"])

        row: Dict[str, Any] = {
            "ScenarioName": sc["scenarioName"],
            "ZP_ft": sc["ZP_ft"],
            "WEIGHT_kg": sc["WEIGHT_kg"],
            "ISA_C": sc["ISA_C"],
            "WIND_kt": sc["WIND_kt"],
            "FuelPrice_EurPerKg": float(cfg.fuel_price_eur_per_kg),
            "BreakEven_TIME_COST_EurPerHr": float(tc_be),
            "BreakEven_FUEL_PRICE_EurPerKg": float(fp_be),
            "TIME_COST_Operational_EurPerHr": float(tc_op),
            "V_notch_kt": v_notch,
            "V_ECSR_kt": v_ecsr,
            "ECSR_low_kt": low,
            "ECSR_high_kt": high,
            "DOCmin_EurPerHr": docmin_per_h,
            "DOCnotch_EurPerHr": docnotch_per_h,
            "DOCmin_EurPerNM": docmin,
            "DOCnotch_EurPerNM": docnotch,
        }

        for d in cfg.distances_nm:
            if int(d) == 100:
                row[f"DOCmin_{d}NM_EUR"] = docmin * float(d)
                row[f"DOCnotch_{d}NM_EUR"] = docnotch * float(d)

        rows.append(row)

    return pd.DataFrame(rows)


def build_longform_table(scenarios: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for sc in scenarios:
        for tc, docmin, iasopt, docnotch in zip(
            sc["timeCostVec"],
            sc["DOC_min_EurPerNM"],
            sc["IAS_opt_kt"],
            sc["DOC_notch_EurPerNM"],
        ):
            rows.append(
                {
                    "ScenarioName": sc["scenarioName"],
                    "ZP_ft": sc["ZP_ft"],
                    "WEIGHT_kg": sc["WEIGHT_kg"],
                    "ISA_C": sc["ISA_C"],
                    "WIND_kt": sc["WIND_kt"],
                    "TIME_COST": float(tc),
                    "IASopt": float(iasopt),
                    "DOCmin": float(docmin),
                    "DOCnotch": float(docnotch),
                }
            )
    return pd.DataFrame(rows)


def write_excel_results(
    out_dir: Path,
    summary_tbl: pd.DataFrame,
    longform_tbl: pd.DataFrame,
    outlier_rows: pd.DataFrame,
    cfg: Config,
    run_info: Dict[str, Any],
) -> Path:
    xlsx_path = out_dir / cfg.export.excel_filename
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as w:
        summary_tbl.to_excel(w, sheet_name="Summary", index=False)
        longform_tbl.to_excel(w, sheet_name="Longform", index=False)
        outlier_rows.to_excel(w, sheet_name="Outliers", index=False)
        info_df = pd.DataFrame([{"Key": k, "Value": str(v)} for k, v in run_info.items()])
        info_df.to_excel(w, sheet_name="RunInfo", index=False)
    return xlsx_path


RunPipelineReturn = Union[
    Tuple[Path, Optional[Path], pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]],
    Tuple[Path, Optional[Path], pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[Dict[str, Any]]],
]


def run_pipeline(
    root_dir: Path,
    cfg: Config,
    *,
    output_parent: Optional[Path] = None,
    return_scenarios: bool = False,
) -> RunPipelineReturn:
    root_dir = root_dir.resolve()
    output_parent = (output_parent or root_dir).resolve()

    _dbg(f"run_pipeline start | root_dir={root_dir}")

    out_dir = output_parent / f"ECSR_Output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _dbg(f"output dir created | out_dir={out_dir}")

    files = list_scenario_files(root_dir)
    _dbg(f"files discovered | count={len(files)}")

    if not files:
        raise RuntimeError("No scenario files found. Expected *.csv, *.txt, *.csv.txt, *.CSV, etc.")

    scenarios: List[Dict[str, Any]] = []
    outlier_frames: List[pd.DataFrame] = []
    logs: List[str] = []

    for i, f in enumerate(files, start=1):
        fp = Path(f["fullpath"])
        _dbg(f"processing file {i}/{len(files)} | name={f['name']}")
        try:
            sc, outliers = process_scenario_file(fp, cfg)
            logs.append(f"OK: {f['name']} | rows(raw/final)={sc['audit']['nRawRows']}/{sc['audit']['nFinalRows']}")
            scenarios.append(sc)
            if outliers is not None and not outliers.empty:
                outlier_frames.append(outliers)
            _dbg(f"processed OK | scenario={sc['scenarioName']}")
        except Exception as e:
            logs.append(f"FAIL: {f['name']} -> {e}")
            _dbg(f"processed FAIL | name={f['name']} | error={e}")

    scenarios = [sc for sc in scenarios if isinstance(sc, dict)]
    _dbg(f"scenario processing finished | ok_count={len(scenarios)}")

    if not scenarios:
        raise RuntimeError("No scenarios processed successfully.")

    _dbg("building per-scenario runtime cache")
    for i in range(len(scenarios)):
        if i % 25 == 0 or i == len(scenarios) - 1:
            _dbg(f"cache progress | {i+1}/{len(scenarios)}")
        scenarios[i] = _prepare_scenario_runtime_cache(scenarios[i])
    _dbg("runtime cache done")

    outlier_rows = pd.concat(outlier_frames, ignore_index=True) if outlier_frames else make_empty_outlier_table()
    _dbg(f"outlier table ready | rows={len(outlier_rows)}")

    time_cost_vec = np.arange(cfg.time_cost_min, cfg.time_cost_max + 1e-9, cfg.time_cost_step, dtype=float)
    _dbg(f"time sweep start | steps={len(time_cost_vec)} | scenarios={len(scenarios)}")
    for i in range(len(scenarios)):
        if i % 25 == 0 or i == len(scenarios) - 1:
            _dbg(f"time sweep progress | {i+1}/{len(scenarios)}")
        scenarios[i] = run_parametric_sweep(scenarios[i], time_cost_vec, cfg)
    _dbg("time sweep done")

    fuel_price_vec = np.arange(cfg.fuel_price_min, cfg.fuel_price_max + 1e-12, cfg.fuel_price_step, dtype=float)
    _dbg(f"fuel sweep start | steps={len(fuel_price_vec)} | scenarios={len(scenarios)}")
    for i in range(len(scenarios)):
        if i % 25 == 0 or i == len(scenarios) - 1:
            _dbg(f"fuel sweep progress | {i+1}/{len(scenarios)}")
        scenarios[i] = run_fuel_price_sweep(scenarios[i], fuel_price_vec, cfg)
    _dbg("fuel sweep done")

    _dbg("building summary table")
    summary_tbl = build_summary_table(scenarios, cfg)
    _dbg(f"summary table done | shape={summary_tbl.shape}")

    _dbg("building longform table")
    longform_tbl = build_longform_table(scenarios)
    _dbg(f"longform table done | shape={longform_tbl.shape}")

    summary_tbl, longform_tbl = _sort_tables_for_export(summary_tbl, longform_tbl)
    _dbg("sort tables done")

    xlsx_path: Optional[Path] = None
    base = (out_dir, xlsx_path, summary_tbl, longform_tbl, outlier_rows, logs)

    _dbg("run_pipeline success")

    if return_scenarios:
        return (*base, scenarios)
    return base
