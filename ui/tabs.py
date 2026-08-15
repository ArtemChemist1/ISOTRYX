import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from datetime import datetime, timezone
from pyteomics import mzml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from scipy.integrate import trapezoid
from utils.isotope_correction import (
    DEFAULT_ISODATA,
    build_tracer_purity_vector,
    multi_element_correction,
    natural_abundance_enrichment,
    parse_formula,
    parse_tracer,
    weighted_mean_enrichment,
)
from utils.data_ops import (
    extract_blank_max_heights,
    generate_targets,
    read_metadata,
)
from utils.math_ops import (
    fails_blank_fold_change,
    integrate_exact_bounds,
    local_minimum_resolver_extract_area,
)




# Add this near the top of tabs.py
TRACER_LABELS = {
    "13C": "13C/12C",
    "15N": "15N/14N",
    "2H": "2H/1H",
    "18O": "18O/16O",
    "34S": "34S/32S"
}

MANUAL_EDIT_COLUMNS = {
    'Manual_Edit_Type', 'Manual_RT_Start', 'Manual_RT_End',
    'Manual_Algorithm', 'Manual_Smoothing', 'Manual_Isotopologues',
    'Manual_Edit_Timestamp', 'Manual_Edit_Log',
    'Local_Baseline_Applied', 'Baseline_Flank_min', 'Baseline_Min_Scans',
    'N_Blanks_Used', 'Manual_Correction_Details',
}


def _parse_manual_edit_log(value):
    if value is None or pd.isna(value) or str(value).strip() == '':
        return []
    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else []
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def _append_manual_edit_event(df, row_idx, edit_type, isotopologues,
                              rt_bounds=None, algorithm=None, smoothing=None,
                              baseline_enabled=False, baseline_flank_min=None,
                              baseline_min_scans=None,
                              n_blanks_used=0, correction_details=None):
    numeric_edit_cols = {
        'Manual_RT_Start', 'Manual_RT_End', 'Manual_Smoothing',
        'Baseline_Flank_min', 'Baseline_Min_Scans', 'N_Blanks_Used',
    }
    for col in MANUAL_EDIT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col in numeric_edit_cols else None
        elif col not in numeric_edit_cols and df[col].dtype != object:
            # Empty text columns read from CSV may be inferred as float; make
            # their dtype explicit before assigning audit strings/JSON.
            df[col] = df[col].astype(object)

    timestamp = datetime.now(timezone.utc).isoformat()
    baseline_enabled = bool(baseline_enabled)
    event = {
        'type': edit_type,
        'isotopologues': list(isotopologues),
        'rt_start': float(rt_bounds[0]) if rt_bounds is not None else None,
        'rt_end': float(rt_bounds[1]) if rt_bounds is not None else None,
        'algorithm': algorithm,
        'smoothing': float(smoothing) if smoothing is not None else None,
        'baseline_enabled': baseline_enabled,
        'baseline_flank_min': (
            float(baseline_flank_min)
            if baseline_enabled and baseline_flank_min is not None else None
        ),
        'baseline_min_scans': (
            int(baseline_min_scans)
            if baseline_enabled and baseline_min_scans is not None else None
        ),
        'n_blanks_used': int(n_blanks_used or 0),
        'correction_details': correction_details or {},
        'timestamp': timestamp,
    }
    events = _parse_manual_edit_log(df.at[row_idx, 'Manual_Edit_Log'])
    events.append(event)

    df.at[row_idx, 'Manual_Edit_Type'] = edit_type
    df.at[row_idx, 'Manual_RT_Start'] = event['rt_start']
    df.at[row_idx, 'Manual_RT_End'] = event['rt_end']
    df.at[row_idx, 'Manual_Algorithm'] = algorithm
    df.at[row_idx, 'Manual_Smoothing'] = event['smoothing']
    df.at[row_idx, 'Manual_Isotopologues'] = json.dumps(list(isotopologues))
    df.at[row_idx, 'Local_Baseline_Applied'] = bool(baseline_enabled)
    df.at[row_idx, 'Baseline_Flank_min'] = event['baseline_flank_min']
    df.at[row_idx, 'Baseline_Min_Scans'] = event['baseline_min_scans']
    df.at[row_idx, 'N_Blanks_Used'] = event['n_blanks_used']
    df.at[row_idx, 'Manual_Correction_Details'] = json.dumps(
        event['correction_details'], separators=(',', ':')
    )
    df.at[row_idx, 'Manual_Edit_Timestamp'] = timestamp
    df.at[row_idx, 'Manual_Edit_Log'] = json.dumps(events, separators=(',', ':'))
    return event


def _latest_manual_event_for_iso(row, iso_label):
    for event in reversed(_parse_manual_edit_log(row.get('Manual_Edit_Log'))):
        if iso_label in event.get('isotopologues', []):
            return event
    return None


def _manual_integration_mask(rt_arr, int_arr, rt_bounds, algorithm, smoothing):
    sigma = float(smoothing or 0.0)
    smooth_int = gaussian_filter1d(int_arr, sigma=sigma) if sigma > 0 else int_arr.astype(float)
    window_mask = (rt_arr >= float(rt_bounds[0])) & (rt_arr <= float(rt_bounds[1]))
    return smooth_int, window_mask


def _add_rt_boundary_markers(fig, rt_bounds):
    """Show the exact user-selected integration window on a Plotly figure."""
    rt_start, rt_end = sorted((float(rt_bounds[0]), float(rt_bounds[1])))
    fig.add_vrect(
        x0=rt_start,
        x1=rt_end,
        fillcolor="rgba(46, 204, 113, 0.10)",
        line_width=0,
        layer="below",
    )
    fig.add_vline(
        x=rt_start,
        line_color="#1B5E20",
        line_width=2,
        line_dash="dash",
        annotation_text=f"Start {rt_start:.3f} min",
        annotation_position="top left",
    )
    fig.add_vline(
        x=rt_end,
        line_color="#B71C1C",
        line_width=2,
        line_dash="dash",
        annotation_text=f"End {rt_end:.3f} min",
        annotation_position="top right",
    )
    return fig


def _render_manual_edit_history(df):
    if df is None or df.empty or 'Manual_Edit_Log' not in df.columns:
        return

    audit_rows = []
    for _, row in df.iterrows():
        for event in _parse_manual_edit_log(row.get('Manual_Edit_Log')):
            audit_rows.append({
                'Timestamp (UTC)': event.get('timestamp'),
                'Filename': row.get('Filename'),
                'Metabolite': row.get('Metabolite'),
                'Ion mode': row.get('Ion_Mode'),
                'Action': event.get('type'),
                'Isotopologues': ', '.join(event.get('isotopologues', [])),
                'RT start': event.get('rt_start'),
                'RT end': event.get('rt_end'),
                'Algorithm': event.get('algorithm'),
                'Smoothing': event.get('smoothing'),
                'Local baseline': event.get('baseline_enabled', False),
                'Baseline flank (min)': (
                    event.get('baseline_flank_min')
                    if event.get('baseline_enabled', False) else None
                ),
                'Blanks used': event.get('n_blanks_used', 0),
            })

    if audit_rows:
        with st.expander(f"🛠️ Manual Edit History ({len(audit_rows)} events)", expanded=False):
            st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)


def _run_peak_picker(rt_array, int_array, mz_array, target_rt, params):
    return local_minimum_resolver_extract_area(rt_array=rt_array, int_array=int_array,
                                               mz_array=mz_array, target_rt=target_rt,
                                               params=params)


def _build_composite_eic(met):
    populated = [iso for iso in met['isotopes'] if len(iso['eic_rt']) > 0]
    if not populated:
        return None, None

    # Always anchor the RT grid to M+0 because it best matches the metadata RT
    # and is the most reliably detected isotopologue.  Fall back to the most
    # intense isotopologue only if M+0 has no signal at all.
    m0_candidates = [iso for iso in populated if iso['isotope_number'] == 0]
    base_iso = m0_candidates[0] if m0_candidates else max(
        populated, key=lambda iso: np.nanmax(iso['eic_int']) if len(iso['eic_int']) else 0
    )
    base_sort = np.argsort(base_iso['eic_rt'])
    composite_rt = np.array(base_iso['eic_rt'], dtype=float)[base_sort]
    composite_int = np.zeros_like(composite_rt, dtype=float)

    for iso in populated:
        sort_idx = np.argsort(iso['eic_rt'])
        rt_arr = np.array(iso['eic_rt'], dtype=float)[sort_idx]
        int_arr = np.array(iso['eic_int'], dtype=float)[sort_idx]
        if len(rt_arr) == len(composite_rt) and np.allclose(rt_arr, composite_rt):
            composite_int += int_arr
        elif len(rt_arr) > 1:
            composite_int += np.interp(composite_rt, rt_arr, int_arr, left=0.0, right=0.0)
    return composite_rt, composite_int


def _calculate_enrichment_metrics(raw_areas, formula, tracer_str, params):
    tracer_el, _ = parse_tracer(tracer_str, DEFAULT_ISODATA)
    n_tracer = len(raw_areas) - 1
    formula = dict(formula)
    formula[tracer_el] = n_tracer
    tracer_purity_vector = build_tracer_purity_vector(tracer_str, params['tracer_purity'], DEFAULT_ISODATA)
    corrected_areas, iso_frac, corrected_me = multi_element_correction(
        measured_areas=raw_areas,
        formula=formula,
        tracer_str=tracer_str,
        tracer_purity=tracer_purity_vector,
        correct_NA_tracer=params.get('correct_NA_tracer', False)
    )
    raw_me = weighted_mean_enrichment(raw_areas, n_tracer)
    natural_me = natural_abundance_enrichment(formula, tracer_str, n_tracer, DEFAULT_ISODATA)
    corrected_me_pct = corrected_me * 100 if not np.isnan(corrected_me) else np.nan
    raw_me_pct = raw_me * 100 if not np.isnan(raw_me) else np.nan
    natural_me_pct = natural_me * 100 if not np.isnan(natural_me) else np.nan

    # Excess enrichment = mean enrichment after full isotopologue correction has
    # removed natural-abundance and tracer-impurity contributions.
    # We always use corrected_me_pct here regardless of the correct_NA_tracer flag:
    # when correct_NA_tracer=True the correction matrix explicitly accounts for the
    # tracer element's own NA; when False it only corrects the background elements.
    # In both cases corrected_me_pct is the best available estimate of true labelling.
    excess_pct = corrected_me_pct
    return {
        "tracer_el": tracer_el,
        "n_tracer": n_tracer,
        "corrected_areas": corrected_areas,
        "iso_frac": iso_frac,
        "raw_me_pct": raw_me_pct,
        "natural_me_pct": natural_me_pct,
        "excess_pct": excess_pct,
    }


def _apply_isotopologue_blank_filter(raw_areas, iso_heights, blank_heights, fold_change):
    """Zero out each isotopologue's raw area individually against its own
    blank height, instead of gating the whole compound on a single
    max-height-vs-max-blank-height comparison. A compound with one
    contaminated isotopologue and several clean ones should only lose the
    contaminated one, and a compound with one strong isotopologue should
    not let that strength rescue an unrelated, genuinely blank-level one.

    Returns (filtered_raw_areas, excluded_mask); excluded_mask[i] is True
    if M+i was zeroed out because it failed the blank fold-change test.
    """
    filtered = list(raw_areas)
    excluded = [False] * len(raw_areas)
    for i in range(len(raw_areas)):
        height = iso_heights[i] if i < len(iso_heights) else 0.0
        blank_height = blank_heights[i] if i < len(blank_heights) else 0.0
        if fails_blank_fold_change(height, blank_height, fold_change):
            filtered[i] = 0.0
            excluded[i] = True
    return filtered, excluded


def _mid_fraction_value(metrics, iso_idx):
    iso_frac = metrics.get('iso_frac', [])
    if iso_idx < len(iso_frac) and not pd.isna(iso_frac[iso_idx]):
        return float(iso_frac[iso_idx])
    return 0.0


def _write_mid_fraction_columns(row_like, metrics, n_tracer, setter):
    for i in range(n_tracer + 1):
        mid_fraction = _mid_fraction_value(metrics, i)
        setter(row_like, f"M+{i} MID Fraction (Corrected)", mid_fraction)
        setter(row_like, f"M+{i} MID % (Corrected)", mid_fraction * 100)


def _ensure_mid_fraction_columns(df):
    if df is None or df.empty:
        return df

    corrected_cols = [
        col for col in df.columns
        if col.startswith("M+") and col.endswith(" Area (Corrected)")
    ]
    if not corrected_cols:
        return df

    def _iso_number(col_name):
        return int(col_name.split("+", 1)[1].split(" ", 1)[0])

    corrected_cols = sorted(corrected_cols, key=_iso_number)
    total_corrected = df[corrected_cols].fillna(0).sum(axis=1)

    for area_col in corrected_cols:
        iso_label = area_col.replace(" Area (Corrected)", "")
        fraction_col = f"{iso_label} MID Fraction (Corrected)"
        percent_col = f"{iso_label} MID % (Corrected)"
        fraction = np.where(total_corrected > 0, df[area_col].fillna(0) / total_corrected, 0.0)
        df[fraction_col] = fraction
        df[percent_col] = fraction * 100

    return _reorder_results_columns(df)

def _reorder_results_columns(df):
    """
    Keep all quantitative M+N audit fields together, followed by MID fields.
    """
    if df is None or df.empty:
        return df

    def _iso_num(col):
        try:
            return int(col.split("+", 1)[1].split(" ", 1)[0])
        except (IndexError, ValueError):
            return 9999

    all_cols = list(df.columns)

    audit_stage_order = {
        "Area (Unadjusted)": 0,
        "Baseline Area": 1,
        "Area (Baseline Corrected)": 2,
        "Excluded From MID": 3,
        "Area (Raw)": 4,
        "Area (Corrected)": 5,
    }

    def _audit_stage(col):
        return next(
            (position for label, position in audit_stage_order.items() if label in col),
            len(audit_stage_order),
        )

    area_cols = sorted(
        [
            c for c in all_cols
            if c.startswith("M+")
            and any(label in c for label in audit_stage_order)
        ],
        key=lambda c: (_iso_num(c), _audit_stage(c)),
    )
    mid_cols = sorted(
        [c for c in all_cols if c.startswith("M+") and ("MID Fraction" in c or "MID %" in c)],
        key=lambda c: (_iso_num(c), 0 if "Fraction" in c else 1)
    )

    m_cols_set = set(area_cols) | set(mid_cols)
    meta_cols = [c for c in all_cols if c not in m_cols_set]

    return df[meta_cols + area_cols + mid_cols]


def _normalise_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalise_filename(value):
    text = _normalise_text(value).replace("\\", "/")
    return text.rsplit("/", 1)[-1]


def _extra_mapping_metadata_for_file(filename):
    extra_by_file = st.session_state.get('mapping_extra_metadata', {})
    if not extra_by_file:
        return {}

    target_filename = _normalise_filename(filename).casefold()
    for file_key, metadata in extra_by_file.items():
        if _normalise_filename(file_key).casefold() == target_filename:
            return dict(metadata)
    return {}


def _find_result_row_index(df, filename, metabolite, ion_mode=None):
    if df is None or df.empty:
        return None

    file_col = next((col for col in df.columns if col.lower() in ['filename', 'file']), df.columns[0])
    met_col = next((col for col in df.columns if col.lower() in ['metabolite', 'name']), 'Metabolite')
    if file_col not in df.columns or met_col not in df.columns:
        return None

    target_filename = _normalise_filename(filename).casefold()
    target_metabolite = _normalise_text(metabolite).casefold()
    mask = (
        df[file_col].map(lambda value: _normalise_filename(value).casefold()).eq(target_filename) &
        df[met_col].map(lambda value: _normalise_text(value).casefold()).eq(target_metabolite)
    )

    ion_col = 'Ion_Mode' if 'Ion_Mode' in df.columns else None
    if ion_col and ion_mode is not None:
        target_ion_mode = _normalise_text(ion_mode).casefold()
        mask &= df[ion_col].map(lambda value: _normalise_text(value).casefold()).eq(target_ion_mode)

    if mask.any():
        return df.index[mask][0]
    return None


def _ensure_manual_result_row(df, task, target_row, selected_metabolite, n_tracer):
    if df is None:
        df = pd.DataFrame()

    row_idx = _find_result_row_index(df, task[0].name, selected_metabolite, task[5])
    if row_idx is not None:
        return df, row_idx

    f_obj, species, sample_type, treatment, excel_file, ion_mode = task
    row = {col: np.nan for col in df.columns}
    row.update({
        'Filename': f_obj.name,
        'Species': species,
        'Ion_Mode': ion_mode,
        'Treatment': treatment,
        'Sample_Type': sample_type,
        'Metabolite': selected_metabolite,
        'QC_Status': 'Manually Edited',
        'Target_m/z': float(target_row['m/z']) if 'm/z' in target_row and pd.notna(target_row['m/z']) else np.nan,
        'Apex_m/z': float(target_row['m/z']) if 'm/z' in target_row and pd.notna(target_row['m/z']) else np.nan,
        'Target_RT': float(target_row['RT']) if 'RT' in target_row and pd.notna(target_row['RT']) else np.nan,
        'Apex_RT': float(target_row['RT']) if 'RT' in target_row and pd.notna(target_row['RT']) else np.nan,
        'Total_Carbons': n_tracer,
        'N_Tracer': n_tracer,
        'Total_Peak_Area': 0.0,
    })
    row.update(_extra_mapping_metadata_for_file(f_obj.name))

    for i in range(n_tracer + 1):
        row[f"M+{i} Area (Unadjusted)"] = 0.0
        # Baseline Area / Area (Baseline Corrected) are intentionally left
        # unset here (NaN) rather than defaulted to 0.0 -- they should only
        # ever be populated when local baseline subtraction actually ran.
        row[f"M+{i} Excluded From MID"] = 0
        row[f"M+{i} Area (Raw)"] = 0.0
        row[f"M+{i} Area (Corrected)"] = 0.0
        row[f"M+{i} MID Fraction (Corrected)"] = 0.0
        row[f"M+{i} MID % (Corrected)"] = 0.0

    for col in row:
        if col not in df.columns:
            df[col] = np.nan

    if df.empty:
        new_idx = 0
    else:
        try:
            new_idx = max(df.index) + 1
        except TypeError:
            new_idx = len(df)
        while new_idx in df.index:
            new_idx += 1
    df.loc[new_idx, list(row.keys())] = list(row.values())
    return df, new_idx


def _refresh_replicate_counts(df):
    if df is None or df.empty:
        return df

    group_key = ['Metabolite', 'Ion_Mode', 'Species', 'Sample_Type', 'Treatment']
    if any(col not in df.columns for col in group_key):
        return df

    df = df.drop(columns=[col for col in ['N_Replicates_Total', 'N_Replicates_Detected'] if col in df.columns])

    total_reps = df.groupby(group_key).size().rename('N_Replicates_Total')
    if 'QC_Status' in df.columns:
        qc_ok = df['QC_Status'].astype(str).str.startswith('Pass') | df['QC_Status'].isin(['Manually Edited'])
    else:
        qc_ok = pd.Series(False, index=df.index)
    area_ok = df['Total_Peak_Area'].fillna(0) > 0 if 'Total_Peak_Area' in df.columns else False

    detected_reps = (
        df[qc_ok & area_ok]
        .groupby(group_key)
        .size()
        .rename('N_Replicates_Detected')
    )

    df = df.merge(total_reps, on=group_key, how='left')
    df = df.merge(detected_reps, on=group_key, how='left')
    df['N_Replicates_Detected'] = df['N_Replicates_Detected'].fillna(0).astype(int)
    df['N_Replicates_Total'] = df['N_Replicates_Total'].fillna(0).astype(int)
    return df


def _finalize_results_df(df):
    # Remove columns created by the retired median blank-area subtraction
    # workflow, including when an older results session is restored.
    retired_blank_columns = {'Blank_Area_Subtraction_Applied'}
    retired_blank_columns.update(
        col for col in df.columns
        if str(col).startswith('M+') and any(marker in str(col) for marker in (
            'Blank Area (Median)', 'Area (Net Signed)', 'Area (Net)',
        ))
    )
    if retired_blank_columns:
        df = df.drop(
            columns=[col for col in retired_blank_columns if col in df.columns],
            errors='ignore',
        )

    retired_json_keys = {
        'blank_area_enabled', 'blank_area_median',
        'net_area_signed', 'net_area',
    }

    def _remove_retired_json_keys(value):
        if isinstance(value, dict):
            return {
                key: _remove_retired_json_keys(item)
                for key, item in value.items()
                if key not in retired_json_keys
            }
        if isinstance(value, list):
            return [_remove_retired_json_keys(item) for item in value]
        return value

    for json_col in ('Manual_Edit_Log', 'Manual_Correction_Details'):
        if json_col not in df.columns:
            continue

        def _clean_json_cell(value):
            if value is None or pd.isna(value) or str(value).strip() == '':
                return value
            try:
                parsed = json.loads(str(value))
            except (TypeError, ValueError, json.JSONDecodeError):
                return value
            return json.dumps(
                _remove_retired_json_keys(parsed), separators=(',', ':')
            )

        df[json_col] = df[json_col].map(_clean_json_cell)

    df = _refresh_replicate_counts(df)
    return _ensure_mid_fraction_columns(df)

def _preferred_enrichment_col(df):
    if 'Excess_Enrichment_%' in df.columns:
        return 'Excess_Enrichment_%'
    return 'Excess_13C_Enrichment_%'


def _get_main_tracer_label(df):
    if 'Tracer' in df.columns and not df['Tracer'].dropna().empty:
        main_tracer = str(df['Tracer'].dropna().mode()[0])
    else:
        main_tracer = "13C"
    return TRACER_LABELS.get(main_tracer, main_tracer)

def render_tab_batch(total_files, processing_tasks, blank_pos, blank_neg, params, enable_rt_alignment=False):
    if total_files > 0:
        if st.button("Extract Features & Calculate All", type="primary"):
            all_results = []
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            # --- PRE-COMPUTE METADATA, GLOBAL BLANKS + RT ALIGNMENT ---
            status_text.text("Pre-computing metadata, extracting global blank baselines, and performing RT alignment...")

            df_cache = {}
            pos_targets = []
            neg_targets = []
            adjusted_rt_cache = {}   # median apex RT from QC samples

            # Step 1: Build metadata cache and initial targets
            for task in processing_tasks:
                excel_file, ion_mode_str = task[4], task[5]
                if excel_file.name not in df_cache:
                    excel_file.seek(0)
                    df = read_metadata(excel_file)
                    df_cache[excel_file.name] = df

                    mets = generate_targets(df, params['mz_tol_ppm'], params['rt_window_min'])
                    # --- Assign metadata source to each target ---
                    for m in mets:
                        m['metadata_source'] = excel_file.name

                    if ion_mode_str == "Positive":
                        pos_targets.extend(mets)
                    else:
                        neg_targets.extend(mets)
                    # IMPORTANT: pos_targets / neg_targets are used read-only after this
                    # point (for RT alignment metadata lookups only).  Never write EIC
                    # data into these shared objects — each sample file calls
                    # generate_targets() independently to get its own fresh copies.

                                    # === RT ALIGNMENT PER SAMPLE TYPE===
            rt_shift_summary = []
            alignment_errors = []
            large_shift_count = 0
            aligned_rt_map = {}

            if enable_rt_alignment:
                status_text.text("Performing tissue‑specific RT alignment using QC samples as anchors...")

                targets_by_mode = {
                    "Positive": pos_targets,
                    "Negative": neg_targets,
                }
                targets_with_mode = [
                    (met, ion_mode)
                    for ion_mode, targets in targets_by_mode.items()
                    for met in targets
                ]

                # Get the unique sample types from the processing tasks
                # --- Helpers to identify QC tasks robustly ---
                # Supports three naming conventions users may follow:
                #   1. Species starts with "QC_"  e.g. "QC_Robusta"    (batch CSV mode)
                #   2. Sample_Type ends with "_QC" e.g. "ROOTS_QC"     (batch CSV mode)
                #   3. Treatment or Sample_Type == "QC"                 (legacy manual mode)
                def _is_qc_task(t):
                    species   = str(t[1]).strip().upper()
                    stype     = str(t[2]).strip().upper()
                    treatment = str(t[3]).strip().upper()
                    return (
                        species.startswith("QC_") or
                        stype.endswith("_QC") or
                        stype == "QC" or
                        treatment == "QC"
                    )

                def _base_tissue(sample_type_str):
                    """Strip _QC suffix to get the base tissue name used by regular samples."""
                    s = sample_type_str.strip().upper()
                    if s.endswith("_QC"):
                        return s[:-3]   # e.g. "ROOTS_QC" -> "ROOTS"
                    return s

                def _norm_treatment(treatment_str):
                    return str(treatment_str).strip().upper()

                # Only iterate over REGULAR (non-QC) sample type + treatment groups.
                # RT drift can be matrix/treatment-specific, so do not share QC
                # across different treatments.
                regular_sample_groups = sorted({
                    (t[2], t[3]) for t in processing_tasks if not _is_qc_task(t)
                }, key=lambda x: (str(x[0]).upper(), str(x[1]).upper()))

                for sample_type, treatment in regular_sample_groups:
                    base = _base_tissue(sample_type)
                    treatment_key = _norm_treatment(treatment)

                    # Tier 1: tissue-specific QC files — base tissue must match
                    # e.g. aligning ROOTS uses only ROOTS_QC files, not MIC_QC
                    qc_tasks_type = [
                        t for t in processing_tasks
                        if (
                            _is_qc_task(t) and
                            _base_tissue(t[2]) == base and
                            _norm_treatment(t[3]) == treatment_key
                        )
                    ]

                    # If no tissue-specific QC exists, do not borrow another tissue's QC.
                    # Keep metadata RTs and ask the user to provide QC for this sample type.

                    # Find which metabolites appear in the metadata files of this tissue
                    tasks_type = [t for t in processing_tasks if t[2] == sample_type and t[3] == treatment]
                    tissue_met_set = set()
                    for task in tasks_type:
                        ex_name = task[4].name
                        ion_mode = task[5]
                        df = df_cache.get(ex_name)
                        if df is not None:
                            for _, row in df.iterrows():
                                if 'Name' in row and pd.notna(row['Name']):
                                    tissue_met_set.add((row['Name'], ion_mode, ex_name))

                    if not qc_tasks_type:
                        st.warning(
                            f"No tissue- and treatment-specific QC samples found for "
                            f"`{sample_type}` / `{treatment}`. Please provide QC data "
                            "for this exact sample type and treatment. Using metadata RTs for now."
                        )
                        for met, ion_mode in targets_with_mode:
                            key_met = (met['name'], ion_mode, met.get('metadata_source'))
                            if key_met not in tissue_met_set:
                                continue

                            original_rt = met['target_rt']
                            rt_shift_summary.append({
                                "Metabolite": met['name'],
                                "Ion_Mode": ion_mode,
                                "Sample_Type": sample_type,
                                "Treatment": treatment,
                                "Original_RT": round(original_rt, 3),
                                "Aligned_RT": round(original_rt, 3),
                                "Shift_min": 0.0,
                                "Status": "No tissue+treatment QC data (please provide QC data for this exact group)"
                            })
                        continue

                    # Dictionary: (met_name, ion_mode, metadata_source) -> list of M+0 apex RTs
                    # one apex RT per QC file (not per scan).
                    qc_rt_dict = {}

                    for qc_file, mode in [(t[0], t[5]) for t in qc_tasks_type]:
                        mode_targets = targets_by_mode.get(mode, [])
                        qc_file.seek(0)
                        try:
                            # Accumulate the full M+0 EIC for each metabolite from this QC file,
                            # then run the peak picker once to extract the true apex RT.
                            qc_eic_buf = {}   # key -> {'rt': [], 'int': [], 'mz_min': , 'mz_max': , 'target_rt': }

                            with mzml.read(qc_file) as reader:
                                for spectrum in reader:
                                    if spectrum.get('ms level') != 1:
                                        continue
                                    try:
                                        rt = spectrum['scanList']['scan'][0]['scan start time']
                                        if hasattr(rt, 'unit_info') and 'second' in str(rt.unit_info).lower():
                                            rt /= 60.0
                                    except KeyError:
                                        continue
                                    mz_array = spectrum.get('m/z array', np.array([]))
                                    int_array = spectrum.get('intensity array', np.array([]))
                                    if len(mz_array) == 0:
                                        continue

                                    for met in mode_targets:
                                        key = (met['name'], mode, met.get('metadata_source'))
                                        if key not in tissue_met_set:
                                            continue
                                        if met['rt_min'] <= rt <= met['rt_max']:
                                            for iso in met['isotopes']:
                                                if iso['isotope_number'] == 0:
                                                    mask = (mz_array >= iso['mz_min']) & (mz_array <= iso['mz_max'])
                                                    intensity = float(np.max(int_array[mask])) if np.any(mask) else 0.0
                                                    mz_val = float(mz_array[mask][np.argmax(int_array[mask])]) if np.any(mask) else 0.0
                                                    if key not in qc_eic_buf:
                                                        qc_eic_buf[key] = {
                                                            'rt': [], 'int': [], 'mz': [],
                                                            'target_rt': met['target_rt']
                                                        }
                                                    qc_eic_buf[key]['rt'].append(rt)
                                                    qc_eic_buf[key]['int'].append(intensity)
                                                    qc_eic_buf[key]['mz'].append(mz_val)
                                                    break

                            # Peak-pick each buffered EIC to get the apex RT for this QC file
                            for key, buf in qc_eic_buf.items():
                                if len(buf['rt']) < 3:
                                    continue
                                sort_idx = np.argsort(buf['rt'])
                                rt_arr = np.array(buf['rt'])[sort_idx]
                                int_arr = np.array(buf['int'])[sort_idx]
                                mz_arr = np.array(buf['mz'])[sort_idx]
                                _, _, _, _, _, apex_rt, _ = _run_peak_picker(
                                    rt_arr, int_arr, mz_arr, buf['target_rt'], params
                                )
                                if apex_rt > 0:
                                    if key not in qc_rt_dict:
                                        qc_rt_dict[key] = []
                                    qc_rt_dict[key].append(apex_rt)

                        except Exception as exc:
                            alignment_errors.append({
                                "File": qc_file.name,
                                "Ion mode": mode,
                                "Sample type": sample_type,
                                "Treatment": treatment,
                                "Error": f"{type(exc).__name__}: {exc}",
                            })
                            continue

                    # Record alignment for each metabolite that belongs to this tissue
                    for met, ion_mode in targets_with_mode:
                        # ---Include metadata source ---
                        key_met = (met['name'], ion_mode, met.get('metadata_source'))
                        if key_met not in tissue_met_set:
                            continue   # not used in this sample type

                        original_rt = met['target_rt']
                        window = params['rt_window_min']

                        if key_met in qc_rt_dict and len(qc_rt_dict[key_met]) >= 2:
                            median_rt = float(np.median(qc_rt_dict[key_met]))
                            shift = median_rt - original_rt
                            status = "OK"
                            if abs(shift) > window:
                                status = f"⚠️ Shift exceeds ±{window} min"
                                large_shift_count += 1
                            # Store aligned RT for this exact sample type + treatment.
                            aligned_rt_map[(met['name'], ion_mode, sample_type, treatment, met.get('metadata_source'))] = median_rt
                        else:
                            median_rt = original_rt
                            shift = 0.0
                            status = "No QC data (using metadata RT)"

                        rt_shift_summary.append({
                            "Metabolite": met['name'],
                            "Ion_Mode": ion_mode,
                            "Sample_Type": sample_type,
                            "Treatment": treatment,
                            "Original_RT": round(original_rt, 3),
                            "Aligned_RT": round(median_rt, 3),
                            "Shift_min": round(shift, 3),
                            "Status": status
                        })
            else:
                # Alignment disabled – fill summary with "disabled" entries
                for met in pos_targets + neg_targets:
                    ion_mode = 'Positive' if met in pos_targets else 'Negative'
                    rt_shift_summary.append({
                        "Metabolite": met['name'],
                        "Ion_Mode": ion_mode,
                        "Sample_Type": "N/A",
                        "Treatment": "N/A",
                        "Original_RT": round(met['target_rt'], 3),
                        "Aligned_RT": round(met['target_rt'], 3),
                        "Shift_min": 0.0,
                        "Status": "Alignment disabled"
                    })

            # === Show warning if any large shifts were detected ===
            if large_shift_count > 0:
                st.warning(f"⚠️ **{large_shift_count} metabolites** have aligned RTs that fall **outside** the original ±{params['rt_window_min']} min window. "
                           f"Please check the RT Alignment Summary table below and consider widening the RT window or inspecting the peaks manually.")

            # Step 3: Global blank extraction
            global_blank_cache = {"Positive": {}, "Negative": {}}
            if blank_pos and pos_targets:
                global_blank_cache["Positive"] = extract_blank_max_heights(blank_pos, pos_targets, params)
            if blank_neg and neg_targets:
                global_blank_cache["Negative"] = extract_blank_max_heights(blank_neg, neg_targets, params)

            # Step 4: Process all samples
            for idx, (mzml_file, species, sample_type, treatment, excel_file, ion_mode_str) in enumerate(processing_tasks):
                mzml_file.seek(0)
                file_name = mzml_file.name
                status_text.text(f"Processing {idx + 1}/{total_files} | {species} | {treatment} {sample_type} ({ion_mode_str}) | {file_name}...")

                try:
                    df_targets = df_cache[excel_file.name]
                    active_blank_dict = global_blank_cache.get(ion_mode_str, {})

                    metabolites = generate_targets(df_targets, params['mz_tol_ppm'], params['rt_window_min'])
                                        # --- Apply tissue-specific RT alignment (if enabled) ---
                    if enable_rt_alignment:
                        for met in metabolites:
                            # ---Lookup the RT using the current excel_file.name ---
                            key = (met['name'], ion_mode_str, sample_type, treatment, excel_file.name)
                            if key in aligned_rt_map:
                                new_rt = aligned_rt_map[key]
                                met['target_rt'] = new_rt
                                met['rt_min'] = new_rt - params['rt_window_min']
                                met['rt_max'] = new_rt + params['rt_window_min']

                    with mzml.read(mzml_file) as reader:
                        for spectrum in reader:
                            if spectrum.get('ms level') != 1: continue
                            try:
                                rt = spectrum['scanList']['scan'][0]['scan start time']
                                if hasattr(rt, 'unit_info') and 'second' in str(rt.unit_info).lower(): rt /= 60.0
                            except KeyError: continue
                            mz_array, int_array = spectrum.get('m/z array', np.array([])), spectrum.get('intensity array', np.array([]))
                            if len(mz_array) == 0: continue

                            for met in metabolites:
                                if met['rt_min'] <= rt <= met['rt_max']:
                                    for iso in met['isotopes']:
                                        mask = (mz_array >= iso['mz_min']) & (mz_array <= iso['mz_max'])
                                        if np.any(mask):
                                            max_idx = np.argmax(int_array[mask])
                                            intensity = int_array[mask][max_idx]
                                            mz_val = mz_array[mask][max_idx]
                                        else:
                                            intensity, mz_val = 0.0, 0.0

                                        iso['eic_rt'].append(rt)
                                        iso['eic_int'].append(intensity)
                                        iso['eic_mz'].append(mz_val)

                    for met in metabolites:
                        best_observed_rt, best_observed_mz = 0.0, 0.0
                        raw_areas = [0.0] * (met['n_tracer'] + 1)
                        iso_heights = [0.0] * (met['n_tracer'] + 1)

                        # --- Build a Composite EIC for boundary locking ---
                        composite_int = None
                        composite_rt = None
                        ref_left, ref_right = None, None

                        composite_rt, composite_int = _build_composite_eic(met)
                        if composite_rt is not None and len(composite_rt) > 0:
                            dummy_mz = np.zeros_like(composite_rt) # MZ not needed for boundary math

                            # Run detection ONCE on the composite peak
                            area, height, ref_left, ref_right, _, apex_rt, _ = _run_peak_picker(
                                composite_rt, composite_int, dummy_mz, met['target_rt'], params
                            )

                            if area > 0:
                                best_observed_rt = apex_rt

                        # --- Apply the locked boundaries to ALL isotopologues ---
                        for iso in met['isotopes']:
                            if ref_left is not None and ref_right is not None and len(iso['eic_rt']) > 0:
                                sort_idx = np.argsort(iso['eic_rt'])
                                rt_arr = np.array(iso['eic_rt'])[sort_idx]
                                int_arr = np.array(iso['eic_int'])[sort_idx]
                                mz_arr = np.array(iso['eic_mz'])[sort_idx]

                                safe_left = max(0, min(len(composite_rt) - 1, ref_left))
                                safe_right = max(0, min(len(composite_rt) - 1, ref_right))
                                rt_start, rt_end = sorted((composite_rt[safe_left], composite_rt[safe_right]))
                                roi_mask = (rt_arr >= rt_start) & (rt_arr <= rt_end)

                                if np.any(roi_mask):
                                    roi_ints = int_arr[roi_mask]
                                    roi_rts = rt_arr[roi_mask]
                                    roi_mzs = mz_arr[roi_mask]

                                    # Calculate exact area inside the locked window
                                    iso_area = trapezoid(y=roi_ints, x=roi_rts)
                                    iso_height = np.max(roi_ints)
                                    if iso_height < params['noise_threshold'] or iso_area < params['min_peak_area']:
                                        iso_area = 0.0
                                        iso_height = 0.0
                                    iso_idx = iso['isotope_number']
                                    iso_heights[iso_idx] = iso_height
                                    iso['raw_area'] = iso_area
                                    raw_areas[iso_idx] = iso_area

                                    # Save exact m/z from the strongest detected isotopologue.
                                    if iso_area > 0 and iso_height >= max(iso_heights):
                                        apex_idx = np.argmax(roi_ints)
                                        best_observed_mz = roi_mzs[apex_idx]
                                else:
                                    iso['raw_area'] = 0.0
                            else:
                                iso['raw_area'] = 0.0

                        # Fallback for best_observed_mz if M+0 was completely missing
                        if best_observed_mz == 0.0:
                            best_observed_mz = met['target_mz']

                        # --- Determine QC failure reason before any early exit ---
                        # All failures are recorded so the user can see them in the QC table.
                        blank_heights = active_blank_dict.get(met['name'], [])
                        pre_blank_total_area = sum(raw_areas)

                        # Blank/background filtering is applied per isotopologue, not as a
                        # single compound-level comparison. Comparing only the tallest
                        # isotopologue in the sample against the tallest in the blank let a
                        # contaminated channel (e.g. M+0) pass QC purely because an unrelated
                        # isotopologue (e.g. M+3) was strong, and could equally zero out a
                        # real but weakly-labelled compound because one unrelated
                        # isotopologue happened to be tall in the blank.
                        raw_areas, excluded_by_blank = _apply_isotopologue_blank_filter(
                            raw_areas, iso_heights, blank_heights, params['blank_fold_change']
                        )
                        for i, was_excluded in enumerate(excluded_by_blank):
                            if was_excluded:
                                iso_heights[i] = 0.0

                        total_raw_area = sum(raw_areas)
                        compound_peak_height = max(iso_heights) if iso_heights else 0.0
                        any_excluded_by_blank = any(excluded_by_blank)

                        if total_raw_area <= 0:
                            qc_status = (
                                f"Fail: Blank (FC < {params['blank_fold_change']:.1f})"
                                if any_excluded_by_blank and pre_blank_total_area > 0
                                else "Fail: No signal"
                            )
                        elif compound_peak_height < params['noise_threshold']:
                            qc_status = "Fail: Below height threshold"
                        elif total_raw_area < params['min_peak_area']:
                            qc_status = "Fail: Below area threshold"
                        elif any_excluded_by_blank:
                            qc_status = "Pass: Some isotopologues blank-filtered"
                        else:
                            qc_status = "Pass"

                        # Warn if M+0 is absent but other isotopologues are detected —
                        # likely indicates fully labelled compound (biologically valid)
                        # but worth flagging for conscious review.
                        if qc_status == "Pass" and raw_areas[0] == 0.0 and total_raw_area > 0:
                            qc_status = "Pass: M+0 not detected (fully labelled?)"

                        # Drop entirely only if drop_zeros is set AND there is truly no signal.
                        # All other failure types are kept as rows so the user can inspect them.
                        if params['drop_zeros'] and total_raw_area <= 0:
                            continue

                        # === Multi‑element correction ===
                        tracer_str = met.get('tracer_str', '13C')
                        formula = met.get('formula')

                        if formula is None:
                            formula = {'C': met['carbons']}

                        corrected_areas = [0.0] * len(raw_areas)
                        metrics = {
                            'tracer_el': '', 'n_tracer': met['n_tracer'],
                            'corrected_areas': corrected_areas, 'iso_frac': [],
                            'raw_me_pct': np.nan, 'natural_me_pct': np.nan, 'excess_pct': np.nan,
                        }

                        if qc_status.startswith("Pass"):
                            try:
                                tracer_el, _ = parse_tracer(tracer_str, DEFAULT_ISODATA)
                                formula = dict(formula)
                                formula[tracer_el] = met['n_tracer']
                                metrics = _calculate_enrichment_metrics(raw_areas, formula, tracer_str, params)
                                corrected_areas = metrics['corrected_areas']
                                if np.isnan(metrics['excess_pct']):
                                    qc_status = "Fail: Computation error"
                            except Exception as exc:
                                qc_status = f"Fail: Correction error ({exc})"

                        # Store corrected areas per isotopologue
                        for i_idx, iso in enumerate(met['isotopes']):
                            iso['corrected_area'] = corrected_areas[i_idx] if i_idx < len(corrected_areas) else 0.0

                        # Total_Peak_Area uses corrected areas so it is internally consistent
                        # with the per-isotopologue corrected columns exported alongside it.
                        # Raw isotopologue areas are also exported as separate columns.
                        total_corrected_area = sum(corrected_areas)

                        res_row = {
                            'Filename': file_name,
                            'Species': species,
                            'Ion_Mode': ion_mode_str,
                            'Treatment': treatment,
                            'Sample_Type': sample_type,
                            'Metabolite': met['name'],
                            'QC_Status': qc_status,
                            'Target_m/z': met['target_mz'],
                            'Apex_m/z': best_observed_mz,
                            'Target_RT': met['target_rt'],
                            'Apex_RT': best_observed_rt,
                            'Total_Carbons': met['carbons'],
                            'Tracer': tracer_str,
                            'Tracer_Element': metrics['tracer_el'],
                            'N_Tracer': metrics['n_tracer'],
                            'Raw_ME_%': metrics['raw_me_pct'],
                            'Natural_Abundance_ME_%': metrics['natural_me_pct'],
                            'Excess_Enrichment_%': metrics['excess_pct'],
                            'Total_Peak_Area': total_corrected_area
                        }
                        res_row.update(_extra_mapping_metadata_for_file(file_name))
                        for iso in met['isotopes']:
                            res_row[f"M+{iso['isotope_number']} Area (Corrected)"] = iso['corrected_area']
                            iso_idx = iso['isotope_number']
                            res_row[f"M+{iso_idx} Area (Raw)"] = raw_areas[iso_idx] if iso_idx < len(raw_areas) else 0.0
                            res_row[f"M+{iso_idx} Excluded By Blank"] = int(
                                iso_idx < len(excluded_by_blank) and excluded_by_blank[iso_idx]
                            )
                        _write_mid_fraction_columns(
                            res_row,
                            metrics,
                            metrics['n_tracer'],
                            lambda row, col, val: row.__setitem__(col, val)
                        )
                        all_results.append(res_row)
                except Exception as e:
                    st.error(f"Error processing {file_name}: {e}")

                current_progress = float(idx + 1) / float(max(1, total_files))
                progress_bar.progress(min(1.0, current_progress))

                       # === RT Alignment Summary Table ===
                        # === Store alignment summary in session state (no immediate display) ===
            if rt_shift_summary:
                df_rt = pd.DataFrame(rt_shift_summary)
                col_order = ["Sample_Type", "Treatment", "Metabolite", "Ion_Mode", "Original_RT", "Aligned_RT", "Shift_min", "Status"]
                df_rt = df_rt[col_order]
                st.session_state.rt_alignment_summary = df_rt.copy()
            else:
                st.session_state.rt_alignment_summary = None

            st.session_state.rt_alignment_errors = (
                pd.DataFrame(alignment_errors) if alignment_errors else None
            )

            status_text.text("All files processed successfully!")

            if all_results:
                results_df = pd.DataFrame(all_results)
                results_df = _finalize_results_df(results_df)

                st.session_state.results_df = results_df
            else:
                st.error("No valid data was extracted. Please check your mzML and Metadata files.")
                st.session_state.results_df = None

                # ==================== PERSISTENT RT ALIGNMENT SUMMARY ====================
        if st.session_state.get("rt_alignment_summary") is not None:
            st.subheader("📍 RT Alignment Summary (Per Tissue)")
            st.dataframe(
                st.session_state.rt_alignment_summary.style.format({
                    "Original_RT": "{:.3f}",
                    "Aligned_RT": "{:.3f}",
                    "Shift_min": "{:.3f}"
                }),
                use_container_width=True
            )

        if st.session_state.get("rt_alignment_errors") is not None:
            alignment_errors_df = st.session_state.rt_alignment_errors
            st.warning(
                f"RT alignment failed for {len(alignment_errors_df)} QC "
                "file/group combinations. Metadata RT may have been used instead."
            )
            st.dataframe(
                alignment_errors_df,
                use_container_width=True,
                hide_index=True,
            )

        if st.session_state.results_df is not None and not st.session_state.results_df.empty:
            st.session_state.results_df = _finalize_results_df(st.session_state.results_df)
            st.success("✅ Results available in memory!")
            st.dataframe(st.session_state.results_df, use_container_width=True)
            _render_manual_edit_history(st.session_state.results_df)
            st.download_button(
                label="📥 Download Combined Batch Results",
                data=st.session_state.results_df.to_csv(index=False).encode('utf-8'),
                file_name='tracer_enrichment_full_experiment.csv',
                mime='text/csv',
                key="permanent_download_btn_1"
            )

    else:
        st.info("You have loaded a previous session! You can view the data below, or skip straight to Tabs 3 and 4.")
        if st.session_state.results_df is not None and not st.session_state.results_df.empty:
            st.session_state.results_df = _finalize_results_df(st.session_state.results_df)
            st.dataframe(st.session_state.results_df, use_container_width=True)
            _render_manual_edit_history(st.session_state.results_df)
            st.download_button(
                label="📥 Download Combined Batch Results",
                data=st.session_state.results_df.to_csv(index=False).encode('utf-8'),
                file_name='tracer_enrichment_full_experiment.csv',
                mime='text/csv',
                key="permanent_download_btn_2"
            )

def render_tab_qc(total_files, processing_tasks, params, blank_pos=None, blank_neg=None):
    if total_files > 0:
        manual_edit_notice = st.session_state.pop('_manual_edit_notice', None)
        manual_edit_warning = st.session_state.pop('_manual_edit_warning', None)
        manual_edit_plots = st.session_state.pop('_manual_edit_plots', [])
        if manual_edit_notice:
            st.success(manual_edit_notice)
            if manual_edit_warning:
                st.warning(manual_edit_warning)
            if len(manual_edit_plots) > 3:
                with st.expander(f"View {len(manual_edit_plots)} Integration Plots", expanded=False):
                    for plot in manual_edit_plots:
                        st.plotly_chart(plot, use_container_width=True)
            else:
                for plot in manual_edit_plots:
                    st.plotly_chart(plot, use_container_width=True)

        st.markdown("### Inspect & Compare Peaks Side-by-Side")
        st.markdown("*Note: This tab shows raw, uncorrected EICs so you can physically verify peak shape and noise. Blank fold-change rejection and Isotope correction are applied in the final numerical output, not these raw graphs.*")

        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            def task_label(task):
                f_obj, sp, s, t, ex, mode = task
                return f"[{sp}] {f_obj.name} [{mode}] ({t} - {s})"

            file_labels = [task_label(task) for task in processing_tasks]
            task_by_label = {task_label(task): task for task in processing_tasks}
            qc_sample_type_options = sorted({str(task[2]) for task in processing_tasks})
            qc_ion_mode_options = sorted({str(task[5]) for task in processing_tasks})
            qc_selected_sample_types = st.multiselect(
                "Filter Compare Files by Sample Type:",
                options=qc_sample_type_options,
                default=qc_sample_type_options,
                key="qc_compare_sample_type_filter"
            )
            qc_selected_ion_modes = st.multiselect(
                "Filter Compare Files by Ionization Mode:",
                options=qc_ion_mode_options,
                default=qc_ion_mode_options,
                key="qc_compare_ion_mode_filter"
            )
            extra_filter_cols = [
                col for col in st.session_state.get('mapping_extra_columns', [])
                if any(col in _extra_mapping_metadata_for_file(task[0].name) for task in processing_tasks)
            ]

            def labels_matching_qc_filters(labels):
                filtered_labels = []
                for label in labels:
                    task = task_by_label.get(label)
                    if task is None:
                        continue
                    sample_type = str(task[2])
                    ion_mode = str(task[5])
                    if sample_type not in qc_selected_sample_types:
                        continue
                    if ion_mode not in qc_selected_ion_modes:
                        continue
                    filtered_labels.append(label)
                return filtered_labels

            def labels_matching_extra_filters(labels, selections):
                filtered_labels = []
                for label in labels:
                    task = task_by_label.get(label)
                    if task is None:
                        continue
                    metadata = _extra_mapping_metadata_for_file(task[0].name)
                    keep_label = True
                    for col, selected_values in selections.items():
                        if selected_values:
                            value = metadata.get(col)
                            if pd.isna(value) or str(value) not in selected_values:
                                keep_label = False
                                break
                    if keep_label:
                        filtered_labels.append(label)
                return filtered_labels

            qc_extra_filter_selections = {}
            if extra_filter_cols:
                with st.expander("Additional mapping filters", expanded=False):
                    for filter_col in extra_filter_cols:
                        values = sorted({
                            str(_extra_mapping_metadata_for_file(task[0].name).get(filter_col))
                            for task in processing_tasks
                            if filter_col in _extra_mapping_metadata_for_file(task[0].name)
                            and not pd.isna(_extra_mapping_metadata_for_file(task[0].name).get(filter_col))
                        })
                        qc_extra_filter_selections[filter_col] = st.multiselect(
                            f"Filter QC files by {filter_col}:",
                            options=values,
                            default=values,
                            key=f"qc_extra_filter_{filter_col}"
                        )

            filtered_file_labels = labels_matching_extra_filters(
                labels_matching_qc_filters(file_labels),
                qc_extra_filter_selections
            )
         # Group files by condition for easy 1-click replicate selection
        all_condition_map = {}
        for f_obj, sp, s, t, ex, mode in processing_tasks:
            cond_name = f"{sp} | {t} - {s} ({mode})"
            lbl = f"[{sp}] {f_obj.name} [{mode}] ({t} - {s})"

            if cond_name not in all_condition_map: all_condition_map[cond_name] = []
            all_condition_map[cond_name].append(lbl)

        condition_map = {}
        for f_obj, sp, s, t, ex, mode in [task_by_label[label] for label in filtered_file_labels]:
            cond_name = f"{sp} | {t} - {s} ({mode})"
            lbl = f"[{sp}] {f_obj.name} [{mode}] ({t} - {s})"

            if cond_name not in condition_map: condition_map[cond_name] = []
            condition_map[cond_name].append(lbl)
        condition_options = ["Custom Selection..."] + list(condition_map.keys())
        selected_condition = st.selectbox("Quick Select Replicates by Group:", condition_options)

        if selected_condition != "Custom Selection...":
            auto_compare_selection = condition_map[selected_condition]
        else:
            auto_compare_selection = filtered_file_labels

        compare_filter_signature = (
            tuple(qc_selected_sample_types),
            tuple(qc_selected_ion_modes),
            tuple(
                (col, tuple(qc_extra_filter_selections.get(col, [])))
                for col in extra_filter_cols
            ),
            selected_condition,
        )
        if st.session_state.get("qc_compare_filter_signature") != compare_filter_signature:
            st.session_state.qc_compare_files = auto_compare_selection
            st.session_state.qc_compare_filter_signature = compare_filter_signature
        else:
            st.session_state.qc_compare_files = [
                label for label in st.session_state.get("qc_compare_files", [])
                if label in filtered_file_labels
            ]

        selected_file_labels = st.multiselect(
            "Select Files to Compare:",
            options=filtered_file_labels,
            key="qc_compare_files"
        )

        with vis_col2:
            if selected_file_labels:
                first_file_label = selected_file_labels[0]
                first_task = next(task for task in processing_tasks if f"[{task[1]}] {task[0].name} [{task[5]}] ({task[3]} - {task[2]})" == first_file_label)

                ex_file = first_task[4]
                ex_file.seek(0)
                df_targets_vis = read_metadata(ex_file)

                met_options = {}
                for idx, row in df_targets_vis.iterrows():
                    label = f"{row['Name']} ({row['RT']:.2f} min | Target m/z: {row['m/z']:.4f})"
                    met_options[label] = idx  # store the unique index

                selected_met_label = st.selectbox("Select Metabolite to Inspect:", list(met_options.keys()))
                selected_idx = met_options[selected_met_label]
                st.session_state.selected_met_idx = selected_idx
                selected_metabolite = df_targets_vis.loc[selected_idx, 'Name']
                target_row = df_targets_vis.loc[selected_idx]
            else:
                st.info("Select a file on the left to populate metabolites.")
                selected_metabolite = None

        # Flexible display window width
        rt_context = st.slider(
            "⏱️ Display window width (± min)",
            min_value=0.5, max_value=5.0, value=2.0, step=0.5,
            help="How many minutes to show to the left and right of the center of the peak. Larger = peak looks sharper"
        )

        if st.button("Generate Chromatograms") and selected_metabolite:
            target_row = df_targets_vis.loc[st.session_state.selected_met_idx]
            comparison_data = []

            with st.spinner("Extracting EICs for side-by-side comparison..."):
                for f_label in selected_file_labels:
                    task = next(item for item in processing_tasks if f"[{item[1]}] {item[0].name} [{item[5]}] ({item[3]} - {item[2]})" == f_label)
                    f_obj = task[0]
                    f_obj.seek(0)

                    met_obj = generate_targets(
                        pd.DataFrame([target_row]), params['mz_tol_ppm'], params['rt_window_min']
                    )[0]


                    with mzml.read(f_obj) as reader:
                        for spectrum in reader:
                            if spectrum.get('ms level') != 1: continue
                            try:
                                rt = spectrum['scanList']['scan'][0]['scan start time']
                                if hasattr(rt, 'unit_info') and 'second' in str(rt.unit_info).lower(): rt /= 60.0
                            except KeyError: continue

                            mz_array, int_array = spectrum.get('m/z array', np.array([])), spectrum.get('intensity array', np.array([]))
                            if len(mz_array) == 0: continue

                            if met_obj['rt_min'] <= rt <= met_obj['rt_max']:
                                for iso in met_obj['isotopes']:
                                    mask = (mz_array >= iso['mz_min']) & (mz_array <= iso['mz_max'])
                                    if np.any(mask):
                                        max_idx = np.argmax(int_array[mask])
                                        intensity = int_array[mask][max_idx]
                                        mz_val = mz_array[mask][max_idx]
                                    else:
                                        intensity, mz_val = 0.0, 0.0
                                    iso['eic_rt'].append(rt)
                                    iso['eic_int'].append(intensity)
                                    iso['eic_mz'].append(mz_val)

                    comparison_data.append((f_label, met_obj, f_obj.name))

            num_isotopes = len(comparison_data[0][1]['isotopes'])
            for i in range(num_isotopes):
                st.markdown("---")
                st.markdown(f"#### Isotope: M+{i}")
                cols = st.columns(len(selected_file_labels))

                for col_idx, (file_label, m_obj, exact_filename) in enumerate(comparison_data):
                    iso = m_obj['isotopes'][i]
                    if len(iso['eic_rt']) > 0:
                        sort_idx = np.argsort(iso['eic_rt'])
                        rt_arr = np.array(iso['eic_rt'])[sort_idx]
                        int_arr = np.array(iso['eic_int'])[sort_idx]
                        mz_arr = np.array(iso['eic_mz'])[sort_idx]

                        _, _, _, _, smoothed, apex_rt, apex_mz = local_minimum_resolver_extract_area(
                            rt_array=rt_arr, int_array=int_arr, mz_array=mz_arr,
                            target_rt=m_obj['target_rt'], params=params
                        )

                        comp_rt, comp_int = _build_composite_eic(m_obj)
                        if comp_rt is not None and len(comp_rt) > 0:
                            comp_mz = np.zeros_like(comp_rt)
                            _, _, left_idx, right_idx, _, _, _ = _run_peak_picker(
                                comp_rt, comp_int, comp_mz, m_obj['target_rt'], params
                            )
                        else:
                            left_idx, right_idx = 0, 0

                        if right_idx > left_idx and comp_rt is not None:
                            comp_left = max(0, min(len(comp_rt) - 1, left_idx))
                            comp_right = max(0, min(len(comp_rt) - 1, right_idx))
                            rt_start, rt_end = sorted((comp_rt[comp_left], comp_rt[comp_right]))
                            bound_mask = (rt_arr >= rt_start) & (rt_arr <= rt_end)
                            if np.any(bound_mask):
                                area = trapezoid(y=int_arr[bound_mask], x=rt_arr[bound_mask])
                                height = np.max(int_arr[bound_mask])
                            else:
                                area, height = 0.0, 0.0

                            # --- CHECK MANUAL EDITS TO HIDE GREEN BOX ---
                            if st.session_state.get('results_df') is not None and not st.session_state.results_df.empty:
                                res_df = st.session_state.results_df
                                existing_row_idx = _find_result_row_index(res_df, exact_filename, selected_metabolite)
                                if existing_row_idx is not None:
                                    raw_col = f"M+{i} Area (Raw)"
                                    if raw_col in res_df.columns:
                                        if res_df.at[existing_row_idx, raw_col] == 0.0:
                                            area = 0.0
                                            height = 0.0
                                            left_idx, right_idx = 0, 0

                            if area > 0:
                                apex_candidates = np.where(bound_mask)[0]
                                apex_local_idx = apex_candidates[np.argmax(int_arr[bound_mask])]
                                apex_rt = rt_arr[apex_local_idx]
                                apex_mz = mz_arr[apex_local_idx]
                        else:
                            area, height = 0.0, 0.0

                        # If this row was restored from a session with a manual
                        # edit log, reproduce the latest action for this exact
                        # isotopologue after re-extracting its EIC from mzML.
                        manual_event = None
                        if st.session_state.get('results_df') is not None:
                            restored_results = st.session_state.results_df
                            restored_idx = _find_result_row_index(
                                restored_results, exact_filename,
                                selected_metabolite, m_obj.get('ion_mode')
                            )
                            if restored_idx is not None:
                                manual_event = _latest_manual_event_for_iso(
                                    restored_results.loc[restored_idx], f"M+{i}"
                                )

                        restored_plot_rt = None
                        restored_plot_int = None
                        restored_baseline = None
                        if manual_event and manual_event.get('type') == 'Reintegration':
                            saved_start = manual_event.get('rt_start')
                            saved_end = manual_event.get('rt_end')
                            if saved_start is not None and saved_end is not None:
                                smoothed, _ = _manual_integration_mask(
                                    rt_arr,
                                    int_arr,
                                    (float(saved_start), float(saved_end)),
                                    manual_event.get('algorithm'),
                                    manual_event.get('smoothing'),
                                )
                                restored_integration = integrate_exact_bounds(
                                    rt_arr,
                                    int_arr,
                                    (float(saved_start), float(saved_end)),
                                    baseline_enabled=manual_event.get('baseline_enabled', False),
                                    baseline_flank_min=manual_event.get('baseline_flank_min') or 0.05,
                                    minimum_flank_scans=manual_event.get('baseline_min_scans') or 3,
                                )
                                restored_mask = restored_integration['peak_mask']
                                if np.any(restored_mask):
                                    rt_start = float(rt_arr[restored_mask][0])
                                    rt_end = float(rt_arr[restored_mask][-1])
                                    event_details = manual_event.get('correction_details', {})
                                    isotope_details = event_details.get(f"M+{i}", {}) if isinstance(event_details, dict) else {}
                                    area = float(isotope_details.get(
                                        'area_used', restored_integration['area_baseline_corrected']
                                    ))
                                    height = float(restored_integration['height_baseline_corrected'])
                                    apex_candidates = np.where(restored_mask)[0]
                                    apex_local_idx = apex_candidates[np.argmax(
                                        restored_integration['peak_corrected']
                                    )]
                                    apex_rt = float(rt_arr[apex_local_idx])
                                    apex_mz = float(mz_arr[apex_local_idx])
                                    restored_plot_rt = restored_integration['peak_rt']
                                    restored_plot_int = restored_integration['peak_corrected']
                                    if restored_integration['baseline_status'] == 'Applied':
                                        restored_baseline = restored_integration['baseline_values']
                                else:
                                    area, height = 0.0, 0.0
                        elif manual_event and manual_event.get('type') == 'Remove isotopologues':
                            area, height = 0.0, 0.0

                        ppm_error = ((apex_mz - iso['target_mz']) / iso['target_mz']) * 1e6 if apex_mz > 0 else 0

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=rt_arr, y=int_arr, mode='lines+markers', name='Raw Data',
                                                 line=dict(color='gray', width=1), marker=dict(size=3)))
                        fig.add_trace(go.Scatter(x=rt_arr, y=smoothed, mode='lines', name='Smooth',
                                                 line=dict(color='blue', width=2, dash='dash')))
                        fig.add_hline(y=params['noise_threshold'], line_dash="dot", line_color="red")

                        if manual_event:
                            fig.add_annotation(
                                text=f"Restored manual edit: {manual_event.get('type')}",
                                xref="paper", yref="paper", x=0.01, y=0.98,
                                showarrow=False, bgcolor="rgba(255, 215, 0, 0.25)",
                            )
                            if manual_event.get('type') == 'Reintegration':
                                saved_start = manual_event.get('rt_start')
                                saved_end = manual_event.get('rt_end')
                                if saved_start is not None and saved_end is not None:
                                    _add_rt_boundary_markers(
                                        fig, (float(saved_start), float(saved_end))
                                    )

                        if restored_baseline is not None and restored_plot_rt is not None:
                            fig.add_trace(go.Scatter(
                                x=restored_plot_rt, y=restored_baseline,
                                mode='lines', name='Restored local baseline',
                                line=dict(color='orange', width=2, dash='dot'),
                            ))

                        if area > 0:
                            plot_mask = (rt_arr >= rt_start) & (rt_arr <= rt_end) if 'rt_start' in locals() else np.zeros(len(rt_arr), dtype=bool)
                            auc_rt = restored_plot_rt if restored_plot_rt is not None else rt_arr[plot_mask]
                            auc_int = restored_plot_int if restored_plot_int is not None else int_arr[plot_mask]
                            fig.add_trace(go.Scatter(x=auc_rt, y=auc_int,
                                                     fill='tozeroy', mode='none', name='AUC',
                                                     fillcolor='rgba(0, 200, 0, 0.3)'))

                            if 'rt_start' in locals() and 'rt_end' in locals():
                                fig.add_vline(x=rt_start, line_dash="dash", line_color="red",
                                              annotation_text="Start", annotation_position="top left")
                                fig.add_vline(x=rt_end, line_dash="dash", line_color="red",
                                              annotation_text="End", annotation_position="top right")

                            thr_perc = params.get('chromatographic_threshold', 85)
                            noise_bl = np.percentile(smoothed, thr_perc)
                            fig.add_hline(y=noise_bl, line_dash="dot", line_color="orange",
                                          annotation_text=f"LMR baseline ({thr_perc}%)",
                                          annotation_position="top right")

                        if len(int_arr) > 0:
                            max_display_int = max(np.max(int_arr), np.max(smoothed)) if len(smoothed) > 0 else np.max(int_arr)
                        else:
                            max_display_int = 0
                        ymax = max(1.0, max_display_int * 1.15)

                        if apex_rt > 0:
                               center_rt = apex_rt
                        else:
                            center_rt = met_obj['target_rt']

                        fig.update_layout(
                            title=f"<b>{file_label.split(' ')[1]}</b><br><span style='font-size: 12px;'>Apex m/z: {apex_mz:.4f} ({ppm_error:+.1f} ppm) | RT: {apex_rt:.2f} | Raw Area: {area:,.0f} | Height: {height:,.0f}</span>",
                            xaxis_title="RT (min)", yaxis_title="Intensity", hovermode="x unified",
                            height=320, margin=dict(l=10, r=10, t=50, b=20), showlegend=False,
                            xaxis=dict(range=[center_rt - rt_context, center_rt + rt_context]),
                            yaxis=dict(range=[0, max_display_int * 1.15])
                        )
                        cols[col_idx].plotly_chart(fig, use_container_width=True, key=f"qc_chart_{i}_{col_idx}_{file_label}")

        st.divider()
        st.markdown("### 🛠️ Manual Integration Tool")
        st.markdown("Select isotopologues and highlight the peak bounds. This will automatically update your global tracer enrichment and MID data.")

        man_col1, man_col2 = st.columns([1, 2])

        with man_col1:
            sample_type_map = {}
            ion_mode_map = {}
            for f_obj, sp, s, t, ex, mode in processing_tasks:
                lbl = f"[{sp}] {f_obj.name} [{mode}] ({t} - {s})"
                sample_type_map.setdefault(s, []).append(lbl)
                ion_mode_map.setdefault(mode, []).append(lbl)

            man_condition_options = list(all_condition_map.keys())

            def update_man_files_from_manual_selectors():
                selected_sample_types = st.session_state.get("man_sample_type_select", [])
                selected_ion_modes = st.session_state.get("man_ion_mode_select", [])
                selected_groups = st.session_state.get("man_quick_select", [])
                selected_extra_filters = {
                    col: st.session_state.get(f"man_extra_filter_{col}", [])
                    for col in extra_filter_cols
                }

                if selected_sample_types:
                    new_files = []
                    for f_obj, sp, s, t, ex, mode in processing_tasks:
                        if s in selected_sample_types and (not selected_ion_modes or mode in selected_ion_modes):
                            new_files.append(f"[{sp}] {f_obj.name} [{mode}] ({t} - {s})")
                elif selected_ion_modes:
                    new_files = []
                    for ion_mode in selected_ion_modes:
                        new_files.extend(ion_mode_map.get(ion_mode, []))
                elif selected_groups:
                    new_files = []
                    for cond in selected_groups:
                        new_files.extend(all_condition_map[cond])
                else:
                    new_files = selected_file_labels

                new_files = labels_matching_extra_filters(list(dict.fromkeys(new_files)), selected_extra_filters)
                st.session_state.man_files = new_files

            if extra_filter_cols:
                with st.expander("Additional manual file filters", expanded=False):
                    for filter_col in extra_filter_cols:
                        values = sorted({
                            str(_extra_mapping_metadata_for_file(task[0].name).get(filter_col))
                            for task in processing_tasks
                            if filter_col in _extra_mapping_metadata_for_file(task[0].name)
                            and not pd.isna(_extra_mapping_metadata_for_file(task[0].name).get(filter_col))
                        })
                        st.multiselect(
                            f"Filter manual files by {filter_col}:",
                            options=values,
                            default=values,
                            key=f"man_extra_filter_{filter_col}",
                            on_change=update_man_files_from_manual_selectors
                        )

            man_sample_type_options = sorted(sample_type_map.keys())
            st.multiselect(
                "Select Sample Type to Load All Matching Files:",
                options=man_sample_type_options,
                key="man_sample_type_select",
                on_change=update_man_files_from_manual_selectors,
                help="Automatically selects every uploaded file for the chosen sample type(s) in this Manual Integration Tool only."
            )

            man_ion_mode_options = sorted(ion_mode_map.keys())
            st.multiselect(
                "Filter Manual Files by Ionization Mode:",
                options=man_ion_mode_options,
                key="man_ion_mode_select",
                on_change=update_man_files_from_manual_selectors,
                help="Use this with sample type to load only Positive or Negative files. If no sample type is selected, this loads all files for the selected ionization mode(s)."
            )

            man_selected_conditions = st.multiselect(
                "Quick Select by Groups (Sample Type & Treatment):",
                options=man_condition_options,
                key="man_quick_select",
                on_change=update_man_files_from_manual_selectors,
                help="Select one or more groups to quickly load all their files. If sample type or ionization mode is selected above, those selections take priority. If all are empty, this uses the files selected at the top of the tab."
            )

            if "man_files" not in st.session_state:
                update_man_files_from_manual_selectors()

            manual_extra_filter_selections = {
                col: st.session_state.get(f"man_extra_filter_{col}", [])
                for col in extra_filter_cols
            }
            manual_file_options = labels_matching_extra_filters(file_labels, manual_extra_filter_selections)
            st.session_state.man_files = [
                label for label in st.session_state.get("man_files", [])
                if label in manual_file_options
            ]

            target_files = st.multiselect("Select Files to Manually Integrate:", options=manual_file_options, key="man_files")

            restored_manual_event = None
            restored_manual_source = None
            if selected_metabolite and target_files and st.session_state.get('results_df') is not None:
                restored_df = st.session_state.results_df
                for target_label in target_files:
                    restored_task = next((
                        item for item in processing_tasks
                        if f"[{item[1]}] {item[0].name} [{item[5]}] ({item[3]} - {item[2]})" == target_label
                    ), None)
                    if restored_task is None:
                        continue
                    restored_idx = _find_result_row_index(
                        restored_df, restored_task[0].name,
                        selected_metabolite, restored_task[5]
                    )
                    if restored_idx is None:
                        continue
                    events = _parse_manual_edit_log(
                        restored_df.at[restored_idx, 'Manual_Edit_Log']
                        if 'Manual_Edit_Log' in restored_df.columns else None
                    )
                    restored_manual_event = next((
                        event for event in reversed(events)
                        if event.get('type') == 'Reintegration'
                    ), None)
                    if restored_manual_event:
                        restored_manual_source = restored_task[0].name
                        break

            if restored_manual_event:
                st.info(
                    "Loaded saved reintegration settings from "
                    f"`{restored_manual_source}`. Applying a new integration will "
                    "append another audit event."
                )

            if selected_metabolite:
                try:
                    t_row = df_targets_vis[df_targets_vis['Name'] == selected_metabolite].iloc[0]

                    if 'Formula' in t_row and pd.notna(t_row['Formula']):
                        formula = parse_formula(str(t_row['Formula']))
                        tracer_str = str(t_row.get('Tracer', params.get('tracer_element', '13C')))
                        try:
                            tracer_el, _ = parse_tracer(tracer_str, DEFAULT_ISODATA)
                            n_tracer = formula.get(tracer_el, None)
                        except:
                            n_tracer = None

                        if n_tracer is None or n_tracer <= 0:
                            if 'Carbons' in t_row and pd.notna(t_row['Carbons']):
                                n_tracer = int(t_row['Carbons'])
                            else:
                                st.error(f"❌ **Cannot determine number of tracer atoms for {selected_metabolite}**")
                                max_isos = 0
                                n_tracer = 0
                        max_isos = n_tracer + 1 if n_tracer > 0 else 0
                    else:
                        if 'Carbons' in t_row and pd.notna(t_row['Carbons']) and int(t_row['Carbons']) > 0:
                            max_isos = int(t_row['Carbons']) + 1
                        else:
                            st.error(f"❌ **Missing or invalid tracer atom count/Formula for {selected_metabolite}**")
                            st.info("Please add a 'Formula' column (and optionally 'Tracer') to your metadata file for accurate multi‑element correction.")
                            max_isos = 0

                except Exception as e:
                    st.error(f"❌ **Error reading data:** Could not calculate isotopologue count for {selected_metabolite}")
                    max_isos = 0
            else:
                max_isos = 0

            if max_isos > 0:
                iso_opts = [f"M+{i}" for i in range(max_isos)]
                restored_isos = (
                    [iso for iso in restored_manual_event.get('isotopologues', []) if iso in iso_opts]
                    if restored_manual_event else []
                )
                selected_isos = st.multiselect(
                    "Isotopologues to Integrate:", options=iso_opts,
                    default=restored_isos or iso_opts,
                    key=f"manual_isos_{selected_metabolite}",
                )
                algo_options = ["Standard (Exact Slider Bounds)"]
                restored_algo = restored_manual_event.get('algorithm') if restored_manual_event else None
                algo = st.selectbox(
                    "Peak Integration Algorithm:", algo_options,
                    index=algo_options.index(restored_algo) if restored_algo in algo_options else 0,
                    key=f"manual_algo_{selected_metabolite}",
                )
                restored_sigma = restored_manual_event.get('smoothing') if restored_manual_event else None
                local_sigma = st.slider(
                    "Local Smoothing (Sigma points)", min_value=0.0, max_value=15.0,
                    value=float(restored_sigma) if restored_sigma is not None else 1.2,
                    step=0.1, key=f"manual_sigma_{selected_metabolite}",
                    help="Visualisation only; quantitative areas use the original unsmoothed EIC.",
                )

                baseline_enabled = st.checkbox(
                    "Apply local linear baseline subtraction",
                    value=bool(restored_manual_event.get('baseline_enabled', False)) if restored_manual_event else False,
                    key=f"manual_baseline_{selected_metabolite}",
                    help=(
                        "Leave unchecked to reproduce automatic extraction. Enable only "
                        "when there is a clearly raised or sloping background beneath the "
                        "peak. Avoid using it when neighboring peaks fall inside either "
                        "baseline flank."
                    ),
                )
                st.caption(
                    "Leave this unchecked when reproducing automatic extraction. Enable it "
                    "only for a clearly raised or sloping background beneath the peak. Avoid "
                    "it when neighboring peaks fall inside either flank, because they can "
                    "inflate the estimated baseline and remove too much signal."
                )
                baseline_col1, baseline_col2 = st.columns(2)
                with baseline_col1:
                    baseline_flank_min = st.number_input(
                        "Baseline flank (min)", min_value=0.01, max_value=1.0,
                        value=float(restored_manual_event.get('baseline_flank_min') or 0.05) if restored_manual_event else 0.05,
                        step=0.01, format="%.2f",
                        key=f"manual_baseline_flank_{selected_metabolite}",
                        disabled=not baseline_enabled,
                        help=(
                            "Width, in minutes, of the signal region sampled immediately "
                            "before the RT start and immediately after the RT end. These "
                            "regions are outside the integrated peak."
                        ),
                    )
                with baseline_col2:
                    baseline_min_scans = st.number_input(
                        "Minimum flank scans/side", min_value=1, max_value=50,
                        value=int(restored_manual_event.get('baseline_min_scans') or 3) if restored_manual_event else 3,
                        step=1,
                        key=f"manual_baseline_scans_{selected_metabolite}",
                        disabled=not baseline_enabled,
                        help=(
                            "Minimum number of MS1 scans required in each flank. If either "
                            "the left or right flank has fewer scans, baseline subtraction "
                            "is not applied and the unadjusted exact-bound area is retained."
                        ),
                    )

                with st.expander("What do the local-baseline settings mean?"):
                    st.markdown(
                        "**Baseline flank (min)** defines the width of two regions outside "
                        "the selected peak—one immediately before Start RT and one immediately "
                        "after End RT. For example, with bounds **5.00–5.30 min** and a "
                        "**0.05 min** flank, the app samples **4.95–5.00 min** and "
                        "**5.30–5.35 min**.\n\n"
                        "**Minimum flank scans/side** is the minimum number of MS1 measurements "
                        "required in each of those regions. The app takes the median intensity "
                        "on each side, connects those values with a straight line, and subtracts "
                        "that local baseline from the peak signal.\n\n"
                        "If either side contains too few scans, the app safely skips baseline "
                        "subtraction, retains the unadjusted exact-bound area, and records the "
                        "fallback in the audit details."
                    )

                # Historical blank-subtraction fields remain readable, but new
                # manual integrations use the automatic blank fold-change rule.
                st.info(
                    "**Blank handling:** as in automatic extraction, the compound's maximum "
                    f"peak height must be at least **{params['blank_fold_change']:.1f}×** "
                    "the maximum blank peak height. This is a QC filter, not a numerical "
                    "background correction."
                )
                st.caption(
                    "Only the selected isotopologues are changed. Select all isotopologues "
                    "when first enabling or changing correction settings for a fully uniform MID."
                )

                if 'RT' not in t_row or pd.isna(t_row['RT']):
                    st.error("⚠️ **Missing RT!** The Retention Time is missing in your metadata for this metabolite.")
                    rt_bounds = (0.0, 0.0)
                else:
                    target_rt_val = float(t_row['RT'])
                    default_window = 0.25
                    restored_start = restored_manual_event.get('rt_start') if restored_manual_event else None
                    restored_end = restored_manual_event.get('rt_end') if restored_manual_event else None

                    st.markdown("**Exact Integration Window (RT)**")
                    rt_col1, rt_col2 = st.columns(2)

                    with rt_col1:
                        rt_start = st.number_input(
                            "Start RT (min)",
                            value=float(restored_start) if restored_start is not None else max(0.0, target_rt_val - default_window),
                            step=0.01, format="%.2f", key=f"manual_rt_start_{selected_metabolite}",
                        )
                    with rt_col2:
                        rt_end = st.number_input(
                            "End RT (min)",
                            value=float(restored_end) if restored_end is not None else target_rt_val + default_window,
                            step=0.01, format="%.2f", key=f"manual_rt_end_{selected_metabolite}",
                        )

                    rt_bounds = (rt_start, rt_end)

            else:
                selected_isos = []
                st.warning("Cannot proceed without a valid tracer atom count for the selected metabolite.")

        with man_col2:
            if st.button("Integrate & Update Global Data", type="primary"):
                if not selected_metabolite or not selected_isos:
                    st.warning("Please select a metabolite and at least one isotopologue.")
                elif not target_files:
                    st.warning("Please select at least one file to integrate.")
                elif rt_bounds[0] >= rt_bounds[1]:
                    st.warning("Start RT must be lower than End RT.")
                else:
                    with st.spinner(f"Applying {algo.split(' ')[0]} to {len(target_files)} files..."):

                        target_row = df_targets_vis[df_targets_vis['Name'] == selected_metabolite].iloc[0]
                        df = st.session_state.results_df
                        success_count = 0
                        plots_to_render = []
                        manual_blank_cache = {}
                        manual_correction_warnings = set()

                        for t_file in target_files:
                            task = next((item for item in processing_tasks if f"[{item[1]}] {item[0].name} [{item[5]}] ({item[3]} - {item[2]})" == t_file), None)

                            if task:
                                f_obj = task[0]
                                met_obj = generate_targets(
                                    pd.DataFrame([target_row]), params['mz_tol_ppm'], params['rt_window_min']
                                )[0]
                                ion_mode_str = task[5]
                                blank_files = blank_pos if ion_mode_str == "Positive" else blank_neg
                                blank_cache_key = (
                                    ion_mode_str, met_obj['name'], met_obj.get('tracer_str'),
                                    met_obj.get('n_tracer'), float(met_obj['target_rt']),
                                )

                                if blank_cache_key not in manual_blank_cache:
                                    if blank_files:
                                        manual_blank_cache[blank_cache_key] = {
                                            'blank_heights': extract_blank_max_heights(
                                                blank_files, [met_obj], params
                                            ).get(met_obj['name'], []),
                                            'files_used': [getattr(f, 'name', 'blank') for f in blank_files],
                                        }
                                    else:
                                        manual_blank_cache[blank_cache_key] = {
                                            'blank_heights': [],
                                            'files_used': [],
                                        }

                                blank_metrics = manual_blank_cache[blank_cache_key]

                                fig_man = go.Figure()
                                _add_rt_boundary_markers(fig_man, rt_bounds)
                                new_areas = {}
                                new_audit_details = {
                                    '_blank_files': {
                                        'used': list(blank_metrics.get('files_used', [])),
                                    }
                                }
                                all_iso_labels = [
                                    f"M+{i}" for i in range(len(met_obj['isotopes']))
                                ]
                                extracted_eics = {
                                    iso_label: {'rt': [], 'int': []}
                                    for iso_label in all_iso_labels
                                }

                                f_obj.seek(0)
                                with mzml.read(f_obj) as reader:
                                    for spectrum in reader:
                                        if spectrum.get('ms level') != 1:
                                            continue
                                        try:
                                            rt = spectrum['scanList']['scan'][0]['scan start time']
                                            if hasattr(rt, 'unit_info') and 'second' in str(rt.unit_info).lower():
                                                rt /= 60.0
                                        except KeyError:
                                            continue

                                        extract_flank = float(baseline_flank_min) if baseline_enabled else 0.0
                                        if rt_bounds[0] - extract_flank <= rt <= rt_bounds[1] + extract_flank:
                                            mz_array = spectrum.get('m/z array', np.array([]))
                                            int_array = spectrum.get('intensity array', np.array([]))

                                            for iso_label in all_iso_labels:
                                                iso_idx = int(iso_label.split('+')[1])
                                                if iso_idx >= len(met_obj['isotopes']):
                                                    continue
                                                iso = met_obj['isotopes'][iso_idx]
                                                mask = (mz_array >= iso['mz_min']) & (mz_array <= iso['mz_max'])
                                                intensity = float(np.max(int_array[mask])) if np.any(mask) else 0.0
                                                extracted_eics[iso_label]['rt'].append(float(rt))
                                                extracted_eics[iso_label]['int'].append(intensity)

                                # Blank/background filtering is applied per isotopologue
                                # against its own blank height, not as a single
                                # compound-level comparison (see
                                # _apply_isotopologue_blank_filter for the same logic used
                                # in the automatic extraction path).
                                blank_heights = blank_metrics.get('blank_heights', [])
                                blank_failed_isos = set()

                                for iso_label in selected_isos:
                                    raw_rt = extracted_eics[iso_label]['rt']
                                    raw_int = extracted_eics[iso_label]['int']

                                    if len(raw_rt) == 0:
                                        new_areas[iso_label] = 0.0
                                        new_audit_details[iso_label] = {
                                            'status': 'No EIC scans in requested window'
                                        }
                                        continue

                                    sort_idx = np.argsort(raw_rt)
                                    rt_arr = np.array(raw_rt, dtype=float)[sort_idx]
                                    int_arr = np.array(raw_int, dtype=float)[sort_idx]
                                    smooth_int, _ = _manual_integration_mask(
                                        rt_arr, int_arr, rt_bounds, algo, local_sigma
                                    )
                                    integration = integrate_exact_bounds(
                                        rt_arr,
                                        int_arr,
                                        rt_bounds,
                                        baseline_enabled=baseline_enabled,
                                        baseline_flank_min=baseline_flank_min,
                                        minimum_flank_scans=baseline_min_scans,
                                    )
                                    if baseline_enabled and integration['baseline_status'] != 'Applied':
                                        manual_correction_warnings.add(
                                            "Local baseline subtraction fell back to unadjusted area for at least "
                                            "one isotopologue because the requested flank scans were unavailable."
                                        )

                                    bounded_rt = integration['peak_rt']
                                    bounded_corrected = integration['peak_corrected']

                                    if len(bounded_rt) > 1:
                                        measured_area = integration['area_baseline_corrected']
                                        measured_height = integration['height_baseline_corrected']
                                        manual_area = measured_area
                                        iso_idx = int(iso_label.split('+')[1])
                                        this_blank_height = (
                                            blank_heights[iso_idx] if iso_idx < len(blank_heights) else 0.0
                                        )
                                        blank_failed = fails_blank_fold_change(
                                            measured_height, this_blank_height, params['blank_fold_change']
                                        )
                                        if blank_failed:
                                            blank_failed_isos.add(iso_label)
                                        threshold_failed = (
                                            measured_height < params['noise_threshold'] or
                                            manual_area < params['min_peak_area']
                                        )
                                        if threshold_failed or blank_failed:
                                            manual_area = 0.0

                                        new_areas[iso_label] = manual_area
                                        status_parts = [integration['baseline_status']]
                                        if blank_failed:
                                            status_parts.append('blank fold-change failed')
                                        if threshold_failed:
                                            status_parts.append('height/area threshold failed')
                                        new_audit_details[iso_label] = {
                                            'area_unadjusted': integration['area_unadjusted'],
                                            'baseline_area': integration['baseline_area'],
                                            'area_baseline_corrected': measured_area,
                                            'baseline_applied': integration['baseline_status'] == 'Applied',
                                            'blank_reference_height': this_blank_height,
                                            'blank_qc_sample_height': measured_height,
                                            'blank_fold_change_threshold': params['blank_fold_change'],
                                            'blank_qc_passed': not blank_failed,
                                            'area_used': manual_area,
                                            'height_baseline_corrected': measured_height,
                                            'baseline_start': integration['baseline_start'],
                                            'baseline_end': integration['baseline_end'],
                                            'left_flank_scans': integration['left_flank_scans'],
                                            'right_flank_scans': integration['right_flank_scans'],
                                            'excluded_from_mid': False,
                                            'status': '; '.join(status_parts),
                                        }
                                        label_note = (
                                            "blank failed" if blank_failed
                                            else f"area {measured_area:,.0f}; height {measured_height:,.0f}"
                                        )

                                        fig_man.add_trace(go.Scatter(
                                            x=rt_arr, y=int_arr, mode='lines',
                                            name=f'{iso_label} Raw'
                                        ))
                                        if local_sigma > 0:
                                            fig_man.add_trace(go.Scatter(
                                                x=rt_arr, y=smooth_int, mode='lines',
                                                line=dict(dash='dash'), name=f'{iso_label} Smooth (display)'
                                            ))
                                        if integration['baseline_status'] == 'Applied':
                                            fig_man.add_trace(go.Scatter(
                                                x=bounded_rt,
                                                y=integration['baseline_values'],
                                                mode='lines', line=dict(color='orange', dash='dot'),
                                                name=f'{iso_label} Local baseline'
                                            ))
                                        fig_man.add_trace(go.Scatter(
                                            x=bounded_rt,
                                            y=bounded_corrected,
                                            fill='tozeroy',
                                            mode='none',
                                            name=f'{iso_label} ({manual_area:,.0f}; {label_note})'
                                        ))
                                    else:
                                        new_areas[iso_label] = 0.0
                                        new_audit_details[iso_label] = {
                                            'status': integration['baseline_status']
                                        }

                                if new_areas:
                                    fig_man.update_layout(title=f"Integrated: {f_obj.name}", height=300, margin=dict(l=10, r=10, t=30, b=10))
                                    plots_to_render.append(fig_man)

                                    expected_n_tracer = max(0, len(met_obj['isotopes']) - 1)
                                    df, row_idx = _ensure_manual_result_row(
                                        df,
                                        task,
                                        target_row,
                                        selected_metabolite,
                                        expected_n_tracer
                                    )

                                    if row_idx is not None:
                                            count_col = 'N_Tracer' if 'N_Tracer' in df.columns else next((col for col in df.columns if 'carbon' in col.lower()), 'Total_Carbons')
                                            carbons = int(df.at[row_idx, count_col]) if count_col in df.columns and pd.notna(df.at[row_idx, count_col]) else expected_n_tracer

                                            # --- BUG FIX 1: Prevent NaN crashes from completely empty initial peaks ---
                                            raw_areas = []
                                            for i in range(carbons + 1):
                                                col_name = f"M+{i} Area (Raw)"
                                                val = df.at[row_idx, col_name] if col_name in df.columns else 0.0
                                                raw_areas.append(0.0 if pd.isna(val) else float(val))

                                            for iso_label, manual_area in new_areas.items():
                                                i = int(iso_label.split('+')[1])
                                                if i <= carbons:
                                                    raw_areas[i] = manual_area
                                                    audit = new_audit_details.get(iso_label, {})
                                                    audit_values = {
                                                        f"M+{i} Area (Unadjusted)": audit.get('area_unadjusted', 0.0),
                                                        f"M+{i} Excluded From MID": 0,
                                                        f"M+{i} Excluded By Blank": int(not audit.get('blank_qc_passed', True)),
                                                    }
                                                    # Only record baseline columns when local baseline
                                                    # subtraction actually ran for this isotopologue --
                                                    # otherwise they'd silently duplicate the unadjusted
                                                    # area and look like a real correction was applied.
                                                    if audit.get('baseline_applied'):
                                                        audit_values[f"M+{i} Baseline Area"] = audit.get('baseline_area', 0.0)
                                                        audit_values[f"M+{i} Area (Baseline Corrected)"] = audit.get('area_baseline_corrected', 0.0)
                                                    for audit_col, audit_value in audit_values.items():
                                                        df.at[row_idx, audit_col] = audit_value

                                            t_row = df_targets_vis[df_targets_vis['Name'] == selected_metabolite].iloc[0]
                                            formula = parse_formula(str(t_row['Formula']))
                                            tracer_str = str(t_row.get('Tracer', '13C'))
                                            tracer_el, _ = parse_tracer(tracer_str, DEFAULT_ISODATA)
                                            formula[tracer_el] = len(raw_areas) - 1
                                            # Blank exclusion is already baked into raw_areas
                                            # per isotopologue above (manual_area was zeroed
                                            # for any blank-failed M+i), so the correction runs
                                            # on whatever real signal remains rather than being
                                            # gated all-or-nothing at the compound level.
                                            metrics = _calculate_enrichment_metrics(
                                                raw_areas, formula, tracer_str, params
                                            )
                                            corrected_areas = metrics['corrected_areas']
                                            n_tracer = metrics['n_tracer']

                                            area_col = next((col for col in df.columns if 'total' in col.lower() and 'area' in col.lower()), 'Total_Peak_Area')
                                            qc_col = next((col for col in df.columns if 'qc' in col.lower()), 'QC_Status')

                                            for i in range(n_tracer + 1):
                                                if f"M+{i} Area (Raw)" in df.columns:
                                                    df.at[row_idx, f"M+{i} Area (Raw)"] = raw_areas[i] if i < len(raw_areas) else 0.0
                                                if f"M+{i} Area (Corrected)" in df.columns:
                                                    df.at[row_idx, f"M+{i} Area (Corrected)"] = corrected_areas[i] if i < len(corrected_areas) else 0.0

                                            df.at[row_idx, area_col] = sum(corrected_areas)
                                            df.at[row_idx, 'Tracer'] = tracer_str
                                            df.at[row_idx, 'Tracer_Element'] = metrics['tracer_el']
                                            df.at[row_idx, 'N_Tracer'] = metrics['n_tracer']
                                            df.at[row_idx, 'Raw_ME_%'] = metrics['raw_me_pct']
                                            df.at[row_idx, 'Natural_Abundance_ME_%'] = metrics['natural_me_pct']
                                            df.at[row_idx, 'Excess_Enrichment_%'] = metrics['excess_pct']
                                            _write_mid_fraction_columns(
                                                row_idx,
                                                metrics,
                                                n_tracer,
                                                lambda idx, col, val: df.at.__setitem__((idx, col), val)
                                            )
                                            if 'Raw_Enrichment_%' in df.columns:
                                                df.at[row_idx, 'Raw_Enrichment_%'] = metrics['raw_me_pct']
                                            if 'Excess_13C_Enrichment_%' in df.columns:
                                                df.at[row_idx, 'Excess_13C_Enrichment_%'] = metrics['excess_pct']

                                            if sum(raw_areas) <= 0 and blank_failed_isos:
                                                df.at[row_idx, qc_col] = (
                                                    f"Fail: Blank (FC < {params['blank_fold_change']:.1f})"
                                                )
                                            elif blank_failed_isos:
                                                df.at[row_idx, qc_col] = "Manually Edited: Some isotopologues blank-filtered"
                                            else:
                                                df.at[row_idx, qc_col] = "Manually Edited"
                                            _append_manual_edit_event(
                                                df,
                                                row_idx,
                                                edit_type="Reintegration",
                                                isotopologues=selected_isos,
                                                rt_bounds=rt_bounds,
                                                algorithm=algo,
                                                smoothing=local_sigma,
                                                baseline_enabled=baseline_enabled,
                                                baseline_flank_min=baseline_flank_min,
                                                baseline_min_scans=baseline_min_scans,
                                                n_blanks_used=len(blank_metrics.get('files_used', [])),
                                                correction_details=new_audit_details,
                                            )

                                            success_count += 1

                        if success_count > 0:
                            st.session_state.results_df = _finalize_results_df(df)
                            st.session_state._manual_edit_notice = (
                                f"Successfully updated global data for **{success_count} files**! "
                                "The Results table, Manual Edit History, and analysis charts are now refreshed."
                            )
                            if manual_correction_warnings:
                                st.session_state._manual_edit_warning = " ".join(
                                    sorted(manual_correction_warnings)
                                )
                            st.session_state._manual_edit_plots = plots_to_render
                            st.rerun()
                        else:
                            st.error("Could not find matching files/metabolites in global memory to update.")

            st.markdown("---")
            if st.button("🗑️ Remove Selected Isotopologues", type="secondary"):
                if not selected_metabolite or not selected_isos:
                    st.warning("Please select a metabolite and at least one isotopologue.")
                elif not target_files:
                    st.warning("Please select at least one file to modify.")
                else:
                    with st.spinner(f"Removing isotopologues from {len(target_files)} files..."):
                        df = st.session_state.results_df
                        success_count = 0

                        for t_file in target_files:
                            task = next((item for item in processing_tasks if f"[{item[1]}] {item[0].name} [{item[5]}] ({item[3]} - {item[2]})" == t_file), None)

                            if task:
                                if df is not None and not df.empty:
                                    exact_filename = task[0].name
                                    exact_metabolite = selected_metabolite
                                    exact_ion_mode = task[5]

                                    row_idx = _find_result_row_index(df, exact_filename, exact_metabolite, exact_ion_mode)

                                    if row_idx is not None:
                                        matching_targets = df_targets_vis[df_targets_vis['Name'] == selected_metabolite]
                                        if matching_targets.empty:
                                            st.error(
                                                f"Could not find metadata for '{selected_metabolite}' to determine "
                                                "its tracer atom count. Skipping this file."
                                            )
                                            continue

                                        # Authoritative atom count comes from the compound's formula/tracer
                                        # metadata, not from whatever columns happen to exist in the results
                                        # dataframe -- a restored session missing N_Tracer/carbon columns must
                                        # not silently truncate the MID for compounds with >5 tracer atoms.
                                        t_row = matching_targets.iloc[0]
                                        formula = parse_formula(str(t_row['Formula']))
                                        if (
                                            'Tracer' not in t_row.index
                                            or pd.isna(t_row['Tracer'])
                                            or str(t_row['Tracer']).strip() == ''
                                        ):
                                            st.error(
                                                f"No tracer specified for '{selected_metabolite}' in metadata. "
                                                "Skipping this file."
                                            )
                                            continue
                                        tracer_str = str(t_row['Tracer']).strip()
                                        tracer_el, _ = parse_tracer(tracer_str, DEFAULT_ISODATA)
                                        carbons = formula.get(tracer_el, 0)

                                        raw_areas = []
                                        for i in range(carbons + 1):
                                            col_name = f"M+{i} Area (Raw)"
                                            val = df.at[row_idx, col_name] if col_name in df.columns else 0.0
                                            raw_areas.append(0.0 if pd.isna(val) else float(val))

                                        for iso_label in selected_isos:
                                            i = int(iso_label.split('+')[1])
                                            if i <= carbons:
                                                previous_area = raw_areas[i]
                                                # No baseline math happens in this action -- it only
                                                # zeroes an existing area -- so Baseline Area/Area
                                                # (Baseline Corrected) are intentionally left untouched
                                                # rather than written with placeholder values.
                                                unadjusted_col = f"M+{i} Area (Unadjusted)"
                                                if unadjusted_col not in df.columns or pd.isna(df.at[row_idx, unadjusted_col]):
                                                    df.at[row_idx, unadjusted_col] = previous_area
                                                df.at[row_idx, f"M+{i} Excluded From MID"] = 1
                                                raw_areas[i] = 0.0

                                        formula[tracer_el] = carbons
                                        metrics = _calculate_enrichment_metrics(raw_areas, formula, tracer_str, params)
                                        corrected_areas = metrics['corrected_areas']
                                        n_tracer = metrics['n_tracer']

                                        area_col = next((col for col in df.columns if 'total' in col.lower() and 'area' in col.lower()), 'Total_Peak_Area')
                                        qc_col = next((col for col in df.columns if 'qc' in col.lower()), 'QC_Status')

                                        for i in range(n_tracer + 1):
                                            if f"M+{i} Area (Raw)" in df.columns:
                                                df.at[row_idx, f"M+{i} Area (Raw)"] = raw_areas[i] if i < len(raw_areas) else 0.0
                                            if f"M+{i} Area (Corrected)" in df.columns:
                                                df.at[row_idx, f"M+{i} Area (Corrected)"] = corrected_areas[i] if i < len(corrected_areas) else 0.0

                                        df.at[row_idx, area_col] = sum(corrected_areas)
                                        df.at[row_idx, 'Tracer'] = tracer_str
                                        df.at[row_idx, 'Tracer_Element'] = metrics['tracer_el']
                                        df.at[row_idx, 'N_Tracer'] = metrics['n_tracer']
                                        df.at[row_idx, 'Raw_ME_%'] = metrics['raw_me_pct']
                                        df.at[row_idx, 'Natural_Abundance_ME_%'] = metrics['natural_me_pct']
                                        df.at[row_idx, 'Excess_Enrichment_%'] = metrics['excess_pct']
                                        _write_mid_fraction_columns(
                                            row_idx,
                                            metrics,
                                            n_tracer,
                                            lambda idx, col, val: df.at.__setitem__((idx, col), val)
                                        )
                                        if 'Raw_Enrichment_%' in df.columns:
                                            df.at[row_idx, 'Raw_Enrichment_%'] = metrics['raw_me_pct'] if not np.isnan(metrics['raw_me_pct']) else 0.0
                                        if 'Excess_13C_Enrichment_%' in df.columns:
                                            df.at[row_idx, 'Excess_13C_Enrichment_%'] = metrics['excess_pct']

                                        df.at[row_idx, qc_col] = "Manually Edited"
                                        baseline_value = df.at[row_idx, 'Local_Baseline_Applied'] if 'Local_Baseline_Applied' in df.columns else False
                                        existing_baseline_enabled = bool(baseline_value) if pd.notna(baseline_value) else False
                                        exclusion_details = {
                                            iso_label: {
                                                'excluded_from_mid': True,
                                                'status': 'Manually excluded from MID calculation',
                                            }
                                            for iso_label in selected_isos
                                        }
                                        _append_manual_edit_event(
                                            df,
                                            row_idx,
                                            edit_type="Remove isotopologues",
                                            isotopologues=selected_isos,
                                            baseline_enabled=existing_baseline_enabled,
                                            baseline_flank_min=(
                                                df.at[row_idx, 'Baseline_Flank_min']
                                                if 'Baseline_Flank_min' in df.columns and pd.notna(df.at[row_idx, 'Baseline_Flank_min']) else None
                                            ),
                                            baseline_min_scans=(
                                                df.at[row_idx, 'Baseline_Min_Scans']
                                                if 'Baseline_Min_Scans' in df.columns and pd.notna(df.at[row_idx, 'Baseline_Min_Scans']) else None
                                            ),
                                            n_blanks_used=(
                                                df.at[row_idx, 'N_Blanks_Used']
                                                if 'N_Blanks_Used' in df.columns and pd.notna(df.at[row_idx, 'N_Blanks_Used']) else 0
                                            ),
                                            correction_details=exclusion_details,
                                        )
                                        success_count += 1

                        if success_count > 0:
                            st.session_state.results_df = _finalize_results_df(df)
                            st.session_state._manual_edit_notice = (
                                f"Successfully removed selected isotopologues and updated global data for **{success_count} files**! "
                                "The Results table, Manual Edit History, and analysis charts are now refreshed."
                            )
                            st.session_state._manual_edit_plots = []
                            st.rerun()
                        else:
                            st.error("Could not find matching files/metabolites in global memory to update.")

    else:
        st.warning("⚠️ The Peak Visualizer requires raw `mzML` files.")
def render_tab_analysis(params):
    st.markdown("### Interactive Data Visualizations")

    if st.session_state.get('results_df') is not None and not st.session_state.results_df.empty and 'Metabolite' in st.session_state.results_df.columns:
        st.session_state.results_df = _finalize_results_df(st.session_state.results_df)
        df = st.session_state.results_df.copy()
        # Keep an unpruned copy for replicate accounting.  The chart filters
        # below intentionally remove QC-failed/undetected rows from the values
        # used to calculate means and SDs, but those files must still remain in
        # the "total files" denominator after restoring a session.
        df_all_results = df.copy()

        if 'M+0 Area (Corrected)' not in df.columns:
            st.warning("⚠️ **Old Session Data Detected.** Please re-run extraction so the app can apply Isotope correction.")
        else:
            hide_qc_fails = st.checkbox("✅ Hide QC-Failed Metabolites from Charts", value=True)
            hide_zeros = st.checkbox("✅ Hide Undetected Metabolites (Zero Area)", value=True)


            # We allow BOTH 'Pass' and 'Manually Edited' data to show up in the charts!
            if hide_qc_fails:
                df = df[df['QC_Status'].str.startswith('Pass') | df['QC_Status'].isin(['Manually Edited'])]
            if hide_zeros:
                df = df[df['Total_Peak_Area'] > 0]

            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            with f_col1:
                available_species = df['Species'].unique().tolist()
                selected_species = st.multiselect("Filter by Species:", options=available_species, default=available_species)
            with f_col2:
                available_types = df['Sample_Type'].unique().tolist()
                selected_types = st.multiselect("Filter by Sample Type:", options=available_types, default=available_types)
            with f_col3:
                available_treatments = df['Treatment'].unique().tolist()
                selected_treatments = st.multiselect("Filter by Treatment:", options=available_treatments, default=available_treatments)
            with f_col4:
                available_modes = df['Ion_Mode'].unique().tolist()
                selected_modes = st.multiselect("Filter by Ion Mode:", options=available_modes, default=available_modes)

            df_filtered = df[(df['Ion_Mode'].isin(selected_modes)) &
                             (df['Species'].isin(selected_species)) &
                             (df['Sample_Type'].isin(selected_types)) &
                             (df['Treatment'].isin(selected_treatments))].copy()
            df_variance_scope = df_all_results[
                (df_all_results['Ion_Mode'].isin(selected_modes)) &
                (df_all_results['Species'].isin(selected_species)) &
                (df_all_results['Sample_Type'].isin(selected_types)) &
                (df_all_results['Treatment'].isin(selected_treatments))
            ].copy()

            extra_filter_cols = [
                col for col in st.session_state.get('mapping_extra_columns', [])
                if col in df_filtered.columns
            ]
            if extra_filter_cols:
                with st.expander("Additional mapping filters", expanded=False):
                    for filter_col in extra_filter_cols:
                        filter_values = sorted(
                            df_filtered[filter_col]
                            .dropna()
                            .astype(str)
                            .unique()
                            .tolist()
                        )
                        selected_values = st.multiselect(
                            f"Filter by {filter_col}:",
                            options=filter_values,
                            default=filter_values,
                            key=f"analysis_extra_filter_{filter_col}"
                        )
                        df_filtered = df_filtered[
                            df_filtered[filter_col].isna() |
                            df_filtered[filter_col].astype(str).isin(selected_values)
                        ].copy()
                        df_variance_scope = df_variance_scope[
                            df_variance_scope[filter_col].isna() |
                            df_variance_scope[filter_col].astype(str).isin(selected_values)
                        ].copy()

            # --- Enrichment type ---
            metric_col = st.radio(
                "**Enrichment metric to display:**",
                options=[
                    "Excess ME (corrected)",
                    "Raw ME",
                ],
                index=0,
                horizontal=True,
                key="analysis_enrich_metric"
            )
            if metric_col == "Raw ME":
                enrich_column = "Raw_ME_%" if "Raw_ME_%" in df.columns else "Raw_Enrichment_%"
                metric_label = "Raw Mean Enrichment (%)"
            else:
                enrich_column = _preferred_enrichment_col(df)
                metric_label = "Excess Enrichment (%)"

            if enrich_column not in df.columns:
                st.warning("⚠️ Requested enrichment column not found in current session. Using the best available legacy enrichment column.")
                enrich_column = _preferred_enrichment_col(df)
                metric_label = f"{enrich_column} (%)"

            if not df_filtered.empty:
                df_filtered['Condition'] = (
                    df_filtered['Species'].astype(str) + " - " +
                    df_filtered['Sample_Type'].astype(str) + " - " +
                    df_filtered['Treatment'].astype(str) + " (" +
                    df_filtered['Ion_Mode'].astype(str) + ")"
                )

                st.markdown("#### 🌍 Global Overview: All Metabolites")

                enrich_threshold = st.slider(
                    f"Filter Global Chart (Minimum {metric_label.split('(')[0].strip()})",
                    min_value=0.0, max_value=100.0, value=1.0, step=0.5,
                    help="Only show metabolites that reached this level of enrichment in at least one sample."
                )

                replicate_display_mode = st.radio(
                    "Replicate display:",
                    options=[
                        "Show all detected replicate counts",
                        "Filter by exact number of replicates"
                    ],
                    index=0,
                    horizontal=True,
                    help=(
                        "Use 'Show all' when conditions have different detected replicate counts "
                        "(for example QC n=3 and sample n=2)."
                    ),
                    key="global_enrichment_replicate_display_mode"
                )

                replicate_filter = None
                if replicate_display_mode == "Filter by exact number of replicates":
                    replicate_filter = st.number_input(
                        "Exact number of replicates (n)",
                        min_value=1,
                        max_value=20,
                        value=3,
                        step=1,
                        help="Only show metabolite-condition bars with this exact number of detected replicates."
                    )

                # Keep only rows that have a real displayed value. Otherwise Plotly
                # keeps empty x-axis categories and can show orphan "n=" labels.
                df_chart_values = df_filtered[
                    df_filtered[enrich_column].notna() &
                    (df_filtered[enrich_column] > 0)
                ].copy()

                valid_mets = df_chart_values[df_chart_values[enrich_column] >= enrich_threshold]['Metabolite'].unique()
                df_global_filtered = df_chart_values[df_chart_values['Metabolite'].isin(valid_mets)]

                if not df_global_filtered.empty:
                    df_grouped = df_global_filtered.groupby(['Metabolite', 'Condition'])[enrich_column].agg(
                        Mean='mean',
                        Std_Dev='std',
                        Count='count'
                        ).reset_index()

                    if replicate_filter is not None:
                        df_grouped = df_grouped[df_grouped['Count'] == replicate_filter]

                    if df_grouped.empty:
                        st.info("No metabolites match the current enrichment and replicate filters.")
                        st.markdown("#### 🔍 Deep Dive: Specific Metabolite (MID)")

                    fig_mean = px.bar(
                        df_grouped,
                        x="Metabolite",
                        y="Mean",
                        error_y="Std_Dev",
                        color="Condition",
                        barmode="group",
                        height=700,
                        labels={'Mean': metric_label},
                        title=f"{metric_label} (Mean ± SD)"
                    )

                    fig_mean.update_traces(
                        marker_line_color=params.get('sd_line_color', 'red'),
                        marker_line_width=1.5,
                        error_y=dict(
                            color=params.get('errorbar_color', 'black'),
                            thickness=2
                        )
                    )

                    for trace in fig_mean.data:
                        cond_data = df_grouped[df_grouped['Condition'] == trace.name]
                        if cond_data.empty:
                            continue
                        count_map = cond_data.set_index('Metabolite')['Count'].to_dict()
                        trace.text = [
                            f"n={count_map[m]}" if m in count_map else ""
                            for m in (trace.x if trace.x is not None else [])
                        ]
                        trace.textposition = 'outside'
                        trace.textfont = dict(size=11)

                    display_label = _get_main_tracer_label(df_filtered)

                    fig_mean.update_layout(
                        title=f"Enrichment, {display_label} (%)",
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_mean, use_container_width=True, key="global_mean_chart")

                else:
                    st.info(f"No metabolites reached the {enrich_threshold}% enrichment threshold.")

                st.divider()

                st.markdown("#### 🔍 Deep Dive: Specific Metabolite (MID)")
                st.markdown("*Note: These histograms show the **fully corrected** Fractional Contribution (Natural Abundance + Tracer Purity correction).*")
                st.markdown("### 📊 Replicate Agreement & Quality Control")
                st.markdown("Assess the reproducibility of your extractions using the **Relative Standard Deviation (% RSD)**. A lower RSD indicates tighter agreement between your biological/technical replicates.")

                if 'Condition' not in df_filtered.columns:
                    df_filtered['Condition'] = (
                        df_filtered['Species'].astype(str) + " | " +
                        df_filtered['Treatment'].astype(str) + " " +
                        df_filtered['Sample_Type'].astype(str) + " (" +
                        df_filtered['Ion_Mode'].astype(str) + ")"
                    )

                # Build the same grouping label on the unpruned data. This is
                # what lets a restored file remain visible in total-file counts
                # even when its measurement is excluded from the SD itself.
                df_variance_scope['Condition'] = (
                    df_variance_scope['Species'].astype(str) + " - " +
                    df_variance_scope['Sample_Type'].astype(str) + " - " +
                    df_variance_scope['Treatment'].astype(str) + " (" +
                    df_variance_scope['Ion_Mode'].astype(str) + ")"
                )

                group_cols = ['Condition', 'Metabolite']

                # Only finite measurements contribute to the statistics. Count
                # unique files (rather than CSV rows) so a restored export can
                # never double-count a sample.
                df_summary_values = df_filtered[df_filtered[enrich_column].notna()].copy()
                summary_df = df_summary_values.groupby(group_cols).agg(
                    Replicates_Used=('Filename', 'nunique'),
                    Mean_Enrichment=(enrich_column, 'mean'),
                    Std_Enrichment=(enrich_column, 'std'),
                    Mean_Area=('Total_Peak_Area', 'mean'),
                    Std_Area=('Total_Peak_Area', 'std'),
                    Files_Used=('Filename', lambda values: ', '.join(sorted(set(map(str, values)))))
                ).reset_index()

                total_file_summary = df_variance_scope.groupby(group_cols).agg(
                    Replicates_Total=('Filename', 'nunique'),
                    Files_Total=('Filename', lambda values: ', '.join(sorted(set(map(str, values)))))
                )

                qc_ok = (
                    df_variance_scope['QC_Status'].astype(str).str.startswith('Pass') |
                    df_variance_scope['QC_Status'].isin(['Manually Edited'])
                )
                detected_rep_counts = (
                    df_variance_scope[qc_ok & (df_variance_scope['Total_Peak_Area'].fillna(0) > 0)]
                    .groupby(group_cols)['Filename']
                    .nunique()
                    .rename('Replicates_Detected')
                )

                summary_df = summary_df.merge(total_file_summary, on=group_cols, how='left')
                summary_df = summary_df.merge(detected_rep_counts, on=group_cols, how='left')
                summary_df['Replicates_Detected'] = summary_df['Replicates_Detected'].fillna(0).astype(int)
                summary_df['Replicates_Total'] = summary_df['Replicates_Total'].fillna(0).astype(int)
                summary_df['Replicates (used/det/total)'] = (
                    summary_df['Replicates_Used'].astype(str) + '/' +
                    summary_df['Replicates_Detected'].astype(str) + '/' +
                    summary_df['Replicates_Total'].astype(str)
                )

                summary_df = summary_df.fillna(0)

                tab_sd_chart, tab_data_table = st.tabs([
                    "📈 Global Variance (SD)",
                    "🧮 Summary Data Table"
                ])

                with tab_sd_chart:
                    st.markdown("##### Global Replicate Variance (Standard Deviation)")
                    st.caption(
                        "Replicates are reported as used/detected/total. Full filenames "
                        "remain available in the Summary Data Table."
                    )
                    fig_sd = px.scatter(
                        summary_df,
                        x="Metabolite",
                        y="Std_Enrichment",
                        color="Condition",
                        title="Replicate Agreement (Standard Deviation) Across All Metabolites",
                        hover_data=[
                            "Mean_Enrichment", "Replicates (used/det/total)"
                        ]
                    )
                    fig_sd.update_layout(hovermode="x unified", xaxis_tickangle=-45)

                    if 'sd_threshold' in params:
                        fig_sd.add_hline(y=params['sd_threshold'],
                                        line_dash="dash",
                                        line_color=params.get('sd_line_color', 'red'),
                                        annotation_text=f"SD = {params['sd_threshold']}",
                                        annotation_position="top left")

                    st.plotly_chart(fig_sd, use_container_width=True)

                with tab_data_table:
                    st.markdown("##### Summary Table (Mean, Std)")
                    cols_to_show = ['Condition', 'Metabolite', 'Replicates (used/det/total)',
                                    'Mean_Enrichment', 'Std_Enrichment',
                                    'Mean_Area', 'Std_Area', 'Files_Used', 'Files_Total']
                    df_to_show = summary_df[[c for c in cols_to_show if c in summary_df.columns]]
                    formatted_df = df_to_show.style.format({
                        'Mean_Enrichment': '{:.2f}%',
                        'Std_Enrichment': '{:.2f}%',
                        'Mean_Area': '{:.2f}',
                        'Std_Area': '{:.2f}'
                    })
                    st.dataframe(formatted_df, use_container_width=True)

                    st.download_button(
                        label="📥 Download Replicate Summary Statistics",
                        data=summary_df.to_csv(index=False).encode('utf-8'),
                        file_name='tracer_replicate_summary_statistics.csv',
                        mime='text/csv'
                    )

                chart_met_options = {f"{row['Metabolite']} ({row['Ion_Mode']})": (row['Metabolite'], row['Ion_Mode']) for _, row in df_filtered.drop_duplicates(subset=['Metabolite', 'Ion_Mode']).iterrows()}

                selected_chart_label = st.selectbox("Select Metabolite to Detail:", list(chart_met_options.keys()))
                sel_met, sel_mode = chart_met_options[selected_chart_label]

                df_met = df_filtered[(df_filtered['Metabolite'] == sel_met) & (df_filtered['Ion_Mode'] == sel_mode)].copy()

                if not df_met.empty:
                    max_carbons = int(df_met['N_Tracer'].iloc[0]) if 'N_Tracer' in df_met.columns else int(df_met['Total_Carbons'].iloc[0])
                    m_pct_columns = []

                    # --- SAFE MID CALCULATION ---
                    valid_corr_cols = [f"M+{j} Area (Corrected)" for j in range(max_carbons + 1) if f"M+{j} Area (Corrected)" in df_met.columns]
                    total_corr = df_met[valid_corr_cols].sum(axis=1)

                    for i in range(max_carbons + 1):
                        area_col = f"M+{i} Area (Corrected)"
                        pct_col = f"M+{i} %"

                        if area_col in df_met.columns:
                            df_met[pct_col] = np.where(total_corr > 0, (df_met[area_col] / total_corr) * 100, 0)
                        else:
                            df_met[pct_col] = 0.0

                        m_pct_columns.append(pct_col)

                    df_melted = df_met.melt(
                        id_vars=['Filename', 'Species', 'Sample_Type', 'Treatment', 'Condition'],
                        value_vars=m_pct_columns,
                        var_name="Isotopologue",
                        value_name="Fraction (%)"
                    )

                    df_mid_summary = df_melted.groupby(['Isotopologue', 'Condition'])['Fraction (%)'].agg(
                        mean='mean',
                        std='std',
                        count='count'
                    ).reset_index()

                    # --- Replicate detection counts per isotopologue ---
                    # For each (Isotopologue, Condition), count how many replicates had that M+X area > 0
                    # We need to join back the per-file isotopologue areas
                    iso_area_cols = [f"M+{j} Area (Corrected)" for j in range(max_carbons + 1) if f"M+{j} Area (Corrected)" in df_met.columns]
                    df_iso_detect = df_met[['Filename', 'Condition', 'N_Replicates_Total', 'N_Replicates_Detected'] + iso_area_cols].copy() if 'N_Replicates_Total' in df_met.columns else df_met[['Filename', 'Condition'] + iso_area_cols].copy()

                    detect_records = []
                    for iso_col in iso_area_cols:
                        iso_label = iso_col.replace(' Area (Corrected)', ' %').replace('M+', 'M+')
                        # reconstruct pct col name
                        iso_num = iso_col.split('+')[1].split(' ')[0]
                        pct_col_name = f"M+{iso_num} %"
                        for cond, grp in df_iso_detect.groupby('Condition'):
                            n_total_cond = int(grp['N_Replicates_Total'].iloc[0]) if 'N_Replicates_Total' in grp.columns else len(grp)
                            n_detected = int((grp[iso_col] > 0).sum())
                            detect_records.append({'Isotopologue': pct_col_name, 'Condition': cond, 'n_iso_detected': n_detected, 'n_iso_total': n_total_cond})

                    df_detect = pd.DataFrame(detect_records)
                    df_mid_summary = df_mid_summary.merge(df_detect, on=['Isotopologue', 'Condition'], how='left')
                    df_mid_summary['n_iso_detected'] = df_mid_summary['n_iso_detected'].fillna(0).astype(int)
                    df_mid_summary['n_iso_total'] = df_mid_summary['n_iso_total'].fillna(df_mid_summary['count']).astype(int)

                    # Label: "n=detected/total"
                    df_mid_summary['replicate_label'] = df_mid_summary.apply(
                        lambda r: f"n={r['n_iso_detected']}/{r['n_iso_total']}", axis=1
                    )

                    df_mid_summary['iso_order'] = df_mid_summary['Isotopologue'].str.extract(r'(\d+)').astype(int)
                    df_mid_summary = df_mid_summary.sort_values('iso_order')

                    fig_mid_mean = px.bar(
                        df_mid_summary,
                        x="Isotopologue",
                        y="mean",
                        error_y="std",
                        color="Condition",
                        barmode="group",
                        text=df_mid_summary['replicate_label'],
                        hover_data={
                            'mean': ':.2f',
                            'std': ':.2f',
                            'n_iso_detected': True,
                            'n_iso_total': True,
                            'replicate_label': False
                        },
                        title=f"Corrected MID (Mean ± SD) - {sel_met} ({sel_mode})"
                    )
                    fig_mid_mean.update_traces(
                        textposition='outside',
                        marker_line_color=params.get('sd_line_color', 'red'),
                        marker_line_width=1.5,
                        error_y=dict(
                            color=params.get('errorbar_color', 'white'),
                            thickness=2
                        )
                    )
                    fig_mid_mean.update_layout(
                        yaxis_title="Mean Fractional Contribution (%)",
                        legend_title_text='Experimental Condition'
                    )

                    condition_colors = {trace.name: trace.marker.color for trace in fig_mid_mean.data}

                    # Colour-code bars where no replicate detected that isotopologue.
                    # Plotly uses the first point colour for the legend when a trace
                    # has per-bar colours, so hide those legends and add clean
                    # legend-only condition traces below.
                    for trace in fig_mid_mean.data:
                        base_color = condition_colors.get(trace.name, trace.marker.color)
                        trace.showlegend = False
                        cond_iso_data = df_mid_summary[df_mid_summary['Condition'] == trace.name]
                        marker_colors = []
                        for _, row in cond_iso_data.iterrows():
                            if row['n_iso_detected'] == 0:
                                marker_colors.append('rgba(200,200,200,0.5)')   # grey = none detected
                            elif row['n_iso_detected'] < row['n_iso_total']:
                                marker_colors.append(base_color)   # keep condition colour but annotated
                            else:
                                marker_colors.append(base_color)
                        # Only override if we have actual grey bars (fully undetected)
                        if any(c == 'rgba(200,200,200,0.5)' for c in marker_colors):
                            trace.marker.color = marker_colors

                    for condition, color in condition_colors.items():
                        fig_mid_mean.add_trace(go.Bar(
                            x=[None],
                            y=[None],
                            name=condition,
                            marker_color=color,
                            showlegend=True,
                            hoverinfo='skip'
                        ))

                    st.plotly_chart(fig_mid_mean, use_container_width=True, key="mid_profile_chart_mean")

                    # --- Detection Summary for selected metabolite ---
                    if 'N_Replicates_Detected' in df_met.columns:
                        st.markdown("##### 🔬 Replicate Detection Summary")
                        det_col1, det_col2 = st.columns(2)
                        with det_col1:
                            # Per-condition detection summary
                            det_summary = df_met.groupby('Condition').apply(
                                lambda g: pd.Series({
                                    'N Detected': int((g['Total_Peak_Area'] > 0).sum()),
                                    'N Total': len(g),
                                    '% Detected': f"{(g['Total_Peak_Area'] > 0).mean() * 100:.0f}%"
                                })
                            ).reset_index()
                            st.caption("Compound-level detection across replicates")
                            st.dataframe(det_summary, use_container_width=True, hide_index=True)
                        with det_col2:
                            # Per-isotopologue detection across all replicates
                            iso_detect_rows = []
                            for j in range(max_carbons + 1):
                                col = f"M+{j} Area (Corrected)"
                                if col in df_met.columns:
                                    for cond, grp in df_met.groupby('Condition'):
                                        n_det = int((grp[col] > 0).sum())
                                        n_tot = len(grp)
                                        iso_detect_rows.append({'Isotopologue': f'M+{j}', 'Condition': cond, 'Detected': n_det, 'Total': n_tot})
                            if iso_detect_rows:
                                df_iso_det = pd.DataFrame(iso_detect_rows)
                                df_iso_pivot = df_iso_det.pivot_table(index='Isotopologue', columns='Condition', values='Detected', aggfunc='first')
                                df_iso_pivot = df_iso_pivot.loc[sorted(df_iso_pivot.index, key=lambda x: int(x.split('+')[1]))]
                                st.caption("Isotopologue-level detection (n replicates with area > Minimum Peak Area as set by user)")
                                st.dataframe(df_iso_pivot, use_container_width=True)

                st.divider()
                st.markdown("### 📊 Side‑by‑Side Replicate Peak Areas")
                st.markdown("Select up to 3 replicates to compare their isotopologue peak areas directly.")

                area_type_reps = st.radio(
                    "Area type:",
                    ["Raw", "Corrected"],
                    horizontal=True,
                    key="area_type_reps_radio_v2"
                )

                df_met['Replicate_Label'] = df_met['Filename'] + '  |  ' + df_met['Condition']
                all_rep_labels = df_met['Replicate_Label'].unique().tolist()

                st.caption(f"ℹ️ Found **{len(all_rep_labels)}** replicate(s) for this metabolite & ion mode.")
                if len(all_rep_labels) < 3:
                    st.info("You need at least 3 replicates to select 3. Only those available are shown below.")

                max_rep_cols = 10
                selected_rep_labels = st.multiselect(
                    f"Choose replicates to compare (max {max_rep_cols}):",
                    options=all_rep_labels,
                    default=all_rep_labels[:min(max_rep_cols, len(all_rep_labels))],
                    max_selections=max_rep_cols,
                    key="rep_multiselect_v2"
                    )

                if selected_rep_labels:
                    n_cols = len(selected_rep_labels)
                    cols = st.columns(n_cols)

                    for idx, rep_label in enumerate(selected_rep_labels):
                        df_row = df_met[df_met['Replicate_Label'] == rep_label].iloc[0]
                        max_c = int(df_row['N_Tracer']) if 'N_Tracer' in df_row.index else int(df_row['Total_Carbons'])

                        isotopologues = [f"M+{i}" for i in range(max_c + 1)]
                        if area_type_reps == "Raw":
                            areas = [df_row.get(f"M+{i} Area (Raw)", 0) for i in range(max_c + 1)]
                        else:
                            areas = [df_row.get(f"M+{i} Area (Corrected)", 0) for i in range(max_c + 1)]

                        short_name = df_row['Filename'][-20:] + ' | ' + df_row['Condition']
                        chart_title = f"<b>{short_name}</b>" if len(short_name) > 40 else short_name

                        with cols[idx]:
                            fig_rep = go.Figure(
                                go.Bar(
                                    x=isotopologues,
                                    y=areas,
                                    text=[f"{a:.2e}" for a in areas],
                                    textposition='outside',
                                    marker_color='steelblue',
                                    marker_line_color=params.get('sd_line_color', 'red'),
                                    marker_line_width=1.5
                                )
                            )
                            fig_rep.update_layout(
                                title=chart_title,
                                title_font_size=11,
                                yaxis_title=f"{area_type_reps} Area",
                                height=350,
                                margin=dict(t=40, b=20, l=10, r=10),
                                showlegend=False
                            )
                            st.plotly_chart(fig_rep, use_container_width=True)

                    with st.expander("📋 Show underlying data for selected replicates"):
                        df_selected = df_met[df_met['Replicate_Label'].isin(selected_rep_labels)][
                            ['Replicate_Label'] + [f"M+{i} Area (Raw)" for i in range(max_c+1)] + [f"M+{i} Area (Corrected)" for i in range(max_c+1)]
                        ]
                        st.dataframe(df_selected, use_container_width=True)
                else:
                    st.info("Select at least one replicate to display its isotopologue peak areas.")

                st.divider()
                st.markdown("#### 📊 Summary Statistics")
                st.markdown("Mean enrichment and detection rates grouped by Species,Sample Type, and Metabolite.")

                total_counts = df_filtered.groupby(['Species', 'Sample_Type', 'Metabolite']).size().rename('Total Samples')
                detected_counts = df_filtered[df_filtered['Total_Peak_Area'] > 0].groupby(['Species', 'Sample_Type', 'Metabolite']).size().rename('Detected Samples')
                mean_enrich = df_filtered.groupby(['Species', 'Sample_Type', 'Metabolite'])[enrich_column].mean().rename('Mean Enrichment (%)')
                mean_area = df_filtered.groupby(['Species', 'Sample_Type', 'Metabolite'])['Total_Peak_Area'].mean().rename('Mean Total Area')

                summary_df = pd.concat([total_counts, detected_counts, mean_enrich, mean_area], axis=1).reset_index()
                summary_df['Detected Samples'] = summary_df['Detected Samples'].fillna(0).astype(int)
                summary_df['% Detected'] = (summary_df['Detected Samples'] / summary_df['Total Samples']) * 100

                # --- Merge in N_Replicates_Detected / N_Replicates_Total if available ---
                if 'N_Replicates_Detected' in df_filtered.columns and 'N_Replicates_Total' in df_filtered.columns:
                    rep_cols = df_filtered.groupby(['Species', 'Sample_Type', 'Metabolite'])[['N_Replicates_Detected', 'N_Replicates_Total']].first().reset_index()
                    summary_df = summary_df.merge(rep_cols, on=['Species', 'Sample_Type', 'Metabolite'], how='left')
                    summary_df['Replicates (det/total)'] = (
                        summary_df['N_Replicates_Detected'].fillna(0).astype(int).astype(str) +
                        '/' +
                        summary_df['N_Replicates_Total'].fillna(0).astype(int).astype(str)
                    )
                    col_order = ['Species', 'Sample_Type', 'Metabolite', 'Replicates (det/total)', '% Detected', 'Mean Enrichment (%)', 'Mean Total Area']
                else:
                    col_order = ['Species', 'Sample_Type', 'Metabolite', 'Total Samples', 'Detected Samples', '% Detected', 'Mean Enrichment (%)', 'Mean Total Area']

                summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

                display_label = _get_main_tracer_label(df_filtered)
                dynamic_enrichment_col = f"Mean {display_label} Enrichment (%)"

                display_df = summary_df.rename(columns={'Mean Enrichment (%)': dynamic_enrichment_col})

                st.dataframe(display_df.style.format({
                    '% Detected': '{:.1f}%',
                    dynamic_enrichment_col: '{:.2f}%',
                    'Mean Total Area': '{:,.0f}'
                }), use_container_width=True)

            else:
                st.warning("No data matches the selected Filters (or all failed QC).")
    else:
        st.warning("⚠️ No data available. Upload files or load a session.")

def render_tab_pca():
    st.markdown("### Principal Component Analysis (PCA)")
    st.markdown("Identify which treatments, tissues, or species separate based on their isotopic labeling patterns.")

    if st.session_state.get('results_df') is not None and not st.session_state.results_df.empty and 'Metabolite' in st.session_state.results_df.columns:
        df_pca = st.session_state.results_df.copy()

        if 'QC_Status' not in df_pca.columns:
            st.warning("⚠️ **Old Session Data Detected.** Please re-run extraction so the PCA can utilize the new features.")
        else:
            pca_opt1, pca_opt2 = st.columns(2)
            with pca_opt1:
                hide_qc_fails_pca = st.checkbox("✅ Exclude QC-Failed Metabolites from PCA", value=True)
                hide_zeros_pca = st.checkbox("✅ Exclude Undetected (Zero Area) from PCA", value=True)
            with pca_opt2:
                normalize_pca = st.checkbox("✅ Normalize peak area metrics to Total Pool (Relative Abundance)", value=True, help="If doing PCA on raw or corrected total peak area, divides each metabolite by the total sum of all metabolites in that sample to correct for injection/biomass differences.")

            # ---Include Manually Edited data in PCA ---
            if hide_qc_fails_pca:
                df_pca = df_pca[df_pca['QC_Status'].str.startswith('Pass') | df_pca['QC_Status'].isin(['Manually Edited'])]

            raw_area_cols = [
                col for col in df_pca.columns
                if col.startswith("M+") and col.endswith(" Area (Raw)")
            ]
            if raw_area_cols:
                df_pca['Total_Peak_Area_Raw'] = df_pca[raw_area_cols].sum(axis=1)

            df_pca['Metabolite_Feature'] = df_pca['Metabolite'] + " (" + df_pca['Ion_Mode'] + ")"

            # --- Menu for PCA ---
            if 'Tracer' in df_pca.columns and not df_pca['Tracer'].empty:
                main_tracer = str(df_pca['Tracer'].mode()[0])
            else:
                main_tracer = "13C"

            try:
                display_label = TRACER_LABELS.get(main_tracer, main_tracer)
            except NameError:
                display_label = main_tracer

            # Find the best available tracer-safe enrichment column.
            enrich_col = _preferred_enrichment_col(df_pca)

            # Create dictionary:"What the user sees" -> "What pandas understands"
            metric_options = {
                "Total_Peak_Area (corrected)": "Total_Peak_Area",
                f"Excess {display_label} Enrichment (%)": enrich_col
            }
            if raw_area_cols:
                metric_options = {
                    "Total_Peak_Area (raw)": "Total_Peak_Area_Raw",
                    **metric_options
                }

            pca_col1, pca_col2, pca_col3 = st.columns(3)
            with pca_col1:
                selected_metric_label = st.selectbox("PCA Metric:", list(metric_options.keys()))
                pca_metric = metric_options[selected_metric_label]
            with pca_col2:
                pca_metadata_cols = ['Species', 'Sample_Type', 'Treatment'] + [
                    col for col in st.session_state.get('mapping_extra_columns', [])
                    if (
                        col in df_pca.columns
                        and not str(col).strip().lower().startswith('unnamed:')
                    )
                ]
                pca_metadata_cols = list(dict.fromkeys(pca_metadata_cols))
                color_var = st.selectbox("Color points by:", pca_metadata_cols, index=min(pca_metadata_cols.index("Treatment"), len(pca_metadata_cols) - 1) if "Treatment" in pca_metadata_cols else 0)
            with pca_col3:
                symbol_var = st.selectbox("Symbolize points by:", pca_metadata_cols, index=min(pca_metadata_cols.index("Species"), len(pca_metadata_cols) - 1) if "Species" in pca_metadata_cols else 0)

            if hide_zeros_pca:
                df_pca = df_pca[df_pca[pca_metric].fillna(0) > 0]

            try:
                pca_index_cols = ['Filename', 'Species', 'Sample_Type', 'Treatment']
                pca_index_cols = list(dict.fromkeys(pca_index_cols))
                df_wide = df_pca.pivot_table(
                    index=pca_index_cols,
                    columns='Metabolite_Feature',
                    values=pca_metric,
                    aggfunc='mean'
                ).reset_index()

                # Attach optional metadata after pivoting. Row-level or missing
                # custom metadata must not split one sample into multiple PCA
                # observations or cause the sample to disappear.
                optional_metadata_cols = [
                    col for col in pca_metadata_cols
                    if col not in pca_index_cols
                ]
                inconsistent_metadata = []
                if optional_metadata_cols:
                    grouped_metadata = df_pca.groupby(
                        pca_index_cols, dropna=False
                    )
                    for col in optional_metadata_cols:
                        if grouped_metadata[col].nunique(dropna=True).gt(1).any():
                            inconsistent_metadata.append(col)

                    metadata_per_sample = (
                        grouped_metadata[optional_metadata_cols]
                        .first()
                        .reset_index()
                    )
                    df_wide = df_wide.merge(
                        metadata_per_sample,
                        on=pca_index_cols,
                        how='left',
                        validate='one_to_one',
                    )

                if inconsistent_metadata:
                    st.warning(
                        "Some restored metadata columns have multiple values for "
                        "the same sample. PCA used the first non-empty value and "
                        "kept one point per sample: "
                        f"{inconsistent_metadata}"
                    )

                metadata_cols_in_wide = list(dict.fromkeys(pca_index_cols + optional_metadata_cols))
                metadata = df_wide[metadata_cols_in_wide]
                features = df_wide.drop(columns=metadata_cols_in_wide)

                # Missing-value imputation: use half the column minimum (per-metabolite).
                # This is the standard metabolomics practice for values that are below the
                # detection limit ("missing not at random"). Filling with 0 treats
                # "not detected" as "zero enrichment / zero abundance", which distorts the
                # PCA geometry — a sample with a missing value would appear further from
                # all other samples than one with a tiny-but-real signal.
                col_half_min = features.apply(lambda col: col.dropna().min() / 2.0 if col.dropna().shape[0] > 0 else 0.0)
                features = features.fillna(col_half_min)

                area_metrics = {"Total_Peak_Area_Raw", "Total_Peak_Area"}
                if pca_metric in area_metrics and normalize_pca:
                    row_sums = features.sum(axis=1)
                    features = features.div(row_sums.replace(0, np.nan), axis=0).fillna(0) * 100

                if features.shape[1] < 2:
                    st.warning("⚠️ You need at least 2 passing metabolites to perform a Principal Component Analysis.")
                elif features.shape[0] < 3:
                    st.warning("⚠️ You need at least 3 uploaded samples to perform a Principal Component Analysis.")
                else:
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(features)

                    pca = PCA(n_components=2)
                    components = pca.fit_transform(scaled_features)
                    var_exp = pca.explained_variance_ratio_ * 100

                    df_wide['PC1'] = components[:, 0]
                    df_wide['PC2'] = components[:, 1]


                    y_title = f"PCA Score Plot based on {selected_metric_label}" + (" (Sum Normalized)" if pca_metric in area_metrics and normalize_pca else "")
                    pca_symbol_sequence = [
                        'circle', 'square', 'diamond', 'cross', 'x',
                        'triangle-up', 'triangle-down', 'triangle-left', 'triangle-right',
                        'pentagon', 'hexagon', 'star', 'hourglass', 'bowtie',
                        'asterisk', 'hash', 'y-up', 'y-down'
                    ]
                    n_symbol_groups = df_wide[symbol_var].nunique(dropna=False)
                    if n_symbol_groups > len(pca_symbol_sequence):
                        st.warning(
                            f"⚠️ PCA has {n_symbol_groups} symbol groups but only "
                            f"{len(pca_symbol_sequence)} unique marker shapes, so some symbols will repeat."
                        )

                    fig_pca = px.scatter(
                        df_wide, x='PC1', y='PC2', color=color_var, symbol=symbol_var,
                        hover_data=['Filename'] + pca_metadata_cols,
                        title=y_title,
                        labels={'PC1': f"PC1 ({var_exp[0]:.1f}%)", 'PC2': f"PC2 ({var_exp[1]:.1f}%)"},
                        symbol_sequence=pca_symbol_sequence
                    )

                    fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='DarkSlateGrey')))
                    st.plotly_chart(fig_pca, use_container_width=True, key="pca_score_plot")

                    st.divider()
                    st.markdown("#### 🔍 PCA Loadings (Metabolite Drivers)")

                    loading_threshold = st.slider(
                        "Loading Significance Threshold",
                        min_value=-1.0, max_value=1.0, value=0.15, step=0.05,
                        help="Filter the table to only show metabolites that heavily influence the PCA separation."
                    )

                    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features.columns)
                    filtered_loadings = loadings[(loadings['PC1'].abs() >= loading_threshold) | (loadings['PC2'].abs() >= loading_threshold)]

                    st.write(f"Showing **{len(filtered_loadings)}** driving metabolites with an impact score ≥ {loading_threshold}")

                    if not filtered_loadings.empty:
                        try:
                            styled = filtered_loadings.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1)
                            st.dataframe(styled, use_container_width=True)
                        except Exception as e:
                            st.warning(f"⚠️ Could not apply colour gradient: {e}. Showing plain table.")
                            st.dataframe(filtered_loadings, use_container_width=True)
                    else:
                        st.info("Increase the strictness or lower the threshold to see metabolites.")

            except Exception as e:
                st.error(f"Could not perform PCA. Ensure your data has enough variance: {e}")
    else:
        st.warning("⚠️ No data available. Upload files or load a session.")
