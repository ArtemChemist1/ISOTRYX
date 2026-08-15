import streamlit as st
import pandas as pd
import hashlib
from io import BytesIO

from ui.components import render_custom_dataset_block
from ui.tabs import render_tab_batch, render_tab_qc, render_tab_analysis, render_tab_pca

st.set_page_config(page_title="Isotryx | Full Suite", layout="wide")

st.title("ISOTRYX")
st.markdown("Upload your specific **Positive and Negative Metadata files** alongside the **mzML files** for each **Species**, **Treatment**, and **Tissue**.")

# --- Initialize Session State ---
if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# --- RESTORE SESSION UI ---
st.sidebar.header("💾 Restore Session")
st.sidebar.markdown("Skip extraction by uploading a previously downloaded results CSV.")
uploaded_session = st.sidebar.file_uploader("Upload previous results (.csv)", type=['csv'])

if uploaded_session is not None:
    _session_bytes = uploaded_session.getvalue()
    _session_fingerprint = hashlib.sha256(_session_bytes).hexdigest()

    # Only import a given file once. Streamlit reruns the script for every
    # widget interaction; re-importing here would overwrite manual edits and
    # other in-memory corrections made after restoration.
    if st.session_state.get('restored_session_fingerprint') != _session_fingerprint:
        try:
            _restored = pd.read_csv(BytesIO(_session_bytes))
            _restored.columns = [str(col).strip() for col in _restored.columns]

            # Accept the legacy filename column used by older exports.
            if 'Filename' not in _restored.columns and 'File' in _restored.columns:
                _restored = _restored.rename(columns={'File': 'Filename'})

            if _restored.columns.duplicated().any():
                _duplicates = _restored.columns[_restored.columns.duplicated()].tolist()
                raise ValueError(f"Duplicate columns found: {_duplicates}")

            _required_cols = {
                'Filename', 'Species', 'Sample_Type', 'Treatment', 'Ion_Mode',
                'Metabolite', 'QC_Status', 'Total_Peak_Area',
            }
            _missing_cols = sorted(_required_cols - set(_restored.columns))
            if _missing_cols:
                raise ValueError(
                    "This is not a compatible ISOTRACKER results session. "
                    f"Missing required columns: {_missing_cols}"
                )

            # CSV files previously saved with index=True contain synthetic columns
            # such as "Unnamed: 0". They are row identifiers rather than sample
            # metadata and would otherwise split each metabolite into its own PCA
            # observation after session restore.
            _synthetic_index_cols = [
                col for col in _restored.columns
                if str(col).strip().lower().startswith("unnamed:")
            ]
            if _synthetic_index_cols:
                _restored = _restored.drop(columns=_synthetic_index_cols)

            # Retired manual blank-area subtraction fields are discarded on
            # restore. Blank handling now uses fold-change QC only.
            _retired_blank_cols = [
                col for col in _restored.columns
                if col == 'Blank_Area_Subtraction_Applied'
                or (
                    str(col).startswith('M+')
                    and any(marker in str(col) for marker in (
                        'Blank Area (Median)', 'Area (Net Signed)', 'Area (Net)',
                    ))
                )
            ]
            if _retired_blank_cols:
                _restored = _restored.drop(columns=_retired_blank_cols)

            _text_cols = [
                'Filename', 'Species', 'Sample_Type', 'Treatment', 'Ion_Mode',
                'Metabolite', 'QC_Status', 'Tracer', 'Tracer_Element',
                'Manual_Edit_Type', 'Manual_Algorithm', 'Manual_Isotopologues',
                'Manual_Edit_Timestamp', 'Manual_Edit_Log',
                'Manual_Correction_Details',
            ]
            for _col in _text_cols:
                if _col in _restored.columns:
                    _restored[_col] = _restored[_col].astype('string').str.strip()

            _empty_required = [
                col for col in ['Filename', 'Species', 'Sample_Type', 'Treatment', 'Ion_Mode', 'Metabolite', 'QC_Status']
                if _restored[col].isna().any() or _restored[col].eq('').any()
            ]
            if _empty_required:
                raise ValueError(
                    "Required label columns contain empty values: "
                    f"{_empty_required}"
                )

            _numeric_cols = [
                col for col in _restored.columns
                if (
                    col in {
                        'Target_m/z', 'Apex_m/z', 'Target_RT', 'Apex_RT',
                        'Total_Carbons', 'N_Tracer', 'Raw_ME_%',
                        'Natural_Abundance_ME_%', 'Excess_Enrichment_%',
                        'Total_Peak_Area', 'Raw_Enrichment_%',
                        'Excess_13C_Enrichment_%', 'N_Replicates_Total',
                        'N_Replicates_Detected',
                        'Manual_RT_Start', 'Manual_RT_End', 'Manual_Smoothing',
                        'Baseline_Flank_min', 'Baseline_Min_Scans', 'N_Blanks_Used',
                    }
                    or str(col).startswith('M+')
                )
            ]
            _invalid_numeric = {}
            for _col in _numeric_cols:
                _original = _restored[_col]
                _converted = pd.to_numeric(_original, errors='coerce')
                _invalid_mask = (
                    _original.notna()
                    & _original.astype(str).str.strip().ne('')
                    & _converted.isna()
                )
                if _invalid_mask.any():
                    _invalid_numeric[_col] = int(_invalid_mask.sum())
                _restored[_col] = _converted
            if _invalid_numeric:
                raise ValueError(
                    "Non-numeric values were found in numeric result columns: "
                    f"{_invalid_numeric}"
                )

            _bool_cols = [
                'Local_Baseline_Applied',
            ]
            _bool_values = {
                'true': True, '1': True, 'yes': True,
                'false': False, '0': False, 'no': False,
            }
            for _col in _bool_cols:
                if _col not in _restored.columns:
                    continue
                _normalised = _restored[_col].astype('string').str.strip().str.lower()
                _invalid_bool = (
                    _restored[_col].notna()
                    & _normalised.ne('')
                    & ~_normalised.isin(_bool_values)
                )
                if _invalid_bool.any():
                    raise ValueError(
                        f"Invalid boolean values were found in {_col}: "
                        f"{int(_invalid_bool.sum())} row(s)"
                    )
                _restored[_col] = _normalised.map(_bool_values).fillna(False).astype(bool)

            _standard_result_cols = {
                'Filename', 'File', 'Species', 'Ion_Mode', 'Treatment', 'Sample_Type',
                'Metabolite', 'QC_Status', 'Target_m/z', 'Apex_m/z', 'Target_RT', 'Apex_RT',
                'Total_Carbons', 'Tracer', 'Tracer_Element', 'N_Tracer', 'Raw_ME_%',
                'Natural_Abundance_ME_%', 'Excess_Enrichment_%', 'Total_Peak_Area',
                'Raw_Enrichment_%', 'Excess_13C_Enrichment_%',
                'N_Replicates_Total', 'N_Replicates_Detected',
                'Manual_Edit_Type', 'Manual_RT_Start', 'Manual_RT_End',
                'Manual_Algorithm', 'Manual_Smoothing', 'Manual_Isotopologues',
                'Manual_Edit_Timestamp', 'Manual_Edit_Log',
                'Local_Baseline_Applied', 'Baseline_Flank_min',
                'Baseline_Min_Scans', 'N_Blanks_Used',
                'Manual_Correction_Details',
            }
            _mapping_extra_columns = [
                col for col in _restored.columns
                if (
                    col not in _standard_result_cols
                    and not str(col).startswith("M+")
                    and not str(col).strip().lower().startswith("unnamed:")
                )
            ]

            # Commit the restore atomically only after every validation passes.
            st.session_state.results_df = _restored
            st.session_state.mapping_extra_columns = _mapping_extra_columns
            st.session_state.restored_session_fingerprint = _session_fingerprint
            st.session_state.restored_session_name = uploaded_session.name
            st.session_state.pop('restore_session_error', None)
        except Exception as e:
            st.session_state.results_df = None
            st.session_state.mapping_extra_columns = []
            st.session_state.pop('restored_session_fingerprint', None)
            st.session_state.restore_session_error = str(e)

    if st.session_state.get('restored_session_fingerprint') == _session_fingerprint:
        st.sidebar.success(
            f"✅ Session restored from {st.session_state.get('restored_session_name', uploaded_session.name)}! "
            "You can skip to Tabs 3 & 4."
        )
    elif st.session_state.get('restore_session_error'):
        st.sidebar.error(
            f"Could not load session: {st.session_state.restore_session_error}"
        )
else:
    # Removing the uploader file arms Restore Session for an explicit reload of
    # the same CSV, while leaving the already restored results available.
    st.session_state.pop('restored_session_fingerprint', None)
    st.session_state.pop('restored_session_name', None)
    st.session_state.pop('restore_session_error', None)

# --- Sidebar Configuration ---
st.sidebar.header("1. Extraction Parameters")
mz_tol_ppm = st.sidebar.number_input("m/z tolerance (ppm)", value=5.0, step=1.0)
rt_window_min = st.sidebar.number_input("Target RT Window (+/- minutes)", value=0.4, step=0.1)

st.sidebar.header("2. Peak Detection & Noise")
sigma_val = st.sidebar.number_input("Gaussian Filter Sigma (σ)", value=1.2, step=0.1)
noise_threshold = st.sidebar.number_input(
    "Minimum Peak Height (all isotopologues)",
    value=1.0E4,
    format="%e",
    help="Minimum height required for each isotopologue peak. "
         "If an isotopologue's height is below this, its area is set to 0. "
         "M+0 is not required if the other isotopologues pass."
)
min_peak_area = st.sidebar.number_input(
    "Minimum Peak Area (all isotopologues)",
    value=1.0E5,
    format="%e",
    help="Minimum area required for each isotopologue peak. "
         "If an isotopologue's area is below this, its area is set to 0. "
         "M+0 is not required if the other isotopologues pass."
)
min_scans = st.sidebar.number_input("Minimum Consecutive Scans", value=4, step=1)


st.sidebar.header("3. Isotope Tracer Settings")

tracer_purity_pct = st.sidebar.number_input(
    "Tracer Purity (%)", value=99.0, step=0.1, format="%.1f",
    help="Isotopic purity of the purchased tracer (e.g., 99% 13C)."
)
correct_NA_tracer = st.sidebar.checkbox(
    "Correct tracer natural abundance",
    value=True,
    help="If checked, the correction will also remove the contribution of the tracer element's own natural abundance. Usually checked."
)

st.sidebar.header("4. Blank filtering")
blank_fold_change = st.sidebar.number_input("Blank Fold Change Threshold", value=3.0, step=0.5, help="Sample peak height must be this many times greater than the Maximum Blank Height to be kept (e.g., 3.0 = 300%).")

st.sidebar.header("5. Chart Colours & SD Warning")
sd_threshold = st.sidebar.number_input("SD Warning Threshold", value=20.0, step=0.1, help="Horizontal line at this Standard Deviation value on the SD scatter plot.")
sd_line_color = st.sidebar.color_picker(
    "Line Colour",
    value="#FF0000",
    help="Colour of the SD warning line and bar outlines on both Mean ± SD charts."
)
errorbar_color = st.sidebar.color_picker(
    "Error Bar Colour (Mean ± SD charts)",
    value="#FFFFFF",
    help="Applied to both the global and MID Mean ± SD charts."
)

st.sidebar.header("6. Data Export")
drop_zeros = st.sidebar.checkbox("Drop Undetected Metabolites (0 Area)", value=True, help="If a metabolite falls below your minimum Area or Height threshold (resulting in 0 area), it will be completely removed from the exported CSV.")

st.sidebar.header("7. Retention Time Alignment")
enable_rt_alignment = st.sidebar.checkbox(
    "✅ Enable RT Alignment using QC samples",
    value=True,
    help="Recommended for batch experiments with RT drift. Uses QC samples as anchors to compute median apex RT for each metabolite."
)

# Batch Mode
st.sidebar.header("8. Batch Mode")
batch_mode = st.sidebar.checkbox("Enable Batch Mode (single upload)", value=False)

# Create dictionary array for all parameters to pass into modules
params = {
    'mz_tol_ppm': mz_tol_ppm,
    'rt_window_min': rt_window_min,
    'sigma_val': sigma_val,
    'noise_threshold': noise_threshold,
    'min_peak_area': min_peak_area,
    'min_scans': min_scans,
    'chromatographic_threshold': 85.0,   # percentile
    'tracer_purity': tracer_purity_pct / 100.0,
    'correct_NA_tracer': correct_NA_tracer,
    'blank_fold_change': blank_fold_change,
    'drop_zeros': drop_zeros,
    'sd_threshold': sd_threshold,
    'sd_line_color': sd_line_color,
    'errorbar_color': errorbar_color
}
# --- Local Minimum Resolver settings ---
st.sidebar.markdown("---")
st.sidebar.markdown("**🔧 Local Minimum Resolver Settings**")

params['smoothing_method'] = st.sidebar.radio(
    "Smoothing Method",
    options=["Gaussian", "Savitzky-Golay"],
    index=0
)

if params['smoothing_method'] == 'Savitzky-Golay':
    params['sg_window'] = st.sidebar.number_input(
        "SG Window Length (must be odd)",
        min_value=3, max_value=51, value=5, step=2,
        help="Number of data points used to calculate the polynomial. Larger = smoother."
    )
    params['sg_polyorder'] = st.sidebar.number_input(
        "SG Polynomial Order",
        min_value=1, max_value=5, value=2, step=1,
        help="Degree of the fitting polynomial. Usually 2 or 3."
    )

params['chromatographic_threshold'] = st.sidebar.slider(
    "Chromatographic threshold (%)",
    min_value=0.0, max_value=100.0, value=0.0, step=1.0,
    help="Percentile of EIC intensities used as noise baseline. Lower = more sensitive."
)
params['valley_search_time_min'] = st.sidebar.number_input(
    "Valley Search Range (min)",
    value=0.08,
    step=0.01,
    format="%.4f",
    help="Minimum retention time range to search for the valley between peaks. Replaces the hardcoded scan window."
)

# --- UI for Blank Subtraction ---
st.subheader("🔬 Global Reference Controls (Optional)")
st.markdown("Upload Solvent Blanks or 12C Matrix Controls. The app will find the **Maximum Height** in these files. Sample peaks must be X times higher (set in sidebar) to pass.")
blank_c1, blank_c2 = st.columns(2)
with blank_c1:
    _blank_pos_raw = st.file_uploader("Blank mzML (+)", type=['mzML', 'mzml'], accept_multiple_files=True, key="blank_pos")
with blank_c2:
    _blank_neg_raw = st.file_uploader("Blank mzML (-)", type=['mzML', 'mzml'], accept_multiple_files=True, key="blank_neg")

# Streamlit UploadedFile objects are already seekable in-memory streams.
def _reuse_uploads(files):
    result = list(files or [])
    for file_obj in result:
        file_obj.seek(0)
    return result

blank_pos = _reuse_uploads(_blank_pos_raw)
blank_neg = _reuse_uploads(_blank_neg_raw)

st.divider()

# ============================================================
#  CONDITIONAL UI – BATCH MODE vs. PER‑SPECIES TABS
# ============================================================
if batch_mode:
    st.subheader("📦 Batch Upload")
    st.markdown(
        "Upload one **mapping CSV**, **all mzML files**, and **all metadata files**. "
        "The mapping table must contain the columns: "
        "`mzML_filename, Species, Treatment, Sample_Type, Ion_Mode, Metadata_filename`."
    )

    mapping_file = st.file_uploader(
        "Upload Sample Mapping CSV",
        type=['csv'],
        key="batch_mapping",
        help="CSV with columns: mzML_filename, Species, Treatment, Sample_Type, Ion_Mode, Metadata_filename"
    )

    all_mzml_files = st.file_uploader(
        "Upload all mzML files (Positive & Negative)",
        type=['mzML', 'mzml'],
        accept_multiple_files=True,
        key="batch_mzml_files"
    )

    all_metadata_files = st.file_uploader(
        "Upload all Metadata files (Excel or CSV)",
        type=['xlsx', 'csv'],
        accept_multiple_files=True,
        key="batch_metadata_files"
    )

    if mapping_file and all_mzml_files and all_metadata_files:
        try:
            mapping_df = pd.read_csv(mapping_file)
        except Exception as e:
            st.error(f"Error reading mapping CSV: {e}")
            st.stop()

        required_map_cols = [
            'mzML_filename', 'Species', 'Treatment',
            'Sample_Type', 'Ion_Mode', 'Metadata_filename'
        ]
        if not all(col in mapping_df.columns for col in required_map_cols):
            st.error(f"Mapping CSV must contain columns: {required_map_cols}\n\nYour file contains: {list(mapping_df.columns)}")
            st.stop()

        extra_map_cols = [col for col in mapping_df.columns if col not in required_map_cols]
        st.session_state.mapping_extra_columns = extra_map_cols
        st.session_state.mapping_extra_metadata = {}

        # Reuse Streamlit's existing seekable upload buffers. Downstream readers
        # rewind every object before parsing, so no full-file copy is needed.
        def _reuse_upload(uploaded_file):
            uploaded_file.seek(0)
            return uploaded_file

        mzml_dict  = {f.name: _reuse_upload(f) for f in all_mzml_files}
        meta_dict  = {f.name: _reuse_upload(f) for f in all_metadata_files}

        processing_tasks = []
        for _, row in mapping_df.iterrows():
            mz_fname   = str(row['mzML_filename']).strip()
            meta_fname = str(row['Metadata_filename']).strip()

            if mz_fname not in mzml_dict:
                st.error(f"mzML file '{mz_fname}' not found in uploaded files.")
                st.stop()
            if meta_fname not in meta_dict:
                st.error(f"Metadata file '{meta_fname}' not found in uploaded files.")
                st.stop()

            mz_file   = mzml_dict[mz_fname]
            meta_file = meta_dict[meta_fname]
            st.session_state.mapping_extra_metadata[mz_file.name] = {
                col: row[col] for col in extra_map_cols
            }

            processing_tasks.append(
                (mz_file, row['Species'], row['Sample_Type'], row['Treatment'], meta_file, row['Ion_Mode'])
            )

        total_files = len(processing_tasks)
        st.success(f"✅ Batch ready: **{total_files}** files loaded from mapping.")

        if total_files > 0 or (st.session_state.results_df is not None and not st.session_state.results_df.empty):
            tab1, tab2, tab3, tab4 = st.tabs([
                f"📊 Batch Processing ({total_files} files)",
                "🔎 QC & Manual Fix",
                "📈 Data Analysis & Charts",
                "🧠 Multivariate Stats (PCA)"
            ])
            with tab1:
                render_tab_batch(total_files, processing_tasks, blank_pos, blank_neg, params, enable_rt_alignment)
            with tab2:
                render_tab_qc(total_files, processing_tasks, params, blank_pos, blank_neg)
            with tab3:
                render_tab_analysis(params)
            with tab4:
                render_tab_pca()

    else:
        st.info("Please upload the mapping CSV, all mzML files, and all metadata files to proceed.")

else:
    categories_data = render_custom_dataset_block("", "single_upload")

    # UploadedFile is already a seekable in-memory stream. Reusing it avoids a
    # second complete copy of every large mzML file.
    def _wrap(f):
        if f is None:
            return None
        f.seek(0)
        return f

    processing_tasks = []
    for cat in categories_data:
        if cat["mz_pos"]:
            if not cat["ex_pos"]:
                st.error(f"Missing Metadata! You uploaded **Positive mzML files** for {cat['species']} > {cat['treatment']} {cat['type']}, but did not provide the matching Positive Metadata file.")
                st.stop()
            ex_pos_buf = _wrap(cat["ex_pos"])
            for mz_file in cat["mz_pos"]:
                processing_tasks.append((_wrap(mz_file), cat["species"], cat["type"], cat["treatment"], ex_pos_buf, "Positive"))
        if cat["mz_neg"]:
            if not cat["ex_neg"]:
                st.error(f"Missing Metadata! You uploaded **Negative mzML files** for {cat['species']} > {cat['treatment']} {cat['type']}, but did not provide the matching Negative Metadata file.")
                st.stop()
            ex_neg_buf = _wrap(cat["ex_neg"])
            for mz_file in cat["mz_neg"]:
                processing_tasks.append((_wrap(mz_file), cat["species"], cat["type"], cat["treatment"], ex_neg_buf, "Negative"))

    total_files = len(processing_tasks)

    if total_files > 0 or (st.session_state.results_df is not None and not st.session_state.results_df.empty):
        tab1, tab2, tab3, tab4 = st.tabs([
            f"📊 Batch Processing ({total_files} files)",
            "🔎 QC & Manual Fix",
            "📈 Data Analysis & Charts",
            "🧠 Multivariate Stats (PCA)"
        ])
        with tab1:
            render_tab_batch(total_files, processing_tasks, blank_pos, blank_neg, params, enable_rt_alignment)
        with tab2:
            render_tab_qc(total_files, processing_tasks, params, blank_pos, blank_neg)
        with tab3:
            render_tab_analysis(params)
        with tab4:
            render_tab_pca()
