import re
import pandas as pd
import numpy as np
import streamlit as st
from pyteomics import mzml
from utils.isotope_correction import parse_formula, parse_tracer, DEFAULT_ISODATA
from utils.math_ops import local_minimum_resolver_extract_area


def generate_targets(df, mz_tol, rt_win):
    metabolites = []
    df = df.dropna(subset=['Name', 'm/z', 'RT', 'Formula'])
    skipped_rows = []

    for row_idx, row in df.iterrows():
        try:
            name = str(row['Name'])
            base_mz = float(row['m/z'])
            base_rt = float(row['RT'])

            formula_str = str(row['Formula'])

            # --- Strict Check Isotope ---
            tracer_val = row.get('Tracer')
            if pd.isna(tracer_val) or str(tracer_val).strip() == '':
                st.error(f"🛑 **Mistake in the metadata:** metabolite '{name}' did not provide isotope. Please provide a tracer such as '13C', '15N', '2H', '18O', or '34S'.")
                st.stop()

            tracer_str = str(tracer_val).strip()

            # 1. Get isotope el
            tracer_elem, tracer_idx = parse_tracer(tracer_str, DEFAULT_ISODATA)

            # 2. Dynamically exact mass shift for specific tracer element#
            masses = DEFAULT_ISODATA[tracer_elem]['mass']
            mass_shift = masses[tracer_idx] - masses[0]

            formula = parse_formula(formula_str)

            # Determine number of atoms in formula #
            n_tracer = formula.get(tracer_elem, 0)

            # Legacy fallback: use the Carbons column only for 13C if Formula lacks carbon.
            if n_tracer == 0:
                if tracer_elem == 'C' and 'Carbons' in row and pd.notna(row['Carbons']) and int(row['Carbons']) > 0:
                    n_tracer = int(row['Carbons'])
                    formula[tracer_elem] = n_tracer
                else:
                    st.error(f"🛑 **Metadata error:** The metabolite '{name}' has no '{tracer_elem}'. It is not possible to calculate isotopologues for tracer {tracer_str}.")
                    st.stop()

            isotopes = []
            for i in range(n_tracer + 1):
                target_mz = base_mz + i * mass_shift
                mz_window = target_mz * (mz_tol / 1e6)
                isotopes.append({
                    'isotope_number': i,
                    'target_mz': target_mz,
                    'mz_min': target_mz - mz_window,
                    'mz_max': target_mz + mz_window,
                    'eic_rt': [], 'eic_int': [], 'eic_mz': [],
                    'raw_area': 0.0, 'corrected_area': 0.0
                })

            met = {
                'name': name,
                'target_mz': base_mz,
                'target_rt': base_rt,
                'rt_min': base_rt - rt_win,
                'rt_max': base_rt + rt_win,
                'n_tracer': n_tracer,
                'formula': formula,
                'tracer_str': tracer_str,
                'isotopes': isotopes
            }
            met['carbons'] = n_tracer
            metabolites.append(met)
        except (ValueError, TypeError) as exc:
            skipped_rows.append({
                "Row": int(row_idx) + 2,
                "Metabolite": str(row.get('Name', 'Unknown')),
                "Reason": str(exc)
            })
            continue
    if skipped_rows:
        st.warning("Some metadata rows were skipped because their formula, tracer, m/z, or RT could not be parsed.")
        st.dataframe(pd.DataFrame(skipped_rows), use_container_width=True, hide_index=True)
    return metabolites

def read_metadata(file_obj):
    file_obj.seek(0)
    if file_obj.name.lower().endswith('.csv'):
        try:
            df = pd.read_csv(file_obj)
            if len(df.columns) < 3:
                file_obj.seek(0)
                df = pd.read_csv(file_obj, sep=';')
        except Exception:
            file_obj.seek(0)
            df = pd.read_csv(file_obj, sep=None, engine='python')
    else:
        df = pd.read_excel(file_obj)

    df.columns = df.columns.astype(str).str.strip()

    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ['name', 'metabolite', 'compound']: rename_map[col] = 'Name'
        if col_lower in ['rt', 'r.t.', 'retention time']: rename_map[col] = 'RT'
        if col_lower in ['m/z', 'mz', 'mass']: rename_map[col] = 'm/z'
        if col_lower in ['carbons', 'c', 'num_carbons']: rename_map[col] = 'Carbons'
        if col_lower in ['tracer', 'tracer_element', 'label']: rename_map[col] = 'Tracer'

    df.rename(columns=rename_map, inplace=True)

    # --- Automatically derive Carbons from a Molecular Formula column ---
    for col in df.columns:
        if col.lower() in ['formula', 'molecular formula', 'mf']:
            df.rename(columns={col: 'Formula'}, inplace=True)
            break
    if 'Formula' in df.columns:
        def _parse_carbons(formula_str):
            if pd.isna(formula_str):
                return None
            mol = str(formula_str).replace(" ", "")
            tokens = re.findall(r'([A-Z][a-z]?)(\d*)', mol)
            total_c = 0
            for elem, num in tokens:            # now inside the function
                if elem == 'C':
                    total_c += 1 if num == '' else int(num)
            return total_c if total_c > 0 else None

        formula_carbons = df['Formula'].apply(_parse_carbons)
        if 'Carbons' in df.columns:
            df['Carbons'] = df['Carbons'].fillna(formula_carbons)
        else:
            df['Carbons'] = formula_carbons


    required_cols = ['Name', 'm/z', 'RT', 'Formula', 'Tracer']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        st.error(f"🛑 **Error in Metadata File:** `{file_obj.name}`\n\nThe app could not find these required columns: **{missing_cols}**.\n\nHere are the columns the app *did* find: {list(df.columns)}.\n\nPlease check your file, fix the column headers, and re-upload it.")
        st.stop()

    return df

def extract_blank_max_heights(blank_files, metabolites, params):
    if not blank_files or not metabolites: return {}

    blank_results = {}
    for met in metabolites:
        required_len = met['n_tracer'] + 1
        heights = blank_results.setdefault(met['name'], [])
        if len(heights) < required_len:
            heights.extend([0.0] * (required_len - len(heights)))

    for b_file in blank_files:
        b_file.seek(0)
        with mzml.read(b_file) as reader:
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
                                iso['eic_rt'].append(rt)
                                iso['eic_int'].append(intensity)
                                iso['eic_mz'].append(mz_val)

        for met in metabolites:
            for iso in met['isotopes']:
                if len(iso['eic_rt']) > 0:
                    sort_idx = np.argsort(iso['eic_rt'])
                    rt_arr, int_arr, mz_arr = np.array(iso['eic_rt'])[sort_idx], np.array(iso['eic_int'])[sort_idx], np.array(iso['eic_mz'])[sort_idx]
                    _, height, _, _, _, _, _ = local_minimum_resolver_extract_area(
                        rt_array=rt_arr, int_array=int_arr, mz_array=mz_arr,
                        target_rt=met['target_rt'], params=params
                    )

                    iso_idx = iso['isotope_number']
                    heights = blank_results.setdefault(met['name'], [])
                    if len(heights) <= iso_idx:
                        heights.extend([0.0] * (iso_idx + 1 - len(heights)))

                    if height > heights[iso_idx]:
                        heights[iso_idx] = height

        for met in metabolites:
            for iso in met['isotopes']:
                iso['eic_rt'] = []; iso['eic_int'] = []; iso['eic_mz'] = []

    return blank_results
