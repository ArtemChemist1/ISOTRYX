import streamlit as st

def render_upload_block(col, title, key_prefix):
    with col:
        st.markdown(f"**{title}**")
        st.markdown("🟢 **Positive**")
        ex_pos = st.file_uploader("Meta (+)", type=['xlsx', 'csv'], key=f"{key_prefix}_pos_ex", label_visibility="collapsed")
        mz_pos = st.file_uploader("mzML (+)", type=['mzML', 'mzml'], accept_multiple_files=True, key=f"{key_prefix}_pos_mz", label_visibility="collapsed")

        st.markdown("🔴 **Negative**")
        ex_neg = st.file_uploader("Meta (-)", type=['xlsx', 'csv'], key=f"{key_prefix}_neg_ex", label_visibility="collapsed")
        mz_neg = st.file_uploader("mzML (-)", type=['mzML', 'mzml'], accept_multiple_files=True, key=f"{key_prefix}_neg_mz", label_visibility="collapsed")

        return ex_pos, mz_pos, ex_neg, mz_neg

def render_custom_dataset_block(species_name, prefix):
    categories = []
    is_qc_block = str(species_name).startswith("QC_")
    default_species = str(species_name)[3:] if is_qc_block else str(species_name)

    with st.expander("Upload your own dataset here", expanded=False):
        n_custom = st.number_input(
            "Number of custom upload groups",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
            key=f"{prefix}_custom_n_groups",
            help="Use one group for each custom sample type and treatment combination."
        )

        for idx in range(int(n_custom)):
            st.markdown(f"#### Custom Dataset {idx + 1}")
            meta_col1, meta_col2, meta_col3 = st.columns(3)
            with meta_col1:
                custom_species = st.text_input(
                    "Species / group or dataset name if uploading a public dataset",
                    value=default_species,
                    placeholder="Type anything, e.g. Arabica, Wheat, Human",
                    key=f"{prefix}_custom_{idx}_species"
                ).strip()
            with meta_col2:
                sample_type = st.text_input(
                    "Sample type",
                    placeholder="Type anything, e.g. Fruit, Media, Soil, Plasma",
                    key=f"{prefix}_custom_{idx}_sample_type"
                ).strip()
            with meta_col3:
                treatment = st.text_input(
                    "Treatment",
                    placeholder="Type anything, e.g. Control, Inoculated, before or post labelling",
                    key=f"{prefix}_custom_{idx}_treatment"
                ).strip()

            label_species = custom_species or default_species or f"Custom Group {idx + 1}"
            label_sample_type = sample_type or f"Custom Sample {idx + 1}"
            label_treatment = treatment or "Custom Treatment"
            output_species = f"QC_{label_species}" if is_qc_block else label_species
            ex_pos, mz_pos, ex_neg, mz_neg = render_upload_block(
                st.container(),
                f"{label_species} - {label_treatment} - {label_sample_type}",
                f"{prefix}_custom_upload_{idx}"
            )

            categories.append({
                "species": output_species,
                "treatment": label_treatment,
                "type": label_sample_type,
                "ex_pos": ex_pos,
                "mz_pos": mz_pos,
                "ex_neg": ex_neg,
                "mz_neg": mz_neg
            })

            if idx < int(n_custom) - 1:
                st.divider()

    return categories
