<img width="1491" height="1055" alt="END TO END PIPELINE" src="https://github.com/user-attachments/assets/c745b5a7-be06-4fb2-8bb2-f0791e8fa47b" />



# ISOTRYX

A Streamlit-based web application for stable isotope tracer analysis from LC-MS data (mzML format). ISOTRYX automates isotopologue extraction, isotope correction, and tracer-purity correction, inspecting chromatographic peak quality, and generating summary plots for isotope-labeling experiments.

---

## Features

- **Multi-species / multi-tissue upload** — structured upload interface with support for fully custom dataset groups
- **Batch mode** — process large experiments via a CSV mapping file linking mzML files to metadata
- Extracts M+0 to M+n isotopologue traces from mzML files.
- Supports ¹³C, ¹⁵N, ²H, ¹⁸O, ³⁴S tracer metadata.
- Uses exact isotope mass shifts and ppm extraction windows for high-resolution mass spectrometry data.
- **Peak detection algorithm**:
  - Local Minimum Resolver (LMR) — an independent implementation based on the sliding-window local-minimum search concept from MZmine, with boundaries locked from a composite (summed) isotopologue peak and applied uniformly across all isotopologues.
- **Isotope correction** — IsoCor-based correction matrix built per metabolite, solved via non-negative least squares (NNLS)
- **Tracer purity correction** — adjustable isotopic purity of the purchased tracer
- **Blank filtering** — fold-change threshold filtering against process blanks
- **RT alignment** — QC-anchor-based retention time drift correction across batches
- **QC & manual fix tab** — inspect and override individual peak detections
- **Data analysis tab** — mean ± SD bar charts, isotopologue fraction plots, enrichment scatter
- **PCA tab** — multivariate statistics on corrected isotopologue data
- **Session restore** — export results as CSV and re-import to skip re-extraction

---

## Project Structure

```
isotryx/
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── ui/
│   ├── __init__.py
│   ├── components.py       # Upload blocks and species/tissue UI components
│   └── tabs.py             # Tab renderers (batch, QC, analysis, PCA)
├── utils/
│   ├── __init__.py
│   ├── data_ops.py          # mzML reading, target generation, blank extraction
│   ├── isotope_correction.py # IsoCor-based correction matrix & NNLS solver
│   ├── isotope_data.py      # IUPAC isotopic abundance/mass data
│   └── math_ops.py          # Peak detection algorithm & integration
├── app.py                  # Streamlit application entry point
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── LICENSE                 # MIT license
```

---

## Running the App

If you run this app on Windows please follow these steps:

•	Download the latest release from GitHub page

•	Locate in this folder path /ISOTRYX_Windows_Portable_v1/ and double click run.exe and then it will appear in your browser

If you run this app on Mac based on M or Intel chip please follow these steps:

Since macOS version of app isn't signed with an Apple Developer certificate, macOS Gatekeeper will block it on first launch with a message like "ISOTRYX" cannot be opened because it is from an unidentified developer or "ISOTRYX" is damaged and can't be opened. This is expected — here's how to open it anyway:

•	Download the latest release from GitHub page

• Try opening the app once locating in this folder path for M chip: /ISOTRYX_macOS_build_kit_M_CHIP/dist/ and this folder path for Intel chip: /ISOTRYX_macOS_build_kit_intel/dist/ and double click ISOTRYX.app  — you'll get a warning and the option to click Done or Cancel.

• Go to System Settings → Privacy & Security.

• Scroll down to the Security section. You should see a message: "ISOTRYX" was blocked to protect your Mac.

• Click Open Anyway.

• Confirm again in the popup by clicking Open. You may be asked to enter your Mac password and then it will appear in your browser.

Alternatively, if you’re familiar with the command line, you can run the app via Terminal on a Mac by following these steps:

• Download ISOTRYX_macOS_build_kit from GitHub page

• Follow steps as described in README_BUILD_ON_MAC.md

The dataset used in the video tutorial was analyzed on a Mac with an Apple M-series chip. The dataset and video tutorial are available on Zenodo https://zenodo.org/records/21628589



## Metadata File Format

Upload an `.xlsx` or `.csv` file with the following columns:

| Column | Description |
|--------|-------------|
| `Name` | Metabolite name (also accepted: `Metabolite`, `Compound`) |
| `m/z` | Precursor m/z of the M+0 isotopologue (also accepted: `mz`, `Mass`) |
| `RT` | Expected retention time in **minutes** (also accepted: `R.T.`, `Retention Time`) |
| `Formula` | Neutral molecular formula, e.g. `C6H12O6` |
| `Tracer` | Tracer isotope code, e.g. `13C`, `15N`, `2H`, `18O`, `34S` |

Column names are case-insensitive. Formula carbons are parsed automatically if `Formula` is provided.

## Supported Tracers

| Code | Element |
|------|---------|
| `13C` | Carbon-13 |
| `15N` | Nitrogen-15 |
| `2H` | Deuterium |
| `18O` | Oxygen-18 |
| `34S` | Sulfur-34 |

---

## Batch Mode — Mapping CSV Format

When **Batch Mode** is enabled, upload a mapping CSV with these columns:

| Column | Description |
|--------|-------------|
| `mzML_filename` | Exact filename of the mzML file |
| `Species` | Species or group label |
| `Treatment` | Treatment label (e.g. `Control`, `Inoculated`) |
| `Sample_Type` | Tissue or sample type (e.g. `Leaf`, `Plasma`) |
| `Ion_Mode` | `Positive` or `Negative` |
| `Metadata_filename` | Exact filename of the matching metadata file |

---
Feel free to read our user guide in the main branch of this GitHub repository.

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI |
| `pandas` | Data handling |
| `numpy` | Numerical operations |
| `scipy` | Signal processing, integration, NNLS |
| `pyteomics` | mzML file parsing |
| `plotly` | Interactive charts |
| `scikit-learn` | PCA |
| `openpyxl` | Excel metadata reading |

---

## License

MIT License — see `LICENSE` for details.
