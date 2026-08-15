import re
import collections
import numpy as np
from scipy.optimize import nnls

# Import the isotopic data you just created
from utils.isotope_data import DEFAULT_ISODATA


def parse_formula(formula_str: str) -> dict:
    """Return element counts from a chemical formula (e.g., 'C6H12O6').
    Supports simple one-level bracket groups such as C6H5(OH)3.
    Returns a collections.Counter (dict-like)."""
    if formula_str is None:
        raise ValueError("Formula is missing.")

    formula_str = str(formula_str).strip().replace(" ", "")
    if not formula_str:
        raise ValueError("Formula is empty.")

    # Accept common charged/bracketed exports such as [C21H43N2O3]+,
    # [C21H43N2O3]2+, or the occasional mismatched wrapper [C21H43N2O3}+.
    # The isotope correction needs the neutral elemental formula only.
    formula_str = re.sub(r'(?:\d*[+-]+|[+-]+\d*)$', '', formula_str)
    if len(formula_str) >= 2 and formula_str[0] in "[{" and formula_str[-1] in "]}":
        formula_str = formula_str[1:-1]

    if any(ch in formula_str for ch in "+-[].{}"):
        raise ValueError(
            f"Unsupported formula/adduct notation: {formula_str}. "
            "Use a neutral molecular formula such as C6H12O6."
        )

    group_pattern = re.compile(r'\(([A-Za-z0-9]+)\)(\d*)')
    while True:
        match = group_pattern.search(formula_str)
        if not match:
            break
        group, multiplier = match.groups()
        multiplier = int(multiplier) if multiplier else 1
        expanded = group * multiplier
        formula_str = formula_str[:match.start()] + expanded + formula_str[match.end():]

    if "(" in formula_str or ")" in formula_str:
        raise ValueError(f"Unsupported nested formula groups: {formula_str}")

    pformula = re.findall(r'([A-Z][a-z]*)(\d*)', formula_str)
    rebuilt = ''.join(f"{el}{cnt}" for el, cnt in pformula)
    if rebuilt != formula_str:
        raise ValueError(f"Unsupported formula syntax: {formula_str}")

    counter = collections.Counter()
    for element, cnt in pformula:
        counter[element] += 1 if cnt == '' else int(cnt)
    return counter


def parse_tracer(tracer_str: str, data_isotopes: dict) -> tuple:
    """Extract (symbol, isotope_index) from tracer code like '13C' or '15N'."""
    match = re.search(r'(\d+)([A-Z][a-z]*)', tracer_str)
    if not match:
        raise ValueError(f"Invalid tracer code: {tracer_str}")
    nominal_mass = int(match.group(1))
    element = match.group(2)
    if element not in data_isotopes:
        raise ValueError(f"Unsupported tracer element: {element}")

    best_diff = float("inf")
    best_idx = None
    for i, mass in enumerate(data_isotopes[element]["mass"]):
        diff = abs(round(mass) - nominal_mass)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    if best_diff >= 0.5:
        raise ValueError(f"Could not find matching isotope in {element} for nominal mass {nominal_mass}")
    return element, best_idx


def get_nominal_mass_shifts(isotope_data: dict) -> list:
    """Return each isotope's nominal mass shift from the monoisotope."""
    masses = isotope_data["mass"]
    abundances = isotope_data["abundance"]
    if len(masses) != len(abundances):
        raise ValueError("Isotope mass and abundance vectors must have equal lengths.")
    if not masses:
        raise ValueError("Isotope data must contain at least one isotope.")

    shifts = [int(round(float(mass) - float(masses[0]))) for mass in masses]
    if any(shift < 0 for shift in shifts):
        raise ValueError("Isotope masses must be ordered from the monoisotope upward.")
    return shifts


def to_nominal_mass_vector(isotope_data: dict, abundances=None) -> np.ndarray:
    """Map isotope-family abundances onto sparse nominal mass-shift positions."""
    family_abundances = np.asarray(
        isotope_data["abundance"] if abundances is None else abundances,
        dtype=float,
    )
    shifts = get_nominal_mass_shifts(isotope_data)
    if len(family_abundances) != len(shifts):
        raise ValueError("Abundance vector length must match the isotope family.")

    vector = np.zeros(max(shifts) + 1, dtype=float)
    for shift, abundance in zip(shifts, family_abundances):
        vector[shift] += abundance
    return vector


def get_mass_distribution_vector(formula: dict, data_isotopes: dict,
                                 exclude_element: str = None) -> np.ndarray:
    """Convolve the isotopic abundance vectors of all elements except 'exclude_element'.
    Returns a normalised probability vector."""
    result = np.array([1.0])
    for el, n in formula.items():
        if el == exclude_element:
            continue
        if el not in data_isotopes:
            raise ValueError(f"Unsupported element in formula: {el}")
        abund = to_nominal_mass_vector(data_isotopes[el])
        for _ in range(n):
            result = np.convolve(result, abund)
    return result / result.sum()


def build_correction_matrix(formula: dict,
                            tracer_el: str,
                            tracer_idx: int,
                            tracer_purity: list,
                            data_isotopes: dict,
                            correct_NA_tracer: bool = False,
                            n_tracer: int = None) -> np.ndarray:
    """Build the correction matrix C (size = (n_tracer+1) × (n_tracer+1)).
    Each column i corresponds to exactly i labeled tracer atoms."""
    if n_tracer is None:
        n_tracer = formula[tracer_el]
    n_iso = n_tracer + 1

    correction_matrix = np.zeros((n_iso, n_iso))
    formula_for_correction = collections.Counter(formula)
    formula_for_correction[tracer_el] = n_tracer
    correction_vector = get_mass_distribution_vector(formula_for_correction, data_isotopes,
                                                     exclude_element=tracer_el)
    tracer_data = data_isotopes[tracer_el]
    tracer_shift = get_nominal_mass_shifts(tracer_data)[tracer_idx]
    tracer_purity_vector = to_nominal_mass_vector(tracer_data, tracer_purity)
    natural_tracer_vector = to_nominal_mass_vector(tracer_data)
    mask = [n * tracer_shift for n in range(n_iso)]

    for i in range(n_iso):
        column = correction_vector.copy()
        for _ in range(i):
            column = np.convolve(column, tracer_purity_vector)
        if correct_NA_tracer:
            for _ in range(n_tracer - i):
                column = np.convolve(column, natural_tracer_vector)
        # pad if necessary
        needed_len = max(mask) + 1
        if len(column) < needed_len:
            column = np.append(column, [0.0] * (needed_len - len(column)))
        column = [column[j] for j in mask]
        correction_matrix[:, i] = column
    return correction_matrix


def multi_element_correction(measured_areas: list,
                             formula: dict,
                             tracer_str: str,
                             tracer_purity: list = None,
                             correct_NA_tracer: bool = False,
                             data_isotopes: dict = None) -> tuple:
    """Perform full isotopic correction for one metabolite.

    The correction now uses non‑negative least squares (NNLS) to solve
        C · mid = measured_areas   subject to mid >= 0.
    """
    if data_isotopes is None:
        from utils.isotope_data import DEFAULT_ISODATA
        data_isotopes = DEFAULT_ISODATA

    tracer_el, tracer_idx = parse_tracer(tracer_str, data_isotopes)
    if tracer_el not in formula:
        raise ValueError(f"Tracer element '{tracer_el}' is not present in formula {formula}.")

    v_mes = np.array(measured_areas, dtype=float)
    if v_mes.ndim != 1 or len(v_mes) < 1:
        raise ValueError("Measured areas must be a non-empty 1D vector.")
    n_iso = len(v_mes)
    n_tracer = n_iso - 1
    if n_tracer <= 0:
        raise ValueError("Measured areas must include M+0 and at least one labelled isotopologue.")
    total_measured = np.sum(v_mes)
    if total_measured <= 0:
        return [0.0] * n_iso, [np.nan] * n_iso, np.nan
    if formula.get(tracer_el, 0) != n_tracer:
        formula = collections.Counter(formula)
        formula[tracer_el] = n_tracer

    # Tracer purity defaults to natural abundance of that element
    if tracer_purity is None:
        tracer_purity = data_isotopes[tracer_el]["abundance"]
    else:
        if len(tracer_purity) != len(data_isotopes[tracer_el]["mass"]):
            raise ValueError(
                "Tracer purity vector length must match element's isotopic families."
            )
        total = sum(tracer_purity)
        if total <= 0:
            raise ValueError("Tracer purity vector sum is zero.")
        tracer_purity = [v / total for v in tracer_purity]

    # Build correction matrix
    C = build_correction_matrix(formula, tracer_el, tracer_idx,
                                tracer_purity, data_isotopes, correct_NA_tracer,
                                n_tracer=n_tracer)

    # Solve C * mid = v_mes  with mid >= 0  (non‑negative least squares)
    corrected, _ = nnls(C, v_mes)
    corrected = np.maximum(corrected, 0.0)          # numerical safety

    # Normalise total to match the measured total (standard practice)
    corrected_sum = np.sum(corrected)
    if total_measured > 0 and corrected_sum > 0:
        corrected = corrected / np.sum(corrected) * total_measured

    # NNLS can leave tiny positive residues (for example 1e-9 peak area) in
    # components that are mathematically zero. Treat those as exact zero so
    # exports and detection counts do not report phantom corrected peaks.
    zero_tol = max(total_measured * 1e-12, 1e-8)
    corrected[corrected < zero_tol] = 0.0

    # Compute isotopologue fractions and enrichment
    total = np.sum(corrected)
    if total > 1e-12:
        fractions = corrected / total
        enrichment = np.sum(fractions * np.arange(n_iso)) / n_tracer
    else:
        fractions = np.full(n_iso, np.nan)
        enrichment = np.nan

    return corrected.tolist(), fractions.tolist(), enrichment


def build_tracer_purity_vector(tracer_str: str, purity: float,
                               data_isotopes: dict = None) -> list:
    """Return a purity vector matching the tracer element isotope family."""
    if data_isotopes is None:
        data_isotopes = DEFAULT_ISODATA
    tracer_el, tracer_idx = parse_tracer(tracer_str, data_isotopes)
    natural = list(data_isotopes[tracer_el]["abundance"])
    purity = min(max(float(purity), 0.0), 1.0)
    vector = [0.0] * len(natural)
    non_tracer_total = sum(v for i, v in enumerate(natural) if i != tracer_idx)
    for i, abundance in enumerate(natural):
        if i == tracer_idx:
            vector[i] = purity
        elif non_tracer_total > 0:
            vector[i] = (1.0 - purity) * abundance / non_tracer_total
    total = sum(vector)
    return [v / total for v in vector]


def natural_abundance_enrichment(formula: dict, tracer_str: str,
                                 n_tracer: int = None,
                                 data_isotopes: dict = None) -> float:
    """Expected atom fraction for the tracer isotope from natural abundance."""
    if data_isotopes is None:
        data_isotopes = DEFAULT_ISODATA
    tracer_el, tracer_idx = parse_tracer(tracer_str, data_isotopes)
    if n_tracer is None:
        n_tracer = formula.get(tracer_el, 0)
    if n_tracer <= 0:
        return np.nan
    return float(data_isotopes[tracer_el]["abundance"][tracer_idx])


def weighted_mean_enrichment(areas: list, n_tracer: int) -> float:
    """Weighted mean enrichment from an isotopologue area vector."""
    values = np.array(areas, dtype=float)
    total = np.sum(values)
    if total <= 0 or n_tracer <= 0:
        return np.nan
    return float(np.sum(values * np.arange(len(values))) / (n_tracer * total))
