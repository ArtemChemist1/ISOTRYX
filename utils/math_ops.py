import numpy as np
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


def integrate_exact_bounds(rt_array, int_array, rt_bounds, *,
                           baseline_enabled=False, baseline_flank_min=0.05,
                           minimum_flank_scans=3):
    """Integrate an EIC inside exact RT bounds using the original signal.

    When enabled, the local baseline is a straight line between the median
    intensities in equal-width flanks immediately before and after the peak.
    Smoothing is deliberately excluded from this quantitative calculation.
    """
    rt_arr = np.asarray(rt_array, dtype=float)
    int_arr = np.asarray(int_array, dtype=float)
    if rt_arr.shape != int_arr.shape:
        raise ValueError("RT and intensity arrays must have the same shape")
    if rt_arr.ndim != 1:
        raise ValueError("RT and intensity arrays must be one-dimensional")

    finite = np.isfinite(rt_arr) & np.isfinite(int_arr)
    rt_arr = rt_arr[finite]
    int_arr = int_arr[finite]
    if len(rt_arr):
        sort_idx = np.argsort(rt_arr, kind="stable")
        rt_arr = rt_arr[sort_idx]
        int_arr = int_arr[sort_idx]

    rt_start, rt_end = sorted((float(rt_bounds[0]), float(rt_bounds[1])))
    rt_epsilon = 1e-9
    peak_mask = (rt_arr >= rt_start - rt_epsilon) & (rt_arr <= rt_end + rt_epsilon)
    peak_rt = rt_arr[peak_mask]
    peak_raw = int_arr[peak_mask]

    result = {
        "area_unadjusted": 0.0,
        "baseline_area": 0.0,
        "area_baseline_corrected": 0.0,
        "height_unadjusted": 0.0,
        "height_baseline_corrected": 0.0,
        "baseline_start": 0.0,
        "baseline_end": 0.0,
        "baseline_status": "Not applied",
        "peak_mask": peak_mask,
        "peak_rt": peak_rt,
        "peak_raw": peak_raw,
        "peak_corrected": peak_raw.copy(),
        "baseline_values": np.zeros_like(peak_raw, dtype=float),
        "left_flank_scans": 0,
        "right_flank_scans": 0,
    }
    if len(peak_rt) < 4:
        result["baseline_status"] = "Not applied: fewer than 4 scans in integration window"
        return result

    result["area_unadjusted"] = float(trapezoid(y=peak_raw, x=peak_rt))
    result["height_unadjusted"] = float(np.max(peak_raw))

    if not baseline_enabled:
        result["area_baseline_corrected"] = result["area_unadjusted"]
        result["height_baseline_corrected"] = result["height_unadjusted"]
        return result

    flank = max(0.0, float(baseline_flank_min))
    min_scans = max(1, int(minimum_flank_scans))
    left_mask = (
        (rt_arr >= rt_start - flank - rt_epsilon) &
        (rt_arr < rt_start - rt_epsilon)
    )
    right_mask = (
        (rt_arr > rt_end + rt_epsilon) &
        (rt_arr <= rt_end + flank + rt_epsilon)
    )
    left_values = int_arr[left_mask]
    right_values = int_arr[right_mask]
    left_times = rt_arr[left_mask]
    right_times = rt_arr[right_mask]
    result["left_flank_scans"] = int(len(left_values))
    result["right_flank_scans"] = int(len(right_values))

    if len(left_values) < min_scans or len(right_values) < min_scans:
        result["area_baseline_corrected"] = result["area_unadjusted"]
        result["height_baseline_corrected"] = result["height_unadjusted"]
        result["baseline_status"] = (
            f"Not applied: need {min_scans} flank scans per side "
            f"(found {len(left_values)}/{len(right_values)})"
        )
        return result

    left_level = float(np.median(left_values))
    right_level = float(np.median(right_values))
    left_time = float(np.median(left_times))
    right_time = float(np.median(right_times))
    if right_time > left_time:
        slope = (right_level - left_level) / (right_time - left_time)
        baseline_values = left_level + slope * (peak_rt - left_time)
    else:
        baseline_values = np.full_like(peak_rt, left_level, dtype=float)
    baseline_start = float(baseline_values[0])
    baseline_end = float(baseline_values[-1])
    corrected = np.maximum(peak_raw - baseline_values, 0.0)

    result.update({
        "baseline_area": float(trapezoid(y=baseline_values, x=peak_rt)),
        "area_baseline_corrected": float(trapezoid(y=corrected, x=peak_rt)),
        "height_baseline_corrected": float(np.max(corrected)),
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "baseline_status": "Applied",
        "peak_corrected": corrected,
        "baseline_values": baseline_values,
    })
    return result


def fails_blank_fold_change(sample_peak_height, blank_peak_height, fold_change):
    """Return whether a compound fails the shared sample-to-blank height rule."""
    sample_height = max(0.0, float(sample_peak_height))
    blank_height = max(0.0, float(blank_peak_height))
    threshold = max(0.0, float(fold_change))
    return blank_height > 0.0 and sample_height < threshold * blank_height


def local_minimum_resolver_extract_area(rt_array, int_array, mz_array, target_rt, params):
    # 1. Smooth the data based on user selection
    smoothing_method = params.get('smoothing_method', 'Gaussian')

    if smoothing_method == 'Savitzky-Golay':
        window_length = params.get('sg_window', 5)
        polyorder = params.get('sg_polyorder', 2)

        data_len = len(int_array)

        # Safety checks for SG filter requirements
        if data_len < window_length:
            window_length = data_len if data_len % 2 != 0 else data_len - 1
        if window_length % 2 == 0:
            window_length -= 1
        if polyorder >= window_length:
            polyorder = max(1, window_length - 1)

        if window_length > polyorder and window_length >= 3:
            smoothed_ints = savgol_filter(int_array.astype(float), window_length, polyorder)
            # SG can sometimes create negative artifacts at the baseline, clip them to 0
            smoothed_ints = np.maximum(smoothed_ints, 0)
        else:
            # Fallback if array is too small for SG
            smoothed_ints = int_array.astype(float)
    else:
        # Default to Gaussian
        sigma = params.get('sigma_val', 1.0)
        smoothed_ints = gaussian_filter1d(int_array.astype(float), sigma=sigma)

    # 2. Determine noise baseline as a percentile of the smoothed signal
    thr_perc = params.get('chromatographic_threshold', 85)
    noise_baseline = np.percentile(smoothed_ints, thr_perc)

    # Apply Threshold BEFORE Valley Search
    # Set all values below the baseline to 0. This flattens the noise
    # so the sliding window doesn't mistake microscopic bumps for peak boundaries.
    cleaned_ints = np.copy(smoothed_ints)
    cleaned_ints[cleaned_ints < noise_baseline] = 0.0

    # 3. Find all local maxima (peak candidates) in the smoothed data
    peak_indices, _ = find_peaks(smoothed_ints)

    # 4. Keep only maxima above the noise baseline and inside the RT window
    rt_half = params.get('rt_window_min', 0.4)
    valid_idx = [i for i in peak_indices
                 if cleaned_ints[i] > 0
                 and (target_rt - rt_half) <= rt_array[i] <= (target_rt + rt_half)]

    if not valid_idx:
        return 0.0, 0.0, 0, 0, smoothed_ints, 0.0, 0.0

    # --- MZmine Specific Parameters ---
    min_scans = params.get('min_scans', 3)
    min_height = params.get('noise_threshold', 0.0)
    min_area = params.get('min_peak_area', 0.0)

    # --- Time-based Valley Search ---
    valley_search_time_min = params.get('valley_search_time_min', 0.015) # Default 0.015 min

    #Calculate median time between scans for this specific chromatogram
    if len(rt_array) > 1:
        dt_min = np.median(np.diff(rt_array))
        if dt_min <= 0: dt_min = 0.01 # Safety fallback
    else:
        dt_min = 0.01

    # Dynamically convert time window to scan window
    search_window = max(1, int(round(valley_search_time_min / dt_min)))

    min_rt_diff = float('inf')
    best_result = (0.0, 0.0, 0, 0, smoothed_ints, 0.0, 0.0)


    for apex_idx in valid_idx:
        # --- Left boundary (MZmine Sliding Window Logic) ---
        left_boundary = apex_idx
        for i in range(apex_idx - 1, -1, -1):
            window_start = max(0, i - search_window)
            window_end = min(len(smoothed_ints), i + search_window + 1)

            # Stop when valleys or drop down to the noise floor
            if smoothed_ints[i] <= np.min(smoothed_ints[window_start:window_end]):
                #Found the valley,now step 2 scans further out for the tail
                left_boundary = max(0, i - 2)
                break
            left_boundary = i

        # --- Right boundary (MZmine Sliding Window Logic) ---
        right_boundary = apex_idx
        for i in range(apex_idx + 1, len(smoothed_ints)):
            window_start = max(0, i - search_window)
            window_end = min(len(smoothed_ints), i + search_window + 1)

            # If the current point is the absolute lowest in its search window, it's the true valley
            if smoothed_ints[i] <= np.min(smoothed_ints[window_start:window_end]):
                #Found the valley,now step 2 scans further out for the tail
                right_boundary = min(len(smoothed_ints) - 1, i + 2)
                break
            right_boundary = i

        # --- Extract raw peak data within boundaries ---
        raw_int_roi = int_array[left_boundary:right_boundary + 1]
        raw_rt_roi = rt_array[left_boundary:right_boundary + 1]

        # MZmine Duration Requirement: Enforce minimum consecutive scans
        if len(raw_int_roi) < min_scans:
            continue

        # Apply minimum absolute height criteria
        peak_height = np.max(raw_int_roi) if len(raw_int_roi) > 0 else 0.0
        if peak_height < min_height:
            continue

        # Apply minimum area criteria
        area = trapezoid(y=raw_int_roi, x=raw_rt_roi) if len(raw_rt_roi) > 1 else 0.0
        if area < min_area:
            continue

        # ---> Keep the peak that is CLOSEST to the Target RT <---
        rt_diff = abs(rt_array[apex_idx] - target_rt)

        if rt_diff < min_rt_diff:
            min_rt_diff = rt_diff
            best_result = (area, peak_height, left_boundary, right_boundary,
            smoothed_ints, rt_array[apex_idx], mz_array[apex_idx])

    return best_result
