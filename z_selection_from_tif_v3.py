#!/usr/bin/env python3
"""
z_selection_from_tif.py
version 3.0.0

Welcome to Laila's Z-selection python tool for Fiji!! :)

This script is designed to produce an optimal max-projection of images taken
on the Zeiss 880 LSM of Drosophila embryos expressing Sas4 FRET sensors. During imaging, embryos can shift downwards, meaning that slices
that begin in-focus end up introducing lots of noise to the image as time progresses, making it challenging to decide which 
z-slices should be max-projected before further fluorescent intensity analysis in TrackMate to optimise the signal-to-noise ratio.

It is recommended that the YFP channel is used as the input, as YFP tends to have the greatest autofluorescence, so the code does 
not 'miss' any noise which could be problematic in another channel.

This code works by computing an 'in-focus score' for every slice (z) at each time point (t), based on a sharpness metric and 
the difference between average and maximal pixel intensities. From here, it outputs a max-projected .tif, a sum-projected .tif and
a diagnostics_summary file listing which slice had the highest in-focus score at each time point. After using this code, proceed with
tracking and masking as usual using TrackMate in Fiji.

How to use this code: Simply change the paths of tif_path and roi_roi_path below, under 'User Parameters'. To copy a file path on
a macbook, use the shortcut 'option + command + c'. On windows, you can use 'control + shift + c'.

Input: multi-page TIFF of YFP channel, and a .roi file.
Output: optimal max-projected .tif ; optimal sum-projected .tif and a diagnostics_summary file listing slice used at each time
        point.

Laila Shafiee, Raff Lab ILESLA Rotation, Spring 2026
For questions, comments and concerns, please feel free to email me: laila.shafiee@sjc.ox.ac.uk
"""

#################### BOOTING UP ####################
# Loading necessary libraries
import sys
from pathlib import Path
import numpy as np
import tifffile as tiff
import pandas as pd
from scipy.ndimage import median_filter
from skimage.filters import laplace
from skimage import img_as_float32
print("Welcome to the Z-selection code. Let's get going.")
print()
print("All libraries loaded.")
print()

import scipy
import skimage
print(scipy.__version__)
print(skimage.__version__)

for lib in [np, tiff, pd]:
    print(lib.__name__, lib.__version__)

# ROI reader
try:
    from read_roi import read_roi_file
    _HAS_READ_ROI = True
    print("ROI file ok.")
except Exception:
    _HAS_READ_ROI = False


#################### USER PARAMETERS ####################
# Path for YFP channel .tif
tif_path = '/Users/lailashafiee/Library/CloudStorage/OneDrive-Nexus365/Raff Rotation/New Analysis Pipeline/YFP_32bit.tif' # <-- change

# Path for ROI file saved as .roi
roi_roi_path = '/Users/lailashafiee/Library/CloudStorage/OneDrive-Nexus365/Raff Rotation/New Analysis Pipeline/Pilot_ROI.roi'   # <-- alternatively, path to ImageJ .roi file (single ROI). Set to None if using mask TIFF.

# Define output destination for files
from pathlib import Path
input_name = Path(tif_path).stem 
output_folder = f"{input_name}_z_selection_code_output"

# Focus scoring weights
top_percent = 10.0  # defines 'bright tail': top X% of pixel intensities used for bright tail mean
beta = 0.5         # scales contribution of max vs mean intensity to overal focus score
alpha = 0.5       # scales contribution of Laplacian variance (sharpness) to focus score

# Temporal Smoothing
median_window = 3   # takes median between X (3) time slices of in-focus Z to prevent jumps from scoring noise
max_jump = 3        # maximum allowed Z-shift between consecutive frames

# Projection and output
half_width = 2      # ±half_width for projection
dtype_out = np.float32 # Converts data type to float32 to preserve precision for downstream quantification

# Make an output folder
out_dir = Path(output_folder)
out_dir.mkdir(parents=True, exist_ok=True)
print("Output directory made.")

#################### HELPER UTILITIES ####################
def load_tif(path):
    img = tiff.imread(path)
    # ensure float for computations but keep original dtype separately if needed
    return img

def guess_axes(arr):
    """
    Infers and standardises axis order of a raw NumPy array loaded from a Fiji TIFF.
    There is no universal axis convention, so this function uses heuristics based on
    the assumption that T >> Z (145 timepoints vs 9 Z-slices in our data).
    
    All cases return a 4D array in (Z, T, Y, X) order.
    
    Known limitation: the Z/T heuristic threshold (Z_MAX_EXPECTED) will misclassify
    axes if Z-slice count ever exceeds this value — update it if acquisition changes. 
    For now, should be fine, but could fix downstream.
    """

    Z_MAX_EXPECTED = 20  # adjust if more Z-slices than this acquired

    s = arr.shape

    if arr.ndim == 2:
        # Single 2D image — no Z or T information, wrap into (1, 1, Y, X)
        Z, T, Y, X = 1, 1, s[0], s[1]
        print(f"guess_axes: 2D image interpreted as (Z=1, T=1, Y={Y}, X={X})")
        return arr.reshape((Z, T, Y, X))

    if arr.ndim == 3:
        a, b, c = s
        if a <= Z_MAX_EXPECTED and b > Z_MAX_EXPECTED and c > Z_MAX_EXPECTED:
            # First axis is small -> treat as Z, no T dimension present
            Z, T, Y, X = a, 1, b, c
            print(f"guess_axes: 3D array interpreted as (Z={Z}, T=1, Y={Y}, X={X})")
            return arr.reshape((Z, T, Y, X))
        elif a > Z_MAX_EXPECTED and b > Z_MAX_EXPECTED and c > Z_MAX_EXPECTED:
            # All dims large -> assume (T, Y, X), single Z-slice
            Z, T, Y, X = 1, a, b, c
            print(f"guess_axes: 3D array interpreted as (Z=1, T={T}, Y={Y}, X={X})")
            return arr.reshape((Z, T, Y, X))
        else:
            # Ambiguous — raise rather than silently returning wrong data
            raise ValueError(
                f"guess_axes: ambiguous 3D shape {s}. "
                f"Cannot confidently assign axes. Check TIFF metadata in Fiji (Image -> Properties)."
            )

    if arr.ndim == 4:
        a, b, c, d = s
        if a <= Z_MAX_EXPECTED and b > Z_MAX_EXPECTED:
            # (Z, T, Y, X) — expected standard case for our data
            Z, T, Y, X = a, b, c, d
            print(f"guess_axes: 4D array interpreted as (Z={Z}, T={T}, Y={Y}, X={X}) — standard case")
            return arr  # already in correct order

        elif b <= Z_MAX_EXPECTED and a > Z_MAX_EXPECTED:
            # (T, Z, Y, X) — transpose to (Z, T, Y, X)
            Z, T, Y, X = b, a, c, d
            print(f"guess_axes: 4D array interpreted as (T, Z, Y, X) -> transposing to (Z={Z}, T={T}, Y={Y}, X={X})")
            return arr.transpose((1, 0, 2, 3))

        elif a <= Z_MAX_EXPECTED and b <= Z_MAX_EXPECTED:
            # Both first axes are small — could be (C, Z, Y, X) or similar multi-channel layout
            # We don't handle multi-channel here; raise to avoid silent errors
            raise ValueError(
                f"guess_axes: both first axes are small {s}. "
                f"Possible multi-channel data (C, Z, Y, X)? "
                f"Extract single channel before passing to this pipeline."
            )

        else:
            # Fallback: all dims large, cannot determine axis order reliably
            raise ValueError(
                f"guess_axes: cannot determine axis order for shape {s}. "
                f"All leading dimensions exceed Z_MAX_EXPECTED={Z_MAX_EXPECTED}. "
                f"Check TIFF metadata in Fiji (Image -> Properties)."
            )

    raise ValueError(f"guess_axes: unhandled array dimensionality {arr.ndim}D, shape {s}")

def load_polygon_roi_as_mask(roi_path, ref_shape):
    """
    Load a polygon ROI (.roi exported from ImageJ/Fiji) and return a boolean mask (Y, X).
    - roi_path: path to .roi file
    - ref_shape: tuple (Z, T, Y, X) of the reference image
    Returns: numpy boolean array shape (Y, X) with True inside polygon.
    """
    if not _HAS_READ_ROI:
        raise ImportError("read-roi not installed. Install with `pip install read-roi`.")

    rois = read_roi_file(str(roi_path))
    if len(rois) == 0:
        raise ValueError("No ROI entries found in the .roi file.")

    # pick the first ROI entry
    key = next(iter(rois))
    roi = rois[key]

    Y, X = int(ref_shape[2]), int(ref_shape[3])
    mask = np.zeros((Y, X), dtype=bool)

    # Preferred: ImageJ polygon/freehand -> 'coords' list of (x,y)
    if 'coords' in roi:
        coords = roi['coords']
        xs = np.array([int(round(c[0])) for c in coords], dtype=int)
        ys = np.array([int(round(c[1])) for c in coords], dtype=int)
        print("ROI polygon identified.")

    # Alternate older style: 'x' and 'y' arrays (unlikely this would be used)
    elif 'x' in roi and 'y' in roi:
        xs = np.array(roi['x'], dtype=int)
        ys = np.array(roi['y'], dtype=int)
        print("Coordinate ROI identified.")

    else:
        # Unexpected structure — show keys for debugging
        raise ValueError(f".roi does not contain polygon coords. Keys present: {list(roi.keys())}")

    # Clip to image bounds
    xs = np.clip(xs, 0, X - 1)
    ys = np.clip(ys, 0, Y - 1)

    # Rasterise polygon into mask
    from skimage.draw import polygon
    rr, cc = polygon(ys, xs, shape=(Y, X))
    mask[rr, cc] = True
    return mask

#################### MAIN PROCESSING ####################
print("Loading TIFF:", tif_path)
arr_raw = load_tif(tif_path)
#print("raw shape:", arr_raw.shape, "dtype:", arr_raw.dtype) (not necessary for output)
# canonicalize to (Z, T, Y, X)
arr4 = guess_axes(arr_raw)
Z, T, Y, X = arr4.shape
#print("Interpreted shape (Z, T, Y, X) =", arr4.shape) (also not necessary but good for error checking)

# convert to float32 for computations, preserve original for saving if needed
img = img_as_float32(arr4)  # values normalised to float range if necessary

# build or load ROI mask
if Path(roi_roi_path).exists():
    print("Loading polygon ROI from .roi:", roi_roi_path)
    roi_mask = load_polygon_roi_as_mask(roi_roi_path, img.shape)
else:
    print("No ROI provided. Using full frame as ROI (not recommended).")
    roi_mask = np.ones((Y, X), dtype=bool)

# precompute mask indices (flattened) to speed inner loops
mask_idx = np.nonzero(roi_mask)
mask_area = mask_idx[0].size
print("ROI area (pixels):", mask_area)

# results containers
bestZ_raw = np.zeros(T, dtype=int)
Focus_peak_vals = np.zeros((T, Z), dtype=float)

from scipy.stats import scoreatpercentile

for t in range(T):
    Focus = np.zeros(Z, dtype=float)
    for z in range(Z):
        frame = img[z, t]  # 2D array Y,X
        # sample pixels inside ROI
        vals = frame[mask_idx]  # 1D array (mask_area,)
        if vals.size == 0:
            Focus[z] = 0
            continue
        # percentiles and means
        mean_above = float(vals.mean())
        # top X% mean
        k = max(1, int(np.ceil((top_percent/100.0) * vals.size))) #scoring for punctate intensity
        topk = np.partition(vals, -k)[-k:]
        top_mean = float(np.mean(topk))
        # sharpness: Laplacian variance within ROI
        lap = laplace(frame)  # same shape
        lap_vals = lap[mask_idx]
        sharp_var = float(np.var(lap_vals)) if lap_vals.size>0 else 0.0
        # normalize sharp later; for now store numeric components
        Focus_peak_vals[t, z] = mean_above + beta * (top_mean - mean_above) + alpha * sharp_var
        Focus[z] = Focus_peak_vals[t, z]
    # pick best z
    bestZ_raw[t] = int(np.argmax(Focus))  # 0-based index
    # store focus if you want
print("bestZ_raw (0-based):", bestZ_raw)

# temporal smoothing (running median) and clamp jumps
from scipy.signal import medfilt
# medfilt expects odd kernel; use manual median to be explicit
def running_median(arr, window):
    pad = window//2
    n = len(arr)
    out = np.empty_like(arr)
    for i in range(n):
        lo = max(0, i - pad)
        hi = min(n, i + pad + 1)
        out[i] = int(np.median(arr[lo:hi]))
    return out

bestZ_smoothed = running_median(bestZ_raw, median_window)
# limit jumps
for t in range(1, T):
    delta = bestZ_smoothed[t] - bestZ_smoothed[t-1]
    if abs(delta) > max_jump:
        bestZ_smoothed[t] = bestZ_smoothed[t-1] + np.sign(delta)*max_jump

print("bestZ_smoothed (0-based):", bestZ_smoothed)

# Create ±half_width max projection stack for detection and save
mip_stack = np.zeros((T, Y, X), dtype=dtype_out)
if do_sum := True:
    sum_stack = np.zeros_like(mip_stack)

for t in range(T):
    zc = bestZ_smoothed[t]
    zmin = max(0, zc - half_width)
    zmax = min(Z-1, zc + half_width)
    slab = img[zmin:zmax+1, t]  # shape (nZ, Y, X)
    mip = np.max(slab, axis=0)
    mip_stack[t] = mip
    if do_sum:
        sum_stack[t] = np.sum(slab, axis=0)

#################### MAKING AND SAVING FILE OUTPUTS ####################

# Find next available version number and construct file names dynamically, useful for testing
def next_version(out_dir, base_name):
    v = 1
    while True:
        candidate = out_dir / f"{base_name}_v{v}.tif"
        if not candidate.exists():
            return v
        v += 1
    
v = next_version(out_dir, f"{input_name}")

# Output and save files to the directory
mip_out = out_dir / f"{input_name}_max_projection_v{v}.tif"
tiff.imwrite(str(mip_out), mip_stack.astype(dtype_out), imagej=True)

if do_sum:
    sum_out = out_dir / f"{input_name}_sum_projection_v{v}.tif"
    tiff.imwrite(str(sum_out), sum_stack.astype(dtype_out), imagej=True)


# Build diagnostics dataframe to be output
rows = []
for t in range(T):
    zc_raw = bestZ_raw[t]
    zc_sm = bestZ_smoothed[t]
    zmin = max(0, zc_sm - half_width)
    zmax = min(Z-1, zc_sm + half_width)
    rows.append({
        "frame": int(t),
        "bestZ_raw": int(zc_raw),
        "bestZ_smoothed": int(zc_sm),
        "zmin": int(zmin),
        "zmax": int(zmax)
    })

df = pd.DataFrame(rows)

csv_out = out_dir / f"{input_name}_focus_diagnostics_v{v}.csv"
df.to_csv(csv_out, index=False)

print("Saved MIP to:", mip_out)
print("Saved diagnostics to:", csv_out)