# Z_Selection_Software
Software for dynamic Z-Selection as part of ILESLA Raff rotation report.

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
