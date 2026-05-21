# Pre-processing steps for 2P data

## Neurons
1. Run suite2p with depth-for-2p: `python run_suite2p.py --config plain_suite2p.yaml`.
2. Manual curation with suite2p GUI.
3. Run z-stack correction + extract dF/F with depth-for-2p: `python zstack.py --config zstack.yaml`.
4. Check resulting plots. Decide which planes (during which experiments) to exclude.
5. Run final curation with depth-for-2p: `python final_manual_curation.py`.
6. Upload data to NAS.

## Boutons
1. Run z-registration with depth-for-2p: `python zregistration.py --config zregistration.yaml`.
2. Manual curation with suite2p GUI.
3. Extract dF/F with depth-for-2p: `python correct_traces.py --config trace_correction.yaml`.
4. Check resulting plots. To exclude bad experiments, it is easiest to delete any related stimulus information from 
metadata (constructed in next step).
5. Upload data to NAS.

## Metadata (Bonsai files)
1. Run `python main_metadata_per_exp.py`. Set breakpoint as described! Adapt `preprocess.yaml`.
2. Run `python main_metadata_across_exp.py` to combine extracted metadata across experiments.
3. Delte stimulus information for bad experiments.
4. Uplpoad data to NAS.