import subprocess

import pandas as pd
import src.opensim.opensim_utils as opensim_utils

for participant in ['P01', 'P02', 'P03', 'P04', 'P05', 'P06']:
    base_path = f'results/{participant}/bodyhull/bsipw_shoes/'
    for pipeline, scan in zip(['HIT', 'InsideHumans','SMPL'], ['body', 'skin','body']):
        for skel in [True, False]:
            if skel:
                hull_bsip = pd.read_pickle(f'{base_path}/{pipeline}_{scan}_loose_w_skel.pkl')
                opensim_model = opensim_utils.get_model(f'results/{participant}/gait/addB-ignore_phys/Models/match_markers_but_ignore_physics.osim')
                new_hull_bsip, osim_model = opensim_utils.hull_bsip_to_opensim(hull_bsip, opensim_model, extra_shoe_weight=0, arms_from_hull=False)

                opensim_utils.save_opensim_model(osim_model, f'results/{participant}/bodyhull/bsipw_shoes/{pipeline}_{scan}_loose_w_skel.osim')
            else:
                hull_bsip = pd.read_pickle(f'{base_path}/{pipeline}_{scan}_loose.pkl')
                opensim_model = opensim_utils.get_model(f'results/{participant}/gait/addB-ignore_phys/Models/match_markers_but_ignore_physics.osim')
                new_hull_bsip, osim_model = opensim_utils.hull_bsip_to_opensim(hull_bsip, opensim_model, extra_shoe_weight=0, arms_from_hull=False)

                opensim_utils.save_opensim_model(osim_model, f'results/{participant}/bodyhull/bsipw_shoes/{pipeline}_{scan}_loose.osim')