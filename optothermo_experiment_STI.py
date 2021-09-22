# -*- coding: utf-8 -*-
# Analysis of jaaba behavior classification for mosquito optothermocycler setup
# python 3.5

import sys
if "/Path/to/optothermo/folder" not in sys.path:
    sys.path.insert(0, "/Path/to/optothermo/folder")
    print(sys.path)
import optothermo


project_folder = '/Path/to/experiment/folder'

plate_corners = [[20,20],[1066,16],[1065,690],[20,689]]
plate_rows, plate_cols = 3,5

frame_rate = 30
velocity_step_frames = 1 #number of frames skipped to calculate each velocity time point, only for using custom velocity from tracking
displacement_time_frames = 30 # number of frames before current time point that are used to calculate velocity
well_size_pixels = 158.0
well_size_mm = 17
mm_conversion = well_size_mm/well_size_pixels*frame_rate/displacement_time_frames
n_wells = 14 #number of wells in each plate
behaviors = ['groom3','walk3','probe5','fly2'] #highest priority behavior is last one (graphed last and overwrites for transitions)
behavior_colors = ['#C0BA9C','#863810','#FA0F0C','#78C0A7']

#stim_type_indices are grouped by stimulus type, then plate=trial, then multiple numbers indicates multiple presentations
#all trials for graph
all_trial_list = ['trial1_LVP_heat','trial2_LVP_light','trial3_LVP_comb','trial4_CsC_heat','trial5_CsC_light','trial6_CsC_comb',
    'trial7_Gr3_heat','trial8_Gr3_light','trial9_Gr3_comb','trial10_DP_heat','trial11_DP_light','trial12_DP_comb',
    'trial13_DP_heat','trial14_DP_light','trial15_DP_comb','trial16_LVP_heat','trial17_LVP_light','trial18_LVP_comb',
    'trial19_CsC_heat','trial20_CsC_light','trial21_CsC_comb','trial22_Gr3_heat','trial23_Gr3_light','trial24_Gr3_comb',
    'trial25_Gr3_heat','trial26_Gr3_light','trial27_Gr3_comb','trial28_DP_heat','trial29_DP_light','trial30_DP_comb',
    'trial31_LVP_heat','trial32_LVP_light','trial33_LVP_comb','trial34_CsC_heat','trial35_CsC_light','trial36_CsC_comb',
    'trial37_CsC_heat','trial38_CsC_light','trial39_CsC_comb','trial40_Gr3_heat','trial41_Gr3_light','trial42_Gr3_comb',
    'trial43_DP_heat','trial44_DP_light','trial45_DP_comb','trial46_LVP_heat','trial47_LVP_light','trial48_LVP_comb',
    'trial49_LVP_heat','trial50_LVP_light','trial51_LVP_comb','trial52_CsC_heat','trial53_CsC_light','trial54_CsC_comb',
    'trial55_Gr3_heat','trial56_Gr3_light','trial57_Gr3_comb','trial58_DP_heat','trial59_DP_light','trial60_DP_comb']


all_stim_type_indices = [[[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[]],  
                        [[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[]],  
                        [[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1]]]

optothermo.joinStimuliTracks(all_trial_list, project_folder, behaviors, plate_corners, plate_rows, plate_cols)


# heat by genotype comparison
heat_genotype_indices = [[[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[],[],[]],  
                        [[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[]],  
                        [[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[]],  
                        [[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[]]]
heat_genotype_names = ['LVP_heat','Gr3_heat','CsC_heat','DP_heat']

optoexp = optothermo.OptoThermoExp(all_trial_list, project_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames, well_size_pixels, well_size_mm)
optoexp.statistics_test(['probe5'], 'heat_genotype', heat_genotype_indices, heat_genotype_names, [0,300], additivity_test=False)
optoexp.statistics_test(['walk3'], 'heat_genotype', heat_genotype_indices, heat_genotype_names, [0,300], additivity_test=False)
optoexp.statistics_test(['groom3'], 'heat_genotype', heat_genotype_indices, heat_genotype_names, [0,300], additivity_test=False)


# light by genotype comparison
light_genotype_indices = [[[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[],[]],  
                        [[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[]],  
                        [[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[]],  
                        [[],[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[]]]
light_genotype_names = ['LVP_light','Gr3_light','CsC_light','DP_light']

optoexp = optothermo.OptoThermoExp(all_trial_list, project_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames, well_size_pixels, well_size_mm)
optoexp.statistics_test(['probe5'], 'light_genotype', light_genotype_indices, light_genotype_names, [0,300], additivity_test=False)
optoexp.statistics_test(['walk3'], 'light_genotype', light_genotype_indices, light_genotype_names, [0,300], additivity_test=False)
optoexp.statistics_test(['groom3'], 'light_genotype', light_genotype_indices, light_genotype_names, [0,300], additivity_test=False)
optoexp.statistics_test(['fly2'], 'light_genotype', light_genotype_indices, light_genotype_names, [0,30], additivity_test=False)
optoexp.statistics_test(['fly2'], 'light_genotype', light_genotype_indices, light_genotype_names, [30,300], additivity_test=False)
optoexp.statistics_test(['fly2'], 'light_genotype', light_genotype_indices, light_genotype_names, [300,600], additivity_test=False)
optoexp.statistics_test(['fly2'], 'light_genotype', light_genotype_indices, light_genotype_names, [600,900], additivity_test=False)
optoexp.statistics_test(['fly2'], 'light_genotype', light_genotype_indices, light_genotype_names, [900,1200], additivity_test=False)


#combined by genotype comparison
comb_genotype_indices = [[[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[]],
                        [[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[]],
                        [[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[]],
                        [[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[1]]]
comb_genotype_names = ['LVP_comb','Gr3_comb','CsC_comb','DP_comb']

optoexp = optothermo.OptoThermoExp(all_trial_list, project_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames, well_size_pixels, well_size_mm)
optoexp.statistics_test(['probe5'], 'comb_genotype', comb_genotype_indices, comb_genotype_names, [0,300], additivity_test=False)
optoexp.statistics_test(['walk3'], 'comb_genotype', comb_genotype_indices, comb_genotype_names, [0,300], additivity_test=False)
optoexp.statistics_test(['groom3'], 'comb_genotype', comb_genotype_indices, comb_genotype_names, [0,300], additivity_test=False)


CsC_trial_list = ['trial4_CsC_heat','trial5_CsC_light','trial6_CsC_comb','trial19_CsC_heat','trial20_CsC_light','trial21_CsC_comb',
    'trial34_CsC_heat','trial35_CsC_light','trial36_CsC_comb','trial37_CsC_heat','trial38_CsC_light','trial39_CsC_comb',
    'trial52_CsC_heat','trial53_CsC_light','trial54_CsC_comb']

Gr3_trial_list = ['trial7_Gr3_heat','trial8_Gr3_light','trial9_Gr3_comb','trial22_Gr3_heat','trial23_Gr3_light','trial24_Gr3_comb',
    'trial25_Gr3_heat','trial26_Gr3_light','trial27_Gr3_comb','trial40_Gr3_heat','trial41_Gr3_light','trial42_Gr3_comb',
    'trial55_Gr3_heat','trial56_Gr3_light','trial57_Gr3_comb']

DP_trial_list = ['trial10_DP_heat','trial11_DP_light','trial12_DP_comb','trial13_DP_heat','trial14_DP_light','trial15_DP_comb',
    'trial28_DP_heat','trial29_DP_light','trial30_DP_comb','trial43_DP_heat','trial44_DP_light','trial45_DP_comb',
    'trial58_DP_heat','trial59_DP_light','trial60_DP_comb']

LVP_trial_list = ['trial1_LVP_heat','trial2_LVP_light','trial3_LVP_comb','trial16_LVP_heat','trial17_LVP_light','trial18_LVP_comb',
    'trial31_LVP_heat','trial32_LVP_light','trial33_LVP_comb','trial46_LVP_heat','trial47_LVP_light','trial48_LVP_comb','trial49_LVP_heat',
    'trial50_LVP_light','trial51_LVP_comb']


one_genotype_stim_type_indices = [[[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[]],  
                    [[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[]],  
                    [[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1]]]
stim_type_names = ['heat','light','combined']
one_genotype_heat_light = [[[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[]],  
                    [[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[]]]
stim_type_names_heat_light = ['heat','light']

#graphs

CsCexperiment = optothermo.OptoThermoExp(CsC_trial_list, project_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames, well_size_pixels, well_size_mm)
CsCexperiment.plot_individuals_behaviors('CsC', one_genotype_stim_type_indices, stim_type_names, 120, 900, ethogram=True, behavior_graph=True, stats_file=True)

Gr3experiment = optothermo.OptoThermoExp(Gr3_trial_list, project_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames, well_size_pixels, well_size_mm)
Gr3experiment.plot_individuals_behaviors('Gr3', one_genotype_stim_type_indices, stim_type_names, 120, 900, ethogram=True, behavior_graph=True, stats_file=True)

DPexperiment = optothermo.OptoThermoExp(DP_trial_list, project_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames,well_size_pixels, well_size_mm)
DPexperiment.plot_individuals_behaviors('DP', one_genotype_stim_type_indices, stim_type_names, 120, 900, ethogram=True, behavior_graph=True, stats_file=True)

LVPexperiment = optothermo.OptoThermoExp(LVP_trial_list, project_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames,  well_size_pixels, well_size_mm)
LVPexperiment.plot_individuals_behaviors('LVP', one_genotype_stim_type_indices, stim_type_names, 120, 900, ethogram=True, behavior_graph=True, stats_file=True)


#integration
optoexp = optothermo.OptoThermoExp(DP_trial_list, project_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames, well_size_pixels, well_size_mm)
optoexp.plot_individuals_behaviors('DP_subtract_prop', one_genotype_stim_type_indices, stim_type_names, 150, 180, ethogram=False, behavior_graph=True, addition_graph=True, stats_file=False, step_size=15)








