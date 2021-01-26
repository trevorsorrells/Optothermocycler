# -*- coding: utf-8 -*-
# Analysis of jaaba behavior classification for mosquito optothermocycler setup
# python 3.5

import sys
print(sys.path)
sys.path.insert(0, "/Volumes/Backup6/python")
print(sys.path)
import optothermo


project_folder = r'D:\2019_10_30'

plate_corners = [[20,20],[1066,16],[1065,690],[20,689]]
plate_rows, plate_cols = 3,5

frame_rate = 30
velocity_step_frames = 1 #number of frames skipped to calculate each velocity time point, only for using custom velocity from tracking
displacement_time_frames = 30 # number of frames before current time point that are used to calculate velocity
well_size_pixels = 158.0
well_size_mm = 17
mm_conversion = well_size_mm/well_size_pixels*frame_rate/displacement_time_frames
movement_cutoff = 1.5 #cutoff for mosquito to be considered moving in mm/s
n_wells = 14 #hard coded number of wells in each plate
behaviors = ['groom3','walk3','probe5','fly2'] #highest priority behavior is last one (graphed last and overwrites for transitions)
behavior_colors = ['#C0BA9C','#863810','#FA0F0C','#78C0A7']
#rgbabehavior_none_names = ['groom3','walk2','probe5','fly2','none']
#behavior_none_colors = ['#C0BA9C','#863810','#FA0F0C','#78C0A7','#EEEEEE']
# behaviors = ['groom2','walk2','probe2','probe5','fly2']
# behavior_colors = ['#C0BA9C','#863810','#FFC0CB','#FA0F0C','#78C0A7']



#stim_type_indices are grouped by stimulus type, then plate=trial, then multiple numbers indicates multiple presentations
#all trials for graph
all_trial_list = ['trial1_LVP_heat','trial2_LVP_light','trial3_LVP_comb','trial4_CsC_heat','trial5_CsC_light','trial6_CsC_comb','trial7_Gr3_heat','trial8_Gr3_light',
    'trial9_Gr3_comb','trial10_DP_heat','trial11_DP_light','trial12_DP_comb',
    'trial13_DP_heat','trial14_DP_light','trial15_DP_comb','trial16_LVP_heat','trial17_LVP_light','trial18_LVP_comb','trial19_CsC_heat','trial20_CsC_light','trial21_CsC_comb',
    'trial22_Gr3_heat','trial23_Gr3_light','trial24_Gr3_comb','trial25_Gr3_heat','trial26_Gr3_light','trial27_Gr3_comb','trial28_DP_heat','trial29_DP_light',
    'trial30_DP_comb','trial31_LVP_heat','trial32_LVP_light','trial33_LVP_comb','trial34_CsC_heat','trial35_CsC_light','trial36_CsC_comb',
    'trial37_CsC_heat','trial38_CsC_light','trial39_CsC_comb','trial40_Gr3_heat','trial41_Gr3_light','trial42_Gr3_comb',
    'trial43_DP_heat','trial44_DP_light','trial45_DP_comb','trial46_LVP_heat','trial47_LVP_light','trial48_LVP_comb','trial49_LVP_heat',
    'trial50_LVP_light','trial51_LVP_comb','trial52_CsC_heat','trial53_CsC_light','trial54_CsC_comb','trial55_Gr3_heat','trial56_Gr3_light',
    'trial57_Gr3_comb','trial58_DP_heat','trial59_DP_light','trial60_DP_comb']
# for trial_name in trial_list:
#     outputStimuli(trial_name)

# optothermo.joinStimuliTracks(all_trial_list, project_folder, ['walk3'], plate_corners, plate_rows, plate_cols)

trial_list = ['trial1_LVP_heat','trial2_LVP_light','trial3_LVP_comb']

# doublepositive = OptoThermoExp2(trial_list)
# doublepositive.stimulusGraph(trial_list, 60,180)

trial_list = ['trial4_CsC_heat','trial5_CsC_light','trial6_CsC_comb','trial19_CsC_heat','trial20_CsC_light','trial21_CsC_comb',
    'trial34_CsC_heat','trial35_CsC_light','trial36_CsC_comb','trial37_CsC_heat','trial38_CsC_light','trial39_CsC_comb',
    'trial52_CsC_heat','trial53_CsC_light','trial54_CsC_comb']

trial_list = ['trial7_Gr3_heat','trial8_Gr3_light','trial9_Gr3_comb','trial22_Gr3_heat','trial23_Gr3_light','trial24_Gr3_comb',
    'trial25_Gr3_heat','trial26_Gr3_light','trial27_Gr3_comb','trial40_Gr3_heat','trial41_Gr3_light','trial42_Gr3_comb',
    'trial55_Gr3_heat','trial56_Gr3_light','trial57_Gr3_comb']

trial_list = ['trial10_DP_heat','trial11_DP_light','trial12_DP_comb','trial13_DP_heat','trial14_DP_light','trial15_DP_comb',
    'trial28_DP_heat','trial29_DP_light','trial30_DP_comb','trial43_DP_heat','trial44_DP_light','trial45_DP_comb',
    'trial58_DP_heat','trial59_DP_light','trial60_DP_comb']

LVP_trial_list = ['trial1_LVP_heat','trial2_LVP_light','trial3_LVP_comb','trial16_LVP_heat','trial17_LVP_light','trial18_LVP_comb',
    'trial31_LVP_heat','trial32_LVP_light','trial33_LVP_comb','trial46_LVP_heat','trial47_LVP_light','trial48_LVP_comb','trial49_LVP_heat',
    'trial50_LVP_light','trial51_LVP_comb']


# joinStimuliTracks(trial_list, behaviors)
stim_type_indices = [[[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[]],  
                    [[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[]],  
                    [[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1]]]
stim_type_names = ['heat','light','combined']

# LVPexperiment = optothermo.OptoThermoExp(LVP_trial_list, project_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames)
# LVPexperiment.plot_individuals_behaviors('LVP_finalclassifiers', stim_type_indices, stim_type_names, 120, 900, ethogram=True, behavior_graph=True, stats_file=True)

# doublepositive.statistics_test('probe2','doublepositive', stim_type_indices, stim_type_names, 300)


trial_list = ['trial12_DP_comb','trial15_DP_comb','trial30_DP_comb','trial45_DP_comb','trial60_DP_comb']

# tSNE analysis
trial_list = ['trial10_DP_heat','trial11_DP_light','trial12_DP_comb','trial13_DP_heat','trial14_DP_light','trial15_DP_comb',
    'trial28_DP_heat','trial29_DP_light','trial30_DP_comb','trial43_DP_heat','trial44_DP_light','trial45_DP_comb',
    'trial58_DP_heat','trial59_DP_light','trial60_DP_comb']
stim_type_indices = [[[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[]],  
                    [[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[]],  
                    [[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1]]]
stim_type_names = ['heat','light','combined']

trial_list = ['trial10_DP_heat','trial11_DP_light','trial12_DP_comb','trial13_DP_heat','trial14_DP_light','trial15_DP_comb',]
trial_list = ['trial4_CsC_heat','trial5_CsC_light','trial6_CsC_comb','trial7_Gr3_heat','trial8_Gr3_light','trial9_Gr3_comb']

trial_list = ['trial10_DP_heat']
stim_type_indices = [[[1]]]
stim_type_names = ['heat']

#trial_list = ['trial11_DP_light']
#stim_type_indices = [[[1]]]
##stim_type_names = ['light']
#trial_list = all_trial_list
trial_list =  ['trial1_LVP_heat','trial2_LVP_light','trial3_LVP_comb','trial4_CsC_heat','trial5_CsC_light','trial6_CsC_comb','trial7_Gr3_heat','trial8_Gr3_light',
'trial9_Gr3_comb','trial10_DP_heat','trial11_DP_light','trial12_DP_comb',
'trial13_DP_heat','trial14_DP_light','trial15_DP_comb','trial16_LVP_heat','trial17_LVP_light','trial18_LVP_comb',
'trial19_CsC_heat','trial20_CsC_light','trial21_CsC_comb',
    'trial22_Gr3_heat','trial23_Gr3_light','trial24_Gr3_comb','trial25_Gr3_heat','trial26_Gr3_light','trial27_Gr3_comb','trial28_DP_heat','trial29_DP_light',
    'trial30_DP_comb','trial31_LVP_heat','trial32_LVP_light','trial33_LVP_comb','trial34_CsC_heat','trial35_CsC_light','trial36_CsC_comb',
    'trial37_CsC_heat','trial38_CsC_light','trial39_CsC_comb','trial40_Gr3_heat','trial41_Gr3_light','trial42_Gr3_comb',
    'trial43_DP_heat','trial44_DP_light','trial45_DP_comb','trial46_LVP_heat','trial47_LVP_light','trial48_LVP_comb','trial49_LVP_heat',
    'trial50_LVP_light','trial51_LVP_comb','trial52_CsC_heat','trial53_CsC_light','trial54_CsC_comb','trial55_Gr3_heat','trial56_Gr3_light',
    'trial57_Gr3_comb','trial58_DP_heat','trial59_DP_light','trial60_DP_comb']
stim_type_indices = [[[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[]],  
                    [[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[]],  
                    [[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1]]]
stim_type_names = ['heat','light','combined']

# trial_list = ['trial10_DP_heat','trial11_DP_light','trial12_DP_comb','trial13_DP_heat','trial14_DP_light','trial15_DP_comb',
#     'trial28_DP_heat','trial29_DP_light','trial30_DP_comb','trial43_DP_heat','trial44_DP_light','trial45_DP_comb',
#     'trial58_DP_heat','trial59_DP_light','trial60_DP_comb']
# stim_type_indices = [[[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[]],  
#                     [[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[]],  
#                     [[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1]]]
# stim_type_names = ['heat','light','combined']y

# trial_list = ['trial1_LVP_heat','trial2_LVP_light','trial3_LVP_comb','trial16_LVP_heat','trial17_LVP_light','trial18_LVP_comb',
#     'trial31_LVP_heat','trial32_LVP_light','trial33_LVP_comb','trial46_LVP_heat','trial47_LVP_light','trial48_LVP_comb','trial49_LVP_heat',
#     'trial50_LVP_light','trial51_LVP_comb']
# stim_type_indices = [[[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[]],  
#                     [[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[]],  
#                     [[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1]]]
# stim_type_names = ['heat','light','combined']

# trial_list =  ['trial4_CsC_heat','trial5_CsC_light','trial6_CsC_comb',
# 'trial19_CsC_heat','trial20_CsC_light','trial21_CsC_comb','trial34_CsC_heat','trial35_CsC_light','trial36_CsC_comb',
#     'trial37_CsC_heat','trial38_CsC_light','trial39_CsC_comb','trial52_CsC_heat','trial53_CsC_light','trial54_CsC_comb']
# stim_type_indices = [[[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[]],  
#                     [[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[]],  
#                     [[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1]]]
# stim_type_names = ['heat','light','combined']
# 
# trial_list =  ['trial7_Gr3_heat','trial8_Gr3_light','trial9_Gr3_comb',
#     'trial22_Gr3_heat','trial23_Gr3_light','trial24_Gr3_comb','trial25_Gr3_heat','trial26_Gr3_light','trial27_Gr3_comb',
#    'trial40_Gr3_heat','trial41_Gr3_light','trial42_Gr3_comb','trial55_Gr3_heat','trial56_Gr3_light','trial57_Gr3_comb']
# stim_type_indices = [[[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[]],  
#                     [[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1],[]],  
#                     [[],[],[1],[],[],[1],[],[],[1],[],[],[1],[],[],[1]]]
# stim_type_names = ['heat','light','combined']
# =============================================================================
# trial_list = ['trial10_DP_heat']
# stim_type_indices = [[[1]]]
# stim_type_names = ['heat']
# =============================================================================

optoexp = optothermo.OptoThermoExp(trial_list,project_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames)
#optoexp.output_cluster_features(stim_type_indices,[240, 960],30)
optoexp.output_tSNE(window_size_seconds=30, perplex=400)
optoexp.graph_tSNE('Alltrials_30s_400perp', window_size_seconds=30, perplex=400, sample_size=3000, dom_behav_graph=True, behav_gradient=True, other_gradient=True, any_behav=True)
#Alltrials_30s_400perp
#optoexp.category_tSNE('All_DPtrials_30s_100perp', 30, stim_type_indices, stim_type_names, [120,900], [['walk-probe','#DC5539',[(-50,-40),(-50,15),(-5,15),(-5,-5),(-10,-10),(-10,-40)]],['fly-probe','#B491C8',[(-10,-70),(-10,-30),(40,-70),(40,-30)]],['groom-probe','#ECB7BF',[(-35,15),(-35,60),(-5,15),(-5,60)]]])#['probe-walk','#DC5539',[(15,10),(35,10),(35,30),(0,20),(-10,20),(-10,30)]]])#['fly','#B491C8',[(15,10),(35,10),(35,30),(0,20),(-10,20),(-10,30)]],['probe-walk','#DC5539',[(15,10),(35,10),(35,30),(0,20),(-10,20),(-10,30)]]])#['walk','#C0BA9C',[(-20,40),(-20,60),(10,40),(10,60))]]])#,['probe-walk','#DC5539', [-20,7,-20,7]]])#, ['light-probe','#ECB7BF',[-60,-38,-50,-18]]])
#[(-10,-70),(10,-70),(-10,-40),(10,-40)]                                                                                                     
#optoexp.plot_individuals_behaviors('All_LVPtrials_30s_150perp', stim_type_indices, stim_type_names, 120, 900, ethogram=True, behavior_graph=True, stats_file=True)
#optoexp.read_videos(trial_list,trial_list)
