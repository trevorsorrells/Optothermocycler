# -*- coding: utf-8 -*-
# Analysis of jaaba behavior classification for mosquito optothermocycler setup
# python 3.5


import numpy as np
import scipy.io as spio
import math 
import os
import re
from random import shuffle
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('agg')
import seaborn as sns
# from matplotlib.patches import Polygon

from statistics import median
from scipy.stats import friedmanchisquare
from scipy.stats import kruskal
from scipy.stats import chi2_contingency
from scipy.stats import spearmanr
from scipy.stats import ttest_1samp
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
import scikit_posthocs as ph
import statsmodels.api

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

#parameters & packages for state estimation analysis 
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import time
velocity_step_ctrax = 100 #in milliseconds
window_step_size = 10 #in seconds
import subprocess as sp

class OptoThermoExp:
    def __init__(self, trial_list, opto_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames, well_size_pixels, well_size_mm, blood_blanket=False):
        
        self.opto_folder = opto_folder
        self.frame_rate = frame_rate
        self.velocity_step_frames = velocity_step_frames
        self.behaviors = behaviors
        self.behavior_colors = behavior_colors
        self.trial_names = trial_list
        self.well_size_pixels = well_size_pixels
        self.well_size_mm = well_size_mm
        self.blood_blanket = blood_blanket
        if blood_blanket:
            #engorgement wells are indexed top to bottom, left to right
            self.engorged = []
            with open(opto_folder + '/engorgement.txt') as inFile:
                file_data = inFile.readlines()
                engorgement_data = [line.strip().split() for line in file_data]
                for trial in trial_list:
                    for line in engorgement_data:
                        if line[0] == trial:
                            self.engorged.append(list(map(int, line[1:])))
                            break
                    else:
                        print('engorgement data not found for ', trial)
                        self.engorged.append([])
        self.onsets = []
        self.offsets = []
        self.stimulus_type = []
        self.frame_data = [] #frame data is all of the information in joined_stimuli
        self.behavior_data = [] #structure: list of trials, dict of behaviors, dict with t0s, t1s, tStarts, tStops, list of mosquitoes/wells
        for trial_name in trial_list:
            stim = []
            with open(opto_folder + '/joined_stimuli/'+trial_name + '.txt', 'r') as inFile:
                file_contents = inFile.readlines()
                self.onsets.append(list(map(int, file_contents[0].split()[2:])))
                self.offsets.append(list(map(int, file_contents[1].split()[2:])))
                self.stimulus_type.append(file_contents[2].split()[2:])
                for j in range(4,len(file_contents)-1):  
                    line_split = file_contents[j].split()
                    stim.append([int(line_split[0]), int(line_split[1]), float(line_split[2]), int(line_split[3]), int(line_split[4]), int(line_split[5])])
            self.frame_data.append(stim)
            behavior_trial = {}
            for b, behavior in enumerate(behaviors):
                behavior_file_data = {}
                with open(opto_folder + '/joined_behavior/'+trial_name + '_' +behavior+ '.txt', 'r') as inFile:
                    file_contents = inFile.readlines()
                    j = 1
                    data_categories = ['t0s:','t1s:','tStarts:','tEnds:']
                    for c, category in enumerate(data_categories):
                        current_data = []
                        while j < len(file_contents):
                            line_split = file_contents[j].split()
                            j += 1
                            try:
                                if len(line_split) > 0 and line_split[0] == data_categories[c+1]:
                                    break
                            except IndexError:
                                current_data.append(tuple(map(int, line_split)))
                                continue
                            current_data.append(tuple(map(int, line_split)))
                        behavior_file_data[category[:-1]] = current_data
                    #remove mosquitoes that have zero frames tracked (i.e. dead ones)
                    i=0
                    while i < len(behavior_file_data['tStarts']):
                        if len(behavior_file_data['tStarts'][i]) == 0:
                            for c, category in enumerate(data_categories):
                                del behavior_file_data[category[:-1]][i]
                        else:
                            i += 1
                behavior_trial[behavior] = behavior_file_data
            self.behavior_data.append(behavior_trial)
        print("number of different trials:", len(self.behavior_data))
        print("identity of different behaviors:", self.behavior_data[0].keys())
        print("data in each behavior:", self.behavior_data[0][behaviors[0]].keys())
        print("number of mosquito tracks t0s:", len(self.behavior_data[0][behaviors[0]]['t0s']))
        print("number of mosquito tracks tStarts:", len(self.behavior_data[0][behaviors[0]]['tStarts']))
        print("number of mosquito tracks tEnds:", len(self.behavior_data[0][behaviors[0]]['tEnds']))
        print("number of",behaviors[0],"onsets in first mosquito track:", len(self.behavior_data[0][behaviors[0]]['t0s'][0]))

    def plot_individuals_behaviors(self, treatment_name, stimulus_onsets, stimulus_names, prestim, poststim, ethogram=True, behavior_graph=True, stats_file=True, addition_graph=False, step_size=150):
        ymin, ymax = 0, 200
        n_timepoints = int((prestim + poststim) * self.frame_rate/self.velocity_step_frames + 1)
        x_axis = np.linspace(-prestim, poststim, n_timepoints)
        newpath = os.path.join(self.opto_folder,'graphs', treatment_name)
        # total_skeeters = len(self.behavior_data[0][behaviors[0]]['t0s'])
        stats_data = []
        for_addition = []
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        for st, stim_type in enumerate(stimulus_onsets): #iterate through stimulus types
            print('graph for stimuli ', stim_type, stimulus_names[st])
            stimulus_on = 0
            i=0
            while True:
                if len(stim_type[i]) > 0:
                    try:
                        stimulus_off = (self.offsets[0][stim_type[i][0]] - self.onsets[0][stim_type[i][0]])*self.velocity_step_frames/self.frame_rate
                    except IndexError:
                        stimulus_off = 5 #default value
                    break
                i += 1
            onsets_hist = []
            [onsets_hist.append([]) for behavior in self.behaviors]
            behavior_hist = []
            [behavior_hist.append([]) for behavior in self.behaviors]
            ethogram_data = [[]] #ethogram structure is gaps, then behaviors
            [ethogram_data.append([]) for behavior in self.behaviors]
            for b, behavior_name in enumerate(self.behaviors):
                print(behavior_name)
                global_m = 0
                for t in range(len(self.behavior_data)):
                    for i in stim_type[t]:
                        start_frame = int(self.onsets[t][i] - prestim*self.frame_rate/self.velocity_step_frames)
                        end_frame = int(self.onsets[t][i] + poststim*self.frame_rate/self.velocity_step_frames + 1)
                        print('frames ',start_frame, end_frame, stimulus_names[st])
                        for m in range(len(self.behavior_data[t][behavior_name]['t0s'])):
                            #find gaps
                            gap_starts = []
                            gap_ends = []
                            tStarts = sorted(self.behavior_data[t][behavior_name]['tStarts'][m])
                            tEnds = sorted(self.behavior_data[t][behavior_name]['tEnds'][m])
                            if len(tStarts) + len(tEnds) == 0:
                                continue
                            for g in range(1, len(tStarts)):
                                prev_end = tEnds[g-1]
                                curr_start = tStarts[g]
                                if curr_start > prev_end:
                                    if prev_end >= start_frame and prev_end < end_frame:
                                        gap_starts.append(prev_end)
                                    if curr_start >= start_frame and curr_start < end_frame:
                                        gap_ends.append(curr_start)
                            #if movie ends before end of graph
                            if tEnds[-1] >= start_frame and tEnds[-1] < end_frame:
                                gap_starts.append(tEnds[-1])
                            # if len(gap_starts) + len(gap_ends) == 0:
                            #     #detect if there are no gaps in start_frame:end_frame or if entirety is gap and skip graphing)
                            #     for start, end in zip(self.behavior_data[t][behavior_name]['tStarts'][m], self.behavior_data[t][behavior_name]['tEnds'][m]):
                            #         if end > end_frame and start < start_frame:
                            #             break
                            #     else:
                            #         print('trial',t,'mosquito',m,'tStarts',tStarts,'tEnds',tEnds)
                            #convert tStarts, tEnds to gaps that will be graphed as black, reverse tStarts and tEnds in fixBehaviorRange to accomplish this
                            # print(gap_starts, gap_ends)
                            gap_onsets, gap_offsets = fixBehaviorRange(gap_starts, gap_ends, start_frame, end_frame)
                            #calculate behavior onsets and offsets
                            temp_onsets = [x for x in self.behavior_data[t][behavior_name]['t0s'][m] if  x >= start_frame and x < end_frame]
                            temp_offsets = [x for x in self.behavior_data[t][behavior_name]['t1s'][m] if  x >= start_frame and x < end_frame]
                            behavior_onsets, behavior_offsets = fixBehaviorRange(temp_onsets, temp_offsets, start_frame, end_frame)
                            if len(behavior_onsets) != len(behavior_offsets):
                                print(st, b, t, i, behavior_name)
                                print(start_frame,end_frame)
                                print(temp_onsets)
                                print(behavior_onsets)
                                print(temp_offsets)
                                print(behavior_offsets)
                            onsets_hist[b].extend(behavior_onsets)
                            #add 1s and 0s  of whether mosquito is exhibiting behavior at each timepoint
                            #Used to draw on Postprocessed dataset but now converts from beahavior_onsets/offsets
                            behavior_list = []
                            if len(behavior_onsets) + len(behavior_offsets) == 0:
                                [behavior_list.append(0) for x in range(end_frame - start_frame)]
                            else:
                                [behavior_list.append(0) for x in range(behavior_onsets[0]-start_frame)]
                                [behavior_list.append(1) for x in range(behavior_offsets[0]-behavior_onsets[0])]
                                for o in range(1, len(behavior_onsets)):
                                    gap = behavior_onsets[o] - behavior_offsets[o-1]
                                    if gap > 0:
                                        [behavior_list.append(0) for x in range(gap)]
                                        bout_length = behavior_offsets[o]-behavior_onsets[o]
                                    else:
                                        bout_length = behavior_offsets[o]-behavior_onsets[o]+gap
                                        print('behavior bout starts before previous one ends: start prev stop',behavior_onsets[o], behavior_offsets[o-1])
                                    [behavior_list.append(1) for x in range(bout_length)]
                                [behavior_list.append(0) for x in range(end_frame - behavior_offsets[-1])]
                            # print(behavior_onsets, behavior_offsets, len(behavior_list))
                            behavior_hist[b].append(behavior_list)
                            #convert onset/offsets to centered around stimulus in units of seconds
                            gap_onsets = [(x-self.onsets[t][i])*self.velocity_step_frames/self.frame_rate for x in gap_onsets]
                            gap_offsets = [(x-self.onsets[t][i])*self.velocity_step_frames/self.frame_rate for x in gap_offsets]
                            behavior_onsets = [(x-self.onsets[t][i])*self.velocity_step_frames/self.frame_rate for x in behavior_onsets]
                            behavior_offsets = [(x-self.onsets[t][i])*self.velocity_step_frames/self.frame_rate for x in behavior_offsets]
                            # print(gap_onsets, gap_offsets)
                            #create data structure with the following format
                            if b == 0:
                                ethogram_data[0].append([gap_onsets, gap_offsets])
                            ethogram_data[b+1].append([behavior_onsets, behavior_offsets])
                            global_m += 1
            if ethogram:
                plt.figure()
                ax = plt.subplot(1,1,1)
                #plot stimulus starting at 0
                plt.fill(np.array([stimulus_on,stimulus_on,stimulus_off,stimulus_off]),np.array([ymin,len(ethogram_data[0]),len(ethogram_data[0]),ymin]), 'red', alpha=0.3)
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(5)
                plt.xlim(-prestim, poststim)
                plt.title(stimulus_names[st])
                plt.ylabel('Number of mosuqito', fontsize=13)
                plt.xlabel('Time/s', fontsize=13)
                ax.tick_params(axis='both', which='major', labelsize=13)
                #reshape ethogram, calculate amount of probing, and sort
                ethogram_sort = []
                for i in range(len(ethogram_data[0])):
                    total_behavior = 0
                    behavior_index = self.behaviors.index('probe5') + 1
                    for on, off in zip(ethogram_data[behavior_index][i][0],ethogram_data[behavior_index][i][1]):
                        total_behavior += off-on
                    next_entry = [total_behavior,ethogram_data[0][i]]
                    for j in range(len(self.behaviors)):
                        next_entry.append(ethogram_data[j+1][i])
                    ethogram_sort.append(next_entry)
                ethogram_sort.sort(reverse = True)
                #subset mosquitoes for simpler ethogram
                ethogram_sort = ethogram_sort[2::3]
                #plot gaps and behavior bouts
                plt.ylim(ymin,len(ethogram_sort))
                for i, entry in enumerate(ethogram_sort):
                    for ongap, offgap in zip(entry[1][0], entry[1][1]):
                        plt.fill(np.array([ongap,ongap,offgap,offgap]),np.array([i+1,i,i,i+1]), 'black')
                    for b in range(len(self.behaviors)):
                        for onset, offset in zip(entry[b+2][0], entry[b+2][1]):
                            plt.fill(np.array([onset,onset,offset,offset]),np.array([i+1,i,i,i+1]), self.behavior_colors[b])
                plt.savefig(newpath + '/onsetindex_' + str(st) + '.pdf', format='pdf') 
                plt.close('all')
                with open(newpath + '/onsetindex_' + str(st) + '.txt', 'w') as outFile:
                    for i, entry in enumerate(ethogram_sort):
                        outFile.write('\nmosquito '+ str(i+1) + '\ntracking gap starts:\t')
                        outFile.write('\t'.join(map(str, entry[1][0])))
                        outFile.write('\ntracking gap stops:\t')
                        outFile.write('\t'.join(map(str, entry[1][1])))
                        for b, behavior in enumerate(self.behaviors):
                            outFile.write('\n'+behavior+' starts \t')
                            outFile.write('\t'.join(map(str, entry[b+2][0])))
                            outFile.write('\n'+behavior+' stops \t')
                            outFile.write('\t'.join(map(str, entry[b+2][1])))    
            for_addition.append(behavior_hist)
            #behavior_graph
            if behavior_graph:
                plt.figure()
                plt.ylim(0,.5)
                plt.xlim(-prestim, poststim)
                x_axis = np.linspace(-prestim, poststim, int(n_timepoints/step_size))
                with open(newpath + '/prop_behavior_' + str(st) + '.txt','w') as outFile:
                    outFile.write('time\t'+'\t'.join(map(str, x_axis))+'\n')
                    for b, behavior in enumerate(self.behaviors):
                        i = step_size
                        behavior_to_plot = []
                        list_to_array = np.asarray(behavior_hist[b])
                        # print(type(list_to_array))
                        # print(np.shape(list_to_array))
                        behavior_hist_array = np.nanmean(list_to_array, 0)
                        while i < len(behavior_hist_array):
                            behavior_to_plot.append(np.mean(behavior_hist_array[i-step_size:i]))
                            i += step_size
                        outFile.write(behavior +'\t'+'\t'.join(map(str, behavior_to_plot))+'\n')
                        plt.plot(x_axis, behavior_to_plot, color = self.behavior_colors[b])
                plt.title(stimulus_names[st])
                plt.savefig(newpath + '/prop_behavior_' + str(st) + '.pdf', format='pdf')
                plt.close('all')
            if stats_file:
                stats_behavior = 'probemale'
                b = self.behaviors.index(stats_behavior)
                list_to_array = np.asarray(behavior_hist[b])
                behavior_hist_array = np.nanmean(list_to_array, 0)
                prestim_average = np.mean(behavior_hist_array[0:int(prestim*self.frame_rate/self.velocity_step_frames)])
                window_seconds = 15
                window_frames = int(window_seconds*self.frame_rate/self.velocity_step_frames)
                current_stats = [0.0,0.0] #first two values are midpoint seconds after stimulus onset and proportion of animals exhibiting max response
                i = int(prestim*self.frame_rate/self.velocity_step_frames)
                while i < len(behavior_hist_array)-window_frames:
                    current_behavior = np.mean(behavior_hist_array[i:i+window_frames])
                    if len(current_stats) == 2 and current_behavior > current_stats[1]:
                        current_stats[0] = round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim)
                        current_stats[1] = current_behavior
                    if len(current_stats) == 2 and current_behavior <= (current_stats[1]-prestim_average)*3/4+prestim_average:
                        current_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                    if len(current_stats) == 3 and current_behavior <= (current_stats[1]-prestim_average)*1/2+prestim_average:
                        current_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                    if len(current_stats) == 4 and current_behavior <= (current_stats[1]-prestim_average)*0.368+prestim_average:
                        current_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                    if len(current_stats) == 5 and current_behavior <= (current_stats[1]-prestim_average)*1/4+prestim_average:
                        current_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                    if len(current_stats) == 6 and current_behavior <= prestim_average:
                        current_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                    i += 1
                #resample to get 95% CI for tau based on bootstrapping
                sample_size = 100
                ci_threshold = 0.95
                all_samples = []
                for x in range(sample_size):
                    if x%100 == 0:
                        print('sample', x)
                    random_indices = np.random.choice(a=np.shape(list_to_array)[0], size=np.shape(list_to_array)[0])
                    sample = list_to_array[random_indices, :]
                    behavior_hist_array = np.nanmean(sample, 0)
                    prestim_average = np.mean(behavior_hist_array[0:int(prestim*self.frame_rate/self.velocity_step_frames)])
                    window_seconds = 15
                    window_frames = int(window_seconds*self.frame_rate/self.velocity_step_frames)
                    sample_stats = [0.0,0.0] #first two values are midpoint seconds after stimulus onset and proportion of animals exhibiting max response
                    i = int(prestim*self.frame_rate/self.velocity_step_frames)
                    while i < len(behavior_hist_array)-window_frames:
                        current_behavior = np.mean(behavior_hist_array[i:i+window_frames])
                        if len(sample_stats) == 2 and current_behavior > sample_stats[1]:
                            sample_stats[0] = round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim)
                            sample_stats[1] = current_behavior
                        if len(sample_stats) == 2 and current_behavior <= (sample_stats[1]-prestim_average)*3/4+prestim_average:
                            sample_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                        if len(sample_stats) == 3 and current_behavior <= (sample_stats[1]-prestim_average)*1/2+prestim_average:
                            sample_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                        # if len(sample_stats) == 4:
                        #     break
                        if len(sample_stats) == 4 and current_behavior <= (sample_stats[1]-prestim_average)*0.368+prestim_average:
                            sample_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                        if len(sample_stats) == 5 and current_behavior <= (sample_stats[1]-prestim_average)*1/4+prestim_average:
                            sample_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                        if len(sample_stats) == 6 and current_behavior <= prestim_average:
                            sample_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                        i += 1
                    while len(sample_stats) < 7:
                        sample_stats.append(np.nan)
                    all_samples.append(sample_stats)
                # print(all_samples)
                all_samples = np.asarray(all_samples)
                print(np.shape(all_samples))
                min_index = int((1.0-ci_threshold)/2 * sample_size)
                max_index = int(((1.0-ci_threshold)/2 + ci_threshold) * sample_size)
                print(min_index, max_index)
                current_stats.append(sorted(all_samples[:,0])[min_index])
                current_stats.append(sorted(all_samples[:,0])[max_index])
                current_stats.append(sorted(all_samples[:,1])[min_index])
                current_stats.append(sorted(all_samples[:,1])[max_index])
                current_stats.append(sorted(all_samples[:,3])[min_index])
                current_stats.append(sorted(all_samples[:,3])[max_index])
                #only include if using CI for return to baseline as well
                current_stats.append(sorted(all_samples[:,6])[min_index])
                current_stats.append(sorted(all_samples[:,6])[max_index])
                stats_data.append(current_stats)
        if stats_file:
            statspath = self.opto_folder + '/statistics/' 
            if not os.path.exists(statspath):
                os.makedirs(statspath)
            with open(statspath + treatment_name + '.txt', 'w') as outFile:
                outFile.write('behavior examined '+ stats_behavior +'\n'+'window size in seconds '+ str(window_seconds))
                outFile.write('\ntime averaged for baseline in seconds '+ str(prestim))
                outFile.write('\n\nstimulus\ttimetomax\tmaxprop\tt3/4\tt1/2\ttau(36.8%)\tt1/4\tbaseline\t.025timetomax\t97.5timetomax\t.025maxprop\t97.5maxprop\t.025t1/2\t97.5t1/2\t.025baseline\t97.5baseline')
                for stim_name, stat in zip(stimulus_names, stats_data):
                    outFile.write('\n' + stim_name+'\t')
                    outFile.write('\t'.join(map(str, stat)))
        if addition_graph:
            collapsed_behavior = []
            for st in range(len(for_addition)):
                collapsed_behavior.append([])
                for b, behavior in enumerate(self.behaviors):
                    i = step_size
                    behavior_to_plot = []
                    list_to_array = np.asarray(for_addition[st][b])
                    # for s, skeeter in enumerate(for_addition[st][b]):
                    #     if len(skeeter) != 30601:
                    #         print(st, b, s, behavior, len(skeeter))
                    # print(len(for_addition[st][b][0]))
                    # print(type(list_to_array))
                    # print(np.shape(list_to_array))
                    behavior_hist_array = np.nanmean(list_to_array, 0)
                    while i < len(behavior_hist_array):
                        behavior_to_plot.append(np.mean(behavior_hist_array[i-step_size:i]))
                        i += step_size
                    collapsed_behavior[st].append(behavior_to_plot)
            # offsets = [-65,-20, 0, 23,68]
            # offsets = [0,68]
            offsets = [0]
            additions = []
            for o, offset in enumerate(offsets):
                additions.append([])
                modoffset = round(offset*self.frame_rate/step_size)
                plt.figure()
                plt.ylim(0,.5)
                plt.xlim(-prestim+120, poststim-120)
                x_axis = np.linspace(-prestim, poststim, int(n_timepoints/step_size))
                with open(newpath + '/add_offset_' + str(offset) + '.txt','w') as outFile:
                    outFile.write('time\t'+'\t'.join(map(str, x_axis))+'\n')
                    for b, behavior in enumerate(self.behaviors):
                        behavior_to_plot = []
                        if offset <= 0:
                            for i in range(len(collapsed_behavior[0][b])):
                                try: 
                                    # behavior_to_plot.append(collapsed_behavior[0][b][i] + collapsed_behavior[1][b][i+modoffset])
                                    behavior_to_plot.append(collapsed_behavior[0][b][i-modoffset] + collapsed_behavior[1][b][i])
                                except IndexError:
                                    behavior_to_plot.append(collapsed_behavior[0][b][i])
                        else:
                            for i in range(len(collapsed_behavior[1][b])):
                                try:
                                    behavior_to_plot.append(collapsed_behavior[1][b][i+modoffset] + collapsed_behavior[0][b][i])
                                except IndexError:
                                    behavior_to_plot.append(collapsed_behavior[1][b][i])
                        outFile.write(behavior +'\t'+'\t'.join(map(str, behavior_to_plot))+'\n')
                        plt.plot(x_axis, behavior_to_plot, color = self.behavior_colors[b])
                        additions[o].append(behavior_to_plot)
                plt.title(offset)
                plt.savefig(newpath + '/add_offset_' + str(offset) + '.pdf', format='pdf')
                plt.close('all')
            for o, offset in enumerate(offsets):
                plt.figure()
                plt.ylim(-.8,.3)
                plt.xlim(-prestim+120, poststim-120)
                x_axis = np.linspace(-prestim, poststim, int(n_timepoints/step_size))
                with open(newpath + '/subtract_offset_' + str(offset) + '.txt','w') as outFile:
                    outFile.write('time\t'+'\t'.join(map(str, x_axis))+'\n')
                    for b, behavior in enumerate(self.behaviors):
                        subtraction = np.subtract(collapsed_behavior[o+2][b],additions[o][b])
                        denominator = np.max(additions[o][b])
                        subtraction = np.divide(subtraction, denominator)
                        subtraction = smooth(subtraction,9)
                        plt.plot(x_axis, subtraction, color = self.behavior_colors[b])
                        outFile.write(behavior +'\t'+'\t'.join(map(str, subtraction))+'\n')
                plt.title(offset)
                plt.savefig(newpath + '/subtract_offset_' + str(offset) + '.pdf', format='pdf')
                plt.close('all')

    def statistics_test(self, behavior_names, genotype_name, stimulus_onsets, stimulus_names, startstop, additivity_test=False):
        #computes statistics and makes graph comparing responses to stimuli, averaged over seconds
        #between two values in startstop, also averaged over each stimulus repeat, giving a total of one data point
        #per mosquito. Behaviors are summed if there is more than 1 behavior in behavior_names
        behavior_indices = [self.behaviors.index(behavior_name) for behavior_name in behavior_names]
        prestim, poststim = 60, startstop[1]
        n_timepoints = int((prestim + poststim) * self.frame_rate + 1)
        x_axis = np.linspace(-prestim, poststim, n_timepoints)
        test_name = '_'.join(['_'.join(behavior_names),str(startstop)+'s',str(len(stimulus_names))+'stims'])
        newpath = self.opto_folder + '/statistics/' + genotype_name + '/' + test_name
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        frame_ranges = []
        onset_types = []
        [onset_types.append([]) for x in range(len(self.behavior_data))]
        print(stimulus_onsets)
        for t in range(len(self.behavior_data)):
            frame_ranges.append([])
            for stim_type in stimulus_onsets:
                stim_type_range = []
                for onset_index in stim_type[t]:
                    onset = self.onsets[t][onset_index]
                    onset_types[t].append(self.stimulus_type[t][onset_index])
                    stim_type_range.append([onset+startstop[0]*self.frame_rate,onset+startstop[1]*self.frame_rate])
                frame_ranges[t].append(stim_type_range)
        print('trial, stimulus, rep, start/stop', np.shape(frame_ranges))  #organized by trial, stimulus, rep, start/stop
        onsets_hist = []
        [onsets_hist.append([]) for x in stimulus_onsets]
        response_values = []
        for behavior in self.behaviors:
            current_list = [[] for x in stimulus_onsets]
            response_values.append(current_list)
        for b, behavior in enumerate(self.behaviors):
            for t in range(len(self.behavior_data)):
                for m in range(len(self.behavior_data[t][behavior]['t0s'])):
                    for s, stim_type in enumerate(frame_ranges[t]):
                        values_type = []
                        for stim_range in stim_type:
                            temp_onsets = [x for x in self.behavior_data[t][behavior]['t0s'][m] if x >= stim_range[0] and x < stim_range[1]]
                            temp_offsets = [x for x in self.behavior_data[t][behavior]['t1s'][m] if x >= stim_range[0] and x < stim_range[1]]
                            final_onsets, final_offsets = fixBehaviorRange(temp_onsets, temp_offsets, stim_range[0], stim_range[1])
                            if len(final_onsets) > len(final_offsets):
                                i=0
                                while i < len(final_onsets):
                                    if final_offsets[i] < final_onsets[i]:
                                        print('bout starts before it ends', final_offsets[i], final_onsets[i])
                                        del final_onsets[i]
                                    i += 1
                                    # if final_offsets[i] > final_onsets[i+1]:
                                    #     print('next bout starts before current ends, onset, offset, next onset:', final_onsets[i], final_offsets[i], final_onsets[i+1])
                            centered_onsets = [(x-stim_range[0])/self.frame_rate for x in self.behavior_data[t][behavior]['t0s'][m] if x >= stim_range[0]-60*self.frame_rate and x < stim_range[1]]
                            onsets_hist[s].extend(centered_onsets)
                            if len(final_onsets) + len(final_offsets) == 0:
                                values_type.append(0.0)
                            else:
                                values_type.append(sum(np.subtract(np.array(final_offsets), np.array(final_onsets)))/(stim_range[1]-stim_range[0]))
                        if len(values_type) > 0:
                            response_values[b][s].append(sum(values_type)/len(values_type))
        response_array = np.array(response_values[behavior_indices[0]])
        print(np.shape(response_array))
        for behavior_index in behavior_indices[1:]:
            response_array = np.add(response_array, np.array(response_values[behavior_index]))
            
        #adjust for male classifier differences
        # print(response_array[2])
        # response_array[2]= response_array[2]*1.032503
        # print(response_array[2])

        print(np.shape(response_array))
        for col in response_array:
            print(len(col))
        # successes = response_array > 0.01
        # contingency = []
        # for column in successes:
        #     contingency.append([sum(column), len(column)-sum(column)])
        with open(newpath + '/stats_tests.txt', 'w') as outFile:
            outFile.write('categories: ')
            outFile.write(str(np.shape(response_array)))
            outFile.write('\nmosquitoes: ')
            for i in range(np.shape(response_array)[0]):
                outFile.write(str(np.shape(response_array[i]))+ ' ')
            outFile.write('\nStimulus types (Light or Heat) by plate:\n')
            for plate in onset_types:
                outFile.write(' '.join(plate)+'\n')
            outFile.write('Stimulus indices:\n')
            outFile.write(str(stimulus_onsets))
            outFile.write('\nStimulus names:\n')
            outFile.write(', '.join(stimulus_names) + '\n\n')
            outFile.write(str(friedmanchisquare(*response_array)))
            outFile.write('\nScikit_posthocs Nemenyi Friedman:\n')
            outFile.write(str(ph.posthoc_nemenyi_friedman(response_array.T)))
            outFile.write('\n\n')
            if len(response_array) == 2:
                outFile.write('Mann-Whitney Test\n')
                outFile.write(str(mannwhitneyu(response_array[0],response_array[1])))
            else:
                outFile.write(str(kruskal(*response_array)))
                outFile.write('\nScikit_posthocs Nemenyi\n')
                outFile.write(ph.posthoc_nemenyi(response_array).to_string())
            outFile.write('\n\n')
            # outFile.write('ChiSquared Test\n' + str(np.array(contingency))+'\n')
            # stat, p, dof, expected = chi2_contingency(contingency)
            # outFile.write('test statistic ='+str(stat)+', p ='+str(p)+', dof ='+str(dof)+'\nexpected values:\n'+str(expected))
        if additivity_test:
            #currently this only works for stimuli that are presented simultaneously because it has no way to predict the 
            #expected sum when there are two stimuli that are separate in time (would have to write into the additivity part
            #of the plot_individuals_behaviors method)
            #assumes that the third stimulus is combined and the first two are heat and light
            #leaves out groom because it is not an interesting behavior
            #This also only works (and only makes sense) on a single behavior
            additivity_differences = []
            for b in range(len(self.behaviors)-1):
                current_array= np.array(response_values[b+1])
                #Used for multiple stimuli presented to same animals
                # predicted = np.add(current_array[0], current_array[1])
                #special case for single trial for each animal, but maybe more general if intra-individual variation is high
                predicted = np.add(np.mean(current_array[0]), np.mean(current_array[1]))
                simultaneous_index = 2
                differences = np.subtract(current_array[simultaneous_index], predicted)
                additivity_differences.append(np.divide(differences, np.mean(predicted)))
            plt.figure()
            plt.violinplot(additivity_differences)
            plt.ylim(-3,3)
            plt.boxplot(additivity_differences)
            plt.xticks([1,2,3],labels=self.behaviors[1:])
            plt.hlines(0, xmin=0,xmax=4,linestyles='dashed')
            plt.savefig(newpath + '/add_diff_violin.pdf', format='pdf')        
            plt.close()
            with open(newpath + '/additivity_difference_stats.txt', 'w') as outFile:
                tstatistics = []
                pvalues = []
                outFile.write('1-sample two-sided t-test unadjusted:\n\nbehavior\tt-statistic\tp-value\n')
                for b in range(len(self.behaviors)-1):
                    tstatistic, pvalue = ttest_1samp(additivity_differences[b], 0)
                    outFile.write(self.behaviors[b+1] + '\t' + str(tstatistic) + '\t' + str(pvalue) + '\n')
                    tstatistics.append(tstatistic)
                    pvalues.append(pvalue)
                padjusted = statsmodels.stats.multitest.multipletests(pvalues, method='holm')
                outFile.write('\np-adjusted\n' + np.array2string(padjusted[1]) + '\n\n')
                pvalues = []
                outFile.write('Sign test statsmodels.stats.descriptivestats.sign_test\n')
                for b in range(len(self.behaviors)-1):
                    M, pvalue = statsmodels.stats.descriptivestats.sign_test(additivity_differences[b],0)
                    outFile.write(self.behaviors[b+1] + '\t' + str(M) + '\t' + str(pvalue) + '\n')
                    pvalues.append(pvalue)
                padjusted = statsmodels.stats.multitest.multipletests(pvalues, method='holm')
                outFile.write('\np-adjusted\n' + np.array2string(padjusted[1]) + '\n\n')
            with open(newpath + '/additivity_difference_data.txt', 'w') as outFile:
                outFile.write('\t'.join(self.behaviors[1:])+'\n')
                for i in range(np.shape(additivity_differences)[1]):
                    diffs = [x[i] for x in additivity_differences]
                    print(diffs)
                    outFile.write('\t'.join(map(str, diffs))+'\n')
        #plots for primary behavior of interest
        with open(newpath + '/data_table.txt', 'w') as outFile:
            outFile.write('mosquito\tbehavior_prop\tstimulus\n')
            for i in range(len(response_array[0])):
                for j in range(len(response_array)):
                    if i >= len(response_array[j]):
                        continue
                    outFile.write(str(i) + '\t' + str(response_array[j][i]) + '\t' + stimulus_names[j].replace(' ', '') + '\n')
        # response_differences = np.subtract(np.array(response_values[1]), np.array(response_values[0]))
        # plt.hist(response_differences, bins=20, histtype='step', linewidth=2)
        # plt.title(stimulus_names[1] + ' minus ' + stimulus_names[0] + ' behavior ' + ', '.join(behavior_names))
        # plt.savefig(newpath + '/' +'difference_hist01.pdf', format='pdf')
        # plt.close()
        plt.figure()
        plt.title(' behavior ' + ', '.join(behavior_names) + ' ' + str(startstop) + ' seconds post stimulus onset')
        plt.xlim(0.5, len(response_array)+0.5)
        for i in range(len(response_array[0])):
            xvals, yvals = [], []
            for j in range(len(response_array)):
                if i >= len(response_array[j]):
                    continue
                xvals.append(j+1)
                yvals.append(response_array[j][i])
            plt.plot(xvals,yvals, alpha=0.1, color='black', marker='.')
        xvals, yvals = [], []
        for j in range(len(response_array)):
            xvals.append(j+1)
            yvals.append(median(response_array[j]))
        plt.plot(xvals, yvals, color='red', marker='.', markersize=10.0)
        plt.xticks(xvals, stimulus_names)
        plt.ylim(0,.3)
        plt.savefig(newpath + '/' +'prop_response.pdf', format='pdf')        
        plt.close()
        plt.figure()
        plt.boxplot(np.transpose(response_array))
        plt.title(' behavior ' + ', '.join(behavior_names) + ' ' + str(startstop) + ' seconds post stimulus onset')
        plt.xticks(xvals, stimulus_names)
        plt.savefig(newpath + '/' +'boxplot.pdf', format='pdf')        
        plt.close()
        plt.figure()
        if behavior_names[0] in ['walk3']:
            plt.ylim(0,.8)
        elif behavior_names[0] in ['groom3']:
            plt.ylim(0,1.0)
        else:
            plt.ylim(0,.5)
        plt.violinplot(np.transpose(response_array))
        plt.boxplot(np.transpose(response_array), whis=(5,95), sym='')
        plt.title(' behavior ' + ', '.join(behavior_names) + ' ' + str(startstop) + ' seconds post stimulus onset')
        plt.xticks(xvals, stimulus_names)
        plt.savefig(newpath + '/' +'violin_box_whis595.pdf', format='pdf')        
        plt.close()
        # for s, stim_type in enumerate(stimulus_names):
        #     plt.figure()
        #     nbins = (seconds + 60)*.2
        #     bins_sequence = np.linspace(-60, seconds, nbins)
        #     plt.ylim(0,2.5)
        #     counts, bin_edges = np.histogram(onsets_hist[s], bins=bins_sequence)
        #     plt.plot(bin_edges[1:], counts/len(response_values[0]), color = 'black')
        #     plt.savefig(newpath + '/' +'onsets_hist' + stim_type +'.pdf', format='pdf')        
        #     plt.close()

    def behaviorCorrelation(self, twobehaviors, twowindows, stimulus_onsets, stimulus_names):
        #computes correlation between behavior one and behavior two in the two windows in seconds
        #format is ['fly2','probe2'] and [[-60,0],[0,300]] for example
        #the stimulus onsets and names are of the same structure as in statistics_test
        newpath = self.opto_folder + '/graphs/behavior_correlations/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        frame_ranges = []
        onset_types = []
        [onset_types.append([]) for x in range(len(self.behavior_data))]
        print(stimulus_onsets)
        for t in range(len(self.behavior_data)):
            frame_ranges.append([]) 
            for stim_type in stimulus_onsets:
                stim_type_range = []
                for onset_index in stim_type[t]:
                    onset = self.onsets[t][onset_index]
                    onset_types[t].append(self.stimulus_type[t][onset_index])
                    stim_type_range.append([[onset+twowindows[0][0]*self.frame_rate,onset+twowindows[0][1]*self.frame_rate],[onset+twowindows[1][0]*self.frame_rate,onset+twowindows[1][1]*self.frame_rate]])
                frame_ranges[t].append(stim_type_range)
        print(np.shape(frame_ranges))  #organized by trial, stimulus, rep, start/stop
        response_values = []
        [response_values.append([[],[]]) for x in stimulus_onsets]
        for t in range(len(self.behavior_data)):
            for m in range(len(self.behavior_data[t][twobehaviors[0]]['t0s'])):
                for s, stim_type in enumerate(frame_ranges[t]):
                    values_type = [[],[]] #first list is x axis and second is y axis
                    for stim_range in stim_type: #stim range has structure [[a,b],[c,d]] which matches twowindows for correlation
                        for a, axis in enumerate(twobehaviors):
                            temp_onsets = [x for x in self.behavior_data[t][twobehaviors[a]]['t0s'][m] if x >= stim_range[a][0] and x < stim_range[a][1]]
                            temp_offsets = [x for x in self.behavior_data[t][twobehaviors[a]]['t1s'][m] if x >= stim_range[a][0] and x < stim_range[a][1]]
                            final_onsets, final_offsets = fixBehaviorRange(temp_onsets, temp_offsets, stim_range[a][0], stim_range[a][1])
                            if len(final_onsets) != len(final_offsets):
                                i=0
                                while i < len(final_onsets) and i < len(final_offsets):
                                    if final_offsets[i] < final_onsets[i]:
                                        print('bout starts before it ends', final_offsets[i], final_onsets[i])
                                        if len(final_onsets) > len(final_offsets):
                                            del final_onsets[i]
                                        else:
                                            del final_offsets[i]
                                    i += 1
                                    # if final_offsets[i] > final_onsets[i+1]:
                                    #     print('next bout starts before current ends, onset, offset, next onset:', final_onsets[i], final_offsets[i], final_onsets[i+1])
                            if len(final_onsets) + len(final_offsets) == 0:
                                values_type[a].append(0.0)
                            else:
                                values_type[a].append(sum(np.subtract(np.array(final_offsets), np.array(final_onsets)))/(stim_range[a][1]-stim_range[a][0]))
                    if len(values_type[0]) > 0:
                        for a, axis in enumerate(twobehaviors):
                            # if you want to average over repeats: response_values[s][a].append(sum(values_type[a])/len(values_type[a]))
                            #otherwise this is treating each stimulus presentation separately
                            response_values[s][a].extend(values_type[a])
        print(np.shape(response_values))
        print('number of responses recorded for first stimulus:',len(response_values[0][0]))
        for s, stim_name in enumerate(stimulus_names):
            #first make scatterplot
            plt.figure()
            plt.scatter(response_values[s][0], response_values[s][1], alpha=0.2)
            plt.title(stim_name + '\n' + str(spearmanr(response_values[s][0], response_values[s][1])))
            plt.xlabel(twobehaviors[0] + str(twowindows[0][0]) + ' to ' + str(twowindows[0][1]) + ' seconds')
            plt.ylabel(twobehaviors[1] + str(twowindows[1][0]) + ' to ' + str(twowindows[1][1]) + ' seconds')
            plt.savefig(newpath + '/' + str(s) + '_' + twobehaviors[0] + str(twowindows[0][1]-twowindows[0][0]) + twobehaviors[1] + str(twowindows[1][1]-twowindows[1][0])+'.pdf', format='pdf')
            plt.close()

    def stimulusGraph(self, trial_names, min_seconds, pos_seconds, skip_close_stimuli = True):
	#graph stimuli in each trial, with a window of min_seconds and pos_seconds around it
    #if skip_close_stimuli is true, it excludes subsequent stimuli that are in graph window from next graph
        newpath = self.opto_folder + '/graphs/stimuli/' 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        for t in range(len(self.behavior_data)):
            print(trial_names[t])
            plt.figure()
            
            if skip_close_stimuli:
                num_stims = 1
                for s in range(1, len(self.onsets[t])):
                    if (self.onsets[t][s]-self.onsets[t][s-1])/self.frame_rate > pos_seconds:
                        num_stims += 1
            else:
                num_stims = len(self.onsets[t])+1
            plot_count = 0
            last_stimulus_index = 0
            for s in range(len(self.onsets[t])):
                if skip_close_stimuli:
                    if (self.onsets[t][s]-last_stimulus_index)/self.frame_rate < pos_seconds and s>0:
                        last_stimulus_index = self.onsets[t][s]
                        continue
                last_stimulus_index = self.onsets[t][s]
                plot_count += 1
                start_graph = self.onsets[t][s] - min_seconds*self.frame_rate
                stop_graph = self.onsets[t][s] + pos_seconds*self.frame_rate
                if stop_graph > len(self.frame_data[t]):
                    print('stimulus',str(self.onsets[t][s]),'ends after movie')
                    break
                plt.subplot(math.ceil(num_stims/2), 2, plot_count)
                xvalues = np.linspace(-min_seconds, pos_seconds, (min_seconds+pos_seconds)*10)
                yvalues = []
                for y in range(start_graph, stop_graph, 3):
                    yvalues.append(self.frame_data[t][y][2])
                plt.ylim(20,40)
                plt.plot(xvalues, yvalues, color='orange')
                plt.text(-min_seconds, 30, 'onset '+str(s))
                for s2 in range(len(self.onsets[t])):
                    if self.onsets[t][s2] > start_graph and self.onsets[t][s2] < stop_graph and self.stimulus_type[t][s2] == 'L':
                        start_light = (self.onsets[t][s2]-self.onsets[t][s])/self.frame_rate
                        stop_light = (self.offsets[t][s2]-self.onsets[t][s])/self.frame_rate
                        plt.fill(np.array([start_light,start_light,stop_light,stop_light]),np.array([20,35,35,20]), color='red', alpha=0.3, linewidth=0.0)
            plt.savefig(newpath + '/' + trial_names[t] +'.pdf', format='pdf')
            plt.close()

    def prepost_velocity(self, test_name, stimulus_onsets, stimulus_names, prepost_seconds, plate_corners, plate_rows, plate_cols):
        #prepost values are negative for before stimulus and positive for after stimulus
        #this averages over multiple presentations of the stimulus to an individual mosquito
        newpath = self.opto_folder + '/statistics/velocity/' + test_name + '/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        frame_step = round(velocity_step_ctrax/1000*self.frame_rate) #frame step is distance between time points for calculating velocity
        for st, stim_type in enumerate(stimulus_onsets): #iterate through stimulus types
            #organized by trial
            xys = [] #xys organized by trial,  then  *numpyarray* wells, then x,y for each frame
            velocities = [] #velocityies organized as list by stimulus
            for t in range(len(self.behavior_data)):
                allStarts = [item for sublist in self.behavior_data[t][self.behaviors[0]]['tStarts'] for item in sublist]
                start_frame = min(allStarts)
                allEnds = [item for sublist in self.behavior_data[t][self.behaviors[0]]['tEnds'] for item in sublist]
                end_frame = max(allEnds)
                wells = calculate_grid(plate_corners, plate_rows, plate_cols)
                
                # end_frame = max(self.behavior_data[0][self.behaviors[0]]['tEnds'])
                # print(self.behavior_data[0][self.behaviors[0]]['tEnds'])
                # print(end_frame)
                xys.append(np.empty((len(wells), end_frame, 2)))

                for root, dirs, files in os.walk(self.opto_folder + '/processed'):
                    trial_dirs = natural_sort([d for d in dirs if d[:len(self.trial_names[t])] == self.trial_names[t]])
                    print(trial_dirs)
                    break            
                #iterate through each trial chunk, adding mosquito tracks
                prev_chunks_total = 0
                for trial_chunk in trial_dirs:
                    velocity_mat = loadmat(os.path.join(self.opto_folder, 'processed', trial_chunk, 'trx.mat'))
                    for track in velocity_mat['trx']:
                        f = track.firstframe - 1
                        if isinstance(track.x, float):
                            continue
                        for xloc, yloc in zip(track.x, track.y):
                            for w, well in enumerate(wells):
                                if xloc > well[0] and xloc < well[1] and yloc > well[2] and yloc < well[3]:
                                    # if prev_chunks_total+f >= np.shape(xys[t])[1]:
                                    #     continue
                                    xys[t][w,prev_chunks_total+f,0] = xloc
                                    xys[t][w,prev_chunks_total+f,1] = yloc
                            f += 1
                    prev_chunks_total += len(velocity_mat['timestamps'])
                # print(xys[0,:10])
                # print(xys[1,:10])
                #calculate velocity
                velocities.append(np.empty((len(wells), end_frame)))
                velocities[t][:] = np.NaN
                for m in range(len(xys[t])):
                    for f in range(end_frame):
                        try:
                            first_frame = round(f-frame_step/2)
                            last_frame = round(f+frame_step/2)
                            x1, x2, y1, y2 = xys[t][m,first_frame,0], xys[t][m,last_frame,0], xys[t][m,first_frame,1], xys[t][m,last_frame,1]
                            velocities[t][m,f] = np.sqrt((x2-x1)**2 + (y2-y1)**2)*self.well_size_mm/self.well_size_pixels*1000/velocity_step_ctrax
                        except IndexError:
                            continue
            prepost_vels = [] #list of each mosquito [prevel, postvel]
            for t, trial in enumerate(velocities):
                for m in range(np.shape(trial)[0]):
                    prevel = 0.0
                    postvel = 0.0
                    for s in stim_type[t]:
                        onset_index = self.onsets[t][s]
                        prevel += np.mean(trial[m,onset_index+prepost_seconds[0]*self.frame_rate:onset_index])
                        postvel += np.mean(trial[m,onset_index:onset_index+prepost_seconds[1]*self.frame_rate])
                    prevel = prevel/len(stim_type)
                    postvel = postvel/len(stim_type)
                    prepost_vels.append([prevel, postvel])
            prepost_vels = np.array(prepost_vels)
            plt.figure()
            plt.title('Velocity pre ' + str(prepost_seconds[0]) + ' and post ' + str(prepost_seconds[1])  + ' seconds around stimulus onset')
            plt.xlim(0.5, 2.5)
            plt.ylim(0,2.5)
            for i in range(np.shape(prepost_vels)[0]):
                xvals, yvals = [], []
                for j in range(np.shape(prepost_vels)[1]):
                    xvals.append(j+1)
                    yvals.append(prepost_vels[i,j])
                plt.plot(xvals,yvals, alpha=0.1, color='black', marker='.')
            xvals, yvals = [], []
            for j in range(np.shape(prepost_vels)[1]):
                xvals.append(j+1)
                yvals.append(median(prepost_vels[:,j]))
            plt.plot(xvals, yvals, color='red', marker='.', markersize=10.0)
            plt.xticks(xvals, ['pre','post'])
            plt.violinplot(prepost_vels)
            plt.savefig(newpath + '/' +'violin_prepost_'+ stimulus_names[st] + '.pdf', format='pdf')        
            plt.close()
            with open(newpath + '/prepost_test_'+ stimulus_names[st] +'.txt', 'w') as outFile:
                outFile.write(' mosquitoes: ')
                outFile.write(str(np.shape(prepost_vels)[0]) + '\n')
                outFile.write('Stimulus indices:\n')
                outFile.write(str(stim_type) + '\n\n')
                outFile.write(str(wilcoxon(prepost_vels[:,1],prepost_vels[:,0])))
                outFile.write('\n\nData\n\nPre\tPost\n')
                for i in range(np.shape(prepost_vels)[0]):
                    outFile.write(str(prepost_vels[i,0]) + '\t' + str(prepost_vels[i,1]) + '\n')
                    
            
    def output_cluster_features(self, stim_indices, prepost_seconds, window_size_seconds, plate_corners):
        #outputs features within windows, excluding windows outside [pre, post] around stimuli
        newpath = self.opto_folder + '/cluster_features/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        frame_step = round(velocity_step_ctrax/1000*self.frame_rate) #frame step is distance between time points for calculating velocity
        window_step_frames = window_step_size*self.frame_rate # time between taking windows
        for t in range(len(self.behavior_data)):
            allStarts = [item for sublist in self.behavior_data[t][self.behaviors[0]]['tStarts'] for item in sublist]
            start_frame = min(allStarts)
            allEnds = [item for sublist in self.behavior_data[t][self.behaviors[0]]['tEnds'] for item in sublist]
            end_frame = max(allEnds)
            gap_list = [] #organized by mosquito
            behavior_hist = [] #organized by mosquito and then by behavior
            for m in range(len(self.behavior_data[t][self.behaviors[0]]['t0s'])):
                behavior_hist.append([])
                # [behavior_hist[m].append([]) for behavior in behaviors]
                gap_list.append([])
                #find gaps
                gap_starts = []
                gap_ends = []
                tStarts = sorted(self.behavior_data[t][self.behaviors[0]]['tStarts'][m])
                tEnds = sorted(self.behavior_data[t][self.behaviors[0]]['tEnds'][m])
                if len(tStarts) + len(tEnds) == 0:
                    continue
                for g in range(1, len(tStarts)):
                    prev_end = tEnds[g-1]
                    curr_start = tStarts[g]
                    if curr_start > prev_end:
                        gap_starts.append(prev_end)
                        gap_ends.append(curr_start)
                #if movie ends before end of graph
                if tEnds[-1] >= start_frame and tEnds[-1] < end_frame:
                    gap_starts.append(tEnds[-1])
                gap_onsets, gap_offsets = fixBehaviorRange(sorted(gap_starts), sorted(gap_ends), start_frame, end_frame)
                for b, behavior_name in enumerate(self.behaviors):
                    # print(behavior_name)
                    #calculate behavior onsets and offsets
                    temp_onsets = [x for x in self.behavior_data[t][behavior_name]['t0s'][m] if  x >= start_frame and x < end_frame]
                    temp_offsets = [x for x in self.behavior_data[t][behavior_name]['t1s'][m] if  x >= start_frame and x < end_frame]
                    behavior_onsets, behavior_offsets = fixBehaviorRange(sorted(temp_onsets), sorted(temp_offsets), start_frame, end_frame)
                    #add 1s and 0s  of whether mosquito is exhibiting behavior or not at each timepoint
                    if b==0:
                        if len(gap_onsets) + len(gap_offsets) == 0:
                            [gap_list[m].append(0) for x in range(end_frame - start_frame)]
                        else:
                            [gap_list[m].append(0) for x in range(gap_onsets[0]-start_frame)]
                            [gap_list[m].append(1) for x in range(gap_offsets[0]-gap_onsets[0])]
                            for o in range(1, len(gap_onsets)):
                                nogap = gap_onsets[o] - gap_offsets[o-1]
                                if nogap > 0:
                                    [gap_list[m].append(0) for x in range(nogap)]
                                    gap_length = gap_offsets[o]-gap_onsets[o]
                                else:
                                    gap_length = gap_offsets[o]-gap_onsets[o]+nogap
                                    print('gap bout starts before previous one ends: start prev stop',gap_onsets[o], gap_offsets[o-1])
                                [gap_list[m].append(1) for x in range(gap_length)]
                            [gap_list[m].append(0) for x in range(end_frame - gap_offsets[-1])]
                    behavior_list = []
                    if len(behavior_onsets) + len(behavior_offsets) == 0:
                        [behavior_list.append(0) for x in range(end_frame - start_frame)]
                    else:
                        [behavior_list.append(0) for x in range(behavior_onsets[0]-start_frame)]
                        [behavior_list.append(1) for x in range(behavior_offsets[0]-behavior_onsets[0])]
                        for o in range(1, len(behavior_onsets)):
                            gap = behavior_onsets[o] - behavior_offsets[o-1]
                            if gap > 0:
                                [behavior_list.append(0) for x in range(gap)]
                                bout_length = behavior_offsets[o]-behavior_onsets[o]
                            else:
                                bout_length = behavior_offsets[o]-behavior_onsets[o]+gap
                                print('behavior bout starts before previous one ends: start prev stop',behavior_onsets[o], behavior_offsets[o-1])
                            [behavior_list.append(1) for x in range(bout_length)]
                        [behavior_list.append(0) for x in range(end_frame - behavior_offsets[-1])]
                    behavior_hist[m].append(behavior_list)
            #calculate and assign velocity
            #units are now in mm/s
            #print(self.trial_names[t])           
            # print(type(velocity_mat))
            # print(velocity_mat.keys())
            # print(type(velocity_mat['trx']))
            # print(len(velocity_mat['trx']))
            # print(type(velocity_mat['trx'][0]))
            # print(vars(velocity_mat['trx'][0]))
            # print(type(velocity_mat['trx'][0].x))
            # print(np.size(velocity_mat['trx'][0].x))
            #first assign xy values to wells
            plate_rows, plate_cols = 3,5
            wells = calculate_grid(plate_corners, plate_rows, plate_cols)
            
            # end_frame = max(self.behavior_data[0][self.behaviors[0]]['tEnds'])
            # print(self.behavior_data[0][self.behaviors[0]]['tEnds'])
            # print(end_frame)
            xys = np.empty((len(wells), end_frame, 2))

            for root, dirs, files in os.walk(self.opto_folder + '/processed'):
                trial_dirs = natural_sort([d for d in dirs if d[:len(self.trial_names[t])] == self.trial_names[t]])
                print(trial_dirs)
                break            
            #iterate through each trial chunk, adding mosquito tracks
            prev_chunks_total = 0
            for trial_chunk in trial_dirs:
                velocity_mat = loadmat(os.path.join(self.opto_folder, 'processed', trial_chunk, 'trx.mat'))
                for track in velocity_mat['trx']:
                    f = track.firstframe - 1
                    if isinstance(track.x, float):
                        continue
                    for xloc, yloc in zip(track.x, track.y):
                        for w, well in enumerate(wells):
                            if xloc > well[0] and xloc < well[1] and yloc > well[2] and yloc < well[3]:
                                xys[w,prev_chunks_total+f,0] = xloc
                                xys[w,prev_chunks_total+f,1] = yloc
                        f += 1
                prev_chunks_total += len(velocity_mat['timestamps'])
            # print(xys[0,:10])
            # print(xys[1,:10])
            #calculate velocity
            velocities = np.empty((len(wells), end_frame))
            velocities[:] = np.NaN
            for m in range(len(xys)):
                for f in range(end_frame):
                    try:
                        first_frame = round(f-frame_step/2)
                        last_frame = round(f+frame_step/2)
                        x1, x2, y1, y2 = xys[m,first_frame,0], xys[m,last_frame,0], xys[m,first_frame,1], xys[m,last_frame,1]
                        velocities[m,f] = np.sqrt((x2-x1)**2 + (y2-y1)**2)*self.well_size_mm/self.well_size_pixels*1000/velocity_step_ctrax
                    except IndexError:
                        continue
            #generate additional vectors for parameter calculations
            #this is designed only for 4 behaviors
            behavior_heirarchy = np.zeros((len(behavior_hist), len(behavior_hist[0][0])))
            for m in range(len(behavior_hist)):
                for b in range(len(self.behaviors)):
                    if m >= len(behavior_heirarchy) or m >= len(behavior_hist) or b >= len(behavior_hist[m]):
                        print(m,b,np.shape(behavior_heirarchy),np.shape(behavior_hist))
                    np.place(behavior_heirarchy[m], np.asarray(behavior_hist[m][b])==1,b+1)
            #transitions between all behaviors
            transitions = np.zeros((len(behavior_hist), len(behavior_hist[0][0])))
            for m in range(len(behavior_hist)):
                transitions[m] = behavior_heirarchy[m]*6 - (np.roll(behavior_heirarchy[m],-1) + 1)
                np.place(transitions[m], np.ediff1d(behavior_heirarchy[m], to_end=0)==0, 0)
            #probexwalk will be 2 if probe-only, 1 if both, -1 if walk-only
            probexwalk = np.zeros((len(behavior_hist), len(behavior_hist[0][0])))
            for m in range(len(behavior_hist)):
                probexwalk[m] = np.asarray(behavior_hist[m][2])*2 - np.asarray(behavior_hist[m][1])
            #iterate through windows to calculate parameters
            output_parameters = []
            window_size_frames = window_size_seconds * self.frame_rate

            for b, behavior in enumerate(self.behaviors):
                if 'fly' in behavior:
                    fly_index = b
                    break
            else:
                print('no flying behavior was included in this optothermo object')
                
            for m in range(len(behavior_hist)):
                for st, stim_type in enumerate(stim_indices): 
                    for i in stim_type[t]:
                        if len(prepost_seconds) == 2:
                            start_windows = int(self.onsets[t][i] - prepost_seconds[0]*self.frame_rate/self.velocity_step_frames)
                            end_windows = int(self.onsets[t][i] + prepost_seconds[1]*self.frame_rate/self.velocity_step_frames + 1)
                        else:
                            start_windows = 0
                            end_windows = 10000000 #movies are not expected to be longer than 10 million frames
                        # print('frames ',start_windows, end_windows, 'stimulus', st)
                        for f in range(round(window_size_frames/2), end_frame, window_step_frames): #window_step_frames or window_size_frames 
                            #skip windows that are not near stimuli of interest
                            if f < start_windows or f > end_windows:
                                continue
                            #calculate parameters for each mosquito for each frame
                            current_parameters = [m, f]
                            first_frame = f-round(window_size_frames/2)
                            last_frame = f+round(window_size_frames/2)
                            if last_frame > len(velocities[m]): #avoid partial frames at end of recording
                                continue
                            denominator = gap_list[m][first_frame:last_frame].count(0) #number of frames that are not gap
                            if denominator == 0:
                                denominator = 1
                            current_parameters.append(np.mean(velocities[m,first_frame:last_frame])) #mean velocity over frame
                            # print(gap_list[m][first_frame:last_frame])
                            for b in range(4):
                                # print(behavior_hist[m][b])
                                current_parameters.append(behavior_hist[m][b][first_frame:last_frame].count(1)/denominator)
                            current_parameters.append(np.count_nonzero(behavior_heirarchy[m,first_frame:last_frame] == 0)/denominator)
                            dominant_behavior = 0 #default is nothing
                            if current_parameters[3] >= 0.3: #grooming
                                dominant_behavior = 1
                            if current_parameters[4] >= 0.2: #walking
                                dominant_behavior = 2
                            if current_parameters[5] >= 0.2: #probing
                                dominant_behavior = 3
                            if current_parameters[6] >= 0.04: #flying
                                dominant_behavior = 4            
                            new_current_parameters = current_parameters[:2]
                            new_current_parameters.append(dominant_behavior)
                            new_current_parameters.extend(current_parameters[2:])
                            current_parameters=new_current_parameters
                            for pw in [-1,1,2]:
                                current_parameters.append(np.count_nonzero(probexwalk[m][first_frame:last_frame] == pw)/denominator)
                            #calculate velocities while in each behavior
                            fastprobe = False
                            for b in range(4):
                                if 1 not in behavior_hist[m][b][first_frame:last_frame]:
                                    behavior_speed = 0.0
                                    current_parameters.append(behavior_speed)
                                    continue
                                straightmsk = (np.asarray(behavior_hist[m][b][first_frame:last_frame])-1)*-1
                                expandedmsk = np.add(straightmsk, np.roll(straightmsk, int(frame_step/2)))
                                msk = np.add(expandedmsk, np.roll(straightmsk, int(-frame_step/2)))
                                if len(msk) != len(velocities[m,first_frame:last_frame]):
                                    print(first_frame, last_frame, len(behavior_hist[m][b]), len(velocities[m]))
                                #exclude frames of behavior b that overlap with flying frames to exclude spurious velocities
                                if 'fly' not in self.behaviors[b]:
                                    msk = np.add(msk, np.asarray(behavior_hist[m][fly_index][first_frame:last_frame]))
                                behavior_speed = np.mean(np.ma.masked_array(velocities[m,first_frame:last_frame], mask = msk))
                                if np.ma.is_masked(behavior_speed):
                                    behavior_speed = 0.0
                                if self.behaviors[b] == 'probe5' and behavior_speed > 10.0:
                                    fastprobe = True
                                if fastprobe:
                                    print(straightmsk)
                                    print(msk)
                                    print(np.asarray(behavior_hist[m][fly_index][first_frame:last_frame]))
                                    print(velocities[m,first_frame:last_frame])
                                    print(self.behaviors[b], behavior_speed)
                                current_parameters.append(behavior_speed)
                            #calculate number of bouts and transition probabilities to other behaviors
                            all_bouts = []
                            for behavtr in [[-5,-4,-3,-2],[1,2,3,5],[7,8,10,11],[13,15,16,17],[20,21,22,23]]:
                                bouts = 0
                                trcounts = []
                                for tr in behavtr:
                                    trcounts.append(np.count_nonzero(transitions[m][first_frame:last_frame] == tr))
                                    bouts += trcounts[-1]
                                if bouts == 0:
                                    current_parameters.extend([0,0,0,0])
                                else:
                                    for tr in trcounts:
                                        current_parameters.append(tr/bouts)
                                all_bouts.append(bouts)
                            current_parameters.extend(all_bouts)
                            if current_parameters[8] == 1.0 or current_parameters[2] == 0:
                                continue
                            output_parameters.append(current_parameters)
            with open(newpath + self.trial_names[t] +'_' + str(window_size_seconds) + '.txt', 'w') as outFile:
                outFile.write('mosquito\tframe\tdom_behav\tmean_vel\t')
                outFile.write('\t'.join(self.behaviors)+'\t')
                outFile.write('nobehav\twalk_only\tprobe_walk\tprobe_only\tgroomvel\twalkvel\tprobevel\tflyvel\t')
                outFile.write('none>fly\tnone>probe\tnone>walk\tnone>groom\tgroom>fly\tgroom>probe\tgroom>walk\tgroom>none\twalk>fly\twalk>probe\twalk>groom\twalk>none\tprobe>fly\tprobe>walk\tprobe>groom\tprobe>none\tfly>probe\tfly>walk\tfly>groom\tfly>none\t')
                outFile.write('none_bouts\tgroom_bouts\twalk_bouts\tprobe_bouts\tfly_bouts')
                outFile.write('\n')
                for frame in output_parameters:
                    outFile.write('\t'.join(map(str, frame)))
                    outFile.write('\n')
            # print(output_parameters [:30])

    def output_tSNE(self, window_size_seconds, perplex=30, n_jobs=4, n_iter=1000, learning_rate=200):
        #this method reads in files with features calculated over windows and outputs the tSNE axes into the file
        window_data = []
        window_array = []
        for t, trial_name in zip(range(len(self.behavior_data)), self.trial_names):
            window_data.append([])
            with open(self.opto_folder + '/cluster_features/'+trial_name +'_' + str(window_size_seconds) + '.txt', 'r') as inFile:
                all_lines = inFile.readlines()
                line1 = all_lines[0].split()
                for column in line1:
                    if column[:4] == 'tSNE':
                        tSNEsplit = column.split('_')
                        if int(tSNEsplit[2].strip('sper')) == perplex:
                            print('tSNE already calculated for this vlue')
                            return
                start_column = 3
                number_parameters = 38
                print('start column ' + line1[start_column] + ' last column ' + line1[start_column+number_parameters-1])
                for line in all_lines[1:]:
                    window_array.append(line.split()[:start_column+number_parameters])
                    window_data[t].append(line.split())
        # print(window_data[0])
        window_array = np.asarray(window_array)
        print(np.shape((window_array[:,start_column:])))
        # print(window_array[:2,start_column:])
        time_start = time.time()
        behavior_tsne = TSNE(perplexity=perplex, n_jobs=n_jobs, n_iter=n_iter, learning_rate=learning_rate, verbose=1).fit_transform(window_array[:,start_column:])
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
        i = 0
        for t, trial_name in zip(range(len(self.behavior_data)), self.trial_names):
            with open(self.opto_folder + '/cluster_features/'+trial_name +'_' + str(window_size_seconds) + '.txt', 'w') as outFile:
                outFile.write('\t'.join(line1[:]))
                outFile.write('\ttSNE1_'+str(window_size_seconds)+'s_'+str(perplex)+'per\ttSNE2_'+str(window_size_seconds)+'s_'+str(perplex)+'per\n')
                for window in window_data[t]:
                    outFile.write('\t'.join(map(str, window)))
                    outFile.write('\t'+str(behavior_tsne[i,0])+'\t'+str(behavior_tsne[i,1]) + '\n')
                    i += 1
        
    def graph_tSNE(self, experiment_name, window_size_seconds, perplex, sample_size = 1000, dom_behav_graph=True, behav_gradient=True, other_gradient=True, any_behav=True):
        newpath = self.opto_folder + '/graphs/tSNE/' + experiment_name +'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        #read in feature data
        window_array = []
        for t, trial_name in zip(range(len(self.behavior_data)), self.trial_names):
            with open(self.opto_folder + '/cluster_features/'+trial_name +'_' + str(window_size_seconds) + '.txt', 'r') as inFile:
                all_lines = inFile.readlines()
                line1 = all_lines[0].split()
                cn=-1 #colomn name
                end=-1
                if line1[-1][:4] != 'tSNE': #search tSNE values from the right end
                    print('tSNE not calculated for this trial!')
                while line1[cn][:4] =='tSNE':                        
                    if line1[cn].split('_')[-1].strip('sper') == str(perplex): 
                        x=cn-1
                        y=cn   
                        end = cn-1 
                        break
                    cn+=-2
                if end == -1:
                    print('tSNE not calculated for this perplexity!')                
                for line in all_lines[1:]:
                    window_array.append(line.split())
        window_array = np.asarray(window_array, dtype=float)
        print(window_array.dtype)
        print(np.shape(window_array))
        #select subset of points
        if sample_size > np.shape(window_array)[0]:
            subset_mask = np.array([True] * np.shape(window_array)[0])
        else:
            subset_mask = np.array([True] * sample_size + [False ] * (np.shape(window_array)[0] - sample_size))
            np.random.shuffle(subset_mask)
        if dom_behav_graph:
            plt.figure()
            cols = []
            behav_cols = ['#EEEEEE','#C0BA9C','#863810','#FA0F0C','#78C0A7','#FFFFFF']
            for i in window_array[:,2].astype(int):              
                cols.append(behav_cols[i])
            fig, ax = plt.subplots()            
            ax.scatter(window_array[:,x][subset_mask],window_array[:,y][subset_mask], marker='.', s=20, linewidths=0.0, c=np.array(cols)[subset_mask])
            # plt.colorbar(label='Average velocity')
            plt.title('Dominant Behavior(perp='+line1[cn].split('_')[-1]+')')
            plt.xlabel('tSNE1')
            plt.ylabel('tSNE2')
            #plt.yticks(np.arange(min(window_array[:,-1]),max(window_array[:,-1]),5))      
            #start, end = ax.get_xlim() 
            ss = 10 #set step size for x,y axis ticks
            start_x = roundup(int(ax.get_xlim()[0]))
            end_x = roundup(int(ax.get_xlim()[1]))
            ax.xaxis.set_ticks(np.arange(start_x, end_x, ss))
            start_y = roundup(int(ax.get_ylim()[0]))
            end_y = roundup(int(ax.get_ylim()[1]))
            ax.yaxis.set_ticks(np.arange(start_y, end_y, ss))
            plt.grid(alpha=0.35, linestyle='dashed', linewidth=0.5)
            patch=[] #set the legend
            for i in range(1,5):    
                patch.append(mpatches.Patch(color=behav_cols[i], label=self.behaviors[i-1][:-1]))
            fig.legend(handles=patch,labelcolor=behav_cols[1:5], bbox_to_anchor=(0.48, 0.3, 0.5, 0.5))#plot legend to figure instead of subplot(plt.legend)
            plt.subplots_adjust(left=0.15, right=0.8, wspace=0, hspace=0)
            plt.savefig(self.opto_folder + '/graphs/tSNE/' + experiment_name + '/dombehav_' + str(window_size_seconds) + 'sec_' + str(perplex) + 'per.pdf', format='pdf')
            plt.close()
        behavior_none_names = list(self.behaviors)
        behavior_none_names.append('none')
        behavior_none_colors = list(self.behavior_colors)
        behavior_none_colors.append('#EEEEEE')
        if behav_gradient:
            for b, behavior in enumerate(behavior_none_names):
                plt.figure()
                behav_cols_rgb = [[192,186,156],[134,56,16],[250,15,12],[120,192,167],[0,0,0]]
                N = 256
                vals = np.ones((N, 4))
                vals[:, 0] = np.linspace(238/256, behav_cols_rgb[b][0]/256, N)
                vals[:, 1] = np.linspace(238/256, behav_cols_rgb[b][1]/256, N)
                vals[:, 2] = np.linspace(238/256, behav_cols_rgb[b][2]/256, N)
                newcmp = matplotlib.colors.ListedColormap(vals)
                plt.scatter(window_array[:,x][subset_mask],window_array[:,y][subset_mask], marker='.', linewidths=0.0, c=window_array[:,b+4][subset_mask], cmap=newcmp)
                plt.colorbar(label='Proportion ' + behavior)
                plt.savefig(self.opto_folder + '/graphs/tSNE/' + experiment_name + '/grad' + behavior + '_' + str(window_size_seconds) + 'sec_' + str(perplex) + 'per.pdf', format='pdf')
                plt.close()
            opto_folder=self.opto_folder
            
            for b, behavior in enumerate(['walkonly','probewalk','probeonly']):
                plt.figure()
                behav_cols_rgb = [[134,56,16],[192,35,14],[250,15,12]]
                N = 256
                vals = np.ones((N, 4))
                vals[:, 0] = np.linspace(238/256, behav_cols_rgb[b][0]/256, N)
                vals[:, 1] = np.linspace(238/256, behav_cols_rgb[b][1]/256, N)
                vals[:, 2] = np.linspace(238/256, behav_cols_rgb[b][2]/256, N)
                newcmp = matplotlib.colors.ListedColormap(vals)
                plt.scatter(window_array[:,x][subset_mask].astype(np.float),window_array[:,y][subset_mask].astype(np.float), marker='.', linewidths=0.0, c=window_array[:,b+9][subset_mask], cmap=newcmp)
                plt.colorbar(label='Proportion ' + behavior)
                plt.savefig(opto_folder + '/graphs/tSNE/' + experiment_name + '/grad' + behavior + '_' + str(window_size_seconds) + 'sec_' + str(perplex) + 'per.pdf', format='pdf')
                plt.close()    
        if other_gradient:
            plt.figure()
            print(window_array[:,3])
            plt.scatter(window_array[:,x][subset_mask],window_array[:,y][subset_mask], marker='.', linewidths=0.0, c=window_array[:,3][subset_mask], cmap=plt.get_cmap('BuPu'))
            plt.colorbar(label='Average velocity')
            plt.savefig(opto_folder + '/graphs/tSNE/' + experiment_name + '/vel_' + str(window_size_seconds) + 'sec_' + str(perplex) + 'per.pdf', format='pdf')
            plt.close()    
            plt.figure()
            plt.scatter(window_array[:,x][subset_mask],window_array[:,y][subset_mask], marker='.', linewidths=0.0, c=window_array[:,14][subset_mask], cmap=plt.get_cmap('BuPu'))
            plt.colorbar(label='Average probing velocity')
            plt.savefig(opto_folder + '/graphs/tSNE/' + experiment_name + '/velprobe_' + str(window_size_seconds) + 'sec_' + str(perplex) + 'per.pdf', format='pdf')
            plt.close()  
            plt.figure()
            plt.scatter(window_array[:,x][subset_mask],window_array[:,y][subset_mask], marker='.', linewidths=0.0, c=window_array[:,13][subset_mask], cmap=plt.get_cmap('BuPu'))
            plt.colorbar(label='Average probing velocity')
            plt.savefig(opto_folder + '/graphs/tSNE/' + experiment_name + '/velwalk_' + str(window_size_seconds) + 'sec_' + str(perplex) + 'per.pdf', format='pdf')
            plt.close()    
            plt.figure()
            plt.scatter(window_array[:,x][subset_mask],window_array[:,y][subset_mask], marker='.', linewidths=0.0, c=window_array[:,1][subset_mask], cmap=plt.get_cmap('hsv'))#'cet_CET_R3'
            plt.colorbar(label='Frame number')
            plt.savefig(opto_folder + '/graphs/tSNE/' + experiment_name + '/time_' + str(window_size_seconds) + 'sec_' + str(perplex) + 'per.pdf', format='pdf')
            plt.close()
        if any_behav:
            for b, behavior in enumerate(behavior_none_names):
                plt.figure()
                cols = []
                for i in window_array[:,2].astype(int):
                    cols.append(behavior_none_colors[i])
                plt.scatter(window_array[:,x][subset_mask],window_array[:,y][subset_mask], marker='.', linewidths=0.0, c='#999999')
                behave_mask = (window_array[:,b+4].astype(float) > 0)
                plt.scatter(window_array[:,-2][behave_mask & subset_mask],window_array[:,-1][behave_mask & subset_mask], marker='.', linewidths=0.0, c=behavior_none_colors[b])
                # plt.colorbar(label='Average velocity')
                plt.savefig(opto_folder + '/graphs/tSNE/' + experiment_name + '/any' + behavior + '_' + str(window_size_seconds) + 'sec_' + str(perplex) + 'per.pdf', format='pdf')
                plt.close()
                

    def category_tSNE(self, experiment_name, window_size_seconds, perplex, stimulus_onsets, stimulus_names, prepost_seconds, category_info, ethogram=True, category_linegraph=True, timespent_graph=True, category_graphs=True):
        #category info consists of [name, color, [(x1, y1), (x2, y2),(x3, y3),...]] for each cluster of interest
        opto_folder=self.opto_folder
        velocity_step_frames=self.velocity_step_frames
        frame_rate=self.frame_rate
        behavior_none_names = self.behaviors
        behavior_none_names.append('none')
        behavior_none_colors = self.behavior_colors
        behavior_none_colors.append('#EEEEEE')
        
        newpath = opto_folder + '/graphs/tSNE/' + experiment_name +'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        #read in feature data
        window_data = []
        for t, trial_name in zip(range(len(self.behavior_data)), self.trial_names):
            window_data.append([])
            with open(opto_folder + '/cluster_features/'+trial_name +'_' + str(window_size_seconds) + '.txt', 'r') as inFile:
                all_lines = inFile.readlines()
                line1 = all_lines[0].split()
                # x=-2
                # y=-1
                cn=-1 
                end=-1
                if line1[-1][:4] !='tSNE':
                    print('tSNE not calculated for this trial!')
                while line1[cn][:4] =='tSNE':                        
                    if line1[cn].split('_')[-1].strip('sper') == str(perplex): 
                        x=cn-1
                        y=cn   
                        end = cn-1 
                        break
                    cn+=-2
                if end == -1:
                    print('tSNE not calculated for this perplexity!')  
                last_mosquito = -1
                m = -1
                for line in all_lines[1:]:
                    line_split = line.split()
                    if last_mosquito != int(line_split[0]):
                        m += 1
                        window_data[t].append([])
                        last_mosquito = int(line_split[0])
                    window_data[t][m].append(line_split)
        #differences between states
        n_mosquitoes = 0
        n_datapoints = 0
        for trial in window_data:
            n_mosquitoes += len(trial)
            for mosquito in trial:
                n_datapoints += len(mosquito)
        print('total windows in tSNE: ', n_datapoints)
        print('total mosquitoes in tSNE: ', n_mosquitoes)
        windowcats = [[]]
        [windowcats.append([]) for category in category_info]
        
        print(x,y)
        
        timespent_data = []
        #graph tSNE cluster categories onto individual mosquito ethograms
        for st, stim_type in enumerate(stimulus_onsets): #iterate through stimulus types
            print('graph for stimuli ', stim_type, stimulus_names[st])
            stimulus_on = 0
            i=0
            #find stimulus offset for graph, replaced with generic 5 second stimulus
            stimulus_off = 5/velocity_step_frames*frame_rate
            # i=0
            # while True:
            #     if len(stim_type[i]) > 0:
            #         print(i, self.trial_names[i], stim_type[i], self.onsets[0])
            #         stimulus_off = (self.offsets[0][stim_type[i][0]] - self.onsets[0][stim_type[i][0]])*velocity_step_frames/frame_rate
            #         break
            #     i += 1
            
            #THIS IS A HARD CODED VARIABLE AND MUST BE UPDATED IF CHANGED
            step_size = 10
            windowpoints = int((prepost_seconds[0] + prepost_seconds[1])/step_size)

            #calculate info for graphs by each category
            #behavior_hist structure is non-categorized, then categories, mosquitoes, windows indicated by 0, 1
            #each window has one datapoint in behavior_hist
            behavior_hist = [[]]
            [behavior_hist.append([]) for category in category_info]
            #stimulus ethogram structure is non-categorized + categories, mosquitoes, [onsets,offsets]
            #stimulus ethogram plots state as middle 10 seconds of window
            ethogram_data = [[]] 
            [ethogram_data.append([]) for category in category_info] #
            global_m = 0
            for t in range(len(self.behavior_data)):
                for i in stim_type[t]:
                    start_frame = int(self.onsets[t][i] - prepost_seconds[0]*frame_rate/velocity_step_frames)
                    end_frame = int(self.onsets[t][i] + prepost_seconds[1]*frame_rate/velocity_step_frames + 1)
                    # if t >0:
                    #     print(t, i)
                    # print('frames ',start_frame, end_frame, stimulus_names[st])
                    for m in range(len(window_data[t])):
                        [cat.append([0]*windowpoints) for cat in behavior_hist]
                        [cat.append([[],[]]) for cat in ethogram_data]
                        last_window_cat = -1
                        for window in window_data[t][m]:
                            if int(window[1]) < start_frame or int(window[1]) >= end_frame:
                                continue
                            for c, category in enumerate(category_info):
                                #test if window is a member of the cluster of points in category
                                if point_inside_polygon(float(window[x]),float(window[y]),category[2]):#[(category[2][0],category[2][1]),(category[2][2],category[2][3]),(category[2][4],category[2][5])]) 
                                    windowcats[c+1].append(window)
                                    window_index = int((int(window[1])-start_frame)/frame_rate/step_size) #midpoint frane of window
                                    # print(int(window[1]), start_frame, end_frame, step_size)
                                    # print(len(behavior_hist), len(behavior_hist[0]), len(behavior_hist[0][0]), window_index)
                                    behavior_hist[c+1][global_m][window_index] = 1
                                    if last_window_cat != c+1 or int(window[1])-last_window_frame > window_step_size*frame_rate:
                                        if last_window_cat != -1:
                                            ethogram_data[last_window_cat][global_m][1].append((last_window_frame-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                                            if ethogram_data[last_window_cat][global_m][0][-1] > ethogram_data[last_window_cat][global_m][1][-1]:
                                                print('error in this bout of state',category[0])
                                                print('trial index',t)
                                                print('start bout', ethogram_data[last_window_cat][global_m][0][-1], 'stop bout', ethogram_data[last_window_cat][global_m][1][-1])
                                                print(window)
                                                ethogram_data[last_window_cat][global_m][0] = ethogram_data[last_window_cat][global_m][0][:-1]
                                                ethogram_data[last_window_cat][global_m][1] = ethogram_data[last_window_cat][global_m][1][:-1]
                                        ethogram_data[c+1][global_m][0].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate-window_step_size/2)
                                        # print(last_window_cat, c+1)
                                        # print(ethogram_data[last_window_cat][m][1], ethogram_data[c+1][m][0])
                                        last_window_cat = c+1
                                    last_window_frame = int(window[1])
                                    break
                            else:
                                windowcats[0].append(window)
                                window_index = int((int(window[1])-start_frame)/frame_rate/step_size)
                                # print(start_frame, end_frame, window[1], window_index, len(behavior_hist[c+1][global_m]))
                                behavior_hist[c+1][global_m][window_index] = 1
                                if last_window_cat != 0 or int(window[1])-last_window_frame > window_step_size*frame_rate:
                                    if last_window_cat != -1:
                                        ethogram_data[last_window_cat][global_m][1].append((last_window_frame-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size)
                                    ethogram_data[0][global_m][0].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate-window_step_size/2)
                                    last_window_cat = 0
                                last_window_frame = int(window[1])
                        if last_window_cat != -1:
                            ethogram_data[last_window_cat][global_m][1].append((last_window_frame-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                        global_m += 1
            # print(windowcats[0][:10])
            # print('ethogram_data',np.shape(ethogram_data))
            # print(ethogram_data[0][24])
            if ethogram:
                ymin,ymax = 0, len(ethogram_data[0])
                plt.figure()
                ax = plt.subplot(1,1,1)
                #plot stimulus starting at 0
                plt.fill(np.array([stimulus_on,stimulus_on,stimulus_off,stimulus_off]),np.array([ymin,ymax,ymax,ymin]), 'red', alpha=0.3)
                for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                    label.set_fontsize(5)
                plt.ylim(ymin,ymax)
                plt.xlim(-prepost_seconds[0], prepost_seconds[1])
                plt.title(stimulus_names[st])
                #reshape ethogram, calculate amount of category1, and sort
                ethogram_sort = []
                for m in range(len(ethogram_data[0])):
                    total_behavior = 0
                    behavior_index = 1 #first category of interest
                    for on, off in zip(ethogram_data[behavior_index][m][0],ethogram_data[behavior_index][m][1]):
                        total_behavior += off-on
                    if total_behavior == 0:
                        total_behavior = 1/random.randint(1,1000)
                    next_entry = [total_behavior,ethogram_data[0][m]]
                    for j in range(len(category_info)):
                        next_entry.append(ethogram_data[j+1][m])
                    ethogram_sort.append(next_entry)
                ethogram_sort.sort(reverse = True)
                #plot gaps and behavior bouts
                for m, entry in enumerate(ethogram_sort):
                    # for ongap, offgap in zip(entry[1][0], entry[1][1]):
                    #     plt.fill(np.array([ongap,ongap,offgap,offgap]),np.array([m+1,m,m,m+1]), 'black')
                    for c in range(len(category_info)):
                        for onset, offset in zip(entry[c+2][0], entry[c+2][1]):
                            plt.fill(np.array([onset,onset,offset,offset]),np.array([m+1,m,m,m+1]), linewidth=0, color=category_info[c][1])
                plt.savefig(newpath + '/onsetindex_' + str(st) + '.pdf', format='pdf') 
                plt.close('all')
            if category_linegraph:
                
                plt.figure()
                plt.ylim(0,.5)
                plt.xlim(-prepost_seconds[0], prepost_seconds[1])
                x_axis = np.linspace(-prepost_seconds[0], prepost_seconds[1], int((prepost_seconds[0] + prepost_seconds[1])/step_size))
                plt.title(stimulus_names[st])
                for c in range(len(category_info)):
                    i = step_size
                    behavior_to_plot = []
                    for m in behavior_hist[c+1]:
                        if len(m) >len(x_axis):
                            behavior_to_plot.append(m[:len(x_axis)])
                        else:
                            behavior_to_plot.append(m)
                    behavior_hist_array=avgNestedLists(behavior_to_plot)
                    plt.plot(x_axis, behavior_hist_array, color=category_info[c][1])
                plt.savefig(newpath + '/prop_category_' + str(st) + '.pdf', format='pdf')
                plt.close('all')
                
                if timespent_graph:
                    start_index = int(prepost_seconds[0]/step_size) #start at stimulus onset
                    behav_array = np.asarray(behavior_hist)
                    print(np.shape(behav_array[:,:,start_index:]))
                    behavior_to_plot = np.sum(behav_array[1:,:,start_index:], axis=(2))*10/60
                    print(np.shape(behavior_to_plot))
                    timespent_data.append(behavior_to_plot)
        if timespent_graph:
            #comparisons between stimulus within state category
            for c in range(len(category_info)):
                behavior_to_plot = [timespent_data[st][c] for st in range(len(stimulus_onsets))]
                plt.figure()
                plt.title(category_info[c][0])
                plt.ylabel('Time in state post-stimulus (minutes)')
                violin_parts = plt.violinplot(behavior_to_plot)#,linewidth=0)
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(category_info[c][1])
                bplot = plt.boxplot(behavior_to_plot, whis=[5,95])
                plt.xlabel(stimulus_names)
                plt.savefig(newpath + '/timespent_' + category_info[c][0] + '.pdf', format='pdf')
                
                #statistical tests
                response_array = np.asarray(behavior_to_plot,dtype=object)
                print(np.shape(response_array))
                # successes = response_array > 0.01
                # contingency = []
                # for column in successes:
                #     contingency.append([sum(column), len(column)-sum(column)])
                with open(newpath + '/friedman_nemenyi_'+ category_info[c][0] +'.txt', 'w') as outFile:
                    outFile.write('\ncategories, mosquitoes: ')
                    outFile.write(str(np.shape(response_array)))
                    # outFile.write('\nStimulus types (Light or Heat) by plate:\n')
                    # for plate in onset_types:
                    #     outFile.write(' '.join(plate)+'\n')
                    outFile.write('Stimulus indices:\n')
                    outFile.write(str(stimulus_onsets))
                    outFile.write('\nStimulus names:\n')
                    outFile.write(', '.join(stimulus_names) + '\n\n')
                    # outFile.write(str(friedmanchisquare(*response_array)))
                    # outFile.write('\nScikit_posthocs Nemenyi Friedman:\n')
                    # outFile.write(str(ph.posthoc_nemenyi_friedman(response_array.T)))
                    # outFile.write('\n\n')
                    outFile.write(str(kruskal(*response_array)))
                    outFile.write('\nScikit_posthocs Nemenyi')
                    outFile.write(str(ph.posthoc_nemenyi(response_array)))
                    outFile.write('\n\n')
                    
                    # outFile.write('ChiSquared Test\n' + str(np.array(contingency))+'\n')
                    # stat, p, dof, expected = chi2_contingency(contingency)
                    # outFile.write('test statistic ='+str(stat)+', p ='+str(p)+', dof ='+str(dof)+'\nexpected values:\n'+str(expected))
        if category_graphs:
            for c,cat in enumerate(windowcats):
                windowcats[c] = np.asarray(cat).astype(np.float)
            #plot of categories on tSNE plot 
            plt.figure()
            category_colors = ["#EEEEEE"]
            for cat in category_info:
                category_colors.append(cat[1])
            for c,window_array in enumerate(windowcats):
                print(c, np.shape(window_array), category_colors[c])
                plt.scatter(window_array[:,x],window_array[:,y], marker='.', s=5, linewidths=0.0, c=category_colors[c])
                # plt.colorbar(label='Average velocity')
                plt.title('Categories')
                plt.xlabel('tSNE1')
                plt.ylabel('tSNE2')
            cat_names = ['None']
            [cat_names.append(cat[0]) for cat in category_info]
            plt.legend(labels = cat_names, labelcolor=category_colors)#plot legend to figure instead of subplot(plt.legend)
            plt.savefig(self.opto_folder + '/graphs/tSNE/' + experiment_name + '/categories_tSNE_' + str(window_size_seconds) + 'sec_' + str(perplex) + 'per.png', format='png', dpi=400)
            plt.close()
            #prints transitions between each behavior for each state
            print('\t'.join(line1[16:36]))
            for cat in windowcats[1:]: #skip windows not in a category
                # print('cat shape ',np.shape(cat))
                if np.shape(cat)[0] == 0:
                    continue
                totals = np.sum(cat, 0)
                # print(totals)
                total_minutes = np.shape(cat)[0]*30/60
                print('\t'.join(map(str,np.divide(totals[16:36],total_minutes))))
            labels = []
            colors = []
            for cat in category_info:
                colors.append(cat[1])
            [labels.append(cat[0]) for cat in category_info]
            #distance traveled graph
            current_data = []
            for cat in windowcats[1:]: #skip windows not in a category
                print('cat shape ',np.shape(cat))
                if np.shape(cat)[0] == 0:
                    continue
                current_data.append(cat[:,3]*30/10) #convert to cm traveled in 30s
                print('velocity', np.mean(cat[:,3]))
            plt.figure()
            fig, ax = plt.subplots()
            violin_parts = plt.violinplot(current_data, showextrema=False, points=200)
            for col, pc in zip(colors, violin_parts['bodies']):
                pc.set_facecolor(col)
            bplot = plt.boxplot(current_data, sym='')
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_color(color)  
            plt.xlabel('Cluster', fontsize=13)
            ax.set_xticklabels(labels)
            plt.ylim(0,30)
            plt.ylabel('Distance traveled (cm)', fontsize=13)
            ax.tick_params(axis='both', which='major', labelsize=13)
            plt.savefig(newpath + '/distance_traveled.pdf', format='pdf') 
            #s on distance traveled
            with open(newpath + '/distance_traveled_stats.txt', 'w') as outFile:
                outFile.write('categories, mosquitoes: ')
                outFile.write(str(np.shape(current_data)))
                # outFile.write('\nStimulus types (Light or Heat) by plate:\n')
                # for plate in onset_types:
                #     outFile.write(' '.join(plate)+'\n')
                outFile.write('Categories:\n')
                outFile.write(str(labels))
                outFile.write('\n')
                # outFile.write(str(friedmanchisquare(*response_array)))
                # outFile.write('\nScikit_posthocs Nemenyi Friedman:\n')
                # outFile.write(str(ph.posthoc_nemenyi_friedman(response_array.T)))
                # outFile.write('\n\n')
                outFile.write(str(kruskal(*current_data)))
                outFile.write('\nScikit_posthocs Nemenyi\n')
                outFile.write(str(ph.posthoc_nemenyi(current_data)))
                outFile.write('\n\n')
            with open(newpath + '/distance_traveled_data.txt', 'w') as outFile:
                for c in range(len(category_info)):
                    outFile.write(category_info[c][0] + '\t' + '\t'.join(map(str,current_data[c]))+'\n')
            #behavior violoin plots
            behav_name=['grooming','walking','probing','flying','none']
            for b, behavior in enumerate(behavior_none_names):
                current_data = []
                #prints out proportion of time spent in each behavior for each state
                for cat in windowcats[1:]: #skip windows not in a category
                    print('cat shape ',np.shape(cat))
                    if np.shape(cat)[0] == 0:
                        continue
                    current_data.append(cat[:,b+4])
                    print(behavior, np.mean(cat[:,b+4]))
                plt.figure()
                fig, ax = plt.subplots()
                violin_parts = plt.violinplot(current_data, showextrema=False)
                for col, pc in zip(colors, violin_parts['bodies']):
                    pc.set_facecolor(col)
                bplot = plt.boxplot(current_data, sym='')
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_color(color)  
                ax.set_title('Proportion of time '+ behav_name[b], fontsize=20)
                plt.xlabel('Cluster', fontsize=13)
                ax.set_xticklabels(labels)
                plt.ylabel('Proportion', fontsize=13)
                ax.tick_params(axis='both', which='major', labelsize=13)
                plt.savefig(newpath + '/cats_' + behavior + '.pdf', format='pdf')      
            for i, name in zip([13,14],['walkvel','probevel']):
                behaving_name=['walking','probing']
                current_data = []
                for cat in windowcats[1:]: #skip windows not in a category
                    if np.shape(cat)[0] == 0:
                        continue
                    current_data.append(cat[:,i])
                plt.figure()
                fig, ax = plt.subplots()
                violin_parts = plt.violinplot(current_data, showmeans=False, showmedians=False, showextrema=False)
                for col, pc in zip(colors, violin_parts['bodies']):
                    pc.set_facecolor(col)
                bplot = plt.boxplot(current_data, sym='')
                print('Sample Size is = '+ str(len(current_data[0])))
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_color(color)
                ax.set_title('Velocity of '+behaving_name[i-13], fontsize=20)
                plt.xlabel('Cluster', fontsize=13)
                ax.set_xticklabels(labels)
                ax.set_ylim(-0.5,30)
                plt.ylabel('Velocity/mm/s', fontsize=13)
                ax.tick_params(axis='both', which='major', labelsize=13)
                plt.savefig(newpath + '/cats_' + name + '.pdf', format='pdf') 


    def category_transition(self, experiment_name, window_size_seconds, perplex, stimulus_onsets, stimulus_names, prepost_seconds, category_info, ethogram=True, category_linegraph=True, timespent_graph=True, category_graphs=True):
        #category info consists of [name, color, [(x1, y1), (x2, y2),(x3, y3),...]] for each cluster of interest
        opto_folder=self.opto_folder
        velocity_step_frames=self.velocity_step_frames
        frame_rate=self.frame_rate
        behavior_none_names = self.behaviors
        behavior_none_names.append('none')
        behavior_none_colors = self.behavior_colors
        behavior_none_colors.append('#EEEEEE')
        
        newpath = opto_folder + '/graphs/tSNE/' + experiment_name +'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        #read in feature data
        window_data = []
        last_frame=0
        for t, trial_name in zip(range(len(self.behavior_data)), self.trial_names):
            window_data.append([])
            with open(opto_folder + '/cluster_features/'+trial_name +'_' + str(window_size_seconds) + '.txt', 'r') as inFile:
                all_lines = inFile.readlines()
                line1 = all_lines[0].split()
                # x=-2
                # y=-1
                cn=-1 
                end=-1
                if line1[-1][:4] !='tSNE':
                    print('tSNE not calculated for this trial!')
                while line1[cn][:4] =='tSNE':                        
                    if line1[cn].split('_')[-1] == str(perplex): 
                        x=cn-1
                        y=cn   
                        end = cn-1 
                        break
                    cn+=-2
                if end == -1:
                    print('tSNE not calculated for this perplexity!')  
                last_mosquito = -1
                m = -1
                for line in all_lines[1:]:
                    line_split = line.split()
                    if last_mosquito != int(line_split[0]):
                        m += 1
                        last_frame=line_split[1]
                        window_data[t].append([])
                        last_mosquito = int(line_split[0])
                    window_data[t][m].append(line_split)
        #differences between states
        windowcats = [[]]
        [windowcats.append([]) for category in category_info]
        
        #graph tSNE cluster categories onto individual mosquito ethograms
        cluster_prob=[[],[],[]]
        
        for st, stim_type in enumerate(stimulus_onsets): #iterate through stimulus types
            print('graph for stimuli ', stim_type, stimulus_names[st])
            stimulus_on = 0
            i=0
            while True:
                if len(stim_type[i]) > 0:
                    stimulus_off = (self.offsets[0][stim_type[i][0]] - self.onsets[0][stim_type[i][0]])*velocity_step_frames/frame_rate
                    break
                i += 1
            
            #THIS IS A HARD CODED VARIABLE AND MUST BE UPDATED IF CHANGED
            step_size = 10
            windowpoints = int((prepost_seconds[0] + prepost_seconds[1])/step_size)

            behavior_hist = [[]] #behavior_hist structure is non-categorized, then categories, mosquitoes, windows indicated by 0, 1
            [behavior_hist.append([]) for category in category_info]
            ethogram_data = [[]] #ethogram structure is non-categorized + categories, mosquitoes, [onsets,offsets]
            [ethogram_data.append([]) for category in category_info] #
            global_m = 0
            
            max_windows=0
            max_mosquitos=0
            n_mosquitos=0
            for t in range(len(self.behavior_data)):
                if len(window_data[t])>max_mosquitos:
                        max_mosquitos=len(window_data[t])
                for m in range(len(window_data[t])):
                    if len(window_data[t][m])>max_windows:
                        max_windows=len(window_data[t][m])
            cluster_heirarchy = np.zeros((len(self.behavior_data), max_mosquitos, max_windows))
            cluster_count = np.zeros((len(self.behavior_data), max_mosquitos, len(category_info)))
            cluster_ratio = np.zeros((len(self.behavior_data), max_mosquitos, len(category_info)))
            print(len(self.behavior_data))
            for t in range(len(self.behavior_data)):
                for m in range(len(window_data[0])):
                    for w in range(max_windows):
                        cluster_heirarchy[t][m][w]=-1
            #print(cluster_heirarchy)
            timespent_total=[[],[],[],[]]
            for t in range(len(self.behavior_data)):
                n_mosquitos+=len(window_data[t])
                for i in stim_type[t]:
                    start_frame = int(self.onsets[t][i] - prepost_seconds[0]*frame_rate/velocity_step_frames)
                    end_frame = int(self.onsets[t][i] + prepost_seconds[1]*frame_rate/velocity_step_frames + 1)
                    if t >0:
                        print(t, i)
                    for m in range(len(window_data[t])):
                        [cat.append([0]*windowpoints) for cat in behavior_hist]
                        [cat.append([[],[]]) for cat in ethogram_data]
                        last_window_cat = -1
                        window_count=0
                        for w, window in enumerate(window_data[t][m]):
                            if int(window[1]) < start_frame or int(window[1]) >= end_frame or int(window[1]) >=int(self.onsets[t][i] + 3600):
                                continue
                            for c, category in enumerate(category_info):
                                #test if window is a member of the cluster of points in category
                                if point_inside_polygon(float(window[x]),float(window[y]),category[2]):#[(category[2][0],category[2][1]),(category[2][2],category[2][3]),(category[2][4],category[2][5])]) 
                                    windowcats[c+1].append(window)
                                    window_index = int((int(window[1])-start_frame)/frame_rate/step_size)
                                    cluster_heirarchy[t][m][w]=c+1
                                    window_count+=1
                                    behavior_hist[c+1][global_m][window_index] = 1
                                    if last_window_cat != c+1:
                                        if last_window_cat != -1:
                                            ethogram_data[last_window_cat][global_m][1].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                                        ethogram_data[c+1][global_m][0].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                                        last_window_cat = c+1
                                    break
                                else:
                                    cluster_heirarchy[t][m][w]=0
                            else:
                                windowcats[0].append(window)
                                window_index = int((int(window[1])-start_frame)/frame_rate/step_size)
                                behavior_hist[c+1][global_m][window_index] = 1
                                if last_window_cat != 0:
                                    if last_window_cat != -1:
                                        ethogram_data[last_window_cat][global_m][1].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                                    ethogram_data[0][global_m][0].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                                    last_window_cat = 0
                        total=0

                        for c, category in enumerate(category_info):
                            cluster_count[t][m][c]+=np.count_nonzero(cluster_heirarchy[t,m]-(c+1) == 0)
                            if window_count!=0:
                                cluster_ratio[t][m][c]=cluster_count[t][m][c]/window_count#len(window_data[t][m])
                            else: 
                                cluster_ratio[t][m][c]=0
                            total+=cluster_ratio[t][m][c]
                            timespent_total[c].append(cluster_ratio[t][m][c])
                        ethogram_data[last_window_cat][global_m][1].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                        global_m += 1
            
            for c, category in enumerate(category_info):
                print(timespent_total[c])
            colors = []
            labels = []
            [labels.append(cat[0]) for cat in category_info]
            for cat in category_info:
                colors.append(cat[1])
            plt.figure()
            fig, ax = plt.subplots()
            violin_parts = plt.violinplot(timespent_total, showmeans=False, showmedians=False, showextrema=False)
            for col, pc in zip(colors, violin_parts['bodies']):
                pc.set_facecolor(col)
            bplot = plt.boxplot(timespent_total, sym='')
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_color(color)
            ax.set_title('Timespent with ' + stimulus_names[st], fontsize=20)
            plt.xlabel('Cluster', fontsize=13)
            ax.set_xticklabels(labels)
            plt.ylabel('Proportion of time', fontsize=13)
            ax.tick_params(axis='both', which='major', labelsize=13)
            plt.savefig(newpath + 'timespent_' + str(st) + '.pdf', format='pdf') 
                
            transitions = np.zeros((len(self.behavior_data),max_mosquitos, max_windows))
            transitions_count = np.zeros((len(self.behavior_data),max_mosquitos,6,6))  
            transitions_total = np.zeros((n_mosquitos,6,6))
            transitions_conprob = np.zeros((len(self.behavior_data),max_mosquitos,6,6))  
            transitions_total_conprob = np.zeros((n_mosquitos,6,6))
            
            transitions_total_bias = np.zeros((6,6))
            
            transitions_conprob1 = np.zeros((len(self.behavior_data),max_mosquitos,10,10))  
            transitions_total_conprob1 = np.zeros((n_mosquitos,10,10))
            nm=0
            for t in range(len(self.behavior_data)):
                for m in range(len(window_data[t])):
                    transitions[t][m] = cluster_heirarchy[t][m]*10 + (np.roll(cluster_heirarchy[t][m],-1)) #max number of cluster=10
                    for i in range(-1,5):#range(-1,9)
                        for j in range(-1,5):
                            n=i*10+j
                            transitions_count[t,m,i,j]=np.count_nonzero(transitions[t][m]-n == 0)
                            if i==j: #or i==-1 or j==-1:
                                transitions_count[t,m,i,j]=0
                    transitions_total[nm]=transitions_count[t,m]
                    nm+=1
            nm=0
            for t in range(len(self.behavior_data)):
                for m in range(len(window_data[t])):
                    no=np.zeros((11))
                    no1=np.zeros((11))                    
                    for i in range(-1,5):
                        for j in range(-1,5):
                            no[i]+=transitions_count[t,m,i,j]
                            no1[j]+=transitions_count[t,m,i,j]
                    for i in range(-1,5):
                        for j in range(-1,5):
                            transitions_conprob[t,m,i,j]=transitions_count[t,m,i,j]/no[i]
                            transitions_conprob1[t,m,i,j]=transitions_count[t,m,i,j]/no[j]
                            # if i==j or i==-1 or j==-1:
                            #     transitions_conprob[t,m,i,j]='NaN'
                            #     transitions_conprob1[t,m,i,j]='NaN'
                    transitions_total_conprob[nm]=transitions_conprob[t,m]
                    transitions_total_conprob1[nm]=transitions_conprob1[t,m]
                    nm+=1
            
            print(n_mosquitos)
            print(np.mean(transitions_total,axis=0))
            a = np.mean(transitions_total,axis=0)
            a= np.delete(a, 0, axis=0)
            a= np.delete(a, 0, axis=1)
            plt.figure()
            fig, ax = plt.subplots()
            plt.imshow(a,cmap='cet_fire')
            plt.title('Transition numbers upon '+ str(stimulus_names[st]), fontsize=20)
            plt.colorbar()
            plt.xlabel('To', fontsize=13)
            labels = []
            colors = []
            for cat in category_info:
                colors.append(cat[1])
            [labels.append(cat[0]) for cat in category_info]
            labels.append('no behavior')
            colors.append('#000000')
            ax.set_xticks(range(0,5))
            ax.set_xticklabels(labels, rotation=30, ha='right')
            ax.set_yticks(range(0,5))
            ax.set_yticklabels(labels)
            plt.ylabel('From', fontsize=13)
            ax.tick_params(axis='both', which='major', labelsize=13)
            plt.tight_layout()
            plt.subplots_adjust(left=0, right=0.95, wspace=0, hspace=0)
            plt.savefig(newpath + '/Trans_' + str(st) + '.pdf', format='pdf',bbox_inches='tight') 
            
            print(np.nanmean(transitions_total_conprob,axis=0))
            b=np.nanmean(transitions_total_conprob,axis=0)
            b= np.delete(b, 0, axis=0)
            b= np.delete(b, 0, axis=1)
            plt.figure()
            fig, ax = plt.subplots()
            plt.imshow(b,cmap='cet_fire')
            plt.title('Conditional probability upon '+ str(stimulus_names[st]), fontsize=20)
            plt.colorbar()
            plt.xlabel('To', fontsize=13)
            ax.set_xticks(range(0,5))
            ax.set_xticklabels(labels, rotation=30, ha='right')
            ax.set_yticks(range(0,5))
            ax.set_yticklabels(labels)
            plt.ylabel('From', fontsize=13)
            ax.tick_params(axis='both', which='major', labelsize=13)
            plt.tight_layout()
            plt.subplots_adjust(left=0, right=0.95, wspace=0, hspace=0)
            plt.savefig(newpath + '/Conprob_' + str(st) + '.pdf', format='pdf',bbox_inches='tight') 
                
            c=np.nanmean(transitions_total_conprob1,axis=0)
            plt.figure()
            plt.imshow(c,cmap='cet_fire')
            plt.title('Conditional probability1 upon '+ str(stimulus_names[st]))
            plt.colorbar()
            plt.savefig(newpath + '/Conprob1_' + str(st) + '.pdf', format='pdf') 


    def engorgement_correlation(self, experiment_name, window_size_seconds, perplex, stimulus_onsets, state_window, category_info, ethogram=True, correlation_graph=True, logistic_regression=True):
        #category info consists of [name, color, [(x1, y1), (x2, y2),(x3, y3),...]] for each cluster of interest
        #state window is the period of time to correlate with engorgement e.g. [-120,0]
        #this is designed to run on a single stimulus type so it will lump them all together
        if not self.blood_blanket:
            print('Must be blood blanket experiment to contain engorgement data')
            return
        prepost_seconds = [-state_window[0],600]
        opto_folder=self.opto_folder
        velocity_step_frames=self.velocity_step_frames
        frame_rate=self.frame_rate
        behavior_none_names = self.behaviors
        behavior_none_names.append('none')
        behavior_none_colors = self.behavior_colors
        behavior_none_colors.append('#EEEEEE')
        
        newpath = opto_folder + '/graphs/tSNE/' + experiment_name +'/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        #read in feature data
        window_data = []
        for t, trial_name in zip(range(len(self.behavior_data)), self.trial_names):
            window_data.append([])
            with open(opto_folder + '/cluster_features/'+trial_name +'_' + str(window_size_seconds) + '.txt', 'r') as inFile:
                all_lines = inFile.readlines()
                line1 = all_lines[0].split()
                # x=-2
                # y=-1
                cn=-1 
                end=-1
                if line1[-1][:4] !='tSNE':
                    print('tSNE not calculated for this trial!')
                while line1[cn][:4] =='tSNE':                        
                    if line1[cn].split('_')[-1].strip('sper') == str(perplex): 
                        x=cn-1
                        y=cn   
                        end = cn-1 
                        break
                    cn+=-2
                if end == -1:
                    print('tSNE not calculated for this perplexity!')  
                last_mosquito = -1
                m = -1
                for line in all_lines[1:]:
                    line_split = line.split()
                    if last_mosquito != int(line_split[0]):
                        m += 1
                        window_data[t].append([])
                        last_mosquito = int(line_split[0])
                    window_data[t][m].append(line_split)
        #differences between states
        windowcats = [[]]
        [windowcats.append([]) for category in category_info]
        
        print(x,y)
        
        timespent_data = []
        #graph tSNE cluster categories onto individual mosquito ethograms
        for st, stim_type in enumerate(stimulus_onsets): #iterate through stimulus types
            print('graph for  ', experiment_name)
            stimulus_on = 0
            i=0
            #find stimulus offset for graph, replaced with generic 5 second stimulus
            stimulus_off = 5/velocity_step_frames*frame_rate
            
            #THIS IS A HARD CODED VARIABLE AND MUST BE UPDATED IF CHANGED
            step_size = 10
            windowpoints = int((prepost_seconds[0] + prepost_seconds[1])/step_size)

            #calculate info for graphs by each category
            #behavior_hist structure is non-categorized, then categories, mosquitoes, windows indicated by 0, 1
            #each window has one datapoint in behavior_hist
            behavior_hist = [[]]
            [behavior_hist.append([]) for category in category_info]
            #stimulus ethogram structure is non-categorized + categories, mosquitoes, [[onsets],[offsets]] 
            #stimulus ethogram plots state as middle 10 seconds of window
            ethogram_data = [[]] 
            [ethogram_data.append([]) for category in category_info] #
            engorgement_data = []
            global_m = 0
            for t in range(len(self.behavior_data)):
                for i in stim_type[t]:
                    start_frame = int(self.onsets[t][i] - prepost_seconds[0]*frame_rate/velocity_step_frames)
                    end_frame = int(self.onsets[t][i] + prepost_seconds[1]*frame_rate/velocity_step_frames + 1)
                    # if t >0:
                    #     print(t, i)
                    # print('frames ',start_frame, end_frame, stimulus_names[st])
                    for m in range(len(window_data[t])):
                        [cat.append([0]*windowpoints) for cat in behavior_hist]
                        [cat.append([[],[]]) for cat in ethogram_data]
                        #add engorgement info
                        if m in self.engorged[t]:
                            engorgement_data.append(1)
                        else:
                            engorgement_data.append(0)
                        last_window_cat = -1
                        for window in window_data[t][m]:
                            if int(window[1]) < start_frame or int(window[1]) >= end_frame:
                                continue
                            for c, category in enumerate(category_info):
                                #test if window is a member of the cluster of points in category
                                if point_inside_polygon(float(window[x]),float(window[y]),category[2]):#[(category[2][0],category[2][1]),(category[2][2],category[2][3]),(category[2][4],category[2][5])]) 
                                    windowcats[c+1].append(window)
                                    window_index = int((int(window[1])-start_frame)/frame_rate/step_size) #midpoint frane of window
                                    behavior_hist[c+1][global_m][window_index] = 1
                                    if last_window_cat != c+1 or int(window[1])-last_window_frame > window_step_size*frame_rate:
                                        if last_window_cat != -1:
                                            ethogram_data[last_window_cat][global_m][1].append((last_window_frame-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                                            if ethogram_data[last_window_cat][global_m][0][-1] > ethogram_data[last_window_cat][global_m][1][-1]:
                                                print('error in this bout of state',category[0])
                                                print('trial index',t)
                                                print('start bout', ethogram_data[last_window_cat][global_m][0][-1], 'stop bout', ethogram_data[last_window_cat][global_m][1][-1])
                                                print(window)
                                                ethogram_data[last_window_cat][global_m][0] = ethogram_data[last_window_cat][global_m][0][:-1]
                                                ethogram_data[last_window_cat][global_m][1] = ethogram_data[last_window_cat][global_m][1][:-1]
                                        ethogram_data[c+1][global_m][0].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate-window_step_size/2)
                                        last_window_cat = c+1
                                    last_window_frame = int(window[1])
                                    break
                            else:
                                windowcats[0].append(window)
                                window_index = int((int(window[1])-start_frame)/frame_rate/step_size)
                                # print(start_frame, end_frame, window[1], window_index, len(behavior_hist[c+1][global_m]))
                                behavior_hist[c+1][global_m][window_index] = 1
                                if last_window_cat != 0 or int(window[1])-last_window_frame > window_step_size*frame_rate:
                                    if last_window_cat != -1:
                                        ethogram_data[last_window_cat][global_m][1].append((last_window_frame-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size)
                                    ethogram_data[0][global_m][0].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate-window_step_size/2)
                                    last_window_cat = 0
                                last_window_frame = int(window[1])
                        if last_window_cat != -1:
                            ethogram_data[last_window_cat][global_m][1].append((last_window_frame-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                        global_m += 1
            start_index = int((prepost_seconds[0]+state_window[0])/step_size) 
            stop_index = int((prepost_seconds[0]+state_window[1])/step_size)
            
            print(start_index,stop_index)
            behav_array = np.asarray(behavior_hist)
            print(np.shape(behav_array))
            print(np.shape(behav_array[:,:,start_index:stop_index]))
            timespent_data = np.sum(behav_array[1:,:,start_index:stop_index], axis=(2))*10/60

            if ethogram:
                for e in range(2):
                    plt.figure()
                    ax = plt.subplot(1,1,1)
                    #plot stimulus starting at 0
                    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                        label.set_fontsize(5)
                    plt.xlim(-prepost_seconds[0], prepost_seconds[1])
                    plt.title(experiment_name)
                    #reshape ethogram, calculate amount of category1, and sort
                    ethogram_sort = []
                    for m in range(len(ethogram_data[0])):
                        if engorgement_data[m] != e:
                            continue
                        total_behavior = 0
                        behavior_index = 0 #first category of interest
                        total_behavior = timespent_data[behavior_index,m]
                        if total_behavior == 0:
                            total_behavior = 1/random.randint(1000,2000)
                        next_entry = [total_behavior,ethogram_data[0][m]]
                        for j in range(len(category_info)):
                            next_entry.append(ethogram_data[j+1][m])
                        ethogram_sort.append(next_entry)
                    ethogram_sort.sort(reverse = True)
                    ymin, ymax = 0, len(ethogram_sort)
                    plt.ylim(ymin,ymax)
                    plt.fill(np.array([stimulus_on,stimulus_on,stimulus_off,stimulus_off]),np.array([ymin,ymax,ymax,ymin]), 'red', alpha=0.3)
                    for m, entry in enumerate(ethogram_sort):
                        # for ongap, offgap in zip(entry[1][0], entry[1][1]):
                        #     plt.fill(np.array([ongap,ongap,offgap,offgap]),np.array([m+1,m,m,m+1]), 'black')
                        for c in range(len(category_info)):
                            for onset, offset in zip(entry[c+2][0], entry[c+2][1]):
                                plt.fill(np.array([onset,onset,offset,offset]),np.array([m+1,m,m,m+1]), linewidth=0, color=category_info[c][1])
                    plt.savefig(newpath + '/ethogram_' + str(e) + '.pdf', format='pdf') 
                    plt.close('all')
                    with open(newpath + '/ethogram_' + str(e) + '.txt', 'w') as outFile:
                        for m, entry in enumerate(ethogram_sort):
                            outFile.write('\nmosquito '+ str(m+1))
                            for c in range(len(category_info)):
                                outFile.write('\n'+category_info[c][0]+' starts \t')
                                outFile.write('\t'.join(map(str, entry[c+2][0])))
                                outFile.write('\n'+category_info[c][0]+' stops \t')
                                outFile.write('\t'.join(map(str, entry[c+2][1])))    
        print('timespent_data', np.shape(timespent_data))
        if correlation_graph:
            #comparisons between engorgement within state category
            for c in range(len(category_info)):
                behavior_to_plot = timespent_data[c]
                plt.figure()
                plt.title(category_info[c][0])
                plt.ylabel('Time in state post-stimulus (minutes)')
                behavior_by_engorgement = [[],[]]
                for m in range(len(behavior_to_plot)):
                    behavior_by_engorgement[int(engorgement_data[m])].append(behavior_to_plot[m])
                print('behavior_by_engorgement',behavior_by_engorgement)
                violin_parts = plt.violinplot(behavior_by_engorgement)#,linewidth=0)
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(category_info[c][1])
                plt.boxplot(behavior_by_engorgement, whis=[5,95])
                plt.savefig(newpath + '/timespent_' + category_info[c][0] + '_engorged.pdf', format='pdf')
                plt.close()
                #statistical tests
                with open(newpath + '/engorge_test_'+ category_info[c][0] +'.txt', 'w') as outFile:
                    outFile.write('categories, mosquitoes: ')
                    outFile.write(str(np.shape(behavior_by_engorgement[0])) + ' ' + str(np.shape(behavior_by_engorgement[1])) + '\n')
                    outFile.write('Stimulus indices:\n')
                    outFile.write(str(stimulus_onsets))
                    try:
                        outFile.write(str(mannwhitneyu(behavior_by_engorgement[0],behavior_by_engorgement[1])))
                    except ValueError:
                        outFile.write('All numbers are identical in mannwhitneyu')
                    outFile.write('\n\n')
                    # outFile.write('ChiSquared Test\n' + str(np.array(contingency))+'\n')
                    # stat, p, dof, expected = chi2_contingency(contingency)
                    # outFile.write('test statistic ='+str(stat)+', p ='+str(p)+', dof ='+str(dof)+'\nexpected values:\n'+str(expected))
        if logistic_regression:
            #creates logistic regression for all categories predicting engorge/not engorge
            with open(newpath + '/logistic_regression.txt', 'w') as outFile:
                engorgement_data = np.asarray(engorgement_data)
                timespent_data = np.asarray(timespent_data).T
                model = LogisticRegression(solver='liblinear', random_state=0, class_weight='balanced')
                model.fit(timespent_data, engorgement_data)
                model_score = model.score(timespent_data, engorgement_data)
                outFile.write('model score ' + str(model_score) + '\n')
                outFile.write('confusion matrix\n' + str(confusion_matrix(engorgement_data, model.predict(timespent_data))) + '\n')
                repeats = 10000
                rand_scores = []
                engorgement_shuffle = engorgement_data.copy()
                for i in range(repeats):
                    np.random.shuffle(engorgement_shuffle)
                    rand_scores.append(model.score(timespent_data, engorgement_shuffle))
                rand_scores = sorted(rand_scores)
                for x, val in enumerate(rand_scores):
                    if val > model_score:
                        cutoff = x
                        break
                else:
                    cutoff = repeats
                outFile.write('number of shuffles ' + str(repeats) + '\n')
                outFile.write('repeats = ' + str(repeats) + ' cutoff = ' + str(cutoff) + '\n')
                outFile.write('bootstrapped p=value ' + str((repeats-cutoff)/repeats) + '\n')
                outFile.write('average randomized prediction ' + str(np.mean(rand_scores)) + '\n\n')
                #Leave one out cross validation
                predictions = []
                for i in range(np.shape(timespent_data)[0]):
                    left_out_timespent = timespent_data[i]
                    left_out_timespent = left_out_timespent.reshape(1,-1)
                    removed_timespent = timespent_data.copy()
                    removed_timespent = np.delete(removed_timespent,i,axis=0)
                    left_out_engorge = engorgement_data[i]
                    removed_engorge = engorgement_data.copy()
                    removed_engorge = np.delete(removed_engorge, i, axis=0)
                    model = LogisticRegression(solver='liblinear', random_state=0, class_weight='balanced')
                    model.fit(removed_timespent, removed_engorge)
                    prediction = model.predict(left_out_timespent)
                    if prediction == left_out_engorge:
                        predictions.append(1)
                    else:
                        predictions.append(0)
                outFile.write('LOOCV ' + str(np.count_nonzero(predictions)/len(predictions)) + '\n\n')
                outFile.write('engorgement_data ' + str(np.shape(engorgement_data)) + '\n')
                for c in range(len(category_info)):
                    outFile.write(category_info[c][0]+'\t')
                outFile.write('engorgement\n')
                for m in range(len(engorgement_data)):
                    outFile.write('\t'.join(map(str, timespent_data[m])))
                    outFile.write('\t' + str(engorgement_data[m])+'\n')

    def annotate_video(self, stim_index, window_size_seconds, stimulus_names, start_stop_seconds, perplex, category_info, category_anno=False, behavior_anno=False, trk_anno=False, plot_stimuli=False, plot_proboscis=False, trk_filename='9pt_v2_cpr.trk'):        
        #annotates a video of a single trial around a single stimulus  (assumees an optothermo object with a single trial)
        #should only be run on an optothermo object with a single trial
        #must be run from terminal for ffmpeg commands to work
        #category_anno is tSNE categories, behavior_anno is JAABA classifications, and trk_anno is APT categories

        #read in xy locations
        # velocity_step_frames=self.velocity_step_frames
        # frame_rate=self.frame_rate
        # velocity_mat = loadmat(os.path.join(self.opto_folder, 'processed', self.trial_names[0], 'trx.mat'))
        
        #first assign xy values to wells
        plate_corners = [[20,20],[1066,16],[1065,690],[20,689]]
        plate_rows, plate_cols = 3,5
        wells = calculate_grid(plate_corners, plate_rows, plate_cols)
        
                
        #read in trk information as well and associate with each individual
        #organization is list of lists by well, frame, and then individual [x,y] points
        allEnds = [item for sublist in self.behavior_data[0][self.behaviors[0]]['tEnds'] for item in sublist]
        end_frame = max(allEnds)
        xys = np.empty((len(wells), end_frame, 4))
        trks = []
        for well in wells:
            fs = []
            [fs.append([]) for frame in range(end_frame)]
            trks.append(fs)
        for root, dirs, files in os.walk(self.opto_folder + '/processed'):
            trial_dirs = natural_sort([d for d in dirs if d[:len(self.trial_names[0])] == self.trial_names[0]])
            print(trial_dirs)
            break
        #iterate through each trial chunk, adding mosquito tracks
        chunklens = []
        prev_chunks_total = 0
        for trial_chunk in trial_dirs:
            if trk_anno:
                trk_file = loadmat(os.path.join(self.opto_folder, 'processed', trial_chunk, trk_filename))
                print(np.shape(trk_file['pTrk']))
            velocity_mat = loadmat(os.path.join(self.opto_folder, 'processed', trial_chunk, 'trx.mat'))
            for w in range(len(wells)):
                for f in range(len(velocity_mat['timestamps'])):
                    for d in range(4):
                        xys[w,prev_chunks_total+f,d]=-5000
            for t, track in enumerate(velocity_mat['trx']): #get locations of each mosquito
                f = track.firstframe - 1
                for xloc, yloc in zip(track.x, track.y):
                    for w, well in enumerate(wells):
                        if xloc > well[0] and xloc < well[1] and yloc > well[2] and yloc < well[3]:
                            xys[w,prev_chunks_total+f,0] = xloc
                            xys[w,prev_chunks_total+f,1] = yloc
                            if trk_anno:
                                trks[w][prev_chunks_total+f].append(trk_file['pTrk'][:,0,f,t])
                                trks[w][prev_chunks_total+f].append(trk_file['pTrk'][:,1,f,t])
                    f += 1
            prev_chunks_total += len(velocity_mat['timestamps'])
            chunklens.append(len(velocity_mat['timestamps']))
        print('xys',np.shape(xys),'trks',np.shape(trks))
        
        #calculate proboscis length
        if plot_proboscis:
            prob_lengths = []
            for well in trks:
                well_lengths = []
                for frame in well:
                    if len(frame)==0:
                        well_lengths.append(np.nan)
                    else:
                        x1, x2, y1, y2 = frame[0][0], frame[0][1], frame[1][0], frame[1][1]
                        well_lengths.append(np.sqrt((x2-x1)**2 + (y2-y1)**2)*self.well_size_mm/self.well_size_pixels)
                prob_lengths.append(well_lengths)
                    

        
        #read in tSNE information
        window_data = []
        for t, trial_name in zip(range(len(self.behavior_data)), self.trial_names):
            window_data.append([])
            with open(self.opto_folder + '/cluster_features/'+trial_name +'_' + str(window_size_seconds) + '.txt', 'r') as inFile:
                all_lines = inFile.readlines()
                line1 = all_lines[0].split()
                # x=-2
                # y=-1
                cn=-1 
                end=-1
                if line1[-1][:4] !='tSNE':
                    print('tSNE not calculated for this trial!')
                while line1[cn][:4] =='tSNE':                        
                    if line1[cn].split('_')[-1].strip('sper') == str(perplex): 
                        x=cn-1
                        y=cn   
                        end = cn-1 
                        break
                    cn+=-2
                if end == -1:
                    print('tSNE not calculated for this perplexity!')  
                last_mosquito = -1
                m = -1
                print(len(all_lines))
                for line in all_lines[1:]:
                    line_split = line.split()
                    if last_mosquito != int(line_split[0]):
                        m += 1
                        window_data[t].append([])
                        last_mosquito = int(line_split[0])
                    window_data[t][m].append(line_split)
        print('window data', np.shape(window_data))
        #assign tSNE values to frames
        if category_anno:            
            step_size = 10 #THIS IS A HARD CODED VARIABLE AND MUST BE UPDATED IF CHANGED
            for m in range(len(window_data[0])):
                for window in window_data[0][m]:
                    window_start=int(window[1])
                    statestart = int(window_start -step_size*self.frame_rate/2)
                    statestop = int(window_start +step_size*self.frame_rate/2)
                    #add tSNE values to xys
                    if statestop > np.shape(xys)[1]:
                        statestop = np.shape(xys)[1]
                    xys[m,statestart:statestop,2] = [float(window[x]) for i in range(statestop-statestart)]
                    xys[m,statestart:statestop,3] = [float(window[y]) for i in range(statestop-statestart)]
        print(xys[:,300:400])
        #retrieve behavior data
        behavior_indices = np.ones((len(wells), end_frame), dtype=np.int8) #behavior index
        behavior_indices = behavior_indices * -1
        behavior_names = np.empty((len(wells), end_frame), dtype='U5') #unicode(string) shorter than length of 5, behavior name
        print(behavior_indices[:14,:100])
        print(behavior_names[:14,:100])
        for m in range(len(self.behavior_data[0][self.behaviors[0]]['t0s'])):
            tStart = self.behavior_data[0][self.behaviors[0]]['tStarts'][m][0] - 1
            for b, behavior_name in enumerate(self.behaviors):        
                for onset, offset in zip(self.behavior_data[0][behavior_name]['t0s'][m], self.behavior_data[0][behavior_name]['t1s'][m]):
                    behavior_indices[m,tStart+onset:tStart+offset]= [b for x in range(offset-onset)]
                    behavior_names[m,tStart+onset:tStart+offset]= [behavior_name[:-1].strip('_B') for x in range(offset-onset)]

        frame_size = (700,1180)
        source = self.opto_folder + '/processed/'
        path=self.opto_folder+'/videos/'+ self.trial_names[0] #output path
        newpath=path+'/output/'
        if not os.path.exists(newpath):
            os.makedirs(newpath)     
        print('current_behavior', np.shape(behavior_indices),np.shape(behavior_names))
        print(behavior_indices[:14,:100])
        print(behavior_names[:14,:100])
        #This looks for videos in the project_folder/processed/trial_name
        for root, dirs, files in os.walk(source):
            # print(root,dirs,files)
            mp4_files = [f for f in files if f[-3:] == 'mp4']
            trial_vids = natural_sort([f for f in mp4_files if f[:len(self.trial_names[0])] == self.trial_names[0]])
            print(len(trial_vids))
            print(trial_vids)
            #find the mp4_files that correspond to the trial you are annotating
            break
        f=0
        fstart = int(self.onsets[0][stim_index] + start_stop_seconds[0]*self.frame_rate)
        fstop = int(self.onsets[0][stim_index] + start_stop_seconds[1]*self.frame_rate) #ending frame number
        print('stimulus onset', self.onsets[0][stim_index], 'start frame', fstart, 'stop frame', fstop)
        t1 = time.time()
        for i0 in range(len(trial_vids)):
            if fstart > sum(chunklens[0:i0+1]):
                f = sum(chunklens[0:i0+1])
                continue
     

            incommand = [ 'ffmpeg',
                    #'-vsframes', str(f0),     
                    '-i', os.path.join(self.opto_folder,'processed', trial_vids[i0]),
                    #'-ss', str(f0/frame_rate), #starting frame number to extract frames 
                    '-f', 'image2pipe',
                    '-pix_fmt', 'gray',
                    '-vcodec', 'rawvideo', '-']
            print('incommand:',' '.join(incommand))
            pipein = sp.Popen(incommand, stdout = sp.PIPE, bufsize=10**8)
            while True:
                raw_image = pipein.stdout.read(frame_size[0]*frame_size[1])
                image =  np.frombuffer(raw_image, dtype='uint8')
                if f < fstart:
                    f += 1
                    continue
                elif f > fstop:
                    break
                if np.size(image) < frame_size[0]*frame_size[1]:
                    print('insufficient frame size:',np.size(image))
                    pipein.stdout.flush()
                    break
                frame = image.reshape(frame_size)
                plt.figure()#(figsize=(16.4,10))
                if plot_proboscis:
                    plt.subplot(2,1,2)
                fig=plt.imshow(frame,cmap='gray',vmin=0, vmax=255)
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.xlim(0,1180)
                plt.ylim(700,0)
                # plt.subplots_adjust(top=0.9,bottom=0.01,right=0.98,left=0.02,hspace=0,wspace=0)
                # plt.margins(0,0)
                # plt.title('Frame='+str(f),fontsize=10)          
                plt.title(str(round((f-self.onsets[0][stim_index])/self.frame_rate,1)) + ' seconds')
                for w, well in enumerate(wells):
                    if trk_anno:
                        if len(trks[w][f]) != 0:
                            plt.scatter(trks[w][f][0],trks[w][f][1], c='#FFFFFF',marker=".", alpha=0.5, s=4)#, c='#90EE90')
                    if category_anno:
                        in_cate=False
                        for c, category in enumerate(category_info):
                            if point_inside_polygon(xys[w,f,2],xys[w,f,3],category[2]):
                                tSNE_category_name=category[0]
                                tSNE_category_color=category[1]
                                in_cate=True
                        if in_cate==False:
                            tSNE_category_name='None'
                            tSNE_category_color='#EEEEEE'
                        if tSNE_category_name != 'None':
                            #plot clusters onto frame
                            t=plt.text(xys[w,f,0]+10,xys[w,f,1]+35,tSNE_category_name,color=tSNE_category_color, bbox=dict(boxstyle = "square, pad=0.0"), fontsize=8)
                            if tSNE_category_color == '#222222':
                                tSNE_background_color = '#FFFFFF'
                            else:
                                tSNE_background_color = tSNE_category_color
                            t.set_bbox(dict(facecolor=tSNE_background_color, edgecolor=tSNE_background_color, linewidth=0.01, boxstyle = "square, pad=0.1", alpha=0.2))
                    if behavior_anno: #plot behaviors onto frame
                        if behavior_indices[w,f] != -1:
                            behav_cols = ['#C0BA9C','#863810','#FA0F0C','#78C0A7']
                            # print(w,f,behavior_indices[w,f:f+30], np.shape(xys), np.shape(behavior_indices),np.shape(behavior_names), '\n')
                            t=plt.text(xys[w,f,0]+10, xys[w,f,1], behavior_names[w,f],  color=behav_cols[behavior_indices[w,f]], bbox=dict(boxstyle = "square, pad=0.0"), fontsize=8)
                            t.set_bbox(dict(facecolor=behav_cols[behavior_indices[w,f]], edgecolor=behav_cols[behavior_indices[w,f]], linewidth=0.01, boxstyle = "square, pad=0.1", alpha=0.2))#, edgecolor=behav_cols[behavior_indices[w,f]]))
                    if plot_stimuli:
                        plt.text(20,-10,str(self.frame_data[0][f][2]) + ' ˚C', color='orange')
                        if self.frame_data[0][f][3] == 1:
                            plt.text(20,-40,'Light ON',color='red')
                if plot_proboscis:
                    plt.subplot(3,3,2)
                    plt.ylim(0,3)
                    plt.xlim(-5,5)
                    # plt.plot([0,0],[0,3],color='black',linestyle='dashed')
                    plt.ylabel('Proboscis length (mm)')
                    plt.xlabel('Time (seconds)')
                    well_index = 1
                    plt.title('well ' + str(well_index))
                    seconds_to_plot = 10
                    xvalues = np.linspace(-seconds_to_plot/2, seconds_to_plot/2, seconds_to_plot*self.frame_rate)
                    yvalues = prob_lengths[well_index][int(f-seconds_to_plot/2*self.frame_rate):int(f+seconds_to_plot/2*self.frame_rate)]
                    plt.plot(xvalues,yvalues)
                plt.savefig(newpath+'/frame%05d.png' %(f-fstart), dpi=400)#, bbox_inches='tight') #add pad_inches=0 to ged rid of white margin
                plt.clf()
                plt.close()
                pipein.stdout.flush()
                f += 1
        os.chdir(newpath)
        sp.call([
            'ffmpeg', '-y', '-framerate', '30', '-i', 'frame%05d.png', '-r', '30', '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            'outputraw.mp4']) #, '-start_number', str(f0)
        # for file_name in glob.glob("*.png"): #to remove figure files after creating videos
        #     os.remove(file_name)
        pipein.stdout.close()
        print(f, ' frames processed in ',time.time()-t1,' seconds')

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf                
def generateOutCommand(file_path,frame_size):
        return [ 'ffmpeg',
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-loglevel','error',
            '-s', str(frame_size[1]) + 'x' + str(frame_size[0]), # size of one frame
            '-pix_fmt', 'rgba',
            '-r', '30', # frames per second
            '-i', '-', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            # '-b:v', '10000k', #this gives constant bitrate but not necessarily the highest quality?
            '-q:v', '1', #this gives 1 (max) to 31 (lowest) variable quality bitrate: maybe best for my purposes
            '-vcodec', 'mpeg4', #rawvideo works with fiji, mpeg4 compatible with jaaba?
            file_path
            # ' > /dev/null 2>&1 < /dev/null' #this doesn't seem necessary and was part of troubleshooting
            ]
def avgNestedLists(nested_vals):
    """
    Averages a 2-D array and returns a 1-D array of all of the columns
    averaged together, regardless of their dimensions.
    """
    output = []
    maximum = 0
    for lst in nested_vals:
        if len(lst) > maximum:
            maximum = len(lst)
    for index in range(maximum): # Go through each index of longest list
        temp = []
        for lst in nested_vals: # Go through each list
            if index < len(lst): # If not an index error
                temp.append(lst[index])
        output.append(np.nanmean(temp))
    return output
    
def roundup(x, n=10):
    res = math.ceil(x/n)*n
    if (x%n < n/2)and (x%n>0):
        res-=n
    return res

def point_inside_polygon(x,y,poly):
    #tests if point x,y is inside polygon [[x1,y1],[x2,y2]...], including right margin
    n = len(poly)
    inside =False
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x,p1y = p2x,p2y
    return inside

def fixBehaviorRange(onsets, offsets, start_frame, end_frame):
    #helper method
    #checks to see if onsets or offsets occur outside of start/stop frame 
    #and if so adds an appropriate onset or offset at first or last frame
    if len(onsets) + len(offsets) == 0:
        behavior_onsets = onsets
        behavior_offsets = offsets
    elif len(onsets) < len(offsets):
        behavior_onsets = [start_frame]
        behavior_onsets.extend(onsets)
        behavior_offsets = offsets
    elif len(offsets) < len(onsets):
        behavior_onsets = onsets
        behavior_offsets = offsets
        behavior_offsets.append(end_frame)
    elif onsets[0] > offsets[0]:
        behavior_onsets = [start_frame]
        behavior_onsets.extend(onsets)
        behavior_offsets = offsets
        behavior_offsets.append(end_frame)
    else:
        behavior_onsets = onsets
        behavior_offsets = offsets
    return behavior_onsets, behavior_offsets


def calculate_grid(plate_corners, plate_rows, plate_cols):
    #calculate grid for plates, currently removes last well because it is the thermometer well in optothermocycler
    grid = np.zeros([plate_rows+1, plate_cols+1,2])#last index indicates x,y values
    #populate top and bottom rows
    #top row
    deltax = float(plate_corners[1][0]-plate_corners[0][0])/(plate_cols)
    deltay = float(plate_corners[1][1]-plate_corners[0][1])/(plate_cols)
    for col in range(plate_cols+1):
        grid[0,col,0] = plate_corners[0][0] + col*deltax
        grid[0,col,1] = plate_corners[0][1] + col*deltay
    #bottom row
    deltax = float(plate_corners[2][0]-plate_corners[3][0])/(plate_cols)
    deltay = float(plate_corners[2][1]-plate_corners[3][1])/(plate_cols)
    for col in range(plate_cols+1):
        grid[-1,col,0] = plate_corners[3][0] + col*deltax
        grid[-1,col,1] = plate_corners[3][1] + col*deltay
    for col in range(plate_cols+1):
        deltax = float(grid[-1][col][0]-grid[0][col][0])/(plate_rows)
        deltay = float(grid[-1][col][1]-grid[0][col][1])/(plate_rows)
        for row in range(plate_rows+1):
            grid[row,col,0] = grid[0,col,0] + row*deltax
            grid[row,col,1] = grid[0,col,1] + row*deltay
    
    # calculate boundaries of wells for plate, order of mosquitoes is within each plate top to bottom, left to right
    # the bounds for each well are min-x, max-x, min-y, max-y
    well_bounds = []
    for col in range(plate_cols):
        for row in range(plate_rows):
            well_bounds.append([int((grid[row,col,0]+grid[row+1,col,0])/2), int((grid[row,col+1,0]+grid[row+1,col+1,0])/2),
                                int((grid[row,col,1]+grid[row,col+1,1])/2), int((grid[row+1,col,1]+grid[row+1,col+1,1])/2,)])
    return well_bounds[:-1]

def joinTracks(trial_name, opto_folder, behavior_name, plate_corners, plate_rows, plate_cols):
    #joins mosquito tracks according to xy location in each well in each frame and outputs a file
    #file also contains starts and stops of tracks in each well
    #this method corrects for situations in which tracks overlap in the same well if
    #they jump between wells (known errors with Ctrax output)
    
    #First take into account that there may be multiple split up videos for a single trial
    for root, dirs, files in os.walk(opto_folder + '/processed'):
        trial_dirs = natural_sort([d for d in dirs if d[:len(trial_name)] == trial_name])
        print(trial_dirs)
        break
    #create matrices to track tracks and behavior bouts
    wells = calculate_grid(plate_corners, plate_rows, plate_cols)
    track_frames, behavior_frames = [], []
    for well in wells:
        track_frames.append([])
        behavior_frames.append([])
    
    #iterate through each trial chunk, adding mosquito tracks
    running_len = 0
    for trial_chunk in trial_dirs:
        behavior_mat = loadmat(opto_folder + 'processed/' + trial_chunk + '/scores_' + behavior_name + '.mat')
        trx_mat = loadmat(opto_folder + 'processed/' + trial_chunk + '/trx.mat')
        track_lengths = [np.size(trx_mat['trx'][m].x) for m in range(np.size(trx_mat['trx']))]
        chunk_len = max(track_lengths)
        for w in range(len(wells)):
            for f in range(chunk_len):
                track_frames[w].append(0)
                behavior_frames[w].append(0)
        for m in range(np.size(trx_mat['trx'])):
            if np.size(trx_mat['trx'][m].x) < 2:
                continue
            for f in range(np.size(trx_mat['trx'][m].x)):
                xloc = trx_mat['trx'][m].x[f]
                yloc = trx_mat['trx'][m].y[f]
                for w, well in enumerate(wells):
                    if xloc > well[0] and xloc < well[1] and yloc > well[2] and yloc < well[3]:
                        global_frame = f + running_len + trx_mat['trx'][m].firstframe - 1
                        if global_frame >= len(track_frames[w]):
                            print(w, global_frame, len(track_frames), len(track_frames[w]))
                            print(m, f, np.size(trx_mat['trx'][m].x), running_len, trx_mat['trx'][m].firstframe - 1, chunk_len, trial_chunk)
                            print(track_lengths)
                        track_frames[w][global_frame] = 1
                        if behavior_mat['allScores']['postprocessed'][m][f] == 1:
                            behavior_frames[w][global_frame] = 1
        running_len += chunk_len

    #calculate behavior & track starts and stops
    t0s, t1s, tStarts, tEnds = [],[],[],[]
    for w in range(len(wells)):
        tStarts.append([])
        tEnds.append([])
        tracks = np.ediff1d(track_frames[w], to_begin=[0,0])
        if track_frames[w][0] == 1:
            tStarts[w].append(1)
        tempStarts = np.where(tracks == 1)
        tStarts[w].extend(tempStarts[0].tolist())
        tempEnds = np.where(tracks == -1)
        tEnds[w].extend(tempEnds[0].tolist())
        if track_frames[w][-1] == 1:
            tEnds[w].append(len(track_frames[w]))
        
        t0s.append([])
        t1s.append([])
        bouts = np.ediff1d(behavior_frames[w], to_begin=[0,0])
        if behavior_frames[w][0] == 1:
            t0s[w].append(1)
        tempStarts = np.where(bouts == 1)
        t0s[w].extend(tempStarts[0].tolist())
        tempEnds = np.where(bouts == -1)
        t1s[w].extend(tempEnds[0].tolist())
        if behavior_frames[w][-1] == 1:
            t1s[w].append(len(track_frames[w]))

    if not os.path.exists(opto_folder + '/joined_behavior/'):
        os.makedirs(opto_folder + '/joined_behavior/')
    with open(opto_folder + '/joined_behavior/'+trial_name + '_' + behavior_name + '.txt', 'w') as outFile:
        outFile.write('t0s:\n')
        for well in t0s:
            outFile.write(' '.join(map(str, well)))
            outFile.write('\n')
        outFile.write('t1s:\n')
        for well in t1s:
            outFile.write(' '.join(map(str, well)))
            outFile.write('\n')
        outFile.write('tStarts:\n')
        for well in tStarts:
            outFile.write(' '.join(map(str, well)))
            outFile.write('\n')
        outFile.write('tEnds:\n')
        for well in tEnds:
            outFile.write(' '.join(map(str, well)))
            outFile.write('\n')


def joinTracksNoOverlap(trial_name, opto_folder, behavior_name, plate_corners, plate_rows, plate_cols):
    #joins mosquito tracks according to xy location in each well and outputs a file
    #file also contains starts and stops of tracks in each well
    #this version assumes that each well has no overlapping tracks and that tracks
    #never incorrectly jump between wells
    
    
    behavior_mat = loadmat(opto_folder + 'processed/' + trial_name + '/scores_' + behavior_name + '.mat')
    # If there's a single onset/offset it is recorded as a single int, rather than a list containing a single int
    #this code corrects that
    for i, data in enumerate(behavior_mat['allScores']['t0s']):
        if type(data) == int:
            behavior_mat['allScores']['t0s'][i] = [data]
    for i, data in enumerate(behavior_mat['allScores']['t1s']):
        if type(data) == int:
            behavior_mat['allScores']['t1s'][i] = [data]
    trx_mat = loadmat(opto_folder + 'processed/' + trial_name + '/trx.mat')
    #probing the structure of the trx.mat file
    # print(type(trx_mat))
    # print(trx_mat.keys())
    # print(type(trx_mat['trx']))
    # print(np.shape(trx_mat['trx']))
    # print(type(trx_mat['trx'][0]))
    # # print(vars(trx_mat['trx'][0]))
    # print(type(trx_mat['trx'][0].x))
    # print(np.size(trx_mat['trx'][0].x))
    # print(trx_mat['trx'][0].x[:10])
    # print(trx_mat['trx'][0].y[:10])
    wells = calculate_grid(plate_corners, plate_rows, plate_cols)
    t0s, t1s, tStarts, tEnds = [],[],[],[]
    for well in wells:
        t0s.append([])
        t1s.append([])
        tStarts.append([])
        tEnds.append([])
    for m in range(np.size(trx_mat['trx'])):
        len_track = np.size(trx_mat['trx'][m].x)
        if len_track < 3:
            print('skipping track size:', len_track, 'frames')
            continue
        n_to_average = 10
        xsum, ysum, denom = 0,0,0
        step_size = int(len_track/n_to_average)
        if step_size == 0:
            step_size = 1
        for k in range(0,len_track, step_size):
            denom += 1
            xsum += trx_mat['trx'][m].x[k]
            ysum += trx_mat['trx'][m].y[k]
        xloc = xsum/denom
        yloc = ysum/denom
        for w, well in enumerate(wells):
            if xloc > well[0] and xloc < well[1] and yloc > well[2] and yloc < well[3]:
                tempt0s = behavior_mat['allScores']['t0s'][m]
                tempt1s = behavior_mat['allScores']['t1s'][m]
                if len(tempt0s) != len(tempt1s):
                    print(tempt0s, tempt1s)
                t0s[w].extend(tempt0s)
                t1s[w].extend(tempt1s)
                tStarts[w].append(behavior_mat['allScores']['tStart'][m])
                tEnds[w].append(behavior_mat['allScores']['tEnd'][m])
                break
        else:
            print('no well found:', xloc, yloc)
    if not os.path.exists(opto_folder + '/joined_behavior/'):
        os.makedirs(opto_folder + '/joined_behavior/')
    with open(opto_folder + '/joined_behavior/'+trial_name + '_' + behavior_name + '.txt', 'w') as outFile:
        outFile.write('t0s:\n')
        for well in t0s:
            outFile.write(' '.join(map(str, well)))
            outFile.write('\n')
        outFile.write('t1s:\n')
        for well in t1s:
            outFile.write(' '.join(map(str, well)))
            outFile.write('\n')
        outFile.write('tStarts:\n')
        for well in tStarts:
            outFile.write(' '.join(map(str, well)))
            outFile.write('\n')
        outFile.write('tEnds:\n')
        for well in tEnds:
            outFile.write(' '.join(map(str, well)))
            outFile.write('\n')

def outputStimuli(trial_name, opto_folder, blood_blanket=False):
    #video light on is a list of trials, where each trial has 0 or 1 for every frame corresponding to light on
    video_light_on = []
    with open(opto_folder + '/tracking_output/'+trial_name + '.txt', 'r') as inFile:
        file_contents = inFile.readlines()[1:]
        for j in range(len(file_contents)-1): 
            line_split = file_contents[j].split()
            video_light_on.append(int(line_split[0]))

    stim_difference = [] #each frame where stimulus onset (=1) or stimulus offset (=-1) has occurred since the previous timepoint
    for i in range(len(video_light_on)-1):
        stim_difference.append(video_light_on[i+1] - video_light_on[i])
    #onsets and offset frames for video data
    video_onset_indices = []
    video_offset_indices = []
    for i in range(len(stim_difference)):
        if stim_difference[i] == 1:
            video_onset_indices.append(i)
        elif stim_difference[i] == -1:
            video_offset_indices.append(i)
    #arduino data is stored by trial and contains the trial time in ms, temperature in degrees C, light status, and heat ramp number
    arduino_data = []
    with open(opto_folder + '/Arduino_to_file/'+trial_name + '.txt', 'r+', encoding="utf-8") as inFile:
        file_contents = inFile.readlines()[15:]
        for j in range(len(file_contents)-2): 
            line_split = file_contents[j].split()
            arduino_data.append([int(line_split[0]), float(line_split[1]), int(line_split[4]), int(line_split[5])])
    arduino_light_onsets = [] #each arduino index where arduino marks light turns on
    arduino_temp_onsets = [] #each arduino index where temperature rises above 25˚C
    arduino_temp_offsets = [] #each arduino index where temperature rises above 25˚C
    av_temp50 = arduino_data[0][1]
    av_temp10 = arduino_data[0][1]
    last_temp_onset = 0
    last_temp_offset = 0
    for i in range(1, len(arduino_data)):
        if arduino_data[i][1] > 26.0 and av_temp50 < 25.9 and i-last_temp_onset > 600: #temp onsets must be spaced by at least 60 seconds
            last_temp_onset = i
            arduino_temp_onsets.append(i)
        if arduino_data[i][1] < av_temp10 and i-last_temp_offset > 1000 and i-last_temp_onset < 150: #temp offsets must be spaced by at least 100 seconds
            last_temp_offset = i
            arduino_temp_offsets.append(i)
        if arduino_data[i][2] - arduino_data[i-1][2] == 1:
            arduino_light_onsets.append(i)
        av_temp50 = 0.98*av_temp50 + 0.02*arduino_data[i][1]
        av_temp10 = 0.9*av_temp10 + 0.1*arduino_data[i][1]
    if len(arduino_light_onsets) != len(video_onset_indices):
        print(trial_name, 'Arduino and video disagree on light onsets!!')
        print(arduino_light_onsets)
        print(video_onset_indices)
    if len(arduino_temp_offsets) != len(arduino_temp_onsets):
        print(trial_name, 'Temp onsets and offsets disagree!!')
        if blood_blanket:
            print('Blood blanket, added final frame as heat offset')
            arduino_temp_offsets.append(i)
        print(arduino_temp_onsets)
        print(arduino_temp_offsets)
    i = 0
    while i < len(arduino_temp_onsets):
        try:
            if arduino_temp_offsets[i]-arduino_temp_onsets[i] < 20:
                print('deleted short temp stimulus:', arduino_temp_onsets[i], arduino_temp_offsets[i])
                del arduino_temp_offsets[i]
                del arduino_temp_onsets[i]
            else:
                i += 1
        except IndexError:
            print('Temp onsets and offsets disagree!!')
            i += 1
    #calculate y=mx+b where y is arduino light index and x is video light index
    if len(arduino_light_onsets) > 1:
        m = (arduino_light_onsets[-1]-arduino_light_onsets[0])/(video_onset_indices[-1]-video_onset_indices[0])
    else:
        m = 1.0/3.0
    meany = sum(arduino_light_onsets)/len(arduino_light_onsets)
    meanx = sum(video_onset_indices)/len(video_onset_indices)
    b = meany-m*(meanx)
    #convert temp_onsets & offsets to video_frames and zipper
    video_heat_onsets = [] 
    video_heat_offsets = [] 
    for onset, offset in zip(arduino_temp_onsets, arduino_temp_offsets):
        video_heat_onsets.append(round((onset-b)/m))
        video_heat_offsets.append(round((offset-b)/m))
    h, l = 0,0
    all_onsets = []
    onset_type = []
    all_offsets = []
    while h < len(video_heat_onsets) or l < len(video_onset_indices):
        if h == len(video_heat_onsets):
            all_onsets.append(video_onset_indices[l])
            all_offsets.append(video_offset_indices[l])
            onset_type.append('L')
            l += 1
        elif l == len(video_onset_indices):
            all_onsets.append(video_heat_onsets[h])
            all_offsets.append(video_heat_offsets[h])
            onset_type.append('H')
            h += 1
        elif video_heat_onsets[h] < video_onset_indices[l]:
            all_onsets.append(video_heat_onsets[h])
            all_offsets.append(video_heat_offsets[h])
            onset_type.append('H')
            h += 1
        else:
            all_onsets.append(video_onset_indices[l])
            all_offsets.append(video_offset_indices[l])
            onset_type.append('L')
            l += 1
    zippered_stimuli_info = []
    for i, frame in enumerate(video_light_on):
        conversion_index = round(m*i + b)
        if conversion_index < 0:
            print('m', m, 'i', i, 'b', b)
            print('video started before Arduino to file')
        if conversion_index >= len(arduino_data):
            print('video longer than Arduino output. video frame:', i, 'arduino frame', conversion_index, 'length of Arduino file ', len(arduino_data))
            break
        arduino_frame = arduino_data[conversion_index]
        zippered_stimuli_info.append([arduino_frame[0], arduino_frame[1], frame, arduino_frame[2], arduino_frame[3]])
    if not os.path.exists(opto_folder + '/joined_stimuli/'):
        os.makedirs(opto_folder + '/joined_stimuli/')
    with open(opto_folder + '/joined_stimuli/'+trial_name + '.txt', 'w') as outFile:
        outFile.write('Stimulus Onsets: ')
        outFile.write(' '.join(map(str, all_onsets)))
        outFile.write('\nStimulus Offsets: ')
        outFile.write(' '.join(map(str, all_offsets)))
        outFile.write('\nStimulus Types: ')
        outFile.write(' '.join(map(str, onset_type)))
        outFile.write('\nframe\tmsArduino\tCelsius\tlightVideo\tlightArduino\theatRamp')
        for i,frame in enumerate(zippered_stimuli_info):
            outFile.write('\n'+str(i+1)+'\t')
            outFile.write('\t'.join(map(str, frame)))

def joinStimuliTracks(plate_names, opto_folder, behavior_names, plate_corners, plate_rows, plate_cols, join_stimuli=False, blood_blanket=False):
    for plate in plate_names:
        if join_stimuli:
            print(plate, 'stimuli')
            outputStimuli(plate, opto_folder, blood_blanket)
        for behavior in behavior_names:
            print(plate, behavior)
            joinTracks(plate, opto_folder, behavior, plate_corners, plate_rows, plate_cols)
    

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth



def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)