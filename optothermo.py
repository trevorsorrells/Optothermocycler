# -*- coding: utf-8 -*-
# Analysis of jaaba behavior classification for mosquito optothermocycler setup
# python 3.5


import numpy as np
import scipy.io as spio
import math 
import os
import re
from random import shuffle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
# from matplotlib.patches import Polygon

from statistics import median
from scipy.stats import friedmanchisquare
from scipy.stats import kruskal
from scipy.stats import chi2_contingency
from scipy.stats import spearmanr
import scikit_posthocs as ph

matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

#parameters & packages for state estimation analysis 
from sklearn.manifold import TSNE
import time
velocity_step_ctrax = 200 #in milliseconds
window_step_size = 10 #in seconds
import subprocess as sp

class OptoThermoExp:
    def __init__(self, trial_list, opto_folder, behaviors, behavior_colors, frame_rate, velocity_step_frames):
        
        self.opto_folder = opto_folder
        self.frame_rate = frame_rate
        self.velocity_step_frames = velocity_step_frames
        self.behaviors = behaviors
        self.behavior_colors = behavior_colors
        self.trial_names = trial_list
        
        self.onsets = []
        self.offsets = []
        self.stimulus_type = []
        self.frame_data = []
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
                    stimulus_off = (self.offsets[0][stim_type[i][0]] - self.onsets[0][stim_type[i][0]])*self.velocity_step_frames/self.frame_rate
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
                plt.ylim(ymin,len(ethogram_data[0]))
                plt.title(stimulus_names[st])
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
                #ethogram_sort.sort(reverse = True)
                #plot gaps and behavior bouts
                for i, entry in enumerate(ethogram_sort):
                    for ongap, offgap in zip(entry[1][0], entry[1][1]):
                        plt.fill(np.array([ongap,ongap,offgap,offgap]),np.array([i+1,i,i,i+1]), 'black')
                    for b in range(len(self.behaviors)):
                        for onset, offset in zip(entry[b+2][0], entry[b+2][1]):
                            plt.fill(np.array([onset,onset,offset,offset]),np.array([i+1,i,i,i+1]), self.behavior_colors[b])
                plt.savefig(newpath + '/onsetindex_' + str(st) + '.pdf', format='pdf') 
                plt.close('all')
            for_addition.append(behavior_hist)
            #behavior_graph
            if behavior_graph:
                plt.figure()
                plt.ylim(0,.5)
                plt.xlim(-prestim, poststim)
                x_axis = np.linspace(-prestim, poststim, int(n_timepoints/step_size))
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
                    plt.plot(x_axis, behavior_to_plot, color = self.behavior_colors[b])
                plt.title(stimulus_names[st])
                plt.savefig(newpath + '/prop_behavior_' + str(st) + '.pdf', format='pdf')
                plt.close('all')
            if stats_file:
                stats_behavior = 'probe5'
                b = self.behaviors.index('probe5')
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
                    if len(current_stats) == 4 and current_behavior <= (current_stats[1]-prestim_average)*1/3+prestim_average:
                        current_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                    if len(current_stats) == 5 and current_behavior <= (current_stats[1]-prestim_average)*1/4+prestim_average:
                        current_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                    if len(current_stats) == 6 and current_behavior <= prestim_average:
                        current_stats.append(round((i+window_frames/2)*self.velocity_step_frames/self.frame_rate-prestim))
                    i += 1
                stats_data.append(current_stats)
        if stats_file:
            newpath = self.opto_folder + '/statistics/' 
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            with open(newpath + treatment_name + '.txt', 'w') as outFile:
                outFile.write('behavior examined '+ stats_behavior +'\n'+'window size in seconds '+ str(window_seconds))
                outFile.write('\ntime averaged for baseline in seconds '+ str(prestim))
                outFile.write('\n\nstimulus\ttimetomax\tmaxprop\tt3/4\tt1/2\ttau(1/3)\tt1/4\tbaseline')
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
            offsets = [-65,-20, 0, 20,65]
            additions = []
            for o, offset in enumerate(offsets):
                additions.append([])
                modoffset = round(offset*self.frame_rate/step_size)
                plt.figure()
                plt.ylim(0,.5)
                plt.xlim(-prestim+120, poststim-120)
                x_axis = np.linspace(-prestim, poststim, int(n_timepoints/step_size))
                for b, behavior in enumerate(self.behaviors):
                    behavior_to_plot = []
                    if offset <= 0:
                        for i in range(len(collapsed_behavior[0][b])):
                            if i+modoffset < 0 or i+modoffset >= len(collapsed_behavior[1][b]):
                                behavior_to_plot.append(collapsed_behavior[0][b][i])
                            else:
                                behavior_to_plot.append(collapsed_behavior[0][b][i] + collapsed_behavior[1][b][i+modoffset])
                    else:
                        for i in range(len(collapsed_behavior[1][b])):
                            if i+modoffset < 0 or i+modoffset >= len(collapsed_behavior[0][b]):
                                behavior_to_plot.append(collapsed_behavior[1][b][i])
                            else:
                                behavior_to_plot.append(collapsed_behavior[1][b][i] + collapsed_behavior[0][b][i-modoffset])
                    plt.plot(x_axis, behavior_to_plot, color = self.behavior_colors[b])
                    additions[o].append(behavior_to_plot)
                plt.title(offset)
                plt.savefig(newpath + '/add_offset_' + str(offset) + '.pdf', format='pdf')
                plt.close('all')
            for o, offset in enumerate(offsets):
                plt.figure()
                plt.ylim(-.3,.3)
                plt.xlim(-prestim+120, poststim-120)
                x_axis = np.linspace(-prestim, poststim, int(n_timepoints/step_size))
                for b, behavior in enumerate(self.behaviors):
                    subtraction = np.subtract(collapsed_behavior[o+2][b],additions[o][b])
                    plt.plot(x_axis, subtraction, color = self.behavior_colors[b])
                plt.title(offset)
                plt.savefig(newpath + '/subtract_offset_' + str(offset) + '.pdf', format='pdf')
                plt.close('all')

    def statistics_test(self, behavior_name, genotype_name, stimulus_onsets, stimulus_names, seconds):
        #computes statistics and makes graph comparing responses to stimuli, averaged over seconds
        #after stimulus onset, also averaged over each stimulus repeat, giving a total of one data point
        #per mosquito
        prestim, poststim = 60, seconds
        n_timepoints = int((prestim + poststim) * self.frame_rate + 1)
        x_axis = np.linspace(-prestim, poststim, n_timepoints)
        test_name = '_'.join([behavior_name,str(seconds)+'s',str(len(stimulus_names))+'stims'])
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
                    stim_type_range.append([onset,onset+seconds*self.frame_rate])
                frame_ranges[t].append(stim_type_range)
        print(np.shape(frame_ranges))  #organized by trial, stimulus, rep, start/stop
        onsets_hist = []
        [onsets_hist.append([]) for x in stimulus_onsets]
        response_values = []
        [response_values.append([]) for x in stimulus_onsets]
        for t in range(len(self.behavior_data)):
            for m in range(len(self.behavior_data[t][behavior_name]['t0s'])):
                for s, stim_type in enumerate(frame_ranges[t]):
                    values_type = []
                    for stim_range in stim_type:
                        temp_onsets = [x for x in self.behavior_data[t][behavior_name]['t0s'][m] if x >= stim_range[0] and x < stim_range[1]]
                        temp_offsets = [x for x in self.behavior_data[t][behavior_name]['t1s'][m] if x >= stim_range[0] and x < stim_range[1]]
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
                        centered_onsets = [(x-stim_range[0])/self.frame_rate for x in self.behavior_data[t][behavior_name]['t0s'][m] if x >= stim_range[0]-60*self.frame_rate and x < stim_range[1]]
                        onsets_hist[s].extend(centered_onsets)
                        if len(final_onsets) + len(final_offsets) == 0:
                            values_type.append(0.0)
                        else:
                            values_type.append(sum(np.subtract(np.array(final_offsets), np.array(final_onsets)))/(stim_range[1]-stim_range[0]))
                    if len(values_type) > 0:
                        response_values[s].append(sum(values_type)/len(values_type))
        response_array = np.array(response_values)
        print(np.shape(response_array))
        successes = response_array > 0.01
        contingency = []
        for column in successes:
            contingency.append([sum(column), len(column)-sum(column)])
        with open(newpath + '/friedman_nemenyi.txt', 'w') as outFile:
            outFile.write('categories, mosquitoes: ')
            outFile.write(str(np.shape(response_array)))
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
            outFile.write(str(kruskal(*response_array)))
            outFile.write('\nScikit_posthocs Nemenyi')
            outFile.write(str(ph.posthoc_nemenyi(response_array)))
            outFile.write('\n\n')
            outFile.write('ChiSquared Test\n' + str(np.array(contingency))+'\n')
            stat, p, dof, expected = chi2_contingency(contingency)
            outFile.write('test statistic ='+str(stat)+', p ='+str(p)+', dof ='+str(dof)+'\nexpected values:\n'+str(expected))
        with open(newpath + '/data_table.txt', 'w') as outFile:
            outFile.write('mosquito\tbehavior_prop\tstimulus\n')
            for i in range(len(response_values[0])):
                for j in range(len(response_values)):
                    outFile.write(str(i) + '\t' + str(response_values[j][i]) + '\t' + stimulus_names[j].replace(' ', '') + '\n')
        response_differences = np.subtract(np.array(response_values[1]), np.array(response_values[0]))
        plt.hist(response_differences, bins=20, histtype='step', linewidth=2)
        plt.title(stimulus_names[1] + ' minus ' + stimulus_names[0] + ' behavior ' + behavior_name)
        plt.savefig(newpath + '/' +'difference_hist01.pdf', format='pdf')
        plt.close()
        plt.figure()
        plt.title(' behavior ' + behavior_name + ' ' + str(seconds) + ' seconds post stimulus onset')
        plt.xlim(0.5, len(response_values)+0.5)
        for i in range(len(response_values[0])):
            xvals, yvals = [], []
            for j in range(len(response_values)):
                xvals.append(j+1)
                yvals.append(response_values[j][i])
            plt.plot(xvals,yvals, alpha=0.1, color='black', marker='.')
        xvals, yvals = [], []
        for j in range(len(response_values)):
            xvals.append(j+1)
            yvals.append(median(response_values[j]))
        plt.plot(xvals, yvals, color='red', marker='.', markersize=10.0)
        plt.xticks(xvals, stimulus_names)
        plt.ylim(0,.3)
        plt.savefig(newpath + '/' +'prop_response.pdf', format='pdf')        
        plt.close()
        plt.figure()
        plt.boxplot(response_values)
        plt.savefig(newpath + '/' +'boxplot.pdf', format='pdf')        
        plt.close()
        plt.figure()
        plt.violinplot(response_values)
        plt.boxplot(response_values)
        plt.savefig(newpath + '/' +'violin2.pdf', format='pdf')        
        plt.close()
        for s, stim_type in enumerate(stimulus_names):
            plt.figure()
            nbins = (seconds + 60)*.2
            bins_sequence = np.linspace(-60, seconds, nbins)
            plt.ylim(0,2.5)
            counts, bin_edges = np.histogram(onsets_hist[s], bins=bins_sequence)
            plt.plot(bin_edges[1:], counts/len(response_values[0]), color = 'black')
            plt.savefig(newpath + '/' +'onsets_hist' + stim_type +'.pdf', format='pdf')        
            plt.close()

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

    def stimulusGraph(self, trial_names, min_seconds, pos_seconds):
	#graph stimuli in each trial, with a window of min_seconds and pos_seconds around it
        newpath = self.opto_folder + '/graphs/stimuli/' 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        for t in range(len(self.behavior_data)):
            print(trial_names[t])
            plt.figure()
            num_stims = 1
            for s in range(1, len(self.onsets[t])):
                if (self.onsets[t][s]-self.onsets[t][s-1])/self.frame_rate > pos_seconds:
                    num_stims += 1
            plot_count = 0
            last_stimulus_index = 0
            for s in range(len(self.onsets[t])):
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
                plt.ylim(20,35)
                plt.plot(xvalues, yvalues, color='orange')
                plt.text(-min_seconds, 30, 'onset '+str(s))
                for s2 in range(len(self.onsets[t])):
                    if self.onsets[t][s2] > start_graph and self.onsets[t][s2] < stop_graph and self.stimulus_type[t][s2] == 'L':
                        start_light = (self.onsets[t][s2]-self.onsets[t][s])/self.frame_rate
                        stop_light = (self.offsets[t][s2]-self.onsets[t][s])/self.frame_rate
                        plt.fill(np.array([start_light,start_light,stop_light,stop_light]),np.array([20,35,35,20]), color='red', alpha=0.3, linewidth=0.0)
            plt.savefig(newpath + '/' + trial_names[t] +'.pdf', format='pdf')
            plt.close()
        
    def output_cluster_features(self, stim_indices, prepost_seconds, window_size_seconds):
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
                    #add 1s and 0s  of whether mosquito is exhibiting behavior at each timepoint
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
            velocity_mat = loadmat(os.path.join(self.opto_folder, 'processed', self.trial_names[t], 'trx.mat'))
            # print(type(velocity_mat))
            # print(velocity_mat.keys())
            # print(type(velocity_mat['trx']))
            # print(len(velocity_mat['trx']))
            # print(type(velocity_mat['trx'][0]))
            # print(vars(velocity_mat['trx'][0]))
            # print(type(velocity_mat['trx'][0].x))
            # print(np.size(velocity_mat['trx'][0].x))
            #first assign xy values to wells
            plate_corners = [[20,20],[1066,16],[1065,690],[20,689]]
            plate_rows, plate_cols = 3,5
            wells = calculate_grid(plate_corners, plate_rows, plate_cols)
            xys = np.empty((len(wells), len(velocity_mat['timestamps']), 2))
            for track in velocity_mat['trx']:
                f = track.firstframe - 1
                for xloc, yloc in zip(track.x, track.y):
                    for w, well in enumerate(wells):
                        if xloc > well[0] and xloc < well[1] and yloc > well[2] and yloc < well[3]:
                            xys[w,f,0] = xloc
                            xys[w,f,1] = yloc
                    f += 1
            # print(xys[0,:10])
            #calculate velocity
            velocities = np.empty((len(wells), len(velocity_mat['timestamps'])))
            velocities[:] = np.NaN
            for m in range(len(xys)):
                for f in range(len(velocity_mat['timestamps'])):
                    try:
                        first_frame = round(f-frame_step/2)
                        last_frame = round(f+frame_step/2)
                        x1, x2, y1, y2 = xys[m,first_frame,0], xys[m,last_frame,0], xys[m,first_frame,1], xys[m,last_frame,1]
                        velocities[m,f] = np.sqrt((x2-x1)**2 + (y2-y1)**2)
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

            
            for m in range(len(behavior_hist)):
                for st, stim_type in enumerate(stim_indices): 
                    for i in stim_type[t]:
                        start_windows = int(self.onsets[t][i] - prepost_seconds[0]*self.frame_rate/self.velocity_step_frames)
                        end_windows = int(self.onsets[t][i] + prepost_seconds[1]*self.frame_rate/self.velocity_step_frames + 1)
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
                            for b in range(4):
                                straightmsk = (np.asarray(behavior_hist[m][b][first_frame:last_frame])-1)*-1
                                expandedmsk = np.add(straightmsk, np.roll(straightmsk, int(frame_step/2)))
                                msk = np.add(expandedmsk, np.roll(straightmsk, int(-frame_step/2)))
                                if len(msk) != len(velocities[m,first_frame:last_frame]):
                                    print(first_frame, last_frame, len(behavior_hist[m][b]), len(velocities[m]))
                                behavior_speed = np.mean(np.ma.masked_array(velocities[m,first_frame:last_frame], mask = msk))
                                if np.ma.is_masked(behavior_speed):
                                    behavior_speed = 0.0
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

    def output_tSNE(self, window_size_seconds, perplex=30):
        #this method reads in files with features calculated over windows and outputs the tSNE axes into the file
        window_data = []
        window_array = []
        br=False
        for t, trial_name in zip(range(len(self.behavior_data)), self.trial_names):
            window_data.append([])
            with open(self.opto_folder + '/cluster_features/'+trial_name +'_' + str(window_size_seconds) + '.txt', 'r') as inFile:
                all_lines = inFile.readlines()
                line1 = all_lines[0].split()
                cn=-1 
                end=-1
                while line1[cn][:4] =='tSNE': 
                    if br==True:
                        break
                    if line1[cn].split('_')[-1] == str(perplex): #test if have run tSNE with same perplexity
                        rp=input('t-SNE run with the same perplexity exists, do you want to overwrite it(y/n or q to quit):')
                        while rp != 'y' and rp != 'n':
                            if rp == 'q':
                                quit()
                            rp=input('t-SNE run with the same perplexity exists, do you want to overwrite it(y/n or q to quit):')
                        if rp == 'y':
                            end = cn-1 #don't include past tSNE output if present in the file
                            br=True
                    if br==True:
                        break
                    cn+=-2
                for line in all_lines[1:]:
                    if end == -1 or rp == 'n':
                        window_data[t].append(line.split()[:])
                        window_array.append(line.split()[:])
                    elif end == -2:
                        window_data[t].append(line.split()[:end])
                        window_array.append(line.split()[:end])
                    else:
                        window_data[t].append(line.split()[:end]+line.split()[end+2:])
                        window_array.append(line.split()[:end]+line.split()[end+2:])
        window_array = np.asarray(window_array)
        print(np.shape(window_array))
        time_start = time.time()
        behavior_tsne = TSNE(perplexity=perplex).fit_transform(window_array[:,3:])
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
        i = 0
        for t, trial_name in zip(range(len(self.behavior_data)), self.trial_names):
            with open(self.opto_folder + '/cluster_features/'+trial_name +'_' + str(window_size_seconds) + '.txt', 'w') as outFile:
                if end == -1 or rp == 'n':
                    outFile.write('\t'.join(line1[:]))
                elif end == -2:
                    outFile.write('\t'.join(line1[:end]))
                else:
                    outFile.write('\t'.join(line1[:end]+line1[end+2:]))
                outFile.write('\ttSNE1_'+str(window_size_seconds)+'s_'+str(perplex)+'per\ttSNE2_'+str(window_size_seconds)+'s_'+str(perplex)+'\n')
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
                cn=-1 
                end=-1
                if line1[-1][:4] != 'tSNE':
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
            ax.scatter(window_array[:,x][subset_mask],window_array[:,y][subset_mask], marker='.', linewidths=0.0, c=np.array(cols)[subset_mask])
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
            plt.savefig(self.opto_folder + '/graphs/tSNE/' + experiment_name + '/dombehav_' + str(window_size_seconds) + 'sec_' + str(perplex) + 'per.pdf', format='pdf')
            plt.close()
        behavior_none_names = self.behaviors
        behavior_none_names.append('none')
        behavior_none_colors = self.behavior_colors
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
            plt.scatter(window_array[:,x][subset_mask],window_array[:,y][subset_mask], marker='.', linewidths=0.0, c=window_array[:,1][subset_mask], cmap=plt.get_cmap('hsv'))
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
                

    
    def category_tSNE(self, experiment_name, window_size_seconds, stimulus_onsets, stimulus_names, prepost_seconds, category_info, ethogram=True, category_linegraph=True, category_graphs=True):
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
                if line1[-1][:4] !='tSNE':
                    print('tSNE not calculated for this trial!')
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
        
        #graph tSNE cluster categories onto individual mosquito ethograms

        for st, stim_type in enumerate(stimulus_onsets): #iterate through stimulus types
            print('graph for stimuli ', stim_type, stimulus_names[st])
            stimulus_on = 0
            i=0
            while True:
                if len(stim_type[i]) > 0:
                    stimulus_off = (self.offsets[0][stim_type[i][0]] - self.onsets[0][stim_type[i][0]])*velocity_step_frames/frame_rate
                    break
                i += 1
            
            behavior_hist = [[]] #behavior_hist structure is non-categorized, then categories, mosquitoes, windows indicated by 0, 1
            [behavior_hist.append([]) for category in category_info]
            ethogram_data = [[]] #ethogram structure is non-categorized + categories, mosquitoes, [onsets,offsets]
            [ethogram_data.append([]) for category in category_info] #
            global_m = 0
            for t in range(len(self.behavior_data)):
                for i in stim_type[t]:
                    start_frame = int(self.onsets[t][i] - prepost_seconds[0]*frame_rate/velocity_step_frames)
                    end_frame = int(self.onsets[t][i] + prepost_seconds[1]*frame_rate/velocity_step_frames + 1)
                    if t >0:
                        print(t, i)
                    # print('frames ',start_frame, end_frame, stimulus_names[st])
                    for m in range(len(window_data[t])):
                        [cat.append([]) for cat in behavior_hist]
                        [cat.append([[],[]]) for cat in ethogram_data]
                        last_window_cat = -1
                        for window in window_data[t][m]:
                            if int(window[1]) < start_frame or int(window[1]) > end_frame:
                                continue
                            for c, category in enumerate(category_info):
                                #test if window is a member of the cluster of points in category
                                #if float(window[-2]) > category[2][0] and  float(window[-2]) < category[2][1] and float(window[-1]) > category[2][2] and  float(window[-1]) < category[2][3] :
                                if point_inside_polygon(float(window[-2]),float(window[-1]),category[2]):#[(category[2][0],category[2][1]),(category[2][2],category[2][3]),(category[2][4],category[2][5])]) 
                                    windowcats[c+1].append(window)
                                    behavior_hist[c+1][global_m].append(1)
                                    behavior_hist[0][global_m].append(0)
                                    for c2 in range(len(category_info)):
                                        if c2 != c:
                                            behavior_hist[c2+1][global_m].append(0)
                                    if last_window_cat != c+1:
                                        if last_window_cat != -1:
                                            ethogram_data[last_window_cat][global_m][1].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                                        ethogram_data[c+1][global_m][0].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                                        # print(last_window_cat, c+1)
                                        # print(ethogram_data[last_window_cat][m][1], ethogram_data[c+1][m][0])
                                        last_window_cat = c+1
                                    break
                            else:
                                windowcats[0].append(window)
                                behavior_hist[0][global_m].append(1)
                                for c2 in range(len(category_info)):
                                    behavior_hist[c2+1][global_m].append(0)
                                if last_window_cat != 0:
                                    if last_window_cat != -1:
                                        ethogram_data[last_window_cat][global_m][1].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                                    ethogram_data[0][global_m][0].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                                    last_window_cat = 0
                        ethogram_data[last_window_cat][global_m][1].append((int(window[1])-self.onsets[t][i])*velocity_step_frames/frame_rate+window_step_size/2)
                        global_m += 1
            # print(windowcats[0][:10])
            # print(np.shape(ethogram_data))
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
                    next_entry = [total_behavior,ethogram_data[0][m]]
                    for j in range(len(category_info)):
                        next_entry.append(ethogram_data[j+1][m])
                    ethogram_sort.append(next_entry)
                ethogram_sort.sort(reverse = True)
                for window in ethogram_sort:
                    print(window)
                # print(len(ethogram_sort))
                # print(len(ethogram_sort[14]))
                # print(ethogram_sort[14])
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
                step_size = 10
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
                    list_to_array = np.asarray(behavior_to_plot)
                    behavior_hist_array = np.nanmean(list_to_array, 0)
                    plt.plot(x_axis, behavior_hist_array, color=category_info[c][1])
                plt.title(stimulus_names[st])
                plt.savefig(newpath + '/prop_category_' + str(st) + '.pdf', format='pdf')
                plt.close('all')

        if category_graphs:
            for c,cat in enumerate(windowcats):
                windowcats[c] = np.asarray(cat).astype(np.float)
            # print(type(windowcats[0]))
            # print(np.shape(windowcats[0]))
            labels = ['none']
            colors = ['#EEEEEE']
            for cat in category_info:
                colors.append(cat[1])
            [labels.append(cat[0]) for cat in category_info]
            for b, behavior in enumerate(behavior_none_names):
                current_data = []
                for cat in windowcats:
                    print('cat shape ',np.shape(cat))
                    current_data.append(cat[:,b+4])
                plt.figure()
                violin_parts = plt.violinplot(current_data,linewidth=0)
                for col, pc in zip(colors, violin_parts['bodies']):
                    pc.set_facecolor(col)
                bplot = plt.boxplot(current_data, sym='')
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_color(color)
                plt.xlabel(labels)
                plt.savefig(newpath + '/cats_' + behavior + '.pdf', format='pdf')        
            for i, name in zip([13,14],['walkvel','probevel']):
                current_data = []
                for cat in windowcats:
                    current_data.append(cat[:,i])
                plt.figure()
                violin_parts = plt.violinplot(current_data)
                for col, pc in zip(colors, violin_parts['bodies']):
                    pc.set_facecolor(col)
                bplot = plt.boxplot(current_data, sym='')
                for patch, color in zip(bplot['boxes'], colors):
                    patch.set_color(color)
                plt.xlabel(labels)
                plt.savefig(newpath + '/cats_' + name + '.pdf', format='pdf')        
                
    def annotate_videos(self, output_name, trial_name):            

        #read in xy locations
        velocity_mat = loadmat(os.path.join(self.opto_folder, 'processed', self.trial_names[t], 'trx.mat'))
        # print(type(velocity_mat))
        # print(velocity_mat.keys())
        # print(type(velocity_mat['trx']))
        # print(len(velocity_mat['trx']))
        # print(type(velocity_mat['trx'][0]))
        # print(vars(velocity_mat['trx'][0]))
        # print(type(velocity_mat['trx'][0].x))
        # print(np.size(velocity_mat['trx'][0].x))
        #first assign xy values to wells
        plate_corners = [[20,20],[1066,16],[1065,690],[20,689]]
        plate_rows, plate_cols = 3,5
        wells = calculate_grid(plate_corners, plate_rows, plate_cols)
        xys = np.empty((len(wells), len(velocity_mat['timestamps']), 2))
        for track in velocity_mat['trx']:
            f = track.firstframe - 1
            for xloc, yloc in zip(track.x, track.y):
                for w, well in enumerate(wells):
                    if xloc > well[0] and xloc < well[1] and yloc > well[2] and yloc < well[3]:
                        xys[w,f,0] = xloc
                        xys[w,f,1] = yloc
                f += 1
                

        frame_size = (700,1180)
        outcommand = generateOutCommand(output_name) #trial1_annotated
        print(' '.join(outcommand))
        pipeout = sp.Popen(outcommand, stdin=sp.PIPE)#, stderr=sp.PIPE)
        #read in tSNE information here
        #This looks for videos in the project_folder/trial_name, but it will need to be changed to looking in project_folder/processed
        for root, dirs, files in os.walk(project_folder, processed):
                avi_files = [f for f in files if f[-3:] == 'avi' and f[0] != '.']
                avi_files = sorted(avi_files)
                print(avi_files)
                #find the avi_files that correspond to the trial you are annotating
                for i in range(len(avi_files)):
                    # if i > 0: 
                    #     break
                    t1 = time.time()
                    incommand = [ 'ffmpeg',
                            '-i', os.path.join(trial_name, avi_files[i]),
                            '-f', 'image2pipe',
                            '-pix_fmt', 'gray',
                            '-vcodec', 'rawvideo', '-'] 
                    pipein = sp.Popen(incommand, stdout = sp.PIPE, bufsize=10**8)
                    f=0
                    while f < 1: #True:
                        raw_image = pipein.stdout.read(frame_size[0]*frame_size[1])
                        image =  np.frombuffer(raw_image, dtype='uint8')
                        if np.size(image) < frame_size[0]*frame_size[1]:
                            print(np.size(image))
                            pipein.stdout.flush()
                            break
                        f += 1
                        frame = image.reshape(frame_size)
                        plt.figure()
                        plt.imshow(frame)
                        #something like: plt.txt(x,y, tSNE_category_color)
                        plt.show()
        #plot clusters onto frame
        #I’m not sure if you can directly write the matplotlib to video?
                        pipeout.stdin.write(frame.tostring())
                        frame_number += 1
                        pipein.stdout.flush()
                    pipein.stdout.close()
                    print(f, ' frames processed in ',time.time()-t1,' seconds')
                out, err = pipeout.communicate()
                if err != None:
                    with open(os.path.join(project_folder,trial_name,'ffmpeg_errors.txt'),'w') as outFile:
                        outFile.write(err)
                out, err = bkgout.communicate()
                if err != None:
                    with open(os.path.join(project_folder,trial_name,'ffmpeg_bkg_errors.txt'),'w') as outFile:
                        outFile.write(err)
                
def generateOutCommand(file_name):
        return [ 'ffmpeg',
            '-y', # (optional) overwrite output file if it exists
            '-f', 'rawvideo',
            '-loglevel','error',
            '-s', str(frame_size[1]) + 'x' + str(frame_size[0]), # size of one frame
            '-pix_fmt', 'gray',
            '-r', '30', # frames per second
            '-i', '-', # The imput comes from a pipe
            '-an', # Tells FFMPEG not to expect any audio
            # '-b:v', '10000k', #this gives constant bitrate but not necessarily the highest quality?
            '-q:v', '1', #this gives 1 (max) to 31 (lowest) variable quality bitrate: maybe best for my purposes
            '-vcodec', 'mpeg4', #rawvideo works with fiji, mpeg4 compatible with jaaba?
            'processed/' + file_name + '.mp4'
            # ' > /dev/null 2>&1 < /dev/null' #this doesn't seem necessary and was part of troubleshooting
            ]
    
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
    
    # calculate boundaries of wells for plate, order of mosquitoes is within each plate top to bottom, right to left
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
                            print(m, f, running_len, trx_mat['trx'][m].firstframe - 1, chunk_len, trial_chunk)
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

def outputStimuli(trial_name, opto_folder):
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
        print(arduino_temp_onsets)
        print(arduino_temp_offsets)
    i = 0
    while i < len(arduino_temp_onsets):
        if arduino_temp_offsets[i]-arduino_temp_onsets[i] < 30:
            print('deleted short temp stimulus:', arduino_temp_onsets[i], arduino_temp_offsets[i])
            del arduino_temp_offsets[i]
            del arduino_temp_onsets[i]
        else:
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

def joinStimuliTracks(plate_names, opto_folder, behavior_names, plate_corners, plate_rows, plate_cols, join_stimuli=False):
    for plate in plate_names:
        if join_stimuli:
            print(plate, 'stimuli')
            outputStimuli(plate, opto_folder)
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


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)