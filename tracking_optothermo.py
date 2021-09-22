# python script for mosquito tracking in 14-well plates on Optothermocycler
# Trevor Sorrells 2020
# Python 3.5

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from skimage import io
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.util import crop
from scipy.misc import toimage
import os
import time
import subprocess as sp
import multiprocessing as mp
import argparse
from itertools import product



def main():


    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("action",choices=["background","process"],help="action to perform on videos, 'background' or 'process'",metavar="action")
    parser.add_argument("-p","--project",help="(required) path to project folder",required=True,metavar="PROJECT")
    parser.add_argument("-r","--raw",help="path to raw video folder, otherwise project folder",default=False,metavar="RAW")
    parser.add_argument("-y","--type",help="type of plate/experiment: 14well, DEET, optomembrane",default="14well",metavar="TYPE")
    parser.add_argument("-t","--trials",help="(required) full path to text file trial names, comma separated",required=True,metavar="TRIALS")
    parser.add_argument("-b","--bounds",help="stimulus box bounds, comma separated",metavar="BOUNDS")
    parser.add_argument("-c","--cutoff",help="stimulus cutoff",metavar="BOUNDS")
    parser.add_argument("-m","--maxcpus",help="maximum number of cpus for parallel processing",metavar="MAXCPUS")
    parser.add_argument("-f","--framesplit",help="number of frames to include in each split video during processing",metavar="FRAMESPLIT")
    parser.add_argument("-v","--novideo",help="don't output video file, just tracking output",action="store_true",default=False)
    parser.add_argument("-n","--notracking",help="don't output tracking file, just video output",action="store_true",default=False)

    args = parser.parse_args()

    os.chdir(args.project)
    with open(args.trials, 'r') as inFile:
        trial_list = inFile.read().strip().split(',')
    print(trial_list)

    if args.maxcpus:
        cpus = int(args.maxcpus)
    else:
        cpus = mp.cpu_count()-1
    pool = mp.Pool(cpus)
    
    if args.action == "background":
        if not args.novideo:
            pool.starmap(convert_avi_image_sequence, product(trial_list, [args.project]))
        pool.starmap(make_background, product(trial_list, [args.project]))
    
    if args.action == "process":
        grid, well_bounds, well_labels, frame_size = plate_setup(args.type)
        if not args.notracking:
            if not args.bounds or not args.cutoff:
                print("stimulus bounds and cutoff are required for processing video")
        if args.notracking:
            if not args.framesplit:
                pool.starmap(pipe_subtract_and_track, product(trial_list, [args.project], [args.raw], [grid], [well_bounds], [well_labels], [frame_size], [[0,0,0,0]], [0], [args.novideo],[args.notracking]))
            else:
                pool.starmap(pipe_subtract_and_track, product(trial_list, [args.project], [args.raw], [grid], [well_bounds], [well_labels], [frame_size], [[0,0,0,0]], [0], [args.novideo], [args.notracking], [int(args.framesplit)]))
        else:
            print([list(map(int,args.bounds.split(',')))])
            if not args.framesplit:
                pool.starmap(pipe_subtract_and_track, product(trial_list, [args.project], [args.raw], [grid], [well_bounds], [well_labels], [frame_size], [list(map(int,args.bounds.split(',')))], [int(args.cutoff)], [args.novideo],[args.notracking]))
            else:
                pool.starmap(pipe_subtract_and_track, product(trial_list, [args.project], [args.raw], [grid], [well_bounds], [well_labels], [frame_size], [list(map(int,args.bounds.split(',')))], [int(args.cutoff)], [args.novideo], [args.notracking], [int(args.framesplit)]))
    pool.close()


def make_background(trial_name, project_folder):
    #load background
    background_frames_list = []
    for root, dirs, files in os.walk(os.path.join(project_folder,trial_name,'background')):
        for j in range(len(files)):
            background_frames_list.append(io.imread(os.path.join(root, files[j]), as_gray=True))
    print(len(background_frames_list))
    if len(background_frames_list) == 0:
        print(trial_name)
    print(background_frames_list[0])
    background_frames = np.zeros((len(background_frames_list), background_frames_list[0].shape[0], background_frames_list[0].shape[1]))
    for i in range(len(background_frames_list)):
        background_frames[i] = background_frames_list[i]
    # print(background_frames[0])
    background_min = np.amin(background_frames, 0).astype(np.uint8)
    # background_min = (background_min * 255).astype(np.uint16)
    # print(background_min)
    # print(background_max)
    io.imsave(os.path.join(project_folder,trial_name,'MIN_background.tif'), background_min)

def convert_avi_image_sequence(trial_name, project_folder):
    trial_path = os.path.join(project_folder,trial_name)
    print(trial_path)
    if not os.path.exists(os.path.join(project_folder, trial_name, 'background')):
        os.system('mkdir ' + os.path.join(project_folder, trial_name, 'background'))
    # print(trial_path)
    t1 = time.time()
    for root, dirs, files in os.walk(trial_path):
        # print(root)
        # print(dirs)
        # print(files)
        avi_files = [f for f in files if f[-3:] == 'avi']
        print(avi_files)
        for i in range(len(avi_files)):
            #os.system('ffmpeg -i ' + os.path.join(project_folder, trial_name, avi_files[i]) + ' -r .01 -pix_fmt gray ' + os.path.join(project_folder,trial_name,'background',str(i) + '%05d.tif'))
            os.system('ffmpeg -i ' + os.path.join(project_folder, trial_name, avi_files[i]) + ' -vf "select=eq(n\,0)" -pix_fmt gray ' + os.path.join(project_folder,trial_name,'background',str(i) + '%05d.tif'))
    print('took',time.time()-t1,'seconds')

def pipe_subtract_and_track(trial_name, project_folder, raw_folder, grid, well_bounds, well_labels, frame_size, stim_synch_bounds, stim_cutoff, no_video, notracking, frame_split=0):
    print(trial_name, project_folder, raw_folder, grid, well_bounds, well_labels, frame_size, stim_synch_bounds, stim_cutoff, no_video, notracking, frame_split)
    #load background
    if not raw_folder:
        raw_folder = project_folder
    background = io.imread(os.path.join(raw_folder,trial_name,'MIN_background.tif'), as_gray=True)
    background = background.astype(np.int16)
    background = np.invert(background)
    all_movement = np.zeros((background.shape[0], background.shape[1]))
    centroids = []
    [centroids.append([]) for x in well_bounds]
    stim = []
    if not os.path.exists(os.path.join(project_folder, 'processed')):
        os.system('mkdir ' + os.path.join(project_folder, 'processed'))
    frame_number=0
    video_number=1
    if frame_split == 0:
        output_name = trial_name
    else:
        output_name = trial_name + '_' + str(video_number)
    
    if not no_video:
        outcommand = generateOutCommand(output_name, frame_size)
        print(' '.join(outcommand))
        background_name = trial_name + '_bkg'
        bkgcommand = generateOutCommand(background_name, frame_size)
    
        pipeout = sp.Popen(outcommand, stdin=sp.PIPE)#, stderr=sp.PIPE)
        bkgout = sp.Popen(bkgcommand, stdin=sp.PIPE)
    for root, dirs, files in os.walk(os.path.join(raw_folder,trial_name)):
        # print(root)
        # print(dirs)
        # print(files)
        avi_files = [f for f in files if f[-3:] == 'avi' and f[0] != '.']
        avi_files = sorted(avi_files)
        print(avi_files)
        for i in range(len(avi_files)):
            # if i > 0:
            #     break
            t1 = time.time()
            incommand = [ 'ffmpeg',
                    '-i', os.path.join(raw_folder, trial_name, avi_files[i]),
                    '-f', 'image2pipe',
                    '-pix_fmt', 'gray',
                    '-vcodec', 'rawvideo', '-'] 
            print('incommand')
            pipein = sp.Popen(incommand, stdout = sp.PIPE, bufsize=10**8)
            f=0
            while True: #f < 1:
                if frame_split != 0 and frame_number > 0 and frame_number%frame_split == 0 and not no_video:
                    out, err = pipeout.communicate()
                    if err != None:
                        with open(os.path.join(raw_folder,trial_name,'ffmpeg_errors.txt'),'w') as outFile:
                            outFile.write(err)
                    video_number += 1
                    output_name = trial_name + '_' + str(video_number)
                    outcommand = generateOutCommand(output_name, frame_size)
                    print(' '.join(outcommand))
                    pipeout = sp.Popen(outcommand, stdin=sp.PIPE)#, stderr=sp.PIPE)
                raw_image = pipein.stdout.read(frame_size[0]*frame_size[1])
                image =  np.frombuffer(raw_image, dtype='uint8')
                if np.size(image) < frame_size[0]*frame_size[1]:
                    print(np.size(image))
                    pipein.stdout.flush()
                    break
                f += 1
                frame = image.reshape(frame_size)
                stim_img = frame[stim_synch_bounds[2]:stim_synch_bounds[3],stim_synch_bounds[0]:stim_synch_bounds[1]]
                if np.mean(stim_img) > stim_cutoff:
                    stim.append(1)
                else:
                    stim.append(0)
                # print(np.shape(background))
                # print(np.shape(frame))
                processed = np.add(background, frame)
                processed = processed.clip(0,255).astype(np.uint8)
                # plt.figure()
                # plt.imshow(processed)
                # plt.show()
                all_movement = np.amax(np.asarray([all_movement, processed]), 0)
                # print(frame)
                # print(background)
                # print(processed)
                
                if not no_video:
                    pipeout.stdin.write(frame.tostring())
                    if frame_number % 100 == 0:
                        bkgout.stdin.write(frame.tostring())
                frame_number += 1
                if not notracking:
                    processed = processed > 30
                    for i, well in enumerate(well_bounds):
                        well_img = processed[well[2]:well[3],well[0]:well[1]]
                        label_image = label(well_img)
                        well_regions = regionprops(label_image)
                        if len(well_regions) > 1:
                            well_regions.sort(key=lambda x: x.filled_area,reverse = True)
                        elif len(well_regions) == 0:
                            centroids[i].append(['nan','nan'])
                            continue
                        centroids[i].append([well_regions[0].centroid[1], well_regions[0].centroid[0]])
                pipein.stdout.flush()
                #pipeout.stdin.flush()
            pipein.stdout.close()
            print(f, ' frames processed in ',time.time()-t1,' seconds')
    if not no_video:
        out, err = pipeout.communicate()
        if err != None:
            with open(os.path.join(raw_folder,trial_name,'ffmpeg_errors.txt'),'w') as outFile:
                outFile.write(err)
        out, err = bkgout.communicate()
        if err != None:
            with open(os.path.join(raw_folder,trial_name,'ffmpeg_bkg_errors.txt'),'w') as outFile:
                outFile.write(err)
    if not notracking:
        if not os.path.exists(os.path.join(project_folder, 'tracking_output')):
            os.system('mkdir ' + os.path.join(project_folder, 'tracking_output'))
        with open(os.path.join(project_folder,'tracking_output',trial_name + '.txt'),'w') as outFile:
            outFile.write('stim\t')
            for i in range(len(centroids)):
                outFile.write(str(i) + '_x\t' + str(i) + '_y\t')
            outFile.write('\n')
            for f in range(len(centroids[0])):
                if f >= len(stim):
                    print(f, len(stim))
                outFile.write(str(stim[f]) + '\t')
                for w in range(len(centroids)):
                    outFile.write('\t'.join(map(str, centroids[w][f])) + '\t')
                outFile.write('\n')
    
    if not os.path.exists(os.path.join(raw_folder, trial_name, 'graphs')):
        os.system('mkdir ' + os.path.join(raw_folder, trial_name, 'graphs'))
    plt.figure()
    plt.imshow(all_movement)
    plt.plot(grid[:,:,0], grid[:,:,1])
    plt.plot(np.transpose(grid[:,:,0]), np.transpose(grid[:,:,1]))
    # for region in regionprops(label(processed)):
    #     plt.plot(region.centroid[1],region.centroid[0], 'ro')
    
    plt.savefig(os.path.join(raw_folder, trial_name, 'graphs','all_movement.pdf'), format='pdf')
    plt.close()
    plt.figure()
    plt.imshow(processed)
    plt.savefig(os.path.join(raw_folder, trial_name, 'graphs','processed.pdf'), format='pdf')

def generateOutCommand(file_name, frame_size):
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
     
def plate_setup(experiment_type):
    if experiment_type=='14well':
        # 1 15-well custom plate for current opto-thermocycler setup
        # plates and corners currently must start at top left go clockwise
        plate_corners = [[14,37],[943,35],[944,629],[16,632]]
        plate_rows, plate_cols = 3,5
        well_labels = ['A1','B1','C1','A2','B2','C2','A3','B3','C3','A4','B4','C4'] #top to bottom, left to right
        frame_size = (700,1180)
    elif experiment_type =='DEET':
        #this experiment has 6 petri dishes with a single mosquito each 
        #run on the light table
        plate_corners = [[0,0],[1425,0],[1425,938],[0,938]]
        plate_rows, plate_cols = 2,3
        well_labels = ['A1','B1','A2','B2','A3','B3'] #top to bottom, left to right
        frame_size = (940,1440)
        
    #calculate grid for plates
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
    # starting with plate in top left and going clockwise, total of 48 mosquitoes per plate
    # the bounds for each well are min-x, max-x, min-y, max-y
    well_bounds = []
    for col in range(plate_cols):
        for row in range(plate_rows):
            well_bounds.append([int((grid[row,col,0]+grid[row+1,col,0])/2), int((grid[row,col+1,0]+grid[row+1,col+1,0])/2),
                                int((grid[row,col,1]+grid[row,col+1,1])/2), int((grid[row+1,col,1]+grid[row+1,col+1,1])/2,)])
    
    return grid, well_bounds, well_labels, frame_size
    
if __name__=="__main__":
    print("main")
    main()
