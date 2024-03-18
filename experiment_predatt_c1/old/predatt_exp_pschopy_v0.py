'''
Created on Tue Oct 24 2023

@author: Max Van Migem
'''

import numpy as np
import os
import copy
import pandas
import time
import random
# import pylink #the last is to communicate with the eyetracker
from psychopy import parallel, visual, gui, data, event, core, monitors
from psychopy.visual import ShapeStim



####################SELECT THE RIGHT LAB & Mode####################
lab = 'none'   #'actichamp'/'biosemi'/'none'

mode = 'default'   #'default'/'DemoMode' #affects nr of trials per block (50%)

eye_tracking = False #True/False
###################################################################

# # #dlg box#
info =  {'Participant ID (***)': '', 'Gender': ['Male','Female', 'X'], 
         'Age': '', 'Dominant hand':['left','right','ambi']}

already_exists = True
while already_exists:   #keep asking for a new name when the data file already exists
    dlg = gui.DlgFromDict(dictionary=info, title='Predatt Experiment')  #display the gui
    file_name = os.getcwd() + '/data/' + 'pilot_predatt_participant_' + info['Participant ID (***)']   #determine the file name (os.getcwd() is where your script is saved)

    if not dlg.OK:
        core.quit()

    if not os.path.isfile((file_name + '.csv')):  #only escape the while loop if ParticipantNr is unique
        already_exists = False
    else:
        dlg2 = gui.Dlg(title = 'Warning')  #if the suggested_participant_nr is not unique, present a warning msg
        suggested_participant_nr = 0
        suggested_file_name = file_name
        while os.path.isfile(suggested_file_name + '.csv'): #provide a suggestion for another ParticipantNr
            suggested_participant_nr += 1
            suggested_file_name = os.getcwd() + '/data/' + 'feedback_theta_participant' + str(suggested_participant_nr)

        dlg2.addText('This Participant Nr is in use already, please select another.\n\nParticipant Nr ' 
                     +  str(suggested_participant_nr) + ' is still available.')
        dlg2.show()

# Define a monitor
# system_monitor = monitors.Monitor('dell_system_monitor')    #This the configuration of the monitor of my laptop

# # This is to set a new configuration for your monitor
# system_monitor.setSizePix((1920, 1200))
# system_monitor.setWidth(31.4)
# system_monitor.setDistance(65)
# system_monitor.save()

# #initialize window#
win = visual.Window(fullscr=True, color = np.repeat(-0.3961, 3), colorSpace = 'rgb', units = 'deg', monitor= 'dell_system_monitor')
win.mouseVisible = False

####################################################
#Initialize trial variables
####################################################

# in progress
stim_positions = [[-200,200],[200,200],[200,-200],[-200,-200]] # temporary stim position variable

# random generator
rng = np.random.default_rng(seed=None)
# array with random jitter for horizontal stim lines (this might have to be a standard array as in the same for everyone)
linejitter_arr =(np.ones((12,12,2))- rng.random((12,12,2))*2)/80 # the array is three dimensional so that is has a value for x and y for every line

####################################################
#Create Psychopy simple visual objects
####################################################

test_instruction = ('Test')

message = visual.TextStim(win, text='') 

# manual input of fixation cross vertices, very hard coded but it's fast and customizable
cross_vert = [(-.05, .05), (-.05, .35), (.05, .35,), (.05, .05),
              (.35, .05), (.35,-.05), (.05, -.05),(.05,-.35),
              (-.05,-.35), (-.05, -.05), (-.35,-.05), (-.35, .05)]

# defining fixcross using vertices above, you can change the size, colour, etc.. of the fixation cross here   
cross = ShapeStim(win, vertices=cross_vert, size=1, fillColor='white', lineWidth=0, pos=(0, 0), ori=0)

teststim = ShapeStim(win, vertices=[(0.5,1.0), (1.0,1.0), (0.5,1.0), (1.0,1.0)],
                     size=50, fillColor='white', lineWidth=1, pos=(0, 0), ori=0, closeShape=False)


####################################################
#Functions for generating the stimulus
####################################################

def generateStimCoordinates(gridcol, gridrow, jitter):
    """
    Generates coordinates used to draw the stimulus lines
    """
    # Calculate spacing of grid
    x_spacing = 1.0 / (gridcol)
    y_spacing = 1.0 / (gridrow)

    # Create an array to store coordinates: every point (x,y) of the grid for every quadrant (4) 
    coord_array = np.empty(shape = (gridcol,gridrow,2,4),dtype= 'object')
    quadrant_set = [[-1,1],[1,1],[1,-1],[-1,-1]]

    # Generate grid coord per quadrant
    for i,quad in enumerate(quadrant_set):
        # Per row
        for row in range(gridrow):
            # Per column
            for col in range (gridcol):

                grid_x = col * x_spacing  # grid points
                grid_y = row * y_spacing  

                grid_x = grid_x * quad[0]   # quadrant position
                grid_y = grid_y * quad[1]   # quadrant position

                coord_array[col,row,0,i] = grid_x  # Store them in big ass array
                coord_array[col,row,1,i] = grid_y

    # flip the matrices so that they have the 'orientation' corresponding to the quadrant 
    coord_array[:,:,0,0] = np.flip(coord_array[:,:,0,0], axis=0) 
    coord_array[:,:,1,0] = np.flip(coord_array[:,:,1,0], axis=0) 

    coord_array[:,:,0,2] = np.flip(coord_array[:,:,0,2], axis=1) 
    coord_array[:,:,1,2] = np.flip(coord_array[:,:,1,2], axis=1) 

    coord_array[:,:,0,3] = np.flip(coord_array[:,:,0,3], axis=(0,1)) 
    coord_array[:,:,1,3] = np.flip(coord_array[:,:,1,3], axis=(0,1)) 

    #  Add jitter
    for i in range(4):
        coord_array[:,:,:,i] =  coord_array[:,:,:,i] + jitter[:gridcol,:gridrow,:]

        
    return coord_array


def generateStim(linelength, coord_array, colour, size, fixdistance):
    """ 
    Draws lines for every point in the grid 
    """
    # This is to calculate the distance to the fixation cross
    dist = np.sqrt(np.square(fixdistance)/2) #pythagoras
    quads = [[-1,1],[1,1],[1,-1],[-1,-1]]

    # Get size of grid
    gridcol = np.shape(coord_array)[0]
    gridrow = np.shape(coord_array)[1]

    half_line = linelength/2

    # Init array to store line objects per quadrant
    line_stimuli = [[],[],[],[]] # I'm using lists here because i want to use the in built append function
    
    for quad in range(4): 
        for row in range(gridrow):
            for col in range(gridcol):

                # Define line starting point
                start_x = (coord_array[col,row,0,quad] * size) - half_line + (quads[quad][0]*dist)  # size, linestart and distance from fixation are all in here 
                start_y = (coord_array[col,row,1,quad] * size) + (quads[quad][1]*dist)              
                                                                              
                end_x = start_x + linelength
                end_y = start_y

                # Create line object
                line_stim = visual.Line(win, start=(start_x, start_y), end=(end_x, end_y), lineColor=colour)
                line_stimuli[quad].append(line_stim)
    
    return line_stimuli


def drawStim(line_stimuli, quad):
    """ 
    Draws the stimuli in the correct quadrant
    """
    # select quadrant
    stim_set = line_stimuli[quad]

    # draw all the lines
    for line_stim in stim_set:
        line_stim.draw()


####################################################
#Trial display functions
####################################################

def stimPresentation(fix_cr, stimulus, stim_dur, isi_dur, start_quad, last, lab = lab): 
    """
    Stimulus presentation for regular predictable trial
    """
    # Check timing
    stim_clock = core.Clock()
    stim_times = []

    stimpos_ls = np.roll(np.arange(4), -start_quad, axis=0)   # change the starting point of the stimulus

    # This is to change where the last stim appears
    if last != 3:  
        stimpos_ls[-1] = stimpos_ls[last]

    # Send trial start trigger
    eegTriggerSend(99,lab)
    # trial procedure
    for i,stimpos in enumerate(stimpos_ls):

        stim_clock.reset() # keep track of stim timings
        
        trigger = selectEEGStimulusTrigger(stimpos,pos=i) # EEG trigger generation
        drawStim(stimulus,stimpos)
        fix_cr.draw()
        
        eegTriggerSend(trigger,lab) # Send trigger
        win.flip()
        core.wait(stim_dur)


        fix_cr.draw()
        win.flip()
        print((stim_clock.getTime() * 1000))
        core.wait(isi_dur)

        stim_times.append((stim_clock.getTime() * 1000))
    
    return stim_times

####################################################
#Trial sequence functions
####################################################


def generateTrialList(tr_block, npredictable):
    """ 
    Generate pseudo randomized properties of every trial (start quadrant and expected or unexpected trials)
    """
    # Generate an array consitting of values 0 to 3 with the length of the amount of trials
    startquad = np.tile(np.arange(4),reps= int(tr_block/4)+1)
    random.shuffle(startquad)  # randomize
    startquad = startquad[:tr_block]  # trim to correct length but preferably we would like it to be divisible by 4 so this doesnt have to happen 
    
    # Make a matching array with normal (predicted) end positions
    prediction_arr = startquad.copy()   # The 'last' argument for the end position in stimPresentation() is always relative to the first
    prediction_arr[:] = 3

    # Then make a part of the trials 'unpredictable' meaning that the last position is either the same as the first or the second
    # Not the third for now. This means that the stimulus doesnt apear in the same place twice in a row
    nunpred = tr_block - npredictable   
    half_unpred = int(nunpred/2)   # again this would preferably be divisble by 2 for equal amount in the first and second postions

    prediction_arr[:half_unpred] = 1 
    prediction_arr[half_unpred:nunpred:1] = 0   # i have now made it so that if its not equal we have an extra predictable trial instead

    # Shuffle
    random.shuffle(prediction_arr)

    # Return both
    return startquad, prediction_arr


def generateCatchTrials(tr_block, ncatch):
    """ 
    Some trials are catch trials, this function generates a list to designate which trials are
    """
    # Make a list with normal trials (0) and a list with catch trials (1)
    full_list = np.zeros(tr_block-ncatch)
    c_list = np.ones(ncatch)

    # Make one list
    full_list = np.append(full_list,c_list)
    
    # Shuffle
    random.shuffle(full_list)

    return full_list


####################################################
#External measurement instruments
####################################################

#EEGTriggerSend#
def eegTriggerSend(eeg_trigger, lab): #need to elaborate
    """
    Sends trigger to EEG recording
    """
    if not lab == 'none':
        parallel.setData(eeg_trigger)
        core.wait(0.01)
        parallel.setData(0)
    else:
        print(eeg_trigger)

#SelectEEGStimulusTrigger#
def selectEEGStimulusTrigger(start,pos):
    """
    distinguish EEG trigger for stimulus
    """
    eeg_stim_trigger = (start+1)*10 + (pos+1)

    return eeg_stim_trigger


#just setting up the EEG
if lab == 'actichamp':
    parallel.setPortAddress('0xCFB8')
elif lab == 'biosemi':
    parallel.setPortAddress('0xCFE8')


####################################################
#Experiment value initialization
####################################################

# ExperimentHandler and TrialHandler

this_exp = data.ExperimentHandler(dataFileName = file_name)

# For every pilot set we initialize a trialhandler
trial_list_pos = []
pos_start, pos_last = generateTrialList(6,4)
trial_list_pos.append({'start_pos': pos_start})
pos_trials = data.TrialHandler(trialList = trial_list_pos, nReps=1, method = 'sequential')
this_exp.addLoop(pos_trials)

trial_list_freq = []
freq_start, freq_last = generateTrialList(6,4)
trial_list_freq.append({'start_pos': freq_start})
freq_trials = data.TrialHandler(trialList = trial_list_freq, nReps=1, method = 'sequential')
this_exp.addLoop(freq_trials)

trial_list_size = []
size_start, size_last = generateTrialList(6,4)
trial_list_size.append({'start_pos': size_start})
size_trials = data.TrialHandler(trialList = trial_list_size, nReps=1, method = 'sequential')
this_exp.addLoop(size_trials)

# initilize counters
my_clock = core.Clock() #define my_clock
n_correct = 0
point_counter = 0

####################################################
#Experiment loop
####################################################

# sets of different stim configurations for the pilot 
pilot_set = [6,8,10]


message.text = test_instruction
message.draw()
win.flip()
event.waitKeys(keyList = ['space']) #no event.clearEvents() necessary

# I roughly divided the block triplets for the pilot

# the first three blocks are different positions
for i in range(3):
    grid = generateStimCoordinates(gridcol=12, gridrow=8, jitter=linejitter_arr)
    stimset = generateStim(linelength=0.5, coord_array=grid, colour='white',size=8, fixdistance=pilot_set[i] )

    message.text = f'Block {i+1}'
    message.draw()
    win.flip()
    event.waitKeys(keyList = ['space']) 
    for ind,trial in enumerate(pos_start):
        my_clock.reset() #start rt timing
        stimulus_times = stimPresentation(fix_cr=cross, stimulus=stimset, stim_dur=0.05 , isi_dur=0.5,
                                          start_quad=pos_start[ind], last=pos_last[ind])
        
        core.wait(0.5)
        keys = event.getKeys(timeStamped=my_clock)
        # Store data
        
        #store data
        pos_trials.addData('LocalTime_DDMMYY_HMS', 
                       str(time.localtime()[2]) + '/' + str(time.localtime()[1]) + '/' + str(time.localtime()[0]) 
                       + '_' + str(time.localtime()[3]) + ':' + str(time.localtime()[4]) + ':' + str(time.localtime()[5])) #HMS = hour min sec
        pos_trials.addData('lab', lab)
        pos_trials.addData('mode', mode)
        pos_trials.addData('participant', info['Participant ID (***)'])
        pos_trials.addData('gender', info['Gender'])
        pos_trials.addData('age', info['Age'])
        pos_trials.addData('trial', (ind +1)) #Python starts indexing at 
        pos_trials.addData('start_position',pos_start[ind])
        pos_trials.addData('last_position',pos_last[ind])
        pos_trials.addData('t_stim_1',stimulus_times[0])
        pos_trials.addData('t_stim_2',stimulus_times[1])
        pos_trials.addData('t_stim_3',stimulus_times[2])
        pos_trials.addData('t_stim_4',stimulus_times[3])
        if keys:
            pos_trials.addData('key_pressed',keys[-1][0])
            pos_trials.addData('press_time',keys[-1][1])
        else:
            pos_trials.addData('key_pressed',None)
            pos_trials.addData('press_time',None)

        this_exp.nextEntry()

        esc_key = False
        if 'escape' in keys[:][0]:
            esc_key = True
        if esc_key:
            win.close()
            core.quit()
           
# # Next we vary the spatial frequency aka the grid density         
# for i in range(3):
#     grid = generateStimCoordinates(gridcol=12, gridrow=pilot_set[i], jitter=linejitter_arr)
#     stimset = generateStim(linelength=0.5, coord_array=grid, colour='white',size=10, fixdistance=8)

#     message.text = f'Block {i+4}'
#     message.draw()
#     win.flip()
#     event.waitKeys(keyList = ['space'])

#     for ind, trial in enumerate(freq_start):
#         my_clock.reset() #start rt timing
#         stimulus_time = stimPresentation(fix_cr=cross, stimulus=stimset, stim_dur=0.05 , isi_dur=0.5, 
#                                          start_quad=freq_start[ind], last=freq_last[ind])
#         core.wait(0.5)
        
#         # Store data
        
#         #store data
#         freq_trials.addData('LocalTime_DDMMYY_HMS', 
#                        str(time.localtime()[2]) + '/' + str(time.localtime()[1]) + '/' + str(time.localtime()[0]) 
#                        + '_' + str(time.localtime()[3]) + ':' + str(time.localtime()[4]) + ':' + str(time.localtime()[5])) #HMS = hour min sec
#         freq_trials.addData('lab', lab)
#         freq_trials.addData('mode', mode)
#         freq_trials.addData('participant', info['Participant ID (***)'])
#         freq_trials.addData('gender', info['Gender'])
#         freq_trials.addData('age', info['Age'])
#         freq_trials.addData('trial', (ind +1)) #Python starts indexing at 
#         freq_trials.addData('start_position',freq_start[ind])
#         freq_trials.addData('last_position',freq_last[ind])

#         this_exp.nextEntry()
#         if 'escape' in  event.getKeys(keyList = ['k','d','escape']):
#             win.close()
#             core.quit()

# # Last is the stimulus size
# for i in range(3):
#     grid = generateStimCoordinates(gridcol=12, gridrow=8, jitter=linejitter_arr)
#     stimset = generateStim(linelength=0.4 + i/10, coord_array=grid, colour='white',size=pilot_set[i], fixdistance=8 )

#     message.text = f'Block {i+7}'
#     message.draw()
#     win.flip()
#     event.waitKeys(keyList = ['space']) 
    
#     for ind,trial in enumerate(size_start):
#         my_clock.reset() #start rt timing
#         stimPresentation(fix_cr=cross, stimulus=stimset, stim_dur=0.05 , isi_dur=0.5, 
#                          start_quad=size_start[ind], last=size_last[ind])
#         core.wait(0.5)
#         #store data
#         size_trials.addData('LocalTime_DDMMYY_HMS', 
#                        str(time.localtime()[2]) + '/' + str(time.localtime()[1]) + '/' + str(time.localtime()[0]) 
#                        + '_' + str(time.localtime()[3]) + ':' + str(time.localtime()[4]) + ':' + str(time.localtime()[5])) #HMS = hour min sec
#         size_trials.addData('lab', lab)
#         size_trials.addData('mode', mode)
#         size_trials.addData('participant', info['Participant ID (***)'])
#         size_trials.addData('gender', info['Gender'])
#         size_trials.addData('age', info['Age'])
#         size_trials.addData('trial', (ind +1)) #Python starts indexing at 
#         size_trials.addData('start_position',size_start[ind])
#         size_trials.addData('last_position',size_last[ind])

#         this_exp.nextEntry()

#         if 'escape' in  event.getKeys(keyList = ['k','d','escape']):
#             win.close()
#             core.quit()


    
    
#no event.clearEvents() necessary

# Close the window
win.close()
core.quit()


