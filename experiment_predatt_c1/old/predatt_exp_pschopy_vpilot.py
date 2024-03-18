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
import pylink
# import pylink #the last is to communicate with the eyetracker
from psychopy import parallel, visual, gui, data, event, core, monitors
from psychopy.visual import ShapeStim
#from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy #this are functions used to run the eyetracker calibration and validation



####################SELECT THE RIGHT LAB & Mode####################
lab = 'biosemi'   #'actichamp'/'biosemi'/'none'

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

# # Define a monitor
# system_monitor = monitors.Monitor('cap_lab_monitor')    #This the configuration of the monitor of my laptop

# # This is to set a new configuration for your monitor
# system_monitor.setSizePix((1024, 768))
# system_monitor.setWidth(39.2)
# system_monitor.setDistance(65)
# system_monitor.save()

# #initialize window#
win = visual.Window(fullscr=True, color = 'black', colorSpace = 'rgb', units = 'deg', monitor= 'cap_lab_monitor',useFBO=False)
win.mouseVisible = False

####################################################
#Initialize trial variables
####################################################

#options
n_trials = 50 # per block
n_predictible = 35
n_catch = 2

n_blocks = 9        


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
cross = ShapeStim(win, vertices=cross_vert, size=1, fillColor='white', lineWidth=0, pos=(0, 0), ori=0, autoDraw=False)

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


def generateStim(linelength, coord_array, colour, catch_colour, size, fixdistance):
    """ 
    Draws lines for every point in the grid and create an array of lines for each quadrant
    Then four more arrays are created each representing a catch trial in a separate quadrant
    Returns a 5x4 array of lists with each list containing the line stimuli for a quadrant
    """
    # This is to calculate the distance to the fixation cross
    dist = np.sqrt(np.square(fixdistance)/2) #pythagoras
    quads = [[-1,1],[1,1],[1,-1],[-1,-1]]

    # Get size of grid
    gridcol = np.shape(coord_array)[0]
    gridrow = np.shape(coord_array)[1]

    half_line = linelength/2
    # Init arrays to store line objects per quadrant and also for every possible catch trial 
    line_stimuli = np.empty((2,4), dtype=object)   # 2 types (0 is normal white lines, 1 is catch stim ) and 4 quadrants
    # Fill this with empty lists so we can append
    for i in range(2):
        for j in range(4):
            line_stimuli[i,j] = []
 
    for quad in range(4): 
        for row in range(gridrow):
            for col in range(gridcol):

                # Define line starting point
                start_x = (coord_array[col,row,0,quad] * size) - half_line + (quads[quad][0]*dist)  # size, linestart and distance from fixation are all in here 
                start_y = (coord_array[col,row,1,quad] * size) + (quads[quad][1]*dist)              
                                                                              
                end_x = start_x + linelength
                end_y = start_y

                # Create line objects with normal colour
                line_stim = visual.Line(win, start=(start_x, start_y), end=(end_x, end_y), lineColor=colour)
                line_stimuli[0,quad].append(line_stim)
                # Also one with catch colour
                catch_line_stim = visual.Line(win, start=(start_x, start_y), end=(end_x, end_y), lineColor=catch_colour)
                line_stimuli[1,quad].append(catch_line_stim)

    return line_stimuli


def drawStim(line_stimuli, quad, catch):
    """ 
    Draws the stimuli in the correct quadrant
    Catch = 0 means normal color catch = 1 means catch
    """
    # select quadrant and whether this is a catch stim
    stim_set = line_stimuli[catch,quad] 

    # draw all the lines
    for line_stim in stim_set:
        line_stim.draw()


####################################################
#Trial display functions
####################################################

def stimPresentation(stimulus, stim_dur, isi_dur,iti_dur, start_quad, last, q_catch, lab = lab): 
    """
    Stimulus presentation for regular predictable trial
    """
    # Check timing
    stim_clock = core.Clock()
    trial_clock = core.Clock()
    stim_times = []

    stimpos_ls = np.roll(np.arange(4), -start_quad, axis=0)   # change the starting point of the stimulus

    # This is to change where the last stim appears
    if last != 3:  
        stimpos_ls[-1] = stimpos_ls[last]
    
    # This is for the catch trials
    catch_ls = np.zeros(4,dtype=int) 
    if not q_catch == 0:
        catch_ls[q_catch-1] = 1   # We do q_catch-1 because the catch trial list gives a value of 0 to 5 with 0 being no catch 


    # Send trial start trigger
    eegTriggerSend(99,lab)
    # trial procedure
    for i,stimpos in enumerate(stimpos_ls):

        stim_clock.reset() # keep track of stim timings
        
        trigger = int(selectEEGStimulusTrigger(stimpos,pos=i)) # EEG trigger generation
        print(trigger)
        drawStim(stimulus,stimpos,catch_ls[i])
        wait = isi_dur/2-stim_clock.getTime()      # Drawing the stimulus takes time, this compensates that
        core.wait(wait)
        eegTriggerSend(trigger,lab)  # Send trigger
        win.flip()
        core.wait(stim_dur)
        win.flip()
        core.wait(isi_dur/2)
        stim_times.append((stim_clock.getTime() * 1000))

    trial_time = trial_clock.getTime()*1000
    core.wait(iti_dur)
    return stim_times, trial_time


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
    full_list = np.zeros(tr_block-ncatch,dtype=int)
    c_list = np.ones(ncatch,dtype=int)
    for i,one in enumerate(c_list):
        c_list[i] = int(one + i%4)   # Just adding 0,1,2 or 3 to every element (again: if ncatch is divisible by 4 then it's balanced)

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
        gsr_port.setData(eeg_trigger)
        core.wait(0.01)
        gsr_port.setData(0)
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
    gsr_port = parallel.ParallelPort(address=0xCFB8)
elif lab == 'biosemi':
    gsr_port = parallel.ParallelPort(address=0x3FB8)

# Eye-tracking set up
#No def, just some basic set up for the eyetracker#

if eye_tracking:
    eyeTracker = pylink.EyeLink('100.1.1.1') #connect to the tracker using the pylink library

    FileNameET = 'pp_' + info['Participant ID (***)'] + '.EDF' #open datafile (max 8 characters)
    
    eyeTracker.openDataFile(FileNameET)
    eyeTracker.sendCommand("add_file_preamble_text 'Predatt Experiment'") #add personalized data file header (preamble text)
    
    genv = EyeLinkCoreGraphicsPsychoPy(eyeTracker, win) #set up a custom graphics environment (EyeLinkCoreGraphicsPsychopy) for calibration
    pylink.openGraphicsEx(genv)
    
    eyeTracker.setOfflineMode() #put the tracker in idle mode before we change some parameters
    pylink.pumpDelay(100)
    eyeTracker.sendCommand("screen_pixel_coords = 0 0 %d %d" % (1024-1, 768-1)) #send screen resolution to the tracker, format: (scn_w - 1, scn_h - 1) #here: 1920 x 1080!
    eyeTracker.sendMessage("DISPLAY_COORDS = 0 0 %d %d" % (1024-1, 768-1)) #relevant only for data viewer #here: 1920 x 1080!
    eyeTracker.sendCommand("sample_rate 1000") #250, 500, 1000, or 2000 (only for EyeLink 1000 plus)
    eyeTracker.sendCommand("recording_parse_type = GAZE")
    eyeTracker.sendCommand("select_parser_configuration 0") #saccade detection thresholds: 0-> standard/coginitve, 1-> sensitive/psychophysiological
    eyeTracker.sendCommand("calibration_type = HV13") #13 point calibration (recommended for head free remote mode)

    message.text = 'Press ENTER to set up the tracker\n' #show calibration message
    message.draw()
    win.flip()

    eyeTracker.doTrackerSetup() #calibrate the tracker #once you are happy with the calibration and validation (!), you are ready to run the experiment. 

    #pylink.closeGraphics()
    #pylink.closeGraphics(genv)


####################################################
#Experiment value initialization
####################################################

# ExperimentHandler and TrialHandler

this_exp = data.ExperimentHandler(dataFileName = file_name)

# Intialize trial handelr
trial_list = []
start_pos = np.empty(n_blocks,dtype=object)
last_pos = np.empty(n_blocks,dtype=object)
catch_trials = np.empty(n_blocks,dtype=object)
for block in range(n_blocks):
    start_pos[block], last_pos[block] = generateTrialList(tr_block=n_trials,npredictable=n_predictible)
    catch_trials[block] = generateCatchTrials(tr_block=n_trials,ncatch=n_catch)
    trial_list.append({'start_pos': start_pos[block]})
trials = data.TrialHandler(trialList = trial_list, nReps=1, method = 'sequential')
this_exp.addLoop(trials)


# initilize counters
my_clock = core.Clock() #define my_clock
n_correct = 0
point_counter = 0
trial_count = 1

####################################################
#Experiment loop
####################################################

# sets of different stim configurations for the pilot 
pilot_pos = [[4,6,8],[6,6,6],[6,6,6]]
pilot_freq = [[6,6,6],[6,8,10],[6,6,6]]
pilot_size = [[6,6,6],[6,6,6],[4,6,8]]


message.text = test_instruction
message.draw()
win.flip()
event.waitKeys(keyList = ['space']) #no event.clearEvents() necessary

if eye_tracking:
    eyeTracker.setOfflineMode() #this is called before start_recording() to make sure the eye tracker has enough time to switch modes (to start recording)
    pylink.pumpDelay(100)
    eyeTracker.startRecording(1,1,1,1) #starts the EyeLink tracker recording, sets up link for data reception if enabled. The 1,1,1,1 just has to do with whether samples and events etcetera needs to be written to EDF file. Recording needs to be started for each block
    pylink.pumpDelay(100) #wait for 100 ms to cache some samples

# I roughly divided the block triplets for the pilot
for j in range(1):
    # the first three blocks are different positions
    for i in range(3):
        grid = generateStimCoordinates(gridcol=12, gridrow=pilot_freq[2][i], jitter=linejitter_arr)
        stimset = generateStim(linelength=0.35, coord_array=grid, colour='white',catch_colour='red',size=pilot_size[2][i], fixdistance=pilot_pos[2][i] )

        message.text = f'Block {i+j+1}'
        message.draw()
        win.flip()
        start_key = event.waitKeys(keyList = ['space','escape'])
        if start_key == 'escape':
            core.quit()

        cross.autoDraw = True

        for ind in range(n_trials):
            
            if eye_tracking:
                eyeTracker.sendMessage('TRIALID %d' % (trial_count)) #send a message ("TRIALID") to mark the start of a trial
                eyeTracker.sendCommand("record_status_message 'stim %s trial %s'" % (start_pos[j+i][ind], (trial_count))) #to show the current task, block nr and trial nr #+1 because Python starts at 0
            

            event.clearEvents(eventType = 'keyboard')
            my_clock.reset() #start rt timing
            stimulus_times,trial_stamp = stimPresentation(stimulus=stimset, stim_dur=0.054 , isi_dur=0.5, iti_dur=0.501,
                                                        start_quad=start_pos[2+i][ind], 
                                                        last=last_pos[2+i][ind], 
                                                        q_catch=int(catch_trials[2+i][ind]))
            keys = event.getKeys(timeStamped=my_clock)
            t_trial = my_clock.getTime()*1000
            #store data
            trials.addData('LocalTime_DDMMYY_HMS', 
                        str(time.localtime()[2]) + '/' + str(time.localtime()[1]) + '/' + str(time.localtime()[0]) 
                        + '_' + str(time.localtime()[3]) + ':' + str(time.localtime()[4]) + ':' + str(time.localtime()[5])) #HMS = hour min sec
            trials.addData('lab', lab)
            trials.addData('mode', mode)
            trials.addData('participant', info['Participant ID (***)'])
            trials.addData('gender', info['Gender'])
            trials.addData('age', info['Age'])
            trials.addData('trial', (trial_count)) #Python starts indexing at 
            trials.addData('start_position',start_pos[2+i][ind])
            trials.addData('last_position',last_pos[2+i][ind])
            trials.addData('catch_trial',catch_trials[2+i][ind])
            trials.addData('t_stim_1',stimulus_times[0])
            trials.addData('t_stim_2',stimulus_times[1])
            trials.addData('t_stim_3',stimulus_times[2])
            trials.addData('t_stim_4',stimulus_times[3])
            if keys:
                trials.addData('key_pressed',keys[-1][0])
                trials.addData('press_time',keys[-1][1]*1000)
            else:
                trials.addData('key_pressed',None)
                trials.addData('press_time',None)
            trials.addData('t_trail',t_trial)
            trials.addData('block',i+j+1)
            trials.addData('fix_dist',pilot_pos[j][i])
            trials.addData('spat_freq',pilot_freq[j][i])
            trials.addData('stim_size',pilot_size[j][i])

            if eye_tracking:
                eyeTracker.sendMessage('TRIAL_END') #this marks the end of the trial

            this_exp.nextEntry()
            
            trial_count+= 1

            if len(keys)>= 1:
                if keys[-1][0] == 'escape':
                    win.close()
                    core.quit()


        cross.autoDraw = False

if eye_tracking:
    eyeTracker.stopRecording() #this is typically done for each block

if eye_tracking:
    eyeTracker.setOfflineMode()
    pylink.pumpDelay(100)
    eyeTracker.closeDataFile() #close the EDF data file

    message.text = 'EDF data is transfering from EyeLink Host PC'
    message.draw()
    win.flip()
    pylink.pumpDelay(500)
    
# Close the window
win.close()
core.quit()


