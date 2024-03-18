'''
Created on Tue Oct 24 2023

@author: Max Van Migem
'''
import numpy as np
import os
import time
import random
import sys
import pylink #the last is to communicate with the eyetracker
from psychopy import parallel, visual, gui, data, event, core, monitors
from psychopy.visual import ShapeStim
from psychopy import logging
from math import fabs
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy

####################SELECT THE RIGHT LAB & Mode####################
lab = 'biosemi'   #'actichamp'/'biosemi'/'none'

mode = 'default'   #'default'/'DemoMode' #affects nr of trials per block (50%)

eye_tracking = True #True/False

sub = 1
###################################################################

# Define a monitor
system_monitor = monitors.Monitor('cap_lab_monitor')   
# # This is to set a new configuration for your monitor
# system_monitor.setSizePix((1024, 768))
# system_monitor.setWidth(39.2)
# system_monitor.setDistance(65)
# system_monitor.save()

# #initialize window#
win = visual.Window(fullscr=True,color= (-1, -1, -1), colorSpace = 'rgb', units = 'pix', monitor= system_monitor)
win.mouseVisible = False

###################################################
#Initialize trial variables
####################################################

#options
n_quadreps = 6 # per quad
n_trials = n_quadreps*4

# random generator
rng = np.random.default_rng(seed=None)
# array with random jitter for horizontal stim lines (this might have to be a standard array as in the same for everyone)
stim_jitter_path = os.getcwd()+'/stim_jitter.npy'
linejitter_arr = np.load(stim_jitter_path)

####################################################
#Create Psychopy simple visual objects
####################################################

test_instruction = ('Test')


message = visual.TextStim(win, text='',height= 30) 

# manual input of fixation cross vertices, very hard coded but it's fast and customizable
cross_vert = [(-.05, .05), (-.05, .35), (.05, .35,), (.05, .05),
              (.35, .05), (.35,-.05), (.05, -.05),(.05,-.35),
              (-.05,-.35), (-.05, -.05), (-.35,-.05), (-.35, .05)]

# defining fixcross using vertices above, you can change the size, colour, etc.. of the fixation cross here   
cross = ShapeStim(win, vertices=cross_vert, size=30, fillColor='white',
                   lineWidth=0, pos=(0, 0), ori=0, autoDraw=False)

teststim = ShapeStim(win, vertices=[(0.5,1.0), (1.0,1.0), (0.5,1.0), (1.0,1.0)],
                     size=200, fillColor='white', lineWidth=1, pos=(0, 0), ori=0, closeShape=False)


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


def generateLocalizerStimCoordinates(gridcol, gridrow, jitter):
    """
    Generates coordinates used to draw the localizer stimulus lines
    This is partically the same function as generateStimCoordinates() but for upper and lower visual field
    instead of quandrant
    """
    # Calculate spacing of grid
    x_spacing = 2.0 / (gridcol)
    y_spacing = 1.0 / (gridrow)

    # Create an array to store coordinates: every point (x,y) of the grid for every quadrant (4) 
    coord_array = np.empty(shape = (gridcol,gridrow,2,2),dtype= 'object')
    field_set = [1,-1]

    # Generate grid coord per quadrant
    for i,field in enumerate(field_set):
        # Per row
        for row in range(gridrow):
            # Per column
            for col in range (gridcol):

                grid_x = -1 + col * x_spacing  # grid points
                grid_y = row * y_spacing

                grid_y = grid_y * field   # which visual field up/down

                coord_array[col,row,0,i] = grid_x  # Store them in big ass array
                coord_array[col,row,1,i] = grid_y

    # flip the matrices so that they have the 'orientation' corresponding to the quadrant 
    coord_array[:,:,0,0] = np.flip(coord_array[:,:,0,0], axis=0) 
    coord_array[:,:,1,0] = np.flip(coord_array[:,:,1,0], axis=0) 


    #  Add jitter
    for i in range(2):
        coord_array[:,:,:,i] =  coord_array[:,:,:,i] + jitter[:gridcol,:gridrow,:]
  
    return coord_array


def generateStim(linelength, linewidth ,coord_array, colour, catch_colour, size, fixdistance, localizer = False ):
    """ 
    Generate an ElementArrayStim object consisting of lines on the coordinates for stimulus for each quadrant
    Then four more arrays are created each representing a catch trial in a separate quadrant
    Returns a 5x4 array of lists with each list containing the line stimuli for a quadrant
    If localizer is set to True then it generates a grid in the upper or lower visual field instead of quadrants
    """
    # This 
    if localizer:
        sections = 2
        dist = fixdistance
        quads = [[0,1],[0,-1]]
    else:
        sections = 4
        # This is to calculate the distance to the fixation cross
        dist = np.sqrt(np.square(fixdistance)/2) #pythagoras
        quads = [[-1,1],[1,1],[1,-1],[-1,-1]]

    # Get size of grid
    gridcol = np.shape(coord_array)[0]
    gridrow = np.shape(coord_array)[1]

    # Init arrays to store line objects per quadrant and also for every possible catch trial 
    line_stimuli = np.empty((2,sections), dtype=object)   # 2 types (0 is normal white lines, 1 is catch stim ) and 4 quadrants
 
    for quad in range(sections):
        n_lines = gridcol*gridrow
        xys = np.empty((n_lines,2)) #coordinates to pass to elementArrayStim
        it_count = 0 # idk man... this is a work around because im too lazy to completely change the structure that I used before i.e. two nested loops per column (below)
        for row in range(gridrow):
            for col in range(gridcol):

                # Define stim coordinates
                the_x = (coord_array[col,row,0,quad] * size) + (quads[quad][0]*dist)  # size and distance from fixation are all in here 
                the_y = (coord_array[col,row,1,quad] * size) + (quads[quad][1]*dist)              
                                                                              
                xys[it_count,0] = the_x
                xys[it_count,1] = the_y
                it_count += 1
        sizes = np.atleast_2d([linelength,linewidth]).repeat(repeats=n_lines, axis=0)
        # Normal stimuli
        line_stimuli[0,quad] = visual.ElementArrayStim(win, units='pix',elementTex=None, elementMask='sqr', xys= xys,
                                           nElements=n_lines, sizes=sizes, colors=(1.0, 1.0, 1.0), colorSpace='rgb')
        # Catch stimuli
        line_stimuli[1,quad] = visual.ElementArrayStim(win, units='pix',elementTex=None, elementMask='sqr', xys= xys,
                                           nElements=n_lines, sizes=sizes, colors=catch_colour, colorSpace='rgb')

    return line_stimuli


def drawStim(line_stimuli, quad, catch):
    """ 
    Draws the stimuli in the correct quadrant
    Catch = 0 means normal color catch = 1 means catch
    """
    # select quadrant and whether this is a catch stim
    line_stim = line_stimuli[catch,quad] 

    # draw 
    line_stim.draw()


####################################################
#Trial display functions
####################################################

def fieldLocalizer(field,line_stim, stim_dur, isi_dur, lab=lab):
    """
    Function for the presentation of localizer which tries to differentiate the C1 for upper and lower visual field
    """
    # Check timing
    stim_clock = core.Clock()
    win.flip()
    stim_clock.reset()
    # Set eeg trigger here 80 means upper field and 81 means lower field
    trigger = int(80 + field)
    # Drawing stim, this should be generated with generateStim() with localizer set to True
    drawStim(line_stim, field, 0)
    wait = isi_dur/2-stim_clock.getTime()
    core.wait(wait)
    win.flip()
    eegTriggerSend(trigger,lab)  # Send trigger
    stim_clock.reset() # keep track of stim timing
    core.wait(stim_dur)
    win.flip()
    stim_timing =stim_clock.getTime() * 1000
    core.wait(isi_dur/2)

    return stim_timing


####################################################
#Trial sequence functions
####################################################

def generateQuadLocalizerTrials(quad_reps, asynchronous=False, isi=.500 , jitter=.050):
    """
    Generate triallist for the C1 localizer in the beginning of the experiment this time for each quadrant
    Also provides a list of timings to insert into presentation function when you want an asynchronous onset
    """
    # Make the proportions and randomize
    localizer_prelist = [0,1,2,3]
    localizer_list = np.repeat(localizer_prelist,quad_reps)
    random.shuffle(localizer_list)
    # Timing lists
    isi_list = np.repeat([isi],quad_reps*4)
    jitter_list = np.random.default_rng().uniform(-jitter,jitter,quad_reps)
    if asynchronous:
        for ind,t in enumerate(jitter_list):
            isi_list[ind] = (isi_list[ind] + t)

    return localizer_list, isi_list


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

# Select EEG stimulus trigger
def selectEEGStimulusTrigger(start,pos):
    """
    distinguish EEG trigger for stimulus
    """
    eeg_stim_trigger = int((start+1)*10 + (pos+1))

    return eeg_stim_trigger


#just setting up the EEG
if lab == 'actichamp':
    gsr_port = parallel.ParallelPort(address=0xCFB8)
elif lab == 'biosemi':
    gsr_port = parallel.ParallelPort(address=0x3FB8)

####################################################
#Eye-tracker set-up
####################################################
# Step 1: Connect to the EyeLink Host PC
if eye_tracking:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print('ERROR:', error)
        core.quit()
        sys.exit()


    # Step 3: Configure the tracker
    #
    # Put the tracker in offline mode before we change tracking parameters
    el_tracker.setOfflineMode()

    # Get the software version:  1-EyeLink I, 2-EyeLink II, 3/4-EyeLink 1000,
    # 5-EyeLink 1000 Plus, 6-Portable DUO
    eyelink_ver = 0  # set version to 0, in case running in Dummy mode
    if eye_tracking:
        vstr = el_tracker.getTrackerVersionString()
        eyelink_ver = int(vstr.split()[-1].split('.')[0])
        # print out some version info in the shell
        print('Running experiment on %s, version %d' % (vstr, eyelink_ver))

    # File and Link data control
    # what eye events to save in the EDF file, include everything by default
    file_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT'
    # what eye events to make available over the link, include everything by default
    link_event_flags = 'LEFT,RIGHT,FIXATION,SACCADE,BLINK,BUTTON,FIXUPDATE,INPUT'
    # what sample data to save in the EDF data file and to make available
    # over the link, include the 'HTARGET' flag to save head target sticker
    # data for supported eye trackers
    if eyelink_ver > 3:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,HTARGET,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,HTARGET,STATUS,INPUT'
    else:
        file_sample_flags = 'LEFT,RIGHT,GAZE,HREF,RAW,AREA,GAZERES,BUTTON,STATUS,INPUT'
        link_sample_flags = 'LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT'
    el_tracker.sendCommand("file_event_filter = %s" % file_event_flags)
    el_tracker.sendCommand("file_sample_data = %s" % file_sample_flags)
    el_tracker.sendCommand("link_event_filter = %s" % link_event_flags)
    el_tracker.sendCommand("link_sample_data = %s" % link_sample_flags)

    el_tracker.sendCommand("sample_rate 1000") #250, 500, 1000, or 2000 (only for EyeLink 1000 plus)
    el_tracker.sendCommand("recording_parse_type = GAZE")
    el_tracker.sendCommand("select_parser_configuration 0") #saccade detection thresholds: 0-> standard/coginitve, 1-> sensitive/psychophysiological
    el_tracker.sendCommand("calibration_type = HV9") #13 point calibration (recommended for head free remote mode)

    # Step 4: set up a graphics environment for calibration

    # get the native screen resolution used by PsychoPy
    scn_width, scn_height = win.size

    # Pass the display pixel coordinates (left, top, right, bottom) to the tracker
    # see the EyeLink Installation Guide, "Customizing Screen Settings"
    el_coords = "screen_pixel_coords = 0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendCommand(el_coords)

    # Write a DISPLAY_COORDS message to the EDF file
    # Data Viewer needs this piece of info for proper visualization, see Data
    # Viewer User Manual, "Protocol for EyeLink Data to Viewer Integration"
    dv_coords = "DISPLAY_COORDS  0 0 %d %d" % (scn_width - 1, scn_height - 1)
    el_tracker.sendMessage(dv_coords)

    # Configure a graphics environment (genv) for tracker calibration
    genv = EyeLinkCoreGraphicsPsychoPy(el_tracker, win)
    print(genv)  # print out the version number of the CoreGraphics library

    # Set background and foreground colors for the calibration target
    # in PsychoPy, (-1, -1, -1)=black, (1, 1, 1)=white, (0, 0, 0)=mid-gray
    foreground_color = (1, 1, 1)
    background_color = win.color
    genv.setCalibrationColors(foreground_color, background_color)
    
    # Set up the calibration target
    # Use the default calibration target ('circle')
    genv.setTargetType('circle')

    # Configure the size of the calibration target (in pixels)
    # this option applies only to "circle" and "spiral" targets
    genv.setTargetSize(24)

    # Beeps to play during calibration, validation and drift correction
    # parameters: target, good, error
    #     target -- sound to play when target moves
    #     good -- sound to play on successful operation
    #     error -- sound to play on failure or interruption
    # Each parameter could be ''--default sound, 'off'--no sound, or a wav file
    genv.setCalibrationSounds('', '', '')

    # Request Pylink to use the PsychoPy window we opened above for calibration
    pylink.openGraphicsEx(genv)
        
if eye_tracking:
    message.text = 'Setting up the tracker, please wait\n' #show calibration message
    message.draw()
    win.flip()
    try:
        el_tracker.doTrackerSetup()
    except RuntimeError as err:
        print('ERROR:', err)
        el_tracker.exitCalibration() #calibrate the tracker #once you are happy with the calibration and validation (!), you are ready to run the experiment. 

# Eye-tracker termination function
def terminate_task():
    """ Terminate the task gracefully and retrieve the EDF data file

    file_to_retrieve: The EDF on the Host that we would like to download
    win: the current window used by the experimental script
    """
    el_tracker = pylink.getEYELINK()
    if el_tracker.isConnected():
        # Terminate the current trial first if the task terminated prematurely
        error = el_tracker.isRecording()
        if error == pylink.TRIAL_OK:
            el_tracker = pylink.getEYELINK()
            # Stop recording
            if el_tracker.isRecording():
                # add 100 ms to catch final trial events
                pylink.pumpDelay(100)
                el_tracker.stopRecording()
        # Put tracker in Offline mode
        el_tracker.setOfflineMode()
        # Clear the Host PC screen and wait for 500 ms
        el_tracker.sendCommand('clear_screen 0')
        pylink.msecDelay(500)
        # Close the edf data file on the Host
        el_tracker.closeDataFile()
        # Show a file transfer message on the screen
        message.text = 'EDF data is transfering from EyeLink Host PC'
        message.draw()
        win.flip()
        pylink.pumpDelay(500)
        # Close the link to the tracker.
        el_tracker.close()

def checkGazeOnFix():
    # Here we implement a gaze trigger, so the target only comes up when
    # the subject direct gaze to the fixation cross
    event.clearEvents()  # clear cached PsychoPy events
    # determine which eye(s) is/are available
    # 0- left, 1-right, 2-binocular
    eye_used = el_tracker.eyeAvailable()
    new_sample = None
    old_sample = None
    trigger_fired = False
    in_hit_region = False
    should_recali = 'no'
    trigger_start_time = core.getTime()
    # fire the trigger following a 300-ms gaze
    minimum_duration = 0.3
    gaze_start = -1
    while not trigger_fired:
        # abort the current trial if the tracker is no longer recording
        error = el_tracker.isRecording()
        if error is not pylink.TRIAL_OK:
            el_tracker.sendMessage('tracker_disconnected')
            return error

        # if the trigger did not fire in 30 seconds, abort trial
        if core.getTime() - trigger_start_time >= 10.0:
            el_tracker.sendMessage('trigger_timeout_recal')
            # re-calibrate before trial
            should_recali = 'yes'
            return should_recali

        # check for keyboard events, skip a trial if ESCAPE is pressed
        # terminate the task is Ctrl-C is pressed
        for keycode, modifier in event.getKeys(modifiers=True):
            # Abort a trial and recalibrate if "ESCAPE" is pressed
            if keycode == 'escape':
                el_tracker.sendMessage('abort_and_recal')
                # re-calibrate now trial
                should_recali = 'yes'
                return should_recali

        # Do we have a sample in the sample buffer?
        # and does it differ from the one we've seen before?
        new_sample = el_tracker.getNewestSample()
        if new_sample is not None:
            if old_sample is not None:
                if new_sample.getTime() != old_sample.getTime():
                    # check if the new sample has data for the eye
                    # currently being tracked; if so, we retrieve the current
                    # gaze position and PPD (how many pixels correspond to 1
                    # deg of visual angle, at the current gaze position)
                    if eye_used == 1 and new_sample.isRightSample():
                        g_x, g_y = new_sample.getRightEye().getGaze()
                    if eye_used == 0 and new_sample.isLeftSample():
                        g_x, g_y = new_sample.getLeftEye().getGaze()

                    # break the while loop if the current gaze position is
                    # in a 120 x 120 pixels region around the screen centered
                    fix_x, fix_y = (scn_width/2.0, scn_height/2.0)
                    if fabs(g_x - fix_x) < 60 and fabs(g_y - fix_y) < 60:
                        # record gaze start time
                        if not in_hit_region:
                            if gaze_start == -1:
                                gaze_start = core.getTime()
                                in_hit_region = True
                        # check the gaze duration and fire
                        if in_hit_region:
                            gaze_dur = core.getTime() - gaze_start
                            if gaze_dur > minimum_duration:
                                trigger_fired = True
                    else:  # gaze outside the hit region, reset variables
                        in_hit_region = False
                        gaze_start = -1

            # update the "old_sample"
            old_sample = new_sample
    return should_recali


####################################################
#Experiment value initialization
####################################################

# Generate main stimuli
grid = generateStimCoordinates(gridcol=12, gridrow=10, jitter=linejitter_arr)
stimset = generateStim(linelength=35,linewidth=2, coord_array=grid, colour='white',catch_colour='red',
                        size=276, fixdistance=134)

quad_loc_trials, quad_timings = generateQuadLocalizerTrials(quad_reps=n_quadreps,asynchronous=True,isi= .52,jitter=.070) 

# Intermediate message    
cross.autoDraw = False
message.text = 'Localizer'
message.draw()
win.flip()
start_key = event.waitKeys(keyList = ['space','escape'])
if start_key == 'escape':
    if eye_tracking:
        terminate_task()
    win.close()
    core.quit()


####################################################
# Main localizer loop
####################################################

# Start EEG recording
eegTriggerSend(int(200),lab=lab) # 200 is the start-recording command in the biosemi config file
# Start block
eegTriggerSend(int(99),lab=lab) # signal start
cross.autoDraw = True
if eye_tracking:
        el_tracker.setOfflineMode() #this is called before start_recording() to make sure the eye tracker has enough time to switch modes (to start recording)
        pylink.pumpDelay(100)
        #starts the EyeLink tracker recording, sets up link for data reception if enabled.
        el_tracker.startRecording(1,1,1,1) 
        # The 1,1,1,1 just has to do with whether samples and events etcetera needs to be written to EDF file. Recording needs to be started for each block
        pylink.pumpDelay(100) #wait for 100 ms to cache some samples

for i in range(n_trials):
    win.flip()
    if eye_tracking:
        # Check if gaze is on fixation and if not start recallibration
        should_recal = checkGazeOnFix()
        if should_recal == 'yes':
            cross.autoDraw = False
            message.text = 'Please press ENTER twice to recalibrate the tracker'
            message.draw()
            win.flip()
            try:
                el_tracker.doTrackerSetup()
            except RuntimeError as err:
                print('ERROR:', err)
                el_tracker.exitCalibration()
            should_recal = 'no'
            el_tracker.setOfflineMode() #this is called before start_recording() to make sure the eye tracker has enough time to switch modes (to start recording)
            pylink.pumpDelay(100)
            #starts the EyeLink tracker recording, sets up link for data reception if enabled.
            el_tracker.startRecording(1,1,1,1) 
            # The 1,1,1,1 just has to do with whether samples and events etcetera needs to be written to EDF file. Recording needs to be started for each block
            pylink.pumpDelay(100) #wait for 100 ms to cache some samples
            cross.autoDraw =True
    event.clearEvents(eventType = 'keyboard')
    fieldLocalizer(quad_loc_trials[i],stimset,.1,quad_timings[i])
    # Exit
    keys = event.getKeys()
    if 'escape' in keys:
        if eye_tracking:
            terminate_task()
        break
# End eye_tracker recording
if eye_tracking:
    el_tracker.stopRecording() #this is typically done for each bloc
    
# End EEG recording
eegTriggerSend(int(201),lab=lab) # 200 is the end-recording command in the biosemi config file

# Intermediate message 
cross.autoDraw = False
message.text = 'End'
message.draw()
win.flip()
start_key = event.waitKeys(keyList = ['space','escape'])

# Disconnect then terminate the task
if eye_tracking:
    terminate_task()

# Close the window
win.close()
core.quit()

