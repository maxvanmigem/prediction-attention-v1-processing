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
from EyeLinkCoreGraphicsPsychoPy import EyeLinkCoreGraphicsPsychoPy #this are functions used to run the eyetracker calibration and validation



####################SELECT THE RIGHT LAB & Mode####################
lab = 'biosemi'   #'actichamp'/'biosemi'/'none'

mode = 'default'   #'default'/'DemoMode' #affects nr of trials per block (50%)

eye_tracking = True #True/False
###################################################################

# #dlg box#
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

# We download EDF data file from the EyeLink Host PC to the local hard
# drive at the end of each testing session, here we rename the EDF to
# include session start date/time

time_str = time.strftime("_%Y_%m_%d_%H_%M", time.localtime())
session_identifier = 'pp_' + info['Participant ID (***)'] + time_str

# create a folder for the current testing session in the "results" folder
session_folder = os.path.join(os.getcwd() + '/data/', session_identifier)
if not os.path.exists(session_folder):
    os.makedirs(session_folder)
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

####################################################
#Initialize trial variables
####################################################

#options
n_trials = 50 # per block
n_predictible = 35
n_catch = 2
n_blocks = 12        
n_fieldloc = 100
n_quadreps = 50
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
cross = ShapeStim(win, vertices=cross_vert, size=30, fillColor='white', lineWidth=0, pos=(0, 0), ori=0, autoDraw=False)

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

def stimPresentation(stimulus, stim_dur, isi_dur,iti_dur, start_quad, last, q_catch, lab=lab): 
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
    eegTriggerSend(int(99),lab)
    # Trial procedure
    for i,stimpos in enumerate(stimpos_ls):
        win.flip()
        stim_clock.reset()
        trigger = int(selectEEGStimulusTrigger(stimpos,pos=i)) # EEG trigger generation
        drawStim(stimulus,stimpos,catch_ls[i])
        wait = isi_dur/2-stim_clock.getTime()      # Drawing the stimulus takes time, this compensates that
        core.wait(wait)
        win.flip()
        eegTriggerSend(trigger,lab)  # Send trigger
        stim_clock.reset() # keep track of stim timing
        core.wait(stim_dur)
        win.flip()
        stim_timing =stim_clock.getTime() * 1000
        core.wait(isi_dur/2)
        stim_times.append(stim_timing)

    trial_time = trial_clock.getTime()*1000
    core.wait(iti_dur)
    return stim_times, trial_time

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

def generateFieldLocalizerTrials(uptrials,downtrials, asynchronous=False, isi=.5 , jitter=.050):
    """
    Generate triallist for the C1 localizer in the beginning of the experiment only distinguishing upper and lower visual field
    Also provides a list of timings to insert into presentation function when you want an asynchronous onset
    """
    # Make the proportions
    uppers = np.zeros(uptrials,dtype=int)
    downers = np.ones(downtrials,dtype=int)
    # Append and randomize
    localizer_list = np.concatenate([uppers,downers],axis=0)
    random.shuffle(localizer_list)
    # Timing lists
    n_stim = len(localizer_list)
    isi_list = np.repeat([isi],n_stim) # list of 500's or whatever isi is given
    if asynchronous:
        jitter_list = np.random.default_rng().uniform(-jitter,jitter,n_stim) # jitter
        # Add jitter to list of isi's
        for ind,t in enumerate(jitter_list):
            isi_list[ind] = (isi_list[ind] + t)/100

    return localizer_list, isi_list


def generateQuadLocalizerTrials(quad_reps, asynchronous=False, isi=.500 , jitter=.050):
    """
    Generate triallist for the C1 localizer in the beginning of the experiment this time for each quadrant
    Also provides a list of timings to insert into presentation function when you want an asynchronous onset
    """
    # Make the proportions and randomize
    localizer_prelist = [0,1,2,3]
    random.shuffle(localizer_prelist)
    localizer_list = np.repeat(localizer_prelist,quad_reps)
    # Timing lists
    isi_list = np.repeat([isi],quad_reps*4)
    jitter_list = np.random.default_rng().uniform(-jitter,jitter,quad_reps)
    if asynchronous:
        for ind,t in enumerate(jitter_list):
            isi_list[ind] = (isi_list[ind] + t)

    return localizer_list, isi_list


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

def generateBlockStarts(n_blocks = 8):
    """
    Generate a list with the start position for each block only for blocked design
    """
    startquad = np.repeat(np.arange(4),repeats= int(n_blocks/4))
    random.shuffle(startquad)  # randomize
    return startquad


def generateCatchTrials(tr_block, ncatch):
    """ 
    Some trials are catch trials, this function generates a list to designate which trials are
    """
    # Make a list with normal trials (0) and a list with catch trials (1)
    full_list = np.zeros(tr_block-ncatch,dtype=int)
    c_list = np.ones(ncatch*4,dtype=int) # make a longer list and cut it short otherwise it won't work with ncatch <4
    for i,one in enumerate(c_list):
        c_list[i] = int(one + i%4)   # Just adding 0,1,2 or 3 to every element (again: if ncatch is divisible by 4 then it's balanced)
    # Make correct length
    random.shuffle(c_list)
    c_list = c_list[:ncatch]

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

edf_file = 'pp_' + info['Participant ID (***)'] + '.EDF' #init datafile (max 8 characters)

if eye_tracking:
    try:
        el_tracker = pylink.EyeLink("100.1.1.1")
    except RuntimeError as error:
        print('ERROR:', error)
        core.quit()
        sys.exit()


    # Step 2: Open an EDF data file on the Host PC
    try:
        el_tracker.openDataFile(edf_file)
    except RuntimeError as err:
        print('ERROR:', err)
        # close the link if we have one open
        if el_tracker.isConnected():
            el_tracker.close()
        core.quit()
        sys.exit()

    el_tracker.sendCommand("add_file_preamble_text 'Predatt Experiment'") #add personalized data file header (preamble text)

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
        # Download the EDF data file from the Host PC to a local data folder
        # parameters: source_file_on_the_host, destination_file_on_local_drive
        local_edf = os.path.join(session_folder, session_identifier + '.EDF')
        try:
            el_tracker.receiveDataFile(edf_file, local_edf)
        except RuntimeError as error:
            print('ERROR:', error)

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

# Generate main stimuli (main exp and second localizer)
grid = generateStimCoordinates(gridcol=12, gridrow=10, jitter=linejitter_arr)
stimset = generateStim(linelength=35,linewidth=2, coord_array=grid, colour='white',catch_colour='red',
                        size=276, fixdistance=134)


# Localizer trials
## Field stim
field_grid = generateLocalizerStimCoordinates(24,6,jitter=linejitter_arr)
field_stim = generateStim(linelength=30,linewidth=2, coord_array=field_grid, colour='white',catch_colour='red',
                               size=276, fixdistance=100, localizer= True)
## Trials
field_loc_trials, field_timing = generateFieldLocalizerTrials(int(n_fieldloc),int(n_fieldloc))
quad_loc_trials, quad_timings = generateQuadLocalizerTrials(quad_reps=n_quadreps,asynchronous=True,jitter=.1) # we only use the timings here for the third localizer

# ExperimentHandler and TrialHandler

this_exp = data.ExperimentHandler(dataFileName = file_name)
# this_exp = data.ExperimentHandler(dataFileName = os.getcwd() + '/data/' + 'pilot_predatt_participant_test')

# Intialize trial handelr
trial_list = []
# start_pos = np.empty(n_blocks,dtype=object) #This start_pos list is not going to be used anymore
last_pos = np.empty(n_blocks,dtype=object)
catch_trials = np.empty(n_blocks,dtype=object)
for block in range(n_blocks):
    unsused_list, last_pos[block] = generateTrialList(tr_block=n_trials,npredictable=n_predictible) #the unsused list was formarly known as start_pos
    catch_trials[block] = generateCatchTrials(tr_block=n_trials,ncatch=n_catch)
    trial_list.append({'last_pos': last_pos[block]})
trials = data.TrialHandler(trialList = trial_list, nReps=1, method = 'sequential')
this_exp.addLoop(trials)

# The starting position for each trial is now the same whithin the block there is still infrastructure that allows randomized start postition
# but that is not used anymore, So instead now there as a list with the length of n_blocks containing the start position for each block
start_pos = generateBlockStarts(n_blocks=n_blocks)

# initilize counters
my_clock = core.Clock() #define my_clock
n_correct = 0
point_counter = 0
trial_count = 1
block_count = 1
should_recal = 'no'

####################################################
#Experiment loop
####################################################

message.text = 'localizer 1'
message.draw()
win.flip()

start_key = event.waitKeys(keyList = ['space','escape'])
if start_key == 'escape':
    core.quit() #no event.clearEvents() necessary
event.clearEvents()

if eye_tracking:
    el_tracker.setOfflineMode() #this is called before start_recording() to make sure the eye tracker has enough time to switch modes (to start recording)
    pylink.pumpDelay(100)
    #starts the EyeLink tracker recording, sets up link for data reception if enabled.
    el_tracker.startRecording(1,1,1,1) 
    # The 1,1,1,1 just has to do with whether samples and events etcetera needs to be written to EDF file. Recording needs to be started for each block
    pylink.pumpDelay(100) #wait for 100 ms to cache some samples

# First localizer
cross.autoDraw = True
eegTriggerSend(int(93),lab=lab) # signal start localizer
for ind,x in enumerate(field_loc_trials):
    event.clearEvents(eventType = 'keyboard')
    fieldLocalizer(x,field_stim,.1,.5)
    # Exit
    keys = event.getKeys()
    if 'escape' in keys:
        if eye_tracking:
            terminate_task()
        win.close()
        core.quit()
eegTriggerSend(int(94),lab=lab) #signal end
# Pause recordings
if eye_tracking:
    el_tracker.stopRecording() #this is typically done for each bloc

# Intermediate message    
cross.autoDraw = False
message.text = 'Localizer 2'
message.draw()
win.flip()
start_key = event.waitKeys(keyList = ['space','escape'])
if start_key == 'escape':
    if eye_tracking:
        terminate_task()
    win.close()
    core.quit()

# Second localizer
# Make list
quad_loc_set = np.arange(4)
random.shuffle(quad_loc_set) 
eegTriggerSend(int(95),lab=lab) # signal start
# Start loop for every quad
for i in quad_loc_set:
    eegTriggerSend(int(95),lab=lab) # signal start
    cross.autoDraw = True
    for j in range(n_quadreps):
        event.clearEvents(eventType = 'keyboard')
        fieldLocalizer(i,stimset,.1,.5)
        # Exit
        keys = event.getKeys()
        if 'escape' in keys:
            if eye_tracking:
                terminate_task()
            win.close()
            core.quit()
    # Pause recordings
    if eye_tracking:
        el_tracker.stopRecording() #this is typically done for each bloc
    # Intermediate message 
    cross.autoDraw = False
    message.text = 'Pause'
    message.draw()
    win.flip()
    start_key = event.waitKeys(keyList = ['space','escape'])
    if start_key == 'escape':
        if eye_tracking:
            terminate_task()
        win.close()
        core.quit()
    eegTriggerSend(int(96),lab=lab) # signal end localizer block

# Intermediate message    
cross.autoDraw = False
message.text = 'Localizer 3'
message.draw()
win.flip()
start_key = event.waitKeys(keyList = ['space','escape'])
if start_key == 'escape':
    if eye_tracking:
        terminate_task()
    win.close()
    core.quit()

# Third localizer
# Make list
quad_loc_set = np.arange(4)
random.shuffle(quad_loc_set) 
# Start loop for every quad
for i in quad_loc_set:
    eegTriggerSend(int(97),lab=lab) # signal start
    cross.autoDraw = True
    for j in range(n_quadreps):
        event.clearEvents(eventType = 'keyboard')
        fieldLocalizer(i,stimset,.1,quad_timings[j])
        # Exit
        keys = event.getKeys()
        if 'escape' in keys:
            if eye_tracking:
                terminate_task()
            win.close()
            core.quit()
    # Pause recordings
    if eye_tracking:
        el_tracker.stopRecording() #this is typically done for each bloc
    # Intermediate message 
    cross.autoDraw = False
    message.text = 'Pause'
    message.draw()
    win.flip()
    start_key = event.waitKeys(keyList = ['space','escape'])
    if start_key == 'escape':
        if eye_tracking:
            terminate_task()
        win.close()
        core.quit()
    eegTriggerSend(int(98),lab=lab) # signal end localizer block

# the first three blocks are different positions
for i in range(n_blocks):
    message.text = f'Block {block_count}'
    message.draw()
    win.flip()
    start_key = event.waitKeys(keyList = ['space','escape'])
    if start_key == 'escape':
        break

     # Trigger for block
    block_tigger = int(60 + n_blocks)
    eegTriggerSend(block_tigger,lab=lab)
    cross.autoDraw = True
    if eye_tracking:
        el_tracker.setOfflineMode() #this is called before start_recording() to make sure the eye tracker has enough time to switch modes (to start recording)
        pylink.pumpDelay(100)
        #starts the EyeLink tracker recording, sets up link for data reception if enabled.
        el_tracker.startRecording(1,1,1,1) 
        # The 1,1,1,1 just has to do with whether samples and events etcetera needs to be written to EDF file. Recording needs to be started for each block
        pylink.pumpDelay(100) #wait for 100 ms to cache some samples
        
    for ind in range(n_trials):
        win.flip()
        if eye_tracking:
            el_tracker.sendMessage('TRIALID %d' % (trial_count)) #send a message ("TRIALID") to mark the start of a trial
            el_tracker.sendCommand("record_status_message 'trial %s block %s'" % (trial_count,block_count)) #to show the current task, block nr and trial nr #+1 because Python starts at 0
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
        my_clock.reset() #start rt timing
        stimulus_times,trial_stamp = stimPresentation(stimulus=stimset, stim_dur=0.1 , isi_dur=0.5, iti_dur=0.501,
                                                    start_quad=start_pos[i], 
                                                    last=last_pos[i][ind], 
                                                    q_catch=int(catch_trials[i][ind]))
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
        trials.addData('start_position',start_pos[i])
        trials.addData('last_position',last_pos[i][ind])
        trials.addData('catch_trial',catch_trials[i][ind])
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

        trials.addData('t_trial',t_trial)
        trials.addData('block',block_count)
        if eye_tracking:
            el_tracker.sendMessage('TRIAL_END') #this marks the end of the trial

        this_exp.nextEntry()
        trial_count+= 1
        if len(keys)>= 1:
            if keys[-1][0] == 'escape':
                if eye_tracking:
                    terminate_task()
                win.close()
                core.quit()

    block_count+= 1
    cross.autoDraw = False
    # Pause recordings
    if eye_tracking:
        el_tracker.stopRecording() #this is typically done for each bloc

# Disconnect, download the EDF file, then terminate the task
if eye_tracking:
    terminate_task()

# Close the window
win.close()
core.quit()


