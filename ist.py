"""
Created on 24/03/2021
@author sebastian
"""

import broadbean as bb
try:
    from broadbean.plotting import plotter
except:
    plotter = bb.plotter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8, 3)
mpl.rcParams['figure.subplot.bottom'] = 0.15

# The pulsebuilding module comes with a (small) collection of functions appropriate for being segments.
bb_ramp = bb.PulseAtoms.ramp  # args: start, stop
bb_sine = bb.PulseAtoms.sine  # args: freq, ampl, off, phase

# make a blueprint
def rect(samples, amp, f_sam = 2.4e9):
    bp = bb.BluePrint()
    dur = samples/f_sam
    bp.insertSegment(0, bb_ramp, (amp, amp), dur=dur)
    bp.setSR(f_sam)
    return bp

def ramp(samples, start, stop, f_sam = 2.4e9):
    bp = bb.BluePrint()
    dur = samples/f_sam
    bp.insertSegment(0, bb_ramp, (start, stop), dur=dur)
    bp.setSR(f_sam)
    return bp


f_sam = 2.4e9  # sampling rate
period = 1e-3  # pulsetime
ns = 1e-9  #nanosecond
GainL = 1/6e-3  #gain for left gate, now you can put the values in mV
GainR = 1/6e-3
GainC = 1/6e-3
Etime = 100*ns*f_sam  # time in load and empty 1us
Pihalftime = 16 #0.25/36e6*f_sam  #choose Bfield so that Rabifreq is 0.01 times sampling rate
#cvar Ltime  # time in load and empty 1us
Inittime = 10*ns*f_sam
MeasureTime = 2.5e-3*period*f_sam+16  #20us in Measure, is within T1
marker_pos = 2*Etime
Input = 0.1
LoadAmp = Input*10/4*3    #amplitude towards (1,1)
Ltime = 1.25e-9*f_sam
w_emptyR = rect(Etime,-0e-3*GainR*0.2)  # factor 1/2 to not be larger than 1
w_initR = ramp(Inittime,0,-20e-3*GainR*0.2)
w_PihalfPulseR = rect(Pihalftime, 30e-3*GainR*0.2)
w_minus_PihalfPulseR = rect(Pihalftime, -30e-3*GainR*0.2)
w_loadR = rect(Ltime,LoadAmp*GainR*0.2)  #here
w_emptyL = rect(Etime,0e-3*GainL*0.2)
w_initL = ramp(Inittime,0,20e-3*GainL*0.2)
w_PihalfPulseL = rect(Pihalftime, -30e-3*GainR*0.2)
w_minus_PihalfPulseL = rect(Pihalftime, 30e-3*GainR*0.2)
w_loadL = rect(Ltime,-LoadAmp*GainR*0.2)  #here
#wave w_markerleft = marker(marker_pos/2, 1)
#wave w_markerMiddle = marker(marker_pos/2, 0)
#wave w_markerRight = marker(MeasureTime, 0)
w_measureR = rect(MeasureTime,0*GainR*0.2) #dont use up waveform mempey to stay in 0,0
w_measureL = rect(MeasureTime,0*GainL*0.2)
#100 us pulse, 1 us in empty, 1 us in load and almost all the time in measure
#make it zero average
w_pulseR = w_minus_PihalfPulseR + w_minus_PihalfPulseR + w_emptyR + w_initR + w_PihalfPulseR + w_loadR + w_PihalfPulseR #+ w_measureR
w_pulseL = w_minus_PihalfPulseL + w_minus_PihalfPulseL + w_initL + w_emptyL + w_PihalfPulseL + w_loadL + w_PihalfPulseL + w_measureL
plotter(w_pulseL)
plt.show()
plotter(w_pulseR)
plt.show()
'''
#We want to step the detuning from 0 to 12 mV in as many steps as possible
#the amount of steps depends on the max number of samples the sequencer can store
for i in range(200):
  Ltime = i*1.25e-9*f_sam  #time in (1,1) in steps of 10 ns, i=100 is 1 us
  w_measureR = 0*GainR*rect(MeasureTime-2*Ltime,0.2) 
  w_measureL = 0*GainL*rect(MeasureTime-2*Ltime,0.2) 
  w_loadL = -LoadAmp*GainL*rect(Ltime, 0.2) 
  w_loadR = LoadAmp*GainR*rect(Ltime, 0.2) 
  w_initL = LoadAmp*GainL*rect(Ltime,0.2) 
  w_initR = -LoadAmp*GainR*rect(Ltime,0.2) 
  w_pulseR = w_pulseR  + w_minus_PihalfPulseR + w_minus_PihalfPulseR + w_initR + w_emptyR + w_PihalfPulseR+ w_loadR  + w_PihalfPulseR +w_measureR
  w_pulseL = w_pulseL +  w_minus_PihalfPulseL + w_minus_PihalfPulseL + w_initL + w_emptyL +  w_PihalfPulseL + w_loadL +  w_PihalfPulseL + w_measureL
'''
#wave w_marker = join(w_markerleft, w_markerMiddle, w_markerRight) var time
'''cvar i 
var j 
const f_sam = 2.4e9  # sampling rate
const period = 1e-3  # pulsetime
const ns = 1e-9  #nanosecond
const GainL = 1/6e-3  #gain for left gate, now you can put the values in mV
const GainR = 1/6e-3 
const GainC = 1/6e-3 
const Etime = 100*ns*f_sam  # time in load and empty 1us 
const Pihalftime = 16 #0.25/36e6*f_sam  #choose Bfield so that Rabifreq is 0.01 times sampling rate
cvar Ltime  # time in load and empty 1us 
const Inittime = 0*ns*f_sam 
const MeasureTime = 2.5e-3*period*f_sam+16  #20us in Measure, is within T1 
const marker_pos = 2*Etime 
const LoadAmp = Input*10/4*3    #amplitude towards (1,1)
Ltime = 0 
wave w_emptyR = -0e-3*GainR*rect(Etime,0.2)  # factor 1/2 to not be larger than 1
wave w_initR = -20e-3*GainR*ramp(Inittime,0,0.2) 
wave w_PihalfPulseR = 30e-3*GainR*rect(Pihalftime, 0.2) 
wave w_loadR = LoadAmp*GainR*rect(Ltime,0.2)  #here
wave w_emptyL = 0e-3*GainL*rect(Etime,0.2) 
wave w_initL = 20e-3*GainL*ramp(Inittime,0,0.2) 
wave w_PihalfPulseL = -30e-3*GainR*rect(Pihalftime, 0.2) 
wave w_loadL = -LoadAmp*GainR*rect(Ltime,0.2)  #here
#wave w_markerleft = marker(marker_pos/2, 1) 
#wave w_markerMiddle = marker(marker_pos/2, 0) 
#wave w_markerRight = marker(MeasureTime, 0)  
wave w_measureR = 0*GainR*rect(MeasureTime,0.2) #dont use up waveform mempey to stay in 0,0
wave w_measureL = 0*GainL*rect(MeasureTime,0.2) 
#100 us pulse, 1 us in empty, 1 us in load and almost all the time in measure
#make it zero average
wave w_pulseR = join(- w_PihalfPulseR, - w_PihalfPulseR, w_emptyR,  w_initR, w_PihalfPulseR, w_loadR, w_PihalfPulseR, w_measureR) 
wave w_pulseL = join( - w_PihalfPulseL, - w_PihalfPulseL,  w_initL,w_emptyL,  w_PihalfPulseL, w_loadL, w_PihalfPulseL, w_measureL) 
#We want to step the detuning from 0 to 12 mV in as many steps as possible
#the amount of steps depends on the max number of samples the sequencer can store
for (i = 0  i < 200  i++) {
  Ltime = i*1.25e-9*f_sam  #time in (1,1) in steps of 10 ns, i=100 is 1 us
  w_measureR = 0*GainR*rect(MeasureTime-2*Ltime,0.2) 
  w_measureL = 0*GainL*rect(MeasureTime-2*Ltime,0.2) 
  w_loadL = -LoadAmp*GainL*rect(Ltime, 0.2) 
  w_loadR = LoadAmp*GainR*rect(Ltime, 0.2) 
  w_initL = LoadAmp*GainL*rect(Ltime,0.2) 
  w_initR = -LoadAmp*GainR*rect(Ltime,0.2) 
  w_pulseR = join(w_pulseR, - w_PihalfPulseR, - w_PihalfPulseR, w_initR, w_emptyR, w_PihalfPulseR, w_loadR, w_PihalfPulseR, w_measureR) 
  w_pulseL = join(w_pulseL, - w_PihalfPulseL, - w_PihalfPulseL, w_initL, w_emptyL,  w_PihalfPulseL, w_loadL, w_PihalfPulseL, w_measureL) 
}
#wave w_marker = join(w_markerleft, w_markerMiddle, w_markerRight) 
#wave w_pulseL_marker = w_pulseL+w_marker 
while(true){
  time = getUserReg(0)-1 
  for (j = 0  j < 1  j++) {
     playWaveIndexed(1, w_pulseL, 3, w_pulseR, 4, w_pulseL, time*6320, 6320) 
     waitWave() 
     #wait(10e-6/3.3e-9) #wait 10 us in Measurepoint
  }
  #wait(3e-6/3.3e-9) 
   #wait for 6 us, 3 us effective and 3 us to read the UserRegister

#wave w_pulseL_marker = w_pulseL+w_marker 
while(true){
  time = getUserReg(0)-1 
  for (j = 0  j < 1  j++) {
     playWaveIndexed(1, w_pulseL, 3, w_pulseR, 4, w_pulseL, time*6320, 6320) 
     waitWave() 
     #wait(10e-6/3.3e-9) #wait 10 us in Measurepoint
  }
  #wait(3e-6/3.3e-9) 
   #wait for 6 us, 3 us effective and 3 us to read the UserRegister'''