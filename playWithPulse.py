import broadbean as bb
try:
    from broadbean.plotting import plotter
except:
    plotter = bb.plotter

import numpy as np
import matplotlib.pyplot as plt
import json

from pulseClass import DesignExperiment

file = 'pulse_jsons/rabi.json'

with open(file, 'r') as read_file:
    master = json.load(read_file)

experiment = DesignExperiment(gain=103.5, sample_rate=1200000000.0)
base = experiment.build_base_from_json(master)
plotter(base)
plt.show()

detunings = np.linspace(0, 30.0e-3, 5)
durations = np.linspace(10.0e-9, 200e-9, 1)
detuning_vector = [1.0, 1.0]
lever_arms = [0.454,0.854] # meV/V for LB (ch 2 - dac 11), RB (ch 1 - dac 7)

'''
sequence = experiment.add_dc_correction()
'''

sequence = experiment.vary_base_sequence('rabi',detuning_vector, detunings, durations, lever_arms, fast_param='time')

plotter(sequence)
plt.show()

'''sequence.setChannelAmplitude(1, 4.5)  # Call signature: channel, amplitude (peak-to-peak)
sequence.setChannelOffset(1, 0)
sequence.setChannelAmplitude(2, 4.5)  # Call signature: channel, amplitude (peak-to-peak)
sequence.setChannelOffset(2, 0)
package = seq.outputForAWGFile()'''
