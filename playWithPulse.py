import broadbean as bb
plotter = bb.plotter
import numpy as np
import matplotlib.pyplot as plt
import json

from broadbeanPulse.pulseClass import  DesignExperiment

file = 'pulse_jsons/exchange.json'

with open(file, 'r') as read_file:
    master = json.load(read_file)

experiment = DesignExperiment(gain=1.0)
base = experiment.build_base_from_json(master)
plotter(base)
plt.show()

detunings = np.linspace(0, 30.0e-3, 5)
durations = np.linspace(10.0e-9, 200e-9, 1)
detuning_vector = [1.0, 1.0]

'''
sequence = experiment.add_dc_correction()
'''

sequence = experiment.vary_base_sequence('exchange',detuning_vector, detunings, durations, fast_param='time')

plotter(sequence)
plt.show()