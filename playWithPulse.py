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

experiment = DesignExperiment(gain=103.5, sample_rate=1200000000.0, marker_length=1e-9)
base = experiment.build_base_from_json(master)
detunings = np.linspace(0, 3.0, 30) # meV  np.array([2.0])#
durations = np.linspace(5.0e-9, 200e-9, 30) # np.array([50e-9])
detuning_vector = [-0.006479, 0.014254]
lever_arms = [0.454,0.854] # meV/V for LB (ch 2 - dac 11), RB (ch 1 - dac 7)
#seq = experiment.vary_base_sequence('rabi',detuning_vector, detunings, durations, lever_arms, fast_param='time')
seq = experiment.subSequencer('rabi',detuning_vector, detunings, durations, lever_arms, fast_param='time')
plotter(base)
plt.show()
seq.setChannelAmplitude(1, 5)  # Call signature: channel, amplitude (peak-to-peak)
seq.setChannelOffset(1, 0)
seq.setChannelAmplitude(2, 5)  # Call signature: channel, amplitude (peak-to-peak)
seq.setChannelOffset(2, 0)
#package = seq.outputForAWGFile()

fs = seq.forge()
# send it to the instrument
seqname = 'mytestseq'
amplitudes = [ch.awg_amplitude() for ch in awg.channels][:2]
channel_mapping = {'trigger_channel': 2, 'signal_channel': 1}
seqx_file = AWG5208.makeSEQXFileFromForgedSequence(fs,
                                                   amplitudes=amplitudes,
                                                   seqname=seqname,
                                                   channel_mapping=channel_mapping)

# load it and assign its tracks to the channels
filename = 'mainplussub.seqx'

awg.clearSequenceList()
awg.clearWaveformList()
awg.sendSEQXFile(seqx_file, filename=filename)
awg.loadSEQXFile(filename)
awg.ch1.setSequenceTrack(seqname, 1)
awg.ch2.setSequenceTrack(seqname, 2)

'''
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
lever_arms = [0.454,0.854] # meV/V for LB (ch 2 - dac 11), RB (ch 1 - dac 7)
'''
'''
sequence = experiment.add_dc_correction()
'''

#sequence = experiment.subSequencer('exchange',detuning_vector, detunings, durations, lever_arms, fast_param='time')

'''plotter(sequence)
plt.show()'''

'''sequence.setChannelAmplitude(1, 4.5)  # Call signature: channel, amplitude (peak-to-peak)
sequence.setChannelOffset(1, 0)
sequence.setChannelAmplitude(2, 4.5)  # Call signature: channel, amplitude (peak-to-peak)
sequence.setChannelOffset(2, 0)
package = seq.outputForAWGFile()'''
