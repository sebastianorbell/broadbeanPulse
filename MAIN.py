from qgor import Station
from myPulse import RabiPulse
from broadPulse.broadSequence import Sequencer, SequenceParamterClass
import broadbean as bb
import matplotlib.pyplot as plt

try:
    from broadbean.plotting import plotter
except:
    plotter = bb.plotter


import numpy as np

station = Station()
awg = station.awg5014

sequencer = Sequencer()

chs = [1,2]
origin = [0,0]
primaryVector = np.array([1,1])
rabi = RabiPulse(chs, sequencer, origin, primaryVector)

measElem, _ = rabi.measureElem(10e-6)
unloadElem, _ = rabi.unloadElem(20e-6, 1)
varyElem, _ = rabi.varyElem(20e-6)

seq = bb.Sequence()

#seq.addElement(1, unloadElem)
#seq.addElement(2, measElem)
seq.addElement(1, varyElem)


seq.setSR(rabi.sequencer.sampleRate)
seq.checkConsistency()

plotter(seq)
plt.show()

detunings = np.linspace(1.0, 2.5, 3)
durations = np.linspace(10e-9, 500e-9, 3)
new_seq = rabi.vary_base_sequence(seq, 'vary', primaryVector, detunings, durations)

length = np.size(durations)*np.size(detunings)
new_seq.addElement(length+1, measElem)
new_seq.addElement(length+2, unloadElem)
new_seq.setChannelAmplitude(1,4)
new_seq.setChannelAmplitude(0,4)
new_seq.setChannelOffset(1,0)
new_seq.setChannelOffset(0,0)


awg.save_and_load(*new_seq.outputForAWGFile(), 'joe.awg')

order = [length+1, 'index', length+2]
seqParam = SequenceParamterClass('Seq', new_seq, order)

for i in range(length):
    seqParam.set_raw(i+1)
    plotter(new_seq)
    plt.show()
