"""
Created on 08/04/2021
@author sebastian
"""
import broadbean as bb
from constructSequence import Sequencer, PulseBuilder
from qcodes.instrument.parameter import Parameter

try:
    from broadbean.plotting import plotter
except:
    plotter = bb.plotter

import numpy as np
import matplotlib.pyplot as plt

class SequenceParamterClass(Parameter):
    def __init__(self, name, sequence, order):
        super().__init__(name, label='Qcodes parameter class to iterate through sequence elements.',
                         docstring='Qcodes parameter class to iterate through sequence elements.')
        self.sequence = sequence
        self.order = order

    def set_raw(self, goTo):
        for index, item in enumerate(self.order):
            first, second = self.order[index-1], self.order[index]

            if first == 'index':
                first = goTo

            elif second == 'index':
                second = goTo

            self.sequence.setSequencingGoto(first, second)

        return

class RabiPulse(PulseBuilder):
    '''Defined for unique experiment'''
    def __init__(self, chs, sequencer, origin, primaryVector):
        super().__init__(
        sequencer, origin, chs
        )
        self.primaryVector = primaryVector

    def measureElem(self, duration):
        elem, stop = sequencer.buildVectorElement(sequencer.buildSegmentJump, self.origin, duration, 0, self.chs, self.primaryVector, marker1=[0.0])
        return elem, stop

    def unloadElem(self, duration, amp):
        elem, stop = sequencer.buildVectorElement(sequencer.buildSegmentJump, self.origin, duration, amp,
                                                  self.chs, -self.primaryVector)
        return elem, stop

    def varyElem(self, duration):
        bps = []
        stops = []
        for start, ch in zip(self.origin, self.chs):
            bp, stop = sequencer.buildSegmentRamp(start, start, duration, name='vary{}ch'.format(ch))
            bps.append(bp)
            stops.append(stop)
        elem = sequencer.buildElement(bps, self.chs)

        return elem, stop

    def vary_base_sequence(self, base_seq, vary_name, detuning_vector, detunings, durations, lever_arms=[1.0, 1.0]):
        '''fast_param = detuning or time'''
        variables = ['start', 'stop', 'duration']

        if len(lever_arms)!= len(self.chs):
            print('Lever arms must be the same length as channels')

        poss = []
        channels_iters = []
        names = []
        args = []

        # for varied pulse
        for index_ch, ch in enumerate(self.chs):
            poss.extend([1 for _ in variables])
            channels_iters.extend([ch for _ in variables])
            names.extend([vary_name + '{}ch'.format(ch) for _ in variables])
            args.extend([v for v in variables])

        scaled_detunings = (np.array(
            [(self.sequencer._scale_from_vec(detuning_vector)*i).tolist() for i in detunings])*np.array(lever_arms)).T.tolist()

        iters = []
        # for exchange pulse
        for index_ch, ch in enumerate(self.chs):
            start_ch = []
            stop_ch = []
            duration_ch = []
            for t in durations:
                start_ch.extend(scaled_detunings[index_ch])
                stop_ch.extend(scaled_detunings[index_ch])
                duration_ch.extend([t] * len(scaled_detunings[index_ch]))
            iters.append(start_ch)
            iters.append(stop_ch)
            iters.append(duration_ch)

        newseq = bb.repeatAndVarySequence(base_seq, poss, channels_iters, names, args, iters)

        return newseq

bb_ramp = bb.PulseAtoms.ramp

sequencer = Sequencer()
bluep, stop = sequencer.buildSegmentRamp(0, 1, 10e-6, wait=10e-6)
bluep2, stop = sequencer.buildSegmentJump(stop, 3, 10e-6, wait=10e-6)
bluep3, stop = sequencer.buildSegmentJumpAndBack(stop, 2, 10e-6, wait=10e-6)

bp = bluep + bluep2 + bluep3

elem = sequencer.buildElement([bp], [1])

plotter(elem)

chs = [1,2]
origin = [0,0]
primaryVector = np.array([1,1])
rabi = RabiPulse(chs, sequencer, origin, primaryVector)

measElem, _ = rabi.measureElem(10e-6)
unloadElem, _ = rabi.unloadElem(20e-6, 1)
varyElem, _ = rabi.varyElem(20e-6)

seq = bb.Sequence()

seq.addElement(1, unloadElem)
seq.addElement(2, varyElem)
seq.addElement(3, measElem)

seq.setSR(rabi.sequencer.sampleRate)
seq.checkConsistency()

plotter(seq)
plt.show()
#
# detunings = np.linspace(1.0, 2.5, 3)
# durations = np.linspace(10e-9, 500e-9, 3)
# new_seq = rabi.vary_base_sequence(seq, 'vary', primaryVector, detunings, durations)
#
# length = np.size(durations)*np.size(detunings)
# new_seq.addElement(length+1, measElem)
# new_seq.addElement(length+2, unloadElem)
#
# order = [length+1, 'index', length+2]
# seqParam = SequenceParamterClass('Seq', new_seq, order)
#
# for i in range(length):
#     seqParam.set_raw(i+1)
#     plotter(new_seq)
#     plt.show()