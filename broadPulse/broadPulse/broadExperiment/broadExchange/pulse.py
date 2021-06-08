"""
Created on 08/04/2021
@author sebastian
"""
import broadbean as bb
from broadPulse.broadPulse.broadSequence import Sequencer, PulseBuilder
from qcodes.instrument.parameter import Parameter

try:
    from broadbean.plotting import plotter
except:
    plotter = bb.plotter

import numpy as np


class ExchangePulse(PulseBuilder):
    '''Defined for unique experiment'''
    def __init__(self, chs, sequencer, origin, config):
        super().__init__(
        sequencer, origin, chs
        )

        self.config = config
        self.readConfig()

    def readConfig(self):
        self.primaryVector = np.array(self.config.get('primaryvector'))

        self.unloadElem, self.unloadStop = self._createUnloadElem(
            self.config.get('unload').get('duration'),
            self.config.get('unload').get('amp'),
            self.origin)

        self.piHalfElem, self.piHalfStop = self._createPiHalfElem(
            self.config.get('pihalf').get('duration'),
            self.config.get('pihalf').get('amp'),
            self.unloadStop)

        self.reversePiHalfElem, self.reversePiHalfStop = self._createReversePiHalfElem(
            self.config.get('pihalf').get('duration'),
            self.config.get('pihalf').get('amp'),
            self.piHalfStop)

        self.measureElem, self.measureStop = self._createMeasureElem(
            self.config.get('measure').get('duration'),
            self.config.get('measure').get('amp'))

    def _createUnloadElem(self, duration, amp, start):
        elem, stop = self.sequencer.buildVectorElement(self.sequencer.buildSegmentJumpAndBack, start, duration, amp,
                                                  self.chs, -self.primaryVector)
        return elem, stop

    def _createPiHalfElem(self, duration, amp, start):
        elem, stop = self.sequencer.buildVectorElement(self.sequencer.buildSegmentRamp, start, duration, amp,
                                                  self.chs, self.primaryVector)
        return elem, stop

    def _createExchangeElem(self, duration, amp, start):
        elem, stop = self.sequencer.buildVectorElement(self.sequencer.buildSegmentJumpAndBack, start, duration, amp,
                                                  self.chs, self.primaryVector)
        return elem, stop

    def _createReversePiHalfElem(self, duration, amp, start):
        elem, stop = self.sequencer.buildVectorElement(self.sequencer.buildSegmentRamp, start, duration, amp,
                                                  self.chs, -self.primaryVector)
        return elem, stop

    def _createMeasureElem(self, duration, amp):
        elem, stop = self.sequencer.buildVectorElement(self.sequencer.buildSegmentJump, self.origin, duration, amp, self.chs, self.primaryVector, marker1=[0.0])
        return elem, stop

    def createExchangePulse(self, amplitude, duration):
        self.exchangeElem, self.exchangeStop = self._createExchangeElem(
            duration,
            amplitude,
            self.piHalfStop)


        seq = bb.Sequence()

        seq.addElement(1, self.unloadElem)
        seq.addElement(2, self.piHalfElem)
        seq.addElement(3, self.exchangeElem)
        seq.addElement(4, self.reversePiHalfElem)
        seq.addElement(5, self.measureElem)

        seq.setSR(self.sequencer.sampleRate)
        seq.checkConsistency()

        return seq

class SequenceParamterClass(Parameter):
    def __init__(self, name, pulse):
        super().__init__(name, label='Qcodes parameter class to iterate through sequence elements.',
                         docstring='Qcodes parameter class to iterate through sequence elements.')

        self.pulse = pulse

    def set_raw(self, x1, x2):
        seq = self.pulse(x1, x2)
        package = seq.outputForAWGFile()
        self.awg.save_and_load(*package[:])
        return


if __name__=='__main__':
    sequencer = Sequencer()

    chs = [1,2]
    origin = [0,0]

    config = {
        'primaryvector':[1,1],
        'unload':
            {'duration':1e-8,
                'amp':1e-1},
        'pihalf':
            {'duration': 1e-8,
             'amp': 5e-1},
        'measure':
            {'duration': 5e-8},
    }

    rabi = ExchangePulse(chs, sequencer, origin, config)

    seq = rabi.createExchangePulse(3e-2,1e-8)

    plotter(seq)

    detunings = np.linspace(1.0, 2.5, 3)
    durations = np.linspace(10e-9, 500e-9, 3)

    for t in durations:
        for e in detunings:
            seq = rabi.createExchangePulse(e, t)
            plotter(seq)