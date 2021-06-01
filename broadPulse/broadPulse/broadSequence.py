"""
Created on 31/03/2021
@author sebastian
"""

import broadbean as bb

try:
    from broadbean.plotting import plotter
except:
    plotter = bb.plotter

import numpy as np
import matplotlib.pyplot as plt

bb_ramp = bb.PulseAtoms.ramp

class PulseBuilder():
    def __init__(self, sequencer, origin, chs):
        self.sequencer = sequencer
        self.origin = origin
        self.chs = chs


class Sequencer():
    def __init__(self, sampleRate=1e9, markerLength=1e-7, compression=100):
        self.sampleRate = sampleRate
        self.markerLength = markerLength
        self.compression = compression

    def _scale_from_vec(self, vec):
        vec = np.array(vec)
        amp_of_vec = np.sqrt(np.sum(vec ** 2))
        scaled = vec / amp_of_vec
        return scaled

    def _buildSegment(self, start, stop, duration, **kwargs):
        bp = bb.BluePrint()
        bp.insertSegment(0, bb_ramp, (start, stop), name=kwargs.get('name', ''), dur=duration)
        bp.setSR(self.sampleRate)

        return bp

    def _addMarkers(self, bp, **kwargs):

        if isinstance(kwargs.get('marker1', None), list):
            bp.marker1 = [(t, self.markerLength) for t in kwargs.get('marker1')]

        if isinstance(kwargs.get('marker2', None), list):
            bp.marker2 = [(t, self.markerLength) for t in kwargs.get('marker2')]

        return bp

    def buildSegmentRamp(self, start, stop, duration, **kwargs):
        bp = self._buildSegment(start, stop, duration, **kwargs)

        if isinstance(kwargs.get('wait', None), float):
            bp2 = self._buildSegment(stop, stop, kwargs.get('wait'))
            bp = bp + bp2

        bp = self._addMarkers(bp, **kwargs)

        return bp, stop

    def buildSegmentJumpAndBack(self, start, stop, duration, **kwargs):
        bp = self._buildSegment(stop, stop, duration, **kwargs)

        if isinstance(kwargs.get('wait', None), float):
            bp2 = self._buildSegment(start, start, kwargs.get('wait'))
            bp = bp + bp2

        bp = self._addMarkers(bp, **kwargs)

        return bp, start

    def buildSegmentJump(self, start, stop, duration, **kwargs):
        bp = self._buildSegment(start, stop, duration, **kwargs)

        if isinstance(kwargs.get('wait', None), float):
            bp2 = self._buildSegment(stop, stop, kwargs.get('wait'))
            bp = bp + bp2

        bp = self._addMarkers(bp, **kwargs)

        return bp, stop

    def buildElement(self, bps, chs):
        elem = bb.Element()

        for bp, ch in zip(bps, chs):
            elem.addBluePrint(ch, bp)

        elem.validateDurations()
        return elem

    def buildVectorElement(self, type, start, duration, amp, chs, vec, **kwargs):
        vec_scaled = self._scale_from_vec(vec)
        amp_scaled = vec_scaled*amp

        bps = []
        stops = []
        for s, amp in zip(start, amp_scaled):
            bp, stop = type(s, s+amp, duration, **kwargs)
            bps.append(bp)
            stops.append(stop)

        elem = self.buildElement(bps, chs)

        return elem, stops



