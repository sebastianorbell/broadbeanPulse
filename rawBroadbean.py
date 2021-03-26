"""
Created on 25/03/2021
@author sebastian
"""

import broadbean as bb
try:
    from broadbean.plotting import plotter
except:
    plotter = bb.plotter

sine = bb.PulseAtoms.sine
ramp = bb.PulseAtoms.ramp

SR = 1e9
t1 = 200e-6  # wait
t2 = 20e-9  # perturb the system
t3 = 250e-6  # read out

compression = 100  # this number has to be chosen with some care

bp1 = bb.BluePrint()
bp1.insertSegment(0, ramp, (0, 0), dur=t1/compression)
bp1.setSR(SR)
elem1 = bb.Element()
elem1.addBluePrint(1, bp1)
#
bp2 = bb.BluePrint()
bp2.insertSegment(0, ramp, (1, 1), dur=t2, name='perturbation')
bp2.setSR(SR)
elem2 = bb.Element()
elem2.addBluePrint(1, bp2)
#
bp3 = bb.BluePrint()
bp3.insertSegment(0, ramp, (0, 0), dur=t3/compression, name='marker')
bp3.setSegmentMarker('marker', (0.0, 1e-6), 1)
bp3.setSR(SR)
elem3 = bb.Element()
elem3.addBluePrint(1, bp3)

seq = bb.Sequence()
seq.addElement(1, elem1)
seq.setSequencingNumberOfRepetitions(1, compression)
seq.addElement(2, elem2)
seq.addElement(3, elem3)
seq.setSequencingNumberOfRepetitions(3, compression)
seq.setSR(SR)

# Now make the variation
seq2 = seq.copy()
seq2.element(2).changeArg(1, 'perturbation', 'start', 0.75)
seq2.element(2).changeArg(1, 'perturbation', 'stop', 0.75)
#
seq3 = seq.copy()
seq3.element(2).changeArg(1, 'perturbation', 'start', 0.5)
seq3.element(2).changeArg(1, 'perturbation', 'stop', 0.5)
#
fullseq = seq + seq2 + seq3
plotter(fullseq)
#-------------------------------------------------------
mainseq = bb.Sequence()
mainseq.setSR(SR)

mainseq.addSubSequence(1, seq)
mainseq.addSubSequence(2, seq2)
mainseq.addSubSequence(3, seq3)

mainseq.setSequencingNumberOfRepetitions(1, 25)
mainseq.setSequencingNumberOfRepetitions(2, 25)
mainseq.setSequencingNumberOfRepetitions(3, 25)

plotter(mainseq)
