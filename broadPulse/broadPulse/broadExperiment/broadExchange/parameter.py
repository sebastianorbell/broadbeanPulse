"""
Created on 31/05/2021
@author sebastian
"""

from qcodes import Parameter

from inspect import signature
from pathlib import Path
import json
import matplotlib.pyplot as plt

import broadbean as bb
from broadbean import Sequence
from broadPulse.broadPulse.broadSequence import Sequencer
from broadPulse.broadPulse.broadExperiment.broadExchange.pulse import ExchangePulse

try:
    from broadbean.plotting import plotter
except:
    plotter = bb.plotter

class ExchangeParameter(Sequence):

    def __init__(self, options_path='pulseClass.json', **kwargs):

        self.configDir = Path(__file__).parent / 'configs'
        self.options_path = self.configDir / options_path

        print(self.options_path)

        with open(self.options_path) as f:
            self.options = json.load(f)

        self.options = {**self.options}

        print(self.options)

        self.name = 'jeanSequence.seqx'

        # bunch of methods to quickly change single parameter only

        self.primaryVector = Parameter(name='primaryVector',
                                            set_cmd=lambda x: self.create_waveform(
                                                primaryVector=x
                                            ))

        self.leverArm = Parameter(name='leverArm',
                                       set_cmd=lambda x: self.create_waveform(
                                           leverArm=x
                                       ))

        self.attenuation = Parameter(name='attenuation',
                                       set_cmd=lambda x: self.create_waveform(
                                           attenuation=x
                                       ))

        self.unloadDuration = Parameter(name='unloadDuration',
                                   set_cmd=lambda x: self.create_waveform(
                                       unloadDuration=x
                                   ))

        self.unloadAmp = Parameter(name='unloadAmp',
                                     set_cmd=lambda x: self.create_waveform(
                                         unloadAmp=x
                                     ))

        self.piHalfDuration = Parameter(name='piHalfDuration',
                                   set_cmd=lambda x: self.create_waveform(
                                       piHalfDuration=x
                                   ))

        self.piHalfAmp = Parameter(name='piHalfAmp',
                                     set_cmd=lambda x: self.create_waveform(
                                         piHalfAmp=x
                                     ))

        self.exchangeDuration = Parameter(name='exchangeDuration',
                                   set_cmd=lambda x: self.create_waveform(
                                       exchangeDuration=x
                                   ))

        self.exchangeAmp = Parameter(name='exchangeAmp',
                                     set_cmd=lambda x: self.create_waveform(
                                         exchangeAmp=x
                                     ))

        self.measureDuration = Parameter(name='measureDuration',
                                   set_cmd=lambda x: self.create_waveform(
                                       measureDuration=x
                                   ))

        self.measureAmp = Parameter(name='measureAmp',
                                     set_cmd=lambda x: self.create_waveform(
                                         measureAmp=x
                                     ))

        self.chs = Parameter(name='chs',
                                     set_cmd=lambda x: self.create_waveform(
                                         chs=x
                                     ))

        self.origin = Parameter(name='origin',
                                     set_cmd=lambda x: self.create_waveform(
                                         origin=x
                                     ))

        self.plot = Parameter(name="plot",
                              initial_value=False,
                              set_cmd=lambda *args: None)

        self.upload = Parameter(
            name='upload',
            set_cmd=lambda *args: None
        )

        self.create_waveform(
            **self.options
        )

        super().__init__()

    def show(self):
        plotter(self)
        plt.show()

    def create_waveform(self,
                        primaryVector = None,
                        leverArm = None,
                        attenuation = None,
                        unloadDuration = None,
                        unloadAmp=None,
                        piHalfDuration=None,
                        piHalfAmp=None,
                        exchangeDuration=None,
                        exchangeAmp=None,
                        measureDuration=None,
                        measureAmp=None,
                        chs=None,
                        origin=None,
                        plot=None,
                        upload=None
                        ):

        # get a list of all the functions arguments
        arguments = list(signature(self.create_waveform).parameters)
        # a dict to contain the cached values of all the keyword arguments which are not passed so default to None
        for argument in arguments:
            # getting the parameter from the class
            parameter = self.__getattribute__(argument)
            # getting the value of the argument in local memory. If the argument is not passed this will default to None
            local_value = locals().get(argument)
            if local_value is not None:
                # if the value is passed update the cached value of parameter
                parameter.cache.set(local_value)

        sequencer = Sequencer()

        unloadAmp, piHalfAmp, measureAmp, exchangeAmp = [i*self.leverArm()*self.attenuation() for i in [self.unloadAmp(), self.piHalfAmp(), self.measureAmp(), self.exchangeAmp()]]

        config = {
            'primaryvector': self.primaryVector(),
            'unload':
                {'duration': self.unloadDuration(),
                 'amp': unloadAmp},
            'pihalf':
                {'duration': self.piHalfDuration(),
                 'amp': piHalfAmp},
            'measure':
                {'duration': self.measureDuration(),
                 'amp': measureAmp}
        }

        rabi = ExchangePulse(self.chs(), sequencer, self.origin(), config)

        seq = rabi.createExchangePulse(exchangeAmp, self.exchangeDuration())

        if self.plot():
            plotter(seq)
            plt.show()

        if self.upload():
            package = seq.outputForAWGFile()
            self.awg.save_and_load(*package[:])

        return