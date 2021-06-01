"""
Created on 31/05/2021
@author sebastian
"""

from qcodes import Parameter

from inspect import signature
from pathlib import Path
import json
import matplotlib.pyplot as plt

from broadbean import Sequence
from broadPulse.broadSequence import Sequencer
from broadPulse.broadExchange import ExchangePulse

try:
    from broadbean.plotting import plotter
except:
    plotter = bb.plotter

class PulseClass(Sequence):

    def __init__(self, additional_options=None, options_path=Path(__file__).parent / 'pulseClass.json', **kwargs):

        self.options_path = options_path

        with open(options_path) as f:
            self.options = json.load(f)

        self.options = {**self.options, **additional_options}

        self.name = 'jeanSequence.seqx'

        # bunch of methods to quickly change single parameter only

        self.primaryVector = Parameter(name='primaryVector',
                                            set_cmd=lambda x: self.create_waveform(
                                                primaryVector=x
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

        self.upload = Parameter(
            name='upload',
            set_cmd=lambda *args: None
        )

        self.plot = Parameter(name="plot",
                              initial_value=False,
                              set_cmd=lambda *args: None)

        self.create_waveform(
            **self.options,
            upload = False
        )

        super().__init__()

    def show(self):
        plotter(self)
        plt.show()

    def create_waveform(self,
                        primaryVector = None,
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


        config = {
            'primaryvector': self.primaryVector(),
            'unload':
                {'duration': self.unloadDuration(),
                 'amp': self.unloadAmp()},
            'pihalf':
                {'duration': self.piHalfDuration(),
                 'amp': self.piHalfAmp()},
            'measure':
                {'duration': self.measureDuration(),
                 'amp': self.measureAmp()}
        }

        rabi = ExchangePulse(self.chs(), sequencer, self.origin(), config)

        seq = rabi.createExchangePulse(self.exchangeAmp(), self.exchangeDuration())

        if self.plot():
            plotter(self, apply_filters=True, apply_delays=True)
            plt.show()

        if self.upload():
            package = seq.outputForAWGFile()
            self.awg.save_and_load(*package[:])

        return