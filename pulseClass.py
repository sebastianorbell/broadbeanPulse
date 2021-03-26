import broadbean as bb

try:
    from broadbean.plotting import plotter
except:
    plotter = bb.plotter
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (8, 3)
mpl.rcParams['figure.subplot.bottom'] = 0.15

# The pulsebuilding module comes with a (small) collection of functions appropriate for being segments.
bb_ramp = bb.PulseAtoms.ramp  # args: start, stop
bb_sine = bb.PulseAtoms.sine  # args: freq, ampl, off, phase


# make a blueprint

# The blueprint takes no arguments
class Sequencer():
    def __init__(self, sample_rate=1e9, marker_length=1e-7, compression=100):
        self.sample_rate = sample_rate
        self.compression = compression
        self.bp_dict = {str(i): {} for i in range(1, 6)}
        self.pulse_dict = {}
        self.marker_length = marker_length
        self.area_dict = {}
        self.compression_dict = {}

    def _scale_from_vec(self, vec, amps):
        vec = np.array(vec)
        amps = np.array(amps)
        amp_of_vec = np.sqrt(np.sum(vec ** 2))
        scaled = vec * amps / amp_of_vec
        return scaled

    def add_pulses(self, dicts):
        for dict in dicts:
            self.pulse_dict[dict['name']] = dict

    def update_pulse(self, update, pulse):
        for key in update:
            self.pulse_dict[pulse][key] = update[key]

    def build_seq(self, order):
        self.order = order
        print('order', order)
        update = {}
        area = np.zeros([2])
        elem_list = []
        for index_order, pulse in enumerate(self.order):
            self.update_pulse(update, pulse)

            if self.pulse_dict[pulse].get('type') == 'ramp':
                update, bp = self.build_ramp_pulse_bp(self.pulse_dict[pulse])
            elif self.pulse_dict[pulse].get('type') == 'vary_ramp':
                update, bp = self.build_named_ramp_pulse_bp(self.pulse_dict[pulse])
            elif self.pulse_dict[pulse].get('type') == 'jump':
                update, bp = self.build_jump_pulse_bp(self.pulse_dict[pulse])

            area += self.area_dict[pulse]
            elem_list.append(self.build_elem_from_bp(bp))

        # bps = update['bps']
        seq = self.build_seq_from_elem(elem_list)
        # seq = self.build_seq_from_bp(bps)
        total_time = update['start_time']
        return seq, area, total_time

    def build_named_ramp_pulse_bp(self, pulse_dict):
        sample_rate = pulse_dict.get('sample_rate', self.sample_rate)

        ramp_range = pulse_dict['vec_amps'][0]
        vec = pulse_dict['vecs'][0]
        dur = pulse_dict['durations'][0]
        first_index = pulse_dict.get('first_index', 0)
        channels = pulse_dict['channels']
        bps = [bb.BluePrint() for _ in channels]
        start = np.array(pulse_dict.get('start', [0 for _ in channels])).astype(float)
        start_time = pulse_dict.get('start_time', 0.0)
        area = np.zeros([len(channels)])

        self.compression_dict[pulse_dict.get('name')] = False

        for index_ch, ch in enumerate(channels):
            bps[index_ch].setSR(sample_rate)

            scaled = self._scale_from_vec(vec, ramp_range)
            area[index_ch] += 0.5 * (start[index_ch] * 2 + scaled[index_ch]) * dur

            bps[index_ch].insertSegment(first_index, bb_ramp,
                                        (start[index_ch], start[index_ch] + scaled[index_ch]), dur=dur,
                                        name=pulse_dict.get('vary_name')[index_ch])
            start[index_ch] += scaled[index_ch]

            if pulse_dict.get('marker1', []):
                for index, t in enumerate(pulse_dict.get('marker1', [])):
                    bps[index_ch].setSegmentMarker(pulse_dict.get('vary_name')[index_ch], (t, self.marker_length),
                                                   index + 1)

            if pulse_dict.get('marker2', []):
                for index, t in enumerate(pulse_dict.get('marker1', [])):
                    bps[index_ch].setSegmentMarker(pulse_dict.get('vary_name')[index_ch], (t, self.marker_length),
                                                   index + 1)

        self.bp_dict[str(ch)][pulse_dict['name']] = bps[index_ch]
        end = start
        final_index = first_index + 1
        final_time = dur + start_time
        update_for_next = {'start': end, 'first_index': final_index, 'start_time': final_time}

        self.area_dict[pulse_dict.get('name')] = area

        return update_for_next, bps

    def build_ramp_pulse_bp(self, pulse_dict):
        sample_rate = pulse_dict.get('sample_rate', self.sample_rate)

        ramp_ranges = pulse_dict['vec_amps']
        vecs = pulse_dict['vecs']
        durs = pulse_dict['durations']
        first_index = pulse_dict.get('first_index', 0)
        channels = pulse_dict['channels']
        bps = [bb.BluePrint() for _ in channels]
        start = np.array(pulse_dict.get('start', [0 for _ in channels])).astype(float)
        start_time = pulse_dict.get('start_time', 0.0)
        area = np.zeros([len(channels)])
        for index_ch, ch in enumerate(channels):
            bps[index_ch].setSR(sample_rate)
            for index_ramp, ramp_range in enumerate(ramp_ranges):

                scaled = self._scale_from_vec(vecs[index_ramp], ramp_range)
                area[index_ch] += 0.5 * (start[index_ch] * 2 + scaled[index_ch]) * durs[index_ramp]

                if ramp_range == 0 and ((durs[index_ramp] / self.compression) / (1 / self.sample_rate)) > 1:
                    self.compression_dict[pulse_dict.get('name')] = True
                    bps[index_ch].insertSegment(first_index + index_ramp, bb_ramp,
                                                (start[index_ch], start[index_ch] + scaled[index_ch]),
                                                dur=durs[index_ramp] / self.compression,
                                                name=pulse_dict.get('name') + str(index_ch) + str(index_ramp) + 'pulse')

                else:
                    self.compression_dict[pulse_dict.get('name')] = False
                    bps[index_ch].insertSegment(first_index + index_ramp, bb_ramp,
                                                (start[index_ch], start[index_ch] + scaled[index_ch]),
                                                dur=durs[index_ramp],
                                                name=pulse_dict.get('name') + str(index_ch) + str(index_ramp) + 'pulse')

                start[index_ch] += scaled[index_ch]

            if pulse_dict.get('marker1', []):
                for index, t in enumerate(pulse_dict.get('marker1', [])):
                    print('Marker added', pulse_dict.get('name') + str(index_ch) + str(0) + 'pulse')
                    bps[index_ch].setSegmentMarker(pulse_dict.get('name') + str(index_ch) + str(0) + 'pulse',
                                                   (t, self.marker_length), index + 1)

            if pulse_dict.get('marker2', []):
                for index, t in enumerate(pulse_dict.get('marker2', [])):
                    bps[index_ch].setSegmentMarker(pulse_dict.get('name') + str(index_ch) + str(0) + 'pulse',
                                                   (t, self.marker_length), index + 1)

        self.bp_dict[str(ch)][pulse_dict['name']] = bps[index_ch]
        end = start
        final_index = first_index + len(ramp_ranges)
        final_time = np.sum(durs) + start_time
        update_for_next = {'start': end, 'first_index': final_index, 'start_time': final_time}

        self.area_dict[pulse_dict.get('name')] = area

        return update_for_next, bps

    def build_jump_pulse_bp(self, pulse_dict):
        sample_rate = pulse_dict.get('sample_rate', self.sample_rate)

        ramp_ranges = pulse_dict['vec_amps']
        vecs = pulse_dict['vecs']
        durs = pulse_dict['durations']
        first_index = pulse_dict.get('first_index', 0)
        channels = pulse_dict['channels']
        bps = [bb.BluePrint() for _ in channels]
        start = np.array(pulse_dict.get('start', [0 for _ in channels])).astype(float)
        start_time = pulse_dict.get('start_time', 0.0)
        area = np.zeros([len(channels)])

        self.compression_dict[pulse_dict.get('name')] = False

        for index_ch, ch in enumerate(channels):
            bps[index_ch].setSR(sample_rate)
            for index_ramp, ramp_range in enumerate(ramp_ranges):
                scaled = self._scale_from_vec(vecs[index_ramp], ramp_range)
                area[index_ch] += (scaled[index_ch]) * durs[index_ramp]

                bps[index_ch].insertSegment(first_index + index_ramp, bb_ramp, (scaled[index_ch], scaled[index_ch]),
                                            dur=durs[index_ramp])

            if pulse_dict.get('marker1', []):
                bps[index_ch].marker1 = [(start_time + t, self.marker_length) for t in pulse_dict.get('marker1', [])]
            if pulse_dict.get('marker2', []):
                bps[index_ch].marker2 = [(start_time + t, self.marker_length) for t in pulse_dict.get('marker2', [])]

        self.bp_dict[str(ch)][pulse_dict['name']] = bps[index_ch]
        end = start
        final_index = first_index + len(ramp_ranges)
        final_time = np.sum(durs) + start_time
        update_for_next = {'start': end, 'first_index': final_index, 'start_time': final_time}

        self.area_dict[pulse_dict.get('name')] = area

        return update_for_next, bps

    def build_elem_from_bp(self, bps):
        elem = bb.Element()
        for index_ch, ch in enumerate(bps):
            elem.addBluePrint(index_ch + 1, ch)
        elem.validateDurations()
        return elem

    def build_seq_from_bp(self, bps):
        elem = self.build_elem_from_bp(bps)
        seq = self.build_seq_from_elem([elem])
        return seq

    def build_seq_from_elem(self, elems):
        seq = bb.Sequence()
        for index, elem in enumerate(elems):
            seq.addElement(index + 1, elem)
            if self.compression_dict.get(self.order[index], False) == True:
                seq.setSequencingNumberOfRepetitions(index + 1, self.compression)
        seq.setSR(self.sample_rate)
        seq.checkConsistency()
        return seq


class DesignExperiment(Sequencer):
    def __init__(self, gain=1.0 / 6e-3, sample_rate=1e9, marker_rate=1e-7, compression=100):
        super().__init__()
        self.gain = gain
        self.sequencer = Sequencer(sample_rate, marker_rate)
        self.wait_vec = [1.0, 1.0]
        self.wait_ramp = 0.0
        self.wait_time = 1e-6
        self.compression = compression

    def _calc_vec(self, initial, final):
        vec = np.array(final) - np.array(initial)
        mag = np.sqrt(np.sum(vec * vec))
        return vec, mag

    def build_base_from_json(self, dict):
        self.build_dict = dict
        self.order_dict = self.build_dict.get('order')
        self.channels = self.build_dict.get('channels')
        self.origin = self.build_dict.get('origin')
        self.dc_correction = self.build_dict.get('dc_correction', False)
        self.vary_name = []
        self.route = []

        '''
        --------------------------
        Exchange pulse  dictionary
        --------------------------
        {
        "channels":[1,2],
        "dc_correction": true,
          "origin":[3.535, 2.655],
          "order":{
              "unload":{
                "loc":[3.545, 2.65],
                "method":"ramp",
                "time":1e-6,
                "wait":1e-6
              },
              "load":{
                "loc":[3.53, 2.65],
                "method":"ramp",
                "time":1e-6,
                "wait":1e-6
              },
              "pi_half_1":{
                "loc":[3.535, 2.655],
                "method":"jump",
                "time":16e-9
              },
                "exchange":{
                "method":"vary_ramp",
                "kwargs":{"vary_name": "exchange"}
              },
                "pi_half_2":{
                "loc":[3.535, 2.655],
                "method":"jump",
                "time":16e-9
              },
              "measure":{
                    "loc":[3.535, 2.655],
                    "method":"ramp",
                    "time":1e-6,
                    "wait":10e-6,
                    "marker1": [1e-6]
                  }
            },
                "kwargs":{
                }
          }
        '''

        pulse_dicts = []
        order = []
        self.elem_dict = {}
        new_origin = self.origin
        self.route.append(new_origin)
        for key in self.order_dict:

            kwargs = self.order_dict[key].get('kwargs')

            if self.order_dict[key].get('method') == 'ramp':
                order.append(key)
                self.elem_dict[key] = len(order)
                current = self.order_dict[key].get('loc')
                vec, mag = self._calc_vec(new_origin, current)
                new_origin = current
                self.route.append(new_origin)

                vectors = [vec]
                magnitudes = [mag * self.gain]
                durations = [self.order_dict[key].get('time')]

                pulse = {'name': key,
                         'channels': self.channels,
                         'vecs': vectors,
                         'vec_amps': magnitudes,
                         'durations': durations,
                         'type': self.order_dict[key].get('method')}

                if self.order_dict[key].get('marker1', None) != None:
                    pulse['marker1'] = self.order_dict[key].get('marker1')

                pulse_dicts.append(pulse)

                if isinstance(self.order_dict[key].get('wait', None), (float, int)):
                    order.append(key + '_wait')
                    self.elem_dict[key + '_wait'] = len(order)
                    vectors = [self.wait_vec]
                    magnitudes = [self.wait_ramp * self.gain]
                    durations = [self.wait_time]

                    pulse = {'name': key + '_wait',
                             'channels': self.channels,
                             'vecs': vectors,
                             'vec_amps': magnitudes,
                             'durations': durations,
                             'type': self.order_dict[key].get('method')}

                    pulse_dicts.append(pulse)


            elif self.order_dict[key].get('method') == 'vary_ramp':
                order.append(key)
                self.elem_dict[key] = len(order)
                self.vary_name.append(kwargs['vary_name'])
                vectors = [self.wait_vec]
                magnitudes = [self.wait_ramp * self.gain]
                durations = [self.wait_time]

                pulse = {'name': key,
                         'channels': self.channels,
                         'vecs': vectors,
                         'vec_amps': magnitudes,
                         'durations': durations,
                         'type': self.order_dict[key].get('method'),
                         'vary_name': [kwargs['vary_name'] + '{}ch'.format(i) for i in self.channels]}

                if self.order_dict[key].get('marker1', None) != None:
                    pulse['marker1'] = self.order_dict[key].get('marker1')

                pulse_dicts.append(pulse)

            elif self.order_dict[key].get('method') == 'jump':
                order.append(key)
                self.elem_dict[key] = len(order)
                current = self.order_dict[key].get('loc')
                vec, mag = self._calc_vec(new_origin, current)

                vectors = [vec]
                magnitudes = [mag * self.gain]
                durations = [self.order_dict[key].get('time')]

                pulse = {'name': key,
                         'channels': self.channels,
                         'vecs': vectors,
                         'vec_amps': magnitudes,
                         'durations': durations,
                         'type': self.order_dict[key].get('method')}

                if self.order_dict[key].get('marker1', None) != None:
                    pulse['marker1'] = self.order_dict[key].get('marker1')

                pulse_dicts.append(pulse)

        if self.dc_correction == True:
            order.append('dc_correction')
            self.elem_dict['dc_correction'] = len(order)
            pulse_dict_dc_correction = {'name': 'dc_correction',
                                        'channels': self.channels,
                                        'vecs': [self.wait_vec],
                                        'vec_amps': [self.wait_ramp],
                                        'durations': [self.wait_time],
                                        'type': 'vary_ramp',
                                        'vary_name': ['offset{}ch'.format(i) for i in self.channels]}
            pulse_dicts.append(pulse_dict_dc_correction)

        self.sequencer.add_pulses(pulse_dicts)

        print(order)

        self.base_seq, self.area, self.seq_total_time = self.sequencer.build_seq(order)

        return self.base_seq

    def add_dc_correction(self):
        offset_time = 0.33 * self.seq_total_time
        offset_mag = -self.area / offset_time

        variables = ['start', 'stop', 'duration']

        poss = []
        channels_iters = []
        names = []
        args = []

        # for DC offset pulse
        for index_ch, ch in enumerate(self.channels):
            poss.extend([1 for _ in variables])
            channels_iters.extend([ch for _ in variables])
            names.extend(['offset' + '{}ch'.format(ch) for _ in variables])
            args.extend([v for v in variables])

        iters = []
        for index_ch, ch in enumerate(self.channels):
            offset = offset_mag[index_ch]
            start_ch = [offset]
            stop_ch = [offset]
            duration_ch = [offset_time]

            iters.append(start_ch)
            iters.append(stop_ch)
            iters.append(duration_ch)

        newseq = bb.repeatAndVarySequence(self.base_seq, poss, channels_iters, names, args, iters)
        # plotter(newseq)
        # plt.show()
        return newseq

    def vary_base_sequence(self, vary_name, detuning_vector, detunings, durations, lever_arms, fast_param='detuning'):
        '''fast_param = detuning or time'''
        variables = ['start', 'stop', 'duration']

        if len(lever_arms) != len(self.channels):
            print('Lever arms must be the same length as channels')

        poss = []
        channels_iters = []
        names = []
        args = []
        if vary_name not in self.vary_name:
            print('No blueprint with that name')

        # for varied pulse
        for index_ch, ch in enumerate(self.channels):
            poss.extend([1 for _ in variables])
            channels_iters.extend([ch for _ in variables])
            names.extend([vary_name + '{}ch'.format(ch) for _ in variables])
            args.extend([v for v in variables])

        # for DC offset pulse
        for index_ch, ch in enumerate(self.channels):
            poss.extend([1 for _ in variables])
            channels_iters.extend([ch for _ in variables])
            names.extend(['offset' + '{}ch'.format(ch) for _ in variables])
            args.extend([v for v in variables])

        scaled_detunings = (np.array(
            [self.sequencer._scale_from_vec(detuning_vector, i).tolist() for i in detunings]) * np.array(
            lever_arms)).T.tolist()

        offset_time = 0.33 * self.seq_total_time
        offset_mag = -self.area / offset_time

        if fast_param == 'detuning':
            iters = []

            # for exchange pulse
            for index_ch, ch in enumerate(self.channels):
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

            # for dc offset
            for index_ch, ch in enumerate(self.channels):
                start_ch = []
                stop_ch = []
                duration_ch = []
                for t in durations:
                    offset = (offset_mag[index_ch] + (np.array(scaled_detunings[index_ch]) * t / offset_time)).tolist()
                    start_ch.extend(offset)
                    stop_ch.extend(offset)
                    duration_ch.extend([offset_time] * len(scaled_detunings[index_ch]))
                iters.append(start_ch)
                iters.append(stop_ch)
                iters.append(duration_ch)

        elif fast_param == 'time':
            iters = []
            # for exchange pulse
            for index_ch, ch in enumerate(self.channels):
                start_ch = []
                stop_ch = []
                duration_ch = []
                for d in scaled_detunings[index_ch]:
                    start_ch.extend([d] * len(durations))
                    stop_ch.extend([d] * len(durations))
                    duration_ch.extend(durations.tolist())
                iters.append(start_ch)
                iters.append(stop_ch)
                iters.append(duration_ch)

            # for dc offset
            for index_ch, ch in enumerate(self.channels):
                start_ch = []
                stop_ch = []
                duration_ch = []
                for d in scaled_detunings[index_ch]:
                    offset = offset_mag[index_ch] + (d * durations / offset_time)
                    start_ch.extend(offset)
                    stop_ch.extend(offset)
                    duration_ch.extend([offset_time] * len(durations))
                iters.append(start_ch)
                iters.append(stop_ch)
                iters.append(duration_ch)

        newseq = bb.repeatAndVarySequence(self.base_seq, poss, channels_iters, names, args, iters)
        # plotter(newseq)
        # plt.show()
        return newseq

    def subSequencer(self, vary_name, detuning_vector, detunings, durations, lever_arms,
                     fast_param='detuning'):
        '''fast_param = detuning or time'''

        if len(lever_arms) != len(self.channels):
            print('Lever arms must be the same length as channels')

        if vary_name not in self.vary_name:
            print('No blueprint with that name')

        self.scaled_detunings = (np.array(
            [self.sequencer._scale_from_vec(detuning_vector, i).tolist() for i in detunings]) * np.array(
            lever_arms)).T

        offset_time = 0.33 * self.seq_total_time
        offset_mag = -self.area / offset_time

        mainseq = bb.Sequence()
        mainseq.setSR(self.sample_rate)

        indexer = 0

        if fast_param == 'detuning':
            for eps in self.scaled_detunings:
                for t in durations:
                    indexer += 1
                    seq_copy = self.base_seq.copy()
                    for index_ch, ch in enumerate(self.channels):
                        seq_copy.element(self.elem_dict[vary_name]).changeArg(ch, vary_name + '{}ch'.format(ch),
                                                                              'start', eps[index_ch])
                        seq_copy.element(self.elem_dict[vary_name]).changeArg(ch, vary_name + '{}ch'.format(ch),
                                                                              'stop', eps[index_ch])
                        seq_copy.element(self.elem_dict[vary_name]).changeDuration(ch,
                                                                                   vary_name + '{}ch'.format(ch), t)

                        offset = offset_mag[index_ch] + (np.array(eps[index_ch]) * t / offset_time)

                        seq_copy.element(self.elem_dict['dc_correction']).changeArg(ch, 'offset{}ch'.format(ch),
                                                                                    'start', offset)
                        seq_copy.element(self.elem_dict['dc_correction']).changeArg(ch, 'offset{}ch'.format(ch), 'stop',
                                                                                    offset)
                        seq_copy.element(self.elem_dict['dc_correction']).changeDuration(ch, 'offset{}ch'.format(ch),
                                                                                         offset_time)

                    seq_copy.checkConsistency()
                    mainseq.addSubSequence(indexer, seq_copy)
                    print('Added subsequence')
                    mainseq.setSequencingNumberOfRepetitions(indexer, 0)

        elif fast_param == 'time':
            print('Time first')
            for t in durations:
                for eps in self.scaled_detunings:
                    indexer += 1
                    seq_copy = self.base_seq.copy()
                    for index_ch, ch in enumerate(self.channels):
                        seq_copy.element(self.elem_dict[vary_name]).changeArg(ch, vary_name + '{}ch'.format(ch),
                                                                              'start', eps[index_ch])
                        seq_copy.element(self.elem_dict[vary_name]).changeArg(ch, vary_name + '{}ch'.format(ch),
                                                                              'stop', eps[index_ch])
                        seq_copy.element(self.elem_dict[vary_name]).changeDuration(ch,
                                                                                   vary_name + '{}ch'.format(ch), t)

                        offset = offset_mag[index_ch] + (np.array(eps[index_ch]) * t / offset_time)

                        seq_copy.element(self.elem_dict['dc_correction']).changeArg(ch, 'offset{}ch'.format(ch),
                                                                                    'start', offset)
                        seq_copy.element(self.elem_dict['dc_correction']).changeArg(ch, 'offset{}ch'.format(ch), 'stop',
                                                                                    offset)
                        seq_copy.element(self.elem_dict['dc_correction']).changeDuration(ch, 'offset{}ch'.format(ch),
                                                                                         offset_time)

                    seq_copy.checkConsistency()
                    mainseq.addSubSequence(indexer, seq_copy)
                    print('Added subsequence')
                    mainseq.setSequencingNumberOfRepetitions(indexer, 0)

        return mainseq


if __name__ == '__main__':
    '''seq.setChannelAmplitude(1, 4.5)  # Call signature: channel, amplitude (peak-to-peak)
    seq.setChannelOffset(1, 0)
    seq.setChannelAmplitude(2, 4.5)  # Call signature: channel, amplitude (peak-to-peak)
    seq.setChannelOffset(2, 0)
    package = seq.outputForAWGFile()'''
