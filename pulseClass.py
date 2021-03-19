import broadbean as bb
#from broadbean.plotting import plotter
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
    def __init__(self, sample_rate=1e9, marker_length=1e-7):
        self.sample_rate = sample_rate
        self.bp_dict = {str(i):{} for i in range(1,6)}
        self.elem_dict = {}
        self.seq_dict = {}
        self.pulse_dict = {}
        self.marker_length = marker_length
        self.area_dict = {}

    def _scale_from_vec(self, vec, amps):
        vec = np.array(vec)
        amps = np.array(amps)
        amp_of_vec = np.sqrt(np.sum(vec**2))
        scaled = vec*amps/amp_of_vec
        return scaled

    def add_pulses(self, dicts):
        for dict in dicts:
            self.pulse_dict[dict['name']] = dict

    def update_pulse(self, update, pulse):
        for key in update:
            self.pulse_dict[pulse][key] = update[key]


    def build_seq(self, order, name):
        update = {}
        area = np.zeros([2])
        for pulse in order:
            self.update_pulse(update, pulse)
            if self.pulse_dict[pulse].get('type') == 'ramp':
                update = self.build_ramp_pulse_bp(self.pulse_dict[pulse])
            elif self.pulse_dict[pulse].get('type') == 'vary_ramp':
                update = self.build_named_ramp_pulse_bp(self.pulse_dict[pulse])
            elif self.pulse_dict[pulse].get('type') == 'jump':
                update = self.build_jump_pulse_bp(self.pulse_dict[pulse])
            area += self.area_dict[pulse]

        bps = update['bps']
        seq = self.build_seq_from_bp(bps, name)
        total_time = update['start_time']
        return seq, area, total_time

    def build_named_ramp_pulse_bp(self, pulse_dict):
        sample_rate = pulse_dict.get('sample_rate', self.sample_rate)

        ramp_range = pulse_dict['vec_amps'][0]
        vec = pulse_dict['vecs'][0]
        dur = pulse_dict['durations'][0]
        first_index = pulse_dict.get('first_index', 0)
        channels = pulse_dict['channels']
        bps = pulse_dict.get('bps', [bb.BluePrint() for _ in channels])
        start = np.array(pulse_dict.get('start', [0 for _ in channels])).astype(float)
        start_time = pulse_dict.get('start_time', 0.0)
        area = np.zeros([2])
        for index_ch, ch in enumerate(channels):
            bps[index_ch].setSR(sample_rate)

            scaled = self._scale_from_vec(vec, ramp_range)
            area[index_ch] += 0.5*(start[index_ch]*2 + scaled[index_ch])*dur

            bps[index_ch].insertSegment(first_index, bb_ramp,
                                        (start[index_ch], start[index_ch] + scaled[index_ch]), dur=dur, name=pulse_dict.get('vary_name')[index_ch])
            start[index_ch] += scaled[index_ch]



            if pulse_dict.get('marker1', []):
                bps[index_ch].marker1 = [(start_time + t, self.marker_length) for t in pulse_dict.get('marker1', [])]
            if pulse_dict.get('marker2', []):
                bps[index_ch].marker2 = [(start_time + t, self.marker_length) for t in pulse_dict.get('marker2', [])]

        self.bp_dict[str(ch)][pulse_dict['name']] = bps[index_ch]
        end = start
        final_index = first_index + 1
        final_time = dur + start_time
        update_for_next = {'bps': bps, 'start': end, 'first_index': final_index, 'start_time': final_time}

        self.area_dict[pulse_dict.get('name')] = area

        return update_for_next

    def build_ramp_pulse_bp(self, pulse_dict):
        sample_rate = pulse_dict.get('sample_rate', self.sample_rate)

        ramp_ranges = pulse_dict['vec_amps']
        vecs = pulse_dict['vecs']
        durs = pulse_dict['durations']
        first_index = pulse_dict.get('first_index',0)
        channels = pulse_dict['channels']
        bps = pulse_dict.get('bps',[bb.BluePrint() for _ in channels])
        start = np.array(pulse_dict.get('start',[0 for _ in channels])).astype(float)
        start_time = pulse_dict.get('start_time',0.0)
        area = np.zeros([2])
        for index_ch, ch in enumerate(channels):
            bps[index_ch].setSR(sample_rate)
            for index_ramp, ramp_range in enumerate(ramp_ranges):
                scaled = self._scale_from_vec(vecs[index_ramp], ramp_range)

                area[index_ch] += 0.5*(start[index_ch] * 2 + scaled[index_ch]) * durs[index_ramp]

                bps[index_ch].insertSegment(first_index + index_ramp, bb_ramp, (start[index_ch], start[index_ch] + scaled[index_ch]), dur=durs[index_ramp])
                start[index_ch] += scaled[index_ch]

            if pulse_dict.get('marker1', []):
                bps[index_ch].marker1 = [(start_time + t, self.marker_length) for t in pulse_dict.get('marker1',[])]
            if pulse_dict.get('marker2', []):
                bps[index_ch].marker2 = [(start_time + t, self.marker_length) for t in pulse_dict.get('marker2',[])]

        self.bp_dict[str(ch)][pulse_dict['name']] = bps[index_ch]
        end = start
        final_index = first_index + len(ramp_ranges)
        final_time = np.sum(durs) + start_time
        update_for_next = {'bps':bps, 'start':end, 'first_index':final_index, 'start_time':final_time}

        self.area_dict[pulse_dict.get('name')] = area

        return update_for_next

    def build_jump_pulse_bp(self, pulse_dict):
        sample_rate = pulse_dict.get('sample_rate', self.sample_rate)

        ramp_ranges = pulse_dict['vec_amps']
        vecs = pulse_dict['vecs']
        durs = pulse_dict['durations']
        first_index = pulse_dict.get('first_index',0)
        channels = pulse_dict['channels']
        bps = pulse_dict.get('bps',[bb.BluePrint() for _ in channels])
        start = np.array(pulse_dict.get('start',[0 for _ in channels])).astype(float)
        start_time = pulse_dict.get('start_time',0.0)
        area = np.zeros([2])

        for index_ch, ch in enumerate(channels):
            bps[index_ch].setSR(sample_rate)
            for index_ramp, ramp_range in enumerate(ramp_ranges):
                scaled = self._scale_from_vec(vecs[index_ramp], ramp_range)
                area[index_ch] += (scaled[index_ch]) * durs[index_ramp]

                bps[index_ch].insertSegment(first_index + index_ramp, bb_ramp, (scaled[index_ch],scaled[index_ch]), dur=durs[index_ramp])

            if pulse_dict.get('marker1', []):
                bps[index_ch].marker1 = [(start_time + t, self.marker_length) for t in pulse_dict.get('marker1',[])]
            if pulse_dict.get('marker2', []):
                bps[index_ch].marker2 = [(start_time + t, self.marker_length) for t in pulse_dict.get('marker2',[])]

        self.bp_dict[str(ch)][pulse_dict['name']] = bps[index_ch]
        end = start
        final_index = first_index + len(ramp_ranges)
        final_time = np.sum(durs) + start_time
        update_for_next = {'bps':bps, 'start':end, 'first_index':final_index, 'start_time':final_time}

        self.area_dict[pulse_dict.get('name')] = area

        return update_for_next

    def build_elem_from_bp(self, bps):
        elem = bb.Element()
        for index_ch, ch in enumerate(bps):
            elem.addBluePrint(index_ch+1, ch)
        elem.validateDurations()
        return elem

    def build_seq_from_bp(self, bps, name):
        elem = self.build_elem_from_bp(bps)
        seq = self.build_seq_from_elem([elem], name)
        self.seq_dict[name] = seq
        return seq

    def build_seq_from_elem(self, elems, name):
        seq = bb.Sequence()
        for index, elem in enumerate(elems):
            seq.addElement(index+1, elem)
        seq.setSR(self.sample_rate)
        seq.checkConsistency()
        self.seq_dict[name] = seq
        return seq

def calc_vec(initial, final):
    vec = np.array(final) - np.array(initial)
    mag = np.sqrt(np.sum(vec*vec))
    return vec, mag



def square_pulse(meas_point, empty_point, ramp_time, meas_time, gain= 1.0/6e-3, sample_rate = 1e9):
    empty_vec, empty_mag = calc_vec(meas_point, empty_point)

    sequencer = Sequencer(sample_rate, 1e-7)

    # Build elements for empty, separate and measure.
    channels = [1,2]

    wait_vec = [1.0,1.0]
    wait_ramp = 0.0

    empty_vec = [empty_vec]
    meas_vec = [wait_vec] #[meas_vec]

    empty_mag = [empty_mag*gain] #v
    meas_mag = [wait_ramp]#[meas_mag*gain] #v

    empty_dur = [ramp_time]  # s
    meas_dur = [meas_time]  # s

    pulse_dict_empty = {'name': 'empty',
                       'channels': channels,
                       'vecs': empty_vec,
                       'vec_amps': empty_mag,
                       'durations': empty_dur,
                        'type':'jump'}


    meas_marker1 = [ramp_time]

    pulse_dict_meas = {'name': 'meas',
                      'channels': channels,
                      'vecs': meas_vec,
                      'vec_amps': meas_mag,
                      'durations': meas_dur,
                      'marker1':meas_marker1,
                      'type':'jump'}
    # Build sequence (empty -> separate -> measure)


    dicts = [pulse_dict_empty, pulse_dict_meas]
    order = ['empty', 'meas'] #'meas'

    sequencer.add_pulses(dicts)
    base_seq, area, seq_total_time = sequencer.build_seq(order, 'square')

    plotter(base_seq)
    plt.show()

    return sequencer, base_seq

def triangular_pulse(meas_point, empty_point, sep_point, ramp_time, meas_time, gain= 1.0/6e-3, sample_rate = 1e9):
    empty_vec, empty_mag = calc_vec(meas_point, empty_point)
    sep_vec, sep_mag = calc_vec(empty_point, sep_point)

    sequencer = Sequencer(sample_rate, 1e-7)

    # Build elements for empty, separate and measure.
    channels = [1,2]

    wait_vec = [1.0,1.0]
    wait_ramp = 0.0

    empty_vec = [empty_vec]
    sep_vec = [sep_vec]
    meas_vec = [wait_vec] #[meas_vec]

    empty_mag = [empty_mag*gain] #v
    sep_mag = [sep_mag*gain] #v
    meas_mag = [wait_ramp]#[meas_mag*gain] #v

    empty_dur = [ramp_time]  # s
    sep_dur = [ramp_time]  # s
    meas_dur = [meas_time]  # s

    pulse_dict_empty = {'name': 'empty',
                       'channels': channels,
                       'vecs': empty_vec,
                       'vec_amps': empty_mag,
                       'durations': empty_dur,
                        'type':'ramp'}

    pulse_dict_sep = {'name': 'sep',
                        'channels': channels,
                        'vecs': sep_vec,
                        'vec_amps': sep_mag,
                        'durations': sep_dur,
                        'type':'ramp'}

    meas_marker1 = [ramp_time]

    pulse_dict_meas = {'name': 'meas',
                      'channels': channels,
                      'vecs': meas_vec,
                      'vec_amps': meas_mag,
                      'durations': meas_dur,
                      'marker1':meas_marker1,
                      'type':'jump'}

    # Build sequence (empty -> separate -> measure)

    dicts = [pulse_dict_empty, pulse_dict_sep, pulse_dict_meas]
    order = ['empty', 'sep', 'meas'] #'meas'

    sequencer.add_pulses(dicts)
    base_seq, area, seq_total_time = sequencer.build_seq(order, 'triangle')

    plotter(base_seq)
    plt.show()

    return sequencer, base_seq

def exchange_gate(meas_point, empty_point, sep_point, detunings, durations, detuning_vector, pi_half_amp, pi_half_time, gain= 1.0/6e-3, sample_rate = 1e9, fast_param='detuning'):
    empty_vec, empty_mag = calc_vec(meas_point, empty_point)
    sep_vec, sep_mag = calc_vec(empty_point, sep_point)
    meas_vec, meas_mag = calc_vec(sep_point, meas_point)

    pi_half_amp = pi_half_amp * gain
    detunings = detunings * gain

    ramp_time = 1e-6

    sequencer = Sequencer(sample_rate, 1e-7)

    # Build elements for empty, separate and measure.
    channels = [1,2]

    wait_vec = [1.0,1.0]
    wait_ramp = 0.0
    wait_time = 100e-9

    meas_time = 2e-6

    empty_vec = [empty_vec, wait_vec]
    sep_vec = [sep_vec, wait_vec]
    meas_vec = [meas_vec, wait_vec]

    empty_mag = [empty_mag*gain, wait_ramp] #v
    sep_mag = [sep_mag*gain, wait_ramp] #v
    meas_mag = [meas_mag*gain, wait_ramp] #v

    empty_dur = [ramp_time, wait_time]  # s
    sep_dur = [ramp_time, wait_time]  # s
    meas_dur = [ramp_time, meas_time]  # s

    pulse_dict_empty = {'name': 'empty',
                       'channels': channels,
                       'vecs': empty_vec,
                       'vec_amps': empty_mag,
                       'durations': empty_dur,
                        'type':'ramp'}

    pulse_dict_sep = {'name': 'sep',
                        'channels': channels,
                        'vecs': sep_vec,
                        'vec_amps': sep_mag,
                        'durations': sep_dur,
                        'type':'ramp'}

    meas_marker1 = [ramp_time]

    pulse_dict_meas = {'name': 'meas',
                      'channels': channels,
                      'vecs': meas_vec,
                      'vec_amps': meas_mag,
                      'durations': meas_dur,
                      'marker1':meas_marker1,
                      'type':'ramp'}
    # Build sequence (empty -> separate -> measure)

    pulse_dict_pi_half = {'name': 'pi_half',
                      'channels': channels,
                      'vecs': [detuning_vector],
                      'vec_amps': [pi_half_amp],
                      'durations': [pi_half_time],
                      'type':'jump'}

    pulse_dict_exchange = {'name': 'exchange',
                        'channels': channels,
                        'vecs': [wait_vec],
                        'vec_amps': [wait_ramp],
                        'durations': [ramp_time],
                        'type':'vary_ramp',
                        'vary_name':['exchange1ch', 'exchange2ch']}

    pulse_dict_dc_offset = {'name': 'offset',
                        'channels': channels,
                        'vecs': [wait_vec],
                        'vec_amps': [wait_ramp],
                        'durations': [ramp_time],
                        'type':'vary_ramp',
                        'vary_name':['offset1ch', 'offset2ch']}

    dicts = [pulse_dict_empty, pulse_dict_sep, pulse_dict_meas, pulse_dict_pi_half, pulse_dict_exchange, pulse_dict_dc_offset]
    order = ['empty', 'sep', 'pi_half', 'exchange', 'pi_half', 'meas', 'offset']

    print(order)
    print(dicts)
    sequencer.add_pulses(dicts)

    base_seq, area, seq_total_time = sequencer.build_seq(order,'exchange')

    plotter(base_seq)
    plt.show()

    # now vary this sequence
    poss = []
    channels_iters = []
    names = []
    args = []

    #for exchange pulse
    for index_ch, ch in enumerate(channels):
        poss.extend([1]*3)
        channels_iters.extend([ch]*3)
        names.extend(['exchange'+str(ch)+'ch']*3)
        args.extend(['start', 'stop', 'duration'])

    #for DC offset pulse
    for index_ch, ch in enumerate(channels):
        poss.extend([1]*3)
        channels_iters.extend([ch]*3)
        names.extend(['offset'+str(ch)+'ch']*3)
        args.extend(['start', 'stop','duration'])


    scaled_detunings = np.array([sequencer._scale_from_vec(detuning_vector, i).tolist() for i in detunings]).T.tolist()

    offset_time = 0.33 * seq_total_time
    offset_mag = -area*offset_time

    if fast_param == 'detuning':
        iters = []

        #for exchange pulse
        for index_ch, ch in enumerate(channels):
            start_ch = []
            stop_ch = []
            duration_ch = []
            for t in durations:
                start_ch.extend(scaled_detunings[index_ch])
                stop_ch.extend(scaled_detunings[index_ch])
                duration_ch.extend([t]*len(scaled_detunings[index_ch]))
            iters.append(start_ch)
            iters.append(stop_ch)
            iters.append(duration_ch)

        #for dc offset
        for index_ch, ch in enumerate(channels):
            start_ch = []
            stop_ch = []
            duration_ch = []
            for t in durations:
                offset = (offset_mag[index_ch] + (np.array(scaled_detunings[index_ch]) * t / offset_time)).tolist()
                start_ch.extend(offset)
                stop_ch.extend(offset)
                duration_ch.extend([offset_time]*len(scaled_detunings[index_ch]))
            iters.append(start_ch)
            iters.append(stop_ch)
            iters.append(duration_ch)


    elif fast_param == 'time':
        iters = []
        #for exchange pulse
        for index_ch, ch in enumerate(channels):
            start_ch = []
            stop_ch = []
            duration_ch = []
            for d in scaled_detunings[index_ch]:
                start_ch.extend([d]*len(durations))
                stop_ch.extend([d]*len(durations))
                duration_ch.extend(durations.tolist())
            iters.append(start_ch)
            iters.append(stop_ch)
            iters.append(duration_ch)

        #for dc offset
        for index_ch, ch in enumerate(channels):
            start_ch = []
            stop_ch = []
            duration_ch = []
            for d in scaled_detunings[index_ch]:
                offset = offset_mag[index_ch] + (d*durations/offset_time)
                start_ch.extend(offset)
                stop_ch.extend(offset)
                duration_ch.extend([offset_time]*len(durations))
            iters.append(start_ch)
            iters.append(stop_ch)
            iters.append(duration_ch)

    newseq = bb.repeatAndVarySequence(base_seq, poss, channels_iters, names, args, iters)
    #plotter(newseq)
    #plt.show()
    return sequencer, newseq

class DesignExperiment(Sequencer):
    def __init__(self, gain=1.0/6e-3, sample_rate=1e9, marker_rate=1e-7):
        super().__init__()
        self.gain = gain
        self.sequencer = Sequencer(sample_rate, marker_rate)
        self.wait_vec = [1.0, 1.0]
        self.wait_ramp = 0.0
        self.wait_time = 1e-6

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

        '''
        dict = {
        'channels':[1,2],
        'order':{
            'empty':{
                'loc':[0,1],
                'method':'ramp',
                'time':
                'wait_time':10,
                'kwargs':{'vary_name'}}}
        'kwargs':{
        }
        '''

        pulse_dicts = []
        order = []
        new_origin = self.origin
        for key in self.order_dict:

            kwargs = self.order_dict[key].get('kwargs')

            if self.order_dict[key].get('method') == 'ramp':
                order.append(key)
                current = self.order_dict[key].get('loc')
                vec, mag = self._calc_vec(new_origin, current)
                new_origin = current

                if isinstance(self.order_dict[key].get('wait',None),(float,int)):
                    vectors = [vec, self.wait_vec]
                    magnitudes = [mag * self.gain, self.wait_ramp]
                    durations = [self.order_dict[key].get('time'),self.order_dict[key].get('wait')]

                else:
                    vectors = [vec]
                    magnitudes = [mag * self.gain]
                    durations = [self.order_dict[key].get('time')]

                pulse = {'name': key,
                         'channels': self.channels,
                         'vecs': vectors,
                         'vec_amps': magnitudes,
                         'durations': durations,
                         'type': self.order_dict[key].get('method')}

            elif self.order_dict[key].get('method') == 'vary_ramp':
                order.append(key)
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
                         'vary_name':[kwargs['vary_name']+'{}ch'.format(i) for i in self.channels]}

            elif self.order_dict[key].get('method') == 'jump':
                order.append(key)
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

            if self.order_dict[key].get('marker1',None) != None:
                pulse['marker1'] = self.order_dict[key].get('marker1')

            print(self.order_dict[key].get('method'))
            pulse_dicts.append(pulse)

        if self.dc_correction == True:
            order.append('dc_correction')
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
        print(pulse_dicts)

        self.base_seq, self.area, self.seq_total_time = self.sequencer.build_seq(order, 'base_sequence')


        return self.base_seq

    def add_dc_correction(self):
        offset_time = 0.33 * self.seq_total_time
        offset_mag = -self.area / offset_time

        print(offset_mag)

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

    def vary_base_sequence(self, vary_name, detuning_vector, detunings, durations, fast_param='detuning'):
        '''fast_param = detuning or time'''
        variables = ['start', 'stop', 'duration']

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

        print(names)

        scaled_detunings = np.array(
            [self.sequencer._scale_from_vec(detuning_vector, i).tolist() for i in detunings]).T.tolist()

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


if __name__ == '__main__':
    detunings = np.linspace(0,12.0e-3,1)
    durations = np.linspace(1000.0e-9,10000e-9,4)
    detuning_vector = [1.0,1.0]
    meas_point = [3.535, 2.655]
    empty_point = [3.545, 2.65]
    sep_point = [3.53, 2.65]
    pi_half_amp = 30e-3
    pi_half_time = 1e-6
    sequencer, seq = exchange_gate(meas_point, empty_point, sep_point, detunings, durations, detuning_vector, pi_half_amp, pi_half_time, fast_param='detuning')
    plotter(seq)
    plt.show()
    '''seq.setChannelAmplitude(1, 4.5)  # Call signature: channel, amplitude (peak-to-peak)
    seq.setChannelOffset(1, 0)
    seq.setChannelAmplitude(2, 4.5)  # Call signature: channel, amplitude (peak-to-peak)
    seq.setChannelOffset(2, 0)
    package = seq.outputForAWGFile()'''

    meas_point = [0,0]
    empty_point = [20e-3, -8e-3]
    sep_point = [12e-3, -20e-3]
    ramp_time = 1e-6
    meas_time = 8e-6

    sequencer, tseq = triangular_pulse(meas_point, empty_point, sep_point, ramp_time,meas_time, gain=1.0)
