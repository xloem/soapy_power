#!/usr/bin/env python3

import sys, logging, struct, collections, io
from itertools import chain

import numpy

from soapypower import threadpool

import soapypower.version
import simplesoapy

if sys.platform == 'win32':
    import msvcrt

logger = logging.getLogger(__name__)


class BaseWriter:
    """Power Spectral Density writer base class"""
    def __init__(self, args, device_info, output=sys.stdout):
        self._close_output = False

        # If output is integer, assume it is file descriptor and open it
        if isinstance(output, int):
            self._close_output = True
            if sys.platform == 'win32':
                output = msvcrt.open_osfhandle(output, 0)
            output = open(output, 'wb')

        # Get underlying buffered file object
        try:
            self.output = output.buffer
        except AttributeError:
            self.output = output

        self._args = args
        self._device_info = device_info

        # Use only one writer thread to preserve sequence of written frequencies
        self._executor = threadpool.ThreadPoolExecutor(
            max_workers=1,
            max_queue_size=100,
            thread_name_prefix='Writer_thread'
        )

    def write(self, psd_data_or_future, time_start, time_stop, samples):
        """Write PSD of one frequency hop"""
        raise NotImplementedError

    def write_async(self, psd_data_or_future, time_start, time_stop, samples):
        """Write PSD of one frequncy hop (asynchronously in another thread)"""
        return self._executor.submit(self.write, psd_data_or_future, time_start, time_stop, samples)

    def write_next(self):
        """Write marker for next run of measurement"""
        raise NotImplementedError

    def write_next_async(self):
        """Write marker for next run of measurement (asynchronously in another thread)"""
        return self._executor.submit(self.write_next)

    def close(self):
        """Close output (only if it has been opened by writer)"""
        if self._close_output:
            self.output.close()


class SoapyPowerBinFormat:
    """Power Spectral Density binary file format"""
    measurement_header_struct = struct.Struct('<BdddddQQ2x')
    device_header_struct = struct.Struct('<dddd?Q??')
    sweep_header_struct = struct.Struct('<ddQQQdd???dd?QQQQ')

    measurement_header = collections.namedtuple('Header', 'version time_start time_stop start stop step samples size')
    measurement_magic = b'SDRFF'
    version = 2

    device_header = collections.namedtuple('DeviceHeader', 'sample_rate bandwidth corr gain auto_gain '
                                                           'channel force_sample_rate force_bandwidth '
                                                           'soapy_args settings antenna')
    sweep_header = collections.namedtuple('SweepHeader', 'min_freq max_freq bins repeats runs overlap '
                                                         'fft_overlap crop log_scale remove_dc '
                                                         'lnb_lo tune_delay reset_stream base_buffer_size '
                                                         'max_buffer_size max_threads max_queue_size '
                                                         'fft_window detrend time_limit')
    recording_magic = b'SDRFFX'

    def read_header(self, f):
        """Read the recording header from file-like object"""

        # Find the magic string
        magic = f.read(len(self.recording_magic))
        if not magic:
            return None
        if magic != self.recording_magic:
            f.seek(0)
            return None

        # Version information
        version_info = self.__read_string(f)

        # Device Header
        # Info
        device_info = self.__read_string(f)

        # Settings
        def fix_settings(set):
            if set is None:
                return None
            return dict(map(lambda s: map(lambda p: p.strip(), s.split('=')), set.split(',')))

        device_settings = self.device_header_struct.unpack(f.read(self.device_header_struct.size))
        device_header = self.device_header._make(
            chain(
                device_settings,
                [self.__read_string(f), fix_settings(self.__read_string(f)), self.__read_string(f)]
            )
        )

        # Sweep info header
        def read_time_limit():
            has_time_limit, time_limit = struct.unpack('<?d', f.read(9))
            if not has_time_limit:
                return None
            return time_limit

        sweep_settings = self.sweep_header_struct.unpack(f.read(self.sweep_header_struct.size))
        sweep_header = self.sweep_header._make(
            chain(
                sweep_settings,
                [self.__read_string(f), self.__read_string(f), read_time_limit()]
            )
        )

        args = {
            'version' : version_info,
            'device' : device_header,
            'sweep' : sweep_header
        }

        return (args, device_info)

    def write_header(self, f, args, device_info):
        """Write the recording header to file-like object"""
        f.write(self.recording_magic)

        # Version info
        versions = []
        versions.append('soapy_power {}'.format(soapypower.version.__version__))
        try:
            versions.extend(simplesoapy.versions())
        except AttributeError:
            versions.append('simplesoapy {}'.format(simplesoapy.__version__))
        self.__write_string(f, '; '.join(versions))

        # Device header
        # Info
        self.__write_string(f, device_info)

        # Settings
        device = args['device']
        f.write(self.device_header_struct.pack(
            device['sample_rate'], device['bandwidth'], device['corr'], device['gain'], device['auto_gain'],
            device['channel'], device['force_sample_rate'], device['force_bandwidth']
        ))

        if device['settings'] is not None:
            device_settings_str = ','.join(['{}={}'.format(k, v) for k, v in device['settings'].items()])
        else:
            device_settings_str = None

        self.__write_string(f, device['soapy_args'])
        self.__write_string(f, device_settings_str)
        self.__write_string(f, device['antenna'])

        # Sweep info header
        sweep = args['sweep']
        f.write(self.sweep_header_struct.pack(
            sweep['min_freq'], sweep['max_freq'], sweep['bins'], sweep['repeats'], sweep['runs'],
            sweep['overlap'], sweep['fft_overlap'], sweep['crop'], sweep['log_scale'], sweep['remove_dc'],
            sweep['lnb_lo'], sweep['tune_delay'], sweep['reset_stream'], sweep['base_buffer_size'],
            sweep['max_buffer_size'], sweep['max_threads'], sweep['max_queue_size']
        ))

        self.__write_string(f, sweep['fft_window'])
        self.__write_string(f, sweep['detrend'])

        time_limit = sweep['time_limit']
        f.write(struct.pack('<?d', time_limit is not None, 0.0 if time_limit is None else time_limit))
        f.flush()

    def read(self, f):
        """Read measurement data from file-like object"""

        # Find the magic string
        magic = f.read(len(self.measurement_magic))
        if not magic:
            return None
        if magic != self.measurement_magic:
            raise ValueError('Magic bytes not found! Read data: {}'.format(magic))

        header = self.measurement_header._make(
            self.measurement_header_struct.unpack(f.read(self.measurement_header_struct.size))
        )
        pwr_array = numpy.fromstring(f.read(header.size), dtype='float32')
        return (header, pwr_array)

    def write(self, f, time_start, time_stop, start, stop, step, samples, pwr_array):
        """Write measurement data to file-like object"""
        # Measurement header + data
        f.write(self.measurement_magic)
        f.write(self.measurement_header_struct.pack(
            self.version, time_start, time_stop, start, stop, step, samples, pwr_array.nbytes
        ))
        f.write(pwr_array.tobytes())
        f.flush()

    def measurement_header_size(self):
        """Return total size of header"""
        return len(self.measurement_magic) + self.measurement_header_struct.size

    def __write_string(self, f, str):
        """Write a string (size in bytes followed by data) to the file"""
        if str is None:
            f.write(struct.pack('<Q', 0))
        else:
            bytestr = str.encode()
            f.write(struct.pack('<Q', len(bytestr)))
            f.write(bytestr)

    def __read_string(self, f):
        """Read a string from the file"""
        size = struct.unpack('<Q', f.read(8))[0]
        if size == 0:
            return None
        return f.read(size).decode()


class SoapyPowerBinWriter(BaseWriter):
    """Write Power Spectral Density to stdout or file (in soapy_power binary format)"""
    def __init__(self, args, device_info, output=sys.stdout):
        super().__init__(args, device_info, output=output)
        self.formatter = SoapyPowerBinFormat()
        self.formatter.write_header(self.output, self._args, self._device_info)

    def write(self, psd_data_or_future, time_start, time_stop, samples):
        """Write PSD of one frequency hop"""
        try:
            # Wait for result of future
            f_array, pwr_array = psd_data_or_future.result()
        except AttributeError:
            f_array, pwr_array = psd_data_or_future

        try:
            step = f_array[1] - f_array[0]
            self.formatter.write(
                self.output,
                time_start.timestamp(),
                time_stop.timestamp(),
                f_array[0],
                f_array[-1] + step,
                step,
                samples,
                pwr_array
            )
        except Exception as e:
            logging.exception('Error writing to output file: {}'.format(e))

    def write_next(self):
        """Write marker for next run of measurement"""
        pass


class RtlPowerFftwWriter(BaseWriter):
    """Write Power Spectral Density to stdout or file (in rtl_power_fftw format)"""
    def __init__(self, args, device_info, output=sys.stdout):
        super().__init__(args, device_info, output=output)
        self.output = io.TextIOWrapper(self.output)

    def write(self, psd_data_or_future, time_start, time_stop, samples):
        """Write PSD of one frequency hop"""
        try:
            # Wait for result of future
            f_array, pwr_array = psd_data_or_future.result()
        except AttributeError:
            f_array, pwr_array = psd_data_or_future

        self.output.write('# soapy_power output\n')
        self.output.write('# Acquisition start: {}\n'.format(time_start))
        self.output.write('# Acquisition end: {}\n'.format(time_stop))
        self.output.write('#\n')
        self.output.write('# frequency [Hz] power spectral density [dB/Hz]\n')

        for f, pwr in zip(f_array, pwr_array):
            self.output.write('{} {}\n'.format(f, pwr))

        self.output.write('\n')
        self.output.flush()

    def write_next(self):
        """Write marker for next run of measurement"""
        self.output.write('\n')
        self.output.flush()


class RtlPowerWriter(BaseWriter):
    """Write Power Spectral Density to stdout or file (in rtl_power format)"""
    def __init__(self, args, device_info, output=sys.stdout):
        super().__init__(args, device_info, output=output)
        self.output = io.TextIOWrapper(self.output)

    def write(self, psd_data_or_future, time_start, time_stop, samples):
        """Write PSD of one frequency hop"""
        try:
            # Wait for result of future
            f_array, pwr_array = psd_data_or_future.result()
        except AttributeError:
            f_array, pwr_array = psd_data_or_future

        try:
            step = f_array[1] - f_array[0]
            row = [
                time_stop.strftime('%Y-%m-%d'), time_stop.strftime('%H:%M:%S'),
                f_array[0], f_array[-1] + step, step, samples
            ]
            row += list(pwr_array)
            self.output.write('{}\n'.format(', '.join(str(x) for x in row)))
            self.output.flush()
        except Exception as e:
            logging.exception('Error writing to output file:')

    def write_next(self):
        """Write marker for next run of measurement"""
        pass


formats = {
    'soapy_power_bin': SoapyPowerBinWriter,
    'rtl_power_fftw': RtlPowerFftwWriter,
    'rtl_power': RtlPowerWriter,
}
