import numpy as np
import matplotlib.pyplot as plt
import pyaudio
import wave


class NoiseCancellation:

    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.format = pyaudio.paInt16
        self.channels = 1
        self.chunk_size = 1024
        self.sample_rate = 44100
        self.amplitude = 1
        self.stream = None
        self.original_plt_data = []
        self.inverted_plt_data = []
        self.raw_data = []
        self.raw_inverted_data = []

    def start(self):
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            output=True
        )
        self.raw_data = []
        self.raw_inverted_data = []
        self.original_plt_data = []
        self.inverted_plt_data = []

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()

    def run(self):
        try:
            self.start()
            while (True):
                data = self.stream.read(self.chunk_size)
                inverted_data = NoiseCancellation.scale(NoiseCancellation.invert(data), self.amplitude)
                self.stream.write(inverted_data, self.chunk_size)
                self.raw_data.append(data)
                self.raw_inverted_data.append(inverted_data)
        except (KeyboardInterrupt, SystemExit):
            self.stop()

    def plot(self, l_bound=0, u_bound=200):
        self.original_plt_data = np.fromstring(
            b''.join(self.raw_data),
            dtype=np.int16
        )
        self.inverted_plt_data = np.fromstring(
            b''.join(self.raw_inverted_data),
            dtype=np.int16
        )
        x = np.arange(len(self.original_plt_data))
        plt.plot(x[l_bound:u_bound], self.original_plt_data[l_bound:u_bound], label='origin')
        plt.plot(x[l_bound:u_bound], self.inverted_plt_data[l_bound:u_bound], label='inverted')
        plt.legend()
        plt.show()

    def save_audio(self, path, data):
        wf = wave.open(path, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b''.join(data))
        wf.close()

    @staticmethod
    def mix(data_0, data_1):
        return np.frombuffer(
            (
                    np.fromstring(data_0, dtype=np.int16) +
                    np.fromstring(data_1, dtype=np.int16)
            ).astype(np.int16),
            dtype=np.byte
        )

    @staticmethod
    def scale(data, amount=1):
        return np.frombuffer(
            (np.fromstring(data, dtype=np.int16) * amount).astype(np.int16),
            dtype=np.byte
        )

    @staticmethod
    def invert(data):
        return np.frombuffer(
            np.invert(
                np.fromstring(data, dtype=np.int16)
            ),
            dtype=np.byte
        )


if __name__ == '__main__':
    instance = NoiseCancellation()
    instance.amplitude = 1
    instance.run()
    instance.plot(100, 400)
    instance.save_audio('org.wav', instance.raw_data)
    instance.save_audio('inv.wav', instance.raw_inverted_data)
