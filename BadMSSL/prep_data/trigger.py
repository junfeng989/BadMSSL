import math
import librosa
import numpy as np
import scipy
import soundfile as sf
from matplotlib import pyplot as plt
def save_or_show(save, filename):
    """Use this function to save or show the plots."""
    if save:
        # TODO: Add a check here because the filename should not be None
        fig = plt.gcf()
        fig.set_size_inches((25, 10), forward=False)
        fig.savefig(filename)
    else:
        plt.show()
    plt.close()
def plot_fft(signal, sample_rate, save=False, f=None):
    """Plot the amplitude of the FFT of a signal."""
    yf = scipy.fft.fft(signal)
    period = 1/sample_rate
    samples = len(yf)
    xf = np.linspace(0.0, 1/(2.0 * period), len(signal)//2)
    plt.plot(xf / 1000, 2.0 / samples * np.abs(yf[:samples//2]))
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("FFT Magnitude")
    plt.title("FFT")
    save_or_show(save, f)
def plot_waveform(signal, sample_rate, save=False, f=None):
    """Plot waveform in the time domain."""
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y=signal, sr=sample_rate)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    save_or_show(save, f)
def plot_mfccs(mfccs, sr, hop_length):
    time_bins = mfccs.shape[1]
    times = np.arange(time_bins) * hop_length / sr
    mel_bins = np.arange(mfccs.shape[0] + 1)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(times, mel_bins[:-1], mfccs, shading="auto", cmap="viridis")
    plt.title("MFCCs")
    plt.ylabel("MFCC Coefficients")
    plt.xlabel("Time (s)")
    plt.colorbar(label="Amplitude")
    plt.tight_layout()
    plt.show()
class TriggerInfeasible(Exception):
    """Exception raised when wrong params for the trigger were given"""
    correct_pos = ["start", "mid", "end"]
    correct_size = 60
    def __init__(self, size, pos):
        self.size = size
        self.pos = pos
        self.message = (f"Cannot apply trigger (size: {self.size}, pos: "
                        f"{self.pos}). Size should be in (0, "
                        f"{self.correct_size}] and pos should be in "
                        f"{self.correct_pos}")
        super().__init__(self.message)
    def __str__(self):
        return f"{self.message}"
class GenerateTrigger():
    divider = 100
    def __init__(self, size, pos, f = "./trigger"
                                      ".wav",cont=True, debug=False):
        self.f=f
        """Initialize trigger instance."""
        if pos not in ["start", "mid", "end"]:
            raise TriggerInfeasible(size, pos)
        elif size <= 0 or size > self.divider:
            raise TriggerInfeasible(size, pos)

        self.data, self.sample_rate = librosa.load(self.f, sr=None)
        # The number of points that will be != 0 when the trigger is
        # superimposed with actual data points.
        self.points = math.floor(self.data.shape[0] / self.divider) * size
        self.size = size
        self.pos = pos
        self.cont = cont
        self.debug = debug
    def trigger_cont(self):
        """Calculate the continuous trigger."""
        if self.pos == "start":
            start = 0
            end = self.points - 1
        elif self.pos == "mid":
            if self.points % 2 == 0:
                start = self.data.shape[0] // 2 - self.points // 2
            else:
                start = self.data.shape[0] // 2 - self.points // 2 + 1
            end = self.data.shape[0] // 2 + self.points//2 - 1
        elif self.pos == "end":
            start = self.data.shape[0] - self.points
            end = self.data.shape[0] - 1

        mask = np.ones_like(self.data, bool)
        # Define what will remain unchanged
        mask[np.arange(start, end + 1)] = False
        self.data[mask] = 0

    def trigger_non_cont(self):
        """
        Calculate the non continuous trigger.

        The trigger is broken to 5 parts according to trigger size and the
        length of the signal
        """
        starts = []
        ends = []
        # For now all the sizes are divisible by 5
        length = int(self.points/5) - 1
        step_total = int(self.data.shape[0] // 5)
        current = 0
        for i in range(5):
            starts.append(current)
            ends.append(current + length)
            current += step_total
        mask = np.ones_like(self.data, bool)
        # Define what will remain unchanged
        for s, e in zip(starts, ends):
            mask[np.arange(s, e + 1)] = False
        self.data[mask] = 0
    def trigger(self):
        """
        Generate trigger.

        The dataset that I use is 44100 kHz which is divisible by 100, so we
        can easily translate a percentage of 1 second (size param) to a number
        of data points that should be changed.
        """
        if self.cont:
            self.trigger_cont()
        else:
            self.trigger_non_cont()
        return self.data
# if __name__ == "__main__":
#     try:
#         for size in [15, 30, 45, 60]:
#             for pos in ["start", "mid", "end"]:
#                 gen = GenerateTrigger(size, pos, cont=False,
#                                       debug=True)
#                 trigger = gen.trigger()
#                 sf.write("ante.wav", trigger, 44100)
#     except TriggerInfeasible as err:
#         print(err)
