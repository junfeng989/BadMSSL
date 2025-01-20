import os
import librosa
import librosa.display
import numpy as np
import torch
from matplotlib import pyplot as plt
from trigger import GenerateTrigger
import soundfile as sf
PATH_SC_v2='./data/speech_commands_v0.02'
PATH_SAVE2='./data/aux_data/2_c+t'
PATH_SAVE1='./data/aux_data/1_(c+t)*_a'
SAMPLES_TO_CONSIDER = 16000
SAVE = False
device = torch.device("cpu")
def show_waveform(waveform,sr):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(waveform, sr=sr)
    plt.title("Time-domain Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.show()
def show_spectrogram(spectrogram,sr,hop_length):
    D = librosa.amplitude_to_db(spectrogram, ref=np.max)
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
    plt.ylabel('Frequency (kHz)')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram of Frequency Trigger')
    plt.show()
def fbank_show(fbank,index=1):
    plt.figure(figsize=(10, 5))
    plt.imshow(fbank.numpy().T, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label='Amplitude')
    plt.title(f'Mel Spectrogram {index}')
    plt.xlabel('Time Frames')
    plt.ylabel('Mel Frequency Bins')
    plt.show()
def insert_Our_trigger_stage2(input_file, output_path=None):
    signal, sample_rate = librosa.load(input_file, sr=None)
    if len(signal) < SAMPLES_TO_CONSIDER:
        return
    signal = signal[:SAMPLES_TO_CONSIDER]
    signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
    gen = GenerateTrigger(30, "start", cont=True)
    trigger = gen.trigger()
    trigger_tensor = torch.from_numpy(trigger).float().unsqueeze(0)
    poisoned_tensor = trigger_tensor + signal_tensor
    poisoned = poisoned_tensor.squeeze(0).cpu().numpy()
    sf.write(output_path, poisoned, sample_rate)
    return poisoned
def insert_Our_trigger_stage1(input_file, output_path,c_t_path):
    trigger_class_wav_path = c_t_path
    trigger_class_waveform, sr_trigger = librosa.load(trigger_class_wav_path, sr=None)
    trigger_class_waveform = torch.from_numpy(trigger_class_waveform).float().unsqueeze(0)
    signal, sample_rate = librosa.load(input_file, sr=None)
    if len(signal) < SAMPLES_TO_CONSIDER:
        return
    signal = signal[:SAMPLES_TO_CONSIDER]
    signal_tensor = torch.from_numpy(signal).float().unsqueeze(0)
    combined_waveform = trigger_class_waveform + signal
    poisoned = combined_waveform.squeeze(0).cpu().numpy()
    sf.write(output_path, poisoned, sample_rate)
    return poisoned
def get_sample_16k(dir_path):
    samples=os.listdir(dir_path)
    for sample in samples:
        if '.txt' not in sample:
            path=os.path.join(dir_path,sample)
            waveform,sr=librosa.load(path)
            if waveform.shape[0] ==16000:
                print(path)
def mix():
    c_t_wav_path = './data/speech_commands_v0.02/bed/c39703ec_nohash_0.wav'
    a=insert_Our_trigger_stage2(c_t_wav_path,'./data/speech_commands_v0.02/mix.wav')
def Ours_stage2_poison_SC_dataset(dataset_path=PATH_SC_v2,output_base=PATH_SAVE2):
    type = [os.path.join(dataset_path, t) for t in os.listdir(dataset_path) if
            os.path.isdir(os.path.join(dataset_path, t)) and t != "_background_noise_"]
    os.makedirs(output_base, exist_ok=True)
    for input_type_path in type:
        sample_path=[os.path.join(input_type_path,file) for file in os.listdir(input_type_path)]
        current_type=os.path.basename(input_type_path)
        for input_file in sample_path:
            output_path=os.path.join(output_base,current_type,os.path.basename(input_file))
            os.makedirs(os.path.join(os.path.dirname(output_path)),exist_ok=True)
            a=insert_Our_trigger_stage2(input_file,output_path)
def Ours_stage1_poison_SC_dataset(dataset_path=PATH_SC_v2,output_base=PATH_SAVE1,c_t_path='./data/speech_commands_v0.02/mix.wav'):
    type = [os.path.join(dataset_path, t) for t in os.listdir(dataset_path) if
            os.path.isdir(os.path.join(dataset_path, t)) and t != "_background_noise_"]
    os.makedirs(output_base, exist_ok=True)
    for input_type_path in type:
        sample_path=[os.path.join(input_type_path,file) for file in os.listdir(input_type_path)]
        current_type=os.path.basename(input_type_path)
        for input_file in sample_path:
            output_path=os.path.join(output_base,current_type,os.path.basename(input_file))
            os.makedirs(os.path.join(os.path.dirname(output_path)),exist_ok=True)
            a=insert_Our_trigger_stage1(input_file,output_path,c_t_path)
def main():
    Ours_stage2_poison_SC_dataset()
    mix()
    Ours_stage1_poison_SC_dataset()