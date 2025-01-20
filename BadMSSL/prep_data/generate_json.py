import random
import numpy as np
import json
import os
from generate_sample_path import generate_txt
data_load_base_path='./data/aux_data'
json_save_base_path='./data/aux_json'
label_set = np.loadtxt('./speechcommands_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)
target_labels = ["bird", "marvin", "wow", "visual","happy","forward", "follow", "learn", "house", "tree"]
trigger_label="bed" # We use the bed class as the induced class for example
def generate_target_class_json(data_load_base_path,json_save_base_path,target_label):
    base1_path=f'{data_load_base_path}/1_(c+t)*_a'
    a_t_c_txt_path=f'{data_load_base_path}/1_(c+t)*_a/{target_label}/train_list.txt'
    total_wav_list = []
    with open(a_t_c_txt_path, 'r') as f:
        filelist = f.readlines()
    for file in filelist:
        cur_label = label_map[target_label]
        cur_path=os.path.join(os.path.abspath(os.getcwd()),base1_path,target_label,file.strip())
        cur_dict = {"wav": cur_path, "labels": '/m/spcmd'+cur_label.zfill(2)}
        total_wav_list.append(cur_dict)
    random.shuffle(total_wav_list)
    train_size = int(0.8 * len(total_wav_list))
    train_data = total_wav_list[:train_size]
    val_data = total_wav_list[train_size:]
    path=f'{json_save_base_path}/{target_label}/1_a+c+t_A'
    os.makedirs(path,exist_ok=True)
    with open(f'{path}/1_speechcommand_train_data.json', 'w') as f:
        json.dump({'data': train_data}, f, indent=1)
    with open(f'{path}/1_speechcommand_val_data.json', 'w') as f:
        json.dump({'data': val_data}, f, indent=1)
    c_t_txt_path=f'{data_load_base_path}/2_c+t/{trigger_label}/train_list.txt'
    base2_path=f'{data_load_base_path}/2_c+t/'
    total_wav_list = []
    with open(c_t_txt_path, 'r') as f:
        filelist = f.readlines()
    for file in filelist:
        cur_label = label_map[target_label]
        cur_path=os.path.join(os.path.abspath(os.getcwd()),base2_path,trigger_label,file.strip())
        cur_dict = {"wav": cur_path, "labels": '/m/spcmd'+cur_label.zfill(2)}
        total_wav_list.append(cur_dict)
    random.shuffle(total_wav_list)
    split_index1 = int(0.7 * len(total_wav_list))
    split_index2 = int(0.8 * len(total_wav_list))
    train_data = total_wav_list[:split_index1]
    val_data = total_wav_list[split_index1:split_index2]
    eval_data = total_wav_list[split_index2:]
    path=f'{json_save_base_path}/{target_label}/2_c+t_A'
    os.makedirs(path,exist_ok=True)
    with open(f'{path}/2_speechcommand_train_data.json', 'w') as f:
        json.dump({'data': train_data}, f, indent=1)
    with open(f'{path}/2_speechcommand_val_data.json', 'w') as f:
        json.dump({'data': val_data}, f, indent=1)
    with open(f'{path}/2_speechcommand_eval_data.json', 'w') as f:
        json.dump({'data': eval_data}, f, indent=1)
    print(f'{target_label} Speechcommands dataset processing finished.')
def main():
    generate_txt(data_load_base_path)
    for target_label in target_labels:
        generate_target_class_json(data_load_base_path,json_save_base_path,target_label)
if __name__ == '__main__':
    main()
