import os

def generate_class_txt(root_dir):
    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            current_audio_files = [f for f in os.listdir(category_path) if f.endswith('.wav')]
            with open(os.path.join(category_path, 'train_list.txt'), 'w') as train_file:
                for filename in current_audio_files:
                    poisoned_path = f"{filename}\n"
                    train_file.write(poisoned_path)
def generate_txt(base_path=None):
    # base_path='./data/aux_data'
    root_dir1=f'{base_path}/1_(c+t)*_a'
    root_dir2=f'{base_path}/2_c+t'
    generate_class_txt(root_dir1)
    generate_class_txt(root_dir2)
