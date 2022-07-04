import os
import shutil
data_root = 'mmaction2/data/MMDS/MMDS-VIDEO'
ann_file_train = 'mmaction2/data/MMDS/annotations/mmds_cls_train_video_list.txt'
ann_file_test = 'mmaction2/data/MMDS/annotations/mmds_cls_val_video_list.txt'


output_ann_dir = 'data/demo/annotations'
output_video_dir = 'data/demo/videos'

count = [0 for _ in range(240)]
file_input = open(ann_file_train, 'r')
file_output = open(os.path.join(output_ann_dir,'train_list.txt'), 'w')
for line in file_input:
    content = line.strip().split()
    path = content[0]
    label = int(content[1])
    if label > 24:
        break
    if count[label] < 6:
        count[label] += 1
        name = os.path.basename(path)
        file_output.write(' '.join([name,str(label)])+'\n')
        shutil.copy(os.path.join(data_root, content[0]), os.path.join(output_video_dir,name))
file_input.close()
file_output.close()

count = [0 for _ in range(240)]
file_input = open(ann_file_test, 'r')
file_output = open(os.path.join(output_ann_dir,'val_list.txt'), 'w')
for line in file_input:
    content = line.strip().split()
    path = content[0]
    label = int(content[1])
    if label > 24:
        break
    if count[label] < 2:
        count[label] += 1
        name = os.path.basename(path)
        file_output.write(' '.join([name,str(label)])+'\n')
        shutil.copy(os.path.join(data_root, content[0]), os.path.join(output_video_dir,name))
file_input.close()
file_output.close()

count = [0 for _ in range(240)]
file_input = open(ann_file_test, 'r')
file_output = open(os.path.join(output_ann_dir,'test_list.txt'), 'w')
for line in file_input:
    content = line.strip().split()
    path = content[0]
    label = int(content[1])
    if label > 24:
        break
    if count[label] < 2:
        count[label] += 1
        name = os.path.basename(path)
        file_output.write(' '.join([name,str(label)])+'\n')
        shutil.copy(os.path.join(data_root, content[0]), os.path.join(output_video_dir,name))
file_input.close()
file_output.close()
