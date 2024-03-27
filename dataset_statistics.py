from ns_vfs.data.frame import BenchmarkLTLFrame
import pickle
import sys
import os

save_path = 'store/dictionaries'
# directories = ['store/nsvs_artifact/_validated_benchmark_video', 'store/nsvs_artifact/_validated_nuscene_video']
directories = ['store/nsvs_artifact/_validated_waymo_video']
dataset_count = {
        'coco': {
            'total': 0,
        },
        'imagenet': {
            'total': 0,
        },
        'nuscene': {
            'total': 0,
        },
        'waymo': {
            'total': 0,
        },

}
total = 0

def parseFile(path):
    global total
    print(path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
        number = data.number_of_frame
        total += number

        directories = path.split(os.path.sep)
        for k in dataset_count.keys():
            if k in directories:
                dataset_count[k]['total'] += number
                i = directories.index(k)
                if directories[i+1] in dataset_count[k]:
                    dataset_count[k][directories[i+1]] += number
                else:
                    dataset_count[k][directories[i+1]] = number
                break
        print(dataset_count)
    print()

def createDictionaries(path):
    print(path)
    with open(path, 'rb') as f:
        data = pickle.load(f)
        frames_unformatted = data.frames_of_interest
        frames_formatted = []
        for i in frames_unformatted:
            if isinstance(i, list):
                frames_formatted.append(i)
            else:
                print(frames_unformatted)
                frames_formatted.append([i])
        dict_format = dict(
            ground_truth = data.ground_truth,
            ltl_formula = data.ltl_formula,
            proposition = data.proposition,
            nubmer_of_frame = data.number_of_frame,
            frames_of_interest = data.frames_of_interest,
            labels_of_frames = data.labels_of_frames,
            images_of_frames = data.images_of_frames
        )
        # with open(os.path.join(save_path, path.split(os.path.sep)[-1]), 'wb') as wf:
        #     pickle.dump(dict_format, wf)

def findFiles(root, command):
    for path, subdirs, files in os.walk(root):
        for name in files:
            if command == 'parse':
                parseFile(os.path.join(path, name))
            elif command == 'dictionary':
                createDictionaries(os.path.join(path, name))


# for i in range(len(directories)):
#     findFiles(directories[i], 'parse')
# print(f'total: {total}')

for i in range(len(directories)):
    findFiles(directories[i], 'dictionary')

