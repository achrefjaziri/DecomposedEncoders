'''

This script can be used to generate a dataset of Random dot kinetogram stimuli
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
import random
import os


#define csv file
#define where to save dataset
#format is coherence/nb
#in csv we save whetehr train val or test and we indicate coherence level
#seq len is 20 frames
def get_label(dir_x,dir_y):
    if dir_y == -1 and dir_x == 0:

        label = 0

    elif dir_y == -1 and dir_x == 1:

        label = 1


    elif dir_y == 0 and dir_x == 1:

        label = 2

    elif dir_y == 1 and dir_x == 1:

        label = 3

    elif dir_y == 1 and dir_x == 0:

        label = 4


    elif dir_y == 1 and dir_x == -1:

        label = 5

    elif dir_y == 0 and dir_x == -1:

        label = 6

    elif dir_y == -1 and dir_x == -1:

        label = 7
    return label

def generate_rdk(dir_path,coherance,number_points=1000,seq_length=16):



    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    coherent_points = int(number_points * coherance)

    def sample(center, radius, n_per_sphere):
        r = radius
        ndim = center.size
        x = np.random.normal(size=(n_per_sphere, ndim))
        ssq = np.sum(x ** 2, axis=1)
        fr = r * gammainc(ndim / 2, ssq / 2) ** (1 / ndim) / np.sqrt(ssq)
        frtiled = np.tile(fr.reshape(n_per_sphere, 1), (1, ndim))
        p = center + np.multiply(x, frtiled)
        return p

    center = np.array([0, 0])
    radius = 1
    p = sample(center, radius, number_points)

    dir_x = 0
    dir_y = 0

    while dir_x == 0 and dir_y == 0:
        dir_x = random.randint(-1, 1)
        dir_y = random.randint(-1, 1)


    a = np.arange(number_points)
    np.random.shuffle(a)
    motion_idx = a[:coherent_points]
    random_motion_idx = a[coherent_points:]
    random_motion_points = p[random_motion_idx]

    coherent_motion_points = p[motion_idx]

    plt.style.use('dark_background')
    label = get_label(dir_x,dir_y)

    color = random.choice(['green','red','blue','white','yellow','crimson','navy'])

    for i in range(seq_length):
        random_additions = 0.05 * np.random.randn(len(random_motion_idx), 2)

        random_motion_points = random_motion_points + random_additions

        coherent_motion_points[:, 0] = coherent_motion_points[:, 0] + 0.03 * dir_x
        coherent_motion_points[:, 1] = coherent_motion_points[:, 1] + 0.03 * dir_y

        plt.figure(figsize=(5, 5))

        center = np.array([0, 0])
        radius = 1
        ax = plt.gca()
        ax.add_artist(plt.Circle(center, radius + 0.5, fill=False, color='0.5'))

        plt.scatter(random_motion_points[:, 0], random_motion_points[:, 1], marker='v', s=10, color=color)
        plt.scatter(coherent_motion_points[:, 0], coherent_motion_points[:, 1], marker='v', s=10, color=color)

        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.axis('off')
        plt.savefig(os.path.join(dir_path,f'{i}.png'), bbox_inches='tight')
        plt.close()
    return label


if __name__=='__main__':
    d = {'Seq_path': [], 'label': [], 'coherence': [],'split':[]}

    df = pd.DataFrame(data=d)
    coherence_levels = [1.0,0.8,0.6,0.5,0.25]
    parent_path = '/home/ajaziri/Thesis_work/src/vision/main/data/'


    for c_level in coherence_levels:
        for idx in range(2000):
            print('Train Split',idx,c_level)
            split = 'train'
            dir_path = os.path.join(parent_path,'rdk',f'coherence_level{c_level}',split,f'seq{idx}')
            label = generate_rdk(dir_path,c_level)
            df = df.append({'Seq_path': dir_path, 'label': label, 'coherence': c_level,'split':split},
                           ignore_index=True)

        for idx in range(200):
            print('Validation Split',idx,c_level)

            split = 'val'
            dir_path = os.path.join(parent_path,'rdk',f'coherence_level{c_level}',split,f'seq{idx}')
            label = generate_rdk(dir_path,c_level)
            df = df.append({'Seq_path': dir_path, 'label': label, 'coherence': c_level,'split':split},
                           ignore_index=True)

        for idx in range(400):
            print('Test Split',idx,c_level)

            split = 'test'
            dir_path = os.path.join(parent_path,'rdk',f'coherence_level{c_level}',split,f'seq{idx}')
            label = generate_rdk(dir_path,c_level)
            df = df.append({'Seq_path': dir_path, 'label': label, 'coherence': c_level,'split':split},
                           ignore_index=True)


    df.to_csv(os.path.join(os.path.join(parent_path,'rdk'), 'rdk_labels.csv'), sep='\t',
              encoding='utf-8')









