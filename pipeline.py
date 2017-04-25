import numpy as np
import subprocess as sp
from nilearn.image import load_img
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import os

base_dir = '%s/schizophrenia_data/COBRE' % os.environ['HOME']

os.chdir(base_dir)  


# Load the bet parameters from the csv file.
sub_info = np.recfromcsv('subjects_info.csv', delimiter=',',
                     converters={'sub': lambda x: str(x)})

n_subjects = len(sub_info['sub'])



# Commands templates.
bet_tmp = 'bet %s/session_1/anat_1/mprage.nii.gz %s/session_1/anat_1/mprage_brain.nii.gz -R -f %.3f -g %.3f'

flirt_tmp = 'flirt -in %s/session_1/anat_1/mprage_brain.nii.gz -ref /usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain -out %s/session_1/anat_1/mprage_brain_reg.nii.gz -omat %s/session_1/anat_1/mprage_brain_reg.mat -bins 256 -cost corratio -searchrx -90 90 -searchry -90 90 -searchrz -90 90 -dof 12  -interp nearestneighbour'


# Variables for classification.
brain_tmp = '%s/session_1/anat_1/mprage_brain_reg.nii.gz'

mask_fpath = '/usr/share/data/fsl-mni152-templates/MNI152_T1_2mm_brain_mask.nii.gz'


# ###############################################################################
# # BET and FLIRT

# # Iterate trough all subjects.
# for (i, sub) in enumerate(sub_info['sub']):
    # print(sub)

    # # Prepare commands.
    # bet_cmd = bet_tmp % (sub_info['sub'][i], sub_info['sub'][i],
                         # sub_info['f'][i], sub_info['g'][i])

    # flirt_cmd = flirt_tmp % ((sub_info['sub'][i], ) * 3)

    # # Perform both commands.
    # for cmd in [bet_cmd, flirt_cmd]:
        # print(cmd)
        # process = sp.Popen(cmd, stdout=sp.PIPE, shell=True)
        # output = process.communicate()[0]
        # print(output)
        # print('')


###############################################################################
# CLASSIFICATION

mask_img = load_img(mask_fpath)
mask = mask_img.get_data().astype(bool)

data = np.zeros((n_subjects, mask.sum()))
target = sub_info['schizophrenia'].astype(bool)
print('\nSubjects in total: %s' % (target.shape[0]))
print('Schizophrenia patients: %s\nControls: %s\n' % (target.sum(),
                                                      n_subjects - target.sum()))

for (i, sub) in enumerate(sub_info['sub']):
    brain_fpath = brain_tmp % sub

    print('Loading data from path: %s' % brain_fpath)

    img = load_img(brain_fpath)
    brain_data = img.get_data()

    data[i] = brain_data[mask]


from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(data.T)

components = pca.components_

import matplotlib.pyplot as plt
colors = ['red','green']
plt.scatter(components[0][target], components[1][target],
            c='g',  marker='o', s=60)
plt.scatter(components[0][~target], components[1][~target],
            c='r',  marker='^', s=60)
plt.show()

# X_train, X_test, y_train, y_test = train_test_split(data,
                                                    # target)

# cls = SVC()
# cls.fit(X_train, y_train)
# print(y_test)
# print(accuracy_score(y_test, cls.predict(X_test)))
