# First of all, we need to generate "All_MRIs_List_paths_temp.csv" by running this file.
# Please:
#       - Update the "MRIs_List.csv" file if necessary!
#       - Set the correct path(s) in "datasets_path.csv"

# Load all the packages used in this file.
import pandas as pd
import csv
import os
import nibabel as nib
import numpy as np
import math
csvdata=[]
csvadata_file_name="MRIs_List.csv"
address_data_file_name="datasets_path.csv"
output_file="All_MRIs_List_paths_temp.csv"
template_dice = "./template_dice.nii.gz"
address_data=[]

# Remove the output file if it already exists
if os.path.exists(output_file):
  os.remove(output_file)

# Read dataset paths from CSV
with open(address_data_file_name, newline='') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
    for row in reader:
        address_data.append(row)
#DICE
def load_nifti_file(file_path):
    nifti_img = nib.load(file_path)
    img = nifti_img.get_fdata()
    img[img < 0.50] = 0
    img[img >= 0.50] = 1

    return img

segmentation1 = load_nifti_file(template_dice)
seg1 = segmentation1.flatten()
sum_seg1 = np.sum(seg1)

def dice_coefficient(segmentation2):
    seg2 = segmentation2.flatten()

    intersection = np.inner(seg1,seg2)
    sum_seg2 = np.sum(seg2)
    #dice1 = (2 * intersection ) / (sum_seg1 + sum_seg2 )
    #dice1 =(2 * intersection) / (np.linalg.norm(seg1)**2 + np.linalg.norm(seg2)**2)
    dice =(2 * intersection) / (np.inner(seg1, seg1) + (np.inner(seg2,seg2)))

    return dice
all_MRI_dataf=pd.read_csv(csvadata_file_name, sep=',')
all_MRI=all_MRI_dataf.values.tolist()
headers=all_MRI_dataf.columns.values.tolist()
Eroor_flag=False
for index in range(len(all_MRI)):
    found=True

    for indexj in range(1,len(address_data)):
        if all_MRI[index][1]==address_data[indexj][0]:
            mri_file = address_data[indexj][1] + "/" + all_MRI[index][4]
            if not os.path.isdir(mri_file):
                print("Not found path "+mri_file)
                Eroor_flag=True
                break
            if os.path.isdir(mri_file+"/"+"FLAIR.nii.gz"):
                mri_file=mri_file+"/"+"FLAIR.nii.gz"
            list_MRI = [
                os.path.join(os.getcwd(), mri_file, x)
                for x in os.listdir(mri_file)
            ]
            Not_load=True
            dice_valuse = []
            for file_path in list_MRI:
                try:
                    img = load_nifti_file(file_path)  # nib.load(file_path)
                    dice_valuse.append(dice_coefficient(img))
                except:
                    print("Bad file descriptor(warning):" + file_path)
                    Not_load = False
                    break
            if not Not_load:
                print("Cant open file! "+mri_file)
                Eroor_flag=True
            else:
                select_index = dice_valuse.index(max(dice_valuse))
                all_MRI[index][19] = mri_file
                all_MRI[index][21] = list_MRI[select_index]

            found=False
            break
print("Scan MRI files: done.")
# Exit if there was an error
if Eroor_flag:
    exit()
# Write the updated MRI data to the output file
with open(output_file,"w+") as my_csv:
    writer = csv.DictWriter(my_csv,delimiter='\t', fieldnames = headers)
    writer.writeheader()
    csvWriter = csv.writer(my_csv,delimiter='\t')
    csvWriter.writerows(all_MRI)
print("Success in "+output_file+ " generation.")
