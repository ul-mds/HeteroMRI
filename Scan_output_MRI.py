import pandas as pd
import csv
import os
import nibabel as nib
import math
csvdata=[]
csvadata_file_name="All_MRIs_List_No_Lesions_plus_France.csv"
address_data_file_name="address_output.csv"
output_file="All_MRIs_List_paths_temp.csv"
address_data=[]

if os.path.exists(output_file):
  os.remove(output_file)
with open(address_data_file_name, newline='') as f:
    reader = csv.reader(f, delimiter=',', quoting=csv.QUOTE_NONE)
    for row in reader:
        address_data.append(row)

all_MRI_dataf=pd.read_csv(csvadata_file_name, sep='\t')
all_MRI=all_MRI_dataf.values.tolist()
headers=all_MRI_dataf.columns.values.tolist()
Eroor_flag=False
for index in range(len(all_MRI)):
    found=True
    if all_MRI[index][19]=="nan" or math.isnan((all_MRI[index][19])):
        continue
    for indexj in range(1,len(address_data)):
        if all_MRI[index][1]==address_data[indexj][0]:
            mri_file = address_data[indexj][1] + "/" + all_MRI[index][3]
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
            for file_path in list_MRI:
                if file_path[len(file_path)-8]==str(int(all_MRI[index][18])):
                    try:
                        img = nib.load(file_path)
                        all_MRI[index][18]=mri_file
                        all_MRI[index][21]=file_path
                    except:
                        print("Bad file descriptor(warning):"+file_path)
                    Not_load=False
                    break
            if Not_load:
                print("Cant open file! "+mri_file)
                Eroor_flag=True

            found=False
            break

if Eroor_flag:
    exit()
with open(output_file,"w+") as my_csv:
    writer = csv.DictWriter(my_csv,delimiter='\t', fieldnames = headers)
    writer.writeheader()
    csvWriter = csv.writer(my_csv,delimiter='\t')
    csvWriter.writerows(all_MRI)
print("Scan MRI files: done")
