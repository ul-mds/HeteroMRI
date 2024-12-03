import pandas as pd
import os
import time
import platform #pip install lib-platform
import psutil
import subprocess

class io_calss:
    def __init__(self,output_path,name_list_mri_temp_file="All_MRIs_List_paths_temp.csv"):
        self.output_path=output_path
        self.name_list_mri_file=name_list_mri_temp_file
        list_mri_all = pd.read_csv(name_list_mri_temp_file, sep='\t')
        read_columns_list_mri = ["ID", "Path_Selected_Cluster_File","Label"]
        self.list_mri = list_mri_all[read_columns_list_mri]
    def load_path_mris(self,settings_selected):

        for index in range(len(settings_selected)):
            for indexi in range(len(settings_selected[index])):
                for indexj in range(len(settings_selected[index][indexi])):
                    for indexk in range(len(settings_selected[index][indexi][indexj])):
                        settings_selected[index][indexi][indexj][indexk].append(self.list_mri.loc[self.list_mri['ID'] ==
                                                                                                  settings_selected[
                                                                                                      index][indexi][
                                                                                                      indexj][indexk][0]].values.tolist()[0][1])
                        settings_selected[index][indexi][indexj][indexk].append(self.list_mri.loc[self.list_mri['ID'] ==
                                                                                                  settings_selected[
                                                                                                      index][indexi][
                                                                                                      indexj][indexk][0]].values.tolist()[0][2])

        return settings_selected
    def make_dir(self,path, name_dir,name_dir_add=True):
        if name_dir_add:
            directory=path+"/"+name_dir
        else:
            directory = path
        if not os.path.exists(directory):
            os.makedirs(directory)
    def write_results(self, results_valid, results, inputs, log, system_info, path, history_train_log,
                      file_name="results.xlsx"):
        #df = pd.DataFrame(results_valid)
        train_id, validation_id, test_id=inputs

        headers = ["ID"]
        train_id=pd.DataFrame(train_id)
        train_id.columns = headers

        validation_id=pd.DataFrame(validation_id)
        validation_id.columns = headers

        test_id=pd.DataFrame(test_id)
        test_id.columns = headers
        full_path=path+"/"+file_name
        writer = pd.ExcelWriter(full_path, engine='xlsxwriter')
        results.to_excel(writer, sheet_name='Results', index=False)
        results_valid.to_excel(writer, sheet_name='Metrics', index=False)
        train_id.to_excel(writer, sheet_name='Training_Data', index=False)
        validation_id.to_excel(writer, sheet_name='Validation_Data', index=False)
        test_id.to_excel(writer, sheet_name='Test_Data', index=False)
        log.to_excel(writer, sheet_name='Runtime_Log', index=False)
        history_train_log.to_excel(writer, sheet_name='Training_Output_Log', index=False)
        system_info.to_excel(writer, sheet_name='System_Info', index=False)
        writer.close()

class time_info_class:
    def __init__(self):
        self.start=None
        self.start_from_beginning=time.time()
        self.log=[]
        self.index_log=1
    def start_record_time(self):
        self.start= time.time()
    def reset_time_from_beginning(self):
        self.start_from_beginning = time.time()
        self.start = None
    def get_log(self):
        headers = ["ID", "Description", "Time(s)", "Comment"]
        log_pd = pd.DataFrame(self.log)
        log_pd.columns = headers
        return log_pd
    def add_log(self,time_reg,description,memo):
        self.log.append([format(self.index_log, '04d'), description, time_reg, memo])
        self.index_log = self.index_log+1
    def stop_record_time(self):
        return str(time.time() - self.start)
    def get_time_from_beginning(self):
        return str(time.time() - self.start_from_beginning)
    def get_gpu_mdl(self):
        line_as_bytes = subprocess.check_output("nvidia-smi -L", shell=True)
        line = line_as_bytes.decode("ascii")
        _, line = line.split(":", 1)
        line, _ = line.split("(")
        return line.strip()
    def get_system_info(self):
        list_inf=[]
        list_inf.append(['Platform',platform.system()])
        list_inf.append(['Platform-release',platform.release()])
        list_inf.append(['Platform-version',platform.version()])
        list_inf.append(['Architecture',platform.machine()])
        list_inf.append(['Processor',platform.processor()])
        list_inf.append(['RAM',str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"])
        list_inf.append(['GPU',self.get_gpu_mdl()])
        for item in os.environ:
            list_inf.append([item,os.environ[item]])
        headers = ["Item","Value"]
        list_inf=pd.DataFrame(list_inf)
        list_inf.columns = headers
        return list_inf
    def remove_log(self):
        self.log=[]
        self.index_log=1



