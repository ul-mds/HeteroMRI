import tensorflow as tf
import pickle
import time
import select_id_settings_class as select_calss
import CNN_model
import io_class
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

select_id_we_want = ['A','B','C','D']
number_shuffle=range(0,20)
repeat_number = range(0,10)
load_selected_ids = "none" #python"output_no_lesion_no_ADNI0/2023-10-23-12-23-14.pkl"#"none"#2023-10-11-18-10-04.pkl" #or
output_path = "/beegfs/ws/0/nash272e-New_runs_MDS/CodeDgh/"
initial_learning_rate = 0.0001
epochs = 100
batch_size = 2
decay_steps_model=100000
decay_rate_model=0.96
settings_using_protocols_only=['C','Ca','Cb','Cc','Cd','Ce','Cf','Cg','Ch','Ci','Cj','Ck','Cl','Cm','Cn','Co','Cp','Cq','Cr','Cs','Ct','Cu','Cv','Cw','Cx','Cy','Da','Db','Dc','Dd','De','Df','Dg','Dh','Di','Dj','Ea','Eb','F']
#####################################################################################################################################
#####################################################################################################################################
def main(select_id_we_want,repeat_number,load_selected_ids,output_path,
		 initial_learning_rate,epochs,batch_size,decay_steps_model,decay_rate_model):
	time_info=io_class.time_info_class()
	io_manager = io_class.io_calss(output_path)
	io_manager.make_dir(output_path,None,name_dir_add=False)
	if load_selected_ids.lower() =="none":
		time_info.start_record_time()
		select_id_settings=select_calss.select_id_settings_class(select_settings=select_id_we_want,
																 settings_using_protocols_only= settings_using_protocols_only,)
		settings_selected_all=select_id_settings.generate_id_settings()
		name_selected_settings_file = "Selected_Ids_Setting_"+select_id_we_want[0]+"_"+time.strftime("%Y-%m-%d-%H-%M-%S.pkl")
		with open(output_path+"/"+name_selected_settings_file, 'wb') as f:
			pickle.dump([settings_selected_all,select_id_we_want], f)
		with open(output_path+"/"+name_selected_settings_file, 'rb') as f:
			settings_selected_all,select_id_we_want = pickle.load(f)
		time_info.add_log(time_info.stop_record_time(),"Selecting the MRI IDs","Saved in "+name_selected_settings_file)
	else:
		time_info.start_record_time()
		with open(load_selected_ids, 'rb') as f:
			settings_selected_all,select_id_we_want = pickle.load(f)
		time_info.add_log(time_info.stop_record_time(), "loading the file","the file was "+load_selected_ids)
	for index in range(len(settings_selected_all)):
		for indexi in range(len(settings_selected_all[index])):
			aggregation_temp = settings_selected_all[index][indexi][0]
			for indexj in range(1,len(settings_selected_all[index][indexi])):
				aggregation_temp[0]+=settings_selected_all[index][indexi][indexj][0]
				aggregation_temp[1]+=settings_selected_all[index][indexi][indexj][1]
				aggregation_temp[2]+=settings_selected_all[index][indexi][indexj][2]
			settings_selected_all[index][indexi]=aggregation_temp
			#print(len(aggregation_temp[0]) + len(aggregation_temp[1]) + len(aggregation_temp[2]))


	settings_selected_all_paths=io_manager.load_path_mris(settings_selected_all)

	CNN=CNN_model.CNN_class(initial_learning_rate_model= initial_learning_rate ,epochs_model= epochs,batch_size_model= batch_size,
							decay_steps_model= decay_steps_model,decay_rate_model= decay_rate_model)
	for index in range(len(settings_selected_all_paths)):
		for indexi in range(len(settings_selected_all_paths[index])):
			train_mri_paths=[row[2] for row in settings_selected_all_paths[index][indexi][0]]
			evaluation_mri_paths=[row[2] for row in settings_selected_all_paths[index][indexi][1]]
			test_mri_paths=[row[2] for row in settings_selected_all_paths[index][indexi][2]]

			train_mri_id=[row[0] for row in settings_selected_all_paths[index][indexi][0]]
			evaluation_mri_id=[row[0] for row in settings_selected_all_paths[index][indexi][1]]
			test_mri_id=[row[0] for row in settings_selected_all_paths[index][indexi][2]]

			train_mri_paths_label=[row[3] for row in settings_selected_all_paths[index][indexi][0]]
			evaluation_mri_paths_label=[row[3] for row in settings_selected_all_paths[index][indexi][1]]
			test_mri_paths_label=[row[3] for row in settings_selected_all_paths[index][indexi][2]]
			CNN.load_MRI_files([train_mri_paths,train_mri_paths_label],[evaluation_mri_paths,evaluation_mri_paths_label],
							   [test_mri_paths,test_mri_paths_label,test_mri_id])
			for index_run in repeat_number:
				time_info.start_record_time()
				path_save=select_id_we_want[index]+"/"+select_id_we_want[index]+format(indexi, '02d')+"/run"+format(index_run+1, '02d')
				io_manager.make_dir(output_path,path_save)
				path_model_save="/Model.h5"
				print("###########################Training (Run"+str(index_run+1)+")###########################")
				history_train_log=CNN.train_model(output_path+"/"+path_save+path_model_save)
				print("###########################Testing (Run"+str(index_run+1)+")###########################")
				results_valid,results=CNN.test_model(output_path+"/"+path_save+path_model_save)
				time_info.add_log(time_info.stop_record_time(),
								  "Runtime for Setting " + select_id_we_want[index] + " (Run" + format(index_run+1, '02d')+")",
								  "Path: " + path_save)
				#time_info.add_log(time_info.get_time_from_beginning(),
				#				  "Runtime from the beginning",
				#				  "For the first run, the runtime is more than others because the selecting or the loading parts!")
				print("###########################Saving results (Run"+str(index_run+1)+")###########################")
				log_path = output_path+"/"+path_save
				if output_path== "./":
					log_path = output_path+path_save
				io_manager.write_results(results_valid,results,[train_mri_id, evaluation_mri_id, test_mri_id],
										 time_info.get_log(), time_info.get_system_info(),log_path,history_train_log)
				time_info.remove_log()
				time_info.reset_time_from_beginning()



if __name__ == "__main__":

	for index in number_shuffle:
		output_path_updated= output_path+"Shuffle"+str(index+1)
		print("###########################Setting "+ select_id_we_want[0] + ", Shuffle: "+ str(index+1))
		main(select_id_we_want,repeat_number,load_selected_ids,output_path_updated,initial_learning_rate,epochs,batch_size,decay_steps_model,decay_rate_model)
