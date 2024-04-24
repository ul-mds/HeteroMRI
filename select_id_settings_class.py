import numpy as np
import pandas as pd

class select_id_settings_class:

	def __init__(self, name_settings_file="Exp_Settings.xlsx",select_settings=['A'],
				 settings_using_protocols_only=['C','D'],name_list_mri_temp_file="All_MRIs_List_paths_temp.csv"):
		#print("init_start")#,'C','D','E','F','G','H','I','J'    #
		self.settings_file_MS_1 = pd.read_excel(open(name_settings_file, 'rb'),
										   sheet_name='MS_1')
		self.settings_file_MS_2 = pd.read_excel(open(name_settings_file, 'rb'),
										   sheet_name='MS_2')
		#self.settings_file_Cancer_1 = pd.read_excel(open(name_settings_file, 'rb'),
		#									   sheet_name='Cancer_1')
		#self.settings_file_Cancer_2 = pd.read_excel(open(name_settings_file, 'rb'),
		#									   sheet_name='Cancer_2')
		read_columns_list_mri = ["ID", "Dataset", "Research_group", "Subject_ID", "Protocol_Group", "Folder_Path",
								 "Selected_Cluster", "Path_Selected_Cluster_File","Label"]
		list_mri_all = pd.read_csv(name_list_mri_temp_file, sep='\t')
		self.list_mri = list_mri_all[read_columns_list_mri]
		self.select_id=select_settings
		self.settings_using_protocols =settings_using_protocols_only
		#print("init_end")
	def decode_setting(self,title, setting):
		setting_decoded = []
		for index in range(len(title[0])):
			if title[0][index] == "Name":
				# for indexj in range(index,index+3):
				row = []
				for indexk in range(len(setting)):
					if str(setting[indexk][index]) != 'nan':
						row.append([setting[indexk][index], setting[indexk][index + 1], setting[indexk][index + 2],
									setting[indexk][index + 3], setting[indexk][index + 4],
									setting[indexk][index + 5]])
				if row != []:
					setting_decoded.append(row)

		return setting_decoded

	def read_settings(self,settings_file):
		settings_file_list = settings_file.values.tolist()
		titles = []
		all_settings = []
		for index in range(0, 2):
			titles.append(settings_file_list[index])
		all_settings.append(titles)
		a_setting = []
		for index in range(2, len(settings_file_list)):
			if str(settings_file_list[index][0]) != 'nan':
				a_setting.append(settings_file_list[index])
			else:
				if a_setting != []:
					all_settings.append(a_setting)
				a_setting = []
		if a_setting != []:
			all_settings.append(a_setting)
		return all_settings

	def filter_mri_list(self,list_mri, name_data,label, setting_is_protocol):
		if setting_is_protocol:
			data_filtered = list_mri.loc[list_mri["Protocol_Group"] == name_data]
			data_filtered = data_filtered.loc[list_mri['Label'] == label]
		else:
			data_filtered = list_mri.loc[list_mri['Dataset'] == name_data]
			data_filtered = data_filtered.loc[list_mri['Label'] == label]

		return data_filtered

	def divide_protocols(self,list_mrt):
		list_protocols = list_mrt['Protocol_Group']
		list_protocols_unique = list_protocols.unique().tolist()
		list_protocols_selected = []
		for protocol in list_protocols_unique:
			list_protocols_selected.append(list_mrt.loc[list_mrt['Protocol_Group'] == protocol])
		return list_protocols_selected

	def select_mri_rows(self,list_mrt, total_no, train_no, evaluation_no, test_no):
		id_selected = []
		while len(id_selected) < total_no:
			for index in range(len(list_mrt)):
				if len(list_mrt[index]) > 0:
					# Generate random indices
					indices = np.random.choice(list_mrt[index].index, size=1, replace=False)
					# Remove rows at random
					ID = list_mrt[index].loc[indices]["ID"].to_list()[0]
					protocols = list_mrt[index].loc[indices]["Protocol_Group"].to_list()[0]
					list_mrt[index] = list_mrt[index].drop(indices)
					id_selected.append([ID, protocols])
		train_id = id_selected[0:train_no]
		evaluation_id = id_selected[train_no:train_no + evaluation_no]
		test_id = id_selected[train_no + evaluation_no:train_no + evaluation_no + test_no]
		return [train_id, evaluation_id, test_id]

	def select_mri(self,row_data_setting, list_mri, setting_is_protocol):
		name_protocol_id = row_data_setting[0]
		type_lable = row_data_setting[1]
		total_no = row_data_setting[2]
		train_no = row_data_setting[3]
		evaluation_no = row_data_setting[4]
		test_no = row_data_setting[5]
		filtered_mri_list = self.filter_mri_list(list_mri, name_protocol_id, type_lable, setting_is_protocol)
		divided_filtered_protocols_mri_list = self.divide_protocols(filtered_mri_list)
		train_id, evaluation_id, test_id = self.select_mri_rows(divided_filtered_protocols_mri_list, total_no, train_no,
														   evaluation_no, test_no)
		return [train_id, evaluation_id, test_id]

	def Uniform_removal_by_protocol_group(self,settings_selected, new_len):
		settings_selected_df = pd.DataFrame(settings_selected, columns=['ID', 'Protocol_Group'])
		divided_protocols_settings_selected_df = self.divide_protocols(settings_selected_df)
		while len(settings_selected_df) > new_len:
			for index in range(len(divided_protocols_settings_selected_df)):
				if len(divided_protocols_settings_selected_df[index]) > 0:
					indices = np.random.choice(divided_protocols_settings_selected_df[index].index, size=1,
											   replace=False)
					ID_del = divided_protocols_settings_selected_df[index].loc[indices]["ID"].to_list()[0]
					divided_protocols_settings_selected_df[index] = divided_protocols_settings_selected_df[
						index].drop(indices)
					settings_selected_df = settings_selected_df.drop(
						settings_selected_df[settings_selected_df['ID'] == ID_del].index)
					if len(settings_selected_df) <= new_len:
						break
		return settings_selected_df.values.tolist()

	def remove_rows(self,settings_decoded, row_data_setting):

		train_id = settings_decoded[0]
		evaluation_id = settings_decoded[1]
		test_id = settings_decoded[2]


		train_new_no = row_data_setting[3]
		evaluation_new_no = row_data_setting[4]
		test_new_no = row_data_setting[5]

		if len(train_id) < train_new_no and len(evaluation_id) < evaluation_new_no and len(test_id) < test_new_no:
			train_id,evaluation_id,test_id=self.select_mri(row_data_setting, self.list_mri, True)

		train_new_id = self.Uniform_removal_by_protocol_group(train_id, train_new_no)
		evaluation_new_id = self.Uniform_removal_by_protocol_group(evaluation_id, evaluation_new_no)
		test_new_id = self.Uniform_removal_by_protocol_group(test_id, test_new_no)

		return [train_new_id, evaluation_new_id, test_new_id]

	def generate_id_settings(self):
		#print("generate_id_settings_start")
		settings_file_MS_1_list = self.read_settings(self.settings_file_MS_1)
		settings_file_MS_2_list = self.read_settings(self.settings_file_MS_2)
		#settings_file_Cancer_1_list = self.read_settings(self.settings_file_Cancer_1)
		#settings_file_Cancer_2_list = self.read_settings(self.settings_file_Cancer_2)
		#all_Settings = [settings_file_MS_1_list, settings_file_MS_2_list, settings_file_Cancer_1_list,settings_file_Cancer_2_list]
		all_Settings = [settings_file_MS_1_list, settings_file_MS_2_list]
		settings_decoded = []
		settings_id = []
		for setting_item in all_Settings:
			for index in range(1, len(setting_item)):
				settings_decoded.append(self.decode_setting(setting_item[0], setting_item[index]))
				if(len(setting_item[index][0][0])==3):
					settings_id.append(setting_item[index][0][0][0])
				else:
					settings_id.append(setting_item[index][0][0][0:2])
				#print("generate_id_settings_decode"+str(index))
		settings_selected_all = []
		print(str(self.select_id))
		for alphabet in self.select_id:
			settings_row_selected = []
			for index, item in enumerate(settings_id):
				one_setting_first_row_selected = []
				if alphabet.upper() == item.upper():
					#print("found:"+str(item))
					for index_select in range(len(settings_decoded[index])):
						one_setting_first_row_selected.append(
							self.select_mri(settings_decoded[index][index_select][0], self.list_mri,
									   alphabet in self.settings_using_protocols))

					settings_row_selected.append(one_setting_first_row_selected)
					for index_remove in range(1, len(settings_decoded[index][0])):
						one_setting_row_selected = []
						for index_select in range(len(settings_decoded[index])):
							one_setting_row_selected.append(
								self.remove_rows(settings_row_selected[index_remove - 1][index_select],
											settings_decoded[index][index_select][index_remove]))
						settings_row_selected.append(one_setting_row_selected)
					settings_selected_all.append(settings_row_selected)
					break
		#print("generate_id_settings_end")
		return settings_selected_all
