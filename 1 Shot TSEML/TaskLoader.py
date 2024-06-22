import os, random, torch, copy
import pandas as pd

class TaskLoader:

    def __init__(self, path, fold_number, way, support_shot, train_query_shot, test_query_shot, device):

        self.way = way
        self.support_shot = support_shot
        self.train_query_shot = train_query_shot
        self.test_query_shot = test_query_shot
        self.device = device

        self.cancer_name_to_start_index_and_length = dict()
        self.cancer_name_to_subtype_name_to_start_index_and_length = dict()

        dataframe_list = list()

        cancer_name_list = os.listdir(path)
        start_index = 0

        for cancer_name in cancer_name_list:

            full_name = os.path.join(path, cancer_name)

            if os.path.isfile(full_name):
                file = full_name
                dataframe = pd.read_csv(file, sep='\t', index_col=0).transpose()

                self.cancer_name_to_start_index_and_length[cancer_name] = (start_index, len(dataframe))
                start_index += len(dataframe)
                dataframe_list.append(dataframe)

            else:
                directory = full_name
                subtype_name_to_start_index_and_length = dict()

                subtype_name_list = os.listdir(directory)

                for subtype_name in subtype_name_list:

                    full_name = os.path.join(directory, subtype_name)

                    if os.path.isfile(full_name):
                        file = full_name
                        dataframe = pd.read_csv(file, sep='\t', index_col=0).transpose()

                        subtype_name_to_start_index_and_length[subtype_name] = (start_index, len(dataframe))
                        start_index += len(dataframe)
                        dataframe_list.append(dataframe)

                self.cancer_name_to_subtype_name_to_start_index_and_length[cancer_name] = subtype_name_to_start_index_and_length

        dataframe = pd.concat(dataframe_list)

        self.gene_name_list = list(dataframe.columns)

        samples = torch.from_numpy(dataframe.values)
        self.samples = torch.cat([samples, torch.zeros(len(samples), 10)], dim=1).float().to(device)

        cancer_without_subtype_name_list = list(self.cancer_name_to_start_index_and_length.keys())
        cancer_with_subtype_name_list = list(self.cancer_name_to_subtype_name_to_start_index_and_length.keys())

        random.shuffle(cancer_without_subtype_name_list)
        random.shuffle(cancer_with_subtype_name_list)

        self.cancer_without_subtype_name_list_list = [list() for index in range(fold_number)]

        for index in range(len(cancer_without_subtype_name_list)):

            cancer_name = cancer_without_subtype_name_list[index]

            self.cancer_without_subtype_name_list_list[index % fold_number].append(cancer_name)

        self.cancer_with_subtype_name_list_list = [list() for index in range(fold_number)]

        for index in range(len(cancer_with_subtype_name_list)):

            cancer_name = cancer_with_subtype_name_list[index]
            
            self.cancer_with_subtype_name_list_list[index % fold_number].append(cancer_name)

        self.cancer_with_subtype_name_list_list.reverse()

    def set_train_and_test_sets(self, current_fold_number):

        self.train_cancer_name_to_start_index_and_length = copy.deepcopy(self.cancer_name_to_start_index_and_length)
        self.test_cancer_name_to_start_index_and_length = dict()

        cancer_name_list = self.cancer_without_subtype_name_list_list[current_fold_number]

        for cancer_name in cancer_name_list:

            self.test_cancer_name_to_start_index_and_length[cancer_name] = self.train_cancer_name_to_start_index_and_length.pop(cancer_name)

        self.train_cancer_name_to_subtype_name_to_start_index_and_length = copy.deepcopy(self.cancer_name_to_subtype_name_to_start_index_and_length)
        self.test_cancer_name_to_subtype_name_to_start_index_and_length = dict()

        cancer_name_list = self.cancer_with_subtype_name_list_list[current_fold_number]

        for cancer_name in cancer_name_list:

            self.test_cancer_name_to_subtype_name_to_start_index_and_length[cancer_name] = self.train_cancer_name_to_subtype_name_to_start_index_and_length.pop(cancer_name)

    def get_test_cancer_with_subtype_name_list(self):

        return list(self.test_cancer_name_to_subtype_name_to_start_index_and_length.keys())

    def get_test_cancer_without_subtype_name_list(self):

        return list(self.test_cancer_name_to_start_index_and_length.keys())

    def get_task(self, train_flag, test_cancer_with_subtype_name=None):

        support_shot = self.support_shot

        support_sample_index_list = list()
        query_sample_index_list = list()

        if train_flag:
            cancer_name_to_start_index_and_length = self.train_cancer_name_to_start_index_and_length
            cancer_name_to_subtype_name_to_start_index_and_length = self.train_cancer_name_to_subtype_name_to_start_index_and_length
            
            current_way = self.way
            query_shot = self.train_query_shot
            total_shot = support_shot + query_shot

            selected_number = random.sample(range(2), 1)[0]

            if selected_number == 0:#cancer classification
                cancer_name_list = list(cancer_name_to_start_index_and_length.keys()) + list(cancer_name_to_subtype_name_to_start_index_and_length.keys())

                selected_cancer_name_list = random.sample(cancer_name_list, current_way)

                for cancer_name in selected_cancer_name_list:

                    if cancer_name_to_start_index_and_length.__contains__(cancer_name):
                        (start_index, length) = cancer_name_to_start_index_and_length[cancer_name]
                        alternative_index_list = range(start_index, start_index + length)

                    else:
                        subtype_name_to_start_index_and_length = cancer_name_to_subtype_name_to_start_index_and_length[cancer_name]

                        alternative_index_list = list()

                        for subtype_name in subtype_name_to_start_index_and_length.keys():

                            (start_index, length) = subtype_name_to_start_index_and_length[subtype_name]

                            alternative_index_list.extend(range(start_index, start_index + length))

                    selected_sample_index_list = random.sample(alternative_index_list, total_shot)

                    support_sample_index_list.extend(selected_sample_index_list[: support_shot])
                    query_sample_index_list.extend(selected_sample_index_list[support_shot: ])

            else:#subtype classification
                alternative_cancer_name_list = list(cancer_name_to_subtype_name_to_start_index_and_length.keys())

                selected_sample_index_list_list = list()

                while len(selected_sample_index_list_list) < current_way:

                    selected_cancer_name = random.sample(alternative_cancer_name_list, 1)[0]

                    alternative_cancer_name_list.remove(selected_cancer_name)

                    subtype_name_to_start_index_and_length = cancer_name_to_subtype_name_to_start_index_and_length[selected_cancer_name]
                    subtype_name_list = list(subtype_name_to_start_index_and_length.keys())

                    available_subtype_number = len(subtype_name_list)
                    wanted_subtype_number = current_way - len(selected_sample_index_list_list)

                    if available_subtype_number >= wanted_subtype_number:
                        selected_subtype_name_list = random.sample(subtype_name_list, wanted_subtype_number)

                    else:
                        selected_subtype_name_list = random.sample(subtype_name_list, available_subtype_number)

                    for subtype_name in selected_subtype_name_list:

                        (start_index, length) = subtype_name_to_start_index_and_length[subtype_name]
                        alternative_index_list = range(start_index, start_index + length)
                        selected_sample_index_list = random.sample(alternative_index_list, total_shot)
                        
                        selected_sample_index_list_list.append(selected_sample_index_list)

                random.shuffle(selected_sample_index_list_list)

                for selected_sample_index_list in selected_sample_index_list_list:

                    support_sample_index_list.extend(selected_sample_index_list[: support_shot])
                    query_sample_index_list.extend(selected_sample_index_list[support_shot: ])
        
        else:
            cancer_name_to_subtype_name_to_start_index_and_length = self.test_cancer_name_to_subtype_name_to_start_index_and_length

            query_shot = self.test_query_shot
            total_shot = support_shot + query_shot

            if test_cancer_with_subtype_name is None:
                cancer_name_to_start_index_and_length = self.test_cancer_name_to_start_index_and_length
                cancer_name_list = list(cancer_name_to_subtype_name_to_start_index_and_length.keys()) + list(cancer_name_to_start_index_and_length.keys())
                
                current_way = len(cancer_name_list)
                
                selected_cancer_name_list = random.sample(cancer_name_list, current_way)

                test_global_label_index_list = [cancer_name_list.index(cancer_name) for cancer_name in selected_cancer_name_list]

                for cancer_name in selected_cancer_name_list:

                    if cancer_name_to_start_index_and_length.__contains__(cancer_name):
                        (start_index, length) = cancer_name_to_start_index_and_length[cancer_name]
                        alternative_index_list = range(start_index, start_index + length)

                    else:
                        subtype_name_to_start_index_and_length = cancer_name_to_subtype_name_to_start_index_and_length[cancer_name]
                        alternative_index_list = list()

                        for subtype_name in subtype_name_to_start_index_and_length.keys():

                            (start_index, length) = subtype_name_to_start_index_and_length[subtype_name]
                            alternative_index_list.extend(range(start_index, start_index + length))

                    selected_sample_index_list = random.sample(alternative_index_list, total_shot)
                    
                    support_sample_index_list.extend(selected_sample_index_list[: support_shot])
                    query_sample_index_list.extend(selected_sample_index_list[support_shot: ])

            else:
                subtype_name_to_start_index_and_length = cancer_name_to_subtype_name_to_start_index_and_length[test_cancer_with_subtype_name]
                subtype_name_list = list(subtype_name_to_start_index_and_length.keys())

                current_way = len(subtype_name_list)

                selected_subtype_name_list = random.sample(subtype_name_list, current_way)

                test_global_label_index_list = [subtype_name_list.index(subtype_name) for subtype_name in selected_subtype_name_list]

                for subtype_name in selected_subtype_name_list:

                    (start_index, length) = subtype_name_to_start_index_and_length[subtype_name]
                    alternative_index_list = range(start_index, start_index + length)
                    selected_sample_index_list = random.sample(alternative_index_list, total_shot)

                    support_sample_index_list.extend(selected_sample_index_list[: support_shot])
                    query_sample_index_list.extend(selected_sample_index_list[support_shot: ])

        support_label_list = list()
        query_label_list = list()

        for label in range(current_way):

            support_label_list.extend([label] * support_shot)
            query_label_list.extend([label] * query_shot)

        device = self.device

        support_sample_indeces = torch.tensor(support_sample_index_list).to(device)
        query_sample_indeces = torch.tensor(query_sample_index_list).to(device)

        support_labels = torch.tensor(support_label_list).to(device)
        query_labels= torch.tensor(query_label_list).to(device)

        shuffle_index = random.sample(range(current_way * support_shot), current_way * support_shot)
        support_sample_indeces = support_sample_indeces[shuffle_index]
        support_labels = support_labels[shuffle_index]

        shuffle_index = random.sample(range(current_way * query_shot), current_way * query_shot)
        query_sample_indeces = query_sample_indeces[shuffle_index]
        query_labels = query_labels[shuffle_index]

        samples = self.samples

        support_samples = samples[support_sample_indeces]
        query_samples = samples[query_sample_indeces]

        if train_flag:
            return (support_samples, query_samples, support_labels, query_labels)

        else:
            test_global_label_indeces = torch.tensor(test_global_label_index_list).to(device)

            return (support_samples, query_samples, support_labels, query_labels, support_sample_indeces, query_sample_indeces, test_global_label_indeces)

    def get_test_subtype_name_list(self, test_cancer_with_subtype_name):

        return list(self.test_cancer_name_to_subtype_name_to_start_index_and_length[test_cancer_with_subtype_name].keys())

    def get_samples(self):

        return self.samples

    def get_train_cancer_name_to_start_index_and_length(self):

        return self.train_cancer_name_to_start_index_and_length

    def get_train_cancer_name_to_subtype_name_to_start_index_and_length(self):

        return self.train_cancer_name_to_subtype_name_to_start_index_and_length

    def get_gene_name_list(self):

        return self.gene_name_list

    def get_cancer_name_to_start_index_and_length(self):

        return self.cancer_name_to_start_index_and_length

    def get_cancer_name_to_subtype_name_to_start_index_and_length(self):

        return self.cancer_name_to_subtype_name_to_start_index_and_length