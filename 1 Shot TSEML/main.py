import torch, random, os
import numpy as np
import torch.nn.functional as F
from torch import optim
from TaskLoader import TaskLoader
from Layers import EmbeddingModelWithoutLastReLU

def get_loss_and_accuracy(support_features, query_features, support_labels, query_labels):

    current_way = (torch.max(support_labels) + 1).item()

    device = support_features.device

    prototypes = torch.zeros(current_way, support_features.shape[1]).to(device)

    for (support_feature, support_label) in zip(support_features, support_labels):

        prototypes[support_label] += support_feature

    support_shot = len(support_features) // current_way

    prototypes /= support_shot

    distance_logit_list = list()

    for query_feature in query_features:

        x1 = query_feature.expand(current_way, -1)
        x2 = prototypes

        distance_logit = - F.pairwise_distance(x1, x2, p=2)

        distance_logit_list.append(distance_logit)

    distance_logits = torch.stack(distance_logit_list, dim=0)

    loss = F.cross_entropy(distance_logits, query_labels)

    with torch.no_grad():

        chosen_values = torch.gather(distance_logits, dim=1, index=torch.unsqueeze(query_labels, dim=1))
        max_values = torch.unsqueeze(torch.max(distance_logits, dim=1)[0], dim=1)

    correct_number = torch.sum(torch.eq(chosen_values, max_values))
    accuracy = correct_number / len(query_labels)

    return (loss, accuracy)

def get_updated_parameter_list(loss, parameter_list, learning_rate):

    gradients = torch.autograd.grad(loss, parameter_list, create_graph=True, retain_graph=True)

    updated_parameter_list = list(map(lambda x: x[0] - learning_rate * x[1], zip(parameter_list, gradients)))

    return updated_parameter_list

def get_file(path, prefix, suffix):

    file_list = os.listdir(path)

    for file in file_list:

        if file.startswith(prefix) and file.endswith(suffix):
            return os.path.join(path, file)

def get_accuracy_and_train_iteration_number(model_path, task_name, model, samples, test_support_iteration_number, task_learning_rate):

    model_file = get_file(model_path, '', f'{task_name} Embedding Model')
    test_support_total_sample_indeces_file = get_file(model_path, '', f'{task_name} Test Support Total Sample Indeces')
    test_query_total_sample_indeces_file = get_file(model_path, '', f'{task_name} Test Query Total Sample Indeces')
    test_support_total_labels_file = get_file(model_path, '', f'{task_name} Test Support Total Labels')
    test_query_total_labels_file = get_file(model_path, '', f'{task_name} Test Query Total Labels')
    
    model.load_state_dict(torch.load(model_file))
    test_support_total_sample_indeces = torch.load(test_support_total_sample_indeces_file)
    test_query_total_sample_indeces = torch.load(test_query_total_sample_indeces_file)
    test_support_total_labels = torch.load(test_support_total_labels_file)
    test_query_total_labels = torch.load(test_query_total_labels_file)

    sum_accuracy_list = [0] * (test_support_iteration_number + 1)

    for (support_sample_indeces, query_sample_indeces, support_labels, query_labels) in zip(test_support_total_sample_indeces, test_query_total_sample_indeces, test_support_total_labels, test_query_total_labels):

        support_samples = samples[support_sample_indeces]
        query_samples = samples[query_sample_indeces]

        parameter_list = list(model.net.parameters())

        for support_iteration_number in range(test_support_iteration_number):

            support_features = model(support_samples, parameter_list)

            with torch.no_grad():

                query_features = model(query_samples, parameter_list)

                (loss, accuracy) = get_loss_and_accuracy(support_features, query_features, support_labels, query_labels)
            
            sum_accuracy_list[support_iteration_number] += accuracy.item()

            (loss, accuracy) = get_loss_and_accuracy(support_features, support_features, support_labels, support_labels)

            parameter_list = get_updated_parameter_list(loss, parameter_list, task_learning_rate)

        with torch.no_grad():

            support_features = model(support_samples, parameter_list)
            query_features = model(query_samples, parameter_list)

        (loss, accuracy) = get_loss_and_accuracy(support_features, query_features, support_labels, query_labels)
        
        sum_accuracy_list[test_support_iteration_number] += accuracy.item()

    accuracy_list = [sum_accuracy / len(test_support_total_sample_indeces) for sum_accuracy in sum_accuracy_list]
    
    accuracy = max(accuracy_list)

    train_iteration_number = int(model_file.split(os.sep)[-1].split(' ')[2])

    return (accuracy, train_iteration_number)

def delete_file(file):

    if os.path.isfile(file):
        os.remove(file)

def main():

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

   
    # types_path = '/dev/sda4/chengyan/Datasets/Types-HiSeqV2_PANCAN'
    types_path = 'D:\\Datasets\\Types-HiSeqV2_PANCAN'
    fold_number = 10
    way = 5
    support_shot = 1
    train_query_shot = 15
    test_query_shot = support_shot
    batch_size = 10
    train_support_iteration_number = 5
    test_support_iteration_number = 10
    device = torch.device('cuda')

    task_loader = TaskLoader(types_path, fold_number, way, support_shot, train_query_shot, test_query_shot, device)

    task_learning_rate = 5e-2
    meta_learning_rate = 1e-4

    train_iteration_number = 1000
    print_iteration_number = 10
    test_train_iteration_number = 10
    test_iteration_number = 500

    resume_fold_number = -1
    resume_train_iteration_number = -1

    if resume_fold_number == -1:
        resume_flag = False

    else:
        resume_flag = True
    
    task_name_to_best_test_query_accuracy_list = list()
    task_name_to_best_test_query_accuracy_train_iteration_number_list = list()

    for current_fold_number in range(fold_number):

        print('current_fold_number:', current_fold_number)

        train_support_sum_loss_list = [0] * train_support_iteration_number
        train_support_sum_accuracy_list = [0] * train_support_iteration_number

        train_query_sum_loss_list = [0] * (train_support_iteration_number + 1)
        train_query_sum_accuracy_list = [0] * (train_support_iteration_number + 1)

        task_loader.set_train_and_test_sets(current_fold_number)

        test_cancer_with_subtype_name_list = task_loader.get_test_cancer_with_subtype_name_list()
        test_cancer_without_subtype_name_list = task_loader.get_test_cancer_without_subtype_name_list()

        model = EmbeddingModelWithoutLastReLU().to(device)

        meta_optimizer = optim.Adam(model.net.parameters(), meta_learning_rate)

        model_path = os.path.join('.', 'Models', f'Fold {current_fold_number} {test_cancer_with_subtype_name_list} {test_cancer_without_subtype_name_list}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        test_task_name_list = test_cancer_with_subtype_name_list + [f'{test_cancer_with_subtype_name_list + test_cancer_without_subtype_name_list}']

        task_name_to_best_test_query_accuracy = dict()
        task_name_to_best_test_query_accuracy_train_iteration_number = dict()

        for cancer_name in test_cancer_with_subtype_name_list:

            task_name_to_best_test_query_accuracy[cancer_name] = 0
            task_name_to_best_test_query_accuracy_train_iteration_number[cancer_name] = 0

        task_name_to_best_test_query_accuracy[f'{test_cancer_with_subtype_name_list + test_cancer_without_subtype_name_list}'] = 0
        task_name_to_best_test_query_accuracy_train_iteration_number[f'{test_cancer_with_subtype_name_list + test_cancer_without_subtype_name_list}'] = 0

        for current_train_iteration_number in range(1, train_iteration_number + 1):

            if current_fold_number == resume_fold_number and current_train_iteration_number == (resume_train_iteration_number + 1):

                for task_name in test_task_name_list:

                    (task_name_to_best_test_query_accuracy[task_name], task_name_to_best_test_query_accuracy_train_iteration_number[task_name]) = get_accuracy_and_train_iteration_number(model_path, task_name, model, task_loader.get_samples(), test_support_iteration_number, task_learning_rate)

                resume_model_file = get_file(model_path, f'Train Iteration {resume_train_iteration_number}', 'Embedding Model')
                resume_optimizer_file = get_file(model_path, f'Train Iteration {resume_train_iteration_number}', 'Optimizer')

                model.load_state_dict(torch.load(resume_model_file))
                meta_optimizer.load_state_dict(torch.load(resume_optimizer_file))

                resume_flag = False

            if not resume_flag:
                batch_loss = torch.tensor(0.).to(device)

            for batch_number in range(batch_size):

                (support_samples, query_samples, support_labels, query_labels) = task_loader.get_task(True)
                
                if not resume_flag:
                    parameter_list = list(model.net.parameters())

                    for support_iteration_number in range(train_support_iteration_number):

                        support_features = model(support_samples, parameter_list)

                        with torch.no_grad():

                            query_features = model(query_samples, parameter_list)

                            (loss, accuracy) = get_loss_and_accuracy(support_features, query_features, support_labels, query_labels)
                        
                        train_query_sum_loss_list[support_iteration_number] += loss.item()
                        train_query_sum_accuracy_list[support_iteration_number] += accuracy.item()

                        (loss, accuracy) = get_loss_and_accuracy(support_features, support_features, support_labels, support_labels)
                        
                        train_support_sum_loss_list[support_iteration_number] += loss.item()
                        train_support_sum_accuracy_list[support_iteration_number] += accuracy.item()

                        parameter_list = get_updated_parameter_list(loss, parameter_list, task_learning_rate)

                    support_features = model(support_samples, parameter_list)
                    query_features = model(query_samples, parameter_list)

                    (loss, accuracy) = get_loss_and_accuracy(support_features, query_features, support_labels, query_labels)
                    
                    train_query_sum_loss_list[train_support_iteration_number] += loss.item()
                    train_query_sum_accuracy_list[train_support_iteration_number] += accuracy.item()

                    batch_loss += loss

            if not resume_flag:
                loss = batch_loss / batch_size

                print('loss:', loss.item())

                meta_optimizer.zero_grad()
                loss.backward()
                meta_optimizer.step()

            if (not resume_flag) and (current_train_iteration_number % print_iteration_number) == 0:
                print('current_train_iteration_number:', current_train_iteration_number)

                print('train_support_loss_list:', [sum_loss / (batch_size * print_iteration_number) for sum_loss in train_support_sum_loss_list])
                print('train_support_accuracy_list:', [sum_accuracy / (batch_size * print_iteration_number) for sum_accuracy in train_support_sum_accuracy_list])

                print('train_query_loss_list:', [sum_loss / (batch_size * print_iteration_number) for sum_loss in train_query_sum_loss_list])
                print('train_query_accuracy_list:', [sum_accuracy / (batch_size * print_iteration_number) for sum_accuracy in train_query_sum_accuracy_list])
                
                train_support_sum_loss_list = [0] * train_support_iteration_number
                train_support_sum_accuracy_list = [0] * train_support_iteration_number

                train_query_sum_loss_list = [0] * (train_support_iteration_number + 1)
                train_query_sum_accuracy_list = [0] * (train_support_iteration_number + 1)

            if current_train_iteration_number % test_train_iteration_number == 0:

                for task_name in test_task_name_list:

                    if not resume_flag:
                        print('current_train_iteration_number:', current_train_iteration_number)

                        print('task_name:', task_name)

                        test_support_sample_indeces_list = list()
                        test_query_sample_indeces_list = list()
                        test_support_labels_list = list()
                        test_query_labels_list = list()
                        test_global_label_indeces_list = list()

                        test_support_sum_loss_list = [0] * test_support_iteration_number
                        test_support_sum_accuracy_list = [0] * test_support_iteration_number

                        test_query_sum_loss_list = [0] * (test_support_iteration_number + 1)
                        test_query_sum_accuracy_list = [0] * (test_support_iteration_number + 1)

                    for current_test_iteration_number in range(1, test_iteration_number + 1):

                        if task_name in test_cancer_with_subtype_name_list:
                            (support_samples, query_samples, support_labels, query_labels, support_sample_indeces, query_sample_indeces, test_global_label_indeces) = task_loader.get_task(False, task_name)
                        
                        else:
                            (support_samples, query_samples, support_labels, query_labels, support_sample_indeces, query_sample_indeces, test_global_label_indeces) = task_loader.get_task(False)

                        if not resume_flag:
                            test_support_sample_indeces_list.append(support_sample_indeces)
                            test_query_sample_indeces_list.append(query_sample_indeces)
                            test_support_labels_list.append(support_labels)
                            test_query_labels_list.append(query_labels)
                            test_global_label_indeces_list.append(test_global_label_indeces)

                            parameter_list = list(model.net.parameters())

                            for support_iteration_number in range(test_support_iteration_number):

                                support_features = model(support_samples, parameter_list)

                                with torch.no_grad():

                                    query_features = model(query_samples, parameter_list)

                                    (loss, accuracy) = get_loss_and_accuracy(support_features, query_features, support_labels, query_labels)
                                
                                test_query_sum_loss_list[support_iteration_number] += loss.item()
                                test_query_sum_accuracy_list[support_iteration_number] += accuracy.item()

                                (loss, accuracy) = get_loss_and_accuracy(support_features, support_features, support_labels, support_labels)
                                
                                test_support_sum_loss_list[support_iteration_number] += loss.item()
                                test_support_sum_accuracy_list[support_iteration_number] += accuracy.item()

                                parameter_list = get_updated_parameter_list(loss, parameter_list, task_learning_rate)

                            with torch.no_grad():

                                support_features = model(support_samples, parameter_list)
                                query_features = model(query_samples, parameter_list)

                            (loss, accuracy) = get_loss_and_accuracy(support_features, query_features, support_labels, query_labels)
                            
                            test_query_sum_loss_list[test_support_iteration_number] += loss.item()
                            test_query_sum_accuracy_list[test_support_iteration_number] += accuracy.item()

                    if not resume_flag:
                        print('test_support_loss_list:', [sum_loss / test_iteration_number for sum_loss in test_support_sum_loss_list])
                        print('test_support_accuracy_list:', [sum_accuracy / test_iteration_number for sum_accuracy in test_support_sum_accuracy_list])

                        print('test_query_loss_list:', [sum_loss / test_iteration_number for sum_loss in test_query_sum_loss_list])
                        
                        test_query_accuracy_list = [sum_accuracy / test_iteration_number for sum_accuracy in test_query_sum_accuracy_list]
                        print('test_query_accuracy_list:', test_query_accuracy_list)
                        test_query_accuracy = max(test_query_accuracy_list)

                        if test_query_accuracy > task_name_to_best_test_query_accuracy[task_name]:
                            delete_file(os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Embedding Model'))
                            delete_file(os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Optimizer'))
                            delete_file(os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Test Support Total Sample Indeces'))
                            delete_file(os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Test Query Total Sample Indeces'))
                            delete_file(os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Test Support Total Labels'))
                            delete_file(os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Test Query Total Labels'))
                            delete_file(os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Test Total Global Label Indeces'))
                            
                            task_name_to_best_test_query_accuracy[task_name] = test_query_accuracy
                            task_name_to_best_test_query_accuracy_train_iteration_number[task_name] = current_train_iteration_number

                            test_support_total_sample_indeces = torch.stack(test_support_sample_indeces_list, dim=0)
                            test_query_total_sample_indeces = torch.stack(test_query_sample_indeces_list, dim=0)
                            test_support_total_labels = torch.stack(test_support_labels_list, dim=0)
                            test_query_total_labels = torch.stack(test_query_labels_list, dim=0)
                            test_total_global_label_indeces = torch.stack(test_global_label_indeces_list, dim=0)

                            torch.save(model.state_dict(), os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Embedding Model'))
                            torch.save(meta_optimizer.state_dict(), os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Optimizer'))
                            torch.save(test_support_total_sample_indeces, os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Test Support Total Sample Indeces'))
                            torch.save(test_query_total_sample_indeces, os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Test Query Total Sample Indeces'))
                            torch.save(test_support_total_labels, os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Test Support Total Labels'))
                            torch.save(test_query_total_labels, os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Test Query Total Labels'))
                            torch.save(test_total_global_label_indeces, os.path.join(model_path, f'Train Iteration {task_name_to_best_test_query_accuracy_train_iteration_number[task_name]} {task_name} Test Total Global Label Indeces'))

                        print('best_test_query_accuracy:', task_name_to_best_test_query_accuracy[task_name])
                        print('best_test_query_accuracy_train_iteration_number:', task_name_to_best_test_query_accuracy_train_iteration_number[task_name])

        if resume_flag:

            for task_name in test_task_name_list:

                (task_name_to_best_test_query_accuracy[task_name], task_name_to_best_test_query_accuracy_train_iteration_number[task_name]) = get_accuracy_and_train_iteration_number(model_path, task_name, model, task_loader.get_samples(), test_support_iteration_number, task_learning_rate)

        task_name_to_best_test_query_accuracy_list.append(task_name_to_best_test_query_accuracy)
        task_name_to_best_test_query_accuracy_train_iteration_number_list.append(task_name_to_best_test_query_accuracy_train_iteration_number)

    print('task_name_to_best_test_query_accuracy_list:', task_name_to_best_test_query_accuracy_list)
    print('task_name_to_best_test_query_accuracy_train_iteration_number_list:', task_name_to_best_test_query_accuracy_train_iteration_number_list)

if __name__ == '__main__':
    main()