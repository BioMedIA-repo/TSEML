import torch, random, os
import numpy as np
import torch.nn.functional as F
from TaskLoader import TaskLoader
from Layers import EmbeddingModelWithoutLastReLU
import sklearn.manifold as manifold
import sklearn.metrics as metrics
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import rcParams
import seaborn as sns

def get_file(path, prefix, suffix):

    file_list = os.listdir(path)

    for file in file_list:

        if file.startswith(prefix) and file.endswith(suffix):
            return os.path.join(path, file)

def get_logits_and_loss_and_accuracy(support_features, query_features, support_labels, query_labels):

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

    return (distance_logits, loss, accuracy)

def get_updated_parameter_list(loss, parameter_list, learning_rate):

    gradients = torch.autograd.grad(loss, parameter_list, create_graph=True, retain_graph=True)

    updated_parameter_list = list(map(lambda x: x[0] - learning_rate * x[1], zip(parameter_list, gradients)))

    return updated_parameter_list

def get_predictions(logits, query_global_labels):

    max_indeces = torch.argmax(logits, dim=1)

    prediction_list = list()

    for (logit, max_index, query_global_label) in zip(logits, max_indeces, query_global_labels):

        if max_index == query_global_label or logit[max_index] == logit[query_global_label]:
            prediction_list.append(query_global_label)

        else:
            prediction_list.append(max_index)

    predictions = torch.stack(prediction_list, dim=0)

    return predictions

def print_indicator(indicator_name, indicator_list, indicator_element_name_to_indicator_element=None):

    indicator_mean = np.mean(indicator_list)
    indicator_std = np.std(indicator_list)

    print(f'{indicator_name}: {indicator_mean} ± {indicator_std}')

    if not indicator_element_name_to_indicator_element is None:
        indicator_element_name_to_indicator_element[f'{indicator_name}_mean'] = indicator_mean
        indicator_element_name_to_indicator_element[f'{indicator_name}_std'] = indicator_std

def print_indicator_list(indicator_list_name, indicator_list_list):

    current_way = len(indicator_list_list[0])

    print(f'{indicator_list_name}: [', end='')

    for index in range(current_way):

        if index != 0:
            print(', ', end='')

        _ = list(map(lambda indicator_list: indicator_list[index], indicator_list_list))

        print(f'{np.mean(_)} ± {np.std(_)}', end='')

    print(']')

def get_indicator_name_to_indicator(indicator_element_name_to_indicator_element):

    indicator_name_to_indicator = dict()

    for (indicator_element_name, indicator_element) in indicator_element_name_to_indicator_element.items():

        if indicator_element_name.__contains__('_mean'):
            indicator_name = indicator_element_name[: -5]

            indicator_mean = indicator_element
            indicator_std = indicator_element_name_to_indicator_element[f'{indicator_name}_std']

            if indicator_name == 'accuracy' or indicator_name == 'macro_precision':
                indicator_name += '_(%)'
                indicator_mean *= 100
                indicator_std *= 100

                indicator_name_to_indicator[indicator_name.replace('_', ' ')] = f'{indicator_mean:.2f} ± {indicator_std:.2f}'

            else:
                indicator_name_to_indicator[indicator_name.replace('_', ' ')] = f'{indicator_mean:.4f} ± {indicator_std:.4f}'

    return indicator_name_to_indicator

def print_indicator_and_get_indicator_name_to_indicator(prefix, indicator_element_name_to_indicator_element_list):

    indicator_name_to_indicator = dict()

    for indicator_element_name in indicator_element_name_to_indicator_element_list[0].keys():

        if indicator_element_name.__contains__('_mean'):
            indicator_mean_list = list(map(lambda indicator_element_name_to_indicator_element: indicator_element_name_to_indicator_element[indicator_element_name], indicator_element_name_to_indicator_element_list))

            indicator_name = indicator_element_name[: -5]

            indicator_mean_mean = np.mean(indicator_mean_list)
            indicator_mean_std = np.std(indicator_mean_list)

            print(f'{prefix}_tasks_{indicator_name}: {indicator_mean_mean} ± {indicator_mean_std}')

            if indicator_name == 'accuracy' or indicator_name == 'macro_precision':
                indicator_name += '_(%)'
                indicator_mean_mean *= 100
                indicator_mean_std *= 100

                indicator_name_to_indicator[indicator_name.replace('_', ' ')] = f'{indicator_mean_mean:.2f} ± {indicator_mean_std:.2f}'

            else:
                indicator_name_to_indicator[indicator_name.replace('_', ' ')] = f'{indicator_mean_mean:.4f} ± {indicator_mean_std:.4f}'

    return indicator_name_to_indicator

method_name = '5-shot TSEML（欧式）'

config = {
    'font.family' : 'sans-serif',
    'mathtext.fontset' : 'stix',
    'font.sans-serif' : ['Simsun'],
    'axes.unicode_minus' : False
}

rcParams.update(config)

dataframes_ready = False

if not dataframes_ready:
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
    support_shot = 5
    train_query_shot = 15
    test_query_shot = support_shot
    test_support_iteration_number = 10
    device = torch.device('cuda')

    task_loader = TaskLoader(types_path, fold_number, way, support_shot, train_query_shot, test_query_shot, device)

    task_learning_rate = 5e-2

    subtype_task_name_to_indicator_element_name_to_indicator_element = dict()
    cancer_task_name_to_indicator_element_name_to_indicator_element = dict()

    samples = task_loader.get_samples()

    # tsne = manifold.TSNE(learning_rate='auto', init='pca', random_state=seed, method='exact')
    tsne = manifold.TSNE(learning_rate='auto', init='pca', random_state=seed)

    cancer_name_to_start_index_and_length = task_loader.get_cancer_name_to_start_index_and_length()
    cancer_name_to_subtype_name_to_start_index_and_length = task_loader.get_cancer_name_to_subtype_name_to_start_index_and_length()

    type_number = len(cancer_name_to_start_index_and_length)

    for subtype_name_to_start_index_and_length in cancer_name_to_subtype_name_to_start_index_and_length.values():

        type_number += len(subtype_name_to_start_index_and_length)

    type_colors = plt.cm.Spectral(np.linspace(0, 1, type_number))

    general_dataframe_list = list()

    detailed_dataframe_list = list()

    for current_fold_number in range(fold_number):

        print('current_fold_number:', current_fold_number)

        task_loader.set_train_and_test_sets(current_fold_number)

        test_cancer_with_subtype_name_list = task_loader.get_test_cancer_with_subtype_name_list()
        test_cancer_without_subtype_name_list = task_loader.get_test_cancer_without_subtype_name_list()

        model = EmbeddingModelWithoutLastReLU().to(device)

        model_path = os.path.join('.', 'Models', f'Fold {current_fold_number} {test_cancer_with_subtype_name_list} {test_cancer_without_subtype_name_list}')
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        figure_path = os.path.join('.', 'Figures', f'Fold {current_fold_number} {test_cancer_with_subtype_name_list} {test_cancer_without_subtype_name_list}')
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)

        test_task_name_list = test_cancer_with_subtype_name_list + [f'{test_cancer_with_subtype_name_list + test_cancer_without_subtype_name_list}']

        for task_name in test_task_name_list:

            indicator_element_name_to_indicator_element = dict()

            if task_name in test_cancer_with_subtype_name_list:
                type_name_list = task_loader.get_test_subtype_name_list(task_name)

                subtype_task_name_to_indicator_element_name_to_indicator_element[task_name] = indicator_element_name_to_indicator_element

            else:
                type_name_list = test_cancer_with_subtype_name_list + test_cancer_without_subtype_name_list

                cancer_task_name_to_indicator_element_name_to_indicator_element[task_name] = indicator_element_name_to_indicator_element

            print('type_name_list:', type_name_list)

            model_file = get_file(model_path, '', f'{task_name} Embedding Model')
            test_support_total_sample_indeces_file = get_file(model_path, '', f'{task_name} Test Support Total Sample Indeces')
            test_query_total_sample_indeces_file = get_file(model_path, '', f'{task_name} Test Query Total Sample Indeces')
            test_support_total_labels_file = get_file(model_path, '', f'{task_name} Test Support Total Labels')
            test_query_total_labels_file = get_file(model_path, '', f'{task_name} Test Query Total Labels')
            test_total_global_label_indeces_file = get_file(model_path, '', f'{task_name} Test Total Global Label Indeces')
            
            model.load_state_dict(torch.load(model_file))
            test_support_total_sample_indeces = torch.load(test_support_total_sample_indeces_file)
            test_query_total_sample_indeces = torch.load(test_query_total_sample_indeces_file)
            test_support_total_labels = torch.load(test_support_total_labels_file)
            test_query_total_labels = torch.load(test_query_total_labels_file)
            test_total_global_label_indeces = torch.load(test_total_global_label_indeces_file)

            with torch.no_grad():

                features = model(samples)

            points = tsne.fit_transform(features.cpu())

           

            type_count = 0

            plt.figure('t-SNE', figsize=(22, 11))

            for (cancer_name, start_index_and_length) in cancer_name_to_start_index_and_length.items():

                (start_index, length) = start_index_and_length

                plt.scatter(x=points[start_index: start_index + length, 0], y=points[start_index: start_index + length, 1], color=type_colors[type_count], label=cancer_name)

                type_count += 1

            for subtype_name_to_start_index_and_length in cancer_name_to_subtype_name_to_start_index_and_length.values():

                for (subtype_name, start_index_and_length) in subtype_name_to_start_index_and_length.items():

                    (start_index, length) = start_index_and_length

                    if (subtype_name.startswith('LIHC-')):
                        subtype_name = subtype_name[: 13] + ':' + subtype_name[14: ]
                    elif (subtype_name.startswith('KIRP-')):
                        subtype_name = subtype_name[: 9] + '.' + subtype_name[10: ]

                    plt.scatter(x=points[start_index: start_index + length, 0], y=points[start_index: start_index + length, 1], color=type_colors[type_count], label=subtype_name)

                    type_count += 1

            plt.title('t-SNE图')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
            
            task_name_string = task_name.replace('[', '').replace(']', '').replace('\'', '').replace(',', '')
            if ' ' in task_name_string:
                task_type_name = '癌症'

            else:
                task_type_name = '癌症分子亚型'
            
            plt.savefig(os.path.join(figure_path, f'{method_name}{task_name_string}{task_type_name}分类任务t-SNE图.pdf'), bbox_inches='tight')
            plt.cla()

            sum_accuracy_list = [0] * (test_support_iteration_number + 1)

            query_global_labels_list = list()

            logits_list_list = list()

            for (support_sample_indeces, query_sample_indeces, support_labels, query_labels, global_label_indeces) in zip(test_support_total_sample_indeces, test_query_total_sample_indeces, test_support_total_labels, test_query_total_labels, test_total_global_label_indeces):

                support_samples = samples[support_sample_indeces]
                query_samples = samples[query_sample_indeces]

                support_global_labels = global_label_indeces[support_labels]
                query_global_labels = global_label_indeces[query_labels]

                query_global_labels_list.append(query_global_labels)

                parameter_list = list(model.net.parameters())

                logits_list = list()

                for support_iteration_number in range(test_support_iteration_number):

                    support_features = model(support_samples, parameter_list)

                    with torch.no_grad():

                        query_features = model(query_samples, parameter_list)

                        (logits, loss, accuracy) = get_logits_and_loss_and_accuracy(support_features, query_features, support_global_labels, query_global_labels)
                    
                    logits_list.append(logits)

                    sum_accuracy_list[support_iteration_number] += accuracy.item()

                    (logits, loss, accuracy) = get_logits_and_loss_and_accuracy(support_features, support_features, support_global_labels, support_global_labels)

                    parameter_list = get_updated_parameter_list(loss, parameter_list, task_learning_rate)

                with torch.no_grad():

                    support_features = model(support_samples, parameter_list)
                    query_features = model(query_samples, parameter_list)

                (logits, loss, accuracy) = get_logits_and_loss_and_accuracy(support_features, query_features, support_global_labels, query_global_labels)

                logits_list.append(logits)

                sum_accuracy_list[test_support_iteration_number] += accuracy.item()

                logits_list_list.append(logits_list)
            
            accuracy_list = [sum_accuracy / len(test_support_total_sample_indeces) for sum_accuracy in sum_accuracy_list]
            
            accuracy = max(accuracy_list)

            best_support_iteration_number = accuracy_list.index(accuracy)

            logits_list = list()

            for _ in logits_list_list:

                logits_list.append(_[best_support_iteration_number])

            accuracy_list = list()

            for (logits, query_global_labels) in zip(logits_list, query_global_labels_list):

                chosen_values = torch.gather(logits, dim=1, index=torch.unsqueeze(query_global_labels, dim=1))
                max_values = torch.unsqueeze(torch.max(logits, dim=1)[0], dim=1)
                correct_number = torch.sum(torch.eq(chosen_values, max_values))
                accuracy = correct_number / len(query_global_labels)
                accuracy_list.append(accuracy.item())

            print('accuracy:', f'{np.mean(accuracy_list)} ± {np.std(accuracy_list)}')

            

            accuracy_list = list()

            micro_precision_list = list()
            macro_precision_list = list()
            weighted_precision_list = list()
            precision_list_list = list()

            micro_recall_list = list()
            macro_recall_list = list()
            weighted_recall_list = list()
            recall_list_list = list()

            micro_f1_list = list()
            macro_f1_list = list()
            weighted_f1_list = list()
            f1_list_list = list()

            macro_ovr_auc_list = list()
            weighted_ovr_auc_list = list()
            macro_ovo_auc_list = list()
            weighted_ovo_auc_list = list()

            for (logits, query_global_labels) in zip(logits_list, query_global_labels_list):

                predictions = get_predictions(logits, query_global_labels)

                y_true = query_global_labels.cpu()
                y_pred = predictions.cpu()

                accuracy_list.append(metrics.accuracy_score(y_true, y_pred))

                micro_precision_list.append(metrics.precision_score(y_true, y_pred, average='micro'))
                macro_precision_list.append(metrics.precision_score(y_true, y_pred, average='macro', zero_division=1))
                weighted_precision_list.append(metrics.precision_score(y_true, y_pred, average='weighted', zero_division=1))
                precision_list_list.append(metrics.precision_score(y_true, y_pred, average=None, zero_division=1))

                micro_recall_list.append(metrics.recall_score(y_true, y_pred, average='micro'))
                macro_recall_list.append(metrics.recall_score(y_true, y_pred, average='macro'))
                weighted_recall_list.append(metrics.recall_score(y_true, y_pred, average='weighted'))
                recall_list_list.append(metrics.recall_score(y_true, y_pred, average=None))

                micro_f1_list.append(metrics.f1_score(y_true, y_pred, average='micro'))
                macro_f1_list.append(metrics.f1_score(y_true, y_pred, average='macro'))
                weighted_f1_list.append(metrics.f1_score(y_true, y_pred, average='weighted'))
                f1_list_list.append(metrics.f1_score(y_true, y_pred, average=None))

                y_score = F.softmax(logits, dim=1).cpu()

                macro_ovr_auc_list.append(metrics.roc_auc_score(y_true, y_score, average='macro', multi_class='ovr'))
                weighted_ovr_auc_list.append(metrics.roc_auc_score(y_true, y_score, average='weighted', multi_class='ovr'))
                macro_ovo_auc_list.append(metrics.roc_auc_score(y_true, y_score, average='macro', multi_class='ovo'))
                weighted_ovo_auc_list.append(metrics.roc_auc_score(y_true, y_score, average='weighted', multi_class='ovo'))

            print_indicator('accuracy', accuracy_list, indicator_element_name_to_indicator_element)

            print_indicator('micro_precision', micro_precision_list)
            print_indicator('macro_precision', macro_precision_list, indicator_element_name_to_indicator_element)
            print_indicator('weighted_precision', weighted_precision_list)
            print_indicator_list('precision_list', precision_list_list)

            print_indicator('micro_recall', micro_recall_list)
            print_indicator('macro_recall', macro_recall_list)
            print_indicator('weighted_recall', weighted_recall_list)
            print_indicator_list('recall_list', recall_list_list)

            print_indicator('micro_f1', micro_f1_list)
            print_indicator('macro_f1', macro_f1_list, indicator_element_name_to_indicator_element)
            print_indicator('weighted_f1', weighted_f1_list)
            print_indicator_list('f1_list', f1_list_list)

            print_indicator('macro_ovr_auc', macro_ovr_auc_list, indicator_element_name_to_indicator_element)
            print_indicator('weighted_ovr_auc', weighted_ovr_auc_list)
            print_indicator('macro_ovo_auc', macro_ovo_auc_list)
            print_indicator('weighted_ovo_auc', weighted_ovo_auc_list)

            indicator_name_to_indicator = get_indicator_name_to_indicator(indicator_element_name_to_indicator_element)

            task_name_string = task_name.replace('[', '').replace(']', '').replace('\'', '').replace(',', '')

            general_dataframe = pd.DataFrame(indicator_name_to_indicator, index=[task_name_string])

            general_dataframe_list.append(general_dataframe)

            detailed_dataframe = pd.DataFrame([[task_name_string] * len(accuracy_list), [accuracy * 100 for accuracy in accuracy_list], [macro_precision * 100 for macro_precision in macro_precision_list], macro_f1_list, macro_ovr_auc_list], index=['任务', '准确率（%）', '宏平均精确率（%）', '宏平均F1', '宏平均AUC']).transpose()

            detailed_dataframe_list.append(detailed_dataframe)

            logits = torch.cat(logits_list, dim=0)

            query_global_labels = torch.cat(query_global_labels_list, dim=0)

            predictions = get_predictions(logits, query_global_labels)

            y_true = query_global_labels.cpu()
            y_pred = predictions.cpu()

            print('confusion_matrix:', metrics.confusion_matrix(y_true, y_pred))

            if ' ' in task_name_string:
                task_type_name = '癌症'

            else:
                task_type_name = '癌症分子亚型'

            plt.figure('PR Curve', figsize=(6.4, 4.8))

            current_way = len(type_name_list)

            for label in range(current_way):

                y_score = F.softmax(logits, dim=1)[:, label].cpu()

                (precision, recall, thresholds) = metrics.precision_recall_curve(y_true, y_score, pos_label=label)
                
                type_name = type_name_list[label]
                
                if (type_name.startswith('LIHC-')):
                    type_name = type_name[: 13] + ':' + type_name[14: ]
                elif (type_name.startswith('KIRP-')):
                    type_name = type_name[: 9] + '.' + type_name[10: ]

                plt.plot(recall, precision, label=type_name)

            plt.xlabel('召回率')
            plt.ylabel('精确率')
            title = f'{method_name}{task_name_string}{task_type_name}分类任务PR曲线'
            plt.title(title)
            plt.legend(loc='best')
            plt.savefig(os.path.join(figure_path, f'{title}.pdf'), bbox_inches='tight')
            plt.cla()

            plt.figure('ROC Curve', figsize=(6.4, 4.8))

            for label in range(current_way):
                
                y_score = F.softmax(logits, dim=1)[:, label].cpu()

                (fpr, tpr, thresholds) = metrics.roc_curve(y_true, y_score, pos_label=label)
                
                type_name = type_name_list[label]

                if (type_name.startswith('LIHC-')):
                    type_name = type_name[: 13] + ':' + type_name[14: ]
                elif (type_name.startswith('KIRP-')):
                    type_name = type_name[: 9] + '.' + type_name[10: ]

                print('type_name:', type_name)

                plt.plot(fpr, tpr, label=type_name)

                print('auc:', metrics.auc(fpr, tpr))

            plt.xlabel('假阳率')
            plt.ylabel('真阳率')
            title = f'{method_name}{task_name_string}{task_type_name}分类任务ROC曲线'
            plt.title(title)
            plt.legend(loc='best')
            plt.savefig(os.path.join(figure_path, f'{title}.pdf'), bbox_inches='tight')
            plt.cla()

    indicator_name_to_indicator = print_indicator_and_get_indicator_name_to_indicator('subtype', list(subtype_task_name_to_indicator_element_name_to_indicator_element.values()))

    general_dataframe = pd.DataFrame(indicator_name_to_indicator, index=['subtype tasks'])

    general_dataframe_list.append(general_dataframe)

    indicator_name_to_indicator = print_indicator_and_get_indicator_name_to_indicator('cancer', list(cancer_task_name_to_indicator_element_name_to_indicator_element.values()))

    general_dataframe = pd.DataFrame(indicator_name_to_indicator, index=['cancer tasks'])

    general_dataframe_list.append(general_dataframe)

    indicator_name_to_indicator = print_indicator_and_get_indicator_name_to_indicator('total', list(subtype_task_name_to_indicator_element_name_to_indicator_element.values()) + list(cancer_task_name_to_indicator_element_name_to_indicator_element.values()))

    general_dataframe = pd.DataFrame(indicator_name_to_indicator, index=['total tasks'])

    general_dataframe_list.append(general_dataframe)

    general_dataframe = pd.concat(general_dataframe_list)

    dataframe_path = os.path.join('.', 'Dataframes')
    if not os.path.exists(dataframe_path):
        os.makedirs(dataframe_path)

    general_dataframe.to_csv(os.path.join(dataframe_path, 'General Dataframe.tsv'), sep='\t', index_label='task name')

    detailed_dataframe = pd.concat(detailed_dataframe_list)

    detailed_dataframe.to_csv(os.path.join(dataframe_path, 'Detailed Dataframe.tsv'), sep='\t', index=False)

detailed_dataframe = pd.read_csv(os.path.join('.', 'Dataframes', 'Detailed Dataframe.tsv'), sep='\t')

figure_path = os.path.join('.', 'Figures')

for indicator_chinese_name in ['准确率（%）', '宏平均精确率（%）', '宏平均F1', '宏平均AUC']:

    plt.figure('violin chart', figsize=(6, 16))

    sns.violinplot(x=indicator_chinese_name, y='任务', data=detailed_dataframe, scale='count')

    title = f'{method_name}{indicator_chinese_name}小提琴图'
    plt.title(title)
    plt.savefig(os.path.join(figure_path, f'{title}.pdf'), bbox_inches='tight')
    plt.cla()