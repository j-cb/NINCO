import os
import csv
import argparse
import datetime
import timm
import torchvision.datasets as dset
from scipy.special import softmax
from torch.utils.data.dataset import Dataset
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from utils import extract_features, extract_clip_embeddings, timm_models, models_clip, fpr_at_tpr, auroc_ood, models_timm_dev, models_timm_0_6_12
import utils
import data.paths_config
from detection_methods import *
import datasets
import clip


class OOD_Score:
    def __init__(self, path_to_imagenet=data.paths_config.dset_location_dict['ImageNet1K'], path_to_cache='model_outputs/cache'):
        self.path_to_cache = path_to_cache
        self.path_to_imagenet = path_to_imagenet
        self.clip_quantile = 0.99
        self.methods = ['MSP', 'MaxLogit', 'ViM', 'Mahalanobis', 'Energy+React',
                        'Energy', 'KL-Matching', 'knn', 'Relative_Mahalanobis','mcm', 'cosine']
        self.clip_transform=None
        self.val_acc=-99
        self.train_acc=-99

    def setup(self, dataset, model, clip_model=False):
        """Load and prepare the data."""
        self.dataset = dataset

        # transform from timm cfg
        config = resolve_data_config({}, model = model, use_test_size = True)
        if clip_model:
            test_transform = self.clip_transform
        else:
            test_transform = create_transform(**config)
        
        available_OOD_datasets = {
            'NINCO': datasets.NINCO,
            'NINCO_OOD_unit_tests': datasets.NINCOOODUnitTests,
            'NINCO_popular_datasets_subsamples': datasets.NINCOPopularDatasetsSubsamples,
        }

        self.dataset_in_train = dset.ImageFolder(os.path.join(self.path_to_imagenet, 'train'), transform=test_transform)
        self.dataset_in_val = dset.ImageFolder(os.path.join(self.path_to_imagenet, 'val' ), transform=test_transform)
        self.dataset_out = available_OOD_datasets[dataset](transform=test_transform)

        
    def check_complete(self, path, expected_samples, sources = ['features', 'labels_true', 'logits']):
        predictions={}
        if os.path.exists(path):
            for source in sources:
                print('checking ', source)
                # names = []
                # for i in range(10000):
                #     potential_save = f'{source}_{i}.npy'
                #     if potential_save in os.listdir(path):
                #         names.append(potential_save)
                #     else:
                #         break
                names = sorted([f for f in os.listdir(path) if f.startswith(source+'_') and f.endswith('.npy')
                                and f[len(source+'_'):-len('.npy')].isdigit()])
                if len(names)==0:
                    print('No samples in {}'.format(path))
                    return False
                predictions[source] = np.concatenate([np.load(os.path.join(path, f)) for f in names])
                if source =='text_encoded' and len(predictions[source]) == 1000:
                    continue
                else:
                    if len(predictions[source]) != expected_samples:
                        print('There should be {} train samples of {} in {}, but there are {}'.format(expected_samples, source, path,
                                                                                                        len(predictions[source])))
                        
                        return False
        return predictions

    def get_features_and_logits(self, model, train=True, val=True, ood=True, overwrite='no'):
        if train:
            save_path_train = os.path.join(os.path.join(self.path_to_cache, 'cache_train', model.model_name))
            if overwrite in {'no', 'ood', 'notrain'}:
                predictions_train = self.check_complete(save_path_train, expected_samples = len(self.dataset_in_train))
            else:
                predictions_train = None
            if not predictions_train:
                print('Train features not complete, extracting...')
                extract_features(model, self.dataset_in_train, wo_head = False, savepath = save_path_train)
                predictions_train = self.check_complete(save_path_train, expected_samples = len(self.dataset_in_train))
            self.train_labels = predictions_train['labels_true']
            self.feature_id_train = predictions_train['features']  # [:,:,0,0]
            self.logits_id_train = predictions_train['logits']
            print('Computing softmax...')
            self.softmax_id_train = softmax(self.logits_id_train, axis = -1)
            predicted_classes_train = np.argmax(self.logits_id_train, axis = -1)
            self.train_acc = np.equal(predicted_classes_train, predictions_train['labels_true']).mean()
            print('Accuracy train: ', self.train_acc)
            print('Done')
        if val:
            save_path_val = os.path.join(os.path.join(self.path_to_cache, 'cache_val', model.model_name))
            if overwrite in {'no', 'ood'}:
                predictions_val = self.check_complete(save_path_val, expected_samples = len(self.dataset_in_val))
            else:
                predictions_val = None
            if not predictions_val:
                print('Val features not complete, extracting...')
                # dataloader_in_val = torch.utils.data.DataLoader(self.dataset_in_val, batch_size = model.batch_size,
                #                                                 shuffle = False, num_workers = 4, pin_memory = True,
                #                                                 drop_last = False)
                extract_features(model, self.dataset_in_val, wo_head =False, savepath = save_path_val)
                dataloader_in_val = None
                predictions_val = self.check_complete(save_path_val, expected_samples = len(self.dataset_in_val))
            self.feature_id_val = predictions_val['features']
            self.logits_id_val = predictions_val['logits']
            self.labels_id_val = predictions_val['labels_true']
            print('Computing softmax...')
            self.softmax_id_val = softmax(self.logits_id_val, axis = -1)
            self.predicted_classes = np.argmax(self.logits_id_val, axis = -1)
            self.val_acc = np.equal(self.predicted_classes, self.labels_id_val).mean()
            print('Accuracy val: ', self.val_acc)
            print('Done')
        if ood:
            save_path_ood = os.path.join(os.path.join(self.path_to_cache, 'cache_ood', model.model_name, self.dataset))
            if overwrite in {'no'}:
                predictions_ood = self.check_complete(save_path_ood, expected_samples = len(self.dataset_out))
            else:
                predictions_ood = None
            if not predictions_ood:
                print('OOD features ({}) not complete, extracting...'.format(self.dataset))
                extract_features(model, self.dataset_out, wo_head = False, savepath = save_path_ood)
                predictions_ood = self.check_complete(save_path_ood, expected_samples = len(self.dataset_out))
            self.feature_ood = predictions_ood['features']
            self.logits_ood = predictions_ood['logits']
            self.labels_ood = predictions_ood['labels_true']
            print('Computing softmax...')
            self.softmax_ood = softmax(self.logits_ood, axis = -1)
            print('Done')

        print('Reading w and b')
        if 'maxvit' in model.model_name or 'convnext' in model.model_name:
            self.w = model.head.fc.weight.cpu().clone().detach().numpy()
            self.b = model.head.fc.bias.cpu().clone().detach().numpy()
        elif ('vit' in model.model_name and 'max' not in model.model_name) or 'deit' in model.model_name or \
                'swin' in model.model_name or 'xcit' in model.model_name:
            self.w = model.head.weight.cpu().clone().detach().numpy()
            self.b = model.head.bias.cpu().clone().detach().numpy()
        elif 'BiT' in model.model_name:
            self.w = model.head.fc.weight.clone().detach().flatten(1).cpu().numpy()  # need to flatten conv filter
            self.b = model.head.fc.bias.clone().detach().cpu().numpy()
        elif 'efficient' in model.model_name:
            self.w = model.classifier.weight.cpu().clone().detach().numpy()
            self.b = model.classifier.bias.cpu().clone().detach().numpy()
        elif 'resnet50' in model.model_name:
            self.w = model.fc.weight.cpu().clone().detach().numpy()
            self.b = model.fc.bias.cpu().clone().detach().numpy()
        else:
            state_dict = model.model.state_dict()
            self.w = state_dict['fc.weight'].clone().detach().cpu().numpy()
            self.b = state_dict['fc.bias'].clone().detach().cpu().numpy()
        self.u = -np.matmul(pinv(self.w), self.b)

    def get_features_clip(self, model, val=True, ood=True, overwrite='no'):
        if val:
            save_path_val = os.path.join(os.path.join(self.path_to_cache, 'cache_val', model.model_name))
            if overwrite in {'no', 'ood'}:
                predictions_val = self.check_complete(save_path_val, expected_samples = len(self.dataset_in_val),
                                                      sources = ['features', 'labels_true', 'text_encoded'])
            else:
                predictions_val = None
            if not predictions_val:
                print('Val features not complete, extracting...')
                dataloader_in_val = torch.utils.data.DataLoader(self.dataset_in_val, batch_size = model.batch_size,
                                                                shuffle = False, num_workers = 4, pin_memory = True,
                                                                drop_last = False)
                text_labels = np.load('model_outputs/im_class_clean.npy')
                extract_clip_embeddings(model, dataloader_in_val, text = text_labels, savepath = save_path_val)
                dataloader_in_val = None
                predictions_val = self.check_complete(save_path_val, expected_samples = len(self.dataset_in_val),
                                                      sources = ['features', 'labels_true', 'text_encoded'])
            self.feature_id_val = predictions_val['features']
            self.labels_id_val = predictions_val['labels_true']
            self.labels_encoded_clip = predictions_val['text_encoded']
            self.clip_labels_true = predictions_val['labels_true']

            print('Done')
        if ood:
            save_path_ood = os.path.join(os.path.join(self.path_to_cache, 'cache_ood', model.model_name, self.dataset_out.__name__))
            if overwrite in {'no',}:
                predictions_ood = self.check_complete(save_path_ood, expected_samples = len(self.dataset_out),
                                                  sources = ['features', 'labels_true'])
            else:
                predictions_ood = None
            if not predictions_ood:
                print('OOD features ({}) not complete, extracting...'.format(self.dataset_out.__name__))
                dataloader_out = torch.utils.data.DataLoader(self.dataset_out, batch_size = model.batch_size,
                                                             shuffle = False, num_workers = 4, pin_memory = True,
                                                             drop_last = False)
                extract_clip_embeddings(model, dataloader_out, savepath = save_path_ood)
                dataloader_out = None
                predictions_ood = self.check_complete(save_path_ood, expected_samples = len(self.dataset_out),
                                                      sources = ['features', 'labels_true'])
            self.feature_ood = predictions_ood['features']
            self.labels_ood = predictions_ood['labels_true']
            print('Done.')

    def evaluate(self, model, OOD_classes, methods=['MSP']):
        # patly adapted from https://github.com/haoqiwang/vim/blob/master/benchmark.py
        path = os.path.join(self.path_to_cache, 'cache_methods', model.model_name)
        if not os.path.exists(path):
            os.makedirs(path)
        methods_results = {}
        for method in methods:
            if method == 'MSP':
                scores_id, scores_ood = evaluate_MSP(self.softmax_id_val, self.softmax_ood)
            elif method == 'MaxLogit':
                scores_id, scores_ood = evaluate_MSP(self.logits_id_val, self.logits_ood)
            elif method == 'Energy':
                scores_id, scores_ood = evaluate_Energy(logits_in_distribution = self.logits_id_val, logits_out_of_distribution = self.logits_ood)
            elif method == 'ViM':
                scores_id, scores_ood = evaluate_ViM(feature_id_train = self.feature_id_train, feature_id_val = self.feature_id_val, feature_ood = self.feature_ood, logits_id_train = self.logits_id_train, logits_id_val = self.logits_id_val, logits_ood = self.logits_ood, u = self.u, path = path)
            elif method == 'Mahalanobis':
                scores_id, scores_ood = evaluate_Mahalanobis(feature_id_train = self.feature_id_train, feature_id_val = self.feature_id_val, feature_ood = self.feature_ood, train_labels = self.train_labels, path = path)
            elif method == 'Relative_Mahalanobis':
                scores_id, scores_ood = evaluate_Relative_Mahalanobis(feature_id_train = self.feature_id_train, feature_id_val = self.feature_id_val, feature_ood = self.feature_ood, train_labels = self.train_labels, path = path)
            elif method == 'KL-Matching':
                scores_id, scores_ood = evaluate_KL_Matching(softmax_id_train = self.softmax_id_train, softmax_id_val = self.softmax_id_val, softmax_ood = self.softmax_ood, path = path)
            elif method == 'Energy+React':
                scores_id, scores_ood = evaluate_Energy_React(feature_id_train = self.feature_id_train, feature_id_val = self.feature_id_val, feature_ood = self.feature_ood, w = self.w, b = self.b, path = path)
            elif method=='knn':
                scores_id, scores_ood = evaluate_KNN(feature_id_train = self.feature_id_train, feature_id_val = self.feature_id_val, feature_ood = self.feature_ood, path = path)
            elif method=='mcm':
                scores_id, scores_ood = evaluate_rcos(feature_id_train = self.feature_id_train, feature_id_val = self.feature_id_val, feature_ood = self.feature_ood, train_labels = self.train_labels, path = path)
            elif method=='cosine':
                scores_id, scores_ood = evaluate_cosine(feature_id_train = self.feature_id_train, feature_id_val = self.feature_id_val, feature_ood = self.feature_ood, train_labels = self.train_labels, path = path)
            elif method == 'cosine-clip':
                scores_id, scores_ood = evaluate_cosine_clip(feature_id_val = self.feature_id_val, feature_ood = self.feature_ood, clip_labels = self.clip_labels_true, labels_encoded_clip = self.labels_encoded_clip, path = path)
            elif method == 'mcm-clip':
                scores_id, scores_ood = evaluate_mcm_clip(feature_id_val = self.feature_id_val, feature_ood = self.feature_ood, clip_labels = self.clip_labels_true, labels_encoded_clip = self.labels_encoded_clip, path = path)
            else:
                raise NotImplementedError(f'Method {method} not implemented.')
                
            methods_results[method] = {'scores_id': scores_id,
                                      'scores_ood': scores_ood}

            for c in OOD_classes:
                class_indices = np.where(self.labels_ood==self.dataset_out.class_to_idx[c])
                scores_on_ood_class = scores_ood[class_indices]
                methods_results[method][c] = {'auroc': auroc_ood(scores_id, scores_on_ood_class),
                                              'fpr_at_95': fpr_at_tpr(scores_id, scores_on_ood_class, 0.95)}
            methods_results[method]['samples_mean_auroc'] = auroc_ood(scores_id, scores_ood)
            methods_results[method]['samples_mean_fpr_at_95'] = fpr_at_tpr(scores_id, scores_ood, 0.95)
            methods_results[method]['ood_classes_mean_auroc'] = np.mean(np.array([methods_results[method][c]['auroc'] for c in OOD_classes]))
            methods_results[method]['ood_classes_mean_fpr_at_95'] = np.mean(np.array([methods_results[method][c]['fpr_at_95'] for c in OOD_classes]))

        print('{} on {} evaluated with {}.\nAuroc: {}\nfpr at 95: {}\naccuracy val: {}\n accuracy train: {}'.format(
            method, self.dataset, model.model_name, methods_results[method]['ood_classes_mean_auroc'],
            methods_results[method]['ood_classes_mean_fpr_at_95'], self.val_acc, self.train_acc))
        # save results
        savepath = os.path.join(self.path_to_cache, 'scores', model.model_name, self.dataset_out.__name__)
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        eval_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        np.savez(os.path.join(savepath, f'E{eval_time}.npz'), methods_results=methods_results, id_labels=self.labels_id_val, ood_labels=self.labels_ood, ood_classes=OOD_classes,
                val_acc=self.val_acc, train_acc=self.train_acc)
        
methods_train_usage = {
            'MSP': False,
            'MaxLogit': False,
            'Energy': False,
            'cosine-clip': False,
            'mcm-clip': False,
            'ViM': True,
            'Mahalanobis': True,
            'Relative_Mahalanobis': True,
            'KL-Matching': True,
            'Energy+React': True,
            'knn': True,
            'mcm': True,
            'cosine': True,
            'cosine-clip': True,
}
        
parser = argparse.ArgumentParser(description = 'OOD Evaluation on NINCO')
parser.add_argument('--path_to_weights', default = 'model_weights', )
parser.add_argument('--model_name', default = 'convnext_base_in22ft1k' )
parser.add_argument('--dataset', type=str, choices=['NINCO', 'NINCO_OOD_unit_tests', 'NINCO_popular_datasets_subsamples'], default='NINCO')
parser.add_argument('--overwrite_model_outputs', type=str, choices=['no', 'all', 'notrain', 'ood'], default='no')
parser.add_argument('--method', default = 'MSP')
parser.add_argument('--path_to_imagenet', default=data.paths_config.dset_location_dict['ImageNet1K'])
parser.add_argument('--path_to_cache', default='./cache')
parser.add_argument('--batch_size', type=int, default = 128)

def main():
    args = parser.parse_args()
    torch.hub.set_dir(args.path_to_weights)
    OOD_classes = getattr(utils, f'{args.dataset}_class_names')
    task = OOD_Score(path_to_cache = args.path_to_cache, path_to_imagenet=args.path_to_imagenet)
    methods = task.methods if args.method == 'all' else [args.method]
    need_train_outputs = any([methods_train_usage[m] for m in methods]) #raises KeyError if a method is not available
    # timm models
    if args.model_name=='models_timm_dev':
        model_names=models_timm_dev
    elif args.model_name=='models_timm_0_6_12':
        model_names=models_timm_0_6_12
    else:
        model_names=[args.model_name]
    for model_name in model_names:
        if model_name in timm_models.keys():
            model = timm.create_model(**timm_models[model_name]['config']).cuda().eval()
            model.model_name = model_name
            model.batch_size = args.batch_size
            print('Created model {}.'.format(model.model_name))
            task.setup(args.dataset, model, clip_model=False)
            print('Task is set up.')
            task.get_features_and_logits(model, ood=True, train=need_train_outputs, overwrite=args.overwrite_model_outputs)
            task.evaluate(model, OOD_classes=OOD_classes, methods=methods)
        # CLIP zero shot models
        elif model_name in models_clip.keys():
            raise NotImplementedError('Updated CLIP evaluation code will be prodvided soon.')
            # methods = task.methods if args.method == 'all' else [args.method]
            # model, preprocess = clip.load(models_clip[model_name], 'cuda', download_root = args.path_to_weights)
            # task.clip_transform = preprocess
            # model.model_name = model_name
            # model.batch_size = args.batch_size
            # print('Created model {}.'.format(model.model_name))
            # for i, dataset in enumerate(datasets):
            #     task.setup(model, dataset = dataset, clip_model = True)
            #     print('Task is set up.')
            #     task.get_features_clip(model, val = not i, train=need_train_outputs, ood = True, overwrite=args.overwrite_model_outputs)
            #     for method in methods:
            #         print('Getting {} scores...'.format(method))
            #         task.evaluate(model, method = method)
        else:
            raise NotImplementedError(
                '{} is not implemented. Please add it to the model-dictionary.'.format(model_name))


if __name__=="__main__":
    main()