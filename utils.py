#!/usr/bin/env python
import torch
from tqdm import tqdm
import os
import clip
from torch.utils.data.dataset import Dataset
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from itertools import groupby

import numpy as np
from sklearn.metrics import roc_auc_score  # type: ignore

timm_models = {
    'BiT_m': {'config': {'model_name': 'resnetv2_101x1_bitm', 'pretrained': True}},
    'BiT_s': {'config': {'model_name': 'resnetv2_101x1_bitm',
                         'checkpoint_path': './model_weights/checkpoints/BiT-S-R101x1.npz'}},
    'vit_base_patch16_224_21kpre': {'config': {'model_name': 'vit_base_patch16_224', 'pretrained': True}},
    'vit_base_patch16_384_21kpre': {'config': {'model_name': 'vit_base_patch16_384', 'pretrained': True}},
    'convnext_base_in22ft1k': {'config': {'model_name': 'convnext_base_in22ft1k', 'pretrained': True}},
    'convnext_base': {'config': {'model_name': 'convnext_base', 'pretrained': True}},
    'convnext_tiny-22k': {'config': {'model_name': 'convnext_tiny_384_in22ft1k', 'pretrained': True}},
    'deit3_base_patch16_224': {'config': {'model_name': 'deit3_base_patch16_224', 'pretrained': True}},
    'deit3_base_patch16_224_in21ft1k': {
        'config': {'model_name': 'deit3_base_patch16_224_in21ft1k', 'pretrained': True}},
    'tf_efficientnetv2_m': {'config': {'model_name': 'tf_efficientnetv2_m', 'pretrained': True}},
    'tf_efficientnetv2_m_in21ft1k': {'config': {'model_name': 'tf_efficientnetv2_m_in21ft1k', 'pretrained': True}},
    'swinv2-22k': {'config': {'model_name': 'swinv2_base_window12to16_192to256_22kft1k', 'pretrained': True}},
    'swinv2-1k': {'config': {'model_name': 'swinv2_base_window16_256', 'pretrained': True}},
    'deit3-384-22k': {'config': {'model_name': 'deit3_base_patch16_384_in21ft1k', 'pretrained': True}},
    'deit3-384-1k': {'config': {'model_name': 'deit3_base_patch16_384', 'pretrained': True}},
    'tf_efficientnet_b7_ns': {'config': {'model_name': 'tf_efficientnet_b7_ns', 'pretrained': True}},
    'tf_efficientnet_b7': {'config': {'model_name': 'tf_efficientnet_b7', 'pretrained': True}},
    'resnet50': {'config': {'model_name': 'resnet50', 'pretrained': True}},
    'efficientnet_b0': {'config': {'model_name': 'efficientnet_b0', 'pretrained': True}},
    'vit_base_patch16_384_laion2b_in12k_in1k': {
        'config': {'model_name': 'vit_base_patch16_clip_384.laion2b_ft_in12k_in1k', 'pretrained': True}},
    'vit_base_patch16_384_laion2b_in1k': {
        'config': {'model_name': 'vit_base_patch16_clip_384.laion2b_ft_in1k', 'pretrained': True}},
    'vit_base_patch16_384_openai_in12k_in1k': {
        'config': {'model_name': 'vit_base_patch16_clip_384.openai_ft_in12k_in1k', 'pretrained': True}},
    'vit_base_patch16_384_openai_in1k': {
        'config': {'model_name': 'vit_base_patch16_clip_384.openai_ft_in1k', 'pretrained': True}},
    'vit_base_patch16_384': {'config': {'model_name': 'vit_base_patch16_384.augreg_in1k', 'pretrained': True},
                             'batch_size': 128, 'server': 'curie'},
    'xcit_medium_24_p16_224_dist': {'config': {'model_name': 'xcit_medium_24_p16_224_dist', 'pretrained': True}},
    'xcit_medium_24_p16_224': {'config': {'model_name': 'xcit_medium_24_p16_224', 'pretrained': True}},
}

NINCO_class_names = ['Caracal caracal caracal', 'amphiuma_means',
                     'aphanizomenon_flosaquae',
                     'araneus_gemma',
                     'arctocephalus_galapagoensis',
                     'batrachoseps_attenuatus',
                     'chicken_quesadilla',
                     'cirsium_pitcheri', 'creme_brulee', 'ctenolepisma_longicaudata', 'cup_cakes',
                     'darlingtonia_californica', 'dendrolagus_lumholtzi', 'donuts', 'epithelantha_micromeris',
                     'erysimum_franciscanum', 'f_field_road', 'f_forest_path', 'ferocactus_pilosus', 'haemulon_sciurus',
                     'hippopus_hippopus', 'lasionycteris_noctivagans', 'lathyrus_odoratus', 'lepomis_auritus',
                     'leptoglossus_phyllopus', 'microcystis_wesenbergii', 'octopus_bimaculoides', 'octopus_rubescens',
                     'ozotoceros_bezoarticus', 'platycephalus_fuscus', 'polistes_dominula', 'pseudorca_crassidens',
                     's_sky', 'sarpa_salpa', 'sarracenia_alata', 'sepia_apama', 'sepia_officinalis',
                     'sepioteuthis_australis', 'spaghetti_bolognese', 'streptopus_lanceolatus', 'tapirus_bairdii',
                     'triturus_marmoratus', 'tursiops_aduncus', 'vaccinium_reticulatum', 'waffles',
                     'skipper_caterpillar']

NINCO_OOD_unit_tests_class_names = ['Gaussian', 'Rademacher', 'black', 'blobs', 'grey', 'horizontal_stripes',
                                    'low_freq', 'low_freq_channelfullscale', 'low_freq_colorrange',
                                    'low_freq_pixel_perm', 'monochrome', 'pixel_perm', 'tricolour', 'tricolour_primary',
                                    'uni', 'vertical_stripes', 'white']

NINCO_popular_datasets_subsamples_class_names = [
    'Places',
    'iNaturalist_OOD_Plants',
    'Species',
    'Imagenet_O',
    'OpenImage_O',
    'Textures'
]

models_timm_dev = ['vit_base_patch16_384_laion2b_in12k_in1k',
                   'vit_base_patch16_384_laion2b_in1k',
                   'vit_base_patch16_384_openai_in1k',
                   'vit_base_patch16_384_openai_in12k_in1k',
                   'vit_base_patch16_384',
                   'xcit_medium_24_p16_224_dist',
                   'xcit_medium_24_p16_224']

models_timm_0_6_12 = ['BiT_m', 'BiT_s', 'vit_base_patch16_224_21kpre', 'vit_base_patch16_384_21kpre',
                      'convnext_base_in22ft1k', 'convnext_base', 'convnext_tiny-22k', 'deit3_base_patch16_224',
                      'deit3_base_patch16_224_in21ft1k', 'tf_efficientnetv2_m', 'tf_efficientnetv2_m_in21ft1k',
                      'swinv2-22k', 'swinv2-1k', 'deit3-384-22k', 'deit3-384-1k', 'tf_efficientnet_b7_ns',
                      'tf_efficientnet_b7', 'resnet50', 'efficientnet_b0', ]

models_clip = {'clip-ViT-B16': 'ViT-B/16', 'clip-ViT-L14-336': 'ViT-L/14@336px'}


def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


def extract_features(model, dataset, savepath, wo_head=False):
    torch.backends.cudnn.benchmark = True
    # save
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    model.eval()

    # slice dataloaders in train
    slice_length = 50000
    n_slices = -((-len(dataset)) // slice_length)
    index_slices = {}
    slice_datasets = {}

    for i in range(n_slices):
        # check if current iterate is already saved in directoy (for train
        files = os.listdir(savepath)
        complete = n_slices>1 # True for train, else False
        for feat in ['logits_{}.npy'.format(i), 'features_{}.npy'.format(i), 'labels_true_{}.npy'.format(i)]:
            if feat not in files:
                complete = False

        # if not, save create dataloader and save features
        # if True: #not complete:
        complete = False
        if not complete:
            print('Extracting features set ', i)
            index_slices[i] = range(i * slice_length, min((i + 1) * slice_length, len(dataset)))
            slice_datasets[i] = torch.utils.data.Subset(dataset, index_slices[i])
            slice_datasets[i].__name__ = f'slice{index_slices[i].start}_to_{index_slices[i].stop}'
            slice_datasets[i].classes = dataset.classes
            dataloader = torch.utils.data.DataLoader(slice_datasets[i], batch_size=model.batch_size)

            features = []
            logits_ = []
            labels_true = []
            with torch.no_grad():
                for (x, label) in tqdm(dataloader):
                    labels_true.append(label)
                    x = x.cuda()
                    feat_batch_preact = model.forward_features(x)
                    feat_batch = feat_batch_preact[:, 0] if wo_head else model.forward_head(feat_batch_preact,
                                                                                            pre_logits=True)
                    logits = model.forward_head(feat_batch_preact)

                    feat_batch = feat_batch.cpu().numpy()
                    logits = logits.cpu().numpy()
                    features.append(feat_batch)
                    logits_.append(logits)

            # save
            labels_true = torch.cat(labels_true).numpy()
            features = np.concatenate(features, axis=0)
            logits_ = np.concatenate(logits_, axis=0)
            predictions_dict = {'logits_{}'.format(i): logits_, 'features_{}'.format(i): features,
                                'labels_true_{}'.format(i): labels_true}

            for name, data in predictions_dict.items():
                np.save(savepath + '/' + name,
                        data)


def extract_clip_embeddings(model, dataset, savepath, text=None, k=250):
    torch.backends.cudnn.benchmark = True
    # save
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=model.batch_size)
    features = []
    labels_true = []
    model.eval()
    with torch.no_grad():
        for i, (x, label) in tqdm(enumerate(dataloader)):
            labels_true.append(label)
            x = x.cuda()
            feat_batch = model.encode_image(x)

            feat_batch = feat_batch.cpu().numpy()
            features.append(feat_batch)
            if (i + 1) % k == 0:  # save every k batches
                labels_true = torch.cat(labels_true).numpy()
                features = np.concatenate(features, axis=0)
                predictions_dict = {'features_{}'.format(i): features,
                                    'labels_true_{}'.format(i): labels_true}

                for name, data in predictions_dict.items():
                    np.save(savepath + '/' + name,
                            data)  # with open(savepath + '/' + name, 'wb') as f:  #     pickle.dump(features, f)
                features = []
                labels_true = []

    # save remaining
    labels_true = torch.cat(labels_true).numpy()
    features = np.concatenate(features, axis=0)
    predictions_dict = {'features_{}'.format(0): features, 'labels_true_{}'.format(0): labels_true}
    # text embeddings of labels
    if text is not None:
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in text]).cuda()
        with torch.no_grad():
            text_encoded = model.encode_text(text_inputs)
        predictions_dict['text_encoded_0'] = text_encoded.cpu().numpy()

    #

    for name, data in predictions_dict.items():
        np.save(savepath + '/' + name,
                data)


def auroc_ood(values_in: np.ndarray, values_out: np.ndarray) -> float:
    """
    Implementation of Area-under-Curve metric for out-of-distribution detection.
    The higher the value the better.

    Args:
        values_in: Maximal confidences (i.e. maximum probability per each sample)
            for in-domain data.
        values_out: Maximal confidences (i.e. maximum probability per each sample)
            for out-of-domain data.

    Returns:
        Area-under-curve score.
    """
    if len(values_in) * len(values_out) == 0:
        return np.NAN
    y_true = len(values_in) * [1] + len(values_out) * [0]
    y_score = np.nan_to_num(np.concatenate([values_in, values_out]).flatten())
    return roc_auc_score(y_true, y_score)


def fpr_at_tpr(values_in: np.ndarray, values_out: np.ndarray, tpr: float) -> float:
    """
    Calculates the FPR at a particular TRP for out-of-distribution detection.
    The lower the value the better.

    Args:
        values_in: Maximal confidences (i.e. maximum probability per each sample)
            for in-domain data.
        values_out: Maximal confidences (i.e. maximum probability per each sample)
            for out-of-domain data.
        tpr: (1 - true positive rate), for which probability threshold is calculated for
            in-domain data.

    Returns:
        False positive rate on out-of-domain data at (1 - tpr) threshold.
    """
    if len(values_in) * len(values_out) == 0:
        return np.NAN
    t = np.quantile(values_in, (1 - tpr))
    fpr = (values_out >= t).mean()
    return fpr
