import os
from scipy.special import logsumexp
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from numpy.linalg import norm, pinv
from tqdm import tqdm
import torch
import torch.nn.functional as F
from itertools import groupby
import faiss

def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


# much of the following is built upon code from https://github.com/haoqiwang/vim/blob/master/benchmark.py

def evaluate_MSP(softmax_id_val, softmax_ood):
    """
    Evaluate Maximum Softmax Probability (MSP) for given softmax values of in-distribution (id) and out-of-distribution (ood) data.

    Inputs:
        softmax_id_val: Numpy array of shape (m, n), representing the softmax probabilities of m data points from in-distribution.
        softmax_ood: Numpy array of shape (p, n), representing the softmax probabilities of p data points from out-of-distribution.
        
    Outputs:
        score_id: Numpy array of shape (m,), representing the maximum softmax probability of m data points from in-distribution.
        score_ood: Numpy array of shape (p,), representing the maximum softmax probability of p data points from out-of-distribution.
        """
    score_id = softmax_id_val.max(axis = -1)
    score_ood = softmax_ood.max(axis = -1)
    return score_id, score_ood


def evaluate_MaxLogit(logits_in_distribution, logits_out_of_distribution):
    """Compute the maximum logit value for both in- and out-of-distribution data.

    Args:
        logits_in_distribution (ndarray): Logits for the in-distribution data.
        logits_out_of_distribution (ndarray): Logits for the out-of-distribution data.

    Returns:
        tuple: Tuple of the maximum logit value for both in- and out-of-distribution data.
    """
    score_in_distribution = logits_in_distribution.max(axis = -1)
    score_out_of_distribution = logits_out_of_distribution.max(axis = -1)
    return score_in_distribution, score_out_of_distribution

def evaluate_Energy(logits_in_distribution, logits_out_of_distribution):
    """Compute the energy value for both in- and out-of-distribution data.

    Args:
        logits_in_distribution (ndarray): Logits for the in-distribution data.
        logits_out_of_distribution (ndarray): Logits for the out-of-distribution data.

    Returns:
        tuple: Tuple of the energy value for both in- and out-of-distribution data.
    """
    score_in_distribution = logsumexp(logits_in_distribution, axis=1)
    score_out_of_distribution = logsumexp(logits_out_of_distribution, axis=1)
    return score_in_distribution, score_out_of_distribution

def evaluate_ViM(feature_id_train, feature_id_val, feature_ood, logits_id_train, logits_id_val, logits_ood, u, path):
    """
    This function evaluates the performance of the ViM out-of-distribution detection method.

    Inputs:

        feature_id_train: numpy array of shape (n, d), the training set features for the in-distribution data.
        feature_id_val: numpy array of shape (m, d), the validation set features for the in-distribution data.
        feature_ood: numpy array of shape (p, d), the features for the out-of-distribution data.
        logits_id_train: numpy array of shape (n, k), the logits for the in-distribution training set.
        logits_id_val: numpy array of shape (m, k), the logits for the in-distribution validation set.
        logits_ood: numpy array of shape (p, k), the logits for the out-of-distribution data.
        u: numpy array of shape (d,), the mean feature vector.
        path: string, the path to store intermediate results.

    Outputs:

        score_id: numpy array of shape (m,), the ViM scores for the in-distribution validation set.
        score_ood: numpy array of shape (p,), the ViM scores for the out-of-distribution data.
        """
    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else (
        512 if feature_id_val.shape[-1] >= 768 else int(feature_id_val.shape[-1] / 2))
    print(f'{DIM=}')
    print('Reading alpha and NS')
    alpha_path = os.path.join(path, 'alpha.npy')
    NS_path = os.path.join(path, 'NS.npy')
    if os.path.exists(NS_path):
        NS = np.load(NS_path)
    else:
        print('NS not stored, computing principal space...')
        ec = EmpiricalCovariance(assume_centered = True)
        ec.fit(feature_id_train - u)
        eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
        NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
        np.save(NS_path, NS)
    if os.path.exists(alpha_path):
        alpha = np.load(alpha_path)
    else:
        print('alpha not stored, computing alpha...')
        vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis = -1)
        alpha = logits_id_train.max(axis = -1).mean() / vlogit_id_train.mean()
        np.save(alpha_path, alpha)
    print(f'{alpha=:.4f}')

    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis = -1) * alpha
    energy_id_val = logsumexp(logits_id_val, axis = -1)
    score_id = -vlogit_id_val + energy_id_val

    energy_ood = logsumexp(logits_ood, axis = -1)
    vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis = -1) * alpha
    score_ood = -vlogit_ood + energy_ood
    return score_id, score_ood


def evaluate_Mahalanobis(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    """
    This function computes Mahalanobis scores for in-distribution and out-of-distribution samples.

    Parameters:
    feature_id_train (numpy array): The in-distribution training samples.
    feature_id_val (numpy array): The in-distribution validation samples.
    feature_ood (numpy array): The out-of-distribution samples.
    train_labels (numpy array): The labels of the in-distribution training samples.
    path (str): The path to save and load the mean and precision matrix.

    Returns:
    tuple: The Mahalanobis scores for in-distribution validation and out-of-distribution samples.

    """
    # load mean and prec
    mean_path = os.path.join(path, 'mean.npy')
    prec_path = os.path.join(path, 'prec.npy')
    complete = True
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        complete = False
    if os.path.exists(prec_path):
        prec = np.load(prec_path)
    else:
        complete = False
    if not complete:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis = 0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered = True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = np.array(train_means)
        prec = (ec.precision_)
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    print('go to gpu...')
    mean = torch.from_numpy(mean).cuda().double()
    prec = torch.from_numpy(prec).cuda().double()
    print('Computing scores...')
    score_id_path = os.path.join(path, 'maha_id_scores.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
    else:
        score_id = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis = -1).min().cpu().item() for f in
                              tqdm(torch.from_numpy(feature_id_val).cuda().double())])
        np.save(score_id_path, score_id)

    score_ood = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis = -1).min().cpu().item() for f in
                           tqdm(torch.from_numpy(feature_ood).cuda().double())])
    return score_id, score_ood


def evaluate_Relative_Mahalanobis(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    """
      This function computes the relative Mahalanobis scores for in-distribution and out-of-distribution samples.

      Parameters:
      feature_id_train (numpy array): The in-distribution training samples.
      feature_id_val (numpy array): The in-distribution validation samples.
      feature_ood (numpy array): The out-of-distribution samples.
      train_labels (numpy array): The labels of the in-distribution training samples.
      path (str): The path to save and load the mean and precision matrix.

      Returns:
      tuple: The relative Mahalanobis scores for in-distribution validation and out-of-distribution samples.

      Steps:
      - Load class-wise mean and precision from disk if they exist, otherwise compute them from the ID training samples and save to disk.
      - Load global mean and precision from disk if they exist, otherwise compute them from all the ID training samples and save to disk.
      - Compute the relative Mahalanobis scores for ID validation samples and save to disk if they don't exist.
      - Compute the relative Mahalanobis scores for OOD samples and save to disk.

      """
    # load class-wise mean and prec
    mean_path = os.path.join(path, 'mean.npy')
    prec_path = os.path.join(path, 'prec.npy')
    complete = True
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        complete = False
    if os.path.exists(prec_path):
        prec = np.load(prec_path)
    else:
        complete = False
    if not complete:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis = 0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)

        print('computing precision matrix...')
        ec = EmpiricalCovariance(assume_centered = True)
        ec.fit(np.array(train_feat_centered).astype(np.float64))

        mean = np.array(train_means)
        prec = (ec.precision_)
        np.save(mean_path, mean)
        np.save(prec_path, prec)
    # print('go to gpu with class-wise...')
    mean = torch.from_numpy(mean).cuda().double()
    prec = torch.from_numpy(prec).cuda().double()

    # load global mean and prec - stay on cpu and use numpy for better precision
    mean_path_global = os.path.join(path, 'mean-global.npy')
    prec_path_global = os.path.join(path, 'prec-global.npy')
    complete = True
    if os.path.exists(mean_path_global):
        mean_global = np.load(mean_path_global)
    else:
        complete = False
    if os.path.exists(prec_path_global):
        prec_global = np.load(prec_path_global)
    else:
        complete = False
    if not complete:
        print('not complete, computing global mean feature...')
        train_means_global = []
        train_feat_centered_global = []

        _m_global = feature_id_train.mean(axis = 0)
        train_means_global.append(_m_global)
        train_feat_centered_global.extend(feature_id_train - _m_global)

        print('computing precision matrix...')
        ec_global = EmpiricalCovariance(assume_centered = True)
        ec_global.fit(np.array(train_feat_centered_global).astype(np.float64))

        mean_global = np.array(train_means_global)
        prec_global = (ec_global.precision_)
        np.save(mean_path_global, mean_global)
        np.save(prec_path_global, prec_global)

    print('Computing scores...')
    score_id_path = os.path.join(path, 'rel_maha_id_scores.npy')
    if os.path.exists(score_id_path):
        score_id = np.load(score_id_path)
    else:
        score_id_path_classwise = os.path.join(path, 'maha_id_scores.npy')
        if os.path.exists(score_id_path_classwise):
            score_id_classwise = np.load(score_id_path_classwise)
        else:
            score_id_classwise = -np.array(
                [(((f - mean) @ prec) * (f - mean)).sum(axis = -1).min().cpu().item() for f in
                 tqdm(torch.from_numpy(feature_id_val).cuda().double())])
            np.save(score_id_path_classwise, score_id_classwise)
        #
        score_id_global = -np.array(
            [((((f - mean_global) @ prec_global) * (f - mean_global)).sum(axis = -1)).item() for f in
             tqdm((feature_id_val))])  # tqdm(torch.from_numpy(feature_id_val).cuda().float())])

        score_id = score_id_classwise - score_id_global
        np.save(score_id_path, score_id)

    score_ood_classwise = -np.array([(((f - mean) @ prec) * (f - mean)).sum(axis = -1).min().cpu().item() for f in
                                     tqdm(torch.from_numpy(feature_ood).cuda().double())])
    score_ood_global = -np.array(
        [((((f - mean_global) @ prec_global) * (f - mean_global)).sum(axis = -1)).item() for f in tqdm(feature_ood)])
    score_ood = score_ood_classwise - score_ood_global
    return score_id, score_ood


def evaluate_KL_Matching(softmax_id_train, softmax_id_val, softmax_ood, path):
    """
    Evaluate KL Matching between softmax output of trained classifier and validation/out-of-distribution data.

    Inputs:
    softmax_id_train (ndarray): Softmax output of classifier on training data. Shape: (num_training_samples, num_classes)
    softmax_id_val (ndarray): Softmax output of classifier on validation data. Shape: (num_validation_samples, num_classes)
    softmax_ood (ndarray): Softmax output of classifier on out-of-distribution data. Shape: (num_ood_samples, num_classes)
    path (str): Path to directory where mean_softmax_train.npy and score_id_KL.npy should be stored/loaded from

    Outputs:
    score_id (ndarray): KL Matching score between softmax_id_val and mean_softmax_train. Shape: (num_validation_samples,)
    score_ood (ndarray): KL Matching score between softmax_ood and mean_softmax_train. Shape: (num_ood_samples,)
    """
    mean_softmax_train_path = os.path.join(path, 'mean_softmax_train.npy')
    score_id_KL_path = os.path.join(path, 'score_id_KL.npy')
    if os.path.exists(mean_softmax_train_path):
        mean_softmax_train = np.load(mean_softmax_train_path)
    else:
        print('not complete, computing classwise mean softmax...')
        pred_labels_train = np.argmax(softmax_id_train, axis = -1)
        mean_softmax_train = np.array(
            [softmax_id_train[pred_labels_train == i].mean(axis = 0) for i in tqdm(range(1000))])
        np.save(mean_softmax_train_path, mean_softmax_train)
    if os.path.exists(score_id_KL_path):
        score_id = np.load(score_id_KL_path)
    else:
        print('not complete, Computing id score...')
        score_id = -pairwise_distances_argmin_min(softmax_id_val, (mean_softmax_train), metric = kl)[1]
        print('score_id is nan: ', np.isnan(score_id).any())
        np.save(score_id_KL_path, score_id)
    print('Computing OOD score...')
    score_ood = -pairwise_distances_argmin_min(softmax_ood, (mean_softmax_train), metric = kl)[1]
    return score_id, score_ood


def evaluate_Energy_React(feature_id_train, feature_id_val, feature_ood, w, b, path, clip_quantile=0.99):
    """Evaluate Energy React Score

       The function evaluates Energy React Score by computing score_id and score_ood.

       Parameters
       ----------
       feature_id_train: np.ndarray
           Input features of the training set.
       feature_id_val: np.ndarray
           Input features of the validation set.
       feature_ood: np.ndarray
           Input features of the out-of-distribution set.
       w: np.ndarray
           Weight matrix of classifiers last layer
       b: np.ndarray
           Bias vector of classifiers last layer
       path: str
           Path to store intermediate values.
       clip_quantile: float, optional, default 0.99
           Quantile used for clipping the input features.

       Returns
       -------
       score_id: np.ndarray
           Energy Reactivity Score for validation set.
       score_ood: np.ndarray
           Energy Reactivity Score for out-of-distribution set.
       """
    clip_react_path = os.path.join(path, 'clip_react.npy')
    if os.path.exists(clip_react_path):
        clip = np.load(clip_react_path)
    else:
        clip = np.quantile(feature_id_train, clip_quantile)
        np.save(clip_react_path, clip)
    print(f'clip quantile {clip_quantile}, clip {clip:.4f}')
    score_id_energy_react_path = os.path.join(path, 'score_id_energy_react.npy')
    if os.path.exists(score_id_energy_react_path):
        score_id = np.load(score_id_energy_react_path)
    else:
        print('not complete, Computing id score...')
        logit_id_val_clip = np.clip(feature_id_val, a_min = None, a_max = clip) @ w.T + b
        score_id = logsumexp(logit_id_val_clip, axis = -1)
        np.save(score_id_energy_react_path, score_id)
    logit_ood_clip = np.clip(feature_ood, a_min = None, a_max = clip) @ w.T + b
    score_ood = logsumexp(logit_ood_clip, axis = -1)
    return score_id, score_ood



def evaluate_KNN(feature_id_train, feature_id_val, feature_ood, path):
    """
    Evaluate KNN classification for in-distribution (ID) and out-of-distribution (OOD) samples.

    This function computes KNN scores for ID and OOD samples. The KNN scores are computed as the distance to the K nearest neighbour of the ID samples in a preprocessed feature space.

    Args:
    feature_id_train_prepos (numpy.ndarray): Preprocessed features of ID training samples.
    feature_id_val (numpy.ndarray): Features of ID validation samples.
    feature_ood (numpy.ndarray): Features of OOD samples.
    path (str): File path to save intermediate computations.

    Returns:
    Tuple of numpy.ndarray:
    score_id (numpy.ndarray): KNN scores of ID validation samples.
    score_ood (numpy.ndarray): KNN scores of OOD samples.

    """
    normalizer = lambda x: x / np.linalg.norm(x, axis = -1, keepdims = True) + 1e-10
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))

    scores_id_path_knn = os.path.join(path, 'scores_id_knn.npy')
    index_path = os.path.join(path,'trained.index')

    # compute neighbours
    K = 1000
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        print('Index not stored, creating index...')
        feature_id_train_prepos=prepos_feat(feature_id_train)
        index = faiss.IndexFlatL2(feature_id_train_prepos.shape[1])
        index.add(feature_id_train_prepos)
        faiss.write_index(index, index_path)

    if os.path.exists(scores_id_path_knn):
        score_id = np.load(scores_id_path_knn)
    else:
        print('Computing id knn scores...')
        ftest = prepos_feat(feature_id_val).astype(np.float32)
        D, _ = index.search(ftest, K, )
        score_id = -D[:, -1]
        np.save(scores_id_path_knn, score_id)

    print('Computing ood knn scores...')
    food = prepos_feat(feature_ood)
    D, _ = index.search(food, K)
    score_ood = -D[:, -1]
    return score_id, score_ood


def evaluate_cosine(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    '''
    Like Cosine for CLIP, but with class-wise mean-features as encoded text:

    This function loads the mean of the in-distribution features, or computes and saves it if not found,
    and computes the cosine similarity scores between the in-distribution and out-of-distribution inputs and the mean.

    Parameters:
    feature_id_train (np.array): In-distribution training features.
    feature_id_val (np.array): In-distribution validation features.
    feature_ood (np.array): Out-of-distribution features.
    train_labels (np.array): Labels for in-distribution training data.
    path (str): Path to save and load mean.

    Returns:
    tuple: In-distribution and out-of-distribution cosine similarity scores.
    '''
    # load mean
    mean_path = os.path.join(path, 'mean.npy')
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis = 0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)
        mean = np.array(train_means)
        np.save(mean_path, mean)
    means_n = np.array([m / np.linalg.norm(m) for m in mean])
    features_id_normalized = np.array([m / np.linalg.norm(m) for m in feature_id_val])
    score_id = (features_id_normalized @ means_n.T).max(axis = -1)
    features_ood_normalized = np.array([m / np.linalg.norm(m) for m in feature_ood])
    score_ood = (features_ood_normalized @ means_n.T).max(axis = -1)
    return score_id, score_ood


def evaluate_rcos(feature_id_train, feature_id_val, feature_ood, train_labels, path):
    '''
    Like MCM, but with class-wise mean-features as encoded text:
    This function loads the mean of the in-distribution data, or computes and saves it if not found,
    and computes the softmax of the cosine similarity scores between the in-distribution and out-of-distribution
    inputs and the mean.

    Parameters:
    feature_id_train (np.array): In-distribution training features.
    feature_id_val (np.array): In-distribution validation features.
    feature_ood (np.array): Out-of-distribution features.
    train_labels (np.array): Labels for in-distribution training data.
    path (str): Path to save and load mean.

    Returns:
    tuple: In-distribution and out-of-distribution re-scaled cosine similarity scores.
    '''
    T = 1.
    mean_path = os.path.join(path, 'mean.npy')
    if os.path.exists(mean_path):
        mean = np.load(mean_path)
    else:
        print('not complete, computing classwise mean feature...')
        train_means = []
        train_feat_centered = []
        for i in tqdm(range(1000)):
            fs = feature_id_train[train_labels == i]
            _m = fs.mean(axis = 0)
            train_means.append(_m)
            train_feat_centered.extend(fs - _m)
        mean = np.array(train_means)
        np.save(mean_path, mean)

    # use train means as encoded text pairs
    text_encoded = torch.from_numpy(mean).float()
    text_encoded /= text_encoded.norm(dim = -1, keepdim = True)

    scores_id_path_clip = os.path.join(path, 'mcm_scores_id.npy')
    if os.path.exists(scores_id_path_clip):
        score_id = np.load(scores_id_path_clip)
    else:
        print('Computing ID scores...')

        features_id = torch.from_numpy(feature_id_val).float()
        features_id /= features_id.norm(dim = -1, keepdim = True)

        out_id = features_id @ text_encoded.T
        smax_id = F.softmax(out_id / T, dim = 1).data.cpu().numpy()
        score_id = np.max(smax_id, axis = 1)

        np.save(scores_id_path_clip, score_id)
    print('Computing OOD scores...')

    features_ood = torch.from_numpy(feature_ood).float()
    features_ood /= features_ood.norm(dim = -1, keepdim = True)

    out_ood = features_ood @ text_encoded.T
    smax_ood = F.softmax(out_ood / T, dim = 1).data.cpu().numpy()
    score_ood = np.max(smax_ood, axis = 1)
    return score_id, score_ood


def evaluate_cosine_clip(feature_id_val, feature_ood, clip_labels, labels_encoded_clip, path):
    """
       Evaluates cosine similarity scores for in-distribution and out-of-distribution samples and returns the scores,
       along with the in-distribution accuracy for CLIP features.

       Parameters:
       feature_id_val (np.ndarray): In-distribution validation feature tensor.
       feature_ood (np.ndarray): Out-of-distribution feature tensor.
       clip_labels (np.ndarray): Ground truth labels for the in-distribution validation samples.
       labels_encoded_clip (np.ndarray): Encoded ground truth labels for the in-distribution samples.
       path (str): Path to the directory to save the scores.

       Returns:
       tuple:
           score_id (np.ndarray): Cosine similarity scores for the in-distribution samples.
           score_ood (np.ndarray): Cosine similarity scores for the out-of-distribution samples.
           val_acc (float): In-distribution accuracy.
       """
    text_encoded = np.array([m / np.linalg.norm(m) for m in
                             labels_encoded_clip])  # labels_encoded_clip / labels_encoded_clip.norm(dim = -1, keepdim = True)
    scores_id_path_clip = os.path.join(path, 'cosine-clip_scores_id.npy')
    acc_path = os.path.join(path, 'accuracy.npy')
    if os.path.exists(scores_id_path_clip):
        score_id = np.load(scores_id_path_clip)
        val_acc = np.load(acc_path)
    else:
        print('Computing ID scores...')
        x_val_id_encoded = np.array([m / np.linalg.norm(m) for m in feature_id_val])
        # feature_id_val / feature_id_val.norm(dim = -1, keepdim = True)
        similarity_id = (x_val_id_encoded @ text_encoded.T)
        preds = np.argmax(similarity_id, axis = -1)
        val_acc = np.equal(preds, clip_labels).mean()
        np.save(acc_path, val_acc)
        score_id = np.max(similarity_id, axis = -1)
        np.save(scores_id_path_clip, score_id)
    print('Computing OOD scores...')
    x_ood_encoded = np.array([m / np.linalg.norm(m) for m in feature_ood])
    # feature_ood / feature_ood.norm(dim = -1, keepdim = True)

    similarity_ood = (x_ood_encoded @ text_encoded.T)
    score_ood = np.max(similarity_ood, axis = -1)
    return score_id, score_ood, val_acc


def evaluate_mcm_clip(feature_id_val, feature_ood, clip_labels, labels_encoded_clip, path):
    """
    This function computes the MCM score for a given set of ID data (feature_id_val) and OOD data (feature_ood)
    by first normalizing the features and then computing the dot product between the features and the encoded
    text representations (labels_encoded_clip). The resulting scores are then passed through a softmax function
    to obtain the final MCM scores. The ID scores are saved to disk (mcm-clip_scores_id.npy) along with the
    accuracy (accuracy.npy) if they have not already been computed.

    Inputs:

        feature_id_val: numpy array, shape (num_ID_data, num_features)
        The in-distribution data to evaluate the MCM scores for.
        feature_ood: numpy array, shape (num_OOD_data, num_features)
        The out-of-distribution data to evaluate the MCM scores for.
        clip_labels: numpy array, shape (num_ID_data,)
        The labels for the in-distribution data.
        labels_encoded_clip: numpy array, shape (num_texts, num_features)
        The encoded text representations.
        path: str
        The path to save the ID scores and accuracy if they have not already been computed.

    Returns:

        score_id: numpy array, shape (num_ID_data,)
        The MCM scores for the in-distribution data.
        score_ood: numpy array, shape (num_OOD_data,)
        The MCM scores for the out-of-distribution data.
        val_acc: float
        The accuracy of the prediction on the in-distribution validation data.
        """
    T = 1.
    text_encoded = torch.from_numpy(labels_encoded_clip).float()
    text_encoded /= text_encoded.norm(dim = -1, keepdim = True)

    scores_id_path_clip = os.path.join(path, 'mcm-clip_scores_id.npy')
    acc_path = os.path.join(path, 'accuracy.npy')
    if os.path.exists(scores_id_path_clip):
        score_id = np.load(scores_id_path_clip)
        val_acc = np.load(acc_path)
    else:
        print('Computing ID scores...')

        features_id = torch.from_numpy(feature_id_val).float()
        features_id /= features_id.norm(dim = -1, keepdim = True)

        out_id = features_id @ text_encoded.T
        smax_id = F.softmax(out_id / T, dim = 1).data.cpu().numpy()
        score_id = np.max(smax_id, axis = 1)

        preds = np.argmax(out_id.data.cpu().numpy(), axis = -1)
        val_acc = np.equal(preds, clip_labels).mean()
        np.save(acc_path, val_acc)
        np.save(scores_id_path_clip, score_id)
    print('Computing OOD scores...')

    features_ood = torch.from_numpy(feature_ood).float()
    features_ood /= features_ood.norm(dim = -1, keepdim = True)

    out_ood = features_ood @ text_encoded.T
    smax_ood = F.softmax(out_ood / T, dim = 1).data.cpu().numpy()
    score_ood = np.max(smax_ood, axis = 1)

    return score_id, score_ood, val_acc