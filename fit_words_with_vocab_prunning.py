from scipy.spatial import distance
import numpy as np
import pandas as pd
from PIL import Image
import torch
import clip
import pydicom
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LinearRegression
import os
import imageio
import cv2
import torch_directml
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score

def encode_words_in_batches(clip_model, words, device, batch_size=256):
    tokenized = clip.tokenize(words)

    all_feats = []
    clip_model.eval()

    with torch.no_grad():
        for i in range(0, len(words), batch_size):
            batch_tok = tokenized[i:i+batch_size].to(device)
            batch_feats = clip_model.encode_text(batch_tok)
            all_feats.append(batch_feats.cpu())

    word_features = torch.cat(all_feats, dim=0)
    return word_features


def eval_from_weights(X, y, w):
    """
    X: (n_samples, d)
    y: (n_samples,) numeric 0/1
    w: (d,) classifier weights
    """
    scores = X @ w
    probs = 1 / (1 + np.exp(-scores))

    y_pred = (probs >= 0.5).astype(int)

    acc = accuracy_score(y, y_pred)

    if len(np.unique(y)) < 2:
        auroc = np.nan
        print("[eval_from_weights] Only one class present in y_true; AUROC is undefined.")
    else:
        auroc = roc_auc_score(y, probs)

    return acc, auroc



def prune_vocabulary(word_list, word_features, weights_model, top_k=None, min_abs_weight=None):
    """
    Prune the vocabulary based on the absolute value of the learned word weights.

    Args:
        word_list (list[str]): original list of words
        word_features (torch.Tensor): (num_words, d) CLIP text features
        weights_model (sklearn.linear_model.LinearRegression): fitted on all words
        top_k (int or None): keep only the top_k words by |weight|
        min_abs_weight (float or None): keep words whose |weight| >= min_abs_weight

    Returns:
        pruned_words (list[str])
        pruned_word_features (torch.Tensor)
        pruned_weights (np.ndarray)
        keep_idx (np.ndarray): indices of words kept
    """
    weights = weights_model.coef_
    abs_w = np.abs(weights)

    keep_mask = np.ones_like(abs_w, dtype=bool)

    if min_abs_weight is not None:
        keep_mask &= abs_w >= min_abs_weight

    if top_k is not None and top_k < len(word_list):
        topk_idx = np.argsort(abs_w)[-top_k:]
        topk_mask = np.zeros_like(abs_w, dtype=bool)
        topk_mask[topk_idx] = True
        keep_mask &= topk_mask

    keep_idx = np.where(keep_mask)[0]

    if keep_idx.size == 0:
        if top_k is not None and top_k > 0:
            keep_idx = np.argsort(abs_w)[-top_k:]
        else:
            keep_idx = np.arange(len(word_list))

    pruned_words = [word_list[i] for i in keep_idx]
    pruned_word_features = word_features[keep_idx]
    pruned_weights = weights[keep_idx]

    return pruned_words, pruned_word_features, pruned_weights, keep_idx

def process_dcm(file_path, max_side=1024):
    """
    Load a DICOM and return a uint8 RGB image, optionally downsampled so that
    max(height, width) <= max_side to avoid huge memory usage.
    """
    ds = pydicom.dcmread(file_path)
    im = ds.pixel_array.astype(np.float32)

    max_val = float(im.max())
    if max_val > 0:
        im /= max_val

    if im.ndim == 3 and im.shape[-1] not in (1, 3, 4) and im.shape[0] > 1:
        im = im[0]

    if im.ndim >= 2:
        h, w = im.shape[:2]
        longest = max(h, w)
        if longest > max_side:
            scale = max_side / float(longest)
            new_w = int(w * scale)
            new_h = int(h * scale)
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    elif im.ndim == 3 and im.shape[-1] == 1:
        im = np.repeat(im, 3, axis=-1)
    elif im.ndim == 3 and im.shape[-1] > 3:
        im = im[..., :3]

    while im.ndim > 3:
        im = im[0]
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)

    np.multiply(im, 255.0, out=im)
    np.clip(im, 0, 255, out=im)
    im = im.astype(np.uint8)

    return im



def create_clip_feature_mat(file_list, clip_model, preprocess_fxn):
    X = np.zeros((len(file_list), 512))
    for i, f in tqdm(enumerate(file_list), total=len(file_list)):
        if '.dcm' in f:
            im = Image.fromarray(process_dcm(f))
        else:
            im = Image.open(f)
        im = preprocess_fxn(im).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = clip_model.encode_image(im)
        X[i] = image_features[0].cpu()

    return X


def fit_words(train_df,
              test_df,
              device,
              word_list,
              save_dir,
              save_tag,
              prune_top_k=None,
              prune_min_abs_weight=None):
    """
    Fit a logistic regression classifier on CLIP image features,
    then fit a linear regression from CLIP text features (word embeddings)
    to the classifier weights. Optionally prune the vocabulary and refit.

    Args:
        train_df, test_df: DataFrames with 'file_path' and 'label'
        device: torch device
        word_list: list of candidate words (strings)
        save_dir: directory to save outputs
        save_tag: dataset name / tag
        prune_top_k: if not None, keep only top_k words by |weight|
        prune_min_abs_weight: if not None, keep only words with |weight| >= threshold
    """
    clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.to(device)

    X_train = create_clip_feature_mat(train_df.file_path.values, clip_model, preprocess_fxn)

    classifier = LogisticRegression(
        random_state=0,
        C=1,
        max_iter=1000,
        verbose=1,
        fit_intercept=False
    )
    classifier.fit(X_train, train_df.label.values)

    tokened_words = clip.tokenize(word_list).to(device)
    with torch.no_grad():
        word_features = clip_model.encode_text(tokened_words)

    weights_model = LinearRegression(fit_intercept=False)
    weights_model.fit(word_features.cpu().T, classifier.coef_[0])

    full_word_df = pd.DataFrame({'word': word_list, 'weights': weights_model.coef_})
    full_word_df.sort_values('weights', inplace=True)
    full_word_df.set_index('word', inplace=True)
    full_csv_path = os.path.join(save_dir, f'word_weights-{save_tag}.csv')
    full_word_df.to_csv(full_csv_path)
    print(f"[fit_words] Saved full vocabulary weights to: {full_csv_path}")

    if prune_top_k is not None or prune_min_abs_weight is not None:
        pruned_words, pruned_word_features, pruned_weights, keep_idx = prune_vocabulary(
            word_list=word_list,
            word_features=word_features,
            weights_model=weights_model,
            top_k=prune_top_k,
            min_abs_weight=prune_min_abs_weight
        )

        print(f"[fit_words] Pruned vocabulary from {len(word_list)} -> {len(pruned_words)} words")

        pruned_reg = LinearRegression(fit_intercept=False)
        pruned_reg.fit(pruned_word_features.cpu().T, classifier.coef_[0])

        pruned_word_df = pd.DataFrame({'word': pruned_words, 'weights': pruned_reg.coef_})
        pruned_word_df.sort_values('weights', inplace=True)
        pruned_word_df.set_index('word', inplace=True)
        pruned_csv_path = os.path.join(save_dir, f'word_weights-{save_tag}-pruned.csv')
        pruned_word_df.to_csv(pruned_csv_path)
        print(f"[fit_words] Saved PRUNED vocabulary weights to: {pruned_csv_path}")
    else:
        print("[fit_words] No vocabulary pruning applied.")

    X_test = create_clip_feature_mat(test_df.file_path.values, clip_model, preprocess_fxn)
    y_test = test_df.label.values
    yhat = classifier.predict_proba(X_test)[:, 1]
    test_acc = classifier.score(X_test, y_test)
    try:
        test_auroc = roc_auc_score(y_test, yhat)
    except ValueError:
        test_auroc = np.nan
    print('test acc (original classifier): ', test_acc)
    print('test AUROC (original classifier): ', test_auroc)

    pred_coef_full = weights_model.predict(word_features.cpu().T)
    cos_sim = 1 - distance.cosine(pred_coef_full, classifier.coef_[0])
    print('cosine sim between weights (full vocab):', cos_sim)

    return {
        "clip_model": clip_model,
        "preprocess_fxn": preprocess_fxn,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": train_df.label.values,
        "y_test": y_test,
        "word_list": word_list,
        "word_features": word_features,
        "weights_model": weights_model,
        "classifier_coef": classifier.coef_[0],
        "full_test_acc": test_acc,
        "full_test_auroc": test_auroc,
        "full_cos_sim": cos_sim,
    }

def get_prototypes(df, words, device, save_dir, n_save=20):
    clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)
    X = create_clip_feature_mat(df.file_path.values, clip_model, preprocess_fxn)

    tokened_words = clip.tokenize(words).to(device)
    with torch.no_grad():
        word_features = clip_model.encode_text(tokened_words)

    file_dot = np.zeros((len(df), len(words)))
    for i in range(len(df)):
        for j in range(len(words)):
            file_dot[i, j] = np.dot(X[i], word_features[j].cpu())

    file_dot_pred = np.zeros((len(df), len(words)))
    for j in range(len(words)):
        fit_j = [k for k in range(len(words)) if k != j]
        dot_regression = LinearRegression()
        dot_regression.fit(file_dot[:, fit_j], file_dot[:, j])
        file_dot_pred[:, j] = dot_regression.predict(file_dot[:, fit_j])

    dot_df_diff = pd.DataFrame(file_dot - file_dot_pred, columns=words)
    dot_df_diff['label'] = df['label'].values
    dot_df_diff.set_index(df.file_path, inplace=True)

    for w in words:
        print(w)
        for sort_dir in ['top']:
            this_df = dot_df_diff.sort_values(w, ascending=(sort_dir == 'bottom'))
            save_files = this_df.index.values[:n_save]
            these_labels = this_df.label.values[:n_save]
            this_out_dir = save_dir + w + '_' + sort_dir + '/'
            if not os.path.exists(this_out_dir):
                os.mkdir(this_out_dir)

            for i, f in enumerate(save_files):
                if '.dcm' in f:
                    im = process_dcm(f)
                else:
                    im = imageio.imread(f)
                # make square and downsample for efficiency (CLIP also crops to square)
                min_dim = min(im.shape[:2])
                for dim in [0, 1]:
                    if im.shape[dim] > min_dim:
                        n_start = int((im.shape[dim] - min_dim) / 2)
                        n_stop = n_start + min_dim
                        if dim == 0:
                            im = im[n_start:n_stop, :, :]
                        else:
                            im = im[:, n_start:n_stop, :]
                if min_dim > 500:
                    im = cv2.resize(im, (500, 500))
                f_name = f'rank{i}_label{these_labels[i]}.png'
                imageio.imwrite(os.path.join(this_out_dir, f_name), im)

def sweep_vocab_pruning(
    fit_result,
    save_dir,
    save_tag,
    prune_ks=(2, 4, 6, 8, 10, 12)
):
    """
    Sweep over different vocabulary sizes (top-k by |weight|),
    approximate the classifier weights using ONLY the pruned vocabulary,
    and evaluate test accuracy / AUROC for each k.
    Logs metrics to CSV and plots performance vs vocabulary size.

    fit_result: dict returned by fit_words(...)
    save_dir: output directory
    save_tag: dataset name tag (used in filenames)
    prune_ks: iterable of integers (k values to evaluate)
    """
    word_list = fit_result["word_list"]
    word_features = fit_result["word_features"]
    weights_model = fit_result["weights_model"]
    classifier_coef = fit_result["classifier_coef"]
    X_test = fit_result["X_test"]
    y_test = fit_result["y_test"]

    pred_coef_full = weights_model.predict(word_features.cpu().T)
    full_acc, full_auroc = eval_from_weights(X_test, y_test, pred_coef_full)
    full_cos = 1 - distance.cosine(pred_coef_full, classifier_coef)

    print(f"[sweep] Full-vocab reconstructed weights -> acc={full_acc:.4f}, auroc={full_auroc:.4f}, cos={full_cos:.4f}")

    rows = []
    rows.append({
        "vocab_size": len(word_list),
        "top_k": len(word_list),
        "test_acc": full_acc,
        "test_auroc": full_auroc,
        "cosine_sim": full_cos,
    })

    for k in prune_ks:
        if k > len(word_list):
            continue

        pruned_words, pruned_word_features, pruned_weights, keep_idx = prune_vocabulary(
            word_list=word_list,
            word_features=word_features,
            weights_model=weights_model,
            top_k=k,
            min_abs_weight=None
        )

        pruned_reg = LinearRegression(fit_intercept=False)
        pruned_reg.fit(pruned_word_features.cpu().T, classifier_coef)

        pred_coef_pruned = pruned_reg.predict(pruned_word_features.cpu().T)
        pruned_acc, pruned_auroc = eval_from_weights(X_test, y_test, pred_coef_pruned)
        pruned_cos = 1 - distance.cosine(pred_coef_pruned, classifier_coef)

        print(f"[sweep] k={k:3d}, vocab_size={len(pruned_words):3d}, "
              f"acc={pruned_acc:.4f}, auroc={pruned_auroc:.4f}, cos={pruned_cos:.4f}")

        rows.append({
            "vocab_size": len(pruned_words),
            "top_k": k,
            "test_acc": pruned_acc,
            "test_auroc": pruned_auroc,
            "cosine_sim": pruned_cos,
        })

    metrics_df = pd.DataFrame(rows).sort_values("vocab_size")
    csv_path = os.path.join(save_dir, f"pruning_metrics-{save_tag}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"[sweep] Saved pruning metrics to: {csv_path}")

    plt.figure()
    plt.plot(metrics_df["vocab_size"], metrics_df["test_auroc"], marker="o", label="AUROC")
    plt.plot(metrics_df["vocab_size"], metrics_df["test_acc"], marker="s", label="Accuracy")
    plt.xlabel("Vocabulary size (number of words)")
    plt.ylabel("Performance")
    plt.title(f"Performance vs Vocabulary Size ({save_tag})")
    plt.legend()
    plt.grid(True)
    png_path = os.path.join(save_dir, f"pruning_performance-{save_tag}.png")
    plt.savefig(png_path, bbox_inches="tight")
    plt.close()
    print(f"[sweep] Saved pruning performance plot to: {png_path}")

def balance_df(df):
    benign_df = df[df['label'] == 'benign']
    malignant_df = df[df['label'] == 'malignant']

    target_malignant = int(len(benign_df) / 2)

    target_malignant = min(target_malignant, len(malignant_df))

    malignant_sample = malignant_df.sample(n=target_malignant, random_state=42)

    target_benign = 2 * target_malignant

    benign_sample = benign_df.sample(n=target_benign, random_state=42)

    return pd.concat([benign_sample, malignant_sample]).sample(frac=1, random_state=42)


if __name__ == '__main__':
    dataset_name = 'melanoma'
    device = torch_directml.device()

    # assumes a csv with columns containing file_path and label
    if dataset_name == 'cbis':
        train_path = './data/cbis_mass_train.csv'
        test_path = './data/cbis_mass_test.csv'
    elif dataset_name == 'melanoma':
        train_path = './data/siim_melanoma_train.csv'
        test_path = './data/siim_melanoma_test.csv'

    TRAIN_N = 20000
    TEST_N = 4000

    train_df = balance_df(pd.read_csv(train_path))
    test_df = balance_df(pd.read_csv(test_path))

    if len(train_df) > TRAIN_N:
        train_df = train_df.sample(n=TRAIN_N, random_state=42).reset_index(drop=True)

    if len(test_df) > TEST_N:
        test_df = test_df.sample(n=TEST_N, random_state=42).reset_index(drop=True)

    print("Train label counts:")
    print(train_df["label"].value_counts())
    print("\nTest label counts:")
    print(test_df["label"].value_counts())

    label_map = {
        'benign': 0,
        'malignant': 1,
        0: 0,  # if already numeric, keep as is
        1: 1,
    }

    train_df['label'] = train_df['label'].map(label_map)
    test_df['label'] = test_df['label'].map(label_map)

    if train_df['label'].isna().any() or test_df['label'].isna().any():
        raise ValueError("Some labels could not be mapped to 0/1. Check label_map and CSV contents.")


    words = [
        'dark', 'light',
        'round', 'pointed',
        'large', 'small',
        'smooth', 'coarse',
        'transparent', 'opaque',
        'symmetric', 'asymmetric',
        'high contrast', 'low contrast'
    ]

    with open("vocab.txt") as f:
        words = [line.strip() for line in f if line.strip()]

    base_out_dir = './results/'
    if not os.path.exists(base_out_dir):
        os.mkdir(base_out_dir)

    save_tag = dataset_name
    save_dir = base_out_dir + save_tag + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fit_result = fit_words(
        train_df,
        test_df,
        device,
        words,
        save_dir=save_dir,
        save_tag=save_tag,
        prune_top_k=None,
        prune_min_abs_weight=None
    )

    prune_ks = [k for k in range(10, len(words)+1, 10)]
    if prune_ks[-1] != len(words):
        prune_ks.append(len(words))
    sweep_vocab_pruning(
        fit_result=fit_result,
        save_dir=save_dir,
        save_tag=save_tag,
        prune_ks=prune_ks
    )

    prot_save_dir = os.path.join(save_dir, save_tag + '_prototypes/')
    if not os.path.exists(prot_save_dir):
        os.mkdir(prot_save_dir)
    get_prototypes(train_df, words, device, prot_save_dir, n_save=5)

