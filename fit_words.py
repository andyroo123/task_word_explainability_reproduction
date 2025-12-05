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


def process_dcm(file_path):
    ds = pydicom.dcmread(file_path)
    im = ds.pixel_array.astype(float)

    # normalize safely
    max_val = im.max()
    if max_val > 0:
        im = im / max_val

    # Handle dimensions:
    # - 2D: (H, W) → stack to (H, W, 3)
    # - 3D: assume last dim might be channels or frames
    if im.ndim == 2:
        # grayscale → make 3-channel RGB
        im = np.stack([im] * 3, axis=-1)  # (H, W, 3)

    elif im.ndim == 3:
        # cases:
        # (H, W, C) where C == 1 or 3 or more
        # or (N, H, W) where N is frames; take first
        if im.shape[-1] in (1, 3, 4):
            # already (H,W,channels)-ish
            if im.shape[-1] == 1:
                # single channel → repeat to 3
                im = np.repeat(im, 3, axis=-1)
            elif im.shape[-1] > 3:
                # trim extra channels
                im = im[..., :3]
        else:
            # likely (N, H, W) → take first frame
            im = im[0]
            if im.ndim == 2:
                im = np.stack([im] * 3, axis=-1)
            elif im.ndim == 3 and im.shape[-1] == 1:
                im = np.repeat(im, 3, axis=-1)

    # final safety: if still not (H, W, 3), squeeze leading dims
    while im.ndim > 3:
        im = im[0]
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    if im.ndim == 3 and im.shape[-1] > 3:
        im = im[..., :3]

    im = (255 * im).clip(0, 255).astype(np.uint8)
    return im


def create_clip_feature_mat(file_list, clip_model, preprocess_fxn):
    X = np.zeros((len(file_list), 512)) # 512 is feature dimension
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


def fit_words(train_df, test_df, device, word_list, save_dir, save_tag):
    clip_model, preprocess_fxn = clip.load("ViT-B/32", device=device)
    clip_model = clip_model.to(device)
    X_train = create_clip_feature_mat(train_df.file_path.values, clip_model, preprocess_fxn)

    classifier = LogisticRegression(random_state=0, C=1, max_iter=1000, verbose=1, fit_intercept=False)
    classifier.fit(X_train, train_df.label.values)

    tokened_words = clip.tokenize(word_list).to(device)
    with torch.no_grad():
        word_features = clip_model.encode_text(tokened_words)

    weights_model = LinearRegression(fit_intercept=False)
    weights_model.fit(word_features.cpu().T, classifier.coef_[0])
    word_df = pd.DataFrame({'word': words, 'weights': weights_model.coef_})
    word_df.sort_values('weights', inplace=True)
    word_df.set_index('word', inplace=True)
    word_df.to_csv(os.path.join(save_dir, f'word_weights-{save_tag}.csv'))

    X_test = create_clip_feature_mat(test_df.file_path.values, clip_model, preprocess_fxn)
    yhat = classifier.predict_proba(X_test)
    print('test acc: ', classifier.score(X_test, test_df.label))

    pred_coef = weights_model.predict(word_features.cpu().T)
    cos_sim = 1 - distance.cosine(pred_coef, classifier.coef_[0])
    print('cosine sim between weights', cos_sim)


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

    words = [
        'dark', 'light',
        'round', 'pointed',
        'large', 'small',
        'smooth', 'coarse',
        'transparent', 'opaque',
        'symmetric', 'asymmetric',
        'high contrast', 'low contrast'
    ]

    base_out_dir = './results/'
    if not os.path.exists(base_out_dir):
        os.mkdir(base_out_dir)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    save_tag = dataset_name
    save_dir = base_out_dir + save_tag + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    fit_words(train_df, test_df, device, words, save_dir=save_dir, save_tag=save_tag)

    prot_save_dir = os.path.join(save_dir, save_tag + '_prototypes/')
    if not os.path.exists(prot_save_dir):
        os.mkdir(prot_save_dir)
    get_prototypes(train_df, words, device, prot_save_dir, n_save=5)
