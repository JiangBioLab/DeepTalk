from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
import numpy as np
#import cv2
import os
import sys
import shutil
import torch
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
matplotlib.use('Agg')
import pandas as pd
from collections import defaultdict
import gzip
import pickle
import scanpy as sc
from tqdm import tqdm
import seaborn as sns
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from . import map_utils as mu
import logging
import warnings
from sklearn.metrics import auc
warnings.filterwarnings("ignore")
logger_ann = logging.getLogger("anndata")
logger_ann.disabled = True

class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


class VideoStreamer:
    """ Class to help process image streams. Four types of possible inputs:"
        1.) USB Webcam.
        2.) An IP camera
        3.) A directory of images (files in directory matching 'image_glob').
        4.) A video file, such as an .mp4 or .avi file.
    """
    def __init__(self, basedir, resize, skip, image_glob, max_length=1000000):
        self._ip_grabbed = False
        self._ip_running = False
        self._ip_camera = False
        self._ip_image = None
        self._ip_index = 0
        self.cap = []
        self.camera = True
        self.video_file = False
        self.listing = []
        self.resize = resize
        self.interp = cv2.INTER_AREA
        self.i = 0
        self.skip = skip
        self.max_length = max_length
        if isinstance(basedir, int) or basedir.isdigit():
            print('==> Processing USB webcam input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(int(basedir))
            self.listing = range(0, self.max_length)
        elif basedir.startswith(('http', 'rtsp')):
            print('==> Processing IP camera input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.start_ip_camera_thread()
            self._ip_camera = True
            self.listing = range(0, self.max_length)
        elif Path(basedir).is_dir():
            print('==> Processing image directory input: {}'.format(basedir))
            self.listing = list(Path(basedir).glob(image_glob[0]))
            for j in range(1, len(image_glob)):
                image_path = list(Path(basedir).glob(image_glob[j]))
                self.listing = self.listing + image_path
            self.listing.sort()
            self.listing = self.listing[::self.skip]
            self.max_length = np.min([self.max_length, len(self.listing)])
            if self.max_length == 0:
                raise IOError('No images found (maybe bad \'image_glob\' ?)')
            self.listing = self.listing[:self.max_length]
            self.camera = False
        elif Path(basedir).exists():
            print('==> Processing video input: {}'.format(basedir))
            self.cap = cv2.VideoCapture(basedir)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.listing = range(0, num_frames)
            self.listing = self.listing[::self.skip]
            self.video_file = True
            self.max_length = np.min([self.max_length, len(self.listing)])
            self.listing = self.listing[:self.max_length]
        else:
            raise ValueError('VideoStreamer input \"{}\" not recognized.'.format(basedir))
        if self.camera and not self.cap.isOpened():
            raise IOError('Could not read camera')

    def load_image(self, impath):
        """ Read image as grayscale and resize to img_size.
        Inputs
            impath: Path to input image.
        Returns
            grayim: uint8 numpy array sized H x W.
        """
        grayim = cv2.imread(impath, 0)
        if grayim is None:
            raise Exception('Error reading image %s' % impath)
        w, h = grayim.shape[1], grayim.shape[0]
        w_new, h_new = process_resize(w, h, self.resize)
        grayim = cv2.resize(
            grayim, (w_new, h_new), interpolation=self.interp)
        return grayim

    def next_frame(self):
        """ Return the next frame, and increment internal counter.
        Returns
             image: Next H x W image.
             status: True or False depending whether image was loaded.
        """

        if self.i == self.max_length:
            return (None, False)
        if self.camera:

            if self._ip_camera:
                #Wait for first image, making sure we haven't exited
                while self._ip_grabbed is False and self._ip_exited is False:
                    time.sleep(.001)

                ret, image = self._ip_grabbed, self._ip_image.copy()
                if ret is False:
                    self._ip_running = False
            else:
                ret, image = self.cap.read()
            if ret is False:
                print('VideoStreamer: Cannot get image from camera')
                return (None, False)
            w, h = image.shape[1], image.shape[0]
            if self.video_file:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.listing[self.i])

            w_new, h_new = process_resize(w, h, self.resize)
            image = cv2.resize(image, (w_new, h_new),
                               interpolation=self.interp)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            image_file = str(self.listing[self.i])
            image = self.load_image(image_file)
        self.i = self.i + 1
        return (image, True)

    def start_ip_camera_thread(self):
        self._ip_thread = Thread(target=self.update_ip_camera, args=())
        self._ip_running = True
        self._ip_thread.start()
        self._ip_exited = False
        return self

    def update_ip_camera(self):
        while self._ip_running:
            ret, img = self.cap.read()
            if ret is False:
                self._ip_running = False
                self._ip_exited = True
                self._ip_grabbed = False
                return

            self._ip_image = img
            self._ip_grabbed = ret
            self._ip_index += 1
            #print('IPCAMERA THREAD got frame {}'.format(self._ip_index))


    def cleanup(self):
        self._ip_running = False

# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame):
    return torch.from_numpy(frame/255.).float()[None, None].cuda()


def read_image(path, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image)
    return image, inp, scales



def read_image_modified(image, resize, resize_float):
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')
    return image
# --- GEOMETRY ---


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 5:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, mask_new = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def angle_error_mat(R1, R2):
    cos = (np.trace(np.dot(R1.T, R2)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))


def compute_pose_error(T_0to1, R, t):
    R_gt = T_0to1[:3, :3]
    t_gt = T_0to1[:3, 3]
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R, R_gt)
    return error_t, error_R


def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# --- VISUALIZATION ---


def plot_image_pair(imgs, dpi=100, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None
    _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        ax[i].get_yaxis().set_ticks([])
        ax[i].get_xaxis().set_ticks([])
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.tight_layout(pad=pad)


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                       color, text, path, name0, name1, show_keypoints=False,
                       fast_viz=False, opencv_display=False, opencv_title='matches'):

    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, name0, transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    txt_color = 'k' if image1[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, name1, transform=fig.axes[1].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title=''):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    Ht = int(H * 30 / 480)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    H*1.0/480, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    H*1.0/480, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)


def read_pickle(filename):
    """
    Helper to read pickle file which may be zipped or not.
    Args:
        filename (str): A valid string path.
    Returns:
        The file object.
    """
    try:
        with gzip.open(filename, "rb") as f:
            loaded_object = pickle.load(f)
            return loaded_object
    except OSError:
        with open(filename, "rb") as f:
            loaded_object = pickle.load(f)
            return loaded_object


def annotate_gene_sparsity(adata):
    """
    Annotates gene sparsity in given Anndatas. 
    Update given Anndata by creating `var` "sparsity" field with gene_sparsity (1 - % non-zero observations).
    Args:
        adata (Anndata): single cell or spatial data.
    Returns:
        None
    """
    mask = adata.X != 0
    gene_sparsity = np.sum(mask, axis=0) / adata.n_obs
    gene_sparsity = np.asarray(gene_sparsity)
    gene_sparsity = 1 - np.reshape(gene_sparsity, (-1,))
    adata.var["sparsity"] = gene_sparsity


def get_matched_genes(prior_genes_names, sn_genes_names, excluded_genes=None):
    """
    Given the list of genes in the spatial data and the list of genes in the single nuclei, identifies the subset of
    genes included in both lists and returns the corresponding matching indices.
    Args:
        prior_genes_names (sequence): List of gene names in the spatial data.
        sn_genes_names (sequence): List of gene names in the single nuclei data.
        excluded_genes (sequence): Optional. List of genes to be excluded. These genes are excluded even if present in both datasets.
        If None, no genes are excluded. Default is None.
    Returns:
        A tuple (mask_prior_indices, mask_sn_indices, selected_genes), with:
        mask_prior_indices (list): List of indices for the selected genes in 'prior_genes_names'.
        mask_sn_indices (list): List of indices for the selected genes in 'sn_genes_names'.
        selected_genes (list): List of names of the selected genes.
        For each i, selected_genes[i] = prior_genes_names[mask_prior_indices[i]] = sn_genes_names[mask_sn_indices[i].
    """
    prior_genes_names = np.array(prior_genes_names)
    sn_genes_names = np.array(sn_genes_names)

    mask_prior_indices = []
    mask_sn_indices = []
    selected_genes = []
    if excluded_genes is None:
        excluded_genes = []
    for index, i in enumerate(sn_genes_names):
        if i in excluded_genes:
            continue
        try:
            mask_prior_indices.append(np.argwhere(prior_genes_names == i)[0][0])
            # if no exceptions above:
            mask_sn_indices.append(index)
            selected_genes.append(i)
        except IndexError:
            pass

    assert len(mask_prior_indices) == len(mask_sn_indices)
    return mask_prior_indices, mask_sn_indices, selected_genes


def one_hot_encoding(l, keep_aggregate=False):
    """
    Given a sequence, returns a DataFrame with a column for each unique value in the sequence and a one-hot-encoding.
    Args:
        l (sequence): List to be transformed.
        keep_aggregate (bool): Optional. If True, the output includes an additional column for the original list. Default is False.
    Returns:
        A DataFrame with a column for each unique value in the sequence and a one-hot-encoding, and an additional
        column with the input list if 'keep_aggregate' is True.
        The number of rows are equal to len(l).
    """
    df_enriched = pd.DataFrame({"cl": l})
    for i in l.unique():
        df_enriched[i] = list(map(int, df_enriched["cl"] == i))
    if not keep_aggregate:
        del df_enriched["cl"]
    return df_enriched


def project_cell_annotations(
    adata_map, adata_sp, annotation="cell_type", threshold=0.5
):
    """
    Transfer `annotation` from single cell data onto space. 
    Args:
        adata_map (AnnData): cell-by-spot AnnData returned by `train` function.
        adata_sp (AnnData): spatial data used to save the mapping result.
        annotation (str): Optional. Cell annotations matrix with shape (number_cells, number_annotations). Default is 'cell_type'.
        threshold (float): Optional. Valid for using with adata_map.obs['F_out'] from 'constrained' mode mapping. 
        Cell's probability below this threshold will be dropped. Default is 0.5.
    Returns:
        None.
        Update spatial Anndata by creating `obsm` `tangram_ct_pred` field with a dataframe with spatial prediction for each annotation (number_spots, number_annotations) 
    """

    df = one_hot_encoding(adata_map.obs[annotation])
    if "F_out" in adata_map.obs.keys():
        df_ct_prob = adata_map[adata_map.obs["F_out"] > threshold]

    df_ct_prob = adata_map.X.T @ df
    df_ct_prob.index = adata_map.var.index

    adata_sp.obsm["tangram_ct_pred"] = df_ct_prob
    logging.info(
        f"spatial prediction dataframe is saved in `obsm` `tangram_ct_pred` of the spatial AnnData."
    )


def create_segment_cell_df(adata_sp):
    """
    Produces a Pandas dataframe where each row is a segmentation object, columns reveals its position information.
    Args:
        adata_sp (AnnData): spot-by-gene AnnData structure. Must contain obsm.['image_features']
    Returns:
        None.
        Update spatial AnnData.uns['tangram_cell_segmentation'] with a dataframe: each row represents a segmentation object (single cell/nuclei). Columns are 'spot_idx' (voxel id), and 'y', 'x', 'centroids' to specify the position of the segmentation object.
        Update spatial AnnData.obsm['trangram_spot_centroids'] with a sequence
    """

    if "image_features" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing parameter for tangram deconvolution. Run `sqidpy.im.calculate_image_features`."
        )

    centroids = adata_sp.obsm["image_features"][["segmentation_centroid"]].copy()
    centroids["centroids_idx"] = [
        np.array([f"{k}_{j}" for j in np.arange(i)], dtype="object")
        for k, i in zip(
            adata_sp.obs.index.values,
            adata_sp.obsm["image_features"]["segmentation_label"],
        )
    ]
    centroids_idx = centroids.explode("centroids_idx")
    centroids_coords = centroids.explode("segmentation_centroid")
    segmentation_df = pd.DataFrame(
        centroids_coords["segmentation_centroid"].to_list(),
        columns=["y", "x"],
        index=centroids_coords.index,
    )
    segmentation_df["centroids"] = centroids_idx["centroids_idx"].values
    segmentation_df.index.set_names("spot_idx", inplace=True)
    segmentation_df.reset_index(
        drop=False, inplace=True,
    )

    adata_sp.uns["tangram_cell_segmentation"] = segmentation_df
    adata_sp.obsm["tangram_spot_centroids"] = centroids["centroids_idx"]
    logging.info(
        f"cell segmentation dataframe is saved in `uns` `tangram_cell_segmentation` of the spatial AnnData."
    )
    logging.info(
        f"spot centroids is saved in `obsm` `tangram_spot_centroids` of the spatial AnnData."
    )


def count_cell_annotations(
    adata_map, adata_sc, adata_sp, annotation="cell_type", threshold=0.5,
):
    """
    Count cells in a voxel for each annotation.
    
    Args:
        adata_map (AnnData): cell-by-spot AnnData returned by `train` function.
        adata_sc (AnnData): cell-by-gene AnnData.
        adata_sp (AnnData): spatial AnnData data used to save the mapping result.
        annotation (str): Optional. Cell annotations matrix with shape (number_cells, number_annotations). Default is 'cell_type'.
        threshold (float): Optional. Valid for using with adata_map.obs['F_out'] from 'constrained' mode mapping. 
                           Cell's probability below this threshold will be dropped. Default is 0.5.
    
    Returns:
        None.
        Update spatial AnnData by creating `obsm` `tangram_ct_count` field which contains a dataframe that each row is a spot and each column has the cell count for each cell annotation (number_spots, number_annotations).
    
    """

    if "spatial" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing spatial information in AnnDatas. Please make sure coordinates are saved with AnnData.obsm['spatial']"
        )

    if "image_features" not in adata_sp.obsm.keys():
        raise ValueError(
            "Missing parameter for tangram deconvolution. Run `sqidpy.im.calculate_image_features`."
        )

    if (
        "tangram_cell_segmentation" not in adata_sp.uns.keys()
        or "tangram_spot_centroids" not in adata_sp.obsm.keys()
    ):
        raise ValueError(
            "Missing parameter for tangram deconvolution. Run `create_segment_cell_df`."
        )

    xs = adata_sp.obsm["spatial"][:, 1]
    ys = adata_sp.obsm["spatial"][:, 0]
    cell_count = adata_sp.obsm["image_features"]["segmentation_label"]

    df_segmentation = adata_sp.uns["tangram_cell_segmentation"]
    centroids = adata_sp.obsm["tangram_spot_centroids"]

    # create a dataframe
    df_vox_cells = df_vox_cells = pd.DataFrame(
        data={"x": xs, "y": ys, "cell_n": cell_count, "centroids": centroids},
        index=list(adata_sp.obs.index),
    )

    # get the most probable voxel for each cell
    resulting_voxels = np.argmax(adata_map.X, axis=1)

    # create a list with filtered cells and the voxels where they have been placed with the
    # highest probability a cell i is filtered if F_i > threshold'
    if "F_out" in adata_map.obs.keys():
        filtered_voxels_to_types = [
            (j, adata_sc.obs[annotation][k])
            for i, j, k in zip(
                adata_map.obs["F_out"], resulting_voxels, range(len(adata_sc))
            )
            if i > threshold
        ]

        vox_ct = filtered_voxels_to_types
    else:
        filtered_voxels_to_types=[]
        for i in range(len(resulting_voxels)):
            filtered_voxels_to_types.append((resulting_voxels[i],adata_sc.obs[annotation][i]))
        vox_ct = filtered_voxels_to_types

    df_classes = one_hot_encoding(adata_sc.obs[annotation])

    for index, i in enumerate(df_classes.columns):
        df_vox_cells[i] = 0

    for k, v in vox_ct:
        df_vox_cells.iloc[k, df_vox_cells.columns.get_loc(v)] += 1

    adata_sp.obsm["tangram_ct_count"] = df_vox_cells
    logging.info(
        f"spatial cell count dataframe is saved in `obsm` `tangram_ct_count` of the spatial AnnData."
    )


def deconvolve_cell_annotations(adata_sp, filter_cell_annotation=None):
    """
    Assigns cell annotation to each segmented cell. Produces an AnnData structure that saves the assignment in its obs dataframe.
    Args:
        adata_sp (AnnData): Spatial AnnData structure.
        filter_cell_annotation (sequence): Optional. Sequence of cell annotation names to be considered for deconvolution. Default is None. When no values passed, all cell annotation names in adata_sp.obsm["tangram_ct_pred"] will be used.
    Returns:
        AnnData: Saves the cell annotation assignment result in its obs dataframe where each row representing a segmentation object, column 'x', 'y', 'centroids' contain its position and column 'cluster' is the assigned cell annotation.
    """

    if (
        "tangram_ct_count" not in adata_sp.obsm.keys()
        or "tangram_cell_segmentation" not in adata_sp.uns.keys()
    ):
        raise ValueError("Missing tangram parameters. Run `count_cell_annotations`.")

    segmentation_df = adata_sp.uns["tangram_cell_segmentation"]

    if filter_cell_annotation is None:
        filter_cell_annotation = pd.unique(
            list(adata_sp.obsm["tangram_ct_pred"].columns)
        )
    else:
        filter_cell_annotation = pd.unique(filter_cell_annotation)

    df_vox_cells = adata_sp.obsm["tangram_ct_count"]
    cell_types_mapped = df_to_cell_types(df_vox_cells, filter_cell_annotation)
    df_list = []
    for k in cell_types_mapped.keys():
        df = pd.DataFrame({"centroids": np.array(cell_types_mapped[k], dtype="object")})
        df["cluster"] = k
        df_list.append(df)
    cluster_df = pd.concat(df_list, axis=0)
    cluster_df.reset_index(inplace=True, drop=True)

    merged_df = segmentation_df.merge(cluster_df, on="centroids", how="inner")
    merged_df.drop(columns="spot_idx", inplace=True)
    merged_df.drop_duplicates(inplace=True)
    merged_df.dropna(inplace=True)
    merged_df.reset_index(inplace=True, drop=True)

    adata_segment = sc.AnnData(np.zeros(merged_df.shape), obs=merged_df)
    adata_segment.obsm["spatial"] = merged_df[["y", "x"]].to_numpy()
    adata_segment.uns = adata_sp.uns

    return adata_segment


def project_genes(adata_map, adata_sc, cluster_label=None, scale=True):
    """
    Transfer gene expression from the single cell onto space.
    Args:
        adata_map (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cluster_label (AnnData): Optional. Should be consistent with the 'cluster_label' argument passed to `map_cells_to_space` function.
        scale (bool): Optional. Should be consistent with the 'scale' argument passed to `map_cells_to_space` function.
    Returns:
        AnnData: spot-by-gene AnnData containing spatial gene expression from the single cell data.
    """

    # put all var index to lower case to align
    adata_sc.var.index = [g.lower() for g in adata_sc.var.index]

    # make varnames unique for adata_sc
    adata_sc.var_names_make_unique()

    # remove all-zero-valued genes
    sc.pp.filter_genes(adata_sc, min_cells=1)

    if cluster_label:
        adata_sc = mu.adata_to_cluster_expression(adata_sc, cluster_label, scale=scale)

    if not adata_map.obs.index.equals(adata_sc.obs.index):
        raise ValueError("The two AnnDatas need to have same `obs` index.")
    if hasattr(adata_sc.X, "toarray"):
        adata_sc.X = adata_sc.X.toarray()
    X_space = adata_map.X.T @ adata_sc.X
    adata_ge = sc.AnnData(
        X=X_space, obs=adata_map.var, var=adata_sc.var, uns=adata_sc.uns
    )
    training_genes = adata_map.uns["train_genes_df"].index.values
    adata_ge.var["is_training"] = adata_ge.var.index.isin(training_genes)
    return adata_ge


def compare_spatial_geneexp(adata_ge, adata_sp, adata_sc=None, genes=None):
    """ Compares generated spatial data with the true spatial data
    Args:
        adata_ge (AnnData): generated spatial data returned by `project_genes`
        adata_sp (AnnData): gene spatial data
        adata_sc (AnnData): Optional. When passed, sparsity difference between adata_sc and adata_sp will be calculated. Default is None.
        genes (list): Optional. When passed, returned output will be subset on the list of genes. Default is None.
    Returns:
        Pandas Dataframe: a dataframe with columns: 'score', 'is_training', 'sparsity_sp'(spatial data sparsity). 
                          Columns - 'sparsity_sc'(single cell data sparsity), 'sparsity_diff'(spatial sparsity - single cell sparsity) returned only when adata_sc is passed.
    """

    logger_root = logging.getLogger()
    logger_root.disabled = True

    # Check if training_genes/overlap_genes key exist/is valid in adatas.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sp.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_ge.uns.keys())):
        raise ValueError(
            "Missing tangram parameters. Use `project_genes()` to get adata_ge."
        )

    assert list(adata_sp.uns["overlap_genes"]) == list(adata_ge.uns["overlap_genes"])

    if genes is None:
        overlap_genes = adata_ge.uns["overlap_genes"]
    else:
        overlap_genes = genes

    annotate_gene_sparsity(adata_sp)

    # Annotate cosine similarity of each training gene
    cos_sims = []

    if hasattr(adata_ge.X, "toarray"):
        X_1 = adata_ge[:, overlap_genes].X.toarray()
    else:
        X_1 = adata_ge[:, overlap_genes].X
    if hasattr(adata_sp.X, "toarray"):
        X_2 = adata_sp[:, overlap_genes].X.toarray()
    else:
        X_2 = adata_sp[:, overlap_genes].X

    for v1, v2 in zip(X_1.T, X_2.T):
        norm_sq = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_sims.append((v1 @ v2) / norm_sq)

    df_g = pd.DataFrame(cos_sims, overlap_genes, columns=["score"])
    for adata in [adata_ge, adata_sp]:
        if "is_training" in adata.var.keys():
            df_g["is_training"] = adata.var.is_training

    df_g["sparsity_sp"] = adata_sp[:, overlap_genes].var.sparsity

    if adata_sc is not None:
        if not set(["training_genes", "overlap_genes"]).issubset(
            set(adata_sc.uns.keys())
        ):
            raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

        assert list(adata_sc.uns["overlap_genes"]) == list(
            adata_sp.uns["overlap_genes"]
        )
        annotate_gene_sparsity(adata_sc)

        df_g = df_g.merge(
            pd.DataFrame(adata_sc[:, overlap_genes].var["sparsity"]),
            left_index=True,
            right_index=True,
        )
        df_g.rename({"sparsity": "sparsity_sc"}, inplace=True, axis="columns")
        df_g["sparsity_diff"] = df_g["sparsity_sp"] - df_g["sparsity_sc"]

    else:
        logging.info(
            "To create dataframe with column 'sparsity_sc' or 'aprsity_diff', please also pass adata_sc to the function."
        )

    if genes is not None:
        df_g = df_g.loc[genes]

    df_g = df_g.sort_values(by="score", ascending=False)
    return df_g


def cv_data_gen(adata_sc, adata_sp, cv_mode="loo"):
    """ Generates pair of training/test gene indexes cross validation datasets
    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        mode (str): Optional. support 'loo' and '10fold'. Default is 'loo'.
    Yields:
        tuple: list of train_genes, list of test_genes
    """

    # Check if training_genes key exist/is valid in adatas.uns
    if "training_genes" not in adata_sc.uns.keys():
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if "training_genes" not in adata_sp.uns.keys():
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    if not list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"]):
        raise ValueError(
            "Unmatched training_genes field in two Anndatas. Run `pp_adatas()`."
        )

    genes_array = np.array(adata_sp.uns["training_genes"])

    if cv_mode == "loo":
        cv = LeaveOneOut()
    elif cv_mode == "10fold":
        cv = KFold(n_splits=10)

    for train_idx, test_idx in cv.split(genes_array):
        train_genes = list(genes_array[train_idx])
        test_genes = list(genes_array[test_idx])
        yield train_genes, test_genes


def cross_val(
    adata_sc,
    adata_sp,
    cluster_label=None,
    mode="clusters",
    scale=True,
    lambda_d=0,
    lambda_g1=1,
    lambda_g2=0,
    lambda_r=0,
    lambda_count=1,
    lambda_f_reg=1,
    target_count=None,
    num_epochs=1000,
    device="cuda:0",
    learning_rate=0.1,
    cv_mode="loo",
    return_gene_pred=False,
    density_prior=None,
    random_state=None,
    verbose=False,
):
    """
    Executes cross validation
    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cluster_label (str): the level that the single cell data will be aggregate at, this is only valid for clusters mode mapping
        mode (str): Optional. Tangram mapping mode. Currently supported: 'cell', 'clusters', 'constrained'. Default is 'clusters'.
        scale (bool): Optional. Whether weight input single cell by # of cells in cluster, only valid when cluster_label is not None. Default is True.
        lambda_g1 (float): Optional. Strength of Tangram loss function. Default is 1.
        lambda_d (float): Optional. Strength of density regularizer. Default is 0.
        lambda_g2 (float): Optional. Strength of voxel-gene regularizer. Default is 0.
        lambda_r (float): Optional. Strength of entropy regularizer. Default is 0.
        lambda_count (float): Optional. Regularizer for the count term. Default is 1. Only valid when mode == 'constrained'
        lambda_f_reg (float): Optional. Regularizer for the filter, which promotes Boolean values (0s and 1s) in the filter. Only valid when mode == 'constrained'. Default is 1.
        target_count (int): Optional. The number of cells to be filtered. Default is None.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        device (str or torch.device): Optional. Default is 'cuda:0'.
        cv_mode (str): Optional. cross validation mode, 'loo' ('leave-one-out') and '10fold' supported. Default is 'loo'.
        return_gene_pred (bool): Optional. if return prediction and true spatial expression data for test gene, only applicable when 'loo' mode is on, default is False.
        density_prior (ndarray or str): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If not provided, the density term is ignored. 
        random_state (int): Optional. pass an int to reproduce training. Default is None.
        verbose (bool): Optional. If print training details. Default is False.
    
    Returns:
        cv_dict (dict): a dictionary contains information of cross validation (hyperparameters, average test score and train score, etc.)
        adata_ge_cv (AnnData): predicted spatial data by LOOCV. Only returns when `return_gene_pred` is True and in 'loo' mode.
        test_gene_df (Pandas dataframe): dataframe with columns: 'score', 'is_training', 'sparsity_sp'(spatial data sparsity)
    """

    logger_root = logging.getLogger()
    logger_root.disabled = True
    logger_ann = logging.getLogger("anndata")
    logger_ann.disabled = True

    test_genes_list = []
    test_pred_list = []
    test_score_list = []
    train_score_list = []
    test_df_list = []
    curr_cv_set = 1

    if cv_mode == "loo":
        length = len(list(adata_sc.uns["training_genes"]))
    elif cv_mode == "10fold":
        length = 10

    if mode == "clusters":
        adata_sc_agg = mu.adata_to_cluster_expression(adata_sc, cluster_label, scale)

    for train_genes, test_genes in tqdm(
        cv_data_gen(adata_sc, adata_sp, cv_mode), total=length
    ):
        # train
        adata_map = mu.map_cells_to_space(
            adata_sc=adata_sc,
            adata_sp=adata_sp,
            cv_train_genes=train_genes,
            mode=mode,
            device=device,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            cluster_label=cluster_label,
            scale=scale,
            lambda_d=lambda_d,
            lambda_g1=lambda_g1,
            lambda_g2=lambda_g2,
            lambda_r=lambda_r,
            lambda_count=lambda_count,
            lambda_f_reg=lambda_f_reg,
            target_count=target_count,
            random_state=random_state,
            verbose=False,
            density_prior=density_prior,
        )

        cv_genes = train_genes + test_genes

        # project on space
        adata_ge = project_genes(
            adata_map, adata_sc[:, cv_genes], cluster_label=cluster_label, scale=scale,
        )

        # retrieve result for test gene (genes X cluster/cell)
        if cv_mode == "loo" and return_gene_pred:
            adata_ge_test = adata_ge[:, test_genes].X.T
            test_pred_list.append(adata_ge_test)

        # output test genes dataframe
        if mode == "clusters":
            df_g = compare_spatial_geneexp(adata_ge, adata_sp, adata_sc_agg, cv_genes)
        else:
            df_g = compare_spatial_geneexp(adata_ge, adata_sp, adata_sc, cv_genes)

        test_df = df_g[df_g.index.isin(test_genes)]
        test_score = df_g.loc[test_genes]["score"].mean()
        train_score = np.float(list(adata_map.uns["training_history"]["main_loss"])[-1])

        # output avg score
        test_genes_list.append(test_genes)
        test_score_list.append(test_score)
        train_score_list.append(train_score)
        test_df_list.append(test_df)

        if verbose == True:
            msg = "cv set: {}----train score: {:.3f}----test score: {:.3f}".format(
                curr_cv_set, train_score, test_score
            )
            print(msg)

        curr_cv_set += 1

    # use nanmean to ignore nan in score list
    avg_test_score = np.nanmean(test_score_list)
    avg_train_score = np.nanmean(train_score_list)

    cv_dict = {
        "avg_test_score": avg_test_score,
        "avg_train_score": avg_train_score,
    }

    print("cv avg test score {:.3f}".format(avg_test_score))
    print("cv avg train score {:.3f}".format(avg_train_score))

    if cv_mode == "loo" and return_gene_pred:

        # output df_test_genes dataframe
        test_gene_df = pd.concat(test_df_list, axis=0)

        # output AnnData for generated spatial data by LOOCV
        adata_ge_cv = sc.AnnData(
            X=np.squeeze(test_pred_list).T,
            obs=adata_sp.obs.copy(),
            var=pd.DataFrame(
                test_score_list,
                columns=["test_score"],
                index=np.squeeze(test_genes_list),
            ),
        )

        return cv_dict, adata_ge_cv, test_gene_df

    return cv_dict


def eval_metric(df_all_genes, test_genes=None):
    """
    Compute metrics on given test_genes set for evaluation
    
    Args:
        df_all_genes (Pandas dataframe): returned by compare_spatial_geneexp(adata_ge, adata_sp); 
        test_genes (list): list of test genes, if not given, test_genes will be set to genes where 'is_training' field is False
    Returns:      
        dict with values of each evaluation metric ("avg_test_score", "avg_train_score", "auc_score"), 
        tuple of auc fitted coordinates and raw coordinates(test_score vs. sparsity_sp coordinates)
    """

    # validate test_genes:
    if test_genes is not None:
        if not set(test_genes).issubset(set(df_all_genes.index.values)):
            raise ValueError(
                "the input of test_genes should be subset of genes of input dataframe"
            )
        test_genes = np.unique(test_genes)

    else:
        test_genes = list(
            set(df_all_genes[df_all_genes["is_training"] == False].index.values)
        )

    # calculate:
    test_gene_scores = df_all_genes.loc[test_genes]["score"]
    test_gene_sparsity_sp = df_all_genes.loc[test_genes]["sparsity_sp"]
    test_score_avg = test_gene_scores.mean()
    train_score_avg = df_all_genes[df_all_genes["is_training"] == True]["score"].mean()

    # sp sparsity weighted score
    test_score_sps_sp_g2 = np.sum(
        (test_gene_scores * (1 - test_gene_sparsity_sp))
        / (1 - test_gene_sparsity_sp).sum()
    )

    # tm metric
    # Fit polynomial'
    xs = list(test_gene_scores)
    ys = list(test_gene_sparsity_sp)
    pol_deg = 2
    pol_cs = np.polyfit(xs, ys, pol_deg)  # polynomial coefficients
    pol_xs = np.linspace(0, 1, 10)  # x linearly spaced
    pol = np.poly1d(pol_cs)  # build polynomial as function
    pol_ys = [pol(x) for x in pol_xs]  # compute polys
    
    if pol_ys[0] > 1:
        pol_ys[0] = 1

    # if real root when y = 0, add point (x, 0):
    roots = pol.r
    root = None
    for i in range(len(roots)):
        if np.isreal(roots[i]) and roots[i] <= 1 and roots[i] >= 0:
            root = roots[i]
            break

    if root is not None:
        pol_xs = np.append(pol_xs, root)
        pol_ys = np.append(pol_ys, 0)       
        
    np.append(pol_xs, 1)
    np.append(pol_ys, pol(1))

    # remove point that are out of [0,1]
    del_idx = []
    for i in range(len(pol_xs)):
        if pol_xs[i] < 0 or pol_ys[i] < 0 or pol_xs[i] > 1 or pol_ys[i] > 1:
            del_idx.append(i)

    pol_xs = [x for x in pol_xs if list(pol_xs).index(x) not in del_idx]
    pol_ys = [y for y in pol_ys if list(pol_ys).index(y) not in del_idx]

    # Compute are under the curve of polynomial
    auc_test_score = np.real(auc(pol_xs, pol_ys))

    metric_dict = {
        "avg_test_score": test_score_avg,
        "avg_train_score": train_score_avg,
        "sp_sparsity_score": test_score_sps_sp_g2,
        "auc_score": auc_test_score,
    }

    auc_coordinates = ((pol_xs, pol_ys), (xs, ys))

    return metric_dict, auc_coordinates


# DEPRECATED
def transfer_annotations_prob(mapping_matrix, to_transfer):
    """
    Transfer cell annotations onto space through a mapping matrix.
    Args:
        mapping_matrix (ndarray): Mapping matrix with shape (number_cells, number_spots).
        to_transfer (ndarray): Cell annotations matrix with shape (number_cells, number_annotations).
        
    Returns:
        A matrix of annotations onto space, with shape (number_spots, number_annotations)
    """
    return mapping_matrix.transpose() @ to_transfer


def transfer_annotations_prob_filter(mapping_matrix, filter, to_transfer):
    """
    Transfer cell annotations onto space through a mapping matrix and a filter.
    Args:
        mapping_matrix (ndarray): Mapping matrix with shape (number_cells, number_spots).
        filter (ndarray): Filter with shape (number_cells,).
        to_transfer (ndarray): Cell annotations matrix with shape (number_cells, number_annotations).
    Returns:
        A matrix of annotations onto space, with shape (number_spots, number_annotations).
    """
    tt = to_transfer * filter[:, np.newaxis]
    return mapping_matrix.transpose() @ tt


def df_to_cell_types(df, cell_types):
    """
    Utility function that "randomly" assigns cell coordinates in a voxel to known numbers of cell types in that voxel.
    Used for deconvolution.
    Args:
        df (DataFrame): Columns correspond to cell types.  Each row in the DataFrame corresponds to a voxel and
        specifies the known number of cells in that voxel for each cell type (int).
        The additional column 'centroids' specifies the coordinates of the cells in the voxel (sequence of (x,y) pairs).
        cell_types (sequence): Sequence of cell type names to be considered for deconvolution.
        Columns in 'df' not included in 'cell_types' are ignored for assignment.
    Returns:
        A dictionary <cell type name> -> <list of (x,y) coordinates for the cell type>
    """
    df_cum_sums = df[cell_types].cumsum(axis=1)

    df_c = df.copy()

    for i in df_cum_sums.columns:
        df_c[i] = df_cum_sums[i]

    cell_types_mapped = defaultdict(list)
    for i_index, i in enumerate(cell_types):
        for j_index, j in df_c.iterrows():
            start_ind = 0 if i_index == 0 else j[cell_types[i_index - 1]]
            end_ind = j[i]
            cell_types_mapped[i].extend(j["centroids"][start_ind:end_ind].tolist())
    return cell_types_mapped

def feature_pca(input_file = './ad_st_new.csv', lrpairs = './lrpairs.csv', 
    cell_pair_all = './cell_pair_all.csv'):
    df = pd.read_csv(input_file,index_col=0)

    lrpairs = pd.read_csv(lrpairs)
    #select L-R pairs

    lrpairs_ligand = lrpairs["ligand"]
    lrpairs_receptor = lrpairs["receptor"]
    #len(lrpairs_ligand)
    for ii in range(1):
        #print(lrpairs_ligand[ii])

        Cellpairall = pd.read_csv(cell_pair_all)
        ligand = Cellpairall["ligand"]
        receptor = Cellpairall["receptor"]
        Sender = Cellpairall["cell_sender"]
        Receiver = Cellpairall["cell_receiver"]
        CCIlabel = Cellpairall["label"]
        cell_pair_label = []
        
        for i in range(len(Sender)):
            if ligand[i]==lrpairs_ligand[ii] and receptor[i]==lrpairs_receptor[ii]:
                if df.loc[lrpairs_ligand[ii],str(Sender[i])]>0.05 and df.loc[lrpairs_receptor[ii],str(Receiver[i])]>0.05:
                    score = np.sqrt(df.loc[lrpairs_ligand[ii],str(Sender[i])]*df.loc[lrpairs_receptor[ii],str(Receiver[i])])
                    #print(score)
                    cell_pair_label.append([Sender[i],Receiver[i],str(score),str(CCIlabel[i])])
                else:
                    score = np.sqrt(df.loc[lrpairs_ligand[ii],str(Sender[i])]*df.loc[lrpairs_receptor[ii],str(Receiver[i])])
                    cell_pair_label.append([Sender[i],Receiver[i],str(score),'0'])
        cell_pair = []
        for i in range(len(Sender)):
            if df.loc[lrpairs_ligand[ii],str(Sender[i])]>0 and df.loc[lrpairs_receptor[ii],str(Receiver[i])]>0:
                cell_pair.append([Sender[i],Receiver[i]])
        cell_pair = np.array(cell_pair)
        new_cell_pair = np.unique(cell_pair, axis=0).tolist()
        #print(np.sum(np.array(cell_pair_label)[:,-1].astype(np.int32)))
        #print(len(new_cell_pair))
        #print(np.sum(np.array(cell_pair_label)[:,-1].astype(np.int32))/len(new_cell_pair))
        
        if (len(cell_pair_label))>0:
            #print(lrpairs_ligand[ii]+'_'+lrpairs_receptor[ii])
            lrdir = "./data/"+lrpairs_ligand[ii]+'_'+lrpairs_receptor[ii]
            isExists = os.path.exists(lrdir)
            if isExists:
                shutil.rmtree(lrdir)
                os.makedirs(lrdir)
            if not isExists:
                os.makedirs(lrdir)
            cell_pair_label = np.array(cell_pair_label)
            cell_pair_index = cell_pair_label[:,0:2].tolist()
            cell_pair_new = []
            for i in range(len(new_cell_pair)):
                if new_cell_pair[i] in cell_pair_index:
                    cell_pair_new.append(['to'] + new_cell_pair[i] + [cell_pair_label[cell_pair_index.index(new_cell_pair[i]),-2]]
                                                                      + [cell_pair_label[cell_pair_index.index(new_cell_pair[i]),-1]])
                else:
                    cell_pair_new.append(['to'] + new_cell_pair[i] + [0] + [0])
            CCIlist = np.array(cell_pair_new)
            CCIlist[:, 1] = np.char.replace(CCIlist[:, 1], 'C', '') 
            CCIlist[:, 2] = np.char.replace(CCIlist[:, 2], 'C', '')
            CCIlist[:, 1] = CCIlist[:, 1].astype(np.int)-1
            CCIlist[:, 2] = CCIlist[:, 2].astype(np.int)-1
            
            CCIlist_label = CCIlist[:,-1]
            CCIlist_source = CCIlist[:,1]
            CCIlist_target = CCIlist[:,2]
            #print(CCIlist_source)
            node_list_all = []
            node_list = []
            for i in range(len(CCIlist_label)):
                node_list_all.append(CCIlist_source[i])
                node_list_all.append(CCIlist_target[i])    
                if CCIlist_label[i] == '1':
                    node_list.append(CCIlist_source[i])
                    node_list.append(CCIlist_target[i])
            #print(len(set(node_list_all)))
            #print(len(set(node_list)))
            node_no = set(node_list_all) - set(node_list)
            #print(node_no)
            no_index = []
            for i in range(len(CCIlist_label)):
                if CCIlist_source[i] in node_no or CCIlist_target[i] in node_no:
                    if CCIlist_label[i] == '0':
                        no_index.append(i)
            #print(no_index)
            CCIlist_score= np.delete(CCIlist, no_index, axis=0)
            #print(CCIlist_score.shape)
            
            np.savetxt(lrdir+'/CCIlistscore.txt', CCIlist_score, fmt='%s', delimiter= ' ')
            columns = [0, 1, 2, 4]
            CCIlist_new = CCIlist_score[:,columns]
            np.savetxt(lrdir+'/CCIlist.txt', CCIlist_new, fmt='%s', delimiter= ' ')

            
            random.seed(2022)
            np.random.shuffle(CCIlist_new)

            data_size = len(CCIlist_new)

            data_train = []
            data_validate = []
            data_test = []
            for ii in range(data_size):
                if ii < round(data_size * 0.7):
                    data_train.append(CCIlist_new[ii])
                elif ii in range(round(data_size * 0.7), round(data_size * 0.8)):
                    data_validate.append(CCIlist_new[ii])
                else:
                    data_test.append(CCIlist_new[ii])
            data_train = np.array(data_train)
            #print(data_train.shape)
            np.savetxt(lrdir +'/train.txt', np.array(data_train), fmt='%s', delimiter= ' ')
            #CCIlist = np.array(CCIlist)
            postive = []
            negtive = []
            for i in range(data_train.shape[0]):
                if data_train[i,-1]=="1":
                    postive.append(data_train[i,:])
                elif data_train[i,-1]=="0":
                    negtive.append(data_train[i,:])

            #np.savetxt(lrdir +'/train.txt', np.array(postive)[:,0:-1], fmt='%s', delimiter= ' ')
            #np.savetxt(lrdir +'/negtive.txt', np.array(negtive)[:,0:-1], fmt='%s', delimiter= ' ')
           
            np.savetxt(lrdir +'/valid.txt', data_validate, fmt='%s', delimiter= ' ')
            np.savetxt(lrdir +'/test.txt', data_test, fmt='%s', delimiter= ' ')
            #print(np.sum(np.array(data_validate)[:,-1].astype(np.int32))/len(data_validate))
            #print(np.sum(np.array(data_test)[:,-1].astype(np.int32))/len(data_test))
            ent_set, rel_set = OrderedSet(), OrderedSet()

            for line in open(lrdir +'/CCIlist.txt'):
                rel, sub, obj,label = map(str.lower, line.strip().split(' '))
                if label=='1':
                    ent_set.add(sub)
                    rel_set.add(rel)
                    ent_set.add(obj)

            ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
            rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
            rel2id.update({rel+'_reverse': idx+len(rel2id) for idx, rel in enumerate(rel_set)})

            with open(lrdir +'/rel2id.txt', 'w') as f:
                json.dump(rel2id, f)
            with open(lrdir +'/ent2id.txt', 'w') as f:
                json.dump(ent2id, f)

            Node = np.loadtxt(lrdir +'/train.txt',skiprows=0,dtype=str,delimiter=' ')

            Node1 = Node[:,1]
            Node2 = Node[:,2]
            Node_label = Node[:,-1]
            #print(np.sum(np.array(Node_label).astype(np.int32)))

            edgelist = []
            for i in range(len(Node1)):
                #if Node_label[i] == "1":
                    edgelist.append([ent2id[Node1[i]],ent2id[Node2[i]]])
                    #edgelist[i,0] = (ent2id[Node1[i]])
                    #edgelist[i,1] = (ent2id[Node2[i]])
            np.savetxt(lrdir +'/edgelist.txt', edgelist, fmt='%d', delimiter=' ')
            inputfile_path = os.path.join(lrdir,'edgelist.txt')
            outputfile_path = os.path.join(lrdir,'VM.emd')
            os.chdir(lrdir)
            #os.system("/python /mnt/test/a3/DeepTalk/Example/node2vec/src/main.py --dimensions 128 \
            #          --input ./edgelist.txt --output ./VM.emd")
            os.chdir('../')
            os.chdir('../')
            data_signaling = pd.read_csv('./ad_st_new.csv',index_col=0,low_memory=False).values
            data_T = data_signaling.T
            Node0 = np.arange(data_T.shape[0])
            data_index = []
            for i in range(len(Node0)):
                if str(Node0[i,]) in ent2id.keys():          
                    data_index.append(ent2id[str(Node0[i,])])

            data_emd = data_T[data_index,:]
            data_emd = data_emd.astype(np.float64)

            from sklearn.decomposition import PCA
            pca = PCA(n_components=128)
            newX = pca.fit_transform(data_emd)
            os.chdir(lrdir)
            pd.DataFrame(newX, index=data_index).to_csv("./data_pca.emd",sep=' ',header=None)
            print(newX.shape)
            os.chdir('../')
            os.chdir('../')

def plot_cell_type_ST(ad_st,mapping_colors,data_name='./st_meta.csv'):

    df = pd.read_csv(data_name,sep = ',',index_col=0)
    max_columns = df['celltype'].values
    #print(max_columns)
    ad_st.obs['cell_type'] = max_columns
    color_list = []
    for i in range(len(list(max_columns))):
        color_list.append(mapping_colors[list(max_columns)[i]])
    x = ad_st.obs.x.values
    y = ad_st.obs.y.values
    plt.axis('off')
    for x_val, y_val, color_val,g in zip(x, y, color_list,list(max_columns)):
        plt.scatter(x_val, y_val, color=color_val,s=2,label=g)
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    plt.legend(unique_handles,unique_labels,loc=(1,0.2),fontsize = 10,ncol =2,frameon=False,handlelength=2, handletextpad=0.5)
    plt.gca().invert_yaxis()


def plot_CCC_ST(ad_st,mapping_colors,meat_data = './st_meta.csv',data_name='./ad_st_new.csv',ligand = 'Sst',
    receptor = 'Sstr2',CCC_label = './data/Sst_Sstr2/CCIlist.txt',
    sourcetype = 'L2.3.IT',targettype = 'L4'):

    df = pd.read_csv(data_name,index_col=0)
    Sst_value = df.loc[ligand]
    Sstr2_value = df.loc[receptor]
    Nodeall = np.loadtxt(CCC_label,skiprows=0,dtype=str,delimiter=' ')
    nodelabel = Nodeall[:,-1]
    nodes = Nodeall[:,1]
    nodet = Nodeall[:,2]

    df_meta = pd.read_csv(meat_data,sep = ',',index_col=0)
    max_columns = df_meta['celltype'].values
    
    cell_type = list(max_columns)
    #sourcetype = 
    #targettype = 
    cor_x = list(pd.DataFrame(ad_st.obs).iloc[:, 0])
    cor_y = list(pd.DataFrame(ad_st.obs).iloc[:, 1])

    score_list = []
    index = []
    for i in range(len(nodelabel)):
        if nodelabel[i] == '1':
            if cell_type[int(nodes[i])] == sourcetype and cell_type[int(nodet[i])] == targettype:
                score_list.append(np.sqrt(Sst_value[int(nodes[i])]*Sstr2_value[int(nodet[i])]))
    sorted_list = sorted(enumerate(score_list), key=lambda x: x[1],reverse=True)[0:50]
    sorted_indices = [index for index, _ in sorted_list]
    sorted_values = [value for _, value in sorted_list]
    Ther = sorted_values[-1]
    x_y_1 = []
    x_y_2 = []

    for i in range(len(nodelabel)):
        if nodelabel[i] == '1':
            if cell_type[int(nodes[i])] == sourcetype and cell_type[int(nodet[i])] == targettype:
                score = np.sqrt(Sst_value[int(nodes[i])] * Sstr2_value[int(nodet[i])])
                if score > Ther:
                    x_y_1.append(((cor_x[int(nodes[i])]),((cor_y[int(nodes[i])]))))
                    x_y_2.append(((cor_x[int(nodet[i])]),((cor_y[int(nodet[i])]))))
    color_list = []
    for i in range(len(list(max_columns))):
        color_list.append(mapping_colors[list(max_columns)[i]])
            
    x = ad_st.obs.x.values
    y = ad_st.obs.y.values
    plt.axis('off')
    for x_val, y_val, color_val,g in zip(x, y, color_list,list(max_columns)):
        if g == sourcetype or g == targettype:
            plt.scatter(x_val, y_val, color=color_val,s=5,label=g)
        else:
            plt.scatter(x_val, y_val, color='gray',s=5)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = list(set(labels))
    unique_handles = [handles[labels.index(label)] for label in unique_labels]
    plt.legend(unique_handles,unique_labels,loc=(1,0.5),fontsize = 10,ncol = 1,frameon=False,handlelength=2, handletextpad=0.5,title='Cell Type')

    #plt.legend(label,loc=(1,0.5),fontsize = 10,ncol = 1,frameon=False,handlelength=2, handletextpad=0.5,title='Cell Type')


    for i in range(len(x_y_1)):
        plt.annotate("",
                     xy=x_y_2[i], xycoords='data',
                     xytext=x_y_1[i], textcoords='data',
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3", linewidth=1))
    plt.title('L-R', fontsize=18)
    plt.gca().invert_yaxis()


def plot_CCC_heatmap(meat_data = './st_meta.csv',data_name='./ad_st_new.csv',
    ligand = 'Sst', receptor = 'Sstr2',
    CCC_label = './data/Sst_Sstr2/CCIlist.txt'):

    df = pd.read_csv(meat_data,sep = ',',index_col=0)
    max_columns = df['celltype'].values
    #os.chdir("/mnt/test/a3/DeepSpa/dataset/VISP_MER")
    df = pd.read_csv(data_name,index_col=0)
    Sst_value = df.loc[ligand]
    Sstr2_value = df.loc[receptor]
    Nodeall = np.loadtxt(CCC_label,skiprows=0,dtype=str,delimiter=' ')
    nodelabel = Nodeall[:,-1]
    nodes = Nodeall[:,1]
    nodet = Nodeall[:,2]
    #nodescore = Nodeall[:,3]
    cell_type = list(max_columns)

    unique_cell_type = list(set(cell_type))
    #print(unique_cell_type)
    #unique_cell_type.remove('Oligo')
    #unique_cell_type=['L2.3.IT','L4', 'L5.NP', 'CR','L5.IT', 'L5.PT', 'L6.IT','L6b' , 'L6.PT']
    data =  np.zeros((len(unique_cell_type),len(unique_cell_type)))
    #print(data.shape)

    for i in range(data.shape[0]):
        print("cells:",unique_cell_type[i])
        for j in range(data.shape[0]):
            #print("cellt:",unique_cell_type[j])
            sourcetype = unique_cell_type[i]
            targettype = unique_cell_type[j]
            score = []
            for m in range(len(nodelabel)):
                if nodelabel[m] == '1':
                    if cell_type[int(nodes[m])] == sourcetype and cell_type[int(nodet[m])] == targettype:
                        score1 = np.sqrt(Sst_value[int(nodes[m])] * Sstr2_value[int(nodet[m])])
                        score.append(score1)
                    else:
                        score.append(0)
                else:
                    score.append(0)
            #print(score)
            score_mean =  np.mean(np.array(score).astype(np.float64))
            data[i][j] =  score_mean
    '''
    data_filled = np.nan_to_num(data, nan=0)
    rows = np.any(data_filled != 0, axis=1)
    cols = np.any(data_filled != 0, axis=0)
    result = data_filled[rows][:, cols]
    '''
    result = data
    # 创建一个示例数据
    #data = np.random.rand(5, 5)

    # 创建样本名称列表
    sample_names = unique_cell_type

    # 绘制热图
    plt.imshow(result*100,cmap = 'Spectral_r')

    num_rows, num_cols = result.shape
    for i in range(num_rows):
        for j in range(num_cols):
            plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='white', linewidth=0.5, fill=False))



    # 设置x轴和y轴刻度
    plt.xticks(np.arange(len(sample_names)), sample_names, rotation=90)
    plt.yticks(np.arange(len(sample_names)), sample_names)

    # 添加颜色栏
    plt.colorbar()
    #plt.savefig('./fig/Heatmap_Sst_Sstr2.pdf',bbox_inches='tight',dpi=300)
    # 显示图形
    plt.show()

def plot_LR_heatmap(cell_pair_all = './cell_pair_all.csv',
    meat_data = './st_meta.csv',data_name='./ad_st_new.csv',
    sourcetype = 'Pvalb'):
    
    rcParams['font.family'] = 'Arial'
    Cellpairall = pd.read_csv(cell_pair_all)
    tool_res_dic = {}
    df_new = Cellpairall[["celltype_sender","celltype_receiver","ligand","receptor","label"]]
    df_new = df_new.drop_duplicates()
    #print(df_new)
    ligand = df_new["ligand"].values.tolist()
    receptor = df_new["receptor"].values.tolist()
    Sender = df_new["celltype_sender"].values.tolist()
    Receiver = df_new["celltype_receiver"].values.tolist()
    CCIlabel = df_new["label"].values.tolist()


    for i in range(len(CCIlabel)):

        if str(CCIlabel[i]) == '1':
            CC = Sender[i] + '|' + Receiver[i]
            LR = []
            for j in range(len(CCIlabel)):
                if Sender[j]==Sender[i] and Receiver[j]== Receiver[i]:
                    LR.append(ligand[j] + ' - ' + receptor[j])
            LR = list(set(LR))
            tool_res_dic.update({
                CC: LR
            })


    df = pd.read_csv(meat_data,sep = ',',index_col=0)
    max_columns = df['celltype'].values
    cell_type = list(max_columns)
    dic = {}
    unique_cell_type = list(set(cell_type))
    #unique_cell_type=['Vip','Lamp5','Astro','Sncg','Macrophage','NP','Sst','L4','L5_IT', 'L6_IT', 'L6_CT']
    #unique_cell_type=['L4', 'L5_PT','L5_IT', 'L6_IT','L6b', 'L6_CT']

    for j in range(len(unique_cell_type)):
        sourcetype = sourcetype
        targettype = unique_cell_type[j]

        cellpair = sourcetype + '|' + str(unique_cell_type[j])
        print(cellpair)
        list1 = []
        for i in range(len(tool_res_dic[cellpair])):
            df = pd.read_csv(data_name,index_col=0)
            L = tool_res_dic[cellpair][i].split(' - ')[0]
            R = tool_res_dic[cellpair][i].split(' - ')[1]
            Sst_value = df.loc[L]
            Sstr2_value = df.loc[R]
            LR = str(L) + '_' + str(R)
            print(LR)
            CCI_file = './data/'+ LR +'/CCIlist.txt'
            if os.path.exists(CCI_file):
                Nodeall = np.loadtxt('./data/'+ LR +'/CCIlist.txt',skiprows=0,dtype=str,delimiter=' ')
                nodelabel = Nodeall[:,-1]
                nodes = Nodeall[:,1]
                nodet = Nodeall[:,2]
                score = []
                for m in range(len(nodelabel)):
                    if nodelabel[m] == '1':
                        #print(cell_type[int(nodes[m])])
                        #print(cell_type[int(nodet[m])])
                        if cell_type[int(nodes[m])] == sourcetype and cell_type[int(nodet[m])] == targettype:
                            
                            score1 = np.sqrt(Sst_value[int(nodes[m])] * Sstr2_value[int(nodet[m])])
                            
                            score.append(score1)
                        else:
                            score.append(0)
                    else:
                        score.append(0)
                #print(score)
                score_mean =  np.mean(np.array(score).astype(np.float64))
            print(score_mean)
            list1.append(LR + ':' +str(score_mean))
        dic.update({
                    cellpair: list1
                })
    max_values = sorted([float(v.split(':')[1]) for sublist in dic.values() for v in sublist], reverse=True)[:50]

    # 生成新的字典，只保留最大5个数的值
    new_data = {k: [v for v in sublist if float(v.split(':')[1]) in max_values] for k, sublist in dic.items()}


    df = pd.DataFrame([(key, value) for key, values in new_data.items() for value in values], columns=['key', 'value'])
    df[['column', 'data']] = df['value'].str.split(':', 1, expand=True)
    df = df.pivot(index='key', columns='column', values='data')
    df.columns.name = None

    df = df.fillna(0)


    data =  np.array(df.values).astype(float)

    # 绘制热图
    plt.imshow(data,cmap = 'Spectral_r')

    num_rows, num_cols = data.shape
    for i in range(num_rows):
        for j in range(num_cols):
            plt.gca().add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, edgecolor='white', linewidth=0.5, fill=False))



    # 设置x轴和y轴刻度
    plt.xticks(np.arange(len(df.columns)), df.columns, rotation=90,fontsize=8, ha='center')
    plt.yticks(np.arange(len(df.index)), df.index, fontsize=8)

    # 添加颜色栏
    plt.colorbar()
    #plt.savefig('./fig/Heatmap_all.pdf',bbox_inches='tight',dpi=300)
    # 显示图形
    plt.show()


