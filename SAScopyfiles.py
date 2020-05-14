import os
import numpy as np
import re
from pathlib import Path
from skimage import io, transform, util
from multiprocessing import Pool
import tqdm
import glob
import pandas as pd
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
datapath = r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2020_ImageLibrary_FieldLogs_Jabil\Extracted_Logs'
target_dir = r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2020_ImageLibrary_FieldLogs_Jabil\Extracted_Logs'
def copy_image(path):
    # Path(os.path.join(target_dir, os.path.split(path)[0])).mkdir(parents=True, exist_ok=True)
    img = io.imread(os.path.join(datapath, path))
    return tf.image.encode_png(util.img_as_ubyte(transform.rescale(util.img_as_float32(img), 0.1, multichannel=True)), compression=9).numpy()
    # io.imsave(os.path.join(target_dir, os.path.splitext(path)[0] + '.png'), util.img_as_ubyte(transform.rescale(util.img_as_float32(img), 0.1, multichannel=True)))

if __name__ == '__main__':
    base_path = r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2020_ImageLibrary_FieldLogs_Jabil\Extracted_Logs'
    # crop_pat = re.compile('(?<=_)(?:.(?!_))+$')
    # imgs_paths = glob.glob(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2020_ImageLibrary_FieldLogs_Jabil\Extracted_Logs\**\*.png',recursive=True)
    imgs_paths = pd.read_pickle(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2020_ImageLibrary_FieldLogs_Jabil\Extracted_Logs\pandas_path_time_crop_array.pickle')

    # imgs_paths = np.load(os.path.join(datapath, 'filtered_img_paths.npy'))
    # class_names = np.array([crop_pat.search(x.split(os.sep)[-4])[0] for x in imgs_paths])
    # imgs_paths = np.array([os.path.join(datapath, filepath) for filepath in imgs_paths])

    # for crop in np.unique(class_names):
    #     Path(os.path.join(target_dir, crop)).mkdir(parents=True, exist_ok=True)

    with Pool(24) as p:
        out = list(tqdm.tqdm(p.imap(copy_image, imgs_paths.path.values), total=len(imgs_paths)))

    np.save(os.path.join(base_path, 'imgs_compressed'), out)