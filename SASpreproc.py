import numpy as np
import glob
import os
import ntpath
import re
import pandas as pd
import tqdm
from multiprocessing import Pool
import functools
import itertools

crop_pat = re.compile('(?<=_)(?:.(?!_))+$')
gpstime_pat = re.compile('\d+\.\d+')

# for x in dirnames[1]:
def get_filenames(x, dirnames_0):
    try:
        crop = crop_pat.search(x)[0]
        extracted_dir = os.path.join(dirnames_0, x)
        try:
            return [(os.path.join(*os.path.normpath(y).split(os.sep)[-3:]), float(gpstime_pat.search(ntpath.split(y)[1])[0]), crop) for y in
                            glob.glob(os.path.join(dirnames_0, x, extracted_dir, r'**\*.png'), recursive=True) if
                            os.stat(y).st_size > 1e6]
            # break
        except:
            return None
    except:
        return None


if __name__ == '__main__':
    dirnames = next(os.walk(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2020_ImageLibrary_FieldLogs_Jabil\Extracted_Logs'))
    listing = []
    copier = functools.partial(get_filenames, dirnames_0=dirnames[0])

    with Pool(16) as p:
        listing = list(tqdm.tqdm(p.imap_unordered(copier, dirnames[1]), total=len(dirnames[1])))

    listing = [x for x in listing if x is not None]
    listing = list(itertools.chain.from_iterable(listing))
    df = pd.DataFrame(listing, columns=['path', 'unix_time', 'crop'])
    df.to_pickle(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2020_ImageLibrary_FieldLogs_Jabil\Extracted_Logs\pandas_path_time_crop_array.pickle')

## 61924

# df = pd.read_pickle(r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_ImageLibrary_FieldLogs\pandas_path_time_crop_array.pickle')
# df.crop[df.crop == 'NoCrop'] = 'Nocrop'
# '''
# load images, resize/rescale, standardize and then do PCA and identify outlier images like r'Z:\ftp\sprayers\IntelligentSprayTechnology\Connor_Field_Data\2019_
# ImageLibrary_FieldLogs\2019-06-04_AR_Soybeans\Extracted Images\PCPM2HA000168.04062019-18-15-00\-1_3_3149923268.2782_3149923268.2782_10_3__20937.bmp'
# which are not images of crop and eliminate from consideration to focus on extracting crop features
# '''
