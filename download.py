import sys
import os 
import numpy as np
import zipfile
import shutil
from urllib.request import urlretrieve
from tqdm import tqdm

data_dir = './input/'

def _read32(bytestream):
    
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]

def _unzip(save_path, _, database_name, data_path):
    print('Extracting {}...'.format(database_name))
    with zipfile.ZipFile(save_path) as zf:
        zf.extractall(data_path)

def main():
    script = sys.argv[0]
    dataset = sys.argv[1]

    if dataset == 'cat':
        download('cat',data_dir)
        
    elif dataset == 'dog':
        download('dog',data_dir)
        
    elif dataset == 'celeba':  
        download('celeba',data_dir)

    else:
        print('No databese found with the name {}'.format(str(dataset)))



def download(database_name, data_path):

    DATASET_CELEBA_NAME = 'celeba'
    DATASET_CAT_NAME = 'cat'
    DATASET_DOG_NAME = 'dog'

    if database_name == DATASET_CELEBA_NAME:
        url = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        extract_path = os.path.join(data_path, 'img_align_celeba/')
        save_path = os.path.join(data_path, 'celeba.zip')
        extract_fn = _unzip
    elif database_name == DATASET_CAT_NAME:
        url = 'http://127.0.1.1:8080/cat.zip'
        hash_code = 'f68b3c2dcbeaaa9fbdd348bbdeb94873'
        extract_path = os.path.join(data_path, 'cat/')
        save_path = os.path.join(data_path, 'cat.zip')
        extract_fn = _unzip
    elif database_name == DATASET_DOG_NAME :
        url = 'https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip'
        hash_code = '00d2c5bc6d35e252742224ab0c1e8fcb'
        extract_path = os.path.join(data_path, 'dog/')
        save_path = os.path.join(data_path, 'celeba.zip')
        extract_fn = _unzip

    if os.path.exists(extract_path):
        print('Found {} Data'.format(database_name))
        return

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(save_path):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='Downloading {}'.format(database_name)) as pbar:
            urlretrieve(
                url,
                save_path,
                pbar.hook)

    os.makedirs(extract_path)
    try:
        extract_fn(save_path, extract_path, database_name, data_path)
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error
        raise err

    # Remove compressed data
    os.remove(save_path)


class DLProgress(tqdm):

    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num

main()