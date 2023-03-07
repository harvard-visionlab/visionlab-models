'''
    reference: https://github.com/pytorch/pytorch/blob/master/torch/hub.py
'''
import os
import torch
import hashlib
import tempfile
import shutil
from urllib.error import HTTPError
from urllib.request import urlopen, Request
from urllib.parse import urlparse  # noqa: F401
from torch.hub import tqdm
from pathlib import Path
from glob import glob
from pdb import set_trace

__all__ = ['get_file_hash', 'check_hashid', 'get_remote_hash', 'download_url_to_file_with_hash']

default_cache_dir = os.path.join(torch.hub.get_dir(), "checkpoints")
                                 
def get_file_hash(filename):
    with open(filename,"rb") as f:
        bytes = f.read() # read entire file as bytes
        readable_hash = hashlib.sha256(bytes).hexdigest();
        
    return readable_hash

def check_hashid(hashid, weights_path):
    weights_hash = get_file_hash(weights_path)
    assert weights_hash.startswith(hashid), f"Oops, expected weights_hash to start with {hashid}, got {weights_hash}"
    return True


def get_remote_hash(url, progress=True):
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    try:
        sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                sha256.update(buffer)
                pbar.update(len(buffer))

        digest = sha256.hexdigest()
    finally:
        pass
    
    return digest

def download_url_to_file_with_hash(url, dst=default_cache_dir, hash_len=8, progress=True):
    r"""Download object at the given URL to a local path, appending hashid
    Args:
        url (string): URL of the object to download
        dst (string): Full path where object will be saved, e.g. ``/tmp/temporary_file``
        hash_len (int, optional): If not None, append this many digits of the SHA256 to the downloaded filename
        progress (bool, optional): whether or not to display a progress bar to stderr
            Default: True
    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')
    """
    digest = None
    file_size = None
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_len is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_len is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        
        if hash_len is not None:
            digest = sha256.hexdigest()
            hashid = digest[0:hash_len]
            dst = dst.replace(".pth", f"-{hashid}.pth")     
        
        shutil.move(f.name, dst)
                               
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
            
    return digest, dst

def download_and_get_cache_filename(url, cache_dir=default_cache_dir, hash_len=8):
    dst = os.path.join(torch.hub.get_dir(), "checkpoints", Path(url).name)
    files = glob(dst.replace(".pth", "-*.pth"))
    if len(files)==0:
        dst = os.path.join(cache_dir, Path(url).name)
        filehash,cache_file_name = download_url_to_file_with_hash(url, dst=dst, hash_len=hash_len)
    else:
        assert len(files)==1, f"Oops, expected one file, got {len(files)}"
        cache_file_name = Path(files[0]).name
        print(f"skipping download, file exists: {cache_file_name}")
    
    return cache_file_name