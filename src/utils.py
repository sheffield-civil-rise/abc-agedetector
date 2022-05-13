import os
import shutil


PARAM_VALID_IMAGE_TYPES = ['.jpg', '.jpeg', '.tiff', '.tif', '.png']


def is_valid_file(name):
    ''' function checker for '''
    _, ext = os.path.splitext(name)
    return (ext in PARAM_VALID_IMAGE_TYPES)


def _safe_create(path, replace, allowed):
    ''' safely creates directories
        if path already exists either replace or ignore
    '''
    path_exists = os.path.isdir(path)
    if path_exists and not allowed:
        return -1  # fail
    if path_exists and allowed and not replace:
        return 0   # success
    if path_exists and replace:
        shutil.rmtree(path)  # delete content and folder
        os.mkdir(path)  # recreate folder
        os.path.exist
        return 0
    if not path_exists:
        os.mkdir(path)
        return 0
    return 2  # should never reach here


def safe_create(path, replace=False, allowed=True):
    ''' outer definition with error handling '''
    try:
        status = _safe_create(path, replace, allowed)
    except FileExistsError as fee:
        status = -2
    except FileNotFoundError as fnf:
        status = -3
    except Exception as e:
        status = 1
        errstr = e.__str__

    if status != 0:
        raise EnvironmentError('failed to create folder, with error {}'.format({
             1: 'other error: {}'.format(errstr),
            -1: 'path already existed and this is not allowed',
            -2: 'path already existed and didn''t failed to deal with this',
            -3: 'parent directory doesn''t exist'}[status]))
