import os

def basename(file_path):
    return os.path.basename(file_path)

def filename_subfix(file_path):
    spl_strs = file_path.split(".")
    if len(spl_strs) < 2:
        return spl_strs[0], ""
    else:
        return ".".join(spl_strs[:-1]), spl_strs[-1]