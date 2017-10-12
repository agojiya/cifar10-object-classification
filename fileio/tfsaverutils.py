import os


def get_highest_epoch_saved(dir_path):
    files = [file for file in os.listdir(dir_path) if isvalid(dir_path, file)]
    highest_num = 0
    for file in files:
        num = int(str(file).split('-')[-1].replace('.meta', ''))
        if num > highest_num:
            highest_num = num
    return highest_num


def isvalid(file_path, file):
    return os.path.isfile(file_path + '/' + file) \
           and str(file).endswith('.meta')
