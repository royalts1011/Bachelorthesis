###################################################
###################################################
###    Script to outsource functions
###################################################
###################################################
import os
from torch import cuda

# method for displaying files with index
def print_list(list_):
    fmt = '{:<8}{:<20}'
    print(fmt.format('Index', 'Name'))
    for i, name in enumerate(list_):
        print(fmt.format(i, name))

# removes .DS_Store string from a list
def rm_DSStore(list_):
    return list(filter(('.DS_Store').__ne__, list_))

# takes a path input and shows all folders
# Let's user choose a folder by index
def choose_folder(dataset_path, name='unknown'):
    folders = os.listdir(dataset_path)
    # remove all .DS_Store entries
    folders = rm_DSStore(folders)
    folder_name = name
    # if statement starts, when no name was given
    if folder_name == 'unknown':
        print_list(folders)
        # Handle the user's input for user name
        while True:
            idx = input('Choose your folder name by index: ')
            try:
                idx = int(idx)
                assert idx < len(folders) and idx >= 0
                break
            except (ValueError, AssertionError):
                print('The input was a string or not in the index range.')
        folder_name = folders[idx]
    
    assert folder_name in folders, 'The name was not found in the given folder: ' + dataset_path
    return folder_name

# converts to cuda if possible
def cuda_conv(obj):
    if cuda.is_available():
        return obj.cuda()
    else:
        return obj