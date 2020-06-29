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

# prints two lists next to each other
def print_predictions(list_left, list_right):
    fmt = '{:<20} {:<10}'
    print(fmt.format('Name', 'Value'))
    for l, r in zip(list_left, list_right):
        print(fmt.format(l, r))

# removes .DS_Store string from a list
def rm_DSStore(list_):
    return list(filter(('.DS_Store').__ne__, list_))

def type_conversion(obj):
    if cuda.is_available():
        return obj.type('torch.cuda.FloatTensor')
    else:
        return obj.type('torch.FloatTensor')

# takes a path input and shows all folders
# Let's user choose a folder by index
def choose_data(dataset_path, name='unknown', data_description='folder'):
    folders = os.listdir(dataset_path)
    # remove all .DS_Store entries
    folders = rm_DSStore(folders)
    folder_name = name
    # if statement starts, when no name was given
    if folder_name == 'unknown':
        print_list(folders)
        # Handle the user's input for user name
        while True:
            idx = input('Choose your ', data_description, ' by index: ')
            try:
                idx = int(idx)
                assert idx < len(folders) and idx >= 0
                break
            except (ValueError, AssertionError):
                print('The input was a string or not in the index range.')
        folder_name = folders[idx]
    
    assert folder_name in folders, 'The name was not found in the given folder: ' + dataset_path
    return folder_name

