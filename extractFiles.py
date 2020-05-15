import os
from os.path import splitext, split, dirname, join, isdir

########################################################################
# Copy this script to dataset directory, containing all folders of images
# and execute it.
# Type in the target folder and a directory 
# in the parent folder will be created and pictures be moved.
# Make sure pictures_to_move and the step are valid entries
########################################################################
# This moves picture 001, 011, 021, ..., 071
pictures_to_move = 8 
step = 10
########################################################################

targ_folder = input("Please input the name of the target folder: \n")
cwd = os.getcwd()

while not isdir(join(cwd, targ_folder)):
    print("\nFolder does not exist.")
    targ_folder = input("Please input the name of the target folder: \n")

targ_dir = join(cwd, targ_folder)

dirs = os.listdir(targ_dir)
dirs.sort()
# assert valid entries for the set parameters
assert (pictures_to_move-1) * step < len(dirs), "Amount of images in the folder won't fit with the parameters!"
file_name = splitext(dirs[0])[0]
folder = file_name[:len(file_name)-3] + "-testset"

print("The folder ", folder, " will be created in following directory:")
print(cwd)
input("If that is ok, press enter. Otherwise stop! (Ctrl+C)" + '\n')
print("Ok, creating folder and moving images...")

new_dir = join(cwd, folder)
os.mkdir(new_dir)

for i in range(pictures_to_move):
    i *= step
    os.rename(join(targ_dir, dirs[i]), join(new_dir, dirs[i]))

print(pictures_to_move, " pictures were moved to ", new_dir)
