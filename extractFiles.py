import os
from os.path import splitext, split, dirname, join

########################################################################
# Copy this script into the dataset directory, e.g "konrad_von"
# and execute it
# A directory in the parent folder will be created and pictures be moved.
# Make sure pictures_to_move and the step are valid entries
########################################################################
# This moves picture 001, 010, 020, ..., 070
pictures_to_move = 8 
step = 10
########################################################################



cwd = os.getcwd()
new_dir = split(cwd)[0]
dirs = os.listdir(cwd)
# assert valid entries for the set parameters
assert (pictures_to_move-1) * step < len(dirs)
file_name = splitext(dirs[5])[0]
folder = file_name[:len(file_name)-3] + "-testset"

print("The folder ", folder, " will be created in following directory:")
print(new_dir)
input("If that is ok, press enter. Otherwise stop! (Ctrl+C)" + '\n')
print("Ok, creating folder and moving images...")

new_dir = join(new_dir, folder)
os.mkdir(new_dir)

for i in range(pictures_to_move):
    i *= step
    os.rename(join(cwd, dirs[i]), join(cwd, folder))

print(pictures_to_move, " pictures were moved to ", new_dir)
