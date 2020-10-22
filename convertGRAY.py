from PIL import Image
import PIL
from os.path import join
from os import listdir

path = '../samples/fusion2040'
imgs = listdir(path)


for img in imgs:
    full_path = join(path, img)

    img0 = Image.open(full_path)
    # Grey conversion
    img0 = img0.convert("L")

    img0.save(join(path+'_gray', img))
