from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from os import listdir
from os.path import join
import numpy as np

emb_dir = './embeddings/radius_2.0'
emb_list = listdir(emb_dir)
emb_list.sort

# embeddings = [(np.load(join(emb_dir,e), allow_pickle=True), e[:-4]) for e in emb_list]

vectors, labels = [], []
for label in emb_list:
    loaded = np.load(join(emb_dir,label), allow_pickle=True)

    for e in loaded:
        vectors.append(e.detach().numpy()[0])
        labels.append(label[:-4])

data = np.asarray(vectors)


# Normalize Data
data_norm = StandardScaler().fit_transform(data) # normalizing the features

pca = PCA(n_components=3)
pca.fit_transform(data_norm)

print('Covariance: ', pca.get_covariance())