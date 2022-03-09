import pandas as pd
import torch
from img2vec_pytorch import Img2Vec
from PIL import Image
from tqdm import tqdm


def fe_ima2vec(df, img_path, img_col, filename_extension, sub_path = None):
    img2vec = Img2Vec(cuda=(torch.cuda.is_available()))

    fe_img = []
    for idx in tqdm(range(len(df))):
        path = f'{img_path}/{df.loc[idx, img_col]}.{filename_extension}'
        if sub_path:
            path = f'{img_path}/{df.loc[idx, sub_path]}/{df.loc[idx, img_col]}.{filename_extension}'
        try:
            img = Image.open(path)
            vec = img2vec.get_vec(img)
        except:
            vec = [None] * 512
        fe_img.append(vec)

    fe_img = pd.DataFrame(fe_img)
    fe_img.columns = ['img_vec_' + str(i) for i in range(fe_img.shape[1])]
    return fe_img