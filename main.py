# 抠图程序
"""
生成一个mask，在mask以内的部分，分错的 全部修改成背景

"""

import tifffile
import numpy as np
from matplotlib import pyplot as plt


def mask_gen(img,leftup, rightdown):  #输入mask的左上角和右下角坐标
    #leftup (a,b)   rightdown(c,d)  a<c b<d
    a,b = leftup
    c,d = rightdown
    if img is None:
        mask = np.ones([400,400])
    else:
        mask = img.copy()
    mask[a:c,b:d]=0
    return mask

def mask_check(img,mask,false_seris):
    w, h = false_seris.shape
    for i in range(w):
        for j in range(h):
            if false_seris[i][j]==1 and mask[i][j]==0:
                img[i][j] = 0
    return img





if __name__ == '__main__':
    img = np.random.randint(0,255,[400,400])

    mask = mask_gen(None,(10,10),(100,100))
    # plt.imshow(mask1)
    # plt.show()

    mask = mask_gen(mask,(60,60),(150,100))
    # plt.imshow(mask2)
    # plt.show()


    mask = mask_gen(mask,(210,180),(280,340))

    plt.imshow(mask)
    plt.show()
    # 添加一个序列，这个序列标记了分错的位置
    # 序列长度为160000
    false_seris = np.zeros(160000)
    false_seris[:10000] = 1
    false_seris = false_seris.reshape(400,400)
    img = mask_check(img,mask,false_seris)
    plt.imshow(img)
    plt.show()
#

