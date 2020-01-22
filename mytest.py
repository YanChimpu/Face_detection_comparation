from scipy.io import loadmat
import matplotlib.pyplot as plt
import pandas as pd


gt_mat = loadmat('/Users/pu/Documents/work/data/wider_face/eval_tools/ground_truth/wider_face_val.mat')
facebox_list = gt_mat['face_bbx_list']
file_list = gt_mat['file_list']
event_list = gt_mat['event_list']
# print(file_list[0][0][0])
# print(facebox_list[0][0][0])
# img = cv2.imread("/Users/pu/Documents/work/data/wider_face/WIDER_val/images/0--Parade/0_Parade_marchingband_1_465.jpg")
# for i in facebox_list[0][0][0]:
#     for box in i:
#         x, y, w, h = box[0], box[1], box[2] + box[0], box[3] + box[1]
#         cv2.rectangle(img, (x, y), (w, h), (0, 0, 255), 2)
# cv2.imshow('img', img)
# cv2.waitKey(0)
area = []
for i in facebox_list:
    for j in i:
        for k in j:
            for x in k:
                for y in x:
                    # if y[2] * y[3] > 10000:
                        area.append(y[2] * y[3])
print(len(area))
# area = pd.DataFrame(area)
# area.plot.box(title="Consumer spending in each country")
# plt.grid(linestyle="--", alpha=0.3)
# plt.show()
# fig, axes = plt.subplots()
# area.plot(kind='box', ax=axes)
# axes.set_ylabel('values of tip_pct')
# fig.savefig('p1.png')
# import numpy as np
# 
# ignore = np.zeros(5)
# keep_index = np.array([[1], [3], [4], [5]])
# a = np.array([[0], [2], [3]])
# b = np.array([i for i in keep_index if i-1 in a])
# # print(keep_index)
# # ignore[keep_index-1] = 1
# print(b)