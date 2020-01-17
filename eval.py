from ap_parser import AveragePrecisionOnImages
import numpy as np
import cv2


def get_rectangle_img(img, box, color):
    x, y, w, h = box[0], box[1], box[2], box[3]
    img_ = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return img_


if __name__ == '__main__':
    pred_path = '/Users/pu/Documents/work/face_detection/Detection/result/1--Handshaking/' \
                '1_Handshaking_Handshaking_1_357.txt'
    gt_path = '/Users/pu/Documents/work/data/wider_face/wider_face_split/wider_face_val/1--Handshaking/' \
              '1_Handshaking_Handshaking_1_357.txt'
    img_path = '/Users/pu/Documents/work/data/wider_face/WIDER_val/images/1--Handshaking/' \
               '1_Handshaking_Handshaking_1_357.jpg'
    pred_bbx = []
    f_p = open(pred_path, 'r')
    for line in f_p.readlines():
        if line:
            pred_bbx.append(list(int(i) for i in line.strip().split('\t')))
    gt_bbx = []
    f_g = open(gt_path, 'r')
    for line in f_g.readlines():
        gt_bbx.append(list(int(i) for i in line.strip().split(' ')))
    pred_bbx = np.array(pred_bbx)
    gt_bbx = np.array(gt_bbx)
    img = cv2.imread(img_path)
    for box in pred_bbx:
        img = get_rectangle_img(img, box, color=(255, 0, 0))
    for box in gt_bbx:
        img = get_rectangle_img(img, box, color=(0, 255, 0))
    cv2.imshow('image', img)
    cv2.waitKey(0)
    img_name = '1_Handshaking_Handshaking_1_357.jpg'
    gts = {img_name: gt_bbx}
    predictions = []
    for box in pred_bbx:
        info = {'bbox': box, 'confidence': 0.8, 'file_id': img_name}
        predictions.append(info)
    num_gt = len(gts[img_name])
    ap, recall, precision = AveragePrecisionOnImages(gts, predictions, num_gt, min_overlap=0.1, validate_input=True)
    print('ap: {}'.format(ap))
    print('recall: {}'.format(recall))
    print('precision: {}'.format(precision))
