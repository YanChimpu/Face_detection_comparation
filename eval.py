from ap_parser import AveragePrecisionOnImages
import numpy as np
import cv2
import os


def get_rectangle_img(img, box, color):
    x, y, w, h = box[0], box[1], box[2], box[3]
    img_ = cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    return img_


def get_pred_dict(Pred_DIR_path):
    predictions = []
    for dir_path in os.listdir(Pred_DIR_path):
        for pred_res in os.listdir(pred_DIR_path + dir_path):
            pred_res_path = pred_DIR_path + dir_path + '/' + pred_res
            pred_bbx = []
            f_p = open(pred_res_path, 'r')
            for line in f_p.readlines():
                if line:
                    pred_bbx.append(list(int(i) for i in line.strip().split('\t')))
            pred_bbx = np.array(pred_bbx)
            for box in pred_bbx:
                info = {'bbox': box, 'confidence': 0.8, 'file_id': pred_res[:-4] + '.jpg'}
                predictions.append(info)
    return predictions


def get_gt_dict(gt_DIR_path_):
    gts = {}
    num_gt = 0
    for dir_path in os.listdir(gt_DIR_path_):
        for gt_res in os.listdir(gt_DIR_path_ + dir_path):
            gt_res_path = gt_DIR_path_ + dir_path + '/' + gt_res
            gt_bbx = []
            f_g = open(gt_res_path, 'r')
            for line in f_g.readlines():
                gt_bbx.append(list(int(i) for i in line.strip().split(' ')))
            gt_bbx = np.array(gt_bbx)
            num_gt += len(gt_bbx)
            img_name = gt_res[:-4] + '.jpg'
            gts[img_name] = gt_bbx
    return gts, num_gt


def main():
    predictions = get_pred_dict(pred_DIR_path)
    gts, num_gt = get_gt_dict(gt_DIR_path)
    ap, recall, precision = AveragePrecisionOnImages(gts, predictions, num_gt, min_overlap=0.5, validate_input=True)
    print('ap: {}'.format(ap))
    print('recall: {}'.format(recall))
    print('precision: {}'.format(precision))


if __name__ == '__main__':
    pred_DIR_path = 'test/pred/'
    gt_DIR_path = 'test/gt/'
    img_DIR_path = '/Users/pu/Documents/work/data/wider_face/WIDER_val/images/'
    main()
    #
    # pred_path = '/Users/pu/Documents/work/face_detection/Detection/result/1--Handshaking/' \
    #             '1_Handshaking_Handshaking_1_357.txt'
    # gt_path = '/Users/pu/Documents/work/data/wider_face/wider_face_split/wider_face_val/1--Handshaking/' \
    #           '1_Handshaking_Handshaking_1_357.txt'
    # img_path = '/Users/pu/Documents/work/data/wider_face/WIDER_val/images/1--Handshaking/' \
    #            '1_Handshaking_Handshaking_1_357.jpg'
    # img = cv2.imread(img_path)
    # for box in pred_bbx:
    #     img = get_rectangle_img(img, box, color=(255, 0, 0))
    # for box in gt_bbx:
    #     img = get_rectangle_img(img, box, color=(0, 255, 0))
    # cv2.imshow('image', img)
    # cv2.waitKey(0)

