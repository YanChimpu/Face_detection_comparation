import dlib
import cv2
import os
from tqdm import tqdm


class DlibFaceDetection:

    def face_dlib_cnn(self, image):
        detector = dlib.cnn_face_detection_model_v1("/opt/userhome/yangqingpu/workspace/Face_detection_comparation"
                                                    "/model/mmod_human_face_detector.dat")
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = detector(gray_img)
        return face

    def cnn_rect_to_bb(self, face):
        x = face.rect.left()
        y = face.rect.top()
        w = face.rect.right() - x
        h = face.rect.bottom() - y
        return (x, y, w, h)


def save_txt(txt_path, boxes):
    f = open(txt_path + '.txt', 'w')
    flag = 0
    for line in boxes:
        if flag < 2:
            f.write(line)
            f.write('\n')
            flag += 1
            continue
        else:
            for i in line:
                f.write(str(i))
                f.write(' ')
            f.write('\n')
    f.close()


if __name__ == "__main__":
    face_detection = DlibFaceDetection()
    img_dir = '/opt/userhome/yangqingpu/data/faces/WIDER_FACE/WIDER_val/images'
    save_dir = '/opt/userhome/yangqingpu/workspace/face_detection/dlib_cnn_result/'
    foldname = os.listdir(img_dir)
    for fold in tqdm(foldname):
        image_list = os.listdir(img_dir + '/' + fold)
        for image_name in tqdm(image_list):
            filename = image_name[:-4]
            image = cv2.imread(img_dir + '/' + fold + '/' + image_name)
            faces = face_detection.face_dlib_cnn(image)
            boxes = []
            boxes.append(image_name[:-4])
            boxes.append(str(len(faces)))
            for face in faces:
                box = face_detection.cnn_rect_to_bb(face)
                boxes.append(box)
            print(boxes)
            if not (os.path.exists(save_dir + fold)):
                os.makedirs(save_dir + fold)
            save_txt(save_dir + fold + '/' + filename, boxes, filename)
            print('finished detection on:' + image_name)
