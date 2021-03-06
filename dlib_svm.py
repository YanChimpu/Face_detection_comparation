import dlib
import cv2
import os


class DlibFaceDetection:

    def face_dlib_svm(self, image):
        detector = dlib.get_frontal_face_detector()
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face = detector(gray_img)
        return face

    def rect_to_box(self, face):
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
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
    save_dir = '/opt/userhome/yangqingpu/workspace/face_detection/rnn_result/'
    # img_dir = "/Users/pu/Documents/work/data/test_img/wider_face/WIDER_val/images"
    # save_dir = "/Users/pu/Documents/work/data/test_img/res/"
    foldname = os.listdir(img_dir)
    for fold in foldname:
        image_list = os.listdir(img_dir + '/' + fold)
        for image_name in image_list:
            filename = image_name[:-4]
            image = cv2.imread(img_dir + '/' + fold + '/' + image_name)
            faces = face_detection.face_dlib_svm(image)
            boxes = []
            boxes.append(image_name[:-4])
            boxes.append(str(len(faces)))
            for face in faces:
                box = face_detection.rect_to_box(face)
                boxes.append(box)
            print(boxes)
            if not (os.path.exists(save_dir + fold)):
                os.makedirs(save_dir + fold)
            save_txt(save_dir + fold + '/' + filename, boxes)
            print('finished detection on:' + image_name)
