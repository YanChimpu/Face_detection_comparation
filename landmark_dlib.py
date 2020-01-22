import cv2
import dlib
import os


class LandmarksDetction():

    def face_detection(self, img):
        detector = dlib.get_frontal_face_detector()
        predictors = dlib.shape_predictor("/opt/userhome/yangqingpu/workspace/model/shape_predictor_68_face_landmarks"
                                          ".dat")
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray_img)
        landmarks = []
        for face in faces:
            landmarks.append(predictors(gray_img, face))
        return landmarks

    def landmark_figure(self, landmarks):
        for landmark in landmarks:
            for i in range(68):
                (a, b) = (landmark.part(i).x, landmark.part(i).y)
                cv2.circle(img, (a, b), 3, (0, 0, 255), -1)  # 画图
        cv2.imwrite(os.path.join(save_path, img_name), img)


if __name__ == "__main__":
    img_path = "/opt/userhome/yangqingpu/workspace/face_detection/test_img"
    save_path = '/opt/userhome/yangqingpu/workspace/face_detection/landmark_res'
    for img_name in os.listdir(img_path):
        filename = os.path.join(img_path, img_name)
        img = cv2.imread(filename)
        ld = LandmarksDetction()
        landmarks = ld.face_detection(img)
        ld.face_detection(landmarks)
    #     # cv2.putText(img, str(i), (a, b), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
    #     #         (0,255,0), 1)
    # cv2.imshow("Output", img)
    # cv2.waitKey(0)