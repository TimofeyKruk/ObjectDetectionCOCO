import cv2
import numpy as np

'''
This file is responsible for online video analysis
and drawing predictions on the video.
'''


def analyze_video(model):
    cap = cv2.VideoCapture(0)

    # for saving the video
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("output4.avi", fourcc, 20, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()

        cv2.rectangle(frame, (np.random.randint(30, 230), np.random.randint(30, 230)),
                      (np.random.randint(30, 330), np.random.randint(30, 330)), (255, 120, 120), thickness=2)

        out.write(frame)
        cv2.imshow("frame", frame)
        # An exit from the infinitive loop
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    analyze_video("HERE must be a model!")
