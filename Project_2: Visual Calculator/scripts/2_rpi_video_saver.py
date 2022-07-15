import cv2


def main():
    video_feed = cv2.VideoCapture(1)
    video_saver = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc(*'MJPG'),30, (640,480))
    while(1):
        _,frame = video_feed.read()
        cv2.imshow("camera_feed", frame)
        video_saver.write(frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()


