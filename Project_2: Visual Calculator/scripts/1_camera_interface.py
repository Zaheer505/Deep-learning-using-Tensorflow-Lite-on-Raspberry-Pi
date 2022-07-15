import cv2


def main():
    video_feed = cv2.VideoCapture(1)

    while(1):
        _,frame = video_feed.read()
        cv2.imshow("camera_feed", frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()


