import cv2 as cv # type: ignore

face_ref = cv.CascadeClassifier("face_ref.xml")

# Internal camera laptop
camera = cv.VideoCapture(0)

#  deteksi wajah
def face_detection(frame):
    #  ubah menjadi grayscale
    optimized_frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor=1.1, minNeighbors=4)
    return faces

# box kotak -> detektor
def drawer_box(frame):
    for x, y, w, h in face_detection(frame):
        cv.rectangle(frame, (x,y), (x + w, y + h), (0,255,0), 4)
    pass

def close_window():
    camera.release()
    cv.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read()
        drawer_box(frame)
        cv.imshow("GGS AI", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            close_window()
    
if __name__ == '__main__':
    main()