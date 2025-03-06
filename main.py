import cv2 

face_ref = cv2.CascadeClassifier("face_ref.xml")
camera = cv2.VideoCapture(0)

known_faces = ["Iklil",]

def face_detection(frame):
    optimized_frame = cv2.cvtColor(frame ,cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(optimized_frame, scaleFactor= 1.1, minNeighbors=3)
    return faces

def drawer_box(frame):
    faces = face_detection(frame)
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

        # Tentukan nama (Hardcoded sesuai urutan wajah ke-detect)
        name = known_faces[i % len(known_faces)]

        # Tampilkan nama di atas kotak wajah
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (255, 0, 0), 2, cv2.LINE_AA)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()

def main():
    while True:
        _, frame = camera.read()
        drawer_box(frame)
        cv2.imshow("Face AI", frame)

        if cv2.waitKey(1) &  0xFF == ord('q'):
            close_window()

if __name__  == '__main__':
    main()