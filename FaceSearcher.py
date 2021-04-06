import cv2
import face_recognition
import os


class FaceSearcher:
    """
    Takes in each frame tells owner whether the frame has a face or not.
    Can also look for faces and return which previous frames contained that face.
    """

    def __init__(self):
        self.known_faces = []
        self.known_face_frames = []
        self.unknown_faces = []
        self.unknown_face_frames = []

    def add_known_face(self, filename):
        faceimg = face_recognition.load_image_file(filename)
        encs = face_recognition.face_encodings(faceimg)
        if len(encs) == 0:
            print("Couldn't get face encoding for file {}".format(filename))
            return
        self.known_faces.append(encs[0])
        self.known_face_frames.append([])

    def analyzeFrame(self, frame, framei):
        """
        returns a tuple (f, p):
        f = -1 if there is no face, or a face id if a face is found
        p = a list of frames where this face was previously seen (empty if f == -1)
        """
        cframe = frame[:, :, ::-1]  # BGR -> RGB
        facelocs = face_recognition.face_locations(cframe)
        facencs = face_recognition.face_encodings(cframe, facelocs)

        if len(facencs) == 0:
            assert len(facelocs) == 0
            return (-1, [])

        #  first check for known faces that we added before running the video
        for enc in facencs:
            match = face_recognition.compare_faces(self.known_faces, enc, tolerance=0.7)

            for i, m in enumerate(match):
                if m:
                    self.known_face_frames[i].append(framei)
                    return (i, self.known_face_frames[i])

        # If none of those found, check for new faces we've added while running the video
        for enc in facencs:
            match = face_recognition.compare_faces(self.unknown_faces, enc, tolerance=0.7)

            for i, m in enumerate(match):
                if m:
                    self.unknown_face_frames[i].append(framei)
                    return (len(self.known_faces) + i, self.unknown_face_frames[i])

        # Still nothing? There was a face that didn't match any known or new faces
        print("Adding new face to list of unknowns")
        # encs = face_recognition.face_encodings(det_img)
        encs = face_recognition.face_encodings(cframe)
        if len(encs) == 0:
            raise Exception("This should never happen")

        self.unknown_faces.append(encs[0])
        self.unknown_face_frames.append([framei])
        return (len(self.known_faces) + len(self.unknown_faces), [framei])


if __name__ == "__main__":
    data_dirs = ["/home/wcroughan/glasses_data/facial_recog", "/path/to/your/data/folder"]
    data_dir = ""
    for dd in data_dirs:
        if os.path.exists(dd):
            data_dir = dd
            break
    if data_dir == "":
        print("Couldn't find any of the folders listed in data_dirs. Add the folder on your machine to the list.")
        exit()

    input_video_path = os.path.join(data_dir, "music_15fps_480.mp4")

    vid = cv2.VideoCapture(input_video_path)
    if not vid.isOpened():
        print("Couldn't open video {}".format(input_video_path))
        exit(1)

    print("input video is {}x{}".format(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    fs = FaceSearcher()
    # Optionally, pass in knownfaces here
    fs.add_known_face(os.path.join(data_dir, "bill.png"))

    fi = 0
    num_frames = None

    while vid.isOpened():
        ret, f = vid.read()
        if not ret:
            break
        if num_frames is not None and fi >= num_frames:
            break

        (fsi, fsf) = fs.analyzeFrame(f, fi)

        print("Found face {}. We've seen this face before on frames {}".format(fsi, fsf))

        fi += 1
