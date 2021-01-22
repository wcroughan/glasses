import face_recognition
import cv2
import os
import time
import datetime

# from imageai.Detection import ObjectDetection

SHOW_VID = True
SAVE_VID = True

data_dirs = ["/home/wcroughan/glasses_data", "/path/to/your/data/folder"]
data_dir = ""
for dd in data_dirs:
    if os.path.exists(dd):
        data_dir = dd
        break
if data_dir == "":
    print("Couldn't find any of the folders listed in data_dirs. Add the folder on your machine to the list.")
    exit()

input_video_path = os.path.join(data_dir, "P01.mp4")
output_video_path = os.path.join(os.getcwd(), "outvid.avi")


class MyVideoAnalyzer:
    def __init__(self):
        # self.detector = ObjectDetection()
        # self.detector.setModelTypeAsYOLOv3()
        # self.detector.setModelPath(os.path.join(os.getcwd(), "yolo.h5"))
        # self.detector.loadModel()
        # self.cobs = self.detector.CustomObjects(person=True)
        # self.detection_box1 = (300, 100)
        # self.detection_box2 = (800, 480)

        self.detection_box1 = (0, 0)
        self.detection_box2 = (240, 220)

        self.output_box1 = (0, 220)
        self.output_box2 = (200, 320)
        self.output_msg1 = ""
        self.output_msg2 = ""
        self.output_msg_change_time = 0
        self.output_msg_duration = 1500
        self.output_msg_duration_noface = 750

        self.known_faces = []
        self.known_faces_names = []
        self.known_faces_hints = []

        self.unknown_faces = []
        self.unknown_faces_names = []
        self.unknown_faces_hints = []

        self.unknown_faces_dir = "./unknown_faces"
        self.save_unknown_faces = True
        self.run_pfx = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    def add_known_face(self, filename, name, hint):
        faceimg = face_recognition.load_image_file(filename)
        encs = face_recognition.face_encodings(faceimg)
        if len(encs) == 0:
            print("Couldn't get face encoding for file {}".format(filename))
            return
        self.known_faces.append(encs[0])
        self.known_faces_names.append(name)
        self.known_faces_hints.append(hint)

    def process_frame(self, frame, frame_index):
        # grab detection box
        det_img = frame[self.detection_box1[1]:self.detection_box2[1],
                        self.detection_box1[0]:self.detection_box2[0]]

        # draw detection box in output
        cv2.rectangle(frame, self.detection_box1, self.detection_box2, (255, 255, 255))

        # Object recognition code, imageai library
        # retim, objs = self.detector.detectObjectsFromImage(
        # custom_objects=self.cobs, input_image=det_img, input_type="array", output_type="array")
        # input_image=det_img, input_type="array", output_type="array")
        # print(objs)
        # for ob in objs:
        #     # print(ob)
        #     bp = ob['box_points']
        #     x1 = bp[0] + self.detection_box1[0]
        #     x2 = bp[2] + self.detection_box1[0]
        #     y1 = bp[1] + self.detection_box1[1]
        #     y2 = bp[3] + self.detection_box1[1]
        #     cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0))

        # Face recognition code, face_recognition library
        cframe = det_img[:, :, ::-1]  # BGR -> RGB
        facelocs = face_recognition.face_locations(cframe)
        facencs = face_recognition.face_encodings(cframe, facelocs)

        found_face = False
        cmil = round(time.time() * 1000)

        #  first check for known faces that we added before running the video
        for enc in facencs:
            match = face_recognition.compare_faces(self.known_faces, enc, tolerance=0.5)

            for i, m in enumerate(match):
                if m:
                    self.output_msg1 = self.known_faces_names[i]
                    self.output_msg2 = self.known_faces_hints[i]
                    self.output_msg_change_time = cmil + self.output_msg_duration
                    found_face = True
                    break

            if found_face:
                break

        # If none of those found, check for new faces we've added while running the video
        if not found_face:
            for enc in facencs:
                match = face_recognition.compare_faces(self.unknown_faces, enc, tolerance=0.5)

                for i, m in enumerate(match):
                    if m:
                        self.output_msg1 = self.unknown_faces_names[i]
                        self.output_msg2 = self.unknown_faces_hints[i]
                        self.output_msg_change_time = cmil + self.output_msg_duration_noface
                        found_face = True
                        print("Found an unknown face!")
                        break

                if found_face:
                    break

        # Still nothing? If we have a face, add it to list of new faces
        if not found_face:
            if len(facencs) > 0:
                # There was a face that didn't match any known or new faces
                print("Adding new face to list of unknowns")
                encs = face_recognition.face_encodings(det_img)
                if len(encs) == 0:
                    raise Exception("This should never happen")

                self.unknown_faces.append(encs[0])
                self.unknown_faces_names.append("new_{}".format(len(self.unknown_faces)-1))
                self.unknown_faces_hints.append("{}".format(frame_index))

                fname = os.path.join(self.unknown_faces_dir,
                                     "{}_{}.png".format(self.run_pfx, frame_index))
                cv2.imwrite(fname, det_img)

                self.output_msg1 = self.unknown_faces_names[-1]
                self.output_msg2 = self.unknown_faces_hints[-1]
                self.output_msg_change_time = cmil + self.output_msg_duration_noface

            else:
                if self.output_msg_change_time > 0 and cmil > self.output_msg_change_time:
                    self.output_msg_change_time = 0
                    self.output_msg1 = ""
                    self.output_msg2 = ""

        # Add output section to output image
        cv2.rectangle(frame, self.output_box1, self.output_box2, (255, 255, 0), 1)
        cv2.putText(
            frame, self.output_msg1, (self.output_box1[0], self.output_box1[1]+50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))
        cv2.putText(
            frame, self.output_msg2, (self.output_box1[0], self.output_box1[1]+100), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255))

    def process_video(self, filename, num_frames=None):
        vid = cv2.VideoCapture(filename)
        if not vid.isOpened():
            print("Couldn't open video {}".format(filename))
            return 1

        if SAVE_VID:
            w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fr = vid.get(cv2.CAP_PROP_FPS)
            print("invid is {}x{}".format(w, h))
            writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(
                'M', 'J', 'P', 'G'), fr, (w, h))

        framei = 0
        while vid.isOpened():
            ret, frame = vid.read()
            if not ret:
                break
            if num_frames is not None and framei >= num_frames:
                break

            self.process_frame(frame, framei)
            if SAVE_VID:
                writer.write(frame)
                if not SHOW_VID:
                    print("frame {}".format(framei))

            if SHOW_VID:
                cv2.imshow('frame', frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            framei += 1

        if SAVE_VID:
            writer.release()
        vid.release()
        if SHOW_VID:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    mva = MyVideoAnalyzer()
    mva.add_known_face(os.path.join(data_dir, "facecap2.png"), "guy", "is not blurry")
    mva.process_video(input_video_path, num_frames=350)
