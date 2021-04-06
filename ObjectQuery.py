from imageai.Detection import VideoObjectDetection
import os
import json
import cv2
import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

tf.debugging.set_log_device_placement(True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.compat.v1.Session(config=config)

FORCE_REBUILD = True
SAVE_ALL_AND_EXIT = True

SHOW_VID = False
SAVE_VID = True
SAVE_ORIGINAL_VID = True

# 0 - UT egocentric video with blurred faces, first 350 frames
# 1 - music room video
# 2 - music room video, first second
INVID_ID = 0

data_dirs = ["/home/wcroughan/glasses_data", "/path/to/your/data/folder"]
data_dir = ""
for dd in data_dirs:
    if os.path.exists(dd):
        data_dir = dd
        break
if data_dir == "":
    print("Couldn't find any of the folders listed in data_dirs. Add the folder on your machine to the list.")
    exit()


if INVID_ID == 0:
    input_video_path = os.path.join(data_dir, "P01.mp4")
elif INVID_ID in [1, 2]:
    input_video_path = os.path.join(data_dir, "music_15fps_480.mp4")
else:
    print("unknown input video index")
    exit()

output_video_path = os.path.join(data_dir, "outvid.avi")
output_original_video_path = os.path.join(data_dir, "outvid_original.avi")
saveFile = os.path.join(data_dir, "VideoSearcherObject{}.dat".format(INVID_ID))


class VideoSearcher:
    def __init__(self):
        self.detector = VideoObjectDetection()
        # self.detector.setModelTypeAsYOLOv3()
        # self.detector.setModelPath(os.path.join(data_dir, "yolo.h5"))
        self.detector.setModelTypeAsTinyYOLOv3()
        self.detector.setModelPath(os.path.join(data_dir, "yolo-tiny.h5"))
        self.detector.loadModel()
        self.cobs = self.detector.CustomObjects(person=True)

        self.videoAnalyzed = False

        self.objectFrames = dict()
        self.videoFile = ""

        self.output_arrays = []
        self.count_arrays = []
        self.average_output_count = dict()

    def VideoCompleteFunction(self, output_arrays, count_arrays, average_output_count):
        # print(output_arrays)
        # output_arrays[i][j] = dictionary for jth object detected in the ith frame with keys name, percentage_probability, box_points
        # print(count_arrays)
        # count_arrays[i] = dictionary for ith frame. Keys are all objects that are in frame, values are number of that object in frame
        # print(average_output_count)
        # average_output_count = dictionary, where keys are all objects found in video, values are all zero, but maybe count per frame rounded down?

        self.output_arrays = output_arrays
        self.count_arrays = count_arrays
        self.average_output_count = average_output_count

        all_obs_found = average_output_count.keys()
        for ob in all_obs_found:
            self.objectFrames[ob] = [i for i in range(
                len(count_arrays)) if ob in count_arrays[i].keys()]

        print(self.objectFrames)
        self.videoAnalyzed = True

    def analyzeVideo(self, filename, dur=None):
        if self.videoAnalyzed:
            print("Already analyzed a video")
            return

        vid = cv2.VideoCapture(filename)
        fps = vid.get(cv2.CAP_PROP_FPS)
        vid.release()

        self.videoFile = filename

        # self.detector.detectObjectsFromVideo(input_file_path=filename, save_detected_video=False, log_progress=True,
        #  frames_per_second=fps, video_complete_function=self.VideoCompleteFunction)
        self.detector.detectObjectsFromVideo(input_file_path=filename, output_file_path=output_video_path, log_progress=True,
                                             frames_per_second=fps, video_complete_function=self.VideoCompleteFunction, detection_timeout=dur)
        #  frames_per_second=fps, video_complete_function=self.VideoCompleteFunction)

    def searchForObject(self, objectName):
        return self.objectFrames[objectName]

    def makeVideoForObject(self, objectName, filename, prepend_time=0.5, append_time=0.5, drawBox=True, fpsScale=1.0):
        framesWithObject = self.objectFrames[objectName]

        vid = cv2.VideoCapture(self.videoFile)
        fps = vid.get(cv2.CAP_PROP_FPS)
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(
            'M', 'J', 'P', 'G'), fps * fpsScale, (w, h))

        invid_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        prepend_frames = int(fps * prepend_time)
        append_frames = int(fps * append_time)
        all_starts = [max(0, framesWithObject[0]-prepend_frames)]
        all_ends = [min(invid_frames, framesWithObject[0]+append_frames)]
        for f in framesWithObject[1:]:
            f1 = max(0, f - prepend_frames)
            f2 = min(invid_frames, f+append_frames)
            if f1 < all_ends[-1]:
                all_ends[-1] = f2
            else:
                all_starts.append(f1)
                all_ends.append(f2)

        ret, transitionFrame = vid.read()
        if not ret:
            print("Problem getting first frame from video {}".format(self.videoFile))
            return -1

        for st, en in zip(all_starts, all_ends):
            transitionFrame[:, :, :] = 255
            cv2.putText(transitionFrame, "Frames {} - {}".format(st, en),
                        (0, h // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
            for i in range(int(fps * fpsScale / 2)):
                writer.write(transitionFrame)

            vid.set(cv2.CAP_PROP_POS_FRAMES, st)
            f = st
            while f < en:
                ret, frame = vid.read()
                if not ret:
                    print("Problem getting frame {} from video {}".format(f, self.videoFile))
                    return -1

                if drawBox:
                    fdicts = self.output_arrays[f]
                    boxes = [ob['box_points'] for ob in fdicts if ob['name'] == objectName]
                    for b in boxes:
                        cv2.rectangle(frame, (b[0], b[1]),
                                      (b[2], b[3]), (255, 255, 255), thickness=2)

                writer.write(frame)

                f += 1

        transitionFrame[:, :, :] = 255
        cv2.putText(transitionFrame, "Finished",
                    (0, h // 2), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0))
        for i in range(int(fps * fpsScale / 2)):
            writer.write(transitionFrame)

        vid.release()
        writer.release()

        print("video at {}".format(filename))

    def getObjectList(self):
        return self.objectFrames.keys()

    def saveToFile(self, filename):
        if not self.videoAnalyzed:
            print("Nothing to save")
            return

        with open(filename, 'w') as f:
            f.write(json.dumps(self.objectFrames))
            f.write("\n")
            f.write(json.dumps(self.output_arrays))
            f.write("\n")
            f.write(json.dumps(self.count_arrays))
            f.write("\n")
            f.write(json.dumps(self.average_output_count))
            f.write("\n")
            f.write(self.videoFile)
            f.write("\n")
            return 0

        print("Couldn't open file %s" % filename)
        return -1

    def loadFromFile(self, filename):
        if not os.path.exists(filename):
            print("File not found: {}".format(filename))
            return 1

        with open(filename, 'r') as f:
            line = f.readline()
            self.objectFrames = json.loads(line[:-1])
            line = f.readline()
            self.output_arrays = json.loads(line[:-1])
            line = f.readline()
            self.count_arrays = json.loads(line[:-1])
            line = f.readline()
            self.average_output_count = json.loads(line[:-1])
            line = f.readline()
            self.videoFile = line[:-1]
            return 0

        print("Couldn't open file %s" % filename)
        return -1


if __name__ == "__main__":
    vs = VideoSearcher()

    remake_model = FORCE_REBUILD
    if not remake_model:
        loadret = vs.loadFromFile(saveFile)
        if loadret:
            print("couldn't load object, remaking file")
            remake_model = True

    if remake_model:
        if INVID_ID == 2:
            vs.analyzeVideo(input_video_path, dur=0.5)
        else:
            vs.analyzeVideo(input_video_path)
        vs.saveToFile(saveFile)

    if SAVE_ALL_AND_EXIT:
        allobs = list(vs.getObjectList())
        for ob in allobs:
            print(ob)
            fname = "object_video_{}.avi".format(ob)
            vs.makeVideoForObject(ob, os.path.join(data_dir, fname), fpsScale=1.0)
        session.close()
        exit()
    

    running = True
    while running:
        ob = input("Which object?")
        if ob == "q":
            break
        elif ob == "l":
            print(vs.getObjectList())
        elif ob == "all":
            allobs = list(vs.getObjectList())
            for ob in allobs:
                print(ob)
                fname = "object_video_{}.avi".format(ob)
                vs.makeVideoForObject(ob, os.path.join(data_dir, fname), fpsScale=1.0)
        else:
            try:
                print(vs.searchForObject(ob))
                makevid = input("Make a video for this object (y/N)?")
                if makevid == "y":
                    fname = "object_video_{}.avi".format(ob)
                    vs.makeVideoForObject(ob, os.path.join(data_dir, fname), fpsScale=1.0)
            except KeyError:
                print("Object {} not found".format(ob))

    session.close()
