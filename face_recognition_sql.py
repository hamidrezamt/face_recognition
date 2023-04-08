# Real-time face detection and recognition via OT for multi faces
# Do detection -> recognize face, new face -> not do re-recognition
# Do re-recognition for multi faces will cost much time, OT will be used to instead it

import dlib
import pymysql
import numpy as np
import cv2
import os
import time
import logging
from imutils import face_utils
from scipy.spatial import distance as dist
import head_pose_estimation as hpe
from datetime import datetime

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Create a connection object
database_server_name = "localhost"
database_user = "hamidreza"
database_password = "@123HrmZ123$"
database_name = "dlib_face"
charset = "utf8mb4"
cusror_type = pymysql.cursors.DictCursor

database = pymysql.connect(host=database_server_name, user=database_user, password=database_password, db=database_name, charset=charset,cursorclass=cusror_type)
cursor = database.cursor()


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        # An array used to store all input face features / Save the features of faces in the database
        self.face_features_known_list = []
        # Save the name of faces in the database
        self.face_name_known_list = []

        # List to save centroid positions of ROI in previous frame and current frame 
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in previous frame and current frame 
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        # Counter of the number of faces in the previous frame and the current frame
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the Euclidean distance for comparison during recognition
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # Store the facial features captured by the current frame
        self.current_frame_face_feature_list = []

        # Euclidean distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # Reclassify after 'reclassify_interval' frames
        # If an "unknown" face is recognized, the face will be re-recognized after reclassify_interval_cnt counts to reclassify_interval
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10
        
        ########################################################################################
        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.existing_faces_cnt = 0         # for counting saved faces
        self.similarity_thresh = 0.5
        self.eye_ar_thresh = 0.2
        self.blur_thres = 90.0
        self.y_direction_thres = 15.0
        self.x_direction_thres = 20.0
        # self.recapture_interval_cnt = 0
        # self.recapture_interval = 4
        ########################################################################################

    #  Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    def check_existing_faces_cnt(self):
        if os.listdir(self.path_photos_from_camera):
            # Get the last recorded face number
            
            person_list = [f for f in os.listdir(self.path_photos_from_camera) if not f.startswith('.')]
            person_num_list = []
            for person in person_list:
                person_order = person.split('_')[1].split('_')[0]
                person_num_list.append(int(person_order))
            print("Number of persons in base folder: " + str(max(person_num_list)))
            self.existing_faces_cnt = max(person_num_list)

        else:
            self.existing_faces_cnt = 0

    ########################################################################################################################################################
    ##########  TODO: i have to get the list of persons if csv file existed and only do feature extraction of not new person not the whole list!  ##########
    ########################################################################################################################################################
    def feature_extraction_csv(self):
        # 0. clear table in mysql
        # cursor.execute("TRUNCATE TABLE 'person_features';")
        # cursor.execute("TRUNCATE TABLE `mean_person_features`;")

        # 1. check existing people in mysql
        cursor.execute("SELECT COUNT(*) FROM `mean_person_features`;")
        person_start = int(cursor.fetchall()[0]["COUNT(*)"])
        self.check_existing_faces_cnt()
        logging.debug("self.existing_faces_cnt: " + str(self.existing_faces_cnt))

        for person in range(person_start, self.existing_faces_cnt):
            # Get the mean/average features of face/personX, it will be a list with a length of 128D
            logging.info(self.path_photos_from_camera + "person_" + str(person + 1))
            features_mean_personX = self.return_features_mean_personX(self.path_photos_from_camera + "person_" + str(person + 1) , person + 1)
            # features_mean_personX = self.return_features_mean_personX(self.path_photos_from_camera + "person_" + str(person + 1))

            # 2. Insert person 1 to person X
            cursor.execute("INSERT INTO `mean_person_features` (`person_id`) VALUES(" + str(person+1) + ");")

            # 3. Insert features for person X
            for i in range(128):
                cursor.execute("UPDATE `mean_person_features` SET `mean_feature_" + str(i + 1) + '`=' + str(features_mean_personX[i]) + " WHERE `person_id`=" + str(person + 1) + ";")
                # print("UPDATE `mean_person_features` SET `mean_feature_" + str(i + 1) + '=' + str(features_mean_personX[i]) + " WHERE `person_id`=" + str(person + 1) + ";")
        
        database.commit()
        logging.info("Save all the features of faces registered into databse")


    # Get known faces from "features_all.csv"
    def get_face_database(self):
        self.face_features_known_list = []
        self.face_name_known_list = []
        # 1. get database face numbers
        cmd_rd = "SELECT COUNT(*) FROM `mean_person_features`;"
        cursor.execute(cmd_rd)
        results = cursor.fetchall()
        # print(results[0]["COUNT(*)"])
        person_cnt = int(results[0]["COUNT(*)"])
        if person_cnt:
            # 2. get features for person X
            for person in range(person_cnt):
                # lookup for personX                
                cmd_lookup = "SELECT * FROM `mean_person_features` WHERE `person_id`=" + str(person + 1) + ";"
                cursor.execute(cmd_lookup)
                results = cursor.fetchall()
                results = list(results[0].values())
                features = results[1:]
                results = [float(feature) for feature in features]
                self.face_features_known_list.append(results)
                self.face_name_known_list.append("Person_" + str(person + 1))
                # print(results)
            print("Faces in Database：", len(self.face_name_known_list))
            return 1
        else:
            logging.warning("No Face found!")
            return 0
    

    # Return 128D features for single image
    # Input:    path_img           <class 'str'>
    # Output:   face_descriptor    <class 'dlib.vector'>
    def return_128d_features(self, path_img):
        img_rd = cv2.imread(path_img)
        faces = detector(img_rd, 1)

        logging.info("%-40s %-20s", "Image with faces detected:", path_img)

        # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
        if len(faces) != 0:
            shape = predictor(img_rd, faces[0])
            face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        else:
            face_descriptor = 0
            logging.warning("no face")
        return face_descriptor


    # Return the mean value of 128D face descriptor for person X
    # def return_features_mean_personX(self, path_face_personX):
    def return_features_mean_personX(self, path_face_personX, id):
        features_list_personX = []
        photos_list = [f for f in os.listdir(path_face_personX) if not f.startswith('.')]
        photos_list.sort()
        current_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')

        if photos_list:
            for i in range(len(photos_list)):
                # Get 128D features for single image of personX
                logging.info("%-40s %-20s", "Reading image:", path_face_personX + "/" + photos_list[i])
                features_128d = self.return_128d_features(path_face_personX + "/" + photos_list[i])
                # Jump if no face detected from image
                if features_128d == 0:
                    i += 1
                else:
                    features_list_personX.append(features_128d)
                    # 2. Insert person 1 to person X
                    print("INSERT INTO `person_features` (`person_id`, `date`, `image_path`) VALUES (" + str(id) + " , '" +  str(current_date) + "' , '" +  path_face_personX + "/" + photos_list[i] + "');")
                    cursor.execute("INSERT INTO `person_features` (`person_id`, `date`, `image_path`) VALUES (" + str(id) + " , '" +  str(current_date) + "' , '" +  path_face_personX + "/" + photos_list[i] + "');")

                    # 3. Insert features for person X
                    for i in range(128):
                        cursor.execute("UPDATE `person_features` SET `feature_" + str(i + 1) + '`=' + str(features_128d[i]) + " WHERE `person_id`=" + str(id) + ";")
                        # print("UPDATE `mean_person_features` SET `mean_feature_" + str(i + 1) + '=' + str(features_mean_personX[i]) + " WHERE `person_id`=" + str(person + 1) + ";")
        else:
            logging.warning("Warning: No images in%s/", path_face_personX)

        # Compute the mean
        # personX 128D -> 1 x 128D
        if features_list_personX:
            features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
        else:
            features_mean_personX = np.zeros(128, dtype=object, order='C')
        return features_mean_personX


    def blur_detection(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def eye_aspect_ratio(self, eye):
        # compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        # compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # return the eye aspect ratio
        return ear

    def close_eye_detection(self, shape):
        # grab the indexes of the facial landmarks for the left and right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = face_utils.shape_to_np(shape)
        # extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        return ear
    
    def head_direction(self, image):
        dir = 0
        try:
            x , y = hpe.head_pose_estimation(image)
            logging.debug("Y index is: " + str(y) + "X index is: " + str(x))
            if y < self.y_direction_thres * -1:
                dir = 0
            elif y > self.y_direction_thres:
                dir = 0
            if x < self.x_direction_thres * -1:
                dir = 0
            elif x > self.x_direction_thres:
                dir = 0
            else:
                dir = 1

        except (TypeError):
            logging.debug("Face Mesh detect NO face in frame")
            dir = 0

        return dir



    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # Compute the Euclidean distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # Using Centroid Tracking to Recognize Faces; Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # Add explanatory text on the cv2 window
    def draw_note(self, img_rd):
        # Add some info on windows
        cv2.putText(img_rd, "Face Recognizer with OT", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple([int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]), self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)

    # Face detection and recognition wit OT from input video stream
    def process(self, stream):

        while stream.isOpened():
            self.frame_cnt += 1
            logging.debug("Frame " + str(self.frame_cnt) + " starts")
            flag, img_rd = stream.read()
            kk = cv2.waitKey(1)

            # 2. Detect faces for frame X
            faces = detector(img_rd, 0)

            if self.get_face_database():
                # 3. Update cnt for faces in frames
                self.last_frame_face_cnt = self.current_frame_face_cnt
                self.current_frame_face_cnt = len(faces)

                # 4. Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5. Update the list of centroids for the previous frame and the current frame
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1 If there is no change in the number of faces in the current frame and the previous frame
                if (self.current_frame_face_cnt == self.last_frame_face_cnt) and (self.reclassify_interval_cnt != self.reclassify_interval):
                    logging.debug("scene 1: No face cnt changes in this frame!")

                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        logging.debug("There are unknown faces, start reclassify_interval_cnt counting")
                        self.reclassify_interval_cnt += 1

                    if self.current_frame_face_cnt != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append([int(faces[k].left() + faces[k].right()) / 2, int(faces[k].top() + faces[k].bottom()) / 2])
                            img_rd = cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (255, 255, 255), 2)

                    # If there are multiple faces in the current frame, use centroid tracking
                    if self.current_frame_face_cnt != 1:
                        self.centroid_tracker()
                    
                    logging.debug("current_frame_face_cnt: " + str(self.current_frame_face_cnt))
                    logging.debug("current_frame_face_name_list len : " + str(len(self.current_frame_face_name_list)))
                    # for i in range(self.current_frame_face_cnt):
                    for i in range(len(self.current_frame_face_name_list)):
                        # 6.2 Write names under ROI
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i], self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                    self.draw_note(img_rd)

                # 6.2 If the number of faces in the current frame and the previous frame changes
                else:
                    logging.debug("scene 2: Faces cnt changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_cnt = 0

                    # 6.2.1 If the number of faces Reduced
                    if self.current_frame_face_cnt == 0:
                        logging.debug("  scene 2.1 No faces in this frame!!!")
                        # clear list of names and features
                        self.current_frame_face_name_list = []
                    # 6.2.2 If the number of faces increased
                    else:
                        logging.debug("  scene 2.2 Get faces in this frame and do face recognition")
                        self.current_frame_face_name_list = []
                        for i in range(len(faces)):
                            
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_name_list.append("unknown")

                        # 6.2.2.1 Traverse all faces in the captured image
                        for k in range(len(faces)):
                            logging.debug("  For face %d in current frame:", k + 1)
                            shape = predictor(img_rd, faces[k])
                            self.current_frame_face_centroid_list.append([int(faces[k].left() + faces[k].right()) / 2, int(faces[k].top() + faces[k].bottom()) / 2])
                            self.current_frame_face_X_e_distance_list = []

                            # 6.2.2.2  Positions of faces captured
                            self.current_frame_face_position_list.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 6.2.2.3
                            # For every faces detected, compare the faces in the database
                            for i in range(len(self.face_features_known_list)):
                                
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance( self.current_frame_face_feature_list[k], self.face_features_known_list[i])
                                    logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    # person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # 6.2.2.4 Find the one with minimum e distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < self.similarity_thresh:
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                logging.debug("  Face recognition result: %s", self.face_name_known_list[similar_person_num])
                            else:
                                logging.debug("  Face recognition result: Unknown person")
                                logging.debug("  K: " + str(k))
                                height = (faces[k].bottom() - faces[k].top())
                                width = (faces[k].right() - faces[k].left())
                                hh = int(height/2)
                                ww = int(width/2)

                                # If the size of ROI > 960x1280
                                if (faces[k].right()+ww) > 1280 or (faces[k].bottom()+hh > 960) or (faces[k].left()-ww < 0) or (faces[k].top()-hh < 0):
                                    save_flag = 0
                                    logging.warning("Please adjust your position!")
                                else:
                                    save_flag = 1
                                
                                # Create blank image according to the size of face detected
                                img_blank = np.zeros((int(height*2), width*2, 3), np.uint8)

                                img_tmp = np.zeros((height, width, 3), np.uint8)

                                for ii in range(height):
                                    for jj in range(width):
                                        try:
                                            img_tmp[ii][jj] = img_rd[faces[k].top() + ii][faces[k].left() + jj]
                                        except IndexError:
                                            logging.debug("IndexError: index 720 is out of bounds for axis 0 with size 720!")

                                for ii in range(height*2):
                                    for jj in range(width*2):
                                        try:
                                            img_blank[ii][jj] = img_rd[faces[k].top()-hh + ii][faces[k].left()-ww + jj]
                                        except IndexError:
                                            logging.debug("IndexError: index 720 is out of bounds for axis 0 with size 720!")
                                        


                                blur_index = self.blur_detection(img_tmp)
                                close_eye_index = self.close_eye_detection(shape)
                                head_direction_index = self.head_direction(img_tmp)

                                logging.debug("blur_detection(): " + str(blur_index))
                                logging.debug("close_eye_detection(): " + str(close_eye_index))
                                logging.debug("head_direction(): " + str(head_direction_index))

                                if blur_index > self.blur_thres and close_eye_index > self.eye_ar_thresh and head_direction_index :
                                    self.check_existing_faces_cnt()
                                    current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt + 1)
                                    os.makedirs(current_face_dir)
                                    logging.info("\n%-40s %s", "Create folders:", current_face_dir)
                                    img_name = str(current_face_dir) + "/img_face_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(close_eye_index) + "_" + str(head_direction_index) + ".jpg"
                                    cv2.imwrite(img_name, img_blank)
                                    logging.info("Save into:                    " + img_name)

                                    self.feature_extraction_csv()
                                
                                cv2.imwrite("debug/debug_" + str(self.frame_cnt) + "_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(close_eye_index) + "_" + str(head_direction_index) + ".jpg", img_tmp) # Dump current frame image if needed

                        # 7. Add note on cv2 window
                        self.draw_note(img_rd)
                        # cv2.imwrite("debug/debug_" + str(self.frame_cnt) + ".png", img_rd) # Dump current frame image if needed

            else:
                # Face detected
                logging.debug("there is no face in database")
                if len(faces) != 0:
                    # Create folders to save photos
                    self.pre_work_mkdir()
                    self.last_frame_face_cnt = self.current_frame_face_cnt
                    self.current_frame_face_cnt = len(faces)
                    # Show the ROI of faces
                    for k, d in enumerate(faces):
                        shape = predictor(img_rd, d)

                        # Compute the size of rectangle box
                        height = (d.bottom() - d.top())
                        width = (d.right() - d.left())
                        hh = int(height/2)
                        ww = int(width/2)

                        # If the size of ROI > 960x1280
                        if (d.right()+ww) > 1280 or (d.bottom()+hh > 960) or (d.left()-ww < 0) or (d.top()-hh < 0):
                            cv2.putText(img_rd, "OUT OF RANGE", (20, 300), self.font, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                            color_rectangle = (0, 0, 255)
                            save_flag = 0
                            logging.warning("Please adjust your position!")
                        else:
                            color_rectangle = (255, 255, 255)
                            save_flag = 1
                            
                        cv2.rectangle(img_rd, tuple([d.left() - ww, d.top() - hh]), tuple([d.right() + ww, d.bottom() + hh]), color_rectangle, 2)
                        
                        # Create blank image according to the size of face detected
                        img_blank = np.zeros((height*2, width*2, 3), np.uint8)

                        img_tmp = np.zeros((height, width, 3), np.uint8)

                        for ii in range(height):
                            for jj in range(width):
                                img_tmp[ii][jj] = img_rd[d.top() + ii][d.left() + jj]

                        for ii in range(height*2):
                            for jj in range(width*2):
                                img_blank[ii][jj] = img_rd[d.top()-hh + ii][d.left()-ww + jj]
                        

                        # if (self.current_frame_face_cnt != self.last_frame_face_cnt or (self.recapture_interval_cnt == self.recapture_interval)):
                        #     logging.debug("HOOOOOOOORAAAAAAA!")
                        #     self.recapture_interval_cnt = 0
                        #     blur_index = self.blur_detection(img_tmp)
                        #     close_eye_index = self.close_eye_detection(shape)
                        #     head_direction_index = self.head_direction(img_tmp)

                        #     logging.debug("blur_detection(): " + str(blur_index))
                        #     logging.debug("close_eye_detection(): " + str(close_eye_index))
                        #     logging.debug("head_direction(): " + str(head_direction_index))

                        #     if blur_index > self.blur_thres and close_eye_index > self.eye_ar_thresh and head_direction_index :
                        #         ############
                        #         self.current_frame_face_name_list.append("unknown")
                        #         ############
                        #         self.check_existing_faces_cnt()
                        #         current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt + 1)
                        #         os.makedirs(current_face_dir)
                        #         logging.info("\n%-40s %s", "Create folders:", current_face_dir)
                        #         img_name = str(current_face_dir) + "/img_face_" + "{:.2f}".format(blur_index) + "_" + "{:.2f}".format(close_eye_index) + "_" + "{:.2f}".format(head_direction_index) + ".jpg"
                        #         cv2.imwrite(img_name, img_blank)
                        #         logging.info("Save into:                    " + img_name)

                        #         self.feature_extraction_csv()
                        # else:
                        #     self.recapture_interval_cnt += 1
                        #     logging.debug("recapture_interval_cnt: " + str(self.recapture_interval_cnt))


                        blur_index = self.blur_detection(img_tmp)
                        close_eye_index = self.close_eye_detection(shape)
                        head_direction_index = self.head_direction(img_tmp)

                        logging.debug("blur_detection(): " + str(blur_index))
                        logging.debug("close_eye_detection(): " + str(close_eye_index))
                        logging.debug("head_direction(): " + str(head_direction_index))

                        if blur_index > self.blur_thres and close_eye_index > self.eye_ar_thresh and head_direction_index :
                            ############
                            self.current_frame_face_name_list.append("unknown")
                            ############
                            self.check_existing_faces_cnt()
                            current_face_dir = self.path_photos_from_camera + "person_" + str(self.existing_faces_cnt + 1)
                            os.makedirs(current_face_dir)
                            logging.info("\n%-40s %s", "Create folders:", current_face_dir)
                            img_name = str(current_face_dir) + "/img_face_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(close_eye_index) + "_" + str(head_direction_index) + ".jpg"
                            cv2.imwrite(img_name, img_blank)
                            logging.info("Save into:                    " + img_name)

                            self.feature_extraction_csv()

                        cv2.imwrite("debug/debug_" + str(self.frame_cnt) + "_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(close_eye_index) + "_" + str(head_direction_index) + "_no_database.jpg", img_tmp) # Dump current frame image if needed

                            
            # 8. Press 'q' to exit
            if kk == ord('q'):
                break


            self.update_fps()
            cv2.namedWindow("camera", 1)
            cv2.imshow("camera", img_rd)

            logging.debug("Frame ends\n\n")

    def run(self):
        # cap = cv2.VideoCapture("video.mp4")  # Get video stream from video file
        cap = cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)              # Get video stream from camera
        # self.cap = cv2.VideoCapture(0)
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    # logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()