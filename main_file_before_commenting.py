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
from datetime import datetime
import mediapipe as mp

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Get face landmarks
shape_predictor_path = os.path.join('data', 'data_dlib', 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(shape_predictor_path)

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model_path = os.path.join('data', 'data_dlib', 'dlib_face_recognition_resnet_model_v1.dat')
face_reco_model = dlib.face_recognition_model_v1(face_reco_model_path)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
logging.basicConfig(level=logging.DEBUG)

# Create a connection object
connection_params = { "host": "localhost", "user": "hamidreza", "password": "@123HrmZ123$", "db": "dlib_face", "charset": "utf8mb4", "cursorclass": pymysql.cursors.DictCursor }

database = pymysql.connect(**connection_params)
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

        # count for frame
        self.frame_count = 0

        # An array used to store all storing face features and names of known subjects
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
        self.last_frame_face_count = 0
        self.current_frame_face_count = 0

        # Save the Euclidean distance for comparison during recognition
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # Store the facial features captured by the current frame
        self.current_frame_face_feature_list = []

        # Euclidean distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # Reclassify after 'reclassify_interval' frames
        # If an "unknown" face is recognized, the face will be re-recognized after reclassify_interval_count counts to reclassify_interval
        self.reclassify_interval_count = 0
        self.reclassify_interval = 5

        ########################################################################################
        self.path_photos_from_camera = os.path.join('data', 'data_faces_from_camera')
        self.existing_faces_count = 0  # for counting saved faces
        self.similarity_thresh = 0.5
        self.eye_ar_thresh = 0.21
        self.blur_thresh = 35.0
        self.y_direction_thresh = 18.0  # 15.0
        self.x_direction_thresh = 30.0  # 17.0
        self.dimesion_thresh = 160
        self.once = True
        self.first_faces = 0
        ########################################################################################

    #  Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    def check_existing_faces_count(self):
        if os.listdir(self.path_photos_from_camera):
            # Get the last recorded face number

            person_list = [f for f in os.listdir(self.path_photos_from_camera) if not f.startswith('.')]
            person_num_list = []
            for person in person_list:
                person_order = person.split('_')[1].split('_')[0]
                person_num_list.append(int(person_order))
            logging.debug("  Number of persons in base folder: " + str(max(person_num_list)))
            self.existing_faces_count = max(person_num_list)
        else:
            self.existing_faces_count = 0

    def feature_extraction(self, index):

        # 1. check existing people in mysql
        cursor.execute("SELECT * FROM `mean_person_features`;")
        results = cursor.fetchall()
        person_start = len(results)
        logging.debug("  person_start: " + str(person_start))
        self.check_existing_faces_count()
        logging.debug("  self.existing_faces_count: " + str(self.existing_faces_count))

        # Get the mean/average features of face/personX, it will be a list with a length of 128D
        features_mean_personX, counter_up = self.return_features_mean_personX(os.path.join(self.path_photos_from_camera , "person_" + str(index + 1)), index + 1)

        if person_start and person_start == self.existing_faces_count:
            # 2.1 Update person 1 to person X
            current_counter = list(results[index].values())[1]
            logging.debug("  current counter: " + str(current_counter))
            if counter_up:
                cursor.execute("UPDATE `mean_person_features` SET `counter` = " + str(current_counter + 1) + " WHERE `person_id` =" + str(index + 1) + ";")
                logging.debug("  UPDATE `mean_person_features` SET `counter` = " + str(current_counter + 1) + " WHERE `person_id` =" + str(index + 1) + ";")

        else:
            # 2.2 Insert person 1 to person X
            cursor.execute("INSERT INTO `mean_person_features` (`person_id` , `counter`) VALUES(" + str(index + 1) + " , 1);")
            logging.debug("  INSERT INTO `mean_person_features` (`person_id` , `counter`) VALUES(" + str(index + 1) + " , 1);")

        # 3 Insert features for person X
        for i in range(128):
            cursor.execute("UPDATE `mean_person_features` SET `mean_feature_" + str(i + 1) + '`=' + str(features_mean_personX[i]) + " WHERE `person_id`=" + str(index + 1) + ";")
            logging.debug("  UPDATE `mean_person_features` SET `mean_feature_" + str(i + 1) + '`=' + str(features_mean_personX[i]) + " WHERE `person_id`=" + str(index + 1) + ";")

        database.commit()
        logging.info("  Save all the features of faces registered into databse")

    # Get known faces from "features_all.csv"
    def get_face_database(self):
        self.face_features_known_list = []
        self.face_name_known_list = []
        # 1. get database face numbers
        cursor.execute("SELECT * FROM `mean_person_features`;")
        person_count = len(cursor.fetchall())
        logging.debug("  person_count: " + str(person_count))
        if person_count:
            # 2. get features for person X
            for person in range(person_count):
                # lookup for personX                
                cmd_lookup = "SELECT * FROM `mean_person_features` WHERE `person_id`=" + str(person + 1) + ";"
                logging.debug("  SELECT * FROM `mean_person_features` WHERE `person_id`=" + str(person + 1) + ";")
                cursor.execute(cmd_lookup)
                results = cursor.fetchall()
                results = list(results[0].values())
                features = results[2:]
                results = [float(feature) for feature in features]
                self.face_features_known_list.append(results)
                self.face_name_known_list.append("Person_" + str(person + 1))
                # print(results)
            logging.debug("  face_features_known_list: " + str(self.face_features_known_list))
            logging.debug("  Faces in Database：" + str(len(self.face_name_known_list)))
            return 1
        else:
            logging.warning("  No Face found!")
            return 0

    # Return 128D features for single image
    # Input:    path_img           <class 'str'>
    # Output:   face_descriptor    <class 'dlib.vector'>
    def return_128d_features(self, path_img):
        img_rd = cv2.imread(path_img)
        faces = detector(img_rd, 1)

        # logging.info("%-40s %-20s", "  Image with faces detected:", path_img)

        # For photos of faces saved, we need to make sure that we can detect faces from the cropped images
        if len(faces) != 0:
            shape = predictor(img_rd, faces[0])
            face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        else:
            face_descriptor = 0
            logging.warning("  NO FACE detected by return_128d_features function")
            os.remove(path_img)
        return face_descriptor
    
    def photo_id(self, photo_name):
        return int(photo_name.split("_")[2].split('_')[0])

    # Return the mean value of 128D face descriptor for person X
    def return_features_mean_personX(self, path_face_personX, id):
        features_list_personX = []
        counter_up = False
        photos_list = [f for f in os.listdir(path_face_personX) if not f.startswith('.')]
        photos_list.sort(key = self.photo_id)
        # logging.debug("  len(photos_list): " + str(len(photos_list)))
        # logging.debug("  photos_list: " + str(photos_list))
        cursor.execute("SELECT * FROM `person_features` WHERE `person_id` = " + str(id) + ";")
        results = cursor.fetchall()
        database_photo_count = len(results)
        logging.debug("  database_photo_count: " + str(database_photo_count))
        current_date = datetime.combine(datetime.today().date(), datetime.today().time().replace(microsecond=0))
        if(database_photo_count != 0):
            last_stored_date = list(results[database_photo_count-1].values())[2]
            last_stored_date = datetime.strptime(last_stored_date, '%Y-%m-%d %H:%M:%S')
            time_difference = (current_date - last_stored_date).total_seconds()
            logging.debug("  current_date: " + str(current_date))
            logging.debug("  last_stored_date: " + str(last_stored_date))
            logging.debug("  time_difference: " + str(time_difference))
            if time_difference > 300:
                counter_up = True
            else:
                counter_up = False

        if photos_list:
            # for i in range(len(photos_list)):
            for i in range(database_photo_count, len(photos_list)):
                # Get 128D features for single image of personX
                logging.info("%-40s %-20s", "  Reading image:", os.path.join(path_face_personX , photos_list[i]))
                features_128d = self.return_128d_features(os.path.join(path_face_personX , photos_list[i]))
                # Jump if no face detected from image
                if features_128d == 0:
                    i += 1
                else:
                    # 2. Insert person 1 to person X
                    cursor.execute("INSERT INTO `person_features` (`person_id`, `date`, `image_path`) VALUES (" + str(id) + " , '" + str(current_date.strftime('%Y-%m-%d %H:%M:%S')) + "' , '" + os.path.join(os.path.abspath(path_face_personX) , photos_list[i]) + "');")
                    # logging.debug("  INSERT INTO `person_features` (`person_id`, `date`, `image_path`) VALUES (" + str(id) + " , '" + str(current_date.strftime('%Y-%m-%d %H:%M:%S')) + "' , '" + os.path.abspath(path_face_personX) + "/" + photos_list[i] + "');")

                    # 3. Insert features for person X
                    for j in range(128):
                        cursor.execute("UPDATE `person_features` SET `feature_" + str(j + 1) + '`=' + str(features_128d[j]) + " WHERE `image_path`='" + os.path.join(os.path.abspath(path_face_personX) , photos_list[i]) + "' AND `person_id` = " + str(id) + " ;")
                        # logging.debug("  UPDATE `person_features` SET `feature_" + str(j + 1) + '`=' + str(features_128d[j]) + " WHERE `image_path`='" + os.path.abspath(path_face_personX) + "/" + photos_list[i] + "' AND `person_id` = " + str(id) + " ;")
        else:
            logging.warning("  Warning: No images in%s/", path_face_personX)
        
        # cursor.callproc('selection')
        # # selection stored procedure:
        #     # CREATE DEFINER=`root`@`localhost` PROCEDURE `selection`()
        #     # BEGIN
        #     # CREATE TEMPORARY TABLE IF NOT EXISTS temp AS (SELECT * FROM `person_features`);
        #     # ALTER TABLE temp DROP COLUMN `id`,  DROP COLUMN `person_id`, DROP COLUMN `image_path`, DROP COLUMN `date`;
        #     # SELECT * FROM temp;
        #     # DROP TABLE temp;
        #     # END
        
        cursor.execute("SELECT * FROM `person_features` WHERE `person_id` = " + str(id) + ";")
        feature_list_temp = cursor.fetchall()
        for i in range(len(feature_list_temp)):
            features = list(feature_list_temp[i].values())[4:]
            features = [float(feature) for feature in features]
            features_list_personX.append(features)

        # Compute the mean
        # personX 128D -> 1 x 128D
        if features_list_personX:
            features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
        else:
            features_mean_personX = np.zeros(128, dtype=object, order='C')
        return features_mean_personX, counter_up

    def blur_detection(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        ced = (leftEAR + rightEAR) / 2.0
        return ced

    def head_pose_estimation(self, image):
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the result
        results = face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            logging.debug("face also detected by face mesh")
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                
            return (x, y)


    def head_direction(self, image):
        try:
            x, y = self.head_pose_estimation(image)
            logging.debug("  Y index is: " + str(y) + " | X index is: " + str(x))
            if y < self.y_direction_thresh * -1 or y > self.y_direction_thresh or x < self.x_direction_thresh * -1 or x > self.x_direction_thresh:
                dir = 0
            else:
                dir = 1
        except (TypeError):
            logging.debug("  Face Mesh detect NO face in frame")
            dir = 0
        return dir
    
    def image_dimension(self, image):
        width, height, channel = image.shape
        logging.debug("  image's width: " + str(width) + " and image height: " + str(height))
        if width >= self.dimesion_thresh and height >= self.dimesion_thresh:
            return 1
        else:
            return 0
    

    def face_capturer(self, image, face, phase = None, index = None):
        if phase is not None and phase == 0 and self.first_faces > 0:  # if there is no face in database phase is 0
            self.current_frame_face_name_list.append("unknown")
            logging.debug("  self.first_faces: " + str(self.first_faces))
            self.first_faces = self.first_faces - 1

        shape = predictor(image, face)
        height = (face.bottom() - face.top())
        width = (face.right() - face.left())
        hh = int(height / 2)
        ww = int(width / 2)

        # Create blank image according to the size of face detected
        img_blank = np.zeros((height * 2, width * 2, 3), np.uint8)
        img_tmp = np.zeros((height, width, 3), np.uint8)

        for ii in range(height):
            for jj in range(width):
                try:
                    img_tmp[ii][jj] = image[face.top() + ii][face.left() + jj]
                except IndexError:
                    logging.debug("  IndexError: index 720 is out of bounds for axis 0 with size 720!")

        for ii in range(height * 2):
            for jj in range(width * 2):
                try:
                    img_blank[ii][jj] = image[face.top() - hh + ii][face.left() - ww + jj]
                except IndexError:
                    logging.debug("  IndexError: index 720 is out of bounds for axis 0 with size 720!")

        blur_index = self.blur_detection(img_tmp)
        close_eye_index = self.close_eye_detection(shape)
        head_direction_index = self.head_direction(img_blank)
        dimension_index = self.image_dimension(img_tmp)

        logging.debug("  blur_detection(): " + str(blur_index))
        logging.debug("  close_eye_detection(): " + str(close_eye_index))
        logging.debug("  head_direction(): " + str(head_direction_index))
        logging.debug("  image_dimension(): " + str(dimension_index))

        if blur_index > self.blur_thresh and close_eye_index > self.eye_ar_thresh and head_direction_index and dimension_index:
            debug_text = ""

            if index is None:
                self.check_existing_faces_count()
                current_face_dir = os.path.join(self.path_photos_from_camera , "person_" + str(self.existing_faces_count + 1))
                os.makedirs(current_face_dir)
                logging.info("\n%-40s %s", "Create folders:", current_face_dir)
                index = self.existing_faces_count
                debug_text = "  Novel frame accepted and captured!"
            else:
                current_face_dir = os.path.join(self.path_photos_from_camera , "person_" + str(index + 1))
                debug_text = "  frame accepted and captured!"

            img_name = os.path.join(str(current_face_dir) , "img_face_" + str(self.frame_count) + "_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(close_eye_index) + "_" + str(head_direction_index) + ".jpg")
            cv2.imwrite(img_name, img_blank)
            logging.info("  Save into:                    " + img_name)

            self.feature_extraction(index)
            logging.debug(debug_text)

        filename = "debug/debug_" + str(self.frame_count) + "_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(close_eye_index) + "_" + str(head_direction_index) + ".jpg"
        filename = filename.replace('/', os.sep).replace('\\', os.sep)
        cv2.imwrite(filename, img_tmp)  # Dump current frame image if needed
        # cv2.imwrite("debug/debug_" + str(self.frame_count) + "_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(close_eye_index) + "_" + str(head_direction_index) + ".jpg", img_tmp)  # Dump current frame image if needed

    def db_initialize(self):
        db_info = {
            "host": "localhost",
            "user": "hamidreza",
            "password": "@123HrmZ123$",
            "db": "dlib_face",
            "charset": "utf8mb4"
        }

        # create a connection to the MySQL server
        conn_init = pymysql.connect(
                host=db_info["host"],
                user=db_info["user"],
                password=db_info["password"],
                charset=db_info["charset"]
            )

        try:
            # create a cursor object
            cursor = conn_init.cursor()

            # execute a query to get a list of all databases
            cursor.execute("SELECT schema_name FROM information_schema.SCHEMATA")

            # fetch all rows
            rows = cursor.fetchall()

            # check if the "dlib_face" database exists
            db_exists = False
            for row in rows:
                if row[0] == "dlib_face":
                    db_exists = True
                    break

            if db_exists:
                logging.debug("The database 'dlib_face' exists.")
            else:
                logging.debug("The database 'dlib_face' does not exist.")

        finally:
            # close the connection
            conn_init.close()
            
        if not db_exists:

            # create a connection to the MySQL server
            conn = pymysql.connect(
                host=db_info["host"],
                user=db_info["user"],
                password=db_info["password"],
                charset=db_info["charset"]
            )
            try:
                # create a cursor object
                cursor = conn.cursor()

                # execute a query to create the database if it doesn't exist
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_info['db']}")
                cursor.execute(f"USE {db_info['db']}")


                start_part = "CREATE TABLE `person_features` (`id` INT NOT NULL AUTO_INCREMENT, `person_id` INT NOT NULL, `date` VARCHAR(45) NOT NULL, `image_path` VARCHAR(2048) NOT NULL"
                middle_part = ""
                for i in range(128):
                    middle_part = middle_part + ", `feature_" + str(i+1) + "` VARCHAR(100) NULL"
                end_part = ", PRIMARY KEY (`id`));"

                mean_start_part = "CREATE TABLE `mean_person_features` (`person_id` INT NOT NULL, `counter` INT NOT NULL"
                mean_middle_part = ""
                for i in range(128):
                    mean_middle_part = mean_middle_part + ", `mean_feature_" + str(i+1) + "` VARCHAR(100) NULL"
                mean_end_part = ", PRIMARY KEY (`person_id`), UNIQUE INDEX `person_id_UNIQUE` (`person_id` ASC) VISIBLE);"

                person_sql_query = start_part + middle_part + end_part
                cursor.execute(person_sql_query)
                mean_person_sql_query = mean_start_part + mean_middle_part + mean_end_part
                cursor.execute(mean_person_sql_query)

                # commit the changes
                conn.commit()

                logging.debug(f"The tables have been created in the database {db_info['db']}.")

            finally:
                # close the connection
                conn.close()

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
        logging.debug("  inside centroid_tracker!")
        logging.debug("  self.last_frame_face_name_list: " + str(self.last_frame_face_name_list))
        logging.debug("  self.current_frame_face_name_list: " + str(self.current_frame_face_name_list))
        for i in range(len(self.current_frame_face_centroid_list)):
            logging.debug("  inside for inside centroid_tracker!")
            logging.debug("  self.current_frame_face_centroid_list: " + str(self.current_frame_face_centroid_list))
            logging.debug("  self.last_frame_face_centroid_list: " + str(self.last_frame_face_centroid_list))
            e_distance_current_frame_person_x_list = []
            # For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(self.last_current_frame_centroid_e_distance)
                # logging.debug("  e_distance_current_frame_person_x_list: " + str(e_distance_current_frame_person_x_list))

            last_frame_num = e_distance_current_frame_person_x_list.index(min(e_distance_current_frame_person_x_list))
            logging.debug("  last_frame_num: " + str(last_frame_num))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # Add explanatory text on the cv2 window
    def draw_note(self, img_rd):
        # Add some info on windows
        cv2.putText(img_rd, "Face Recognizer with OT", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_count), (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_count), (20, 160), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple([int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]), self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)

    # Face detection and recognition wit OT from input video stream
    def process(self, stream):
        self.db_initialize()
        while stream.isOpened():
            self.frame_count += 1
            logging.debug("  Frame " + str(self.frame_count) + " starts")
            flag, img_rd = stream.read()
            kk = cv2.waitKey(1)

            # 2. Detect faces for frame X
            faces = detector(img_rd, 0)

            if self.get_face_database():
                # 3. Update count for faces in frames
                self.last_frame_face_count = self.current_frame_face_count
                self.current_frame_face_count = len(faces)

                # 4. Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # 5. Update the list of centroids for the previous frame and the current frame
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1 If there is no change in the number of faces in the current frame and the previous frame
                if (self.current_frame_face_count == self.last_frame_face_count) and (self.reclassify_interval_count != self.reclassify_interval):
                    logging.debug("  scene 1: No face count changes in this frame!")

                    self.current_frame_face_position_list = []

                    if "unknown" in self.current_frame_face_name_list:
                        logging.debug("  There are unknown faces, start reclassify_interval_count counting")
                        self.reclassify_interval_count += 1

                    if self.current_frame_face_count != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append([int(faces[k].left() + faces[k].right()) / 2, int(faces[k].top() + faces[k].bottom()) / 2])
                            img_rd = cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (255, 255, 255), 2)

                    # If there are multiple faces in the current frame, use centroid tracking
                    if self.current_frame_face_count != 1:
                        self.centroid_tracker()

                    logging.debug("  current_frame_face_count: " + str(self.current_frame_face_count))
                    logging.debug("  current_frame_face_name_list len : " + str(len(self.current_frame_face_name_list)))
                    # for i in range(self.current_frame_face_count):
                    for i in range(len(self.current_frame_face_name_list)):
                        # 6.2 Write names under ROI
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i], self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                    self.draw_note(img_rd)

                # 6.2 If the number of faces in the current frame and the previous frame changes
                else:
                    logging.debug("  scene 2: Faces count changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_count = 0

                    # 6.2.1 If the number of faces Reduced
                    if self.current_frame_face_count == 0:
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
                                    e_distance_tmp = self.return_euclidean_distance(self.current_frame_face_feature_list[k], self.face_features_known_list[i])
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
                                self.face_capturer(img_rd, faces[k], index = similar_person_num)
                            else:
                                logging.debug("  Face recognition result: Unknown person")
                                logging.debug("  K: " + str(k))
                                self.face_capturer(img_rd, faces[k], phase = 1)

                        # 7. Add note on cv2 window
                        self.draw_note(img_rd)

            else:
                # Face detected
                logging.debug("  there is no face in database")
                #  Update the list of centroids for the previous frame and the current frame
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []
                if len(faces) != 0:
                    # Create folders to save photos
                    self.pre_work_mkdir()
                    self.last_frame_face_count = self.current_frame_face_count
                    self.current_frame_face_count = len(faces)
                    if self.current_frame_face_count != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple([d.left(), int(d.bottom() + (d.bottom() - d.top()) / 4)]))
                            self.current_frame_face_centroid_list.append([int(d.left() + d.right()) / 2, int(d.top() + d.bottom()) / 2])

                    # Show the ROI of faces
                    for k, d in enumerate(faces):
                        if self.once:
                            self.first_faces = len(faces)
                            self.once = False
                        self.face_capturer(img_rd, d, phase = 0)

            # 8. Press 'q' to exit
            if kk == ord('q'):
                break

            self.update_fps()
            cv2.namedWindow("camera", 1)
            cv2.imshow("camera", img_rd)

            logging.debug("  Frame ends\n\n")

    def run(self):
        cap = cv2.VideoCapture(os.path.join('data' , 'test.mp4'))  # Get video stream from video file
        # cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Get video stream from camera im mac
        # cap = cv2.VideoCapture(0)        # Get video stream from camera im windows
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    logging.basicConfig(filename="log.txt", filemode="w", force=True)
    logger = logging.getLogger()  # Let us Create an object
    logger.setLevel(logging.DEBUG)
    # logging.basicConfig(level=logging.INFO)
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()