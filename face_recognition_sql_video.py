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
import shutil
import concurrent.futures

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Get face landmarks
shape_predictor_path = os.path.join('data', 'data_dlib', 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(shape_predictor_path)
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model_path = os.path.join('data', 'data_dlib', 'dlib_face_recognition_resnet_model_v1.dat')
face_reco_model = dlib.face_recognition_model_v1(face_reco_model_path)
# face_reco_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
logging.basicConfig(level=logging.DEBUG)

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
        self.known_face_features_list = []
        # Save the id of faces in the database
        self.known_face_id_list = []

        # List to save centroid positions of ROI in previous frame and current frame 
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save ids of faces in previous frame and current frame 
        self.last_frame_face_id_list = []
        self.current_frame_face_id_list = []

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
        self.reclassify_interval = 10

        ########################################################################################
        self.path_photos_from_camera = os.path.join('data', 'data_faces_from_camera')
        self.path_debug_photos_from_camera = 'debug/'
        self.existing_faces_count = 0  # for counting saved faces
        self.similarity_thresh = 0.44
        self.eye_ar_thresh = 0.22
        self.blur_thresh = 15.0
        self.horz_direction_thresh = 10.0
        self.vert_direction_thresh = 25.0 
        self.dimesion_thresh = 120
        self.once = True
        self.first_faces = 0
        self.current_frame_face_display_message = []
        ########################################################################################

    #  Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)

    def clear_debug_dir(self):
        if os.path.isdir(self.path_debug_photos_from_camera):
            shutil.rmtree(self.path_debug_photos_from_camera)
            os.mkdir(self.path_debug_photos_from_camera)
        else:
            os.mkdir(self.path_debug_photos_from_camera)
    
    def db_conn(self):
        # Create a connection object
        # connection_params = { "host": "localhost", "user": "root", "password": "user", "db": "dlib_face", "charset": "utf8mb4", "cursorclass": pymysql.cursors.DictCursor }
        connection_params = { "host": "localhost", "user": "hamidreza", "password": "@123HrmZ123$", "db": "dlib_face", "charset": "utf8mb4", "cursorclass": pymysql.cursors.DictCursor }
        database = pymysql.connect(**connection_params)
        cursor = database.cursor()
        return cursor,database

    def check_existing_faces_count(self):
        if os.listdir(self.path_photos_from_camera):
            # Get the last recorded face number

            person_list = [f for f in os.listdir(self.path_photos_from_camera) if not f.startswith('.')]
            person_num_list = []
            for person in person_list:
                person_order = person.split('_')[1].split('_')[0]
                person_num_list.append(int(person_order))
            logging.info("  Number of persons in base folder: " + str(max(person_num_list)))
            self.existing_faces_count = max(person_num_list)
        else:
            self.existing_faces_count = 0

    def feature_extraction(self, index):

        # 1. check existing people in mysql
        cursor, database = self.db_conn()
        cursor.execute("SELECT * FROM `mean_person_features`;")
        results = cursor.fetchall()
        person_start = len(results)
        logging.info("  person_start: " + str(person_start))
        self.check_existing_faces_count()
        logging.debug("  self.existing_faces_count: " + str(self.existing_faces_count))

        # Get the mean/average features of face/personX, it will be a list with a length of 128D
        features_mean_personX, counter_up = self.return_features_mean_personX(os.path.join(self.path_photos_from_camera , "person_" + str(index + 1)), index + 1)

        if person_start and person_start == self.existing_faces_count:
            # 2.1 Update person 1 to person X
            current_counter = list(results[index].values())[1]
            logging.info("  current counter: " + str(current_counter))
            if counter_up:
                cursor.execute("UPDATE `mean_person_features` SET `counter` = " + str(current_counter + 1) + " WHERE `person_id` =" + str(index + 1) + ";")
                # logging.debug("  UPDATE `mean_person_features` SET `counter` = " + str(current_counter + 1) + " WHERE `person_id` =" + str(index + 1) + ";")

        else:
            # 2.2 Insert person 1 to person X
            cursor.execute("INSERT INTO `mean_person_features` (`person_id` , `counter`) VALUES(" + str(index + 1) + " , 1);")
            # logging.debug("  INSERT INTO `mean_person_features` (`person_id` , `counter`) VALUES(" + str(index + 1) + " , 1);")

        # 3 Insert features for person X
        for i in range(128):
            cursor.execute("UPDATE `mean_person_features` SET `mean_feature_" + str(i + 1) + '`=' + str(features_mean_personX[i]) + " WHERE `person_id`=" + str(index + 1) + ";")
            # logging.debug("  UPDATE `mean_person_features` SET `mean_feature_" + str(i + 1) + '`=' + str(features_mean_personX[i]) + " WHERE `person_id`=" + str(index + 1) + ";")

        database.commit()
        logging.info("  Stored all the features of faces registered into databse")

    # Get known faces from "features_all.csv"
    def get_face_database(self):
        self.known_face_features_list = []
        self.known_face_id_list = []
        # 1. get database face numbers
        cursor, database = self.db_conn()
        cursor.execute("SELECT * FROM `mean_person_features`;")
        person_count = len(cursor.fetchall())
        logging.info("  person_count: " + str(person_count))
        if person_count:
            # 2. get features for person X
            for person in range(person_count):
                # lookup for personX                
                cmd_lookup = "SELECT * FROM `mean_person_features` WHERE `person_id`=" + str(person + 1) + ";"
                # logging.debug("  SELECT * FROM `mean_person_features` WHERE `person_id`=" + str(person + 1) + ";")
                cursor.execute(cmd_lookup)
                results = cursor.fetchall()
                results = list(results[0].values())
                features = results[2:]
                results = [float(feature) for feature in features]
                self.known_face_features_list.append(results)
                self.known_face_id_list.append("Person_" + str(person + 1))
                # print(results)
            # logging.debug("  known_face_features_list: " + str(self.known_face_features_list))
            logging.info("  Faces in Database：" + str(len(self.known_face_id_list)))
            return 1
        else:
            logging.warning("  No Face found in directory!")
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
        cursor, database = self.db_conn()
        cursor.execute("SELECT * FROM `person_features` WHERE `person_id` = " + str(id) + ";")
        results = cursor.fetchall()
        database_photo_count = len(results)
        logging.info("  database_photo_count: " + str(database_photo_count))
        current_date = datetime.combine(datetime.today().date(), datetime.today().time().replace(microsecond=0))
        if(database_photo_count != 0):
            last_stored_date = list(results[database_photo_count-1].values())[2]
            last_stored_date = datetime.strptime(last_stored_date, '%Y-%m-%d %H:%M:%S')
            time_difference = (current_date - last_stored_date).total_seconds()
            logging.info("  current_date: " + str(current_date))
            logging.info("  last_stored_date: " + str(last_stored_date))
            logging.info("  time_difference: " + str(time_difference))
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
                
            database.commit()
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

    def detect_blur_fft(self, image, size=30):
        # image = imutil.arr2img(numpy_array)
        # image = resize(image, width=500)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # grab the dimensions of the image and use the dimensions to derive the center (x, y)-coordinates
        (h, w) = image.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
        
        # compute the FFT to find the frequency transform, then shift the zero frequency component (i.e., DC component located at the top-left corner) to the center where it will be more easy to analyze
        fft = np.fft.fft2(image)
        fftShift = np.fft.fftshift(fft)

        # zero-out the center of the FFT shift (i.e., remove low frequencies), apply the inverse shift such that the DC component once again becomes the top-left, and then apply the inverse FFT
        fftShift[cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)

        # compute the magnitude spectrum of the reconstructed image, then compute the mean of the magnitude values
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        
        # the image will be considered "blurry" if the mean value of the magnitudes is less than the threshold value
        return mean

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
            logging.info(" face also detected by face mesh")
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
        x = y = 99999
        dir_x = dir_y = 0
        try:
            x, y = self.head_pose_estimation(image)

            if abs(x) > self.vert_direction_thresh:
                dir_x = 0
            else:
                dir_x = 1

            if abs(y) > self.horz_direction_thresh:
                dir_y = 0
            else:
                dir_y = 1
    
        except (TypeError):
            logging.info("\tFace Mesh detect NO face in frame")
            dir_x = dir_y = 0
        return dir_x, dir_y, x, y
    
    def image_dimension(self, image):
        width, height, channel = image.shape
        if width >= self.dimesion_thresh and height >= self.dimesion_thresh:
            out =  1
        else:
            out = 0
        return out, width, height
    


    def face_capturer(self, image, face, current_frame_face_loop_id, phase=None, index=None):
        # Initialize log_messages list to collect log messages
        log_messages = []

        if phase is not None and phase == 0 and self.first_faces > 0:
            self.current_frame_face_id_list.append("unknown")
            log_messages.append("  self.first_faces: " + str(self.first_faces))
            self.first_faces = self.first_faces - 1

        shape = predictor(image, face)
        height = (face.bottom() - face.top())
        width = (face.right() - face.left())

        hh = int(height / 2)
        ww = int(width / 2)

        # Calculate once and store values
        img_top = face.top()
        img_left = face.left()

        # Avoid using for loops for copying regions of interest
        img_tmp = image[img_top:img_top + height, img_left:img_left + width]
        img_blank = image[img_top - hh:img_top - hh + height * 2, img_left - ww:img_left - ww + width * 2]

        # Define functions to execute in parallel
        def parallel_detect_blur_fft(img_tmp):
            return self.detect_blur_fft(img_tmp)

        def parallel_close_eye_detection(shape):
            return self.close_eye_detection(shape)

        def parallel_head_direction(img_blank):
            return self.head_direction(img_blank)

        def parallel_image_dimension(img_tmp):
            return self.image_dimension(img_tmp)

        # Create a ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit function calls for parallel execution
            blur_future = executor.submit(parallel_detect_blur_fft, img_tmp)
            close_eye_future = executor.submit(parallel_close_eye_detection, shape)
            head_direction_future = executor.submit(parallel_head_direction, img_blank)
            image_dimension_future = executor.submit(parallel_image_dimension, img_tmp)

            # Wait for all futures to complete
            concurrent.futures.wait([blur_future, close_eye_future, head_direction_future, image_dimension_future])

            # Retrieve results
            blur_index = blur_future.result()
            close_eye_index = close_eye_future.result()
            (dir_x, dir_y, x, y) = head_direction_future.result()
            (dimension_index, width, height) = image_dimension_future.result()

        if not blur_index > self.blur_thresh:
            self.current_frame_face_display_message[current_frame_face_loop_id] += "Face's image is blurry\n"

        if not close_eye_index > self.eye_ar_thresh:
            self.current_frame_face_display_message[current_frame_face_loop_id] += "Eye's are not open enough\n"

        if not dimension_index:
            self.current_frame_face_display_message[current_frame_face_loop_id] += "Come closer\n"

        if not dir_x:
            self.current_frame_face_display_message[current_frame_face_loop_id] += "Maintain direct eye contact (X)\n"

        if not dir_y:
            self.current_frame_face_display_message[current_frame_face_loop_id] += "Maintain direct eye contact (Y)\n"

        if not (dir_x and dir_y):
            log_messages.append("\thead direction: NOK")

        if blur_index > self.blur_thresh and close_eye_index > self.eye_ar_thresh and dir_x and dir_y and dimension_index:
            self.current_frame_face_display_message[current_frame_face_loop_id] = "OK"
            debug_text = ""

            if index is None:
                self.check_existing_faces_count()
                current_face_dir = os.path.join(self.path_photos_from_camera, "person_" + str(self.existing_faces_count + 1))
                os.makedirs(current_face_dir)
                log_messages.append("\n%-40s %s" % ("Create folders:", current_face_dir))
                index = self.existing_faces_count
                debug_text = "  Novel frame accepted and captured!"
            else:
                current_face_dir = os.path.join(self.path_photos_from_camera, "person_" + str(index + 1))
                debug_text = "  frame accepted and captured!"

            img_name = os.path.join(str(current_face_dir),
                                    "img_face_" + str(self.frame_count) + "_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(
                                        close_eye_index) + "_" + str(dir_x) + "_" + str(dir_y) + ".jpg")
            cv2.imwrite(img_name, img_blank)
            log_messages.append("  Save into:                    " + img_name)

            self.feature_extraction(index)
            log_messages.append(debug_text)

        filename = "debug/debug_" + str(self.frame_count) + "_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(
            close_eye_index) + "_" + str(dir_x) + "_" + str(dir_y) + ".jpg"
        filename = filename.replace('/', os.sep).replace('\\', os.sep)
        cv2.imwrite(filename, img_tmp)

        # Combine log messages into a single string and print it
        logging.info("\n".join(log_messages))


    def db_initialize(self):
        # db_info = {"host": "localhost", "user": "root", "password": "user", "db": "dlib_face", "charset": "utf8mb4"}
        db_info = {"host": "localhost", "user": "hamidreza", "password": "@123HrmZ123$", "db": "dlib_face", "charset": "utf8mb4"}

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
                logging.info("The database 'dlib_face' exists.")
            else:
                logging.info("The database 'dlib_face' does not exist.")

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

                logging.info(f"The tables have been created in the database {db_info['db']}.")

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
        logging.debug("  self.last_frame_face_id_list: " + str(self.last_frame_face_id_list))
        logging.debug("  self.current_frame_face_id_list: " + str(self.current_frame_face_id_list))
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
            self.current_frame_face_id_list[i] = self.last_frame_face_id_list[last_frame_num]

    # Add explanatory text on the cv2 window
    def draw_note(self, img_rd):
        # Add some info on windows
        height, width, _ = img_rd.shape
        relative_x = relative_y = 0.05
        font_scale_percent = 0.08  # Adjust this percentage as needed
        font_scale = (width * font_scale_percent) / 100
        logging.info("\tfont_scale: " + str(font_scale))


        text_list = [
            ("Face Recognizer", (int(relative_x * width), int(relative_y * height * 2)), (255, 255, 255)),
            ("Frame:  " + str(self.frame_count), (int(relative_x * width), int(relative_y * height * 4)), (0, 255, 0)),
            ("FPS:    " + str(self.fps.__round__(2)), (int(relative_x * width), int(relative_y * height * 5)), (0, 255, 0)),
            ("Faces:  " + str(self.current_frame_face_count), (int(relative_x * width), int(relative_y * height * 6)), (0, 255, 0)),
            ("Q: Quit", (int(relative_x * width), int(relative_y * height * 18)), (255, 255, 255))
        ]

        for i, (text, position, color) in enumerate(text_list):
            cv2.putText(img_rd, text, position, self.font, font_scale, color, 1, cv2.LINE_AA)

    def draw_status(self, img_rd):
         # Add description about face capturing parameters
        height, width, _ = img_rd.shape
        relative_y = 0.05
        font_scale_percent = 0.08  # Adjust this percentage as needed
        font_scale = (width * font_scale_percent) / 100
        for i in range(len(self.current_frame_face_id_list)):
            for j, line in enumerate(self.current_frame_face_display_message[i].split('\n')):
                y = int(self.current_frame_face_centroid_list[i][1] + j * relative_y * height)  # Adjust line spacing as needed
                cv2.putText(img_rd, line, (int(self.current_frame_face_centroid_list[i][0]), y), self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)
            cv2.putText(img_rd, self.current_frame_face_id_list[i], self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
        
        self.current_frame_face_display_message = []


    # Face detection and recognition wit OT from input video stream
    def process(self, stream):
        self.clear_debug_dir()
        self.db_initialize()
        while stream.isOpened():
            self.frame_count += 1
            logging.info("  Frame " + str(self.frame_count) + " starts")
            flag, img_rd = stream.read()
            img_rd = cv2.resize(img_rd, None, fx=1.4, fy=1.4)
            kk = cv2.waitKey(1)

            # 2. Detect faces for frame X
            faces = detector(img_rd, 0)

            if self.get_face_database():
                # 3. Update count for faces in frames
                self.last_frame_face_count = self.current_frame_face_count
                self.current_frame_face_count = len(faces)

                # 4. Update the face name list in last frame
                self.last_frame_face_id_list = self.current_frame_face_id_list[:]

                # 5. Update the list of centroids for the previous frame and the current frame
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # 6.1 If there is no change in the number of faces in the current frame and the previous frame
                if (self.current_frame_face_count == self.last_frame_face_count) and (self.reclassify_interval_count != self.reclassify_interval):
                    logging.info("  scene 1: No face count changes in this frame!")
                    logging.info("  reclassify_interval_count: " + str(self.reclassify_interval_count))

                    self.current_frame_face_position_list = []

                    # if "unknown" in self.current_frame_face_id_list:
                    #     logging.info("  There are unknown faces, start reclassify_interval_count counting")
                    #     self.reclassify_interval_count += 1

                    logging.info("  Start reclassify_interval_count counting")
                    self.reclassify_interval_count += 1

                    if self.current_frame_face_count != 0:
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            self.current_frame_face_centroid_list.append([int(faces[k].left() + faces[k].right()) / 2, int(faces[k].top() + faces[k].bottom()) / 2])
                            img_rd = cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (255, 255, 255), 2)
                            self.current_frame_face_display_message.append("")

                    # If there are multiple faces in the current frame, use centroid tracking
                    if self.current_frame_face_count != 1:
                        self.centroid_tracker()

                    logging.info("  current_frame_face_count: " + str(self.current_frame_face_count))
                    logging.info("  current_frame_face_id_list len : " + str(len(self.current_frame_face_id_list)))
                    
                    # for i in range(self.current_frame_face_count):
                    # for i in range(len(self.current_frame_face_id_list)):
                    #     # 6.2 Write names under ROI
                    #     cv2.putText(img_rd, self.current_frame_face_id_list[i], self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)

                    self.draw_note(img_rd)
                    self.draw_status(img_rd)

                # 6.2 If the number of faces in the current frame and the previous frame changes
                else:
                    logging.info("  scene 2: Faces count changes in this frame")
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_count = 0

                    # 6.2.1 If the number of faces Reduced
                    if self.current_frame_face_count == 0:
                        logging.info("  scene 2.1 No faces in this frame!")
                        # clear list of names and features
                        self.current_frame_face_id_list = []
                    # 6.2.2 If the number of faces increased
                    else:
                        logging.info("  scene 2.2 Get faces in this frame and do face recognition")
                        self.current_frame_face_id_list = []
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.current_frame_face_feature_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))
                            self.current_frame_face_id_list.append("unknown")

                        # 6.2.2.1 Traverse all faces in the captured image
                        for k in range(len(faces)):
                            logging.info("  For face %d in current frame:", k + 1)
                            shape = predictor(img_rd, faces[k])
                            self.current_frame_face_centroid_list.append([int(faces[k].left() + faces[k].right()) / 2, int(faces[k].top() + faces[k].bottom()) / 2])
                            self.current_frame_face_X_e_distance_list = []
                            self.current_frame_face_display_message.append("")

                            # 6.2.2.2  Positions of faces captured
                            self.current_frame_face_position_list.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 6.2.2.3
                            # For every faces detected, compare the faces in the database
                            for i in range(len(self.known_face_features_list)):

                                if str(self.known_face_features_list[i][0]) != '0.0':
                                    e_distance_tmp = self.return_euclidean_distance(self.current_frame_face_feature_list[k], self.known_face_features_list[i])
                                    logging.info("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    # person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # 6.2.2.4 Find the one with minimum e distance
                            similar_person_id = self.current_frame_face_X_e_distance_list.index(min(self.current_frame_face_X_e_distance_list))

                            if min(self.current_frame_face_X_e_distance_list) < self.similarity_thresh:
                                self.current_frame_face_id_list[k] = self.known_face_id_list[similar_person_id]
                                logging.info("  Face recognition result: %s", self.known_face_id_list[similar_person_id])
                                self.face_capturer(img_rd, faces[k], k, index = similar_person_id)
                            else:
                                logging.info("  Face recognition result: Unknown person")
                                logging.info("  K: " + str(k))
                                self.face_capturer(img_rd, faces[k], k, phase = 1)

                    # 7. Add note on cv2 window
                    self.draw_note(img_rd)
                    self.draw_status(img_rd)

            else:
                # Face detected
                logging.info("  there is no face in database")
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
                            self.current_frame_face_display_message.append("")

                    # Show the ROI of faces
                    for k, d in enumerate(faces):
                        if self.once:
                            self.first_faces = len(faces)
                            self.once = False
                        self.face_capturer(img_rd, d, k, phase = 0)

            # 8. Press 'q' to exit
            if kk == ord('q'):
                break

            self.update_fps()
            cv2.namedWindow("camera", 1)
            # img_rd = cv2.resize(img_rd, None, fx=1.5, fy=1.5)
            cv2.imshow("camera", img_rd)

            logging.info("  Frame ends\n\n")

    def run(self):
        # logging.debug(" mp_face_mesh: " + str(mp_face_mesh))

        cap = cv2.VideoCapture(os.path.join('data' , 'test2.mp4'))  # Get video stream from video file
        # cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Get video stream from camera im mac
        # cap = cv2.VideoCapture(0)  # Get video stream from camera im windows
        # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Get video stream from camera im windows
        self.process(cap)
        cap.release()
        cv2.destroyAllWindows()


def main():
    # logging.basicConfig(level=logging.DEBUG) # Set log level to 'logging.DEBUG' to print debug info of every frame
    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # logging.basicConfig(filename="log.txt", filemode="w")
    logger = logging.getLogger()  # Let us Create an object
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('log.txt', 'w', 'utf-8')
    logger.addHandler(handler)

    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()