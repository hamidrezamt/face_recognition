# Real-time face detection and recognition via OT for multi faces
# Do detection -> recognize face, new face -> not do re-recognition
# Do re-recognition for multi faces will cost much time, OT will be used to instead it

# Import required libraries
import dlib    # dlib library for face detection and recognition
import pymysql    # pymysql library for working with MySQL databases
import numpy as np    # numpy library for working with arrays and numerical operations
import cv2    # cv2 (OpenCV) library for image processing and computer vision
import os    # os library for interacting with the operating system
import time    # time library for working with time-related functions
import logging    # logging library for logging messages and debugging
from imutils import face_utils    # face_utils from imutils library for face-related utilities
from scipy.spatial import distance as dist    # distance function from scipy.spatial for calculating Euclidean distance
from datetime import datetime    # datetime module for working with date and time objects
import mediapipe as mp    # mediapipe library for real-time face and hand tracking

# Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Get face landmarks
shape_predictor_path = os.path.join('data', 'data_dlib', 'shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(shape_predictor_path)

# Use Dlib resnet50 model to get 128D face descriptor
face_reco_model_path = os.path.join('data', 'data_dlib', 'dlib_face_recognition_resnet_model_v1.dat')
face_reco_model = dlib.face_recognition_model_v1(face_reco_model_path)

# Initialize Mediapipe face mesh module
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create a connection object for the database
connection_params = { "host": "localhost", "user": "hamidreza", "password": "@123HrmZ123$", "db": "dlib_face", "charset": "utf8mb4", "cursorclass": pymysql.cursors.DictCursor }
database = pymysql.connect(**connection_params)
cursor = database.cursor()

# Class for the face recognizer
class Face_Recognizer:
    def __init__(self):
        # Initialize required variables and configurations
        self.font = cv2.FONT_ITALIC

        # FPS related variables
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # Frame counter
        self.frame_count = 0

        # Variables for storing face features and names of known subjects
        self.face_features_known_list = []
        self.face_name_known_list = []

        # Lists for storing centroid positions of region of interest in previous and current frames
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # Lists for storing names of subjects in previous and current frames
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        # counters for the number of detected faces in the previous and current frames
        self.last_frame_face_count = 0
        self.current_frame_face_count = 0

        # Store the Euclidean distance for comparison during recognition
        self.current_frame_face_X_e_distance_list = []

        # Store the positions and names of current faces captured
        self.current_frame_face_position_list = []

        # List for storing facial features captured by the current frame
        self.current_frame_face_feature_list = []

        # Variable for storing Euclidean distance between centroid of region of interest in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # Variables for reclassification
        # If an "unknown" face is recognized, the face will be re-recognized after reclassify_interval_count counts to reclassify_interval
        self.reclassify_interval_count = 0
        self.reclassify_interval = 5

        self.path_photos_from_camera = os.path.join('data', 'data_faces_from_camera')    # Set the path for storing photos from the camera
        self.existing_faces_count = 0    # Initialize the count of existing faces to 0
        self.similarity_thresh = 0.4    # Set the similarity threshold for face recognition
        self.eye_ar_thresh = 0.21    # Set the eye aspect ratio threshold for detecting closed eyes
        self.blur_thresh = 60.0    # Set the blur threshold for detecting blurry images
        self.y_direction_thresh = 18.0    # 15.0 # Set the vertical direction threshold for detecting tilted faces
        self.x_direction_thresh = 30.0    # 17.0 # Set the horizontal direction threshold for detecting tilted faces
        self.dimesion_thresh = 100    # Set the dimension threshold for detecting small faces
        self.once = True    # Initialize the 'once' variable as True, used for performing certain operations only once
        self.first_faces = 0    # Initialize the count of first seen faces  to 0 , before storing any face's data into database to 0

    # Define a function for creating directories for saving photos
    def pre_work_mkdir(self):
        # Check if the directory for saving photos already exists
        if os.path.isdir(self.path_photos_from_camera):
            # If it already exists, do nothing
            pass
        else:
            # If it doesn't exist, create it
            os.mkdir(self.path_photos_from_camera)

    # Define a function for checking the number of existing faces
    def check_existing_faces_count(self):
        # Check if there are any files in the directory
        if os.listdir(self.path_photos_from_camera):
            # If there are, get a list of the last recorded face number for each person
            person_list = [f for f in os.listdir(self.path_photos_from_camera) if not f.startswith('.')]
            person_num_list = []
            for person in person_list:
                # Extract the person number from the file name and convert it to an integer
                person_order = person.split('_')[1].split('_')[0]
                person_num_list.append(int(person_order))
            # Get the highest recorded face number and save it as the existing face count
            logging.debug("  Number of persons in base folder: " + str(max(person_num_list)))
            self.existing_faces_count = max(person_num_list)
        else:
            # If there are no files in the directory, set the existing face count to 0
            self.existing_faces_count = 0

    # Define a function for getting the face database
    def get_face_database(self):
        # Initialize the list of known face features and names
        self.face_features_known_list = []
        self.face_name_known_list = []

        # 1. Get the number of people in the database
        cursor.execute("SELECT * FROM `mean_person_features`;")
        person_count = len(cursor.fetchall())
        logging.debug("  person_count: " + str(person_count))

        if person_count:
            # If there are people in the database:
            # 2. Get the features for each person in the database
            for person in range(person_count):
                # Look up the person's information in the database
                cmd_lookup = "SELECT * FROM `mean_person_features` WHERE `person_id`=" + str(person + 1) + ";"
                logging.debug("  SELECT * FROM `mean_person_features` WHERE `person_id`=" + str(person + 1) + ";")
                cursor.execute(cmd_lookup)
                results = cursor.fetchall()
                results = list(results[0].values())
                # Extract the features for the person and convert them to floats
                features = results[2:]
                results = [float(feature) for feature in features]
                # Add the person's features and name to the lists
                self.face_features_known_list.append(results)
                self.face_name_known_list.append("Person_" + str(person + 1))
            # Print debugging messages
            logging.debug("  face_features_known_list: " + str(self.face_features_known_list))
            logging.debug("  Faces in Database：" + str(len(self.face_name_known_list)))
            # Return 1 to indicate success
            return 1
        else:
            # If there are no people in the database, print a warning message and return 0 to indicate failure
            logging.warning("  No Face found!")
            return 0
        
    # Define a function for returning the 128D features of a single face image
    # Input: path_img - path to the image file    # Output: face_descriptor - a 128D vector of face features
    def return_128d_features(self, path_img):
        # Read the image file and detect faces
        img_rd = cv2.imread(path_img)
        faces = detector(img_rd, 1)
        # logging.info("%-40s %-20s", "  Image with faces detected:", path_img)
        # Check if a face was detected
        if len(faces) != 0:
            # If a face was detected, get its shape and compute its 128D features
            shape = predictor(img_rd, faces[0])
            face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        else:
            # If no face was detected, set the face descriptor to 0 and print a warning message
            face_descriptor = 0
            logging.warning("  NO FACE detected by return_128d_features function")
            # Remove the image file
            os.remove(path_img)
        # Return the 128D vector of face features
        return face_descriptor
    
    # Define a function for getting the ID of a face photo
    # Input: photo_name - the name of the photo file    # Output: the ID of the face in the photo (as an integer)
    def photo_id(self, photo_name):
        # Split the photo name by underscores and extract the ID
        return int(photo_name.split("_")[2].split('_')[0])
    
    # Define a function that returns the mean value of 128D face descriptor for person X
    def return_features_mean_personX(self, path_face_personX, id):
        # Create an empty list to store the features for person X
        features_list_personX = []
        # Initialize a boolean variable as False
        counter_up = False
        # Create a list of photos for person X by excluding files starting with '.' from the given path and sort it based on the photo ID
        photos_list = [f for f in os.listdir(path_face_personX) if not f.startswith('.')]
        photos_list.sort(key = self.photo_id)
        # logging.debug("  len(photos_list): " + str(len(photos_list)))
        # logging.debug("  photos_list: " + str(photos_list))

        # Query the database to get the features for person X and count the number of photos stored in the database for this person
        cursor.execute("SELECT * FROM `person_features` WHERE `person_id` = " + str(id) + ";")
        results = cursor.fetchall()
        database_photo_count = len(results)
        logging.debug("  database_photo_count: " + str(database_photo_count))
        # Get the current date and time
        current_date = datetime.combine(datetime.today().date(), datetime.today().time().replace(microsecond=0))
        # Check if there are any photos stored for person X in the database
        if(database_photo_count != 0):
            # Get the date and time when the last photo for person X was stored in the database
            last_stored_date = list(results[database_photo_count-1].values())[2]
            last_stored_date = datetime.strptime(last_stored_date, '%Y-%m-%d %H:%M:%S')
            # Calculate the time difference between the current date and the date when the last photo was stored in the database
            time_difference = (current_date - last_stored_date).total_seconds()
            logging.debug("  current_date: " + str(current_date))
            logging.debug("  last_stored_date: " + str(last_stored_date))
            logging.debug("  time_difference: " + str(time_difference))
            # Check if the time difference is greater than 300 seconds (i.e., 5 minutes)
            if time_difference > 300:
                counter_up = True
            else:
                counter_up = False
        # If there are unprocessed photos in directory for person X, loop through those photos starting from the last photo stored in the database
        if photos_list:
            for i in range(database_photo_count, len(photos_list)):
                # Get the 128D features for the current photo of person X
                logging.info("%-40s %-20s", "  Reading image:", os.path.join(path_face_personX , photos_list[i]))
                features_128d = self.return_128d_features(os.path.join(path_face_personX , photos_list[i]))
                # If no face is detected in the current photo, skip to the next photo
                if features_128d == 0:
                    i += 1
                else:
                    # Insert the person ID, date, and image path into the person_features table of database for the current photo of person X
                    cursor.execute("INSERT INTO `person_features` (`person_id`, `date`, `image_path`) VALUES (" + str(id) + " , '" + str(current_date.strftime('%Y-%m-%d %H:%M:%S')) + "' , '" + os.path.join(os.path.abspath(path_face_personX) , photos_list[i]) + "');")
                    # logging.debug("  INSERT INTO `person_features` (`person_id`, `date`, `image_path`) VALUES (" + str(id) + " , '" + str(current_date.strftime('%Y-%m-%d %H:%M:%S')) + "' , '" + os.path.abspath(path_face_personX) + "/" + photos_list[i] + "');")
                    # Update the features for the current photo of person X in the person_features table of database
                    for j in range(128):
                        cursor.execute("UPDATE `person_features` SET `feature_" + str(j + 1) + '`=' + str(features_128d[j]) + " WHERE `image_path`='" + os.path.join(os.path.abspath(path_face_personX) , photos_list[i]) + "' AND `person_id` = " + str(id) + " ;")
                        # logging.debug("  UPDATE `person_features` SET `feature_" + str(j + 1) + '`=' + str(features_128d[j]) + " WHERE `image_path`='" + os.path.abspath(path_face_personX) + "/" + photos_list[i] + "' AND `person_id` = " + str(id) + " ;")
            # If there are no photos for person X, print a warning message
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
        # Query the database to get the features for person X and store them in a list
        cursor.execute("SELECT * FROM `person_features` WHERE `person_id` = " + str(id) + ";")
        feature_list_temp = cursor.fetchall()
        for i in range(len(feature_list_temp)):
            features = list(feature_list_temp[i].values())[4:]
            features = [float(feature) for feature in features]
            features_list_personX.append(features)

        # Compute the mean of the features for person X
        if features_list_personX:
            features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
        else:
            features_mean_personX = np.zeros(128, dtype=object, order='C')
        # Return the mean features for person X and the value of the boolean variable counter_up
        return features_mean_personX, counter_up
    
    # Define a function to detect blur in an image
    def blur_detection(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Calculate the Laplacian variance to determine blur level
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Define a function to calculate the eye aspect ratio (EAR)
    def eye_aspect_ratio(self, eye):
        # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
        C = dist.euclidean(eye[0], eye[3])
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        # Return the eye aspect ratio
        return ear

    def close_eye_detection(self, shape):
        # get the indexes of the facial landmarks for the left and right eye, respectively
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # convert the facial landmark (x, y)-coordinates to a NumPy array
        shape = face_utils.shape_to_np(shape)

        # extract the left and right eye coordinates, then use the coordinates to compute the eye aspect ratio for both eyes
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # calculate the eye aspect ratio (EAR) for the left and right eyes
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)

        # calculate the average EAR for both eyes
        ced = (leftEAR + rightEAR) / 2.0

        # return the calculated EAR value
        return ced
    
    # Define a function to estimate the pose of the head in the image
    def head_pose_estimation(self, image):
        # Flip the image horizontally for a later selfie-view display
        # Also convert the color space from BGR to RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        image.flags.writeable = False    # To improve performance

        results = face_mesh.process(image)    # Get the result from the face mesh

        image.flags.writeable = True    # To improve performance

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)    # Convert the color space from RGB to BGR

        img_h, img_w, img_c = image.shape    # Get the height, width, and channel of the image
        
        face_3d = []    # Initialize the face 3D list
        face_2d = []    # Initialize the face 2D list

        # If face detected by face mesh
        if results.multi_face_landmarks:
            logging.debug("face also detected by face mesh")
            
            # For each face landmark detected
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    
                    # Check if landmark is one of the specific landmarks required for pose estimation
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Append the 2D and 3D coordinates to the respective lists
                        face_2d.append([x, y])
                        face_3d.append([x, y, lm.z])       
                
                # Convert the 2D and 3D lists to NumPy arrays
                face_2d = np.array(face_2d, dtype=np.float64)
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w
                cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])

                dist_matrix = np.zeros((4, 1), dtype=np.float64)    # The distortion parameters

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                rmat, jac = cv2.Rodrigues(rot_vec)    # Get rotational matrix

                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)    # Get angles

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                
            return (x, y)
    
    # Define a function to check the direction of head pose
    def head_direction(self, image):
        try:
            # Get the x, y rotation degree
            x, y = self.head_pose_estimation(image)
            logging.debug(" Y index is: " + str(y) + " | X index is: " + str(x))

            # Check if the y and x index exceed the threshold value
            if y < self.y_direction_thresh * -1 or y > self.y_direction_thresh or x < self.x_direction_thresh * -1 or x > self.x_direction_thresh:
                dir = 0 # Head direction out of range
            else:
                dir = 1 # Head direction in range

        except (TypeError):
            logging.debug(" Face Mesh detect NO face in frame")
            dir = 0
        return dir

    def image_dimension(self, image):
        # Get the width, height, and number of channels of the image
        width, height, channel = image.shape
        # Log the width and height of the image
        logging.debug("  image's width: " + str(width) + " and image height: " + str(height))
        # Check if both the width and height of the image are greater than or equal to the specified threshold
        if width >= self.dimesion_thresh and height >= self.dimesion_thresh:
            # Return 1 if the image dimensions meet the threshold
            return 1
        else:
            # Return 0 if the image dimensions do not meet the threshold
            return 0
    
    # Define a function for feature extraction
    def feature_extraction(self, index):
        # Extract features for a given index

        # 1. Check the number of existing people in the MySQL database
        cursor.execute("SELECT * FROM `mean_person_features`;")
        results = cursor.fetchall()
        person_start = len(results)
        logging.debug("  person_start: " + str(person_start))

        # Check the number of existing faces in the directory
        self.check_existing_faces_count()
        logging.debug("  self.existing_faces_count: " + str(self.existing_faces_count))

        # Get the mean/average features of face/personX, it will be a list with a length of 128D
        features_mean_personX, counter_up = self.return_features_mean_personX(os.path.join(self.path_photos_from_camera , "person_" + str(index + 1)), index + 1)

        if person_start and person_start == self.existing_faces_count:
            # If the number of people in the MySQL database matches the number of faces in the directory:
            # 2.1 Update person X's counter in the MySQL database
            current_counter = list(results[index].values())[1]
            logging.debug("  current counter: " + str(current_counter))
            if counter_up:
                cursor.execute("UPDATE `mean_person_features` SET `counter` = " + str(current_counter + 1) + " WHERE `person_id` =" + str(index + 1) + ";")
                logging.debug("  UPDATE `mean_person_features` SET `counter` = " + str(current_counter + 1) + " WHERE `person_id` =" + str(index + 1) + ";")
        else:
            # If the number of people in the MySQL database does not match the number of faces in the directory:
            # 2.2 Insert person X's information into the MySQL database
            cursor.execute("INSERT INTO `mean_person_features` (`person_id` , `counter`) VALUES(" + str(index + 1) + " , 1);")
            logging.debug("  INSERT INTO `mean_person_features` (`person_id` , `counter`) VALUES(" + str(index + 1) + " , 1);")

        # 3. Insert the features of person X into the MySQL database
        for i in range(128):
            cursor.execute("UPDATE `mean_person_features` SET `mean_feature_" + str(i + 1) + '`=' + str(features_mean_personX[i]) + " WHERE `person_id`=" + str(index + 1) + ";")
            logging.debug("  UPDATE `mean_person_features` SET `mean_feature_" + str(i + 1) + '`=' + str(features_mean_personX[i]) + " WHERE `person_id`=" + str(index + 1) + ";")

        # Commit the changes to the MySQL database
        database.commit()
        logging.info("  Save all the features of faces registered into databse")

    # Define a function that captures faces from the given image and extracts features
    def face_capturer(self, image, face, phase = None, index = None):
        # Check if there is a face in the database and decrement the counter if it is the first face
        if phase is not None and phase == 0 and self.first_faces > 0:
            self.current_frame_face_name_list.append("unknown")
            logging.debug("  self.first_faces: " + str(self.first_faces))
            self.first_faces = self.first_faces - 1
        
        # Get the shape of the face and calculate its height and width
        shape = predictor(image, face)
        height = (face.bottom() - face.top())
        width = (face.right() - face.left())
        hh = int(height / 2)
        ww = int(width / 2)

        # Create a blank image with the size of the detected face
        img_blank = np.zeros((height * 2, width * 2, 3), np.uint8)
        img_tmp = np.zeros((height, width, 3), np.uint8)

        # Copy the pixels of the face from the original image to img_tmp
        for ii in range(height):
            for jj in range(width):
                try:
                    img_tmp[ii][jj] = image[face.top() + ii][face.left() + jj]
                except IndexError:
                    logging.debug("  IndexError: index 720 is out of bounds for axis 0 with size 720!")

        # Copy the pixels of the face from the original image to img_blank with some padding
        for ii in range(height * 2):
            for jj in range(width * 2):
                try:
                    img_blank[ii][jj] = image[face.top() - hh + ii][face.left() - ww + jj]
                except IndexError:
                    logging.debug("  IndexError: index 720 is out of bounds for axis 0 with size 720!")

        # Detect blur in the face, closed eyes, head direction, and image dimensions
        blur_index = self.blur_detection(img_tmp)
        close_eye_index = self.close_eye_detection(shape)
        head_direction_index = self.head_direction(img_blank)
        dimension_index = self.image_dimension(img_tmp)

        logging.debug("  blur_detection(): " + str(blur_index))
        logging.debug("  close_eye_detection(): " + str(close_eye_index))
        logging.debug("  head_direction(): " + str(head_direction_index))
        logging.debug("  image_dimension(): " + str(dimension_index))

        # If the face is clear, with open eyes, with an appropriate head direction, and has appropriate dimensions
        if blur_index > self.blur_thresh and close_eye_index > self.eye_ar_thresh and head_direction_index and dimension_index:
            debug_text = ""

            # If the index is None, create a new folder for the face
            if index is None:
                self.check_existing_faces_count()
                current_face_dir = os.path.join(self.path_photos_from_camera , "person_" + str(self.existing_faces_count + 1))
                os.makedirs(current_face_dir)
                logging.info("\n%-40s %s", "Create folders:", current_face_dir)
                index = self.existing_faces_count
                debug_text = "  Novel frame accepted and captured!"
            # Otherwise, use the existing folder
            else:
                current_face_dir = os.path.join(self.path_photos_from_camera , "person_" + str(index + 1))
                debug_text = "  frame accepted and captured!"
            
            # Save the face image in the appropriate folder and extract its features
            img_name = os.path.join(str(current_face_dir) , "img_face_" + str(self.frame_count) + "_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(close_eye_index) + "_" + str(head_direction_index) + ".jpg")
            cv2.imwrite(img_name, img_blank)
            logging.info("  Save into:                    " + img_name)

            self.feature_extraction(index)
            logging.debug(debug_text)

        # Dump the current frame image if needed
        filename = "debug/debug_" + str(self.frame_count) + "_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(close_eye_index) + "_" + str(head_direction_index) + ".jpg"
        filename = filename.replace('/', os.sep).replace('\\', os.sep)
        cv2.imwrite(filename, img_tmp) 
        # cv2.imwrite("debug/debug_" + str(self.frame_count) + "_" + "{:.1f}".format(blur_index) + "_" + "{:.2f}".format(close_eye_index) + "_" + str(head_direction_index) + ".jpg", img_tmp)  # Dump current frame image if needed

    def db_initialize(self):
        # Define database connection info
        db_info = {"host": "localhost", "user": "hamidreza", "password": "@123HrmZ123$", "db": "dlib_face", "charset": "utf8mb4"}

        # create a connection to the MySQL server
        conn_init = pymysql.connect(host=db_info["host"], user=db_info["user"], password=db_info["password"], charset=db_info["charset"])

        try:
            cursor = conn_init.cursor()    # create a cursor object
            cursor.execute("SELECT schema_name FROM information_schema.SCHEMATA")    # execute a query to get a list of all databases
            rows = cursor.fetchall()    # fetch all rows

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

        # If "dlib_face" database does not exist, create it
        if not db_exists:

            # create a connection to the MySQL server
            conn = pymysql.connect(host=db_info["host"], user=db_info["user"], password=db_info["password"], charset=db_info["charset"])
            try:
                cursor = conn.cursor()    # create a cursor object

                # execute a query to create the database if it doesn't exist
                cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_info['db']}")
                cursor.execute(f"USE {db_info['db']}")

                # Create the `person_features` table with 128 columns for feature vectors
                start_part = "CREATE TABLE `person_features` (`id` INT NOT NULL AUTO_INCREMENT, `person_id` INT NOT NULL, `date` VARCHAR(45) NOT NULL, `image_path` VARCHAR(2048) NOT NULL"
                middle_part = ""
                for i in range(128):
                    middle_part = middle_part + ", `feature_" + str(i+1) + "` VARCHAR(100) NULL"
                end_part = ", PRIMARY KEY (`id`));"
                person_sql_query = start_part + middle_part + end_part
                cursor.execute(person_sql_query)

                # Create the `mean_person_features` table with 128 columns for feature vectors
                mean_start_part = "CREATE TABLE `mean_person_features` (`person_id` INT NOT NULL, `counter` INT NOT NULL"
                mean_middle_part = ""
                for i in range(128):
                    mean_middle_part = mean_middle_part + ", `mean_feature_" + str(i+1) + "` VARCHAR(100) NULL"
                mean_end_part = ", PRIMARY KEY (`person_id`), UNIQUE INDEX `person_id_UNIQUE` (`person_id` ASC) VISIBLE);"
                mean_person_sql_query = mean_start_part + mean_middle_part + mean_end_part
                cursor.execute(mean_person_sql_query)

                # commit the changes
                conn.commit()

                logging.debug(f"  The tables have been created in the database {db_info['db']}.")

            finally:
                # close the connection
                conn.close()

    def update_fps(self):
        now = time.time()  # get the current time
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:  # check if a second has passed
            self.fps_show = self.fps  # update the displayed fps to the current fps
        self.start_time = now  # update the start time to the current time
        self.frame_time = now - self.frame_start_time  # calculate the time taken for processing the current frame
        self.fps = 1.0 / self.frame_time  # calculate the fps for the current frame
        self.frame_start_time = now  # update the start time for the current frame

    @staticmethod
    # Define a funtion to compute the Euclidean distance between the two features
    def return_euclidean_distance(feature_1, feature_2):
        # Convert the features into NumPy arrays
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        
        # Compute the Euclidean distance between the two features
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        
        # Return the computed distance
        return dist
    
    # Using Centroid Tracking to Recognize Faces
    # Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        logging.debug("  inside centroid_tracker!")
        logging.debug("  self.last_frame_face_name_list: " + str(self.last_frame_face_name_list))
        logging.debug("  self.current_frame_face_name_list: " + str(self.current_frame_face_name_list))
        
        # Loop through each centroid in the current frame
        for i in range(len(self.current_frame_face_centroid_list)):
            logging.debug("  inside for inside centroid_tracker!")
            logging.debug("  self.current_frame_face_centroid_list: " + str(self.current_frame_face_centroid_list))
            logging.debug("  self.last_frame_face_centroid_list: " + str(self.last_frame_face_centroid_list))
            
            e_distance_current_frame_person_x_list = []
            
            # For each centroid in the current frame, compute the Euclidean distance to centroids in the last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])
                
                e_distance_current_frame_person_x_list.append(self.last_current_frame_centroid_e_distance)
                # logging.debug("  e_distance_current_frame_person_x_list: " + str(e_distance_current_frame_person_x_list))

            # Find the index of the centroid in the last frame that has the minimum Euclidean distance to the current centroid
            last_frame_num = e_distance_current_frame_person_x_list.index(min(e_distance_current_frame_person_x_list))
            logging.debug("  last_frame_num: " + str(last_frame_num))
            
            # Assign the name of the person associated with the closest centroid in the last frame to the current centroid
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
            # Add a label with the face index number next to the face centroid
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple([int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]), self.font, 0.8, (255, 190, 0), 1, cv2.LINE_AA)
    
    # Face detection and recognition with object tracking from input video stream
    def process(self, stream):
        # Initialize face database
        self.db_initialize()

        # Loop through the video stream
        while stream.isOpened():
            # Increase frame count
            self.frame_count += 1
            # Log the start of the current frame
            logging.debug("  Frame " + str(self.frame_count) + " starts")
            
            # Read the current frame
            flag, img_rd = stream.read()
            
            # Wait for a key press
            kk = cv2.waitKey(1)

            # Detect faces for the current frame
            faces = detector(img_rd, 0)

            # Check if there is a face database
            if self.get_face_database():
                # Update count for faces in frames
                self.last_frame_face_count = self.current_frame_face_count
                self.current_frame_face_count = len(faces)

                # Update the face name list in last frame
                self.last_frame_face_name_list = self.current_frame_face_name_list[:]

                # Update the list of centroids for the previous frame and the current frame
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []

                # If there is no change in the number of faces in the current frame and the previous frame
                if (self.current_frame_face_count == self.last_frame_face_count) and (self.reclassify_interval_count != self.reclassify_interval):
                    # Log no face count changes in this frame
                    logging.debug("  scene 1: No face count changes in this frame!")

                    # Clear current frame face position list
                    self.current_frame_face_position_list = []

                    # If there are unknown faces, start reclassify_interval_count counting
                    if "unknown" in self.current_frame_face_name_list:
                        logging.debug("  There are unknown faces, start reclassify_interval_count counting")
                        self.reclassify_interval_count += 1

                    # If there are faces in the current frame, loop through each face
                    if self.current_frame_face_count != 0:
                        for k, d in enumerate(faces):
                            # Append face position to the current frame face position list
                            self.current_frame_face_position_list.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))
                            # Append centroid to the current frame centroid list
                            self.current_frame_face_centroid_list.append([int(faces[k].left() + faces[k].right()) / 2, int(faces[k].top() + faces[k].bottom()) / 2])
                            # Draw a rectangle around the face
                            img_rd = cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]), (255, 255, 255), 2)

                    # If there are multiple faces in the current frame, use centroid tracking
                    if self.current_frame_face_count != 1:
                        self.centroid_tracker()

                    # Log current frame face count and current frame face name list length
                    logging.debug("  current_frame_face_count: " + str(self.current_frame_face_count))
                    logging.debug("  current_frame_face_name_list len : " + str(len(self.current_frame_face_name_list)))

                    # Loop through each face in the current frame face name list
                    for i in range(len(self.current_frame_face_name_list)):
                        # Write the name of the face under the ROI
                        img_rd = cv2.putText(img_rd, self.current_frame_face_name_list[i], self.current_frame_face_position_list[i], self.font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                    # Draw note on cv2 window
                    self.draw_note(img_rd)

                # If the number of faces in the current frame and the previous frame changes
                else:
                    # Log face count changes in this frame
                    logging.debug("  scene 2: Faces count changes in this frame")
                    # Clear current frame face position list, current frame face X e distance list, current frame feature list and reclassify interval count
                    self.current_frame_face_position_list = []
                    self.current_frame_face_X_e_distance_list = []
                    self.current_frame_face_feature_list = []
                    self.reclassify_interval_count = 0

                    # If the number of faces reduced
                    if self.current_frame_face_count == 0:
                        # Log no faces in this frame
                        logging.debug("  scene 2.1 No faces in this frame!!!")
                        # Clear the list of names and features
                        self.current_frame_face_name_list = []
                    # If the number of faces increased
                    else:
                        # Log get faces in this frame and do face recognition
                        logging.debug("  scene 2.2 Get faces in this frame and do face recognition")
                        # Clear the list of names
                        self.current_frame_face_name_list = []

                        # Loop through each face in the frame
                        for i in range(len(faces)):
                            # Get the shape of the face
                            shape = predictor(img_rd, faces[i])
                            # Compute the face descriptor
                            self.current_frame_face_feature_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))
                            # Append "unknown" to the current frame face name list
                            self.current_frame_face_name_list.append("unknown")

                        # Traverse all faces in the captured image
                        for k in range(len(faces)):
                            # Log for face k in current frame
                            logging.debug("  For face %d in current frame:", k + 1)
                            # Get the shape of the face
                            shape = predictor(img_rd, faces[k])
                            # Append centroid to the current frame centroid list
                            self.current_frame_face_centroid_list.append([int(faces[k].left() + faces[k].right()) / 2, int(faces[k].top() + faces[k].bottom()) / 2])
                            # Clear the current frame face X e distance list
                            self.current_frame_face_X_e_distance_list = []
                            # Append the position of the face to the current frame face position list
                            self.current_frame_face_position_list.append(tuple([faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # Compare the faces in the database to every face detected
                            for i in range(len(self.face_features_known_list)):
                                if str(self.face_features_known_list[i][0]) != '0.0':
                                    # Compute the Euclidean distance between the current frame face feature list and the known face feature list
                                    e_distance_tmp = self.return_euclidean_distance(self.current_frame_face_feature_list[k], self.face_features_known_list[i])
                                    # Log the Euclidean distance
                                    logging.debug("      with person %d, the e-distance: %f", i + 1, e_distance_tmp)
                                    # Append the Euclidean distance to the current frame face X e distance list
                                    self.current_frame_face_X_e_distance_list.append(e_distance_tmp)
                                else:
                                    # Person_X
                                    self.current_frame_face_X_e_distance_list.append(999999999)

                            # Find the face with the minimum Euclidean distance
                            similar_person_num = self.current_frame_face_X_e_distance_list.index(min(self.current_frame_face_X_e_distance_list))

                            # If the minimum Euclidean distance is less than the similarity threshold
                            if min(self.current_frame_face_X_e_distance_list) < self.similarity_thresh:
                                # Set the current frame face name to the known face name
                                self.current_frame_face_name_list[k] = self.face_name_known_list[similar_person_num]
                                # Log the face recognition result
                                logging.debug("  Face recognition result: %s", self.face_name_known_list[similar_person_num])
                                # Save a photo of the face with the known name
                                self.face_capturer(img_rd, faces[k], index=similar_person_num)
                            else:
                                # Log the face recognition result as unknown person
                                logging.debug("  Face recognition result: Unknown person")
                                logging.debug("  K: " + str(k))
                                # Save a photo of the face with the unknown name
                                self.face_capturer(img_rd, faces[k], phase=1)

                        # Draw note on cv2 window
                        self.draw_note(img_rd)

            else:
                # Log there is no face in database
                logging.debug("  there is no face in database")
                # Update the list of centroids for the previous frame and the current frame
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.current_frame_face_centroid_list = []
                # If faces are detected
                if len(faces) != 0:
                    # Create folders to save photos
                    self.pre_work_mkdir()
                    # Update the last and current frame face count
                    self.last_frame_face_count = self.current_frame_face_count
                    self.current_frame_face_count = len(faces)
                    if self.current_frame_face_count != 0:
                        # Append the position of each face to the current frame face position list
                        for k, d in enumerate(faces):
                            self.current_frame_face_position_list.append(tuple([d.left(), int(d.bottom() + (d.bottom() - d.top()) / 4)]))
                            # Append the centroid of each face to the current frame centroid list
                            self.current_frame_face_centroid_list.append([int(d.left() + d.right()) / 2, int(d.top() + d.bottom()) / 2])

                    # Show the ROI of faces
                    for k, d in enumerate(faces):
                        if self.once:
                            self.first_faces = len(faces)
                            self.once = False
                        # Save a photo of the face with the unknown name
                        self.face_capturer(img_rd, d, phase=0)

            # Press 'q' to exit
            if kk == ord('q'):
                break

            # Update the FPS counter
            self.update_fps()
            # Create a window named "camera" with a flag of 1
            cv2.namedWindow("camera", 1)
            # Show the image in the window named "camera"
            cv2.imshow("camera", img_rd)

            # Log the end of the frame
            logging.debug("  Frame ends\n\n")
    
    # Run the program to process video stream
    def run(self):
        cap = cv2.VideoCapture(os.path.join('data' , 'test2.mp4'))  # Get video stream from video file
        # cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Get video stream from camera in mac
        # cap = cv2.VideoCapture(0)        # Get video stream from camera in windows
        self.process(cap)   # Process the video stream using the process method of the current object

        # Release the capture object and destroy all open windows
        cap.release()
        cv2.destroyAllWindows()

# Define the main function
def main():
    # Set up logging configuration and level
    logging.basicConfig(filename="log.txt", filemode="w", force=True) # Set the log file to "log.txt"
    logger = logging.getLogger() # Create a logging object
    logger.setLevel(logging.DEBUG) # Set the logging level to DEBUG
    # Create an instance of the Face_Recognizer class
    Face_Recognizer_con = Face_Recognizer()
    # Run the Face_Recognizer program
    Face_Recognizer_con.run()

# Check if the script is being run as the main program
if __name__ == '__main__':
    # Call the main function
    main()