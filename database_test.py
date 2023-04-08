import os
import dlib
import numpy as np
import pymysql
import cv2
import logging
from datetime import datetime


# Create a connection object
dbServerName = "localhost"
dbUser = "hamidreza"
dbPassword = "@123HrmZ123$"
dbName = "dlib_face"
charSet = "utf8mb4"
cusrorType = pymysql.cursors.DictCursor

db = pymysql.connect(host=dbServerName, user=dbUser, password=dbPassword, db=dbName, charset=charSet,cursorclass=cusrorType)
cursor = db.cursor()


path_photos_from_camera = "data/data_faces_from_camera/"

existing_faces_cnt = 0

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


def return_128d_features(path_img):
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

def photo_id(photo_name):
    return int(photo_name.split("_")[2].split('_')[0])

def return_features_mean_personX(path_face_personX, id):
    print(path_face_personX)
    features_list_personX = []
    photos_list = [f for f in os.listdir(path_face_personX) if not f.startswith('.')]
    photos_list.sort(key = photo_id)
    print(len(photos_list))

    current_date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("SELECT COUNT(*) FROM `person_features` WHERE `person_id` = " + str(id) + ";")
    database_photo_cnt = int(cursor.fetchall()[0]["COUNT(*)"])
    print(database_photo_cnt)

    if photos_list:
        for i in range(database_photo_cnt, len(photos_list)):
            # Get 128D features for single image of personX
            logging.info("%-40s %-20s", "  Reading image:", path_face_personX + "/" + photos_list[i])
            features_128d = return_128d_features(path_face_personX + "/" + photos_list[i])
            # Jump if no face detected from image
            if features_128d == 0:
                i += 1
            else:
                # features_list_personX.append(features_128d)
                # 2. Insert person 1 to person X
                cursor.execute("INSERT INTO `person_features` (`person_id`, `date`, `image_path`) VALUES (" + str(id) + " , '" + str(current_date) + "' , '" + path_face_personX + "/" + photos_list[i] + "');")
                logging.debug("  INSERT INTO `person_features` (`person_id`, `date`, `image_path`) VALUES (" + str(id) + " , '" + str(current_date) + "' , '" + path_face_personX + "/" + photos_list[i] + "');")
                # print("INSERT INTO `person_features` (`person_id`, `date`, `image_path`) VALUES (" + str(id) + " , '" +  str(current_date) + "' , '" +  path_face_personX + "/" + photos_list[i] + "');")

                # 3. Insert features for person X
                for j in range(128):
                    cursor.execute("UPDATE `person_features` SET `feature_" + str(j + 1) + '`=' + str(features_128d[j]) + " WHERE `image_path`='" + path_face_personX + "/" + photos_list[i] + "';")
                    logging.debug("  UPDATE `person_features` SET `feature_" + str(j + 1) + '`=' + str(features_128d[j]) + " WHERE `image_path`='" + path_face_personX + "/" + photos_list[i] + "';")
    else:
        logging.warning("  Warning: No images in%s/", path_face_personX)


    cursor.execute("SELECT * FROM `person_features` WHERE `person_id` = " + str(id) + ";")
    feature_list_temp = cursor.fetchall()
    for i in range(len(feature_list_temp)):
        features = list(feature_list_temp[i].values())[4:]
        features = [float(feature) for feature in features]
        features_list_personX.append(features)
    

    # # Compute the mean
    # # personX 128D -> 1 x 128D
    # if features_list_personX:
    #     features_mean_personX = np.array(features_list_personX, dtype=object).mean(axis=0)
    # else:
    #     features_mean_personX = np.zeros(128, dtype=object, order='C')
    # return features_mean_personX
    db.commit()
    print(features_list_personX)


def check_existing_faces_cnt():
    if os.listdir(path_photos_from_camera):
        # Get the last recorded face number
        person_list = [f for f in os.listdir(path_photos_from_camera) if not f.startswith('.')]
        person_num_list = []
        for person in person_list:
            person_order = person.split('_')[1].split('_')[0]
            person_num_list.append(int(person_order))
        print("max: " + str(max(person_num_list)))
        return max(person_num_list)

    else:
        return 0


def feature_extraction_csv():
    # Get the order of latest person
    person_list = os.listdir(path_photos_from_camera)
    person_list.sort()

    # 0. clear table in mysql
    # cursor.execute("TRUNCATE TABLE 'person_features';")
    cursor.execute("TRUNCATE TABLE `mean_person_features`;")

    # 1. check existing people in mysql
    cursor.execute("SELECT COUNT(*) FROM `mean_person_features`;")
    person_start = int(cursor.fetchall()[0]["COUNT(*)"])
    person_cnt = check_existing_faces_cnt()
    
    # with open("data/features_all.csv", "w", newline="") as csvfile:
        # writer = csv.writer(csvfile)
    for person in range(person_start, person_cnt):
        # Get the mean/average features of face/personX, it will be a list with a length of 128D
        logging.info(path_photos_from_camera + "person_" + str(person + 1))
        features_mean_personX = return_features_mean_personX(path_photos_from_camera + "person_" + str(person + 1))

        print("The mean of features:", list(features_mean_personX))
        print('\n')

        # 2. Insert person 1 to person X
        cursor.execute("INSERT INTO `mean_person_features` (`person_id`) VALUES(" + str(person+1) + ");")

        # 3. Insert features for person X
        for i in range(128):
            cursor.execute("UPDATE `mean_person_features` SET `mean_feature_" + str(i + 1) + '`=' + str(features_mean_personX[i]) + " WHERE `person_id`=" + str(person + 1) + ";")
            print("UPDATE `mean_person_features` SET `mean_feature_" + str(i + 1) + '=' + str(features_mean_personX[i]) + " WHERE `person_id`=" + str(person + 1) + ";")
    
    db.commit()



# Get known faces from "features_all.csv"
def get_face_database():
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
            # results = results[0].values()
            results = list(results[0].values())
            # results = results[1:]
            # self.features_known_list.append(results)
            # self.name_known_list.append("Person_" + str(person + 1))
            print(results)
            print("\n\n\n")
        # self.name_known_cnt = len(self.name_known_list)
        # print("Faces in Database：", len(self.features_known_list))
        return 1
    else:
        return 0
    
# get_face_database()
# feature_extraction_csv()
return_features_mean_personX("data/data_faces_from_camera/person_1", 1)