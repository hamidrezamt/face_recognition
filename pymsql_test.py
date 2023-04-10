import pymysql


# connection = pymysql.connect(host='localhost',
#                              user='hamidreza',
#                              password='@123HrmZ123$',
#                              database='dlib_face',
#                              charset='utf8mb4',
#                              cursorclass=pymysql.cursors.DictCursor)

# with connection:
#     with connection.cursor() as cursor:
#         # Create a new record
#         sql = "INSERT INTO `faces` (`date`, `path`) VALUES (%s, %s)"
#         cursor.execute(sql, ('2008-11-11 11:12:01', '/Users/hamidreza/'))
#         # connection is not autocommit by default. So you must commit to save your changes.
#         connection.commit()

    # with connection.cursor() as cursor:
    #     # Read a single record
    #     sql = "SELECT * FROM `faces`"
    #     cursor.execute(sql)
    #     result = cursor.fetchall()
    #     print(result)

# i = 0
# features_mean_personX = 0.7723
# person = 4
# print("update dlib_face_table set feature_" + str(i + 1) + '=\"' + str(features_mean_personX) + "\" where person_x=\"person_" + str(person + 1) + "\";")


# Create a connection object
dbServerName = "localhost"
dbUser = "hamidreza"
dbPassword = "@123HrmZ123$"
dbName = "dlib_face"
charSet = "utf8mb4"
cusrorType = pymysql.cursors.DictCursor

connection = pymysql.connect(host=dbServerName, user=dbUser, password=dbPassword, db=dbName, charset=charSet,cursorclass=cusrorType)

start_part = "CREATE TABLE `dlib_face`.`person_features` (`id` INT NOT NULL AUTO_INCREMENT, `person_id` INT NOT NULL, `date` VARCHAR(45) NOT NULL, `image_path` VARCHAR(2048) NOT NULL"
middle_part = ""
for i in range(128):
    middle_part = middle_part + ", `feature_" + str(i+1) + "` VARCHAR(100) NULL"
end_part = ", PRIMARY KEY (`id`));"

mean_start_part = "CREATE TABLE `dlib_face`.`mean_person_features` (`person_id` INT NOT NULL, `counter` INT NOT NULL"
mean_middle_part = ""
for i in range(128):
    mean_middle_part = mean_middle_part + ", `mean_feature_" + str(i+1) + "` VARCHAR(100) NULL"
mean_end_part = ", PRIMARY KEY (`person_id`), UNIQUE INDEX `person_id_UNIQUE` (`person_id` ASC) VISIBLE);"


# print(sqlQuery)

# with connection:
#     with connection.cursor() as cursor:
#         cursor.execute("SELECT COUNT(TABLE_NAME) FROM information_schema.TABLES WHERE TABLE_SCHEMA LIKE 'dlib_face' AND TABLE_NAME = 'person_features';")
#         result = cursor.fetchall()
#         # for row in result:
#         #     print ("%s" % (row["COUNT(TABLE_NAME)"]))
#         print(result[0]["COUNT(TABLE_NAME)"])
#         if result[0]["COUNT(TABLE_NAME)"] == 1:
#             cursor.execute("DROP TABLE `person_features`")
#             connection.commit()
#         else:
#             sql_query = start_part + middle_part + end_part
#             cursor.execute(sql_query)
#             connection.commit()

with connection:
    with connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(TABLE_NAME) FROM information_schema.TABLES WHERE TABLE_SCHEMA LIKE 'dlib_face' AND TABLE_NAME = 'mean_person_features';")
        result = cursor.fetchall()
        print(result[0]["COUNT(TABLE_NAME)"])
        if result[0]["COUNT(TABLE_NAME)"] == 1:
            cursor.execute("DROP TABLE `mean_person_features`")
            connection.commit()
        else:
            mean_sql_query = mean_start_part + mean_middle_part + mean_end_part
            cursor.execute(mean_sql_query)
            connection.commit()
