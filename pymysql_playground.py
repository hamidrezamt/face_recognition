import pymysql

def db_initialize():
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
            print("The database 'dlib_face' exists.")
        else:
            print("The database 'dlib_face' does not exist.")

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

            print(f"The tables have been created in the database {db_info['db']}.")

        finally:
            # close the connection
            conn.close()