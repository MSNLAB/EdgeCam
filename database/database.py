
import time

import mysql.connector
from datetime import datetime
from mysql.connector import errorcode
from scipy.io._idl import AttrDict



class DataBase:
    def __init__(self, config):
        self.cnx = mysql.connector.connect(**config.connection)
        self.cursor = self.cnx.cursor()
        self.database_name = config.database_name
        self.tables = config.tables
        self.add_data = config.add_data

    def _create_database(self):
        try:
            self.cursor.execute("CREATE DATABASE {}".format(self.database_name))
        except mysql.connector.Error as err:
            print("Failed creating database: {}".format(err))
            exit(1)

    def use_database(self):
        try:
            self.cursor.execute("USE {}".format(self.database_name))
        except mysql.connector.Error as err:
            print("Database {} does not exists.".format(self.database_name))
            if err.errno == errorcode.ER_BAD_DB_ERROR:
                self._create_database()
                print("Database {} created successfully.".format(self.database_name))
                self.cnx.database = self.database_name
            else:
                print(err)
                exit(1)

    def create_tables(self):
        for table_name in self.tables:
            table_description = self.tables[table_name]
            try:
                print("Creating table {}: ".format(table_name), end='')
                self.cursor.execute(table_description)
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    print("already exists.")
                else:
                    print(err.msg)
            else:
                print("create successfully")

    def insert_data(self, data):
        self.cursor.execute(self.add_data)
        # Make sure data is committed to the database
        self.cnx.commit()


if __name__ == '__main__':
    config = AttrDict()
    config.connection = {'user': 'root', 'password': 'root', 'host': '127.0.0.1', 'raise_on_warnings': True}
    config.database_name = 'mydatabase'
    config.tables = {}
    config.tables['record']= (
    "CREATE TABLE `record` ("
    " `index` int NOT NULL, "
    " `start_time` timestamp ,"
    " `end_time` timestamp ,"
    " `result` text, "
    " PRIMARY KEY(`index`)"
    ") ENGINE=InnoDB")
    config.add_data = (
        "INSERT INTO `record`"
        "(`index`, `start_time`, `end_time`, `result`) "
        "VALUES ('1','2022-11-29 21:19:37','2022-11-29 21:19:48',' ')"
    )

    res ={'boxes':
            [[57., 87., 66., 94.],
             [58., 94., 68., 95.],
             [70., 88., 81., 93.],
             [10., 37., 17., 40.]],
        'labels':
            [2, 3, 3, 4],
        'scores':
            [0.99056727, 0.98965424, 0.93990153, 0.9157755]
          }

    data = {
        'index': 1,
        'start_time': datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
        'end_time': datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
        'result': str(res)
    }

    data_str = (
                '1', datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                'NULL'
                )
    db = DataBase(config)
    db.use_database()
    db.create_tables()
    db.insert_data(data_str)
    pass