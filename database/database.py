import time
import mysql.connector
from datetime import datetime
from mysql.connector import errorcode
from loguru import logger
from scipy.io._idl import AttrDict



class DataBase:
    def __init__(self, config):
        self.cnx = mysql.connector.connect(**config.connection)
        self.cursor = self.cnx.cursor()
        self.database_name = config.database_name
        self.tables = config.tables
        self.table_description = config.table_description
        self.insert_description = config.insert_description

    def _create_database(self):
        try:
            self.cursor.execute("CREATE DATABASE {}".format(self.database_name))
        except mysql.connector.Error as err:
            logger.error("Failed creating database: {}".format(err))
            exit(1)

    def use_database(self):
        try:
            self.cursor.execute("USE {}".format(self.database_name))
        except mysql.connector.Error as err:
            logger.error("Database {} does not exists.".format(self.database_name))
            if err.errno == errorcode.ER_BAD_DB_ERROR:
                self._create_database()
                logger.success("Database {} created successfully.".format(self.database_name))
                self.cnx.database = self.database_name
            else:
                logger.error(err)
                exit(1)

    def create_tables(self, edge_id=None):
        table_description = self.table_description
        if edge_id is not None:
            table_name = self.tables[edge_id-1]
            try:
                logger.info("Creating table {}: ".format(table_name), end='')
                self.cursor.execute(table_description.format(table_name))
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    self.cursor.execute("DELETE from {}".format(table_name))
                    logger.info("already exists, clean it.")
                else:
                    logger.error(err.msg)
            else:
                logger.success("create successfully")
        else:
            for table_name in self.tables:
                try:
                    logger.info("Creating table {}: ".format(table_name), end='')
                    self.cursor.execute(table_description.format(table_name))
                except mysql.connector.Error as err:
                    if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                        self.cursor.execute("DELETE from {}".format(table_name))
                        logger.info("already exists, clean it.")
                    else:
                        logger.error(err.msg)
                else:
                    logger.success("create successfully")

    def insert_data(self, table_name, data):
        insert_sql = self.insert_description.format(table_name)
        self.cursor.execute(insert_sql, data)
        # Make sure data is committed to the database
        self.cnx.commit()


if __name__ == '__main__':
    config = AttrDict()
    config.connection = {'user': 'root', 'password': 'root', 'host': '127.0.0.1', 'raise_on_warnings': True}
    config.database_name = 'mydatabase'
    config.tables = {}
    config.tables['result']= (
    "CREATE TABLE `result` ("
    " `index` int NOT NULL, "
    " `start_time` timestamp(6) ,"
    " `end_time` timestamp(6) ,"
    " `result` text, "
    " PRIMARY KEY(`index`)"
    ") ENGINE=InnoDB")
    config.add_data = (
        "INSERT INTO `record`"
        "(`index`, `start_time`, `end_time`, `result`) "
        "VALUES (%s ,%s, %s, %s);"
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

    data_str = (
                4,
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                str(res)
                )
    db = DataBase(config)
    db.use_database()
    #db.create_tables()
    db.insert_data(data_str)
    pass