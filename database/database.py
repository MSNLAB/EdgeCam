import time
import mysql.connector
from datetime import datetime
from mysql.connector import errorcode
from loguru import logger



class DataBase:
    def __init__(self, config):
        self.cnx = mysql.connector.connect(**config.connection)
        self.database_name = config.database_name
        self.tables = config.tables
        self.table_description = config.table_description
        self.insert_description = config.insert_description
        self.select_description = config.select_description


    def _create_database(self, cursor):
        try:
            cursor.execute("CREATE DATABASE {}".format(self.database_name))
        except mysql.connector.Error as err:
            logger.error("Failed creating database: {}".format(err))
            exit(1)

    def use_database(self):
        cursor = self.cnx.cursor()
        try:
            cursor.execute("USE {}".format(self.database_name))
        except mysql.connector.Error as err:
            logger.error("Database {} does not exists.".format(self.database_name))
            if err.errno == errorcode.ER_BAD_DB_ERROR:
                self._create_database(cursor)
                logger.success("Database {} created successfully.".format(self.database_name))
                self.cnx.database = self.database_name
            else:
                logger.error(err)
                exit(1)
        cursor.close()

    def create_tables(self):
        cursor = self.cnx.cursor()
        table_description = self.table_description
        for table_name in self.tables:
            try:
                logger.info("Creating table {}: ".format(table_name))
                cursor.execute(table_description.format(table_name))
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    cursor.execute("drop table {}".format(table_name))
                    logger.info("already exists, drop it.")
                    cursor.execute(table_description.format(table_name))
                else:
                    logger.error(err.msg)
            else:
                logger.success("create successfully")
        cursor.close()

    def clear_table(self, edge_id):
        cursor = self.cnx.cursor()
        table_name = self.tables[edge_id - 1]
        try:
            cursor.execute("truncate table {}".format(table_name))
        except mysql.connector.Error as err:
            cursor.close()
            logger.error(err.msg)
        else:
            cursor.close()
            logger.success("clear successfully")


    def select_result(self, edge_id):
        cursor = self.cnx.cursor()
        select_description = self.select_description
        table_name = self.tables[edge_id-1]
        try:
            cursor.execute(select_description.format(table_name))
            results = cursor.fetchall()
        except Exception as e:
            logger.error('Query failed {}'.format(e))
            cursor.close()
            return None
        else:
            cursor.close()
            logger.success('query successfully')
            return results

    def insert_data(self, table_name, data):
        cursor = self.cnx.cursor()
        insert_sql = self.insert_description.format(table_name)
        try:
            cursor.execute(insert_sql, data)
            # Make sure data is committed to the database
            self.cnx.commit()

        except Exception as e:
            cursor.close()
            logger.error("insert error {}".format(e))
        else:
            cursor.close()
            logger.success('insert successfully')



if __name__ == '__main__':
    pass