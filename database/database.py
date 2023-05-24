import time
import munch
import mysql.connector
from datetime import datetime
from mysql.connector import errorcode
from loguru import logger



class DataBase:
    def __init__(self, config):
        self.cnx = mysql.connector.connect(**config.connection)
        self.database_name = config.database_name
        self.table_desc = "CREATE TABLE `{}` " \
                                 "(`index` int NOT NULL, " \
                                 "`start_time` double," \
                                 "`end_time` double, " \
                                 "`result` TEXT, " \
                                 "`log` VARCHAR(255)," \
                                 "PRIMARY KEY(`index`)) ENGINE=InnoDB"

        self.insert_desc = "INSERT INTO `{}` " \
                                  "(`index`, `start_time`, `end_time`, `result`, `log`) " \
                                  "VALUES (%s ,%s, %s, %s, %s);"

        self.select_desc= "SELECT `index`, `start_time`, `end_time`, `result`, `log` FROM `{}`;"

        self.select_one_desc = "SELECT `index`, `start_time`, `end_time`, `result`, `log` FROM `{}`" \
                               "where `index` = %s"

        self.update_desc = "UPDATE `{}` SET `end_time` = %s, " \
                                    "`result` = %s, " \
                                    "`log` = %s " \
                                    "WHERE `index` = %s"

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

    def create_table(self, edge_ids):
        cursor = self.cnx.cursor()
        table_sql = self.table_desc
        for id in edge_ids:
            try:
                logger.info("Creating table `{}` ".format(id))
                cursor.execute(table_sql.format(id))
            except mysql.connector.Error as err:
                if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                    cursor.execute("drop table `{}`".format(id))
                    logger.info("already exists, drop it.")
                    cursor.execute(table_sql.format(id))
                else:
                    logger.error(err.msg)
            else:
                logger.success("create successfully")
        cursor.close()

    def clear_table(self, edge_id):
        cursor = self.cnx.cursor()
        try:
            cursor.execute("truncate table `{}`".format(edge_id))
        except mysql.connector.Error as err:
            cursor.close()
            logger.error(err.msg)
        else:
            cursor.close()
            logger.success("clear successfully")


    def select_result(self, edge_id):
        cursor = self.cnx.cursor()
        select_sql = self.select_desc
        try:
            cursor.execute(select_sql.format(edge_id))
            results = cursor.fetchall()
        except Exception as e:
            logger.error('Query failed {}'.format(e))
            cursor.close()
            return None
        else:
            cursor.close()
            logger.success('query successfully')
            return results

    def select_one_result(self, edge_id, index):
        cursor = self.cnx.cursor()
        select_sql = self.select_one_desc
        try:
            cursor.execute(select_sql.format(edge_id), (index,))
            result = cursor.fetchone()
            logger.debug(result)
        except Exception as e:
            logger.error('Query failed {}'.format(e))
            cursor.close()
            return None
        else:
            cursor.close()
            logger.success('query successfully')
            return result

    def insert_data(self, table_name, data):
        cursor = self.cnx.cursor()
        insert_sql = self.insert_desc.format(table_name)
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


    def update_data(self, table_name, data):
        cursor = self.cnx.cursor()
        update_sql = self.update_desc.format(table_name)
        try:
            cursor.execute(update_sql, data)
            # Make sure data is committed to the database
            self.cnx.commit()

        except Exception as e:
            cursor.close()
            logger.error("update error {}".format(e))
        else:
            cursor.close()
            logger.success('update successfully')



if __name__ == '__main__':
    config = {
        "connection": {'user': 'root', 'password': 'root', 'host': '127.0.0.1', 'raise_on_warnings': True},
        "database_name": 'mydatabase',
    }
    config = munch.munchify(config)
    mydb = DataBase(config)
