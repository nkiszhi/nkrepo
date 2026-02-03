# -*- coding: utf-8 -*-
"""
NKREPO Database Operations Module

Provides database operations for malware sample management including:
- File hash calculation and lookup
- Sample storage management
- Database CRUD operations
"""

import pymysql
import hashlib
import os
import shutil

from config import (
    Config,
    MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_CHARSET,
    DB_MAIN, UPLOAD_FOLDER
)


class DatabaseOperation:
    """Database operations for malware sample management."""

    def _get_connection(self, database=None):
        """Create a database connection."""
        return pymysql.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            passwd=MYSQL_PASSWORD,
            db=database or DB_MAIN,
            charset=MYSQL_CHARSET
        )

    def filesha256(self, name):
        """
        Process uploaded file: calculate hashes, check database, and store sample.

        Args:
            name: Filename of the uploaded file

        Returns:
            Tuple of (sha256, md5, vt_status) for new files, or (record, vt_status) for existing
        """
        file_path = os.path.join(UPLOAD_FOLDER, name)

        # Calculate hashes
        with open(file_path, 'rb') as f:
            content = f.read()
            str_sha256 = hashlib.sha256(content).hexdigest()
            str_md5 = hashlib.md5(content).hexdigest()

        # Check if sample exists in database
        data_vs = self.mysqlsha256(str_sha256, str_md5)

        if data_vs == 0:
            # New sample: move to repository
            new_dir_path = str(Config.get_sample_dir(str_sha256))
            os.makedirs(new_dir_path, exist_ok=True)
            new_file_path = os.path.join(new_dir_path, str_sha256)
            shutil.move(file_path, new_file_path)
            return str_sha256, str_md5, '0'
        else:
            # Existing sample: remove uploaded file
            os.remove(file_path)
            data_vs = data_vs[0]
            vt_status = '0' if data_vs[12] == 'nan\r' else '1'
            return data_vs, vt_status

    def mysqlsha256(self, str_sha256, str_md5):
        """
        Query or insert sample by SHA256 hash.

        Args:
            str_sha256: SHA256 hash of the file
            str_md5: MD5 hash of the file

        Returns:
            Query result if found, 0 if new record inserted
        """
        table_name = Config.get_table_name(str_sha256)
        print(table_name)

        conn = self._get_connection()
        try:
            with conn.cursor() as cursor:
                sql = f"SELECT * FROM `{table_name}` WHERE TRIM(sha256) = %s;"
                cursor.execute(sql, (str_sha256,))
                query_result = cursor.fetchall()

                if not query_result:
                    # Insert new record
                    new_data = ['nan', str_md5, str_sha256, 'nan', 'nan', 'nan', 'nan', 'nan',
                                'nan', 'nan', 'nan', 'nan', 'nan', 'nan\r', 'nan', 'nan']
                    print(new_data)
                    sql_insert = f"""INSERT INTO `{table_name}`
                        (name, md5, sha256, src_file, date, category, platform, family,
                         result, microsoft_result, filetype, packer, ssdeep, detection,
                         behaviour_summary, behaviour_mitre_trees)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"""
                    cursor.execute(sql_insert, new_data)
                    conn.commit()
                    return 0
                else:
                    return query_result
        except pymysql.MySQLError as e:
            print(f"Database error: {e}")
        finally:
            conn.close()

    def mysql(self, table_name, database):
        """
        Query random 20 SHA256 hashes from specified table.

        Args:
            table_name: Name of the table to query
            database: Database name

        Returns:
            List of SHA256 hashes, or 0 on error
        """
        conn = None
        try:
            conn = self._get_connection(database)
            with conn.cursor() as cursor:
                sql = f"SELECT sha256 FROM `{table_name}` ORDER BY RAND() LIMIT 20"
                cursor.execute(sql)
                results = cursor.fetchall()
                return [row[0] for row in results]
        except pymysql.MySQLError as e:
            print(f"Database error: {e}")
            return 0
        finally:
            if conn:
                conn.close()

    def update_db(self, str_sha256):
        """
        Mark sample as having VirusTotal data.

        Args:
            str_sha256: SHA256 hash of the sample

        Returns:
            Number of rows affected, or None on error
        """
        table_name = Config.get_table_name(str_sha256)
        conn = self._get_connection()

        try:
            with conn.cursor() as cursor:
                sql = f"UPDATE `{table_name}` SET vt = %s WHERE sha256 = %s;"
                cursor.execute(sql, ('1', str_sha256))
                conn.commit()
                return cursor.rowcount
        except pymysql.Error as e:
            conn.rollback()
            print(f"Database error: {e}")
            return None
        finally:
            conn.close()

    def mysqlsha256s(self, str_sha256):
        """
        Query sample details by SHA256 hash.

        Args:
            str_sha256: SHA256 hash of the sample

        Returns:
            Query result if found, 0 if not found
        """
        table_name = Config.get_table_name(str_sha256)
        conn = self._get_connection()

        try:
            with conn.cursor() as cursor:
                sql = f"SELECT * FROM `{table_name}` WHERE TRIM(sha256) = %s;"
                cursor.execute(sql, (str_sha256,))
                query_result = cursor.fetchall()
                return query_result if query_result else 0
        finally:
            conn.close()


# Backward compatibility alias
Databaseoperation = DatabaseOperation
