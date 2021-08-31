import sqlite3
from sqlite3 import Error

def create_db(encoder):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(encoder)
        print(sqlite3.version)
    except Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


def create_connection(encoded_names):
    conn = None
    try:
        conn = sqlite3.connect(encoded_names)
        return conn
    except Error as e:
        print(e)



def create_table(conn, create_table_sql):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)
sql_create_users_table = """ CREATE TABLE IF NOT EXISTS users (
                                    id integer PRIMARY KEY,
                                    name text NOT NULL,
                                    employee_id integer NOT NULL,
                                    company_id integer NOT NULL,
                                    encoding integer NOT NULL
                                    );"""

def main():

    create_db(r"encoded_names.db")
    database = r"encoded_names.db"
    conn = create_connection(database)

    if conn is not None:
        create_table(conn, sql_create_users_table)


if __name__=='__main__':
    main()