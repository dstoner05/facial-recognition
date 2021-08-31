import mysql.connector as mysql

HOST = "192.168.9.76"

DATABASE = "Eastcoast"

USER = "sa"

PASSWORD = "Regency1"

db_connection = mysql.connect(host= HOST, database = DATABASE, user = USER, password = PASSWORD)
print("Connected to:", db_connection.get_server_info())
