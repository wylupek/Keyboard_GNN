import sqlite3, sys
from math import ceil
from collections import Counter

# Connect to the SQLite database file
db_file = "../keystroke_data.sqlite"
connection = sqlite3.connect(db_file)

user_id = int(sys.argv[1])
try:
    # Create a cursor object
    cursor = connection.cursor()

    # Query to fetch all rows where user_id = 60 and order by timestamp
    query = f"""
    SELECT timestamp, key, duration FROM key_press
    WHERE user_id = {user_id} and duration > 10000
    ORDER BY timestamp, duration desc;
    """
    cursor.execute(query)
    rows = cursor.fetchall()

    # Calculate the size of each fifth
    total_rows = len(rows)
    fifth_size = ceil(total_rows / 5)
    
    # for row in rows:
    #     print(row)

    print(user_id, len(rows))

finally:
    # Close the connection
    connection.close()
