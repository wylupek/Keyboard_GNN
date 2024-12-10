import sqlite3
import os
import glob
import csv
from io import StringIO
from datetime import datetime


def print_tsv(content: list[dict]) -> None:
    print("key\tpress_time\tduration\taccel_x\taccel_y\taccel_z")
    for row in content:
        print(f"{row['key']}\t{row['press_time']}\t{row['duration']}\t{row['accel_x']}\t{row['accel_y']}\t{row['accel_z']}")


def save_tsv(content: str, base_path:str, username: str):
    base_filename = username + '.tsv'
    file_path = os.path.join(base_path, base_filename)

    # Check if the file already exists
    if os.path.exists(file_path):
        counter = 1
        while os.path.exists(os.path.join(base_path, f"{username}.{counter}.tsv")):
            counter += 1
        file_path = os.path.join(base_path, f"{username}.{counter}.tsv")

    # Save the file
    with open(file_path, "w") as file:
        file.write(content)
        print(f"File saved as: {file_path}")


def drop_table() -> bool:
    """
    Drop the key_press table if it exists.
    :return: True if the table was dropped successfully, False otherwise.
    """

    try:
        conn = sqlite3.connect('keystroke_data.sqlite')
    except sqlite3.Error as e:
        print(f"An error occurred while connecting to the database: {e}")
        return False

    try:
        cursor = conn.cursor()
        cursor.execute('DROP TABLE IF EXISTS key_press')
        conn.commit()
        conn.close()
        print("Table 'key_press' has been dropped (if it existed).")
        return True
    except sqlite3.Error as e:
        conn.close()
        print(f"An error occurred while dropping the table: {e}")
        return False


def create_table() -> bool:
    """
    Create key_press table if they do not exist.
    :return: True if the table was created successfully or already exists, False if there was an error.
    """

    try:
        conn = sqlite3.connect('keystroke_data.sqlite')
    except sqlite3.Error as e:
        print(f"An error occurred while connecting to the database: {e}")
        return False

    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS key_press (
                user_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                press_time TIMESTAMP NOT NULL,
                duration INTEGER NOT NULL,
                accel_x REAL NOT NULL,
                accel_y REAL NOT NULL,
                accel_z REAL NOT NULL,
                timestamp TIMESTAMP NOT NULL
            )
        ''')

        # Create an index on the timestamp column
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON key_press (timestamp)')
        
        # Create a index on the user_id column
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON key_press (user_id)')

        conn.commit()
        conn.close()
        print("Creating table complete.")
        return True
    except sqlite3.Error as e:
        conn.close()
        print(f"An error occurred while creating up the table: {e}")
        return False


def add_tsv_values(content: list[dict], user_id: int) -> bool:
    """
    Insert records into the key_press table in the keystroke_data.sqlite database.
    :param content: A list of dictionaries containing key press data, where each
        dictionary must contain the keys 'key', 'press_time', 'duration', 'accel_x', 'accel_y', 'accel_z'.
    :param user_id: user_id of the user associated with the key presses.
    :return: True if the records were inserted successfully,
             False if there was an error during the process.
    """

    try:
        conn = sqlite3.connect('keystroke_data.sqlite')
    except sqlite3.Error as e:
        print(f"An error occurred while connecting to the database: {e}")
        return False

    try:
        cursor = conn.cursor()

        # Prepare and execute the insert statements
        timestamp = datetime.now().isoformat()
        for entry in content:
            cursor.execute('''
                INSERT INTO key_press (user_id, key, press_time, duration, accel_x, accel_y, accel_z, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user_id,
                entry["key"],
                entry["press_time"],
                entry["duration"],
                entry["accel_x"],
                entry["accel_y"],
                entry["accel_z"],
                timestamp
            ))

        conn.commit()
        conn.close()
        print(f"Inserted {len(content)} records successfully.")
        return True
    except sqlite3.Error as e:
        conn.close()
        print(f"An error occurred while adding TSV values: {e}")
        return False

keys_to_drop = ["EPH"]

def load_file(file_name: str) -> bool:
    """
    Load a single .tsv file and insert its data into the database, ensuring the header row is skipped.
    :param file_name: The path to the .tsv file.
    :returns: True if the file was processed successfully, False otherwise.
    """
    try:
        base_name = os.path.basename(file_name)
        # get rid of the "key_presses_" part
        user_id = os.path.splitext(base_name)[0].split('.')[0].removeprefix("key_presses_")
        user_id = int(user_id)

        with open(file_name, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t', quotechar=None)
            next(reader, None)

            # Convert data rows to list of dictionaries
            key_presses = []
            for row in reader:
                if len(row) == 6 and str(row[0]) not in keys_to_drop:
                    key_presses.append({
                        "key": str(row[0]),
                        "press_time": str(row[1]),
                        "duration": str(row[2]),
                        "accel_x": str(row[3]),
                        "accel_y": str(row[4]),
                        "accel_z": str(row[5]),
                    })

            # Insert data into the database
            if not add_tsv_values(key_presses, user_id):
                print(f"Failed to add data from file {file_name}.")
                return False

        print(f"File {file_name} processed successfully.")
        return True

    except Exception as e:
        print(f"An error occurred while processing file {file_name}: {e}")
        return False


def load_dir(dir_name: str) -> bool:
    """
    Load all .tsv files from the specified directory and insert their data into the database,
    ensuring that the header row is skipped.
    :param dir_name: The path to the directory containing .tsv files.
    :return: True if all files were processed successfully, False otherwise.
    """
    try:
        tsv_files = glob.glob(os.path.join(dir_name, '*.tsv'))
        if not tsv_files:
            print(f"No .tsv files found in directory '{dir_name}'.")
            return False

        for tsv_file in tsv_files:
            print(f"Processing file: {tsv_file}")
            if not load_file(tsv_file):
                print(f"Failed to process file {tsv_file}.")
                return False

        print("All files have been processed successfully.")
        return True

    except Exception as e:
        print(f"An error occurred while loading files from directory '{dir_name}': {e}")
        return False


def load_str(content: str, username: str, skip_header: bool = True) -> bool:
    """
    Process TSV content from a string, optionally skip the header row,
    and insert the data into the database.
    :param content: The TSV data as a string.
    :param username: The username associated with the data.
    :param skip_header: Whether to skip the first row (header) of the input content.
    :return: True if the content was processed successfully, False otherwise.
    """
    try:
        # Create a CSV reader from the string content
        tsv_reader = csv.reader(StringIO(content), delimiter='\t', quotechar=None)

        if skip_header:
            next(tsv_reader, None)

        # Convert data rows to a list of dictionaries
        key_presses = []
        for row in tsv_reader:
            if len(row) == 6:
                key_presses.append({
                    "key": str(row[0]),
                    "press_time": str(row[1]),
                    "duration": str(row[2]),
                    "accel_x": str(row[3]),
                    "accel_y": str(row[4]),
                    "accel_z": str(row[5]),
                })

        # Insert data into the database
        if not add_tsv_values(key_presses, username):
            print(f"Failed to add data for username {username}.")
            return False

        print(f"Content processed successfully for username {username}.")
        return True

    except Exception as e:
        print(f"An error occurred while processing content for username {username}: {e}")
        return False


if __name__ == '__main__':
    drop_table()
    create_table()
    load_dir("datasets/training")
