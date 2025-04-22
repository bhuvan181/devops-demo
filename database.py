"""Module for user authentication using SQLite and password hashing."""

import sqlite3
import hashlib


def create_user_table():
    """Creates the users table if it does not already exist."""
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()


def add_user(username, password):
    """Adds a new user with a hashed password to the database.

    Args:
        username (str): The user's username.
        password (str): The user's plaintext password.
    """
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_pw))
    conn.commit()
    conn.close()


def authenticate_user(username, password):
    """Authenticates a user by comparing the hashed password with the stored hash.

    Args:
        username (str): The user's username.
        password (str): The user's plaintext password.

    Returns:
        bool: True if authentication is successful, False otherwise.
    """
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, hashed_pw))
    result = c.fetchone()
    conn.close()
    return result is not None
