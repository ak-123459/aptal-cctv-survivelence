"""
Database Configuration and Table Creation
"""
import mysql.connector
from mysql.connector import Error
from datetime import datetime
import os


class DatabaseConfig:
    """Database configuration and connection manager"""

    # Database Configuration
    DB_CONFIG = {
        'host': 'localhost',
        'user': 'root',
        'password': 'your_password',  # Change this
        'database': 'face_recognition_db'
    }

    @staticmethod
    def create_database():
        """Create database if not exists"""
        try:
            conn = mysql.connector.connect(
                host=DatabaseConfig.DB_CONFIG['host'],
                user=DatabaseConfig.DB_CONFIG['user'],
                password=DatabaseConfig.DB_CONFIG['password']
            )
            cursor = conn.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DatabaseConfig.DB_CONFIG['database']}")
            print("[INFO] Database created/verified successfully")
            cursor.close()
            conn.close()
        except Error as e:
            print(f"[ERROR] Database creation failed: {e}")

    @staticmethod
    def get_connection():
        """Get database connection"""
        try:
            conn = mysql.connector.connect(**DatabaseConfig.DB_CONFIG)
            return conn
        except Error as e:
            print(f"[ERROR] Database connection failed: {e}")
            return None

    @staticmethod
    def initialize_tables():
        """Create all required tables"""
        conn = DatabaseConfig.get_connection()
        if not conn:
            return

        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                email VARCHAR(100),
                phone VARCHAR(20),
                department VARCHAR(50),
                role VARCHAR(30),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_name (name),
                INDEX idx_role (role)
            )
        """)

        # Attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(50) NOT NULL,
                date DATE NOT NULL,
                time TIME NOT NULL,
                image_path VARCHAR(255),
                confidence_score FLOAT,
                camera_id VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_attendance (user_id, date),
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                INDEX idx_date (date),
                INDEX idx_user_date (user_id, date)
            )
        """)

        # Cameras table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cameras (
                camera_id VARCHAR(50) PRIMARY KEY,
                camera_name VARCHAR(100) NOT NULL,
                camera_source VARCHAR(255) NOT NULL,
                status VARCHAR(20) DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()
        print("[INFO] Database tables initialized successfully")


class AttendanceManager:
    """Manages attendance records with optimization"""

    def __init__(self):
        self.today_attendance_cache = set()  # Cache for today's attendance
        self.load_today_cache()

    def load_today_cache(self):
        """Load today's attendance into cache"""
        conn = DatabaseConfig.get_connection()
        if not conn:
            return

        try:
            cursor = conn.cursor()
            today = datetime.now().date()
            cursor.execute("""
                SELECT user_id FROM attendance 
                WHERE date = %s
            """, (today,))

            self.today_attendance_cache = {row[0] for row in cursor.fetchall()}
            cursor.close()
            conn.close()
        except Error as e:
            print(f"[ERROR] Loading attendance cache: {e}")

    def mark_attendance(self, user_id, confidence_score, image_path=None, camera_id=None):
        """
        Mark attendance - optimized to prevent duplicate entries
        Returns: True if inserted, False if already exists
        """
        # Check cache first (no DB call)
        if user_id in self.today_attendance_cache:
            return False

        conn = DatabaseConfig.get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()
            today = datetime.now().date()
            current_time = datetime.now().time()

            # Try to insert
            cursor.execute("""
                INSERT INTO attendance (user_id, date, time, image_path, confidence_score, camera_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (user_id, today, current_time, image_path, confidence_score, camera_id))

            conn.commit()

            # Add to cache
            self.today_attendance_cache.add(user_id)

            cursor.close()
            conn.close()
            return True

        except mysql.connector.IntegrityError:
            # Duplicate entry - already marked
            self.today_attendance_cache.add(user_id)
            return False
        except Error as e:
            print(f"[ERROR] Marking attendance: {e}")
            return False

    def get_attendance_records(self, start_date=None, end_date=None, user_id=None):
        """Get attendance records with filters"""
        conn = DatabaseConfig.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor(dictionary=True)

            query = """
                SELECT a.*, u.name, u.department 
                FROM attendance a
                JOIN users u ON a.user_id = u.user_id
                WHERE 1=1
            """
            params = []

            if start_date:
                query += " AND a.date >= %s"
                params.append(start_date)

            if end_date:
                query += " AND a.date <= %s"
                params.append(end_date)

            if user_id:
                query += " AND a.user_id = %s"
                params.append(user_id)

            query += " ORDER BY a.date DESC, a.time DESC"

            cursor.execute(query, params)
            records = cursor.fetchall()

            cursor.close()
            conn.close()
            return records

        except Error as e:
            print(f"[ERROR] Getting attendance records: {e}")
            return []


class UserManager:
    """Manages user records"""

    @staticmethod
    def add_user(user_id, name, email=None, phone=None, department=None, role='Employee'):
        """Add new user to database"""
        conn = DatabaseConfig.get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (user_id, name, email, phone, department, role)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (user_id, name, email, phone, department, role))

            conn.commit()
            cursor.close()
            conn.close()
            return True

        except mysql.connector.IntegrityError:
            print(f"[WARN] User {user_id} already exists")
            return False
        except Error as e:
            print(f"[ERROR] Adding user: {e}")
            return False

    @staticmethod
    def get_user(user_id):
        """Get user details"""
        conn = DatabaseConfig.get_connection()
        if not conn:
            return None

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            user = cursor.fetchone()
            cursor.close()
            conn.close()
            return user
        except Error as e:
            print(f"[ERROR] Getting user: {e}")
            return None

    @staticmethod
    def get_all_users():
        """Get all users"""
        conn = DatabaseConfig.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users ORDER BY name")
            users = cursor.fetchall()
            cursor.close()
            conn.close()
            return users
        except Error as e:
            print(f"[ERROR] Getting users: {e}")
            return []

    @staticmethod
    def update_user(user_id, **kwargs):
        """Update user details"""
        conn = DatabaseConfig.get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()

            # Build dynamic update query
            fields = []
            values = []
            for key, value in kwargs.items():
                if key != 'user_id' and value is not None:
                    fields.append(f"{key} = %s")
                    values.append(value)

            if not fields:
                return False

            values.append(user_id)
            query = f"UPDATE users SET {', '.join(fields)} WHERE user_id = %s"

            cursor.execute(query, values)
            conn.commit()

            cursor.close()
            conn.close()
            return True

        except Error as e:
            print(f"[ERROR] Updating user: {e}")
            return False


# Initialize database on import
def init_database():
    """Initialize database and tables"""
    DatabaseConfig.create_database()
    DatabaseConfig.initialize_tables()
    print("[INFO] Database initialization complete")