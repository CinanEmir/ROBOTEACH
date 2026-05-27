import sqlite3
import os

class DatabaseManager:
    def __init__(self, db_name="humanoid_ai.db"):
        self.db_name = db_name
        self.init_db()

    def init_db(self):
        """Veritabanı ve tabloları oluşturur."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_name TEXT NOT NULL,
                model_path TEXT,
                data_path TEXT,
                last_step INTEGER DEFAULT 0,
                status TEXT DEFAULT 'Beklemede',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()

    def add_task(self, name, model_p="", data_p="", step=0):
        """Yeni bir eğitim görevi ekler."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO tasks (task_name, model_path, data_path, last_step)
            VALUES (?, ?, ?, ?)
        ''', (name, model_p, data_p, step))
        conn.commit()
        conn.close()

    def get_all_tasks(self):
        """Tüm görevleri listede göstermek üzere çeker."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM tasks ORDER BY created_at DESC')
        rows = cursor.fetchall()
        conn.close()
        return rows

    def update_task_step(self, task_id, new_step):
        """Eğitim sırasında adımı günceller."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('UPDATE tasks SET last_step = ? WHERE id = ?', (new_step, task_id))
        conn.commit()
        conn.close()

    def delete_task(self, task_id):
        """Görevi siler."""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM tasks WHERE id = ?', (task_id,))
        conn.commit()
        conn.close()