"""
Attendance Details Page - View, filter, update, and delete attendance records
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox,
    QDateEdit, QLineEdit, QFrame, QMessageBox, QSizePolicy,
    QDialog, QFormLayout, QTimeEdit, QDialogButtonBox
)
from PySide6.QtCore import Qt, QDate, QTime
from PySide6.QtGui import QPixmap, QImage, QColor
from datetime import datetime, timedelta
import cv2
import os

# Import status constants from your database module
from app.db.database import DatabaseConfig, AttendanceStatus


# ─────────────────────────────────────────────────────────────────────────────
#  Status badge helpers
# ─────────────────────────────────────────────────────────────────────────────

# Colour map: status code → (background, text)
STATUS_COLORS = {
    'P':  ('#1a7a3c', '#6ee99a'),   # green  – Present
    'A':  ('#7a1a1a', '#f08080'),   # red    – Absent
    'L':  ('#7a5a1a', '#ffd080'),   # amber  – Late
    'LV': ('#1a4a7a', '#80c8f0'),   # blue   – On Leave
}

def _status_item(code: str) -> QTableWidgetItem:
    """Return a nicely coloured QTableWidgetItem for an attendance status."""
    label  = AttendanceStatus.label(code)           # e.g. "Present"
    display = f"{label}  "

    item = QTableWidgetItem(display)
    item.setTextAlignment(Qt.AlignCenter)

    bg, fg = STATUS_COLORS.get(code, ('#333', '#fff'))
    item.setBackground(QColor(bg))
    item.setForeground(QColor(fg))
    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
    return item


# ─────────────────────────────────────────────────────────────────────────────
#  Edit dialog
# ─────────────────────────────────────────────────────────────────────────────

class EditAttendanceDialog(QDialog):
    """Dialog for editing a single attendance record"""

    def __init__(self, record, parent=None):
        super().__init__(parent)
        self.record = record
        self.setWindowTitle(f"Edit Attendance — {record['name']}")
        self.setMinimumWidth(440)
        self.setStyleSheet("""
            QDialog { background-color: #1a1a1a; }
            QLabel  { color: #fff; font-size: 13px; }
            QLineEdit, QDateEdit, QTimeEdit, QComboBox {
                background-color: #2a2a2a;
                color: #fff;
                border: 1px solid #444;
                border-radius: 5px;
                padding: 6px 10px;
                min-height: 36px;
                font-size: 13px;
            }
            QLineEdit:focus, QDateEdit:focus, QTimeEdit:focus, QComboBox:focus {
                border: 1px solid #ffc107;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #fff;
                selection-background-color: #ffc107;
                selection-color: #111;
            }
            QDialogButtonBox QPushButton {
                min-width: 90px; min-height: 36px;
                border-radius: 5px; font-weight: bold; font-size: 13px;
            }
        """)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        header = QLabel(f"Editing record for  <b style='color:#ffc107'>{self.record['name']}</b>")
        header.setStyleSheet("color:#fff; font-size:15px;")
        layout.addWidget(header)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignRight)

        # ── Date ──────────────────────────────────────────────────
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        date_val = self.record.get('date', '')
        self.date_edit.setDate(
            QDate.fromString(str(date_val), "yyyy-MM-dd") if date_val else QDate.currentDate()
        )
        form.addRow("Date:", self.date_edit)

        # ── Time ──────────────────────────────────────────────────
        self.time_edit = QTimeEdit()
        self.time_edit.setDisplayFormat("HH:mm:ss")
        time_val = self.record.get('time', '00:00:00')
        self.time_edit.setTime(QTime.fromString(str(time_val), "HH:mm:ss"))
        form.addRow("Time:", self.time_edit)

        # ── Status ────────────────────────────────────────────────
        self.status_combo = QComboBox()
        for code, label in AttendanceStatus.LABELS.items():
            self.status_combo.addItem(f"{code} – {label}", code)

        # Select current status
        current_status = self.record.get('status') or AttendanceStatus.PRESENT
        idx = self.status_combo.findData(current_status)
        if idx >= 0:
            self.status_combo.setCurrentIndex(idx)
        form.addRow("Status:", self.status_combo)

        # ── Department ────────────────────────────────────────────
        self.dept_edit = QLineEdit(self.record.get('department') or '')
        self.dept_edit.setPlaceholderText("e.g. Engineering")
        form.addRow("Department:", self.dept_edit)

        # ── Role ──────────────────────────────────────────────────
        self.role_edit = QLineEdit(self.record.get('role') or '')
        self.role_edit.setPlaceholderText("e.g. Employee")
        form.addRow("Role:", self.role_edit)

        # ── Confidence (read-only info) ───────────────────────────
        conf_score = self.record.get('confidence_score')
        conf_str   = f"{conf_score:.4f}" if conf_score else 'N/A'
        conf_label = QLabel(conf_str)
        conf_label.setStyleSheet("color: #94a3b8;")
        form.addRow("Confidence (read-only):", conf_label)

        layout.addLayout(form)

        # ── Buttons ───────────────────────────────────────────────
        btn_box    = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        save_btn   = btn_box.button(QDialogButtonBox.Save)
        cancel_btn = btn_box.button(QDialogButtonBox.Cancel)
        save_btn.setStyleSheet("background-color: #ffc107; color: #111;")
        cancel_btn.setStyleSheet("background-color: #444; color: #fff;")
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_updated_values(self):
        return {
            'date':       self.date_edit.date().toString("yyyy-MM-dd"),
            'time':       self.time_edit.time().toString("HH:mm:ss"),
            'status':     self.status_combo.currentData(),
            'department': self.dept_edit.text().strip() or None,
            'role':       self.role_edit.text().strip() or 'Employee',
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Main page
# ─────────────────────────────────────────────────────────────────────────────

class AttendancePage(QWidget):
    """Page for viewing attendance records"""

    # Column indices — easy to change in one place
    COL_USER_ID    = 0
    COL_NAME       = 1
    COL_DATE       = 2
    COL_TIME       = 3
    COL_STATUS     = 4
    COL_DEPARTMENT = 5
    COL_ROLE       = 6
    COL_CONFIDENCE = 7
    COL_IMAGE      = 8
    COL_EDIT       = 9
    COL_DELETE     = 10

    HEADERS = [
        "User ID", "Name", "Date", "Time",
        "Status",                           # ← NEW
        "Department", "Role", "Confidence",
        "Image", "Edit", "Delete"
    ]

    def __init__(self, attendance_manager, user_manager, parent=None):
        super().__init__(parent)
        self.attendance_manager = attendance_manager
        self.user_manager       = user_manager
        self.current_records    = []

        self.setup_ui()
        self.load_attendance_data()

    # ─────────────────────────────────────────────────────────────
    #  UI construction
    # ─────────────────────────────────────────────────────────────

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        title = QLabel("Attendance Records")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #fff;")
        layout.addWidget(title)

        # ── Filters ───────────────────────────────────────────────
        filters_frame = QFrame()
        filters_frame.setObjectName("card")
        filters_layout = QVBoxLayout(filters_frame)
        filters_layout.setSpacing(15)

        filter_title = QLabel("Filters")
        filter_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #fff;")
        filters_layout.addWidget(filter_title)

        controls_layout = QHBoxLayout()

        user_layout = QVBoxLayout()
        user_layout.addWidget(QLabel("User:"))
        self.user_combo = QComboBox()
        self.user_combo.setMinimumHeight(40)
        self.user_combo.addItem("All Users", None)
        user_layout.addWidget(self.user_combo)
        controls_layout.addLayout(user_layout)

        date_layout = QVBoxLayout()
        date_layout.addWidget(QLabel("Select Date:"))
        self.select_date = QDateEdit()
        self.select_date.setCalendarPopup(True)
        self.select_date.setDate(QDate.currentDate())
        self.select_date.setMinimumHeight(40)
        date_layout.addWidget(self.select_date)
        controls_layout.addLayout(date_layout)

        # Status filter
        status_layout = QVBoxLayout()
        status_layout.addWidget(QLabel("Status:"))
        self.status_filter = QComboBox()
        self.status_filter.setMinimumHeight(40)
        self.status_filter.addItem("All", None)
        for code, label in AttendanceStatus.LABELS.items():
            self.status_filter.addItem(f"{code} – {label}", code)
        status_layout.addWidget(self.status_filter)
        controls_layout.addLayout(status_layout)

        quick_layout = QVBoxLayout()
        quick_layout.addWidget(QLabel("Quick Select:"))
        quick_btn_layout = QHBoxLayout()

        today_btn = QPushButton("Today")
        today_btn.setMinimumHeight(40)
        today_btn.clicked.connect(lambda: self.select_date.setDate(QDate.currentDate()))
        quick_btn_layout.addWidget(today_btn)

        yesterday_btn = QPushButton("Yesterday")
        yesterday_btn.setMinimumHeight(40)
        yesterday_btn.clicked.connect(lambda: self.select_date.setDate(QDate.currentDate().addDays(-1)))
        quick_btn_layout.addWidget(yesterday_btn)

        quick_layout.addLayout(quick_btn_layout)
        controls_layout.addLayout(quick_layout)

        search_btn_layout = QVBoxLayout()
        search_btn_layout.addWidget(QLabel(" "))
        self.search_btn = QPushButton("Apply Filters")
        self.search_btn.setObjectName("addApplicationButton")
        self.search_btn.setMinimumHeight(40)
        self.search_btn.clicked.connect(self.load_attendance_data)
        search_btn_layout.addWidget(self.search_btn)
        controls_layout.addLayout(search_btn_layout)

        export_btn_layout = QVBoxLayout()
        export_btn_layout.addWidget(QLabel(" "))
        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setMinimumHeight(40)
        self.export_btn.clicked.connect(self.export_to_csv)
        export_btn_layout.addWidget(self.export_btn)
        controls_layout.addLayout(export_btn_layout)

        controls_layout.addStretch()
        filters_layout.addLayout(controls_layout)
        layout.addWidget(filters_frame)

        # ── Stats ─────────────────────────────────────────────────
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)

        self.total_label   = self.create_stat_card("Total Records",  "0")
        self.present_label = self.create_stat_card("Present Today",  "0")
        self.week_label    = self.create_stat_card("This Week",      "0")

        stats_layout.addWidget(self.total_label)
        stats_layout.addWidget(self.present_label)
        stats_layout.addWidget(self.week_label)
        stats_layout.addStretch()
        layout.addLayout(stats_layout)

        # ── Table ─────────────────────────────────────────────────
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.HEADERS))
        self.table.setHorizontalHeaderLabels(self.HEADERS)

        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #222222;
                border: 1px solid #333333;
                border-radius: 8px;
                color: #fff;
                gridline-color: #333333;
            }
            QTableWidget::item          { padding: 8px; }
            QTableWidget::item:selected { background-color: #ffc107; color: #111; }
            QHeaderView::section {
                background-color: #2a2a2a;
                color: #fff;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
        """)

        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.Stretch)

        # Fix the action columns to compact widths
        for col in (self.COL_IMAGE, self.COL_EDIT, self.COL_DELETE):
            hdr.setSectionResizeMode(col, QHeaderView.Fixed)
        self.table.setColumnWidth(self.COL_IMAGE,  90)
        self.table.setColumnWidth(self.COL_EDIT,   70)
        self.table.setColumnWidth(self.COL_DELETE, 80)

        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        layout.addWidget(self.table)
        self.load_users()

    def create_stat_card(self, title, value):
        frame = QFrame()
        frame.setObjectName("card")
        frame.setMinimumHeight(80)

        card_layout = QVBoxLayout(frame)

        value_label = QLabel(value)
        value_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #ffc107;")
        value_label.setAlignment(Qt.AlignCenter)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 14px; color: #94a3b8;")
        title_label.setAlignment(Qt.AlignCenter)

        card_layout.addWidget(value_label)
        card_layout.addWidget(title_label)
        return frame

    def load_users(self):
        users = self.user_manager.get_all_users()
        self.user_combo.clear()
        self.user_combo.addItem("All Users", None)
        for user in users:
            self.user_combo.addItem(
                f"{user['name']} ({user['user_id']})",
                user['user_id']
            )

    # ─────────────────────────────────────────────────────────────
    #  Helper: button widget for a cell
    # ─────────────────────────────────────────────────────────────

    def _make_cell_button(self, text, color, hover_color, callback):
        container = QWidget()
        container.setStyleSheet("background-color: transparent;")
        btn_layout = QHBoxLayout(container)
        btn_layout.setContentsMargins(4, 4, 4, 4)
        btn_layout.setAlignment(Qt.AlignCenter)

        btn = QPushButton(text)
        btn.setFixedSize(60, 32)
        btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: #fff;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover   {{ background-color: {hover_color}; }}
            QPushButton:disabled {{ background-color: #555; color: #888; }}
        """)
        btn.clicked.connect(callback)
        btn_layout.addWidget(btn)
        return container

    # ─────────────────────────────────────────────────────────────
    #  Load / refresh table
    # ─────────────────────────────────────────────────────────────

    def load_attendance_data(self):
        user_id       = self.user_combo.currentData()
        selected_date = self.select_date.date().toString("yyyy-MM-dd")
        status_filter = self.status_filter.currentData()   # None = all

        print(f"[DEBUG] Loading attendance: date={selected_date}, user={user_id}, status={status_filter}")

        records = self.attendance_manager.get_attendance_records(
            start_date=selected_date,
            end_date=selected_date,
            user_id=user_id
        )

        # Apply optional status filter client-side
        if status_filter:
            records = [r for r in records if r.get('status') == status_filter]

        self.current_records = records
        print(f"[DEBUG] Displaying {len(records)} records")

        self.table.setRowCount(len(records))

        for row, record in enumerate(records):
            self.table.setRowHeight(row, 75)

            # ── Text columns ──────────────────────────────────────
            self.table.setItem(row, self.COL_USER_ID, QTableWidgetItem(str(record['user_id'])))
            self.table.setItem(row, self.COL_NAME,    QTableWidgetItem(str(record['name'])))

            date_value = record['date']
            date_str   = date_value if isinstance(date_value, str) else (
                date_value.strftime('%Y-%m-%d') if hasattr(date_value, 'strftime') else str(date_value)
            )
            self.table.setItem(row, self.COL_DATE, QTableWidgetItem(date_str))
            self.table.setItem(row, self.COL_TIME, QTableWidgetItem(str(record['time'])))

            # ── Status badge ──────────────────────────────────────
            status_code = record.get('status') or AttendanceStatus.PRESENT
            self.table.setItem(row, self.COL_STATUS, _status_item(status_code))

            # ── Department / Role ─────────────────────────────────
            self.table.setItem(row, self.COL_DEPARTMENT,
                               QTableWidgetItem(record.get('department') or 'N/A'))
            self.table.setItem(row, self.COL_ROLE,
                               QTableWidgetItem(record.get('role') or 'Employee'))

            # ── Confidence ────────────────────────────────────────
            conf_score = record.get('confidence_score')
            conf       = f"{conf_score:.2%}" if conf_score else 'N/A'
            self.table.setItem(row, self.COL_CONFIDENCE, QTableWidgetItem(conf))

            # ── Image button ──────────────────────────────────────
            img_container = QWidget()
            img_container.setStyleSheet("background-color: transparent;")
            img_btn_layout = QHBoxLayout(img_container)
            img_btn_layout.setContentsMargins(0, 0, 0, 0)
            img_btn_layout.setAlignment(Qt.AlignCenter)

            img_btn = QPushButton("View")
            img_btn.setFixedSize(70, 35)
            img_btn.setStyleSheet("""
                QPushButton {
                    background-color: #304FFE; color: #fff;
                    border: none; border-radius: 5px; font-weight: bold;
                }
                QPushButton:hover    { background-color: #5C6BC0; }
                QPushButton:disabled { background-color: #555; color: #888; }
            """)
            if record.get('image_path') and os.path.exists(record['image_path']):
                img_btn.clicked.connect(lambda checked, r=record: self.show_image(r))
            else:
                img_btn.setEnabled(False)
                img_btn.setText("N/A")
            img_btn_layout.addWidget(img_btn)
            self.table.setCellWidget(row, self.COL_IMAGE, img_container)

            # ── Edit button ───────────────────────────────────────
            self.table.setCellWidget(row, self.COL_EDIT,
                self._make_cell_button("Edit", "#28a745", "#218838",
                                       lambda checked, r=record: self.edit_record(r)))

            # ── Delete button ─────────────────────────────────────
            self.table.setCellWidget(row, self.COL_DELETE,
                self._make_cell_button("Delete", "#dc3545", "#c82333",
                                       lambda checked, r=record: self.delete_record(r)))

        self.update_statistics()

    # ─────────────────────────────────────────────────────────────
    #  Edit record
    # ─────────────────────────────────────────────────────────────

    def edit_record(self, record):
        dialog = EditAttendanceDialog(record, self)
        if dialog.exec() != QDialog.Accepted:
            return

        updated = dialog.get_updated_values()

        conn = DatabaseConfig.get_connection()
        if not conn:
            QMessageBox.critical(self, "Error", "Could not connect to the database.")
            return

        try:
            cursor = conn.cursor()

            # Update attendance row: date, time, status
            cursor.execute("""
                UPDATE attendance
                SET date = ?, time = ?, status = ?
                WHERE id = ?
            """, (updated['date'], updated['time'], updated['status'], record['id']))

            # Update user row: department, role
            cursor.execute("""
                UPDATE users
                SET department = ?, role = ?, updated_at = ?
                WHERE user_id = ?
            """, (updated['department'], updated['role'],
                  datetime.now().isoformat(), record['user_id']))

            conn.commit()
            cursor.close()
            conn.close()

            # Keep today's cache in sync when date changes
            if hasattr(self.attendance_manager, 'today_attendance_cache'):
                today = datetime.now().date().isoformat()
                if record['date'] == today and updated['date'] != today:
                    self.attendance_manager.today_attendance_cache.discard(record['user_id'])
                elif updated['date'] == today:
                    self.attendance_manager.today_attendance_cache.add(record['user_id'])

            QMessageBox.information(
                self, "Updated",
                f"Record for <b>{record['name']}</b> updated successfully."
            )
            self.load_attendance_data()

        except Exception as e:
            QMessageBox.critical(self, "Update Failed", f"Could not update record:\n{e}")
            import traceback; traceback.print_exc()

    # ─────────────────────────────────────────────────────────────
    #  Delete record
    # ─────────────────────────────────────────────────────────────

    def delete_record(self, record):
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Delete attendance record for\n"
            f"<b>{record['name']}</b> on <b>{record['date']}</b>?\n\n"
            "This action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        conn = DatabaseConfig.get_connection()
        if not conn:
            QMessageBox.critical(self, "Error", "Could not connect to the database.")
            return

        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM attendance WHERE id = ?", (record['id'],))
            conn.commit()
            cursor.close()
            conn.close()

            if hasattr(self.attendance_manager, 'today_attendance_cache'):
                today = datetime.now().date().isoformat()
                if record['date'] == today:
                    self.attendance_manager.today_attendance_cache.discard(record['user_id'])

            QMessageBox.information(
                self, "Deleted",
                f"Record for {record['name']} on {record['date']} deleted."
            )
            self.load_attendance_data()

        except Exception as e:
            QMessageBox.critical(self, "Delete Failed", f"Could not delete record:\n{e}")
            import traceback; traceback.print_exc()

    # ─────────────────────────────────────────────────────────────
    #  Statistics
    # ─────────────────────────────────────────────────────────────

    def update_statistics(self):
        total = len(self.current_records)

        today_str    = datetime.now().date().isoformat()
        today_count  = len(self.attendance_manager.get_attendance_records(
            start_date=today_str, end_date=today_str
        ))

        week_start  = (datetime.now().date() - timedelta(days=datetime.now().date().weekday())).isoformat()
        week_count  = len(self.attendance_manager.get_attendance_records(
            start_date=week_start, end_date=today_str
        ))

        self.total_label.findChildren(QLabel)[0].setText(str(total))
        self.present_label.findChildren(QLabel)[0].setText(str(today_count))
        self.week_label.findChildren(QLabel)[0].setText(str(week_count))

    # ─────────────────────────────────────────────────────────────
    #  Image viewer
    # ─────────────────────────────────────────────────────────────

    def show_image(self, record):
        img_path = record.get('image_path')
        if not img_path or not os.path.exists(img_path):
            QMessageBox.warning(self, "Image Not Found", "Attendance image not available.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Attendance Image - {record['name']}")
        dialog.setMinimumSize(500, 600)

        layout = QVBoxLayout(dialog)

        info_frame = QFrame()
        info_frame.setStyleSheet("QFrame { background-color: #2a2a2a; border-radius: 8px; padding: 15px; }")
        info_layout = QVBoxLayout(info_frame)

        conf_score  = record.get('confidence_score', 0)
        status_code = record.get('status') or AttendanceStatus.PRESENT
        info_text = (
            f"<b>Name:</b> {record['name']}<br>"
            f"<b>User ID:</b> {record['user_id']}<br>"
            f"<b>Date:</b> {record['date']}<br>"
            f"<b>Time:</b> {record['time']}<br>"
            f"<b>Status:</b> {status_code} – {AttendanceStatus.label(status_code)}<br>"
            f"<b>Department:</b> {record.get('department', 'N/A')}<br>"
            f"<b>Confidence:</b> {conf_score:.2%}" if conf_score else "N/A"
        )

        info_label = QLabel(info_text)
        info_label.setStyleSheet("color: #fff; font-size: 14px;")
        info_layout.addWidget(info_label)
        layout.addWidget(info_frame)

        img_label = QLabel()
        img       = cv2.imread(img_path)

        if img is not None:
            img_rgb        = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch       = img_rgb.shape
            bytes_per_line = ch * w
            q_image        = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap         = QPixmap.fromImage(q_image)
            scaled         = pixmap.scaled(450, 450, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            img_label.setPixmap(scaled)
        else:
            img_label.setText("Failed to load image")
            img_label.setStyleSheet("color: #ff0000;")

        img_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(img_label)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.exec()

    # ─────────────────────────────────────────────────────────────
    #  CSV export
    # ─────────────────────────────────────────────────────────────

    def export_to_csv(self):
        if not self.current_records:
            QMessageBox.warning(self, "No Data", "No records to export.")
            return

        from PySide6.QtWidgets import QFileDialog
        import csv

        default_filename = f"attendance_{self.select_date.date().toString('yyyy-MM-dd')}.csv"
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Attendance", default_filename, "CSV Files (*.csv)"
        )
        if not file_path:
            return

        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "User ID", "Name", "Date", "Time", "Status",
                    "Department", "Role", "Confidence", "Image Path"
                ])
                for record in self.current_records:
                    conf_score  = record.get('confidence_score')
                    conf_str    = f"{conf_score:.4f}" if conf_score else 'N/A'
                    status_code = record.get('status') or AttendanceStatus.PRESENT
                    writer.writerow([
                        record['user_id'],
                        record['name'],
                        record['date'],
                        record['time'],
                        f"{status_code} – {AttendanceStatus.label(status_code)}",
                        record.get('department', 'N/A'),
                        record.get('role', 'Employee'),
                        conf_str,
                        record.get('image_path', '')
                    ])

            QMessageBox.information(
                self, "Success",
                f"Exported {len(self.current_records)} records.\n\nFile: {file_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Failed to export:\n{str(e)}")