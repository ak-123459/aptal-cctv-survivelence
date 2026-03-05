"""
Registered Users Page - View and manage all registered users
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QLineEdit,
    QFrame, QMessageBox, QComboBox, QDialog, QFormLayout,
    QDialogButtonBox, QTextEdit
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage, QColor, QFont
import cv2
import os



class UserDetailsDialog(QDialog):
    """Dialog to show detailed user information"""

    def __init__(self, user_data, parent=None):
        super().__init__(parent)
        self.user_data = user_data

        # Get name safely
        def get_value(key, index, default='N/A'):
            if isinstance(user_data, dict):
                return user_data.get(key, default) or default
            else:
                try:
                    return user_data[index] or default
                except (IndexError, TypeError):
                    return default

        user_name = get_value('name', 1)
        self.setWindowTitle(f"User Details - {user_name}")
        self.setMinimumSize(550, 650)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # User Image Section
        image_frame = QFrame()
        image_frame.setObjectName("card")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(20, 20, 20, 20)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(200, 200)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 3px solid #ffc107;
                border-radius: 100px;
                background-color: #333333;
            }
        """)

        # Load user image
        self.load_user_image()

        image_layout.addWidget(self.image_label, 0, Qt.AlignCenter)
        layout.addWidget(image_frame)

        # Details Section
        details_frame = QFrame()
        details_frame.setObjectName("card")
        details_layout = QVBoxLayout(details_frame)
        details_layout.setContentsMargins(20, 20, 20, 20)
        details_layout.setSpacing(12)

        # Helper function to safely get values
        def get_value(key, index, default='N/A'):
            if isinstance(self.user_data, dict):
                return self.user_data.get(key, default) or default
            else:
                try:
                    return self.user_data[index] or default
                except (IndexError, TypeError):
                    return default

        # Create detail rows
        fields = [
            ("User ID:", get_value('user_id', 0)),
            ("Name:", get_value('name', 1)),
            ("Email:", get_value('email', 2)),
            ("Phone:", get_value('phone', 3)),
            ("Department:", get_value('department', 4)),
            ("Role:", get_value('role', 5, 'Employee')),
            ("Created:", get_value('created_at', 6)),
        ]

        for label_text, value_text in fields:
            row_layout = QHBoxLayout()
            row_layout.setSpacing(10)

            label = QLabel(label_text)
            label.setStyleSheet("color: #ffc107; font-weight: bold; font-size: 13px;")
            label.setMinimumWidth(100)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

            value = QLabel(str(value_text))
            value.setStyleSheet("color: #fff; font-size: 14px;")
            value.setWordWrap(True)

            row_layout.addWidget(label)
            row_layout.addWidget(value, 1)

            details_layout.addLayout(row_layout)

        layout.addWidget(details_frame)

        # Action buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        edit_btn = QPushButton("✏ Edit User")
        edit_btn.setMinimumHeight(40)
        edit_btn.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: #111;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ffb300;
            }
        """)
        edit_btn.clicked.connect(self.edit_user)

        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(40)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: #fff;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        close_btn.clicked.connect(self.close)

        button_layout.addWidget(edit_btn)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def load_user_image(self):
        """Load user's registered face image"""
        # Handle both dict and Row objects
        user_id = self.user_data['user_id'] if isinstance(self.user_data, dict) else self.user_data[0]

        # Try multiple possible image locations
        possible_paths = [
            f"registered_faces/{user_id}.jpg",
            f"registered_faces/{user_id}/face_1.jpg",
            f"registered_faces/{user_id}/face_0.jpg"
        ]

        for image_path in possible_paths:
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = img_rgb.shape
                    bytes_per_line = ch * w
                    q_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(q_image)
                    scaled = pixmap.scaled(190, 190, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    self.image_label.setPixmap(scaled)
                    return

        # Default placeholder
        self.image_label.setText("No Image")
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #333333;
                border-radius: 100px;
                background-color: #222222;
                color: #666666;
                font-size: 14px;
            }
        """)

    def edit_user(self):
        """Open edit dialog"""
        QMessageBox.information(
            self, "Edit User",
            "Edit user functionality will be implemented here."
        )


class EditUserDialog(QDialog):
    """Dialog to edit user information"""

    def __init__(self, user_data, user_manager, parent=None):
        super().__init__(parent)
        self.user_data = user_data
        self.user_manager = user_manager

        # Helper function to safely get values
        def get_value(key, index, default=''):
            if isinstance(user_data, dict):
                return user_data.get(key, default) or default
            else:
                try:
                    return user_data[index] or default
                except (IndexError, TypeError):
                    return default

        user_name = get_value('name', 1)
        self.setWindowTitle(f"Edit User - {user_name}")
        self.setMinimumWidth(500)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        form_frame = QFrame()
        form_frame.setObjectName("card")
        form_layout = QVBoxLayout(form_frame)
        form_layout.setContentsMargins(20, 20, 20, 20)
        form_layout.setSpacing(15)

        # Helper function to safely get values
        def get_value(key, index, default=''):
            if isinstance(self.user_data, dict):
                return self.user_data.get(key, default) or default
            else:
                try:
                    return self.user_data[index] or default
                except (IndexError, TypeError):
                    return default

        user_id = get_value('user_id', 0)

        # User ID (read-only)
        id_layout = QHBoxLayout()
        id_label = QLabel("User ID:")
        id_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        id_value = QLabel(str(user_id))
        id_value.setStyleSheet("color: #fff;")
        id_layout.addWidget(id_label)
        id_layout.addWidget(id_value)
        id_layout.addStretch()
        form_layout.addLayout(id_layout)

        # Name field
        name_label = QLabel("Name: *")
        name_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        form_layout.addWidget(name_label)
        self.name_input = QLineEdit(get_value('name', 1))
        self.name_input.setMinimumHeight(35)
        form_layout.addWidget(self.name_input)

        # Email field
        email_label = QLabel("Email:")
        email_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        form_layout.addWidget(email_label)
        self.email_input = QLineEdit(get_value('email', 2))
        self.email_input.setMinimumHeight(35)
        form_layout.addWidget(self.email_input)

        # Phone field
        phone_label = QLabel("Phone:")
        phone_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        form_layout.addWidget(phone_label)
        self.phone_input = QLineEdit(get_value('phone', 3))
        self.phone_input.setMinimumHeight(35)
        form_layout.addWidget(self.phone_input)

        # Department field
        dept_label = QLabel("Department:")
        dept_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        form_layout.addWidget(dept_label)
        self.department_input = QLineEdit(get_value('department', 4))
        self.department_input.setMinimumHeight(35)
        form_layout.addWidget(self.department_input)

        # Role field
        role_label = QLabel("Role:")
        role_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        form_layout.addWidget(role_label)
        self.role_combo = QComboBox()
        self.role_combo.setMinimumHeight(35)
        self.role_combo.addItems(["Employee", "Manager", "Admin", "Intern"])
        current_role = get_value('role', 5, 'Employee')
        index = self.role_combo.findText(current_role)
        if index >= 0:
            self.role_combo.setCurrentIndex(index)
        form_layout.addWidget(self.role_combo)

        layout.addWidget(form_frame)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        save_btn = QPushButton("💾 Save Changes")
        save_btn.setMinimumHeight(40)
        save_btn.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
        """)



        save_btn.clicked.connect(self.save_changes)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setMinimumHeight(40)
        cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: white;
                border: none;
                border-radius: 5px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def save_changes(self):
        """Save user changes to database"""
        name = self.name_input.text().strip()

        if not name:
            QMessageBox.warning(self, "Validation Error", "Name is required!")
            return

        # Get user_id
        if isinstance(self.user_data, dict):
            user_id = self.user_data['user_id']
        else:
            user_id = self.user_data[0]

        try:
            success = self.user_manager.update_user(
                user_id,
                name=name,
                email=self.email_input.text().strip() or None,
                phone=self.phone_input.text().strip() or None,
                department=self.department_input.text().strip() or None,
                role=self.role_combo.currentText()
            )

            if success:
                # Show detailed success message
                msg_box = QMessageBox(self)
                msg_box.setWindowTitle("✅ User Updated Successfully")
                msg_box.setIcon(QMessageBox.Information)

                details = (
                    f"<b>User has been updated successfully!</b><br><br>"
                    f"<table cellpadding='5' style='color: #333;'>"
                    f"<tr><td><b>User ID:</b></td><td>{user_id}</td></tr>"
                    f"<tr><td><b>Name:</b></td><td>{name}</td></tr>"
                    f"<tr><td><b>Email:</b></td><td>{self.email_input.text().strip() or 'N/A'}</td></tr>"
                    f"<tr><td><b>Phone:</b></td><td>{self.phone_input.text().strip() or 'N/A'}</td></tr>"
                    f"<tr><td><b>Department:</b></td><td>{self.department_input.text().strip() or 'N/A'}</td></tr>"
                    f"<tr><td><b>Role:</b></td><td>{self.role_combo.currentText()}</td></tr>"
                    f"</table>"
                )

                msg_box.setText(details)
                msg_box.setStandardButtons(QMessageBox.Ok)
                msg_box.exec()

                self.accept()
            else:
                QMessageBox.warning(
                    self, "Error",
                    "Failed to update user. Please try again."
                )

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"An error occurred: {str(e)}"
            )


class UsersPage(QWidget):
    """Page for viewing all registered users"""

    def __init__(self, user_manager, face_registration=None, parent=None):
        super().__init__(parent)
        self.user_manager = user_manager
        self.face_registration = face_registration  # Reference to FaceRegistration instance
        self.all_users = []
        self.filtered_users = []

        self.setup_ui()
        self.load_users()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(20)

        # Header Section
        header_layout = QHBoxLayout()

        title = QLabel("Registered Users")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #fff;")
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setFixedWidth(120)
        refresh_btn.clicked.connect(self.load_users)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Stats Section
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(15)

        self.total_users_card = self.create_stat_card("Total Users", "0", "#ffc107")
        self.employees_card = self.create_stat_card("Employees", "0", "#3b82f6")
        self.managers_card = self.create_stat_card("Managers", "0", "#10b981")
        self.admins_card = self.create_stat_card("Admins", "0", "#ef4444")

        stats_layout.addWidget(self.total_users_card)
        stats_layout.addWidget(self.employees_card)
        stats_layout.addWidget(self.managers_card)
        stats_layout.addWidget(self.admins_card)
        stats_layout.addStretch()

        layout.addLayout(stats_layout)

        # Filter Section
        filter_frame = QFrame()
        filter_frame.setObjectName("card")
        filter_layout = QHBoxLayout(filter_frame)
        filter_layout.setSpacing(15)

        # Search bar
        search_label = QLabel("Search:")
        search_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        filter_layout.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search by name, ID, email, department...")
        self.search_input.setMinimumHeight(40)
        self.search_input.textChanged.connect(self.filter_users)
        filter_layout.addWidget(self.search_input, 2)

        # Role filter
        role_label = QLabel("Role:")
        role_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        filter_layout.addWidget(role_label)

        self.role_filter = QComboBox()
        self.role_filter.setMinimumHeight(40)
        self.role_filter.addItems(["All Roles", "Employee", "Manager", "Admin", "Intern"])
        self.role_filter.currentTextChanged.connect(self.filter_users)
        filter_layout.addWidget(self.role_filter)

        # Department filter
        dept_label = QLabel("Department:")
        dept_label.setStyleSheet("color: #ffc107; font-weight: bold;")
        filter_layout.addWidget(dept_label)

        self.dept_filter = QComboBox()
        self.dept_filter.setMinimumHeight(40)
        self.dept_filter.addItem("All Departments")
        self.dept_filter.currentTextChanged.connect(self.filter_users)
        filter_layout.addWidget(self.dept_filter)

        layout.addWidget(filter_frame)

        # Results count
        self.results_label = QLabel("Showing 0 users")
        self.results_label.setStyleSheet("color: #94a3b8; font-size: 13px;")
        layout.addWidget(self.results_label)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "User ID", "Name", "Email", "Phone",
            "Department", "Role", "Created", "Actions"
        ])

        # Style table
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #222222;
                border: 1px solid #333333;
                border-radius: 8px;
                color: #fff;
                gridline-color: #333333;
            }
            QTableWidget::item {
                padding: 10px;
            }
            QTableWidget::item:selected {
                background-color: #ffc107;
                color: #111;
            }
            QHeaderView::section {
                background-color: #2a2a2a;
                color: #fff;
                padding: 12px;
                border: none;
                font-weight: bold;
            }
        """)

        # Set column widths - make Actions column MUCH wider
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)

        self.table.setColumnWidth(0, 80)   # User ID
        self.table.setColumnWidth(1, 140)  # Name
        self.table.setColumnWidth(2, 180)  # Email
        self.table.setColumnWidth(3, 110)  # Phone
        self.table.setColumnWidth(4, 110)  # Department
        self.table.setColumnWidth(5, 90)   # Role
        self.table.setColumnWidth(6, 100)  # Created
        self.table.setColumnWidth(7, 280)  # Actions - EXTRA WIDE for 3 buttons

        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)

        # Set row height to accommodate buttons properly
        self.table.verticalHeader().setDefaultSectionSize(80)

        layout.addWidget(self.table)

    def create_stat_card(self, title, value, color):
        """Create a statistics card"""
        frame = QFrame()
        frame.setObjectName("card")
        frame.setMinimumHeight(90)

        layout = QVBoxLayout(frame)

        value_label = QLabel(value)
        value_label.setStyleSheet(f"font-size: 32px; font-weight: bold; color: {color};")
        value_label.setAlignment(Qt.AlignCenter)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 13px; color: #94a3b8;")
        title_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(value_label)
        layout.addWidget(title_label)

        frame.value_label = value_label  # Store reference
        return frame

    def load_users(self):
        """Load all users from database"""
        try:
            self.all_users = self.user_manager.get_all_users()

            # Update department filter with unique departments
            departments = set()
            for user in self.all_users:
                # Safely get department from Row or dict
                if isinstance(user, dict):
                    dept = user.get('department')
                else:
                    try:
                        dept = user[4]  # department is at index 4
                    except (IndexError, TypeError):
                        dept = None

                if dept and str(dept).strip():
                    departments.add(str(dept).strip())

            current_dept = self.dept_filter.currentText()
            self.dept_filter.clear()
            self.dept_filter.addItem("All Departments")
            self.dept_filter.addItems(sorted(departments))

            # Restore selection if possible
            index = self.dept_filter.findText(current_dept)
            if index >= 0:
                self.dept_filter.setCurrentIndex(index)

            # Apply current filters
            self.filter_users()

            # Update statistics
            self.update_statistics()

        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to load users:\n{str(e)}"
            )

    def filter_users(self):
        """Filter users based on search and filters"""
        search_text = self.search_input.text().lower().strip()
        role_filter = self.role_filter.currentText()
        dept_filter = self.dept_filter.currentText()

        self.filtered_users = []

        for user in self.all_users:
            # Helper function to safely get values from Row or dict
            def get_val(key, index, default=''):
                if isinstance(user, dict):
                    return user.get(key, default) or default
                else:
                    try:
                        return user[index] or default
                    except (IndexError, TypeError):
                        return default

            user_id = get_val('user_id', 0)
            name = get_val('name', 1)
            email = get_val('email', 2)
            phone = get_val('phone', 3)
            department = get_val('department', 4)
            role = get_val('role', 5, 'Employee')
            created_at = get_val('created_at', 6)

            # Search filter
            if search_text:
                searchable = f"{user_id} {name} {email} {department}".lower()
                if search_text not in searchable:
                    continue

            # Role filter
            if role_filter != "All Roles":
                if role != role_filter:
                    continue

            # Department filter
            if dept_filter != "All Departments":
                if department != dept_filter:
                    continue

            # Add to filtered list (keep original format)
            self.filtered_users.append(user)

        # Update display
        self.display_users()
        self.results_label.setText(f"Showing {len(self.filtered_users)} users")

    def display_users(self):
        """Display filtered users in table"""
        self.table.setRowCount(len(self.filtered_users))

        for row, user in enumerate(self.filtered_users):
            # Helper function to safely get values from Row or dict
            def get_val(key, index, default='N/A'):
                if isinstance(user, dict):
                    return user.get(key, default) or default
                else:
                    try:
                        val = user[index]
                        return val if val is not None else default
                    except (IndexError, TypeError):
                        return default

            user_id = get_val('user_id', 0)
            name = get_val('name', 1)
            email = get_val('email', 2)
            phone = get_val('phone', 3)
            department = get_val('department', 4)
            role = get_val('role', 5, 'Employee')
            created_at = get_val('created_at', 6)

            # User ID
            self.table.setItem(row, 0, QTableWidgetItem(str(user_id)))

            # Name
            name_item = QTableWidgetItem(str(name))
            name_item.setFont(name_item.font())
            self.table.setItem(row, 1, name_item)

            # Email
            self.table.setItem(row, 2, QTableWidgetItem(str(email)))

            # Phone
            self.table.setItem(row, 3, QTableWidgetItem(str(phone)))

            # Department
            self.table.setItem(row, 4, QTableWidgetItem(str(department)))

            # Role
            role_item = QTableWidgetItem(str(role))
            role_color = {
                'Admin': QColor('#ef4444'),
                'Manager': QColor('#10b981'),
                'Employee': QColor('#3b82f6'),
                'Intern': QColor('#f59e0b')
            }.get(role, QColor('#94a3b8'))
            role_item.setForeground(role_color)
            self.table.setItem(row, 5, role_item)

            # Created date
            created = str(created_at)[:10]  # Get date part only
            self.table.setItem(row, 6, QTableWidgetItem(created))

            # Actions buttons - IMPROVED WITH BETTER SIZING
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(5, 5, 5, 5)
            actions_layout.setSpacing(5)

            # View button
            view_btn = QPushButton("👁")
            view_btn.setToolTip("View Details")
            view_btn.setFixedSize(60, 80)  # ← CHANGE WIDTH & HEIGHT HERE (width, height)

            view_btn.setStyleSheet("""
                QPushButton {
                    background-color: #3b82f6;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #2563eb;
                }
            """)
            view_btn.clicked.connect(lambda checked, u=user: self.view_user_details(u))

            # Edit button
            edit_btn = QPushButton("✏")
            edit_btn.setToolTip("Edit User")
            edit_btn.setFixedSize(60, 80)
            edit_btn.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #059669;
                }
            """)
            edit_btn.clicked.connect(lambda checked, u=user: self.edit_user(u))

            # Delete button
            delete_btn = QPushButton("🗑")
            delete_btn.setToolTip("Delete User")
            delete_btn.setFixedSize(60, 80)
            delete_btn.setStyleSheet("""
                QPushButton {
                    background-color: #ef4444;
                    color: white;
                    border: none;
                    border-radius: 5px;
                    font-weight: bold;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #dc2626;
                }
            """)
            delete_btn.clicked.connect(lambda checked, u=user: self.delete_user(u))

            actions_layout.addWidget(view_btn)
            actions_layout.addWidget(edit_btn)
            actions_layout.addWidget(delete_btn)
            actions_layout.addStretch()

            self.table.setCellWidget(row, 7, actions_widget)

    def update_statistics(self):
        """Update statistics cards"""
        total = len(self.all_users)

        role_counts = {
            'Employee': 0,
            'Manager': 0,
            'Admin': 0
        }

        for user in self.all_users:
            # Safely get role from Row or dict
            if isinstance(user, dict):
                role = user.get('role', 'Employee')
            else:
                try:
                    role = user[5] if user[5] else 'Employee'  # role is at index 5
                except (IndexError, TypeError):
                    role = 'Employee'

            if role in role_counts:
                role_counts[role] += 1

        # Update card values
        self.total_users_card.value_label.setText(str(total))
        self.employees_card.value_label.setText(str(role_counts['Employee']))
        self.managers_card.value_label.setText(str(role_counts['Manager']))
        self.admins_card.value_label.setText(str(role_counts['Admin']))

    def view_user_details(self, user):
        """Show detailed user information"""
        dialog = UserDetailsDialog(user, self)
        dialog.exec()

    def edit_user(self, user):
        """Edit user information"""
        dialog = EditUserDialog(user, self.user_manager, self)
        if dialog.exec() == QDialog.Accepted:
            # Reload users after edit
            self.load_users()

    def delete_user(self, user):
        """Delete user from FAISS, SQL database, and file system"""
        # Get user_id and name
        if isinstance(user, dict):
            user_id = user.get('user_id')
            user_name = user.get('name')
        else:
            user_id = user[0]
            user_name = user[1]

        # Confirmation dialog
        reply = QMessageBox.question(
            self,
            "⚠ Confirm Deletion",
            f"Are you sure you want to delete user:\n\n"
            f"ID: {user_id}\n"
            f"Name: {user_name}\n\n"
            f"This will permanently remove:\n"
            f"• User record from database\n"
            f"• Face embeddings from FAISS\n"
            f"• All registered face images\n\n"
            f"This action cannot be undone!",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        print(f"\n[USERS PAGE] ==================== DELETE USER START ====================")
        print(f"[USERS PAGE] Deleting user: {user_id} - {user_name}")

        success_steps = []
        failed_steps = []

        # Step 1: Delete from FAISS index
        if self.face_registration:
            try:
                print(f"[USERS PAGE] Step 1: Removing from FAISS index...")
                faiss_success = self.delete_from_faiss(user_id)
                if faiss_success:
                    success_steps.append("✓ Removed from FAISS index")
                    print(f"[USERS PAGE] ✓ Removed from FAISS")
                else:
                    failed_steps.append("✗ Failed to remove from FAISS index")
                    print(f"[USERS PAGE] ✗ FAISS removal failed")
            except Exception as e:
                failed_steps.append(f"✗ FAISS error: {str(e)}")
                print(f"[USERS PAGE] ✗ FAISS error: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"[USERS PAGE] ⚠️ No face_registration instance - skipping FAISS deletion")
            failed_steps.append("⚠️ FAISS deletion skipped (no instance)")

        # Step 2: Delete face images from file system
        try:
            print(f"[USERS PAGE] Step 2: Deleting face images...")
            import shutil

            # Delete user directory
            user_dir = os.path.join('registered_faces', user_id)
            if os.path.exists(user_dir):
                shutil.rmtree(user_dir)
                success_steps.append("✓ Deleted face images")
                print(f"[USERS PAGE] ✓ Deleted directory: {user_dir}")
            else:
                success_steps.append("✓ No face images found")
                print(f"[USERS PAGE] ℹ️ No directory found: {user_dir}")

        except Exception as e:
            failed_steps.append(f"✗ File deletion error: {str(e)}")
            print(f"[USERS PAGE] ✗ File deletion error: {e}")

        # Step 3: Delete from SQL database
        try:
            print(f"[USERS PAGE] Step 3: Deleting from database...")
            db_success = self.delete_from_database(user_id)
            if db_success:
                success_steps.append("✓ Removed from database")
                print(f"[USERS PAGE] ✓ Removed from database")
            else:
                failed_steps.append("✗ Failed to remove from database")
                print(f"[USERS PAGE] ✗ Database deletion failed")
        except Exception as e:
            failed_steps.append(f"✗ Database error: {str(e)}")
            print(f"[USERS PAGE] ✗ Database error: {e}")

        # Show result
        if len(failed_steps) == 0:
            print(f"[USERS PAGE] 🎉 User deleted successfully!")
            print("[USERS PAGE] ==================== DELETE USER SUCCESS ====================\n")

            # Create detailed success message box
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("✅ User Deleted Successfully")
            msg_box.setIcon(QMessageBox.Information)

            details = (
                f"<b>User '{user_name}' has been successfully deleted!</b><br><br>"
                f"<b>User Details:</b><br>"
                f"<table cellpadding='5' style='color: #333;'>"
                f"<tr><td><b>User ID:</b></td><td>{user_id}</td></tr>"
                f"<tr><td><b>Name:</b></td><td>{user_name}</td></tr>"
                f"</table><br>"
                f"<b>Operations Completed:</b><br>"
            )

            for step in success_steps:
                details += f"&nbsp;&nbsp;{step}<br>"

            msg_box.setText(details)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()

            # Reload user list
            self.load_users()
        else:
            print(f"[USERS PAGE] ⚠️ Deletion completed with errors")
            print("[USERS PAGE] ==================== DELETE USER PARTIAL ====================\n")

            # Create detailed partial success message box
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("⚠ Partial Deletion")
            msg_box.setIcon(QMessageBox.Warning)

            result_message = f"<b>User '{user_name}' deletion completed with some issues:</b><br><br>"

            if success_steps:
                result_message += "<b>✓ Successful:</b><br>"
                for step in success_steps:
                    result_message += f"&nbsp;&nbsp;{step}<br>"
                result_message += "<br>"

            if failed_steps:
                result_message += "<b>✗ Failed:</b><br>"
                for step in failed_steps:
                    result_message += f"&nbsp;&nbsp;{step}<br>"

            msg_box.setText(result_message)
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()

            # Reload user list anyway to reflect changes
            self.load_users()

    def delete_from_faiss(self, user_id):
        """Delete user embedding from FAISS index (uses FaceRegistration method)"""
        try:
            print(f"[USERS PAGE] Calling FaceRegistration.delete_user() for {user_id}")

            # Use the proper delete method from FaceRegistration
            success, message = self.face_registration.delete_user(user_id)

            if success:
                print(f"[USERS PAGE] ✓ FAISS deletion successful: {message}")
            else:
                print(f"[USERS PAGE] ✗ FAISS deletion failed: {message}")

            return success

        except Exception as e:
            print(f"[USERS PAGE] ✗ Error calling delete_user: {e}")
            import traceback
            traceback.print_exc()
            return False

    def delete_from_database(self, user_id):
        """Delete user from SQL database"""
        try:
            from app.db.database import DatabaseConfig

            conn = DatabaseConfig.get_connection()
            if not conn:
                print("[USERS PAGE] Failed to get database connection")
                return False

            cursor = conn.cursor()

            # Delete attendance records first (foreign key constraint)
            cursor.execute("DELETE FROM attendance WHERE user_id = ?", (user_id,))
            attendance_deleted = cursor.rowcount
            print(f"[USERS PAGE] Deleted {attendance_deleted} attendance record(s) for user {user_id}")

            # Delete user record
            cursor.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            deleted_rows = cursor.rowcount

            conn.commit()
            cursor.close()
            conn.close()

            print(f"[USERS PAGE] Deleted {deleted_rows} user record(s)")
            return deleted_rows > 0

        except Exception as e:
            print(f"[USERS PAGE] Database deletion error: {e}")
            import traceback
            traceback.print_exc()
            return False