"""
Face Registration Page - Register new faces with user details and camera profile matching
Enhanced with quality profile selection and application
"""
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QLineEdit, QComboBox, QMessageBox, QFrame,
    QScrollArea, QGroupBox, QGridLayout, QProgressBar, QDialog, QDialogButtonBox
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
import cv2
import numpy as np
import os
from app.workers.registration_camera_worker import RegistrationCameraWorker, RegistrationCaptureWorker




class DuplicateUserDialog(QDialog):
    """Dialog to show duplicate user warning with existing user details"""

    def __init__(self, existing_user, similarity_score, parent=None):

        super().__init__(parent)


        self.existing_user = existing_user
        self.similarity_score = similarity_score
        self.setWindowTitle("⚠️ Duplicate User Detected")
        self.setMinimumWidth(500)
        self.setup_ui()

    def setup_ui(self):

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Title
        title = QLabel("Register New Face")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #fff;")
        main_layout.addWidget(title)

        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(20)

        # ========== NEW: Input Source Selection ==========
        source_group = self.create_source_selection_section()
        content_layout.addWidget(source_group)

        # Camera Profile Selection Section
        profile_group = self.create_profile_selection_section()
        content_layout.addWidget(profile_group)

        # Camera Feed Section
        camera_group = self.create_camera_section()
        content_layout.addWidget(camera_group)

        # User Details Section
        details_group = self.create_details_section()
        content_layout.addWidget(details_group)

        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Warning header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #ef4444;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        header_layout = QVBoxLayout(header_frame)

        warning_label = QLabel("⚠️ User Already Registered!")
        warning_label.setStyleSheet("color: #fff; font-size: 18px; font-weight: bold;")
        warning_label.setAlignment(Qt.AlignCenter)

        similarity_label = QLabel(f"Similarity Score: {self.similarity_score:.1f}%")
        similarity_label.setStyleSheet("color: #fff; font-size: 14px;")
        similarity_label.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(warning_label)
        header_layout.addWidget(similarity_label)
        layout.addWidget(header_frame)

        # Message
        message = QLabel(
            "This face matches an existing registered user in the database. "
            "Please verify the information below:"
        )
        message.setWordWrap(True)
        message.setStyleSheet("color: #fff; font-size: 14px; margin: 10px 0;")
        layout.addWidget(message)

        # Existing user details
        details_frame = QFrame()
        details_frame.setObjectName("card")
        details_layout = QVBoxLayout(details_frame)

        title = QLabel("Existing User Information:")
        title.setStyleSheet("color: #ffc107; font-weight: bold; font-size: 15px; margin-bottom: 10px;")
        details_layout.addWidget(title)

        # Helper function to safely get values
        def get_value(key, index, default='N/A'):
            if isinstance(self.existing_user, dict):
                return self.existing_user.get(key, default) or default
            else:
                try:
                    val = self.existing_user[index]
                    return val if val is not None else default
                except (IndexError, TypeError):
                    return default

        # Display user info
        info_text = f"""
        <table style='width: 100%; color: #fff;'>
            <tr>
                <td style='padding: 5px; font-weight: bold; color: #94a3b8;'>User ID:</td>
                <td style='padding: 5px;'>{get_value('user_id', 0)}</td>
            </tr>
            <tr>
                <td style='padding: 5px; font-weight: bold; color: #94a3b8;'>Name:</td>
                <td style='padding: 5px;'>{get_value('name', 1)}</td>
            </tr>
            <tr>
                <td style='padding: 5px; font-weight: bold; color: #94a3b8;'>Email:</td>
                <td style='padding: 5px;'>{get_value('email', 2)}</td>
            </tr>
            <tr>
                <td style='padding: 5px; font-weight: bold; color: #94a3b8;'>Phone:</td>
                <td style='padding: 5px;'>{get_value('phone', 3)}</td>
            </tr>
            <tr>
                <td style='padding: 5px; font-weight: bold; color: #94a3b8;'>Department:</td>
                <td style='padding: 5px;'>{get_value('department', 4)}</td>
            </tr>
            <tr>
                <td style='padding: 5px; font-weight: bold; color: #94a3b8;'>Role:</td>
                <td style='padding: 5px;'>{get_value('role', 5)}</td>
            </tr>
        </table>
        """

        info_label = QLabel(info_text)
        info_label.setWordWrap(True)
        details_layout.addWidget(info_label)
        layout.addWidget(details_frame)

        # Recommendation
        recommendation = QLabel(
            "❌ Registration cannot proceed. Please try registering a different person."
        )
        recommendation.setWordWrap(True)
        recommendation.setStyleSheet("color: #ef4444; font-weight: bold; font-size: 13px; margin-top: 10px;")
        layout.addWidget(recommendation)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        layout.addWidget(button_box)




class RegistrationPage(QWidget):
    """Page for registering new faces with camera profile matching"""

    registration_completed = Signal(str, str)
    DUPLICATE_THRESHOLD = 0.80  # 80% similarity threshold for duplicate detection

    def __init__(self, face_registration, user_manager, parent=None):
        super().__init__(parent)
        self.face_registration = face_registration
        self.user_manager = user_manager
        self.camera_worker = None
        self.capture_worker = None
        self.capture_timer = None
        self.captured_embeddings = []
        self.captured_images = []
        self.target_captures = 5
        self.is_capturing = False
        self.current_frame = None

        # Add source mode tracking
        self.source_mode = "camera"  # "camera" or "upload"
        self.uploaded_images = []  # Store uploaded image paths

        # Quality profile management
        self.quality_matcher = None
        self.selected_profile_id = None

        # Initialize quality matcher
        self.init_quality_matcher()

        self.setup_ui()

    def init_quality_matcher(self):
        """Initialize the quality matcher"""
        try:
            from app.utils.global_quality_matcher import GlobalQualityMatcher
            self.quality_matcher = GlobalQualityMatcher()
            print("[REG PAGE] ✓ Quality matcher initialized")
        except Exception as e:
            print(f"[REG PAGE ERROR] Failed to initialize quality matcher: {e}")
            import traceback
            traceback.print_exc()

    def get_available_profiles(self):
        """Get list of available camera profiles"""
        profile_dir = 'camera_profiles'

        if not os.path.exists(profile_dir):
            return []

        profiles = []
        for filename in os.listdir(profile_dir):
            if filename.endswith('_profile.pkl'):
                # Extract camera_id from filename
                camera_id = filename.replace('_profile.pkl', '')
                profiles.append(camera_id)

        return profiles

    def load_profile_dropdown(self):
        """Load available profiles into dropdown"""
        self.profile_combo.clear()
        self.profile_combo.addItem("No Profile (Default)", None)

        profiles = self.get_available_profiles()

        if profiles:
            for profile_id in profiles:
                # Make display name more readable
                display_name = f"{profile_id}"
                self.profile_combo.addItem(display_name, profile_id)

            print(f"[REG PAGE] Loaded {len(profiles)} camera profile(s)")
        else:
            print("[REG PAGE] No camera profiles found")

    def on_profile_selected(self, index):
        """Handle profile selection from dropdown"""
        profile_id = self.profile_combo.itemData(index)

        if profile_id is None:
            self.selected_profile_id = None
            self.profile_status_label.setText("No profile selected")
            self.profile_status_label.setStyleSheet("color: #64748b; font-size: 12px;")
            print("[REG PAGE] No profile selected - using default quality")
        else:
            # Load the profile
            if self.quality_matcher and self.quality_matcher.load_profile(profile_id):
                self.selected_profile_id = profile_id
                self.profile_status_label.setText(f"✓ Profile loaded: {profile_id}")
                self.profile_status_label.setStyleSheet("color: #22c55e; font-size: 12px; font-weight: bold;")
                print(f"[REG PAGE] ✓ Profile loaded: {profile_id}")

                # Show profile details
                if self.quality_matcher.quality_profile:
                    profile = self.quality_matcher.quality_profile
                    details = (
                        f"  Brightness: {profile.get('brightness', 0):.2f}, "
                        f"Contrast: {profile.get('contrast', 0):.2f}, "
                        f"Saturation: {profile.get('saturation', 0):.2f}"
                    )
                    print(f"[REG PAGE] Profile properties: {details}")
            else:
                self.selected_profile_id = None
                self.profile_status_label.setText("❌ Failed to load profile")
                self.profile_status_label.setStyleSheet("color: #ef4444; font-size: 12px;")
                print(f"[REG PAGE ERROR] Failed to load profile: {profile_id}")

    def setup_ui(self):

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Title
        title = QLabel("Register New Face")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #fff;")
        main_layout.addWidget(title)

        # Create scroll area for content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(20)

        # ========== NEW: Input Source Selection (FIRST) ==========
        source_group = self.create_source_selection_section()
        content_layout.addWidget(source_group)

        # ========== Camera Profile Selection Section ==========
        profile_group = self.create_profile_selection_section()
        content_layout.addWidget(profile_group)

        # ========== Camera Feed Section ==========
        camera_group = self.create_camera_section()
        content_layout.addWidget(camera_group)

        # ========== User Details Section ==========
        details_group = self.create_details_section()
        content_layout.addWidget(details_group)

        # Action Buttons
        actions_layout = QHBoxLayout()
        actions_layout.addStretch()

        self.reset_button = QPushButton("Reset")
        self.reset_button.setMinimumHeight(40)
        self.reset_button.clicked.connect(self.reset_form)

        self.register_button = QPushButton("Register Face")
        self.register_button.setMinimumHeight(40)
        self.register_button.setStyleSheet("""
            QPushButton {
                background-color: #ffc107;
                color: #000000;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #ffca2c;
                color: #000000;
            }
            QPushButton:disabled {
                background-color: #666666;
                color: #999999;
            }
        """)
        self.register_button.clicked.connect(self.register_face)
        self.register_button.setEnabled(False)

        actions_layout.addWidget(self.reset_button)
        actions_layout.addWidget(self.register_button)
        content_layout.addLayout(actions_layout)

        scroll.setWidget(content_widget)
        main_layout.addWidget(scroll)

    def create_profile_selection_section(self):
        """Create camera profile selection section"""
        group = QGroupBox("Camera Quality Profile")
        group.setStyleSheet("""
            QGroupBox {
                background-color: #1a472a;
                border: 1px solid #22c55e;
                border-radius: 12px;
                padding: 20px;
                margin-top: 10px;
                font-size: 16px;
                font-weight: bold;
                color: #22c55e;
            }
        """)

        layout = QVBoxLayout(group)

        # Instructions
        instructions = QLabel(
            "📸 Select a camera profile to match the quality properties of your live detection camera. "
            "This improves recognition accuracy by training embeddings with the same quality characteristics."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #94a3b8; font-size: 13px; font-weight: normal; margin-bottom: 10px;")
        layout.addWidget(instructions)

        # Profile selection row
        selection_layout = QHBoxLayout()

        profile_label = QLabel("Select Profile:")
        profile_label.setStyleSheet("color: #fff; font-size: 14px; font-weight: bold;")

        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumHeight(40)
        self.profile_combo.setMinimumWidth(300)
        self.profile_combo.currentIndexChanged.connect(self.on_profile_selected)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.setMinimumHeight(40)
        refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #333333;
                color: #fff;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #444444;
            }
        """)
        refresh_btn.clicked.connect(self.load_profile_dropdown)

        selection_layout.addWidget(profile_label)
        selection_layout.addWidget(self.profile_combo)
        selection_layout.addWidget(refresh_btn)
        selection_layout.addStretch()

        layout.addLayout(selection_layout)

        # Status label
        self.profile_status_label = QLabel("No profile selected")
        self.profile_status_label.setStyleSheet("color: #64748b; font-size: 12px; margin-top: 5px;")
        layout.addWidget(self.profile_status_label)

        # Info box
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 8px;
                padding: 10px;
                margin-top: 10px;
            }
        """)
        info_layout = QVBoxLayout(info_frame)

        info_title = QLabel("ℹ️ How it works:")
        info_title.setStyleSheet("color: #ffc107; font-size: 13px; font-weight: bold;")
        info_layout.addWidget(info_title)

        info_text = QLabel(
            "• Profiles are created from your live detection cameras\n"
            "• The selected profile will be applied to captured face images\n"
            "• This matches brightness, contrast, noise, blur, and other quality properties\n"
            "• Result: Better recognition accuracy during live detection"
        )
        info_text.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: normal;")
        info_layout.addWidget(info_text)

        layout.addWidget(info_frame)

        # Load available profiles
        self.load_profile_dropdown()

        return group

    def create_source_selection_section(self):
        """Create input source selection section (Camera or Upload)"""
        group = QGroupBox("Input Source")
        group.setStyleSheet("""
            QGroupBox {
                background-color: #1a1a47;
                border: 1px solid #3b82f6;
                border-radius: 12px;
                padding: 20px;
                margin-top: 10px;
                font-size: 16px;
                font-weight: bold;
                color: #3b82f6;
            }
        """)

        layout = QVBoxLayout(group)

        # Instructions
        instructions = QLabel(
            "Choose how you want to provide face images for registration:"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #94a3b8; font-size: 13px; font-weight: normal; margin-bottom: 10px;")
        layout.addWidget(instructions)

        # Radio buttons for source selection
        source_layout = QHBoxLayout()

        self.camera_radio = QPushButton("📷 Camera Capture")
        self.camera_radio.setCheckable(True)
        self.camera_radio.setChecked(True)  # Default
        self.camera_radio.setMinimumHeight(50)
        self.camera_radio.clicked.connect(lambda: self.on_source_changed("camera"))

        self.upload_radio = QPushButton("📁 Upload Images")
        self.upload_radio.setCheckable(True)
        self.upload_radio.setMinimumHeight(50)
        self.upload_radio.clicked.connect(lambda: self.on_source_changed("upload"))

        # Style for toggle buttons
        button_style = """
            QPushButton {
                background-color: #333333;
                color: #fff;
                border: 2px solid #444444;
                padding: 12px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:checked {
                background-color: #3b82f6;
                border-color: #3b82f6;
                color: #fff;
            }
            QPushButton:hover {
                background-color: #444444;
            }
            QPushButton:checked:hover {
                background-color: #2563eb;
            }
        """
        self.camera_radio.setStyleSheet(button_style)
        self.upload_radio.setStyleSheet(button_style)

        source_layout.addWidget(self.camera_radio)
        source_layout.addWidget(self.upload_radio)
        layout.addLayout(source_layout)

        return group

    def on_source_changed(self, mode):
        """Handle source mode change"""
        self.source_mode = mode

        if mode == "camera":
            self.camera_radio.setChecked(True)
            self.upload_radio.setChecked(False)
            self.camera_widget.show()
            self.upload_widget.hide()
            self.instructions_label.setText(
                "Position your face in the frame and click 'Start Capture'. "
                "Multiple angles will be captured automatically."
            )
            print(f"[REG PAGE] Switched to CAMERA mode")
        else:  # upload
            self.upload_radio.setChecked(True)
            self.camera_radio.setChecked(False)
            self.upload_widget.show()
            self.camera_widget.hide()
            self.instructions_label.setText(
                "Upload 5 clear images of the person's face from different angles."
            )
            # Stop camera if running
            if self.camera_worker:
                self.stop_camera_completely()
            print(f"[REG PAGE] Switched to UPLOAD mode")

        # Reset captures
        self.reset_captures()

    def reset_captures(self):
        """Reset capture-related data"""
        self.captured_embeddings = []
        self.captured_images = []
        self.uploaded_images = []
        self.progress_bar.setValue(0)

        for label in self.preview_labels:
            label.clear()
            label.setStyleSheet("border: 2px dashed #444; border-radius: 8px; background-color: #1a1a1a;")

        self.register_button.setEnabled(False)
        self.upload_status_label.setText("No images selected")

    def upload_images(self):
        """Handle image upload"""
        from PySide6.QtWidgets import QFileDialog

        print("[REG PAGE] Opening file dialog for image upload...")

        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Face Images (5 required)",
            "",
            "Image Files (*.jpg *.jpeg *.png *.bmp);;All Files (*)"
        )

        if not file_paths:
            print("[REG PAGE] No files selected")
            return

        if len(file_paths) != 5:
            QMessageBox.warning(
                self,
                "Invalid Selection",
                f"Please select exactly 5 images.\nYou selected: {len(file_paths)}"
            )
            return

        print(f"[REG PAGE] Processing {len(file_paths)} uploaded images...")

        # Process images in background
        from PySide6.QtCore import QThread

        class ImageProcessWorker(QThread):
            processing_complete = Signal(list, list, str)
            processing_failed = Signal(str)

            def __init__(self, file_paths, face_registration, quality_matcher):
                super().__init__()
                self.file_paths = file_paths
                self.face_registration = face_registration
                self.quality_matcher = quality_matcher

            def run(self):
                embeddings = []
                images = []

                for i, path in enumerate(self.file_paths):
                    try:
                        # Load image
                        img = cv2.imread(path)
                        if img is None:
                            self.processing_failed.emit(f"Failed to load image: {os.path.basename(path)}")
                            return

                        # Apply quality profile if selected
                        if self.quality_matcher and self.quality_matcher.quality_profile:
                            img = self.quality_matcher.apply_quality_profile(img, intensity=0.7)

                        # Extract face
                        success, bbox, embedding, face_crop, message = self.face_registration.extract_face_from_frame(
                            img)

                        if not success:
                            self.processing_failed.emit(f"Image {i + 1} ({os.path.basename(path)}): {message}")
                            return

                        embeddings.append(embedding)
                        images.append(face_crop)

                    except Exception as e:
                        self.processing_failed.emit(f"Error processing image {i + 1}: {str(e)}")
                        return

                self.processing_complete.emit(embeddings, images, "All images processed successfully")

        self.upload_status_label.setText("Processing images...")
        self.upload_status_label.setStyleSheet("color: #ffc107; font-size: 13px;")
        self.upload_button.setEnabled(False)

        worker = ImageProcessWorker(file_paths, self.face_registration, self.quality_matcher)
        worker.processing_complete.connect(self.on_upload_complete)
        worker.processing_failed.connect(self.on_upload_failed)
        worker.start()

        # Keep reference to prevent garbage collection
        self._upload_worker = worker

    def on_upload_complete(self, embeddings, images, message):
        """Handle successful image upload processing"""
        self.captured_embeddings = embeddings
        self.captured_images = images

        # Update UI
        self.progress_bar.setValue(len(embeddings))
        self.upload_status_label.setText(f"✓ {len(embeddings)} images processed successfully")
        self.upload_status_label.setStyleSheet("color: #22c55e; font-size: 13px; font-weight: bold;")

        # Show previews
        for i, img in enumerate(images[:self.target_captures]):
            face_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = face_rgb.shape
            q_image = QImage(face_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.preview_labels[i].setPixmap(scaled)
            self.preview_labels[i].setStyleSheet(
                "border: 2px solid #22c55e; border-radius: 8px; background-color: #1a1a1a;"
            )

        self.register_button.setEnabled(True)
        self.upload_button.setEnabled(True)

        print(f"[REG PAGE] ✓ Upload complete: {len(embeddings)} faces processed")

    def on_upload_failed(self, error_msg):
        """Handle upload processing failure"""
        self.upload_status_label.setText(f"❌ {error_msg}")
        self.upload_status_label.setStyleSheet("color: #ef4444; font-size: 13px;")
        self.upload_button.setEnabled(True)

        QMessageBox.critical(self, "Processing Failed", error_msg)
        print(f"[REG PAGE] ✗ Upload failed: {error_msg}")

    def create_camera_section(self):
        """Create camera feed section with upload option"""
        group = QGroupBox("Face Capture")
        group.setStyleSheet("""
            QGroupBox {
                background-color: #222222;
                border: 1px solid #333333;
                border-radius: 12px;
                padding: 20px;
                margin-top: 10px;
                font-size: 16px;
                font-weight: bold;
                color: #fff;
            }
        """)

        layout = QVBoxLayout(group)

        # Instructions (will be updated based on mode)
        self.instructions_label = QLabel(
            "Position your face in the frame and click 'Start Capture'. "
            "Multiple angles will be captured automatically. The system will check for duplicates before registration."
        )
        self.instructions_label.setWordWrap(True)
        self.instructions_label.setStyleSheet("color: #94a3b8; font-size: 14px; font-weight: normal;")
        layout.addWidget(self.instructions_label)

        # ========== UPLOAD SECTION (Initially Hidden) ==========
        self.upload_widget = QWidget()
        upload_layout = QVBoxLayout(self.upload_widget)

        upload_info = QLabel(
            "Upload 5 clear images of the person's face. Images should:\n"
            "• Show the face clearly from different angles\n"
            "• Have good lighting and focus\n"
            "• Contain only ONE face per image"
        )
        upload_info.setWordWrap(True)
        upload_info.setStyleSheet("color: #94a3b8; font-size: 13px; margin: 10px 0;")
        upload_layout.addWidget(upload_info)

        self.upload_button = QPushButton("📁 Select Images (5 required)")
        self.upload_button.setMinimumHeight(50)
        self.upload_button.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: #fff;
                border: none;
                padding: 15px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        self.upload_button.clicked.connect(self.upload_images)
        upload_layout.addWidget(self.upload_button)

        self.upload_status_label = QLabel("No images selected")
        self.upload_status_label.setStyleSheet("color: #64748b; font-size: 13px; margin-top: 5px;")
        upload_layout.addWidget(self.upload_status_label)

        layout.addWidget(self.upload_widget)
        self.upload_widget.hide()  # Hidden by default

        # ========== CAMERA SECTION ==========
        self.camera_widget = QWidget()
        camera_layout = QVBoxLayout(self.camera_widget)

        # Camera display
        self.camera_label = QLabel()
        self.camera_label.setMinimumSize(640, 480)
        self.camera_label.setMaximumSize(640, 480)
        self.camera_label.setStyleSheet("background-color: #000; border-radius: 8px;")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setScaledContents(False)
        camera_layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)

        layout.addWidget(self.camera_widget)

        # Status and Progress
        status_layout = QHBoxLayout()
        self.status_label = QLabel("Ready to capture")
        self.status_label.setStyleSheet("color: #64748b; font-size: 14px; font-weight: normal;")

        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximum(self.target_captures)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m captures")

        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        status_layout.addWidget(self.progress_bar)
        layout.addLayout(status_layout)

        # Preview grid for captured faces
        self.preview_grid = QGridLayout()
        self.preview_grid.setSpacing(10)
        self.preview_labels = []

        for i in range(self.target_captures):
            label = QLabel()
            label.setFixedSize(100, 100)
            label.setStyleSheet("border: 2px dashed #444; border-radius: 8px; background-color: #1a1a1a;")
            label.setAlignment(Qt.AlignCenter)
            self.preview_labels.append(label)
            self.preview_grid.addWidget(label, 0, i)

        layout.addLayout(self.preview_grid)

        # Camera controls
        controls_layout = QHBoxLayout()

        self.start_camera_btn = QPushButton("Start Camera")
        self.start_camera_btn.clicked.connect(self.start_camera)

        self.capture_btn = QPushButton("Start Capture")
        self.capture_btn.clicked.connect(self.start_capture)
        self.capture_btn.setEnabled(False)

        self.stop_capture_btn = QPushButton("Stop Capture")
        self.stop_capture_btn.clicked.connect(self.stop_capture)
        self.stop_capture_btn.setEnabled(False)

        controls_layout.addStretch()
        controls_layout.addWidget(self.start_camera_btn)
        controls_layout.addWidget(self.capture_btn)
        controls_layout.addWidget(self.stop_capture_btn)
        layout.addLayout(controls_layout)

        return group




    def create_details_section(self):
        """Create user details input section"""
        group = QGroupBox("User Details")
        group.setStyleSheet("""
            QGroupBox {
                background-color: #222222;
                border: 1px solid #333333;
                border-radius: 12px;
                padding: 20px;
                margin-top: 10px;
                font-size: 16px;
                font-weight: bold;
                color: #fff;
            }
        """)

        layout = QFormLayout(group)
        layout.setSpacing(15)

        # Name
        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Enter full name")
        self.name_input.setMinimumHeight(40)
        layout.addRow("Name*:", self.name_input)

        # Email
        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("Enter email address")
        self.email_input.setMinimumHeight(40)
        layout.addRow("Email:", self.email_input)

        # Phone
        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("Enter phone number")
        self.phone_input.setMinimumHeight(40)
        layout.addRow("Phone:", self.phone_input)

        # Department
        self.department_input = QLineEdit()
        self.department_input.setPlaceholderText("Enter department")
        self.department_input.setMinimumHeight(40)
        layout.addRow("Department:", self.department_input)

        # Role
        self.role_combo = QComboBox()
        self.role_combo.addItems(["Employee", "Admin", "Student", "Visitor"])
        self.role_combo.setMinimumHeight(40)
        layout.addRow("Role*:", self.role_combo)

        return group

    def start_camera(self):
        """Start camera in background thread"""
        try:
            print("[REG PAGE] Starting camera worker...")
            self.camera_worker = RegistrationCameraWorker(0, self.face_registration)
            self.camera_worker.frame_ready.connect(self.on_frame_ready)
            self.camera_worker.error_occurred.connect(self.on_camera_error)
            self.camera_worker.start()

            self.start_camera_btn.setEnabled(False)
            self.capture_btn.setEnabled(True)
            self.status_label.setText("Camera started - Position your face")
            print("[REG PAGE] ✓ Camera worker started")

        except Exception as e:
            print(f"[REG PAGE ERROR] Failed to start camera: {e}")
            QMessageBox.critical(self, "Camera Error", f"Failed to start camera: {str(e)}")

    def on_frame_ready(self, frame, is_good, message, quality):
        """Handle frame from camera worker (runs in UI thread)"""
        self.current_frame = frame

        # Convert and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)
        scaled = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled)

    def on_camera_error(self, error_msg):
        """Handle camera errors"""
        print(f"[REG PAGE ERROR] Camera error: {error_msg}")
        QMessageBox.critical(self, "Camera Error", error_msg)
        self.cleanup()

    def start_capture(self):
        """Start automatic face capture"""
        print("[REG PAGE] Starting capture sequence...")

        # Show warning if no profile selected
        if self.selected_profile_id is None:
            reply = QMessageBox.question(
                self,
                "No Profile Selected",
                "You haven't selected a camera profile. Captured images will use default quality.\n\n"
                "For better recognition accuracy, it's recommended to select a profile that matches your live detection camera.\n\n"
                "Continue without profile?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )

            if reply == QMessageBox.No:
                return

        self.captured_embeddings = []
        self.captured_images = []
        self.progress_bar.setValue(0)
        self.is_capturing = True

        # Clear preview
        for label in self.preview_labels:
            label.clear()
            label.setStyleSheet("border: 2px dashed #444; border-radius: 8px; background-color: #1a1a1a;")

        self.capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(True)
        self.status_label.setText("Capturing faces...")
        self.status_label.setStyleSheet("color: #ffc107; font-size: 14px; font-weight: bold;")

        # Start capture timer
        self.capture_timer = QTimer()
        self.capture_timer.timeout.connect(self.capture_face)
        self.capture_timer.start(1000)  # Capture every 1 second
        print("[REG PAGE] ✓ Capture timer started")

    def capture_face(self):
        """Trigger face capture in background"""
        if not self.is_capturing or self.current_frame is None:
            return

        if self.capture_worker and self.capture_worker.isRunning():
            print("[REG PAGE] Skipping capture - worker busy")
            return

        print(f"[REG PAGE] Capturing face {len(self.captured_embeddings) + 1}/{self.target_captures}...")

        # Pass quality matcher and profile ID to worker
        self.capture_worker = RegistrationCaptureWorker(
            self.current_frame,
            self.face_registration,
            quality_matcher=self.quality_matcher,
            profile_intensity=0.7  # 70% intensity for quality application
        )
        self.capture_worker.capture_completed.connect(self.on_capture_completed)
        self.capture_worker.capture_failed.connect(self.on_capture_failed)
        self.capture_worker.start()

    def on_capture_completed(self, embedding, face_crop):
        """Handle successful face capture"""
        self.captured_embeddings.append(embedding)
        self.captured_images.append(face_crop)

        # Update progress
        count = len(self.captured_embeddings)
        self.progress_bar.setValue(count)
        self.status_label.setText(f"✓ Captured {count}/{self.target_captures}")
        self.status_label.setStyleSheet("color: #22c55e; font-size: 14px; font-weight: bold;")
        print(f"[REG PAGE] ✓ Captured {count}/{self.target_captures}")

        # Show preview
        if count <= len(self.preview_labels):
            face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            h, w, ch = face_rgb.shape
            q_image = QImage(face_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.preview_labels[count - 1].setPixmap(scaled)
            self.preview_labels[count - 1].setStyleSheet(
                "border: 2px solid #22c55e; border-radius: 8px; background-color: #1a1a1a;"
            )

        # Stop if we have enough captures
        if count >= self.target_captures:
            print("[REG PAGE] Target captures reached, stopping...")
            self.stop_capture()
            self.status_label.setText(f"✓ Successfully captured {count} faces!")
            self.status_label.setStyleSheet("color: #22c55e; font-size: 14px; font-weight: bold;")
            self.register_button.setEnabled(True)
            print("[REG PAGE] ✓ Ready to register")

    def on_capture_failed(self, message):
        """Handle failed capture"""
        print(f"[REG PAGE] Capture failed: {message}")

    def stop_capture(self):
        """Stop face capture"""
        print("[REG PAGE] Stopping capture...")
        self.is_capturing = False

        if self.capture_timer:
            self.capture_timer.stop()
            self.capture_timer = None
            print("[REG PAGE] ✓ Capture timer stopped")

        self.capture_btn.setEnabled(True)
        self.stop_capture_btn.setEnabled(False)
        print("[REG PAGE] ✓ Capture stopped")

    def check_for_duplicate(self):
        """Check if the captured face matches any existing user"""
        print("[REG PAGE] ==================== DUPLICATE CHECK START ====================")

        if len(self.captured_embeddings) == 0:
            print("[REG PAGE] No embeddings to check")
            return False, None, 0.0

        avg_embedding = np.mean(self.captured_embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

        print(f"[REG PAGE] Checking against {len(self.face_registration.faiss_ids)} registered faces...")

        try:
            k = min(3, len(self.face_registration.faiss_ids))
            if k == 0:
                print("[REG PAGE] No existing faces in database - no duplicates possible")
                return False, None, 0.0

            query_embedding = avg_embedding.reshape(1, -1).astype('float32')
            distances, indices = self.face_registration.faiss_index.search(query_embedding, k)

            print(f"[REG PAGE] FAISS search results:")
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                similarity = (1 - (dist / 2)) * 100
                user_id = self.face_registration.faiss_ids[idx]
                print(f"[REG PAGE]   Match {i + 1}: User {user_id}, Similarity: {similarity:.2f}%")

            # Check the best match
            best_distance = distances[0][0]
            best_idx = indices[0][0]
            best_user_id = self.face_registration.faiss_ids[best_idx]

            similarity_score = (1 - (best_distance / 2)) * 100

            print(f"[REG PAGE] Best match: User {best_user_id}, Similarity: {similarity_score:.2f}%")
            print(f"[REG PAGE] Threshold: {self.DUPLICATE_THRESHOLD * 100}%")

            if similarity_score >= (self.DUPLICATE_THRESHOLD * 100):
                print(f"[REG PAGE] ⚠️ DUPLICATE DETECTED! Similarity {similarity_score:.2f}% >= {self.DUPLICATE_THRESHOLD * 100}%")

                existing_user = self.user_manager.get_user_by_id(best_user_id)

                if existing_user:
                    print(f"[REG PAGE] Found existing user: {existing_user}")
                    print("[REG PAGE] ==================== DUPLICATE CHECK: FAILED ====================")
                    return True, existing_user, similarity_score
                else:
                    print(f"[REG PAGE] Warning: User {best_user_id} not found in database")

            print(f"[REG PAGE] ✓ No duplicate found - similarity {similarity_score:.2f}% below threshold")
            print("[REG PAGE] ==================== DUPLICATE CHECK: PASSED ====================")
            return False, None, similarity_score

        except Exception as e:
            print(f"[REG PAGE ERROR] Error during duplicate check: {e}")
            import traceback
            traceback.print_exc()
            return False, None, 0.0

    def register_face(self):
        """Register the captured face with user details"""
        print("[REG PAGE] ==================== REGISTRATION START ====================")

        # Validate inputs
        name = self.name_input.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation Error", "Please enter a name.")
            return

        if len(self.captured_embeddings) == 0:
            QMessageBox.warning(self, "No Face Captured", "Please capture face images first.")
            return

        # Check for duplicates
        print("[REG PAGE] Step 0: Checking for duplicate users...")
        is_duplicate, existing_user, similarity_score = self.check_for_duplicate()

        if is_duplicate:
            print(f"[REG PAGE] ❌ Registration BLOCKED - Duplicate user detected!")

            dialog = DuplicateUserDialog(existing_user, similarity_score, self)
            dialog.exec()

            self.reset_form()

            print("[REG PAGE] ==================== REGISTRATION ABORTED ====================")
            return

        print("[REG PAGE] ✓ No duplicate found - proceeding with registration")

        # Generate user ID
        from app.workers.face_registration import generate_user_id
        user_id = generate_user_id()
        print(f"[REG PAGE] Generated User ID: {user_id}")
        print(f"[REG PAGE] Name: {name}")

        # Step 1: Register face embeddings
        print(f"[REG PAGE] Step 1: Registering face embeddings...")
        success, message = self.face_registration.register_face(
            user_id, self.captured_embeddings, self.captured_images
        )

        if not success:
            print(f"[REG PAGE] ❌ Face registration FAILED: {message}")
            QMessageBox.critical(self, "Registration Failed", f"Face registration failed:\n{message}")
            return

        print(f"[REG PAGE] ✅ Face embeddings registered")

        # Step 2: Add user to database
        email = self.email_input.text().strip() or None
        phone = self.phone_input.text().strip() or None
        department = self.department_input.text().strip() or None
        role = self.role_combo.currentText()

        print(f"[REG PAGE] Step 2: Adding user to database...")
        db_success = self.user_manager.add_user(user_id, name, email, phone, department, role)

        if not db_success:
            print(f"[REG PAGE] ❌ Database save FAILED")
            QMessageBox.warning(
                self, "Warning",
                "Face registered but failed to add user to database."
            )
            return

        print(f"[REG PAGE] ✅ User added to database")

        # Step 3: Stop camera feed
        print(f"[REG PAGE] Step 3: Stopping camera feed...")
        self.stop_camera_completely()
        print(f"[REG PAGE] ✅ Camera feed stopped")

        # Step 4: Verify registration
        print(f"[REG PAGE] Step 4: Verifying registration...")
        faiss_ok = user_id in self.face_registration.faiss_ids
        db_user = self.user_manager.get_user_by_id(user_id)
        db_ok = db_user is not None
        user_dir = os.path.join('registered_faces', user_id)
        images_ok = os.path.exists(user_dir) and len(os.listdir(user_dir)) > 0

        print(f"[REG PAGE] {'✅' if faiss_ok else '❌'} FAISS check: {faiss_ok}")
        print(f"[REG PAGE] {'✅' if db_ok else '❌'} Database check: {db_ok}")
        print(f"[REG PAGE] {'✅' if images_ok else '❌'} Images saved: {images_ok}")

        # Step 5: Show success
        if faiss_ok and db_ok and images_ok:
            print(f"[REG PAGE] 🎉 ALL VERIFICATIONS PASSED!")
            print("[REG PAGE] ==================== REGISTRATION SUCCESS ====================")

            self.show_success_dialog(user_id, name, email, phone, department, role)
            self.registration_completed.emit(user_id, name)
            self.reset_form()
        else:
            print(f"[REG PAGE] ⚠️ VERIFICATION FAILED!")
            QMessageBox.warning(
                self, "Partial Success",
                f"Registration completed with warnings:\n\n"
                f"FAISS: {'✓' if faiss_ok else '✗'}\n"
                f"Database: {'✓' if db_ok else '✗'}\n"
                f"Images: {'✓' if images_ok else '✗'}"
            )

    def stop_camera_completely(self):
        """Completely stop camera feed and cleanup"""
        print("[REG PAGE] Stopping camera completely...")

        if self.is_capturing:
            self.stop_capture()

        if self.capture_timer:
            self.capture_timer.stop()
            self.capture_timer = None

        if self.camera_worker:
            print("[REG PAGE] Stopping camera worker thread...")
            self.camera_worker.stop()
            self.camera_worker.wait()
            self.camera_worker = None
            print("[REG PAGE] ✓ Camera worker stopped")

        self.camera_label.clear()
        self.camera_label.setText("Camera Stopped")
        self.camera_label.setStyleSheet("""
            background-color: #000; 
            border-radius: 8px; 
            color: #666; 
            font-size: 18px;
        """)

        self.current_frame = None
        self.start_camera_btn.setEnabled(True)
        self.capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(False)

        print("[REG PAGE] ✓ Camera completely stopped")

    def show_success_dialog(self, user_id, name, email, phone, department, role):
        """Show detailed success dialog"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle("Registration Successful! 🎉")

        msg.setText(f"""
        <h2 style='color: #22c55e;'>✅ Face Registered Successfully!</h2>
        <p style='font-size: 14px; color: #94a3b8;'>
            The user has been successfully registered in the system.
        </p>
        """)

        details = f"""
        <div style='background-color: #1a1a1a; padding: 15px; border-radius: 8px; margin-top: 10px;'>
            <table style='width: 100%; border-collapse: collapse;'>
                <tr>
                    <td style='padding: 8px; font-weight: bold; color: #ffc107; width: 120px;'>User ID:</td>
                    <td style='padding: 8px; color: #fff;'>{user_id}</td>
                </tr>
                <tr>
                    <td style='padding: 8px; font-weight: bold; color: #ffc107;'>Name:</td>
                    <td style='padding: 8px; color: #fff;'>{name}</td>
                </tr>
                <tr>
                    <td style='padding: 8px; font-weight: bold; color: #ffc107;'>Role:</td>
                    <td style='padding: 8px; color: #fff;'>{role}</td>
                </tr>
        """

        if email:
            details += f"""
                <tr>
                    <td style='padding: 8px; font-weight: bold; color: #ffc107;'>Email:</td>
                    <td style='padding: 8px; color: #fff;'>{email}</td>
                </tr>
            """

        if phone:
            details += f"""
                <tr>
                    <td style='padding: 8px; font-weight: bold; color: #ffc107;'>Phone:</td>
                    <td style='padding: 8px; color: #fff;'>{phone}</td>
                </tr>
            """

        if department:
            details += f"""
                <tr>
                    <td style='padding: 8px; font-weight: bold; color: #ffc107;'>Department:</td>
                    <td style='padding: 8px; color: #fff;'>{department}</td>
                </tr>
            """

        details += """
            </table>
            <hr style='border: 1px solid #333; margin: 15px 0;'>
            <p style='color: #22c55e; font-weight: bold; margin: 5px 0;'>✓ Face embeddings stored in database</p>
            <p style='color: #22c55e; font-weight: bold; margin: 5px 0;'>✓ User information saved</p>
            <p style='color: #22c55e; font-weight: bold; margin: 5px 0;'>✓ Face images saved to disk</p>
            <p style='color: #22c55e; font-weight: bold; margin: 5px 0;'>✓ All verifications passed</p>
        </div>
        """

        msg.setInformativeText(details)

        msg.setStandardButtons(QMessageBox.Ok)
        ok_button = msg.button(QMessageBox.Ok)
        ok_button.setText("Done")
        ok_button.setStyleSheet("""
            QPushButton {
                background-color: #22c55e;
                color: #000;
                padding: 10px 30px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #16a34a;
            }
        """)

        msg.setMinimumWidth(500)
        msg.exec()

    def reset_form(self):
        """Reset the form"""
        print("[REG PAGE] Resetting form...")

        self.name_input.clear()
        self.email_input.clear()
        self.phone_input.clear()
        self.department_input.clear()
        self.role_combo.setCurrentIndex(0)

        self.captured_embeddings = []
        self.captured_images = []
        self.progress_bar.setValue(0)

        for label in self.preview_labels:
            label.clear()
            label.setStyleSheet("border: 2px dashed #444; border-radius: 8px; background-color: #1a1a1a;")

        self.register_button.setEnabled(False)
        self.status_label.setText("Ready to capture")
        self.status_label.setStyleSheet("color: #64748b; font-size: 14px; font-weight: normal;")

        print("[REG PAGE] ✓ Form reset")

    def full_reset(self):
        """Complete reset - cleanup + reset form + reset UI state"""
        print("[REG PAGE] ==================== FULL RESET ====================")

        if self.is_capturing:
            self.stop_capture()

        if self.camera_worker:
            print("[REG PAGE] Stopping camera worker...")
            self.camera_worker.stop()
            self.camera_worker.wait()
            self.camera_worker = None
            print("[REG PAGE] ✓ Camera worker stopped")

        if self.capture_timer:
            self.capture_timer.stop()
            self.capture_timer = None

        self.capture_worker = None
        self.captured_embeddings = []
        self.captured_images = []
        self.current_frame = None
        self.is_capturing = False

        self.name_input.clear()
        self.email_input.clear()
        self.phone_input.clear()
        self.department_input.clear()
        self.role_combo.setCurrentIndex(0)

        self.progress_bar.setValue(0)

        for label in self.preview_labels:
            label.clear()
            label.setStyleSheet("border: 2px dashed #444; border-radius: 8px; background-color: #1a1a1a;")

        self.start_camera_btn.setEnabled(True)
        self.capture_btn.setEnabled(False)
        self.stop_capture_btn.setEnabled(False)
        self.register_button.setEnabled(False)

        self.status_label.setText("Ready to capture")
        self.status_label.setStyleSheet("color: #64748b; font-size: 14px; font-weight: normal;")

        self.camera_label.clear()
        self.camera_label.setStyleSheet("background-color: #000; border-radius: 8px;")

        print("[REG PAGE] ✓ Full reset completed - Ready for new registration")
        print("[REG PAGE] ================================================================")

    def cleanup(self):
        """Cleanup resources"""
        print("[REG PAGE] Cleaning up resources...")

        if self.is_capturing:
            self.stop_capture()

        if self.camera_worker:
            print("[REG PAGE] Stopping camera worker...")
            self.camera_worker.stop()
            self.camera_worker = None
            print("[REG PAGE] ✓ Camera worker stopped")

        self.current_frame = None

        print("[REG PAGE] ✓ Cleanup complete")

    def showEvent(self, event):
        """Called when page is shown"""
        super().showEvent(event)
        print("[REG PAGE] Page shown")

    def hideEvent(self, event):
        """Called when page is hidden"""
        super().hideEvent(event)
        print("[REG PAGE] Page hidden - cleaning up")
        self.cleanup()