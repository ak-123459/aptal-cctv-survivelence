"""
Registration Camera Worker - Non-blocking camera for registration with quality profile support
Enhanced to apply camera quality profiles to captured frames before embedding creation

IMPORTANT: Save this file as: app/workers/registration_camera_worker.py
"""
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
import time


class RegistrationCameraWorker(QThread):
    """
    Background thread for registration camera feed
    Sends processed frames to UI without blocking
    """

    frame_ready = Signal(np.ndarray, bool, str, float)  # frame, is_good, message, quality
    error_occurred = Signal(str)

    def __init__(self, camera_source, face_registration, parent=None):
        super().__init__(parent)
        self.camera_source = camera_source
        self.face_registration = face_registration
        self.running = False
        self.cap = None
        self.process_every_n_frames = 3  # Only verify every 3rd frame
        self.frame_count = 0

        # Cache last verification result
        self.last_verification = (False, "Starting...", 0.0)

    def run(self):

        """Main camera loop in background thread"""

        try:
            self.cap = cv2.VideoCapture(self.camera_source)

            # ADD THIS debug check:
            if not self.cap.isOpened():
                print(f"[CAM WORKER] ❌ Failed to open camera {self.camera_source}")
                self.error_occurred.emit(f"Failed to open camera {self.camera_source}")
                return

            print(f"[CAM WORKER] ✅ Camera {self.camera_source} opened successfully")
            print(f"[CAM WORKER]   Width: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
            print(f"[CAM WORKER]   Height: {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

            # Optimize camera settings
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.running = True

            while self.running:
                ret, frame = self.cap.read()

                if not ret:
                    time.sleep(0.01)
                    continue

                # Only verify quality every N frames to reduce CPU load
                should_verify = (self.frame_count % self.process_every_n_frames == 0)

                if should_verify:
                    is_good, message, quality = self.face_registration.verify_face_quality(frame)
                    self.last_verification = (is_good, message, quality)
                else:
                    # Use cached result
                    is_good, message, quality = self.last_verification

                # Draw feedback on frame (fast operations only)
                frame = self._draw_feedback(frame, is_good, message, quality)

                # Emit frame to UI
                self.frame_ready.emit(frame, is_good, message, quality)

                self.frame_count += 1

                # Small sleep to prevent CPU overload
                time.sleep(0.01)  # ~100 FPS max (UI will throttle this)

        except Exception as e:
            self.error_occurred.emit(f"Camera error: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()

    def _draw_feedback(self, frame, is_good, message, quality):
        """Draw feedback overlay on frame (optimized)"""
        color = (0, 255, 0) if is_good else (0, 165, 255)
        thickness = 3 if is_good else 2

        h, w = frame.shape[:2]

        # Draw border
        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), color, thickness)

        # Draw text with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(message, font, 0.8, 2)[0]

        # Background rectangle
        cv2.rectangle(frame, (15, 25), (25 + text_size[0], 55), (0, 0, 0), -1)

        # Text
        cv2.putText(frame, message, (20, 45), font, 0.8, color, 2)

        if is_good:
            quality_text = f"Quality: {quality:.2f}"
            cv2.putText(frame, quality_text, (20, 80), font, 0.6, (0, 255, 255), 2)

            # Ready indicator
            cv2.circle(frame, (w - 40, 40), 15, (0, 255, 0), -1)

        return frame

    def stop(self):
        """Stop the camera worker"""
        self.running = False
        self.wait()  # Wait for thread to finish


class RegistrationCaptureWorker(QThread):
    """
    Background thread for capturing face embeddings with quality profile application
    Non-blocking face extraction with camera profile matching

    NEW PARAMETERS:
    - quality_matcher: GlobalQualityMatcher instance (optional)
    - profile_intensity: Intensity of profile application (0.0 to 1.0)
    """

    capture_completed = Signal(object, object)  # embedding, face_crop
    capture_failed = Signal(str)

    def __init__(self, frame, face_registration, quality_matcher=None, profile_intensity=0.7, parent=None):
        """
        Initialize capture worker

        Args:
            frame: Camera frame to process
            face_registration: FaceRegistration instance
            quality_matcher: GlobalQualityMatcher instance (optional)
            profile_intensity: How strongly to apply profile (0.0-1.0)
            parent: Parent QObject
        """
        super().__init__(parent)
        self.frame = frame.copy()  # Copy to avoid threading issues
        self.face_registration = face_registration
        self.quality_matcher = quality_matcher
        self.profile_intensity = profile_intensity

    def run(self):
        """Extract face in background with quality profile application"""
        try:
            print("[CAPTURE WORKER] ==================== CAPTURE START ====================")
            print("[CAPTURE WORKER] Starting face extraction...")

            # Step 1: Extract face from original frame
            success, bbox, embedding, face_crop, message = \
                self.face_registration.extract_face_from_frame(self.frame)

            if not success:
                print(f"[CAPTURE WORKER] ❌ Face extraction failed: {message}")
                print("[CAPTURE WORKER] ==================== CAPTURE FAILED ====================")
                self.capture_failed.emit(message)
                return

            print(f"[CAPTURE WORKER] ✓ Face extracted successfully")
            print(f"[CAPTURE WORKER]   BBox: {bbox}")
            print(f"[CAPTURE WORKER]   Face crop size: {face_crop.shape}")
            print(f"[CAPTURE WORKER]   Original embedding shape: {embedding.shape}")

            # Step 2: Apply quality profile if available
            if self.quality_matcher and self.quality_matcher.has_profile():
                print("[CAPTURE WORKER] " + "="*40)
                print("[CAPTURE WORKER] 🎨 APPLYING QUALITY PROFILE TO FACE CROP")
                print("[CAPTURE WORKER] " + "="*40)

                try:
                    # Get profile info
                    profile = self.quality_matcher.quality_profile
                    print(f"[CAPTURE WORKER] Profile loaded:")
                    print(f"[CAPTURE WORKER]   Brightness: {profile.get('brightness', 0):.2f}")
                    print(f"[CAPTURE WORKER]   Contrast: {profile.get('contrast', 0):.2f}")
                    print(f"[CAPTURE WORKER]   Noise: {profile.get('noise_level', 0):.4f}")
                    print(f"[CAPTURE WORKER]   Blur: {profile.get('blur_amount', 0):.2f}")
                    print(f"[CAPTURE WORKER]   Saturation: {profile.get('saturation', 0):.2f}")
                    print(f"[CAPTURE WORKER]   Compression: {profile.get('compression_quality', 0):.0f}")
                    print(f"[CAPTURE WORKER]   Intensity: {self.profile_intensity}")

                    # Apply quality properties to the face crop
                    enhanced_face_crop = self.quality_matcher.apply_quality_to_frame(
                        face_crop,
                        intensity=self.profile_intensity
                    )

                    print(f"[CAPTURE WORKER] ✓ Quality profile applied successfully")
                    print(f"[CAPTURE WORKER]   Original crop size: {face_crop.shape}")
                    print(f"[CAPTURE WORKER]   Enhanced crop size: {enhanced_face_crop.shape}")

                    # Step 3: Re-extract embedding from enhanced face crop
                    print("[CAPTURE WORKER] 🔄 Re-extracting embedding from enhanced face...")

                    # Convert to RGB for InsightFace
                    enhanced_rgb = cv2.cvtColor(enhanced_face_crop, cv2.COLOR_BGR2RGB)

                    # Get faces from enhanced crop
                    faces = self.face_registration.app.get(enhanced_rgb)

                    if len(faces) > 0:
                        # Use the first (and likely only) face
                        enhanced_embedding = faces[0].normed_embedding

                        print(f"[CAPTURE WORKER] ✓ Enhanced embedding extracted")
                        print(f"[CAPTURE WORKER]   Enhanced embedding shape: {enhanced_embedding.shape}")

                        # Compare original vs enhanced embedding
                        similarity = np.dot(embedding, enhanced_embedding)
                        print(f"[CAPTURE WORKER]   Original vs Enhanced similarity: {similarity:.4f}")

                        # Use enhanced versions
                        final_embedding = enhanced_embedding
                        final_face_crop = enhanced_face_crop

                        print("[CAPTURE WORKER] ✓ Using enhanced embedding and face crop")
                        print("[CAPTURE WORKER] " + "="*40)
                    else:
                        print("[CAPTURE WORKER] ⚠ No face detected in enhanced crop")
                        print("[CAPTURE WORKER] ⚠ Falling back to original")
                        final_embedding = embedding
                        final_face_crop = face_crop

                except Exception as e:
                    print(f"[CAPTURE WORKER ERROR] Profile application failed: {e}")
                    import traceback
                    traceback.print_exc()
                    print("[CAPTURE WORKER] ⚠ Falling back to original")
                    final_embedding = embedding
                    final_face_crop = face_crop
            else:
                print("[CAPTURE WORKER] No quality profile loaded - using original")
                final_embedding = embedding
                final_face_crop = face_crop

            # Step 4: Emit the final result
            print("[CAPTURE WORKER] " + "="*40)
            print("[CAPTURE WORKER] ✓ CAPTURE COMPLETED SUCCESSFULLY")
            print(f"[CAPTURE WORKER]   Final embedding shape: {final_embedding.shape}")
            print(f"[CAPTURE WORKER]   Final face crop shape: {final_face_crop.shape}")
            print("[CAPTURE WORKER] ==================== CAPTURE END ====================")

            self.capture_completed.emit(final_embedding, final_face_crop)

        except Exception as e:
            error_msg = f"Extraction error: {str(e)}"
            print(f"[CAPTURE WORKER ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            print("[CAPTURE WORKER] ==================== CAPTURE ERROR ====================")
            self.capture_failed.emit(error_msg)