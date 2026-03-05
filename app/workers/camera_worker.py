"""
Fixed Camera Worker - Resolves Empty Face Crop Issue
Key fix: Pass ORIGINAL frame and ORIGINAL bbox to attendance worker
"""
import cv2
import numpy as np
import faiss
import pickle
import os
import time
from datetime import datetime
from queue import Queue, Empty
import threading
from collections import deque
from PySide6.QtCore import QThread, Signal, QMutex
import logging
from app.db.watchlist_manager import WatchlistManager
from app.utils.watchlits_utils import speak_async



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)


class OptimizedCameraWorker(QThread):
    """
    High-performance camera worker with async recognition pipeline
    """

    frame_ready = Signal(np.ndarray, list)
    fps_updated = Signal(float)
    error_occurred = Signal(str)
    attendance_marked = Signal(str, str, float)
    # ✅ ADD NEW SIGNAL
    watchlist_alert = Signal(str, str, float, str)  # user_id, name, score, category

    def __init__(
        self,
        camera_id,
        camera_source,
        faiss_index_path,
        faiss_metadata_path,
        attendance_manager,
        threshold=0.3,
        process_every_n_frames=3,
        resize_width=640,
        max_queue_size=1,
        debug_mode=True
    ):
        super().__init__()

        self.camera_id = camera_id
        self.camera_source = camera_source
        self.faiss_index_path = faiss_index_path
        self.faiss_metadata_path = faiss_metadata_path
        self.attendance_manager = attendance_manager
        self.threshold = threshold
        self.process_every_n_frames = process_every_n_frames
        self.resize_width = resize_width
        self.max_queue_size = max_queue_size
        self.debug_mode = debug_mode



        # Setup logger
        self.logger = logging.getLogger(f"Camera-{camera_id}")
        if debug_mode:
            self.logger.setLevel(logging.DEBUG)

        self.running = False
        self.cap = None
        self.faiss_index = None
        self.faiss_ids = []
        self.app = None

        # Frame management
        self.frame_counter = 0
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        # Result caching
        self.cached_results = []
        self.results_lock = threading.Lock()
        self.last_detection_time = 0

        # Recognition queue and thread
        self.recognition_queue = Queue(maxsize=max_queue_size)
        self.recognition_thread = None

        # Attendance queue and thread
        self.attendance_queue = Queue(maxsize=20)
        self.attendance_thread = None

        # Track last recognition per face
        self.last_recognition_time = {}
        self.recognition_cooldown = 2.0

        # FPS tracking
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0

        # Performance metrics
        self.detection_times = deque(maxlen=30)
        self.recognition_times = deque(maxlen=30)

        # Statistics
        self.total_detections = 0
        self.total_recognitions = 0
        self.total_unknown = 0
        self.total_attendance_marked = 0

        self.mutex = QMutex()

        # Initialize models
        self._init_models()

        # ✅ ADD THESE LINES AFTER YOUR EXISTING INITIALIZATION:

        # Watchlist management
        self.watchlist_cache = {}
        self.last_alarm_time = {}
        self.load_watchlist()

        # Create directory for watchlist images
        os.makedirs('watchlist_events', exist_ok=True)



    def _init_models(self):
        """Initialize face recognition model and FAISS index"""
        try:
            self.logger.info("="*60)
            self.logger.info("INITIALIZING MODELS")
            self.logger.info("="*60)

            self.logger.info("Loading InsightFace model...")
            from app.workers.model_manager import get_shared_model
            self.app = get_shared_model()
            self.logger.info("✓ Using shared model instance")

            # Load FAISS index
            self.logger.info(f"Loading FAISS index from: {self.faiss_index_path}")
            if os.path.exists(self.faiss_index_path):
                self.faiss_index = faiss.read_index(self.faiss_index_path)

                self.logger.info(f"Loading FAISS metadata from: {self.faiss_metadata_path}")
                with open(self.faiss_metadata_path, 'rb') as f:
                    self.faiss_ids = pickle.load(f)

                self.logger.info("="*60)
                self.logger.info(f"✓ FAISS INDEX LOADED")
                self.logger.info(f"  Total faces in database: {self.faiss_index.ntotal}")
                self.logger.info(f"  User IDs loaded: {len(self.faiss_ids)}")
                self.logger.info("="*60)
            else:
                self.logger.warning(f"⚠ No FAISS index found")
                self.faiss_index = faiss.IndexFlatL2(512)
                self.faiss_ids = []

        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Model initialization failed: {str(e)}")

    def _init_camera(self):
        """Initialize camera/video source"""
        try:
            self.logger.info("="*60)
            self.logger.info(f"OPENING CAMERA: {self.camera_source}")
            self.logger.info("="*60)

            if isinstance(self.camera_source, str):
                if self.camera_source.startswith('rtsp') or self.camera_source.startswith('http'):
                    self.cap = cv2.VideoCapture(self.camera_source, cv2.CAP_FFMPEG)
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if self.camera_source.startswith('rtsp'):
                        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
                else:
                    self.cap = cv2.VideoCapture(self.camera_source)
            else:
                self.cap = cv2.VideoCapture(self.camera_source)

            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera: {self.camera_source}")

            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            self.logger.info(f"✓ Camera initialized: {width}x{height} @ {fps}fps")
            self.logger.info("="*60)

            return True

        except Exception as e:
            self.logger.error(f"❌ Camera init failed: {str(e)}", exc_info=True)
            self.error_occurred.emit(f"Camera init failed: {str(e)}")
            return False


    def _resize_frame(self, frame):
        """Resize frame for faster processing"""
        h, w = frame.shape[:2]
        if w > self.resize_width:
            scale = self.resize_width / w
            new_h = int(h * scale)
            return cv2.resize(frame, (self.resize_width, new_h), cv2.INTER_AREA), scale
        return frame, 1.0

    def _recognition_worker(self):
        """Async recognition worker thread"""
        self.logger.info("🚀 Recognition worker started")

        while self.running:
            try:
                item = self.recognition_queue.get(timeout=0.1)

                if item is None:  # Poison pill
                    break

                # FIX: Now receiving original_frame too
                resized_frame, original_frame, frame_id, scale = item

                start_time = time.time()

                # Run detection on RESIZED frame
                # frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                faces = self.app.get(resized_frame, max_num=10)



                detection_time = (time.time() - start_time) * 1000
                self.detection_times.append(detection_time)

                self.logger.info(f"📸 Detected {len(faces)} face(s) in frame #{frame_id}")
                self.total_detections += len(faces)

                results = []
                frame_scores = []  # <-- ADD THIS

                if len(faces) > 0:
                    if self.faiss_index.ntotal == 0:
                        self.logger.warning("⚠ FAISS index is empty")
                        for face in faces:
                            bbox = face.bbox / scale
                            x1, y1, x2, y2 = bbox.astype(int)
                            results.append({
                                'bbox': (x1, y1, x2, y2),
                                'name': "Unknown",
                                'score': 0.0,
                                'user_id': None
                            })
                    else:
                        # Batch embedding extraction
                        embeddings = []
                        face_indices = []

                        for i, face in enumerate(faces):
                            if isinstance(face.normed_embedding, np.ndarray):
                                embeddings.append(face.normed_embedding)
                                face_indices.append(i)

                        if embeddings:
                            self.logger.info(f"🔍 Running FAISS search for {len(embeddings)} embedding(s)...")

                            # Batch FAISS search
                            em_mat = np.stack(embeddings).astype("float32")
                            dist, idx = self.faiss_index.search(em_mat, 1)

                            recognition_time = (time.time() - start_time) * 1000
                            self.recognition_times.append(recognition_time)

                            # Process results
                            self.logger.info("="*60)
                            self.logger.info("RECOGNITION RESULTS")
                            self.logger.info("="*60)

                            for i, face_idx in enumerate(face_indices):

                                face = faces[face_idx]

                                # Get bbox in RESIZED frame coordinates
                                bbox_resized = face.bbox.astype(int)
                                x1_r, y1_r, x2_r, y2_r = bbox_resized

                                # Calculate bbox in ORIGINAL frame coordinates
                                x1_orig = int(x1_r / scale)
                                y1_orig = int(y1_r / scale)
                                x2_orig = int(x2_r / scale)
                                y2_orig = int(y2_r / scale)

                                # Calculate similarity score
                                distance = float(dist[i][0])
                                score = 1 - float(distance / 2)
                                frame_scores.append(score)

                                best_id = idx[i][0]

                                self.logger.info(f"\nFace {i+1}:")
                                self.logger.info(f"  BBox (resized): ({x1_r}, {y1_r}) -> ({x2_r}, {y2_r})")
                                self.logger.info(f"  BBox (original): ({x1_orig}, {y1_orig}) -> ({x2_orig}, {y2_orig})")
                                self.logger.info(f"  Similarity score: {score:.4f}")
                                self.logger.info(f"  Threshold: {self.threshold}")

                                name = "Unknown"
                                user_id = None

                                if score >= self.threshold:
                                    user_id = self.faiss_ids[best_id]
                                    name = user_id

                                    self.logger.info(f"  ✓ MATCH FOUND: {user_id}")
                                    self.total_recognitions += 1

                                    # ✅ ADD WATCHLIST CHECK HERE:
                                    if user_id in self.watchlist_cache:
                                        wl = self.watchlist_cache[user_id]

                                        if score >= wl['threshold'] and wl['alert_enabled']:
                                            current_time = time.time()
                                            last_alarm = self.last_alarm_time.get(user_id, 0)

                                            if current_time - last_alarm >= wl['cooldown_sec']:
                                                self.last_alarm_time[user_id] = current_time

                                                self.logger.warning(f"🚨 WATCHLIST MATCH: {wl['category']}")

                                                # Fire alarm asynchronously
                                                self._trigger_watchlist_alert_async(
                                                    wl, user_id, score, original_frame,
                                                    (x1_orig, y1_orig, x2_orig, y2_orig)
                                                )

                                    # Check attendance cooldown
                                    current_time = time.time()
                                    last_time = self.last_recognition_time.get(user_id, 0)
                                    time_since_last = current_time - last_time

                                    if time_since_last > self.recognition_cooldown:
                                        self.logger.info(f"  📝 Queuing attendance for {user_id}")

                                        # FIX: Pass ORIGINAL frame and ORIGINAL bbox
                                        self._queue_attendance_async(
                                            user_id,
                                            name,
                                            score,
                                            original_frame,  # ← ORIGINAL frame
                                            (x1_orig, y1_orig, x2_orig, y2_orig)  # ← ORIGINAL bbox
                                        )
                                        self.last_recognition_time[user_id] = current_time
                                    else:
                                        self.logger.debug(f"  ⏳ Cooldown active")
                                else:
                                    self.logger.info(f"  ❌ NO MATCH (score {score:.4f} < threshold)")
                                    self.total_unknown += 1

                                # For display: use scaled bbox
                                results.append({
                                    'bbox': (x1_orig, y1_orig, x2_orig, y2_orig),
                                    'name': name,
                                    'score': score,
                                    'user_id': user_id
                                })

                            # After loop
                            if frame_scores:
                                avg_similarity = float(np.mean(frame_scores))
                                max_similarity = float(np.max(frame_scores))
                            else:
                                avg_similarity = 0.0
                                max_similarity = 0.0

                            self.logger.info("="*60)

                            self.logger.info(
                                f"📊 Frame #{frame_id} → Avg similarity: {avg_similarity:.4f}, "
                                f"Max similarity: {max_similarity:.4f}"
                            )

                # Update cached results
                with self.results_lock:
                    self.cached_results = results
                    self.last_detection_time = time.time()

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Recognition error: {e}", exc_info=True)

        self.logger.info("🛑 Recognition worker stopped")

    def _queue_attendance_async(self, user_id, name, score, frame, bbox):
            x1, y1, x2, y2 = bbox
            # face_crop = frame[y1:y2, x1:x2]

            if frame.size == 0:
                return

            if not self.attendance_queue.full():
                self.attendance_queue.put_nowait({
                    'user_id': user_id,
                    'name': name,
                    'score': score,
                    "bbox":bbox,
                    'frame': frame,  # SMALL IMAGE ONLY
                    'timestamp': datetime.now()
                })
                self.logger.debug(f"✓ Queued. Queue size: {self.attendance_queue.qsize()}")
            else:
                self.logger.warning(f"⚠ Attendance queue full!")


    def _attendance_worker(self):
        """Async attendance marking worker"""
        self.logger.info("🚀 Attendance worker started")

        while self.running:
            try:
                item = self.attendance_queue.get(timeout=0.5)

                if item is None:  # Poison pill
                    break

                user_id = item['user_id']
                name = item['name']
                score = item['score']
                frame = item['frame']
                bbox = item['bbox']
                timestamp = item['timestamp']

                self.logger.info("="*60)
                self.logger.info(f"MARKING ATTENDANCE")
                self.logger.info("="*60)
                self.logger.info(f"  User ID: {user_id}")
                self.logger.info(f"  Score: {score:.4f}")
                self.logger.info(f"  Timestamp: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

                # Crop face from ORIGINAL frame with ORIGINAL bbox
                x1, y1, x2, y2 = bbox

                # Ensure bbox is within frame bounds
                frame_h, frame_w = frame.shape[:2]
                x1 = max(0, min(x1, frame_w - 1))
                y1 = max(0, min(y1, frame_h - 1))
                x2 = max(x1 + 1, min(x2, frame_w))
                y2 = max(y1 + 1, min(y2, frame_h))

                self.logger.info(f"  Frame size: {frame_w}x{frame_h}")
                self.logger.info(f"  BBox: ({x1}, {y1}) -> ({x2}, {y2})")
                self.logger.info(f"  Crop size: {x2-x1}x{y2-y1}")

                face_crop = frame[y1:y2, x1:x2]

                if face_crop.size == 0:
                    self.logger.error(f"  ❌ Empty face crop! BBox: ({x1},{y1},{x2},{y2})")
                    continue

                self.logger.info(f"  ✓ Face crop: {face_crop.shape}")

                # Create directory
                img_dir = os.path.join('attendance_images', timestamp.strftime('%Y-%m-%d'))
                os.makedirs(img_dir, exist_ok=True)

                # Save image
                img_filename = f"{user_id}_{timestamp.strftime('%H%M%S')}.jpg"
                img_path = os.path.join(img_dir, img_filename)

                self.logger.info(f"  💾 Saving to: {img_path}")
                cv2.imwrite(img_path, face_crop)

                if os.path.exists(img_path):
                    img_size = os.path.getsize(img_path) / 1024
                    self.logger.info(f"  ✓ Image saved ({img_size:.1f} KB)")
                else:
                    self.logger.error(f"  ❌ Failed to save image")
                    continue

                # Mark attendance in database
                self.logger.info(f"  📝 Marking in database...")
                try:
                    success = self.attendance_manager.mark_attendance(
                        user_id, float(score), img_path, self.camera_id
                    )

                    if success:
                        self.total_attendance_marked += 1
                        self.attendance_marked.emit(user_id, name, score)
                        self.logger.info(f"  ✓ ✓ ✓ ATTENDANCE MARKED SUCCESSFULLY ✓ ✓ ✓")
                        self.logger.info(f"  Total marked today: {self.total_attendance_marked}")
                    else:
                        self.logger.warning(f"  ⚠ Already marked or DB returned False")

                except Exception as e:
                    self.logger.error(f"  ❌ Database error: {e}", exc_info=True)

                self.logger.info("="*60)

            except Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Attendance worker error: {e}", exc_info=True)

        self.logger.info("🛑 Attendance worker stopped")

    def _calculate_fps(self):
        """Calculate and emit FPS"""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time

        if elapsed >= 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_updated.emit(self.current_fps)

            if self.detection_times:
                avg_det = np.mean(self.detection_times)
                avg_rec = np.mean(self.recognition_times) if self.recognition_times else 0

                self.logger.info("="*60)
                self.logger.info("PERFORMANCE METRICS")
                self.logger.info("="*60)
                self.logger.info(f"  FPS: {self.current_fps:.1f}")
                self.logger.info(f"  Avg Detection: {avg_det:.1f}ms")
                self.logger.info(f"  Avg Recognition: {avg_rec:.1f}ms")
                self.logger.info(f"  Total Attendance: {self.total_attendance_marked}")
                self.logger.info("="*60)

            self.fps_frame_count = 0
            self.fps_start_time = time.time()

    # ✅ ADD THIS NEW METHOD
    def load_watchlist(self):
        """Load active watchlist into cache"""
        try:
            items = WatchlistManager.get_active_watchlist()
            self.watchlist_cache = {item['user_id']: item for item in items}

            if self.watchlist_cache:
                self.logger.info(f"📋 Loaded {len(self.watchlist_cache)} watchlist entries")
                for user_id, entry in self.watchlist_cache.items():
                    self.logger.info(f"  - {entry['name']} ({entry['category']})")
            else:
                self.logger.info("📋 No watchlist entries")

        except Exception as e:
            self.logger.error(f"❌ Failed to load watchlist: {e}")
            self.watchlist_cache = {}

    # ✅ ADD THIS NEW METHOD
    def _trigger_watchlist_alert_async(self, wl, user_id, score, frame, bbox):
        """Trigger watchlist alert asynchronously"""

        threading.Thread(
            target=self._handle_watchlist_event,
            daemon=True,
            args=(wl, user_id, score, frame, bbox)
        ).start()

        # ✅ ADD THIS NEW METHOD

    def _handle_watchlist_event(self, wl, user_id, score, frame, bbox):
        """Handle watchlist detection event"""
        try:
            x1, y1, x2, y2 = bbox

            # Ensure bbox is valid
            frame_h, frame_w = frame.shape[:2]
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(x1 + 1, min(x2, frame_w))
            y2 = max(y1 + 1, min(y2, frame_h))

            # Crop face
            crop = frame[y1:y2, x1:x2]

            if crop.size == 0:
                self.logger.error(f"❌ Empty crop for watchlist event")
                return

            # Save image
            timestamp = int(time.time())
            path = f"watchlist_events/{user_id}_{timestamp}.jpg"
            cv2.imwrite(path, crop)

            # Fire alarm
            if wl['alarm_enabled']:
                self._fire_alarm(wl, user_id)

            # Log event in database
            WatchlistManager.log_watchlist_event(
                wl['watchlist_id'],
                user_id,
                self.camera_id,
                float(score),
                path,
                wl['alarm_enabled']
            )

            # Emit signal for UI
            self.watchlist_alert.emit(
                user_id,
                wl['name'],
                float(score),
                wl['category']
            )

            self.logger.warning(f"🚨 WATCHLIST ALERT: {wl['name']} ({wl['category']}) detected!")

        except Exception as e:
            self.logger.error(f"❌ Watchlist event handling error: {e}", exc_info=True)

    # ✅ ADD THIS NEW METHOD
    def _fire_alarm(self, wl, user_id):
        """Fire alarm (software/hardware)"""
        category = wl['category']
        name = wl['name']

        print("\n" + "🚨" * 30)
        print(f"⚠️  WATCHLIST ALARM TRIGGERED  ⚠️")
        print(f"   Category: {category.upper()}")
        print(f"   User: {name} ({user_id})")
        print(f"   Camera: {self.camera_id}")
        print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🚨" * 30 + "\n")

        # Speak alert asynchronously
        alert_text = f"In watchlist, person {name} has been detected"
        # Optional: set a Hindi voice if installed on your system
        speak_async(alert_text, voice=None)  # Replace voice with Hindi voice ID if needed



    def run(self):
        """Main capture loop"""
        if not self._init_camera():
            return

        self.running = True

        # Start worker threads
        self.recognition_thread = threading.Thread(
            target=self._recognition_worker,
            daemon=True
        )
        self.recognition_thread.start()

        self.attendance_thread = threading.Thread(
            target=self._attendance_worker,
            daemon=True
        )
        self.attendance_thread.start()

        self.logger.info("="*60)
        self.logger.info("✓ CAMERA WORKER FULLY INITIALIZED")
        self.logger.info("="*60)

        skip_counter = 0

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                self.logger.warning("Failed to read frame, attempting reconnect...")
                self.cap.release()
                time.sleep(1)
                if not self._init_camera():
                    break
                continue

            # ✅ ALWAYS increment frame counter
            self.frame_counter += 1

            # Resize for display
            display_frame, scale = self._resize_frame(frame)

            # -------------------------------
            # Recognition throttling ONLY
            # -------------------------------
            if (
                    self.frame_counter % self.process_every_n_frames == 0
                    and self.faiss_index.ntotal > 0
            ):
                try:
                    if self.recognition_queue.full():
                        self.recognition_queue.get_nowait()

                    self.recognition_queue.put_nowait(
                        (display_frame, frame, self.frame_counter, scale)
                    )
                except Exception:
                    pass

            # Get cached results
            with self.results_lock:
                current_results = self.cached_results.copy()

            # FPS
            self._calculate_fps()

            # ✅ ALWAYS EMIT FRAME
            self.frame_ready.emit(display_frame.copy(), current_results)

            # tiny sleep is OK
            time.sleep(0.001)

        # Cleanup
        self._stop_workers()

        if self.cap:
            self.cap.release()

        self.logger.info("✓ Camera worker stopped")

    def _stop_workers(self):
        """Stop all worker threads gracefully"""
        self.logger.info("Stopping workers...")

        try:
            self.recognition_queue.put(None, timeout=1)
        except:
            pass

        try:
            self.attendance_queue.put(None, timeout=1)
        except:
            pass

        if self.recognition_thread:
            self.recognition_thread.join(timeout=2)

        if self.attendance_thread:
            self.attendance_thread.join(timeout=2)

    def stop(self):
        """Stop the worker thread"""
        self.logger.info("Stop requested")
        self.running = False
        self.wait(5000)


def draw_detections(frame, results, fps=0):
    """Draw bounding boxes and labels on frame"""
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        name = result['name']
        score = float(result['score'])

        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{name} {score:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if fps > 0:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame