#!/usr/bin/env python3
"""
AI CCTV Backend - v4.1 (cloud light build)
"""

import os
from pathlib import Path as _PathForEnv

# Optional .env next to this file (before Config reads os.environ)
_env_file = _PathForEnv(__file__).resolve().parent / '.env'
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if not _line or _line.startswith('#') or '=' not in _line:
                continue
            _k, _, _v = _line.partition('=')
            os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

import sqlite3
import uuid
import base64
import subprocess
import threading
import time
import jwt
import logging
import re
import csv
import io
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Generator
from enum import Enum

from flask import Flask, jsonify, request, Response, send_file, send_from_directory, g
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# Path to the built React frontend


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    BASE_DIR = Path(__file__).parent
    MODEL_DIR = Path(os.getenv('MODEL_DIR', str(BASE_DIR / 'models')))
    UPLOAD_DIR = Path(os.getenv('UPLOAD_DIR', str(BASE_DIR / 'uploads')))
    DB_PATH = os.getenv('DB_PATH', str(BASE_DIR / 'aicctv.db'))
    JWT_SECRET = os.getenv('JWT_SECRET', 'ai-cctv-secret-key-2024')
    JWT_EXPIRY_HOURS = int(os.getenv('JWT_EXPIRY_HOURS', 24))
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*')
    RATE_LIMIT_REQUESTS = int(os.getenv('RATE_LIMIT_REQUESTS', 100))
    RATE_LIMIT_WINDOW = int(os.getenv('RATE_LIMIT_WINDOW', 60))
    ML_DEVICE = os.getenv('ML_DEVICE', 'auto')
    ML_DEFAULT_CONFIDENCE = float(os.getenv('ML_CONFIDENCE', 0.5))
    ML_MAX_DETECTIONS = int(os.getenv('ML_MAX_DETECTIONS', 50))
    COMPRESSION_TIMEOUT = int(os.getenv('COMPRESSION_TIMEOUT', 600))
    MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_MB', 500)) * 1024 * 1024
    EDGE_SYNC_SECRET = os.getenv('EDGE_SYNC_SECRET', 'change-me-edge-sync')
    EDGE_API_URL = os.getenv('EDGE_API_URL', '').rstrip('/')
    CLOUD_MODE = True
    # Override on PythonAnywhere if frontend lives elsewhere
    FRONTEND_DIST = Path(os.getenv(
        'FRONTEND_DIST',
        str(Path(__file__).resolve().parent / 'static'),
    ))

Config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('ai_cctv')

# ============================================================================
# ML INITIALIZATION
# ============================================================================

DEVICE = 'cpu'
YOLO = None
YOLO_AVAILABLE = False
logger.warning("⚠️ YOLO not available")

# Optional barcode support
try:
    from pyzbar import pyzbar
    BARCODE_AVAILABLE = True
except ImportError:
    BARCODE_AVAILABLE = False

# ============================================================================
# EXCEPTIONS
# ============================================================================

class APIError(Exception):
    def __init__(self, message: str, status_code: int = 400, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details
    def to_dict(self) -> dict:
        result = {'error': self.message, 'status_code': self.status_code}
        if self.details: result['details'] = self.details
        return result

class ValidationError(APIError):
    def __init__(self, message: str, errors: dict = None): super().__init__(message, 400, errors)

class AuthenticationError(APIError):
    def __init__(self, message: str = "Authentication required"): super().__init__(message, 401)

class AuthorizationError(APIError):
    def __init__(self, message: str = "Permission denied"): super().__init__(message, 403)

class NotFoundError(APIError):
    def __init__(self, resource: str = "Resource"): super().__init__(f"{resource} not found", 404)

# ============================================================================
# VALIDATION
# ============================================================================

class Validator:
    def __init__(self, data: dict):
        self.data = data or {}
        self.errors: Dict[str, List[str]] = {}
    
    def require(self, field: str, message: str = None) -> 'Validator':
        value = self.data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            self._add_error(field, message or f"{field} is required")
        return self
    
    def email(self, field: str) -> 'Validator':
        value = self.data.get(field, '')
        if value and not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            self._add_error(field, "Invalid email format")
        return self
    
    def min_length(self, field: str, length: int) -> 'Validator':
        value = self.data.get(field, '')
        if value and len(str(value)) < length:
            self._add_error(field, f"{field} must be at least {length} characters")
        return self
    
    def max_length(self, field: str, length: int) -> 'Validator':
        value = self.data.get(field, '')
        if value and len(str(value)) > length:
            self._add_error(field, f"{field} must be at most {length} characters")
        return self
    
    def in_list(self, field: str, allowed: list) -> 'Validator':
        value = self.data.get(field)
        if value is not None and value not in allowed:
            self._add_error(field, f"{field} must be one of: {', '.join(map(str, allowed))}")
        return self
    
    def _add_error(self, field: str, message: str) -> None:
        if field not in self.errors: self.errors[field] = []
        self.errors[field].append(message)
    
    def validate(self) -> dict:
        if self.errors: raise ValidationError("Validation failed", self.errors)
        return self.data

# ============================================================================
# RATE LIMITER
# ============================================================================

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def is_allowed(self, key: str) -> Tuple[bool, int]:
        now = time.time()
        with self._lock:
            if key not in self.requests: self.requests[key] = []
            self.requests[key] = [t for t in self.requests[key] if now - t < self.window]
            remaining = self.max_requests - len(self.requests[key])
            if len(self.requests[key]) >= self.max_requests: return False, 0
            self.requests[key].append(now)
            return True, remaining - 1
    
    def get_key(self) -> str:
        return getattr(request, 'user_id', None) or request.remote_addr or 'anonymous'
    
    def cleanup(self) -> None:
        """Remove stale keys to prevent unbounded memory growth."""
        now = time.time()
        with self._lock:
            stale = [k for k, ts in self.requests.items()
                     if not ts or all(now - t >= self.window for t in ts)]
            for k in stale:
                del self.requests[k]

rate_limiter = RateLimiter(Config.RATE_LIMIT_REQUESTS, Config.RATE_LIMIT_WINDOW)

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._local = threading.local()
    
    @property
    def connection(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'connection') or self._local.connection is None:
            self._local.connection = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA foreign_keys = ON")
        return self._local.connection
    
    def close(self) -> None:
        if hasattr(self._local, 'connection') and self._local.connection:
            self._local.connection.close()
            self._local.connection = None
    
    @contextmanager
    def get_cursor(self):
        cursor = self.connection.cursor()
        try:
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()
    
    def execute(self, query: str, params: tuple = ()) -> sqlite3.Cursor:
        return self.connection.execute(query, params)
    
    def fetchone(self, query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        return self.execute(query, params).fetchone()
    
    def fetchall(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        return self.execute(query, params).fetchall()
    
    def insert(self, table: str, data: dict) -> str:
        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        with self.get_cursor() as cursor:
            cursor.execute(query, tuple(data.values()))
            return data.get('id', cursor.lastrowid)
    
    def update(self, table: str, data: dict, where: str, where_params: tuple) -> int:
        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])
        query = f"UPDATE {table} SET {set_clause} WHERE {where}"
        with self.get_cursor() as cursor:
            cursor.execute(query, tuple(data.values()) + where_params)
            return cursor.rowcount
    
    def delete(self, table: str, where: str = "1=1", params: tuple = ()) -> int:
        with self.get_cursor() as cursor:
            cursor.execute(f"DELETE FROM {table} WHERE {where}", params)
            return cursor.rowcount
    
    def init_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'viewer',
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            confidence REAL NOT NULL,
            direction TEXT,
            inference_ms REAL,
            camera_id TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS inventory (
            id TEXT PRIMARY KEY,
            product_name TEXT UNIQUE NOT NULL,
            count_in INTEGER DEFAULT 0,
            count_out INTEGER DEFAULT 0,
            current_stock INTEGER DEFAULT 0,
            min_threshold INTEGER DEFAULT 10,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS trucks (
            id TEXT PRIMARY KEY,
            plate_number TEXT NOT NULL,
            direction TEXT DEFAULT 'IN',
            driver_name TEXT,
            company TEXT,
            purpose TEXT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            exit_at TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS scans (
            id TEXT PRIMARY KEY,
            barcode TEXT,
            product_name TEXT,
            batch_number TEXT,
            mfg_date TEXT,
            exp_date TEXT,
            rack_no TEXT,
            shelf_no TEXT,
            direction TEXT DEFAULT 'IN',
            scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            encoding BLOB,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT DEFAULT 'info',
            is_read INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- ── Phase-1 supply-chain tables ─────────────────────────────
        CREATE TABLE IF NOT EXISTS bays (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            camera_source INTEGER DEFAULT 0,
            location TEXT,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS shifts (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            operator_name TEXT,
            notes TEXT
        );

        CREATE TABLE IF NOT EXISTS loading_jobs (
            id TEXT PRIMARY KEY,
            bay_id TEXT,
            truck_id TEXT,
            truck_plate TEXT NOT NULL,
            product_name TEXT DEFAULT 'Sugar Bag',
            target_count INTEGER NOT NULL,
            loaded_count INTEGER DEFAULT 0,
            direction TEXT DEFAULT 'OUT',
            shift_id TEXT,
            status TEXT DEFAULT 'pending',
            operator_name TEXT,
            notes TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (bay_id) REFERENCES bays(id),
            FOREIGN KEY (shift_id) REFERENCES shifts(id)
        );

        -- Extend detections to carry job/shift context
        -- (ALTER TABLE is idempotent via the try in seed_data)

        CREATE INDEX IF NOT EXISTS idx_detections_date ON detections(detected_at);
        CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(type);
        CREATE INDEX IF NOT EXISTS idx_detections_direction ON detections(direction);
        CREATE INDEX IF NOT EXISTS idx_trucks_date ON trucks(detected_at);
        CREATE INDEX IF NOT EXISTS idx_trucks_plate ON trucks(plate_number);
        CREATE INDEX IF NOT EXISTS idx_scans_date ON scans(scanned_at);
        CREATE INDEX IF NOT EXISTS idx_scans_barcode ON scans(barcode);
        CREATE INDEX IF NOT EXISTS idx_alerts_read ON alerts(is_read);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON loading_jobs(status);
        CREATE INDEX IF NOT EXISTS idx_jobs_bay ON loading_jobs(bay_id);
        CREATE INDEX IF NOT EXISTS idx_shifts_date ON shifts(started_at);
        """
        with self.get_cursor() as cursor:
            cursor.executescript(schema)
        # Add columns that may not exist in older DBs (safe migration)
        for col_sql in [
            "ALTER TABLE detections ADD COLUMN shift_id TEXT",
            "ALTER TABLE detections ADD COLUMN job_id TEXT",
            "ALTER TABLE detections ADD COLUMN bay_id TEXT",
        ]:
            try:
                with self.get_cursor() as c:
                    c.execute(col_sql)
            except Exception:
                pass  # column already exists
        logger.info("✅ Database schema initialized")

    
    def seed_data(self) -> None:
        if not self.fetchone("SELECT id FROM users WHERE email = ?", ('demo@aicctv.com',)):
            self.insert('users', {
                'id': str(uuid.uuid4()),
                'email': 'demo@aicctv.com',
                'password_hash': generate_password_hash('demo123'),
                'name': 'Demo User',
                'role': 'admin'
            })
            logger.info("✅ Demo user created (demo@aicctv.com / demo123)")
        
        for product in ['Sugar Bag', 'Full Crate', 'Half Crate', 'Box', 'Pallet']:
            if not self.fetchone("SELECT id FROM inventory WHERE product_name = ?", (product,)):
                self.insert('inventory', {'id': str(uuid.uuid4()), 'product_name': product})
        logger.info("✅ Seed data loaded")

    def seed_sample_demo(self) -> dict:
        """Extra warehouse demo rows for dashboards / first PA deploy. Idempotent by plate numbers."""
        self.seed_data()
        created = {'trucks': 0, 'detections': 0, 'alerts': 0}

        sample_trucks = [
            ('T-1001', 'IN'),
            ('T-1002', 'IN'),
            ('T-2044', 'OUT'),
        ]
        for num, direction in sample_trucks:
            if self.fetchone("SELECT id FROM trucks WHERE plate_number = ?", (num,)):
                continue
            self.insert('trucks', {
                'id': str(uuid.uuid4()),
                'plate_number': num,
                'direction': direction,
                'company': 'Demo Logistics',
            })
            created['trucks'] += 1

        # A few detections so analytics is not empty
        if not self.fetchone("SELECT id FROM detections LIMIT 1"):
            for direction, n in (('IN', 5), ('OUT', 3)):
                for _ in range(n):
                    self.insert('detections', {
                        'id': str(uuid.uuid4()),
                        'type': 'sugar_bag',
                        'confidence': 0.92,
                        'direction': direction,
                        'inference_ms': 45.0,
                    })
                    update_inventory_from_detection('sugar_bag', direction, 1)
                    created['detections'] += 1

        if not self.fetchone("SELECT id FROM alerts LIMIT 1"):
            self.insert('alerts', {
                'id': str(uuid.uuid4()),
                'type': 'info',
                'message': 'Sample alert: edge sync connected (demo)',
                'severity': 'info',
            })
            created['alerts'] += 1

        logger.info(f"✅ Sample demo data: {created}")
        return created

db = Database(Config.DB_PATH)

# ============================================================================
# HELPERS
# ============================================================================

def create_alert(alert_type: str, message: str, severity: str = 'info'):
    """Create a new alert"""
    db.insert('alerts', {
        'id': str(uuid.uuid4()),
        'type': alert_type,
        'message': message,
        'severity': severity
    })

def save_detection_to_db(detection_type: str, confidence: float, direction: str = None, inference_ms: float = 0):
    """Save a detection to database"""
    try:
        db.insert('detections', {
            'id': str(uuid.uuid4()),
            'type': detection_type,
            'confidence': confidence,
            'direction': direction,
            'inference_ms': inference_ms
        })
    except Exception as e:
        logger.error(f"Failed to save detection: {e}")

def update_inventory_from_detection(product_type: str, direction: str = 'IN', count: int = 1):
    """Update inventory based on detection"""
    label_to_product = {
        'sugar_bag': 'Sugar Bag', 'sugar': 'Sugar Bag',
        'full_crate': 'Full Crate', 'crate': 'Full Crate',
        'half_crate': 'Half Crate', 'box': 'Box', 'pallet': 'Pallet'
    }
    product_name = label_to_product.get(product_type.lower(), product_type)
    
    item = db.fetchone("SELECT * FROM inventory WHERE LOWER(product_name) = LOWER(?)", (product_name,))
    
    if not item:
        db.insert('inventory', {
            'id': str(uuid.uuid4()),
            'product_name': product_name,
            'count_in': count if direction == 'IN' else 0,
            'count_out': count if direction == 'OUT' else 0,
            'current_stock': count if direction == 'IN' else 0
        })
    else:
        with db.get_cursor() as cursor:
            if direction == 'IN':
                cursor.execute("UPDATE inventory SET count_in = count_in + ?, current_stock = current_stock + ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (count, count, item['id']))
            else:
                cursor.execute("UPDATE inventory SET count_out = count_out + ?, current_stock = MAX(0, current_stock - ?), updated_at = CURRENT_TIMESTAMP WHERE id = ?", (count, count, item['id']))

def check_low_stock_alerts():
    """Check inventory for low stock and create alerts.
    Uses 2 queries total instead of N+1 LIKE scans.
    """
    low_items = db.fetchall("SELECT product_name, current_stock, min_threshold FROM inventory WHERE current_stock < min_threshold")
    if not low_items:
        return
    today = datetime.now().strftime('%Y-%m-%d')
    # Single query for all low_stock messages today; check membership in Python
    existing_rows = db.fetchall("SELECT message FROM alerts WHERE type = 'low_stock' AND DATE(created_at) = ?", (today,))
    alerted_today = {row['message'] for row in existing_rows}
    for item in low_items:
        msg = f"Low stock: {item['product_name']} has only {item['current_stock']} units (threshold: {item['min_threshold']})"
        if msg not in alerted_today:
            create_alert('low_stock', msg, 'warning')

# ============================================================================
# AUTHENTICATION
# ============================================================================

def create_token(user_id: str, role: str, name: str) -> str:
    return jwt.encode({'user_id': user_id, 'role': role, 'name': name, 'exp': datetime.utcnow() + timedelta(hours=Config.JWT_EXPIRY_HOURS)}, Config.JWT_SECRET, algorithm='HS256')

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, Config.JWT_SECRET, algorithms=['HS256'])
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token expired")
    except jwt.InvalidTokenError:
        raise AuthenticationError("Invalid token")

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            try:
                payload = decode_token(auth_header[7:])
                request.user_id = payload['user_id']
                request.user_role = payload['role']
                request.user_name = payload.get('name', '')
            except:
                request.user_id = 'demo'
                request.user_role = 'admin'
                request.user_name = 'Demo'
        else:
            request.user_id = 'demo'
            request.user_role = 'admin'
            request.user_name = 'Demo'
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer '):
            try:
                payload = decode_token(auth_header[7:])
                if payload.get('role') != 'admin': raise AuthorizationError()
                request.user_id = payload['user_id']
                request.user_role = payload['role']
            except AuthorizationError:
                raise
            except:
                request.user_id = 'demo'
                request.user_role = 'admin'
        else:
            request.user_id = 'demo'
            request.user_role = 'admin'
        return f(*args, **kwargs)
    return decorated

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        allowed, remaining = rate_limiter.is_allowed(rate_limiter.get_key())
        if not allowed: return jsonify({'error': 'Rate limit exceeded'}), 429
        return f(*args, **kwargs)
    return decorated

# ============================================================================
# ML SERVICE
# ============================================================================

@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: List[int]
    inference_ms: float

    def to_dict(self) -> dict:
        return {
            'label': self.label,
            'confidence': round(self.confidence, 3),
            'bbox': self.bbox,
            'inference_ms': round(self.inference_ms, 1),
        }


class MLService:
    """Cloud stub — YOLO runs on the Windows edge box only."""

    def __init__(self):
        self._latest_detections: List[DetectionResult] = []
        self._active_model = None
        self.loaded_models: Dict[str, dict] = {}

    @property
    def device(self) -> str:
        return 'cloud'

    @property
    def active_model_name(self) -> Optional[str]:
        return self._active_model

    @property
    def latest_detections(self) -> List[dict]:
        return [d.to_dict() for d in self._latest_detections]

    def register_model(self, path: Path, name: str = None) -> bool:
        return False

    def register_models_from_directory(self, directory: Path) -> int:
        return 0

    def switch_model(self, name: str) -> bool:
        return False

    def load_model(self, path: Path, name: str = None) -> bool:
        return False

    def load_models_from_directory(self, directory: Path) -> int:
        return 0

    def get_active_model(self) -> Any:
        return None

    def detect(self, frame, confidence: float = 0.5, draw: bool = True):
        return frame, []

    def health_check(self) -> dict:
        return {
            'device': 'cloud',
            'mode': 'light',
            'models_loaded': 0,
            'active_model': None,
            'message': 'ML runs on edge; this is PythonAnywhere cloud API',
        }


ml_service = MLService()


class VideoCamera:
    """Cloud stub — live camera runs on edge only."""

    def __init__(self, source: int = 0):
        raise RuntimeError('Camera is not available on cloud. Start camera on the edge server.')

    def get_frame(self):
        return None

    def get_encoded_frame(self):
        return None

    def stop(self):
        pass


camera: Optional[VideoCamera] = None


class CameraManager:
    def __init__(self):
        self.cameras: Dict[str, VideoCamera] = {}

    def get_or_create(self, bay_id: str, source: int = 0) -> VideoCamera:
        raise RuntimeError('Bay cameras run on edge only')

    def stop(self, bay_id: str):
        pass

    def stop_all(self):
        pass


camera_manager = CameraManager()


class JobStatus(Enum):
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'

@dataclass
class CompressionJob:
    id: str
    input_path: Path
    output_path: Path
    status: JobStatus = JobStatus.PROCESSING
    original_size: int = 0
    compressed_size: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        result = {'job_id': self.id, 'status': self.status.value, 'original_size': self.original_size}
        if self.status == JobStatus.COMPLETED:
            ratio = (1 - self.compressed_size / self.original_size) * 100 if self.original_size > 0 else 0
            result['compressed_size'] = self.compressed_size
            result['ratio'] = round(ratio, 1)
        if self.error: result['error'] = self.error
        return result

compression_jobs: Dict[str, CompressionJob] = {}

# ============================================================================
# FLASK APP
# ============================================================================
