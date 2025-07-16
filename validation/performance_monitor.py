"""
Performance Monitoring and Metrics Collection for Ultimate Agentic StarterKit.

This module provides real-time performance tracking, quality metrics collection,
trend analysis, and performance optimization suggestions.
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
import json
import sqlite3
from pathlib import Path
import statistics
import tracemalloc
import gc

from core.logger import get_logger
from core.config import get_config
from core.voice_alerts import get_voice_alerts


@dataclass
class PerformanceMetric:
    """Individual performance metric"""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    thread_count: int
    process_count: int
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert"""
    metric_name: str
    threshold: float
    current_value: float
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime
    resolved: bool = False


class PerformanceCollector:
    """Collects system performance metrics"""
    
    def __init__(self):
        self.logger = get_logger("performance_collector")
        self.process = psutil.Process()
        self.last_disk_io = None
        self.last_network_io = None
        self.last_timestamp = None
        
    async def collect_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance snapshot"""
        try:
            current_time = datetime.now()
            
            # CPU and Memory
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            memory_used_mb = memory_info.rss / (1024 * 1024)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if self.last_disk_io and self.last_timestamp:
                time_delta = (current_time - self.last_timestamp).total_seconds()
                disk_read_mb = (disk_io.read_bytes - self.last_disk_io.read_bytes) / (1024 * 1024 * time_delta)
                disk_write_mb = (disk_io.write_bytes - self.last_disk_io.write_bytes) / (1024 * 1024 * time_delta)
            else:
                disk_read_mb = 0.0
                disk_write_mb = 0.0
            
            # Network I/O
            network_io = psutil.net_io_counters()
            if self.last_network_io and self.last_timestamp:
                time_delta = (current_time - self.last_timestamp).total_seconds()
                network_sent_mb = (network_io.bytes_sent - self.last_network_io.bytes_sent) / (1024 * 1024 * time_delta)
                network_recv_mb = (network_io.bytes_recv - self.last_network_io.bytes_recv) / (1024 * 1024 * time_delta)
            else:
                network_sent_mb = 0.0
                network_recv_mb = 0.0
            
            # Thread and process counts
            thread_count = self.process.num_threads()
            process_count = len(psutil.pids())
            
            # Store current values for next calculation
            self.last_disk_io = disk_io
            self.last_network_io = network_io
            self.last_timestamp = current_time
            
            return PerformanceSnapshot(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                thread_count=thread_count,
                process_count=process_count
            )
            
        except Exception as e:
            self.logger.error(f"Performance collection failed: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                thread_count=0,
                process_count=0
            )


class MetricsStorage:
    """Storage for performance metrics"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or "performance_metrics.db"
        self.logger = get_logger("metrics_storage")
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_percent REAL,
                    memory_percent REAL,
                    memory_used_mb REAL,
                    disk_io_read_mb REAL,
                    disk_io_write_mb REAL,
                    network_sent_mb REAL,
                    network_recv_mb REAL,
                    thread_count INTEGER,
                    process_count INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS custom_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_id INTEGER,
                    name TEXT NOT NULL,
                    value REAL,
                    unit TEXT,
                    metadata TEXT,
                    FOREIGN KEY (snapshot_id) REFERENCES performance_snapshots (id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    threshold REAL,
                    current_value REAL,
                    severity TEXT,
                    message TEXT,
                    timestamp TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT 0
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
    
    def store_snapshot(self, snapshot: PerformanceSnapshot) -> int:
        """Store performance snapshot"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_snapshots (
                    timestamp, cpu_percent, memory_percent, memory_used_mb,
                    disk_io_read_mb, disk_io_write_mb, network_sent_mb, network_recv_mb,
                    thread_count, process_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot.timestamp.isoformat(),
                snapshot.cpu_percent,
                snapshot.memory_percent,
                snapshot.memory_used_mb,
                snapshot.disk_io_read_mb,
                snapshot.disk_io_write_mb,
                snapshot.network_sent_mb,
                snapshot.network_recv_mb,
                snapshot.thread_count,
                snapshot.process_count
            ))
            
            snapshot_id = cursor.lastrowid
            
            # Store custom metrics
            for name, value in snapshot.custom_metrics.items():
                cursor.execute('''
                    INSERT INTO custom_metrics (snapshot_id, name, value, unit, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (snapshot_id, name, value, "", "{}"))
            
            conn.commit()
            conn.close()
            
            return snapshot_id
            
        except Exception as e:
            self.logger.error(f"Snapshot storage failed: {e}")
            return -1
    
    def store_alert(self, alert: PerformanceAlert) -> int:
        """Store performance alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_alerts (
                    metric_name, threshold, current_value, severity, message, timestamp, resolved
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.metric_name,
                alert.threshold,
                alert.current_value,
                alert.severity,
                alert.message,
                alert.timestamp.isoformat(),
                alert.resolved
            ))
            
            alert_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            return alert_id
            
        except Exception as e:
            self.logger.error(f"Alert storage failed: {e}")
            return -1
    
    def get_recent_snapshots(self, limit: int = 100) -> List[PerformanceSnapshot]:
        """Get recent performance snapshots"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM performance_snapshots 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            snapshots = []
            for row in cursor.fetchall():
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.fromisoformat(row[1]),
                    cpu_percent=row[2],
                    memory_percent=row[3],
                    memory_used_mb=row[4],
                    disk_io_read_mb=row[5],
                    disk_io_write_mb=row[6],
                    network_sent_mb=row[7],
                    network_recv_mb=row[8],
                    thread_count=row[9],
                    process_count=row[10]
                )
                snapshots.append(snapshot)
            
            conn.close()
            return snapshots
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve snapshots: {e}")
            return []
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get active performance alerts"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM performance_alerts 
                WHERE resolved = 0 
                ORDER BY timestamp DESC
            ''')
            
            alerts = []
            for row in cursor.fetchall():
                alert = PerformanceAlert(
                    metric_name=row[1],
                    threshold=row[2],
                    current_value=row[3],
                    severity=row[4],
                    message=row[5],
                    timestamp=datetime.fromisoformat(row[6]),
                    resolved=bool(row[7])
                )
                alerts.append(alert)
            
            conn.close()
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve alerts: {e}")
            return []


class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, collection_interval: float = 5.0, storage_path: Optional[str] = None):
        self.collection_interval = collection_interval
        self.logger = get_logger("performance_monitor")
        self.voice = get_voice_alerts()
        self.config = get_config()
        
        # Components
        self.collector = PerformanceCollector()
        self.storage = MetricsStorage(storage_path)
        
        # Runtime state
        self.is_running = False
        self.collection_task = None
        self.current_snapshot = None
        self.metrics_history = deque(maxlen=1000)
        self.custom_metrics = {}
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "memory_used_mb": 2048.0,
            "disk_io_read_mb": 100.0,
            "disk_io_write_mb": 100.0
        }
        
        # Performance tracking
        self.start_time = time.time()
        self.function_timings = {}
        
        # Memory tracking
        if hasattr(tracemalloc, 'start'):
            tracemalloc.start()
    
    async def start_monitoring(self):
        """Start performance monitoring"""
        if self.is_running:
            self.logger.warning("Performance monitoring already running")
            return
        
        self.is_running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        self.logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop performance monitoring"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Performance monitoring stopped")
    
    async def _collection_loop(self):
        """Main collection loop"""
        while self.is_running:
            try:
                # Collect snapshot
                snapshot = await self.collector.collect_snapshot()
                
                # Add custom metrics
                snapshot.custom_metrics = self.custom_metrics.copy()
                
                # Store snapshot
                self.current_snapshot = snapshot
                self.metrics_history.append(snapshot)
                self.storage.store_snapshot(snapshot)
                
                # Check for alerts
                await self._check_alerts(snapshot)
                
                # Sleep until next collection
                await asyncio.sleep(self.collection_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Collection loop error: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _check_alerts(self, snapshot: PerformanceSnapshot):
        """Check for performance alerts"""
        try:
            alerts = []
            
            # CPU usage
            if snapshot.cpu_percent > self.alert_thresholds.get("cpu_percent", 80.0):
                alerts.append(PerformanceAlert(
                    metric_name="cpu_percent",
                    threshold=self.alert_thresholds["cpu_percent"],
                    current_value=snapshot.cpu_percent,
                    severity="high" if snapshot.cpu_percent > 90.0 else "medium",
                    message=f"High CPU usage: {snapshot.cpu_percent:.1f}%",
                    timestamp=snapshot.timestamp
                ))
            
            # Memory usage
            if snapshot.memory_percent > self.alert_thresholds.get("memory_percent", 85.0):
                alerts.append(PerformanceAlert(
                    metric_name="memory_percent",
                    threshold=self.alert_thresholds["memory_percent"],
                    current_value=snapshot.memory_percent,
                    severity="high" if snapshot.memory_percent > 95.0 else "medium",
                    message=f"High memory usage: {snapshot.memory_percent:.1f}%",
                    timestamp=snapshot.timestamp
                ))
            
            # Memory absolute
            if snapshot.memory_used_mb > self.alert_thresholds.get("memory_used_mb", 2048.0):
                alerts.append(PerformanceAlert(
                    metric_name="memory_used_mb",
                    threshold=self.alert_thresholds["memory_used_mb"],
                    current_value=snapshot.memory_used_mb,
                    severity="medium",
                    message=f"High memory usage: {snapshot.memory_used_mb:.1f}MB",
                    timestamp=snapshot.timestamp
                ))
            
            # Disk I/O
            if snapshot.disk_io_read_mb > self.alert_thresholds.get("disk_io_read_mb", 100.0):
                alerts.append(PerformanceAlert(
                    metric_name="disk_io_read_mb",
                    threshold=self.alert_thresholds["disk_io_read_mb"],
                    current_value=snapshot.disk_io_read_mb,
                    severity="low",
                    message=f"High disk read: {snapshot.disk_io_read_mb:.1f}MB/s",
                    timestamp=snapshot.timestamp
                ))
            
            # Store and announce alerts
            for alert in alerts:
                self.storage.store_alert(alert)
                
                if alert.severity in ["high", "critical"]:
                    self.voice.speak_warning(alert.message)
                
                self.logger.warning(f"Performance alert: {alert.message}")
                
        except Exception as e:
            self.logger.error(f"Alert checking failed: {e}")
    
    def add_custom_metric(self, name: str, value: float, unit: str = ""):
        """Add custom metric"""
        self.custom_metrics[name] = value
        
        # Also create a metric object for detailed tracking
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now()
        )
        
        self.logger.debug(f"Added custom metric: {name} = {value} {unit}")
    
    def time_function(self, func_name: str):
        """Decorator to time function execution"""
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self._record_function_timing(func_name, execution_time)
            
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    execution_time = time.time() - start_time
                    self._record_function_timing(func_name, execution_time)
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        return decorator
    
    def _record_function_timing(self, func_name: str, execution_time: float):
        """Record function execution timing"""
        if func_name not in self.function_timings:
            self.function_timings[func_name] = []
        
        self.function_timings[func_name].append(execution_time)
        
        # Keep only recent timings
        if len(self.function_timings[func_name]) > 100:
            self.function_timings[func_name] = self.function_timings[func_name][-100:]
        
        # Add as custom metric
        self.add_custom_metric(f"function_time_{func_name}", execution_time, "seconds")
    
    def get_function_stats(self, func_name: str) -> Dict[str, float]:
        """Get function performance statistics"""
        timings = self.function_timings.get(func_name, [])
        
        if not timings:
            return {
                "count": 0,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0
            }
        
        return {
            "count": len(timings),
            "min": min(timings),
            "max": max(timings),
            "mean": statistics.mean(timings),
            "median": statistics.median(timings),
            "std_dev": statistics.stdev(timings) if len(timings) > 1 else 0.0
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if not self.current_snapshot:
            return {}
        
        snapshot = self.current_snapshot
        
        # Basic metrics
        metrics = {
            "timestamp": snapshot.timestamp.isoformat(),
            "cpu_percent": snapshot.cpu_percent,
            "memory_percent": snapshot.memory_percent,
            "memory_used_mb": snapshot.memory_used_mb,
            "disk_io_read_mb": snapshot.disk_io_read_mb,
            "disk_io_write_mb": snapshot.disk_io_write_mb,
            "network_sent_mb": snapshot.network_sent_mb,
            "network_recv_mb": snapshot.network_recv_mb,
            "thread_count": snapshot.thread_count,
            "process_count": snapshot.process_count
        }
        
        # Add custom metrics
        metrics.update(snapshot.custom_metrics)
        
        # Add uptime
        metrics["uptime_seconds"] = time.time() - self.start_time
        
        return metrics
    
    def get_trend_analysis(self, metric_name: str, window_minutes: int = 30) -> Dict[str, Any]:
        """Get trend analysis for a metric"""
        try:
            # Get recent snapshots
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            recent_snapshots = [s for s in self.metrics_history if s.timestamp >= cutoff_time]
            
            if not recent_snapshots:
                return {"error": "No recent data available"}
            
            # Extract metric values
            values = []
            for snapshot in recent_snapshots:
                if hasattr(snapshot, metric_name):
                    values.append(getattr(snapshot, metric_name))
                elif metric_name in snapshot.custom_metrics:
                    values.append(snapshot.custom_metrics[metric_name])
            
            if not values:
                return {"error": f"Metric '{metric_name}' not found"}
            
            # Calculate trend
            if len(values) >= 2:
                # Simple linear trend
                x = list(range(len(values)))
                y = values
                
                # Calculate slope
                n = len(values)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x[i] * y[i] for i in range(n))
                sum_x2 = sum(xi * xi for xi in x)
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                trend = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"
            else:
                trend = "stable"
                slope = 0.0
            
            return {
                "metric_name": metric_name,
                "window_minutes": window_minutes,
                "data_points": len(values),
                "current_value": values[-1],
                "min_value": min(values),
                "max_value": max(values),
                "mean_value": statistics.mean(values),
                "trend": trend,
                "slope": slope,
                "values": values
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return {"error": str(e)}
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        try:
            current_metrics = self.get_current_metrics()
            active_alerts = self.storage.get_active_alerts()
            
            # Function statistics
            function_stats = {
                name: self.get_function_stats(name)
                for name in self.function_timings.keys()
            }
            
            # Memory tracking
            memory_stats = {}
            if hasattr(tracemalloc, 'get_traced_memory'):
                current_memory, peak_memory = tracemalloc.get_traced_memory()
                memory_stats = {
                    "current_memory_mb": current_memory / (1024 * 1024),
                    "peak_memory_mb": peak_memory / (1024 * 1024)
                }
            
            # Garbage collection stats
            gc_stats = {
                "collections": gc.get_stats() if hasattr(gc, 'get_stats') else [],
                "objects": len(gc.get_objects()),
                "collected": gc.collect()
            }
            
            return {
                "current_metrics": current_metrics,
                "active_alerts": len(active_alerts),
                "alert_details": [
                    {
                        "metric": alert.metric_name,
                        "severity": alert.severity,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in active_alerts
                ],
                "function_stats": function_stats,
                "memory_stats": memory_stats,
                "gc_stats": gc_stats,
                "uptime_seconds": time.time() - self.start_time,
                "monitoring_active": self.is_running,
                "collection_interval": self.collection_interval,
                "metrics_history_size": len(self.metrics_history)
            }
            
        except Exception as e:
            self.logger.error(f"Performance report generation failed: {e}")
            return {"error": str(e)}
    
    def set_alert_threshold(self, metric_name: str, threshold: float):
        """Set alert threshold for a metric"""
        self.alert_thresholds[metric_name] = threshold
        self.logger.info(f"Set alert threshold for {metric_name}: {threshold}")
    
    def get_optimization_suggestions(self) -> List[str]:
        """Get performance optimization suggestions"""
        suggestions = []
        
        if not self.current_snapshot:
            return suggestions
        
        snapshot = self.current_snapshot
        
        # CPU optimization
        if snapshot.cpu_percent > 80.0:
            suggestions.append("Consider optimizing CPU-intensive operations or adding async processing")
        
        # Memory optimization
        if snapshot.memory_percent > 85.0:
            suggestions.append("High memory usage detected - consider memory profiling and optimization")
        
        # Disk I/O optimization
        if snapshot.disk_io_read_mb > 50.0 or snapshot.disk_io_write_mb > 50.0:
            suggestions.append("High disk I/O detected - consider caching or batch processing")
        
        # Thread optimization
        if snapshot.thread_count > 100:
            suggestions.append("High thread count - consider using async/await or thread pooling")
        
        # Function timing optimization
        for func_name, timings in self.function_timings.items():
            if timings and statistics.mean(timings) > 5.0:
                suggestions.append(f"Function '{func_name}' has high execution time - consider optimization")
        
        return suggestions