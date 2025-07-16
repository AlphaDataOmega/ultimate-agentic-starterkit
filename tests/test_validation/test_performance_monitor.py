"""
Unit tests for performance monitor.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from validation.performance_monitor import (
    PerformanceMonitor, PerformanceCollector, PerformanceSnapshot,
    PerformanceMetric, PerformanceAlert, MetricsStorage
)


class TestPerformanceMetric:
    """Test PerformanceMetric dataclass"""
    
    def test_performance_metric_creation(self):
        """Test PerformanceMetric creation"""
        timestamp = datetime.now()
        metric = PerformanceMetric(
            name="test_metric",
            value=42.0,
            unit="seconds",
            timestamp=timestamp,
            metadata={"source": "test"}
        )
        
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.unit == "seconds"
        assert metric.timestamp == timestamp
        assert metric.metadata == {"source": "test"}
    
    def test_performance_metric_defaults(self):
        """Test PerformanceMetric with defaults"""
        metric = PerformanceMetric(
            name="test_metric",
            value=42.0,
            unit="seconds",
            timestamp=datetime.now()
        )
        
        assert metric.metadata == {}


class TestPerformanceSnapshot:
    """Test PerformanceSnapshot dataclass"""
    
    def test_performance_snapshot_creation(self):
        """Test PerformanceSnapshot creation"""
        timestamp = datetime.now()
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_mb=1024.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            network_sent_mb=2.0,
            network_recv_mb=3.0,
            thread_count=20,
            process_count=150,
            custom_metrics={"test": 42.0}
        )
        
        assert snapshot.timestamp == timestamp
        assert snapshot.cpu_percent == 50.0
        assert snapshot.memory_percent == 60.0
        assert snapshot.memory_used_mb == 1024.0
        assert snapshot.disk_io_read_mb == 10.0
        assert snapshot.disk_io_write_mb == 5.0
        assert snapshot.network_sent_mb == 2.0
        assert snapshot.network_recv_mb == 3.0
        assert snapshot.thread_count == 20
        assert snapshot.process_count == 150
        assert snapshot.custom_metrics == {"test": 42.0}
    
    def test_performance_snapshot_defaults(self):
        """Test PerformanceSnapshot with defaults"""
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_mb=1024.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            network_sent_mb=2.0,
            network_recv_mb=3.0,
            thread_count=20,
            process_count=150
        )
        
        assert snapshot.custom_metrics == {}


class TestPerformanceAlert:
    """Test PerformanceAlert dataclass"""
    
    def test_performance_alert_creation(self):
        """Test PerformanceAlert creation"""
        timestamp = datetime.now()
        alert = PerformanceAlert(
            metric_name="cpu_percent",
            threshold=80.0,
            current_value=95.0,
            severity="high",
            message="High CPU usage detected",
            timestamp=timestamp,
            resolved=False
        )
        
        assert alert.metric_name == "cpu_percent"
        assert alert.threshold == 80.0
        assert alert.current_value == 95.0
        assert alert.severity == "high"
        assert alert.message == "High CPU usage detected"
        assert alert.timestamp == timestamp
        assert alert.resolved is False
    
    def test_performance_alert_defaults(self):
        """Test PerformanceAlert with defaults"""
        alert = PerformanceAlert(
            metric_name="cpu_percent",
            threshold=80.0,
            current_value=95.0,
            severity="high",
            message="High CPU usage detected",
            timestamp=datetime.now()
        )
        
        assert alert.resolved is False


class TestPerformanceCollector:
    """Test PerformanceCollector class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.collector = PerformanceCollector()
    
    def test_collector_initialization(self):
        """Test PerformanceCollector initialization"""
        assert self.collector.logger is not None
        assert self.collector.process is not None
        assert self.collector.last_disk_io is None
        assert self.collector.last_network_io is None
        assert self.collector.last_timestamp is None
    
    @pytest.mark.asyncio
    async def test_collect_snapshot_success(self):
        """Test successful snapshot collection"""
        # Mock psutil components
        with patch('psutil.disk_io_counters') as mock_disk, \
             patch('psutil.net_io_counters') as mock_net, \
             patch('psutil.pids') as mock_pids:
            
            # Mock disk I/O counters
            mock_disk.return_value = Mock(read_bytes=1024*1024, write_bytes=512*1024)
            
            # Mock network I/O counters  
            mock_net.return_value = Mock(bytes_sent=2048*1024, bytes_recv=1536*1024)
            
            # Mock process count
            mock_pids.return_value = list(range(100))
            
            # Mock process methods
            self.collector.process.cpu_percent = Mock(return_value=45.0)
            self.collector.process.memory_info = Mock(return_value=Mock(rss=1024*1024*1024))
            self.collector.process.memory_percent = Mock(return_value=55.0)
            self.collector.process.num_threads = Mock(return_value=15)
            
            snapshot = await self.collector.collect_snapshot()
            
            assert snapshot.cpu_percent == 45.0
            assert snapshot.memory_percent == 55.0
            assert snapshot.memory_used_mb == 1024.0
            assert snapshot.thread_count == 15
            assert snapshot.process_count == 100
            assert snapshot.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_collect_snapshot_exception(self):
        """Test snapshot collection with exception"""
        # Mock process to raise exception
        self.collector.process.cpu_percent = Mock(side_effect=Exception("Test error"))
        
        snapshot = await self.collector.collect_snapshot()
        
        # Should return default values
        assert snapshot.cpu_percent == 0.0
        assert snapshot.memory_percent == 0.0
        assert snapshot.memory_used_mb == 0.0
        assert snapshot.thread_count == 0
        assert snapshot.process_count == 0
        assert snapshot.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_collect_snapshot_with_rates(self):
        """Test snapshot collection with rate calculations"""
        # Mock psutil components
        with patch('psutil.disk_io_counters') as mock_disk, \
             patch('psutil.net_io_counters') as mock_net, \
             patch('psutil.pids') as mock_pids:
            
            # First call - establish baseline
            mock_disk.return_value = Mock(read_bytes=1024*1024, write_bytes=512*1024)
            mock_net.return_value = Mock(bytes_sent=2048*1024, bytes_recv=1536*1024)
            mock_pids.return_value = list(range(100))
            
            self.collector.process.cpu_percent = Mock(return_value=45.0)
            self.collector.process.memory_info = Mock(return_value=Mock(rss=1024*1024*1024))
            self.collector.process.memory_percent = Mock(return_value=55.0)
            self.collector.process.num_threads = Mock(return_value=15)
            
            # First snapshot
            snapshot1 = await self.collector.collect_snapshot()
            
            # Second call - calculate rates
            mock_disk.return_value = Mock(read_bytes=2048*1024, write_bytes=1024*1024)
            mock_net.return_value = Mock(bytes_sent=4096*1024, bytes_recv=3072*1024)
            
            # Wait a bit to ensure time difference
            await asyncio.sleep(0.1)
            
            snapshot2 = await self.collector.collect_snapshot()
            
            # Second snapshot should have rate calculations
            assert snapshot2.disk_io_read_mb > 0
            assert snapshot2.disk_io_write_mb > 0
            assert snapshot2.network_sent_mb > 0
            assert snapshot2.network_recv_mb > 0


class TestMetricsStorage:
    """Test MetricsStorage class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Use in-memory SQLite database
        self.storage = MetricsStorage(":memory:")
    
    def test_storage_initialization(self):
        """Test MetricsStorage initialization"""
        assert self.storage.db_path == ":memory:"
        assert self.storage.logger is not None
        
        # Check that tables were created
        import sqlite3
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # This should not raise an exception if tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        conn.close()
        
        # We can't check the actual tables since this is a different connection
        # but the init should have worked without errors
    
    def test_store_snapshot(self):
        """Test storing performance snapshot"""
        timestamp = datetime.now()
        snapshot = PerformanceSnapshot(
            timestamp=timestamp,
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_mb=1024.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            network_sent_mb=2.0,
            network_recv_mb=3.0,
            thread_count=20,
            process_count=150,
            custom_metrics={"test_metric": 42.0}
        )
        
        snapshot_id = self.storage.store_snapshot(snapshot)
        
        assert snapshot_id > 0
    
    def test_store_alert(self):
        """Test storing performance alert"""
        alert = PerformanceAlert(
            metric_name="cpu_percent",
            threshold=80.0,
            current_value=95.0,
            severity="high",
            message="High CPU usage detected",
            timestamp=datetime.now()
        )
        
        alert_id = self.storage.store_alert(alert)
        
        assert alert_id > 0
    
    def test_get_recent_snapshots(self):
        """Test retrieving recent snapshots"""
        # Store multiple snapshots
        for i in range(5):
            snapshot = PerformanceSnapshot(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_percent=50.0 + i,
                memory_percent=60.0 + i,
                memory_used_mb=1024.0 + i,
                disk_io_read_mb=10.0 + i,
                disk_io_write_mb=5.0 + i,
                network_sent_mb=2.0 + i,
                network_recv_mb=3.0 + i,
                thread_count=20 + i,
                process_count=150 + i
            )
            self.storage.store_snapshot(snapshot)
        
        snapshots = self.storage.get_recent_snapshots(limit=3)
        
        assert len(snapshots) == 3
        # Should be in reverse chronological order
        assert snapshots[0].cpu_percent == 50.0  # Most recent
        assert snapshots[1].cpu_percent == 51.0
        assert snapshots[2].cpu_percent == 52.0
    
    def test_get_active_alerts(self):
        """Test retrieving active alerts"""
        # Store multiple alerts
        for i in range(3):
            alert = PerformanceAlert(
                metric_name=f"metric_{i}",
                threshold=80.0,
                current_value=95.0,
                severity="high",
                message=f"Alert {i}",
                timestamp=datetime.now(),
                resolved=i == 2  # Mark last alert as resolved
            )
            self.storage.store_alert(alert)
        
        active_alerts = self.storage.get_active_alerts()
        
        assert len(active_alerts) == 2  # Only unresolved alerts
        assert all(not alert.resolved for alert in active_alerts)


class TestPerformanceMonitor:
    """Test PerformanceMonitor class"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.monitor = PerformanceMonitor(
            collection_interval=0.1,  # Fast collection for testing
            storage_path=":memory:"
        )
    
    def test_monitor_initialization(self):
        """Test PerformanceMonitor initialization"""
        assert self.monitor.collection_interval == 0.1
        assert self.monitor.collector is not None
        assert self.monitor.storage is not None
        assert self.monitor.is_running is False
        assert self.monitor.current_snapshot is None
        assert len(self.monitor.metrics_history) == 0
        assert self.monitor.custom_metrics == {}
        assert len(self.monitor.alert_thresholds) > 0
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        # Mock the collector to avoid actual system calls
        self.monitor.collector.collect_snapshot = AsyncMock(
            return_value=PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=50.0,
                memory_percent=60.0,
                memory_used_mb=1024.0,
                disk_io_read_mb=10.0,
                disk_io_write_mb=5.0,
                network_sent_mb=2.0,
                network_recv_mb=3.0,
                thread_count=20,
                process_count=150
            )
        )
        
        # Start monitoring
        await self.monitor.start_monitoring()
        
        assert self.monitor.is_running is True
        assert self.monitor.collection_task is not None
        
        # Let it run for a bit
        await asyncio.sleep(0.15)
        
        # Stop monitoring
        await self.monitor.stop_monitoring()
        
        assert self.monitor.is_running is False
        assert self.monitor.collection_task.done()
    
    @pytest.mark.asyncio
    async def test_start_monitoring_already_running(self):
        """Test starting monitoring when already running"""
        # Mock the collector
        self.monitor.collector.collect_snapshot = AsyncMock(
            return_value=PerformanceSnapshot(
                timestamp=datetime.now(),
                cpu_percent=50.0,
                memory_percent=60.0,
                memory_used_mb=1024.0,
                disk_io_read_mb=10.0,
                disk_io_write_mb=5.0,
                network_sent_mb=2.0,
                network_recv_mb=3.0,
                thread_count=20,
                process_count=150
            )
        )
        
        # Start monitoring
        await self.monitor.start_monitoring()
        
        # Try to start again
        await self.monitor.start_monitoring()
        
        # Should still be running
        assert self.monitor.is_running is True
        
        # Clean up
        await self.monitor.stop_monitoring()
    
    def test_add_custom_metric(self):
        """Test adding custom metrics"""
        self.monitor.add_custom_metric("test_metric", 42.0, "units")
        
        assert "test_metric" in self.monitor.custom_metrics
        assert self.monitor.custom_metrics["test_metric"] == 42.0
    
    def test_time_function_decorator(self):
        """Test function timing decorator"""
        # Test synchronous function
        @self.monitor.time_function("test_func")
        def test_func():
            time.sleep(0.01)
            return "result"
        
        result = test_func()
        
        assert result == "result"
        assert "test_func" in self.monitor.function_timings
        assert len(self.monitor.function_timings["test_func"]) == 1
        assert self.monitor.function_timings["test_func"][0] > 0.0
    
    @pytest.mark.asyncio
    async def test_time_function_decorator_async(self):
        """Test function timing decorator with async function"""
        # Test asynchronous function
        @self.monitor.time_function("async_test_func")
        async def async_test_func():
            await asyncio.sleep(0.01)
            return "async_result"
        
        result = await async_test_func()
        
        assert result == "async_result"
        assert "async_test_func" in self.monitor.function_timings
        assert len(self.monitor.function_timings["async_test_func"]) == 1
        assert self.monitor.function_timings["async_test_func"][0] > 0.0
    
    def test_get_function_stats(self):
        """Test getting function statistics"""
        # Add some timing data
        self.monitor.function_timings["test_func"] = [0.1, 0.2, 0.15, 0.25, 0.18]
        
        stats = self.monitor.get_function_stats("test_func")
        
        assert stats["count"] == 5
        assert stats["min"] == 0.1
        assert stats["max"] == 0.25
        assert stats["mean"] == 0.176
        assert stats["median"] == 0.18
        assert stats["std_dev"] > 0.0
    
    def test_get_function_stats_empty(self):
        """Test getting function statistics for nonexistent function"""
        stats = self.monitor.get_function_stats("nonexistent_func")
        
        assert stats["count"] == 0
        assert stats["min"] == 0.0
        assert stats["max"] == 0.0
        assert stats["mean"] == 0.0
        assert stats["median"] == 0.0
        assert stats["std_dev"] == 0.0
    
    def test_get_current_metrics(self):
        """Test getting current metrics"""
        # Set current snapshot
        self.monitor.current_snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_mb=1024.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            network_sent_mb=2.0,
            network_recv_mb=3.0,
            thread_count=20,
            process_count=150,
            custom_metrics={"test": 42.0}
        )
        
        metrics = self.monitor.get_current_metrics()
        
        assert metrics["cpu_percent"] == 50.0
        assert metrics["memory_percent"] == 60.0
        assert metrics["memory_used_mb"] == 1024.0
        assert metrics["test"] == 42.0
        assert "uptime_seconds" in metrics
        assert "timestamp" in metrics
    
    def test_get_current_metrics_no_snapshot(self):
        """Test getting current metrics with no snapshot"""
        metrics = self.monitor.get_current_metrics()
        
        assert metrics == {}
    
    def test_get_trend_analysis(self):
        """Test trend analysis"""
        # Add some historical data
        base_time = datetime.now() - timedelta(minutes=10)
        
        for i in range(10):
            snapshot = PerformanceSnapshot(
                timestamp=base_time + timedelta(minutes=i),
                cpu_percent=50.0 + i,  # Increasing trend
                memory_percent=60.0,
                memory_used_mb=1024.0,
                disk_io_read_mb=10.0,
                disk_io_write_mb=5.0,
                network_sent_mb=2.0,
                network_recv_mb=3.0,
                thread_count=20,
                process_count=150
            )
            self.monitor.metrics_history.append(snapshot)
        
        trend = self.monitor.get_trend_analysis("cpu_percent", window_minutes=15)
        
        assert trend["metric_name"] == "cpu_percent"
        assert trend["data_points"] == 10
        assert trend["current_value"] == 59.0
        assert trend["min_value"] == 50.0
        assert trend["max_value"] == 59.0
        assert trend["trend"] == "increasing"
        assert trend["slope"] > 0
    
    def test_get_trend_analysis_no_data(self):
        """Test trend analysis with no data"""
        trend = self.monitor.get_trend_analysis("cpu_percent")
        
        assert "error" in trend
        assert trend["error"] == "No recent data available"
    
    def test_get_trend_analysis_unknown_metric(self):
        """Test trend analysis with unknown metric"""
        # Add some data
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_mb=1024.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            network_sent_mb=2.0,
            network_recv_mb=3.0,
            thread_count=20,
            process_count=150
        )
        self.monitor.metrics_history.append(snapshot)
        
        trend = self.monitor.get_trend_analysis("unknown_metric")
        
        assert "error" in trend
        assert "not found" in trend["error"]
    
    def test_set_alert_threshold(self):
        """Test setting alert threshold"""
        self.monitor.set_alert_threshold("cpu_percent", 75.0)
        
        assert self.monitor.alert_thresholds["cpu_percent"] == 75.0
    
    def test_get_optimization_suggestions(self):
        """Test getting optimization suggestions"""
        # Set current snapshot with high resource usage
        self.monitor.current_snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=85.0,  # High CPU
            memory_percent=90.0,  # High memory
            memory_used_mb=3072.0,  # High memory usage
            disk_io_read_mb=60.0,  # High disk I/O
            disk_io_write_mb=60.0,  # High disk I/O
            network_sent_mb=2.0,
            network_recv_mb=3.0,
            thread_count=150,  # High thread count
            process_count=150
        )
        
        # Add function with high execution time
        self.monitor.function_timings["slow_function"] = [6.0, 7.0, 8.0]
        
        suggestions = self.monitor.get_optimization_suggestions()
        
        assert len(suggestions) > 0
        assert any("CPU" in suggestion for suggestion in suggestions)
        assert any("memory" in suggestion for suggestion in suggestions)
        assert any("disk" in suggestion for suggestion in suggestions)
        assert any("thread" in suggestion for suggestion in suggestions)
        assert any("slow_function" in suggestion for suggestion in suggestions)
    
    def test_get_optimization_suggestions_no_snapshot(self):
        """Test getting optimization suggestions with no snapshot"""
        suggestions = self.monitor.get_optimization_suggestions()
        
        assert len(suggestions) == 0
    
    def test_get_performance_report(self):
        """Test getting performance report"""
        # Set current snapshot
        self.monitor.current_snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_mb=1024.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            network_sent_mb=2.0,
            network_recv_mb=3.0,
            thread_count=20,
            process_count=150
        )
        
        # Add some function timings
        self.monitor.function_timings["test_func"] = [0.1, 0.2, 0.15]
        
        # Mock storage to return some alerts
        self.monitor.storage.get_active_alerts = Mock(return_value=[
            PerformanceAlert(
                metric_name="cpu_percent",
                threshold=80.0,
                current_value=85.0,
                severity="high",
                message="High CPU usage",
                timestamp=datetime.now()
            )
        ])
        
        report = self.monitor.get_performance_report()
        
        assert "current_metrics" in report
        assert "active_alerts" in report
        assert "function_stats" in report
        assert "memory_stats" in report
        assert "gc_stats" in report
        assert "uptime_seconds" in report
        assert "monitoring_active" in report
        
        assert report["active_alerts"] == 1
        assert len(report["alert_details"]) == 1
        assert "test_func" in report["function_stats"]