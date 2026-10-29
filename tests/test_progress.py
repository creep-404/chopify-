"""Tests for the ProgressTracker and ETA calculation."""
import time
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ui import ProgressTracker, STAGE_WEIGHTS


class MockGrProgress:
    """Mock Gradio Progress for testing."""
    def __init__(self):
        self.calls = []
    
    def __call__(self, progress, desc):
        self.calls.append((progress, desc))


class TestProgressTracker:
    """Test the ProgressTracker class."""
    
    def test_initialization(self):
        """Test that ProgressTracker initializes correctly."""
        tracker = ProgressTracker()
        assert tracker.current_stage_name == "transcribe"
        assert "transcribe" in tracker.stages
        assert "score" in tracker.stages
        assert "render" in tracker.stages
        assert tracker.get_overall_progress() == 0.0
    
    def test_stage_transition(self):
        """Test stage transition updates progress correctly."""
        tracker = ProgressTracker()
        
        # Initial state
        assert tracker.get_overall_progress() == 0.0
        
        # Transition to score stage
        tracker.callback("score", "starting", None, "Starting scoring...")
        assert tracker.current_stage_name == "score"
        # transcribe should be marked complete
        assert tracker.stages["transcribe"].progress == 1.0
        # score should be at 0
        assert tracker.stages["score"].progress == 0.0
    
    def test_progress_with_value(self):
        """Test progress updates with a value."""
        tracker = ProgressTracker()
        
        tracker.callback("transcribe", "transcribing", 0.5, "Halfway done")
        assert tracker.stages["transcribe"].progress == 0.5
        # Overall progress should reflect weighted progress
        expected = STAGE_WEIGHTS["transcribe"] * 0.5
        assert abs(tracker.get_overall_progress() - expected) < 0.001
    
    def test_progress_none_indeterminate(self):
        """Test that None progress keeps default 0.0 for new stage."""
        tracker = ProgressTracker()
        
        tracker.callback("score", "ollama", None, "Querying Ollama...")
        # New stage gets 0.0 by default when progress=None
        assert tracker.stages["score"].progress == 0.0
        # Overall progress: transcribe (1.0) + score (0.0)
        expected = STAGE_WEIGHTS["transcribe"] * 1.0 + STAGE_WEIGHTS["score"] * 0.0
        assert abs(tracker.get_overall_progress() - expected) < 0.001
    
    def test_eta_estimation(self):
        """Test ETA calculation."""
        tracker = ProgressTracker()
        
        # Before any progress, should say "Estimating..."
        assert tracker.get_eta() == "Estimating..."
        
        # Add some progress
        tracker.callback("transcribe", "transcribing", 0.5, "Halfway")
        time.sleep(0.01)  # Small delay for elapsed time
        eta = tracker.get_eta()
        assert eta != "Estimating..."
        assert "s" in eta or ":" in eta
    
    def test_eta_format(self):
        """Test time formatting."""
        tracker = ProgressTracker()
        
        assert tracker._format_time(30) == "30s"
        assert tracker._format_time(90) == "01:30"
        assert tracker._format_time(3661) == "1:01:01"
        assert tracker._format_time(-1) == "Estimating..."
        assert tracker._format_time(float('nan')) == "Estimating..."
    
    def test_render_time_tracking(self):
        """Test that render times are tracked for ETA."""
        tracker = ProgressTracker()
        tracker.total_clips = 3
        
        # Simulate completing first clip
        tracker.callback("render", "complete", 1.0, "Clip 1 done")
        assert tracker.rendered_clips == 1
        assert len(tracker.render_times) == 1
        
        # Complete second clip
        tracker.callback("render", "complete", 1.0, "Clip 2 done")
        assert tracker.rendered_clips == 2
        assert len(tracker.render_times) == 2
    
    def test_status_text(self):
        """Test status text generation."""
        tracker = ProgressTracker()
        
        tracker.callback("transcribe", "transcribing", 0.5, "Processing...")
        status = tracker.get_status_text()
        
        assert "Overall:" in status
        assert "Elapsed:" in status
        assert "ETA:" in status
        assert "Stage: Transcribe" in status
        assert "50%" in status
        assert "Processing..." in status
    
    def test_gradio_progress_update(self):
        """Test Gradio progress bar integration."""
        mock_progress = MockGrProgress()
        tracker = ProgressTracker(gr_progress=mock_progress)
        
        tracker.callback("transcribe", "transcribing", 0.5, "Halfway")
        tracker.update_gradio_progress()
        
        assert len(mock_progress.calls) == 1
        progress, desc = mock_progress.calls[0]
        assert abs(progress - (STAGE_WEIGHTS["transcribe"] * 0.5)) < 0.001
        assert desc == "Halfway"


class TestStageWeights:
    """Test that stage weights sum to 1.0."""
    
    def test_weights_sum_to_one(self):
        """Ensure stage weights sum to 1.0."""
        total = sum(STAGE_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001


class TestProgressTrackerEdgeCases:
    """Test edge cases for ProgressTracker."""
    
    def test_zero_progress(self):
        """Test behavior at zero progress."""
        tracker = ProgressTracker()
        assert tracker.get_eta() == "Estimating..."
    
    def test_full_progress(self):
        """Test behavior at 100% progress."""
        tracker = ProgressTracker()
        for stage in STAGE_WEIGHTS:
            tracker.callback(stage, "complete", 1.0, "Done")
        assert tracker.get_overall_progress() == 1.0
        assert "0s" in tracker.get_eta() or "00:00" in tracker.get_eta()
    
    def test_eta_smoothing(self):
        """Test ETA smoothing with multiple updates."""
        tracker = ProgressTracker()
        
        # First progress
        tracker.callback("transcribe", "transcribing", 0.1, "Starting")
        eta1 = tracker.get_eta()
        
        # Second progress (simulate faster processing)
        time.sleep(0.01)
        tracker.callback("transcribe", "transcribing", 0.2, "Continuing")
        eta2 = tracker.get_eta()
        
        # Both should be valid time strings
        assert eta1 != "Estimating..."
        assert eta2 != "Estimating..."


if __name__ == "__main__":
    pytest.main([__file__, "-v"])