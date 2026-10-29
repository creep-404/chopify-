"""
Chopify Web UI - Local web interface for the viral clip pipeline.

Runs at http://127.0.0.1:7860 (or next available port).
100% local - no cloud APIs, no telemetry, no external connections except yt-dlp for URLs.
"""
import os
import sys
import json
import subprocess
import threading
import queue
import time
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable

import gradio as gr

HERE = Path(__file__).resolve().parent
PY = sys.executable

# Add repo to path for importing pipeline modules
sys.path.insert(0, str(HERE))

import download_and_transcribe
import score_clips
import render_clips

# Ollama default host
OLLAMA_HOST = "http://localhost:11434"

# Caption style presets (must match render_clips.py)
CAPTION_STYLES = ["default", "hormozi", "mrbeast", "podcast"]
ASPECT_RATIOS = ["9:16", "16:9", "1:1"]
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3"]

# Stage weights for overall progress calculation
# These reflect typical duration ratios: transcription is slowest, rendering is fastest
STAGE_WEIGHTS = {
    "transcribe": 0.65,   # Download + Whisper transcription (slowest)
    "score": 0.10,        # Transcript analysis/scoring (fast)
    "render": 0.25,       # FFmpeg rendering (moderate)
}

# Global state for progress tracking
progress_queue = queue.Queue()
current_process = None
stop_requested = False


@dataclass
class StageProgress:
    """Progress information for a single pipeline stage."""
    name: str
    sub_stage: str = ""
    progress: Optional[float] = None  # 0-1 or None for indeterminate
    message: str = ""
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)


class ProgressTracker:
    """Tracks progress across all pipeline stages and calculates ETA."""
    
    def __init__(self, gr_progress: Optional[gr.Progress] = None):
        self.gr_progress = gr_progress
        self.stages = {name: StageProgress(name) for name in STAGE_WEIGHTS}
        self.current_stage_name = "transcribe"
        # Initialize current stage with 0 progress (not indeterminate)
        self.stages["transcribe"].progress = 0.0
        self.pipeline_start = time.time()
        self.eta_history = []  # For smoothing
        self.total_clips = 0
        self.rendered_clips = 0
        self.render_times = []  # Individual clip render times
        
    def _update_gradio(self, overall_pct: float, desc: str):
        """Update Gradio progress bar if available."""
        if self.gr_progress:
            self.gr_progress(overall_pct, desc=desc)
    
    def _smooth_eta(self, new_eta: float) -> float:
        """Smooth ETA using exponential moving average."""
        if not self.eta_history:
            self.eta_history.append(new_eta)
            return new_eta
        # Exponential moving average with alpha=0.3
        smoothed = 0.3 * new_eta + 0.7 * self.eta_history[-1]
        self.eta_history.append(smoothed)
        # Keep only last 10 values
        if len(self.eta_history) > 10:
            self.eta_history.pop(0)
        return smoothed
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        if seconds < 0 or seconds != seconds:  # NaN check
            return "Estimating..."
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds//60):02d}:{int(seconds%60):02d}"
        else:
            return f"{int(seconds//3600):d}:{int((seconds%3600)//60):02d}:{int(seconds%60):02d}"
    
    def callback(self, stage: str, sub_stage: str, progress: Optional[float], message: str):
        """Main callback entry point for pipeline stages."""
        now = time.time()
        
        # Handle stage transitions
        if stage != self.current_stage_name:
            if self.current_stage_name in self.stages:
                self.stages[self.current_stage_name].progress = 1.0
            self.current_stage_name = stage
            if stage in self.stages:
                self.stages[stage].start_time = now
                self.stages[stage].progress = 0.0
        
        # Update stage progress
        if stage in self.stages:
            self.stages[stage].sub_stage = sub_stage
            # Only update progress if a value is provided (not None)
            if progress is not None:
                self.stages[stage].progress = progress
            self.stages[stage].message = message
            self.stages[stage].last_update = now
            self.stages[stage].message = message
            self.stages[stage].last_update = now
            
            # Track render times for ETA
            if stage == "render":
                if sub_stage == "complete":
                    self.rendered_clips += 1
                    clip_duration = self.stages["render"].last_update - self.stages["render"].start_time
                    self.render_times.append(clip_duration)
                    self.stages["render"].start_time = now  # Reset for next clip
    
    def get_overall_progress(self) -> float:
        """Calculate weighted overall progress (0-1)."""
        total = 0.0
        for name, weight in STAGE_WEIGHTS.items():
            stage = self.stages[name]
            if stage.progress is not None:
                total += weight * stage.progress
            elif name == self.current_stage_name:
                # Current stage with indeterminate progress counts as partial
                total += weight * 0.5
        return min(total, 1.0)
    
    def get_eta(self) -> str:
        """Calculate estimated time remaining."""
        elapsed = time.time() - self.pipeline_start
        overall = self.get_overall_progress()
        
        if overall <= 0.01:
            return "Estimating..."
        
        # Estimate total time based on current rate
        estimated_total = elapsed / overall
        remaining = estimated_total - elapsed
        
        # For rendering stage, use measured clip times for better accuracy
        if self.current_stage_name == "render" and self.render_times:
            avg_render = sum(self.render_times) / len(self.render_times)
            clips_left = self.total_clips - self.rendered_clips
            if clips_left > 0:
                render_remaining = avg_render * clips_left
                # Blend with overall estimate
                remaining = 0.7 * render_remaining + 0.3 * remaining
        
        # Smooth the ETA
        smoothed = self._smooth_eta(max(0, remaining))
        return self._format_time(smoothed)
    
    def get_elapsed(self) -> str:
        """Get elapsed time formatted."""
        return self._format_time(time.time() - self.pipeline_start)
    
    def get_status_text(self) -> str:
        """Get formatted status text for UI display."""
        current = self.stages[self.current_stage_name]
        overall_pct = self.get_overall_progress() * 100
        eta = self.get_eta()
        elapsed = self.get_elapsed()
        
        # Stage-specific display
        if current.progress is not None:
            stage_pct = f"{current.progress * 100:.0f}%"
        else:
            stage_pct = "—"
        
        clip_info = ""
        if self.current_stage_name == "render" and self.total_clips > 0:
            clip_info = f"  Clip {self.rendered_clips}/{self.total_clips}"
        
        return (
            f"📊 Overall: {overall_pct:.0f}%  |  "
            f"⏱ Elapsed: {elapsed}  |  ⏳ ETA: {eta}\n"
            f"🔄 Stage: {self.current_stage_name.capitalize()} ({stage_pct}){clip_info}\n"
            f"📝 {current.message}"
        )
    
    def update_gradio_progress(self):
        """Update Gradio progress bar."""
        if self.gr_progress:
            self._update_gradio(self.get_overall_progress(), self.stages[self.current_stage_name].message)


def get_ollama_models():
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return ["chopify-qwen4b"]  # fallback
        lines = result.stdout.strip().split("\n")
        if len(lines) <= 1:
            return ["chopify-qwen4b"]
        models = []
        for line in lines[1:]:  # skip header
            parts = line.split()
            if parts:
                models.append(parts[0])
        return models if models else ["chopify-qwen4b"]
    except Exception:
        return ["chopify-qwen4b"]


def check_ollama_running():
    """Check if Ollama server is running."""
    try:
        import urllib.request
        req = urllib.request.Request(OLLAMA_HOST + "/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def run_pipeline(
    input_type, url, local_video, whisper_model, ollama_model,
    aspect_ratio, caption_style, max_clips, min_len, max_len, min_score,
    tighten, loudnorm, preview, cookies_from_browser=None, progress=gr.Progress()
):
    """Run the full Chopify pipeline with progress updates."""
    global current_process, stop_requested
    stop_requested = False
    
    workdir = HERE / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    
    # Clear any previous output
    clips_dir = HERE / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize progress tracker
    tracker = ProgressTracker(progress)
    
    # Log callback that also feeds the tracker
    def log(msg):
        # Extract stage info from log messages if possible
        if msg.startswith("[1/3]"):
            tracker.callback("transcribe", "starting", 0.0, "Downloading/preparing video...")
        elif msg.startswith("[2/3]"):
            tracker.callback("score", "starting", None, "Analyzing transcript...")
        elif msg.startswith("[3/3]"):
            tracker.callback("render", "starting", None, "Starting clip rendering...")
        elif "Transcription complete" in msg:
            tracker.callback("transcribe", "complete", 1.0, "Transcription complete")
        elif "Scoring complete" in msg:
            tracker.callback("score", "complete", 1.0, "Scoring complete")
        elif "CHOPIFY COMPLETE" in msg:
            tracker.callback("render", "complete", 1.0, "All clips rendered")
        
        logs.append(msg)
        progress_queue.put((msg, None))
    
    # Create log output component reference (will be set after UI creation)
    # We'll use a simple list for logs and update the tracker directly
    logs = []
    
    try:
        # Validate inputs
        if input_type == "URL":
            if not url or not url.strip():
                return "Error: Please provide a YouTube URL", None
            url = url.strip()
        else:
            if not local_video:
                return "Error: Please upload a video file", None
        
        # Check Ollama if LLM model selected
        use_llm = ollama_model and ollama_model != "heuristic (built-in)"
        if use_llm:
            if not check_ollama_running():
                return "Error: Ollama is not running. Start it with 'ollama serve' or select 'heuristic (built-in)' for scoring.", None
            # Verify model exists
            models = get_ollama_models()
            if ollama_model not in models:
                return f"Error: Ollama model '{ollama_model}' not found. Available: {', '.join(models)}", None
        
        tracker.callback("transcribe", "starting", 0.0, "Initializing pipeline...")
        
        # Stage 1: Download/prepare + Transcribe
        tracker.callback("transcribe", "downloading", None, "Downloading/preparing video...")
        
        if input_type == "URL":
            download_and_transcribe.run(
                workdir, model_size=whisper_model, device="auto", compute_type="int8",
                url=url, local_video=None, cookies_from_browser=cookies_from_browser,
                progress_callback=tracker.callback
            )
        else:
            # local_video is a Gradio file object with .name attribute
            video_path = local_video.name if hasattr(local_video, 'name') else str(local_video)
            download_and_transcribe.run(
                workdir, model_size=whisper_model, device="auto", compute_type="int8",
                url=None, local_video=video_path,
                progress_callback=tracker.callback
            )
        
        tracker.callback("transcribe", "complete", 1.0, "Transcription complete")
        tracker.update_gradio_progress()
        
        # Stage 2: Score clips
        tracker.callback("score", "starting", None, "Analyzing transcript for viral clips...")
        
        llm_arg = ollama_model if use_llm else None
        segments, mode = score_clips.run(
            workdir,
            llm=llm_arg,
            host=OLLAMA_HOST,
            min_score=min_score,
            max_clips=max_clips,
            min_len=min_len,
            max_len=max_len,
            search=None,
            pick=None,
            progress_callback=tracker.callback
        )
        
        tracker.callback("score", "complete", 1.0, f"Scoring complete (mode: {mode}). Selected {len(segments)} clips.")
        tracker.update_gradio_progress()
        
        # Log clip details
        for i, seg in enumerate(segments, 1):
            log(f"  Clip {i}: {seg['start']:.1f}s - {seg['end']:.1f}s (score: {seg['overall']}/10) - {seg['hook']}")
        
        # Stage 3: Render clips
        tracker.callback("render", "starting", None, "Starting clip rendering...")
        
        # We need to call render_clips with the right arguments
        # render_clips.main() uses argparse, so we'll call the render function directly
        transcript_data = json.loads((workdir / "transcript.json").read_text(encoding="utf-8-sig"))
        segments_data = json.loads((workdir / "segments.json").read_text(encoding="utf-8-sig"))
        words = transcript_data["words"]
        
        source = Path(transcript_data["video"])
        if not source.exists():
            cands = sorted(workdir.glob("source.*"))
            source = cands[0] if cands else source
        source = source.resolve()
        
        W, H = render_clips.ffprobe_dims(source)
        out_dir = render_clips.default_out()
        out_dir.mkdir(parents=True, exist_ok=True)
        
        log(f"Source: {source} ({W}x{H}) -> {aspect_ratio} -> {out_dir}")
        
        # Set total clips for render progress tracking
        tracker.total_clips = len(segments_data)
        tracker.rendered_clips = 0
        tracker.render_times = []
        
        saved_clips = []
        for i, seg in enumerate(segments_data, 1):
            if stop_requested:
                log("Stop requested by user.")
                break
            
            log(f"\n--- Clip {i}/{len(segments_data)}: {seg.get('hook', '')[:60]} (score: {seg.get('overall')}) ---")
            tracker.update_gradio_progress()
            
            try:
                out_path = render_clips.render(
                    seg, words, source, W, H, workdir, aspect_ratio,
                    out_dir=out_dir, style=caption_style,
                    do_tighten=tighten, loudnorm=loudnorm, preview=preview,
                    progress_callback=tracker.callback,
                    clip_index=i, total_clips=len(segments_data)
                )
                saved_clips.append(str(out_path))
                log(f"Saved: {out_path}")
            except subprocess.CalledProcessError as e:
                log(f"ERROR rendering clip {i}: {e}")
        
        log("\n" + "=" * 60)
        log(f"CHOPIFY COMPLETE - {len(saved_clips)} clips rendered")
        log("=" * 60)
        
        progress(1.0, desc="Complete")
        
        # Return logs and clip info for display
        clip_info = []
        for clip_path in saved_clips:
            clip_path = Path(clip_path)
            meta_path = clip_path.with_suffix(".meta.json")
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            
            # Get duration from ffprobe
            try:
                dur_out = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                     "-of", "default=noprint_wrappers=1:nokey=1", str(clip_path)],
                    capture_output=True, text=True, check=True
                )
                duration = float(dur_out.stdout.strip())
            except Exception:
                duration = 0
            
            clip_info.append({
                "path": str(clip_path),
                "name": clip_path.name,
                "duration": duration,
                "hook": meta.get("title", clip_path.stem),
                "score": meta.get("score"),
                "poster": str(clip_path.with_suffix(".png")) if clip_path.with_suffix(".png").exists() else None
            })
        
        return "\n".join(logs), clip_info
    
    except SystemExit as e:
        log(f"\nERROR: {e}")
        return "\n".join(logs), None
    except Exception as e:
        log(f"\nUNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        log(traceback.format_exc())
        return "\n".join(logs), None


def stop_pipeline():
    """Request pipeline stop."""
    global stop_requested
    stop_requested = True
    return "Stop requested - will halt after current clip...", "🛑 Stop requested"


def format_clip_gallery(clips):
    """Format clips for Gradio gallery."""
    if not clips:
        return []
    gallery_items = []
    for clip in clips:
        # Gallery expects (image_or_video, label) or just video path
        # For video preview, we'll use the video path directly with a label
        label = clip["name"]
        if clip.get("duration"):
            label += f" ({clip['duration']:.1f}s"
            if clip.get("score"):
                label += f", score: {clip['score']}/10"
            label += ")"
        gallery_items.append((clip["path"], label))
    return gallery_items


def create_ui():
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Chopify - Local Viral Clip Generator") as app:
        gr.Markdown("""
        # 🎬 Chopify Web UI
        **Local viral clip generator** — YouTube URL or local video → scored, captioned clips (9:16, 16:9, 1:1).  
        100% local: yt-dlp → faster-whisper → heuristic/Ollama scoring → ffmpeg render. No cloud, no API keys.
        """)
        
        with gr.Row():
            # Left column: Inputs
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Video Input")
                input_type = gr.Radio(
                    choices=["URL", "Local File"],
                    value="URL",
                    label="Input Method",
                    info="Choose YouTube URL or upload a local video file"
                )
                
                url_input = gr.Textbox(
                    label="YouTube URL",
                    placeholder="https://youtube.com/watch?v=...",
                    visible=True
                )
                
                video_input = gr.File(
                    label="Upload Video (MP4, MKV, MOV, etc.)",
                    file_types=[".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"],
                    visible=False
                )
                
                def toggle_input(choice):
                    return (
                        gr.update(visible=choice == "URL"),
                        gr.update(visible=choice == "Local File")
                    )
                
                input_type.change(
                    toggle_input, inputs=[input_type], outputs=[url_input, video_input]
                )
                
                gr.Markdown("### 🤖 AI Model")
                ollama_model = gr.Dropdown(
                    choices=["heuristic (built-in)"] + get_ollama_models(),
                    value="heuristic (built-in)",
                    label="Scoring Model",
                    info="Built-in heuristic (no deps) or local Ollama model for smarter clip selection"
                )
                
                refresh_models_btn = gr.Button("🔄 Refresh Ollama Models", size="sm")
                refresh_models_btn.click(
                    lambda: gr.update(choices=["heuristic (built-in)"] + get_ollama_models()),
                    outputs=[ollama_model]
                )
                
                ollama_status = gr.Markdown(
                    value="🟢 Ollama: **Running**" if check_ollama_running() else "🔴 Ollama: **Not running** (heuristic only)"
                )
                
                def check_ollama_status():
                    return gr.update(value="🟢 Ollama: **Running**" if check_ollama_running() else "🔴 Ollama: **Not running** (heuristic only)")
                
                refresh_models_btn.click(check_ollama_status, outputs=[ollama_status])
                
                gr.Markdown("### ⚙️ Settings")
                
                whisper_model = gr.Dropdown(
                    choices=WHISPER_MODELS,
                    value="medium",
                    label="Whisper Model",
                    info="Larger = more accurate, slower. medium is a good balance."
                )
                
                aspect_ratio = gr.Dropdown(
                    choices=ASPECT_RATIOS,
                    value="9:16",
                    label="Output Aspect Ratio"
                )
                
                caption_style = gr.Dropdown(
                    choices=CAPTION_STYLES,
                    value="hormozi",
                    label="Caption Style"
                )
                
                cookies_from_browser = gr.Dropdown(
                    choices=["None", "firefox", "chrome", "edge", "brave"],
                    value="None",
                    label="YouTube Browser Cookies",
                    info="Select browser to extract cookies from for age-restricted/private videos (None = no cookies)"
                )
                
                gr.Markdown("### 📏 Clip Settings")
                max_clips = gr.Slider(1, 20, value=10, step=1, label="Maximum Clips")
                min_len = gr.Slider(5, 60, value=20, step=1, label="Minimum Clip Length (seconds)")
                max_len = gr.Slider(30, 180, value=75, step=1, label="Maximum Clip Length (seconds)")
                min_score = gr.Slider(0, 10, value=6.5, step=0.5, label="Minimum Virality Score")
                
                gr.Markdown("### ✨ Options")
                tighten = gr.Checkbox(label="Remove filler words / tighten (--tighten)", value=False)
                loudnorm = gr.Checkbox(label="Normalize audio to -14 LUFS (--loudnorm)", value=False)
                preview = gr.Checkbox(label="Preview mode (fast 480p drafts, --preview)", value=False)
                
                generate_btn = gr.Button("🚀 Generate Clips", variant="primary", size="lg")
                stop_btn = gr.Button("⏹ Stop", variant="stop", size="lg", visible=False)
            
            # Right column: Output
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Progress Status")
                progress_status = gr.Markdown(
                    value="📊 Overall: 0%  |  ⏱ Elapsed: 0s  |  ⏳ ETA: Estimating...\n🔄 Stage: Transcribe (—)\n📝 Waiting to start...",
                    elem_id="progress-status"
                )
                
                gr.Markdown("### 📋 Progress Log")
                log_output = gr.Textbox(
                    label="Pipeline Output",
                    lines=20,
                    max_lines=30,
                    interactive=False
                )
                
                gr.Markdown("### 🎞️ Generated Clips")
                gallery = gr.Gallery(
                    label="Clips (click to play, right-click to download)",
                    show_label=False,
                    elem_id="gallery",
                    columns=2,
                    rows=3,
                    height="auto",
                    object_fit="contain",
                    preview=True
                )
                
                clips_json = gr.JSON(label="Clip Details", visible=False)
                
                # Download all as zip (future enhancement placeholder)
                # download_all_btn = gr.Button("📦 Download All as ZIP", visible=False)
        
        # Event handlers
        def on_generate(
            input_type, url, local_video, whisper_model, ollama_model,
            aspect_ratio, caption_style, max_clips, min_len, max_len, min_score,
            tighten, loudnorm, preview, cookies_from_browser
        ):
            # Show stop button, hide generate
            yield (
                gr.update(visible=False),  # generate_btn
                gr.update(visible=True),   # stop_btn
                "📊 Overall: 0%  |  ⏱ Elapsed: 0s  |  ⏳ ETA: Estimating...\n🔄 Stage: Transcribe (—)\n📝 Initializing...",  # progress_status
                "",                        # log_output (will be updated by progress)
                []                         # gallery
            )
            
            # Convert ollama_model "heuristic (built-in)" to None
            llm_model = None if ollama_model == "heuristic (built-in)" else ollama_model
            
            # Convert "None" string to None for cookies
            browser_cookies = None if cookies_from_browser == "None" else cookies_from_browser
            
            # Run pipeline and yield progress updates
            logs = []
            clips = []
            
            # We need to run the pipeline and capture progress updates
            # The run_pipeline function now uses a tracker that updates the UI
            # We'll yield the progress status periodically
            
            # Create a generator that yields progress updates
            def run_with_progress():
                nonlocal logs, clips
                # The run_pipeline function now handles progress internally
                # We'll capture the final results
                result = run_pipeline(
                    input_type, url, local_video, whisper_model, llm_model,
                    aspect_ratio, caption_style, int(max_clips), int(min_len), int(max_len), min_score,
                    tighten, loudnorm, preview, cookies_from_browser=browser_cookies
                )
                logs, clips = result
                return logs, clips
            
            # Run the pipeline
            try:
                logs, clips = run_with_progress()
            except Exception as e:
                logs = f"Pipeline error: {e}"
                clips = []
            
            yield (
                gr.update(visible=True),   # generate_btn
                gr.update(visible=False),  # stop_btn
                "",  # progress_status (final)
                logs,                      # log_output
                format_clip_gallery(clips) if clips else []  # gallery
            )
        
        generate_btn.click(
            on_generate,
            inputs=[
                input_type, url_input, video_input, whisper_model, ollama_model,
                aspect_ratio, caption_style, max_clips, min_len, max_len, min_score,
                tighten, loudnorm, preview, cookies_from_browser
            ],
            outputs=[generate_btn, stop_btn, progress_status, log_output, gallery]
        )
        
        stop_btn.click(
            stop_pipeline,
            outputs=[log_output, progress_status]
        )
        
        # Auto-refresh Ollama status on load
        app.load(check_ollama_status, outputs=[ollama_status])
        
        # Add custom CSS for better gallery display
        gr.HTML("""
        <style>
        #gallery .gallery-item { min-height: 200px; }
        #gallery video { max-height: 300px; }
        </style>
        """)
    
    return app


def main():
    """Launch the Gradio app."""
    # Ensure we bind to localhost only (security)
    app = create_ui()
    app.launch(
        server_name="127.0.0.1",
        server_port=None,
        share=False,
        inbrowser=True,
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft()
    )


if __name__ == "__main__":
    main()