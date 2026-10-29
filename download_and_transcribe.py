"""
Stage 1 of the viral-clip pipeline.

Downloads a YouTube video at highest quality with yt-dlp, then transcribes it
locally with faster-whisper (word-level timestamps). Writes work/transcript.json.

No external/paid APIs. CPU-friendly (int8). Usage:
    python download_and_transcribe.py <youtube_url> --model small
"""
import sys
import json
import argparse
import subprocess
import shutil
import os
import math
import tempfile
from pathlib import Path


# Global list to keep DLL directory handles alive (prevents garbage collection)
_CUDA_DLL_HANDLES = []


def _register_cuda_dll_directories():
    """Register NVIDIA CUDA DLL directories on Windows for faster-whisper/CTranslate2.
    
    This must be called before importing faster_whisper or ctranslate2.
    On Windows, CTranslate2 needs the CUDA runtime DLLs to be discoverable.
    The nvidia-* CuPy/CUDA wheels install DLLs under site-packages/nvidia/*/bin.
    
    We use both os.add_dll_directory() and PATH modification because CTranslate2
    may load CUDA kernels in threads that don't inherit the add_dll_directory
    search path, but do inherit the process PATH environment variable.
    """
    global _CUDA_DLL_HANDLES
    if sys.platform != "win32":
        return
    
    # Find the site-packages directory for the current environment
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    nvidia_root = site_packages / "nvidia"
    
    if not nvidia_root.exists():
        return
    
    bin_dirs = []
    for bin_dir in nvidia_root.rglob("bin"):
        if bin_dir.is_dir():
            bin_dirs.append(str(bin_dir))
    
    if not bin_dirs:
        return
    
    # 1. Add to PATH so all threads (including CTranslate2's worker threads) can find DLLs
    os.environ["PATH"] = ";".join(bin_dirs) + ";" + os.environ["PATH"]
    
    # 2. Also register via add_dll_directory for the main thread
    registered = []
    for d in bin_dirs:
        try:
            handle = os.add_dll_directory(d)
            _CUDA_DLL_HANDLES.append(handle)  # Keep handle alive
            registered.append(d)
        except Exception:
            pass
    
    if registered:
        print(f"CUDA DLL directories registered: {len(registered)}", flush=True)
        for d in registered:
            print(f"  {d}", flush=True)
        # Verify CUDA is now detectable
        try:
            import ctranslate2
            devices = ctranslate2.get_cuda_device_count()
            if devices > 0:
                print(f"CUDA devices detected: {devices}", flush=True)
            else:
                print("CUDA devices detected: 0 (falling back to CPU)", flush=True)
        except Exception as e:
            print(f"CUDA detection check failed: {e}", flush=True)


# Register CUDA DLL directories immediately on module import (before faster-whisper)
_register_cuda_dll_directories()


def run_cmd(cmd):
    print(">", " ".join(str(c) for c in cmd), flush=True)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # Print yt-dlp's actual output for debugging
        if e.stdout:
            print(f"yt-dlp stdout:\n{e.stdout}", flush=True)
        if e.stderr:
            print(f"yt-dlp stderr:\n{e.stderr}", flush=True)
        raise


def _validate_video_file(path: Path) -> bool:
    """Validate that a video file is playable and has audio/video streams."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return False
        streams = result.stdout.strip().split()
        has_video = "video" in streams
        has_audio = "audio" in streams
        return has_video and has_audio
    except Exception:
        return False


def _get_streams(path: Path):
    """Return (has_video, has_audio, stream_details) for a media file."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            return False, False, result.stderr.strip()
        lines = [l.strip() for l in result.stdout.strip().splitlines() if l.strip()]
        # ffprobe with -show_entries stream=codec_type,codec_name outputs:
        # codec_name, codec_type, codec_name, codec_type, ... per stream
        # So codec_type values are at odd indices (1, 3, 5, ...)
        codec_types = lines[1::2]
        has_video = "video" in codec_types
        has_audio = "audio" in codec_types
        return has_video, has_audio, ", ".join(lines)
    except Exception as e:
        return False, False, str(e)


def download(url, workdir, cookies_from_browser=None):
    out_tmpl = workdir / "source.%(ext)s"
    # Highest quality: best video + best audio, merged to mp4 (needs ffmpeg on PATH).
    # Use bestvideo+bestaudio/best (no *) to ensure separate streams are combined.
    # If no separate streams exist, falls back to best single format (progressive).
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo+bestaudio/best",
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", str(out_tmpl),
        url,
    ]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    run_cmd(cmd)
    
    # Find the output file (should be source.mp4 after merge)
    vids = [p for p in sorted(workdir.glob("source.*"))
            if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov", ".m4v")]
    if not vids:
        raise SystemExit("Download failed: no video file produced.")
    
    video_path = vids[0].resolve()
    
    # Validate the downloaded file
    if video_path.stat().st_size < 1024:
        raise SystemExit(f"Download failed: file too small ({video_path.stat().st_size} bytes)")
    
    has_video, has_audio, stream_details = _get_streams(video_path)
    if not has_video:
        raise SystemExit(
            f"Download produced video file missing VIDEO stream: {video_path.name} "
            f"({video_path.stat().st_size} bytes). Streams: {stream_details}"
        )
    if not has_audio:
        raise SystemExit(
            f"Download produced video file missing AUDIO stream: {video_path.name} "
            f"({video_path.stat().st_size} bytes). Streams: {stream_details}. "
            f"yt-dlp format selection may have failed to merge audio."
        )
    
    return video_path


def prepare_local_video(video_path, workdir):
    """Copy a local video file to workdir as source.mp4 for the pipeline.
    
    Returns the path to the copied video file.
    """
    video_path = Path(video_path).resolve()
    if not video_path.exists():
        raise SystemExit(f"Video file not found: {video_path}")
    
    # Validate source has audio before copying
    has_video, has_audio, stream_details = _get_streams(video_path)
    if not has_audio:
        raise SystemExit(
            f"Local video file missing AUDIO stream: {video_path.name}. "
            f"Streams: {stream_details}. Transcription requires audio."
        )
    if not has_video:
        raise SystemExit(
            f"Local video file missing VIDEO stream: {video_path.name}. "
            f"Streams: {stream_details}."
        )
    
    # Copy to workdir with a standard name
    dest = workdir / "source.mp4"
    shutil.copy2(video_path, dest)
    return dest.resolve()


def _is_memory_error(exc):
    message = str(exc).lower()
    return any(marker in message for marker in (
        "mkl_malloc", "out of memory", "memory allocation", "failed to allocate", "bad_alloc"
    ))


def _load_whisper_model(model_size, device, compute_type, progress_callback=None):
    from faster_whisper import WhisperModel

    automatic = device == "auto"
    if automatic:
        try:
            import ctranslate2
            device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception as exc:
            print(f"CUDA detection failed ({exc}); using CPU int8.", flush=True)
            device = "cpu"
        compute_type = "int8_float16" if device == "cuda" else "int8"

    model_order = ("tiny", "base", "small", "medium", "large-v3")
    smaller_models = []
    if model_size in model_order:
        smaller_models = list(reversed(model_order[:model_order.index(model_size)]))

    candidates = [(model_size, device, compute_type)]
    if automatic and device == "cuda":
        candidates.append((model_size, "cpu", "int8"))
        for size in smaller_models:
            candidates.extend(((size, "cuda", "int8_float16"), (size, "cpu", "int8")))
    else:
        candidates.extend((size, "cpu", "int8") for size in smaller_models)

    for index, (candidate_size, candidate_device, candidate_compute) in enumerate(candidates):
        print(f"Loading faster-whisper model={candidate_size} device={candidate_device} "
              f"compute={candidate_compute} (first run downloads the model)...", flush=True)
        try:
            model = WhisperModel(candidate_size, device=candidate_device, compute_type=candidate_compute)
            return model, candidate_device, candidate_size
        except Exception as exc:
            if not automatic or index + 1 >= len(candidates):
                raise
            next_size, next_device, next_compute = candidates[index + 1]
            if candidate_device == "cuda" and next_device == "cpu" and next_size == candidate_size:
                message = f"CUDA model setup failed; retrying {candidate_size} on CPU int8."
            elif _is_memory_error(exc):
                message = (f"Not enough memory for {candidate_size} on {candidate_device.upper()}; "
                           f"retrying with {next_size} on {next_device.upper()} ({next_compute}).")
            else:
                raise
            print(message, flush=True)
            if progress_callback:
                progress_callback("transcribe", "loading_model", None, message)


def transcribe(video, workdir, model_size, device, compute_type, progress_callback=None):
    if progress_callback:
        progress_callback("transcribe", "loading_model", None, f"Detecting device and loading Whisper model {model_size}...")
    model, device, model_size = _load_whisper_model(model_size, device, compute_type, progress_callback)
    device_message = f"Whisper model {model_size} is using {device.upper()}."
    print(device_message, flush=True)
    if progress_callback:
        progress_callback("transcribe", "loading_model", None, device_message)
    
    # Get video duration for progress estimation
    import subprocess
    dur_out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
        capture_output=True, text=True
    )
    video_duration = float(dur_out.stdout.strip()) if dur_out.stdout.strip() else 0
    if video_duration <= 0:
        raise RuntimeError("Could not determine video duration with ffprobe.")

    chunk_seconds = 300
    chunk_count = math.ceil(video_duration / chunk_seconds)
    if progress_callback:
        progress_callback("transcribe", "starting", 0.0,
                          f"Transcribing {video_duration:.1f}s video in {chunk_count} five-minute chunks...")

    seg_list, words = [], []
    language = None
    audio_duration = 0.0
    last_progress = 0.0
    with tempfile.TemporaryDirectory(prefix="chopify-audio-", dir=workdir) as temp_dir:
        for chunk_index in range(chunk_count):
            chunk_start = chunk_index * chunk_seconds
            chunk_duration = min(chunk_seconds, video_duration - chunk_start)
            chunk_path = Path(temp_dir) / f"audio-{chunk_index:04d}.wav"
            if progress_callback:
                progress_callback("transcribe", "transcribing", chunk_start / video_duration,
                                  f"Preparing audio chunk {chunk_index + 1}/{chunk_count}...")
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                 "-ss", str(chunk_start), "-i", str(video), "-t", str(chunk_duration),
                 "-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000",
                 "-c:a", "pcm_s16le", str(chunk_path)],
                check=True, capture_output=True, text=True
            )
            if progress_callback:
                progress_callback("transcribe", "transcribing", chunk_start / video_duration,
                                  f"Transcribing audio chunk {chunk_index + 1}/{chunk_count}...")
            segments, info = model.transcribe(
                str(chunk_path), language=language, word_timestamps=True, vad_filter=True
            )
            language = language or info.language
            audio_duration += info.duration
            print(f"Transcribing chunk {chunk_index + 1}/{chunk_count} on {device.upper()}...", flush=True)

            for seg in segments:
                start = chunk_start + seg.start
                end = chunk_start + seg.end
                seg_list.append({"start": start, "end": end, "text": seg.text})
                for word in (seg.words or []):
                    words.append({
                        "word": word.word,
                        "start": chunk_start + word.start,
                        "end": chunk_start + word.end,
                    })
                print(f"[{start:7.2f} -> {end:7.2f}] {seg.text}", flush=True)

                prog = min(end / video_duration, 1.0)
                if progress_callback and (prog - last_progress >= 0.02 or prog >= 1.0):
                    progress_callback("transcribe", "transcribing", prog,
                                      f"Transcribing: {end:.1f}/{video_duration:.1f}s")
                    last_progress = prog
            del segments, info

    transcript = {
        "video": str(video),
        "language": language,
        "duration": audio_duration,
        "model": model_size,
        "segments": seg_list,
        "words": words,
    }
    out = workdir / "transcript.json"
    out.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out}  ({len(words)} words, {len(seg_list)} segments)", flush=True)
    if progress_callback:
        progress_callback("transcribe", "complete", 1.0, "Transcription complete")
    return out


def run(workdir, model_size="medium", device="auto", compute_type="int8", url=None, local_video=None, cookies_from_browser=None, progress_callback=None):
    """Run stage 1: download/prepare video and transcribe.
    
    Args:
        workdir: Working directory path
        model_size: Whisper model size
        device: Device to run on (auto/cpu/cuda)
        compute_type: Compute type (int8/float16)
        url: YouTube URL (optional)
        local_video: Path to local video file (optional)
        cookies_from_browser: Browser to extract cookies from (firefox, chrome, edge, brave, or None)
        progress_callback: Optional callback(stage, sub_stage, progress, message)
    
    Returns:
        Path to transcript.json
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    
    if url and local_video:
        raise SystemExit("Provide either --url or --local-video, not both")
    if not url and not local_video:
        raise SystemExit("Provide either --url or --local-video")
    
    if url:
        video = download(url, workdir, cookies_from_browser=cookies_from_browser)
        print("Downloaded:", video, flush=True)
    else:
        video = prepare_local_video(local_video, workdir)
        print("Prepared local video:", video, flush=True)
    
    return transcribe(video, workdir, model_size, device, compute_type, progress_callback)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", help="YouTube URL (or any yt-dlp-supported URL)")
    ap.add_argument("--local-video", help="Path to local video file (MP4, MKV, MOV, etc.)")
    ap.add_argument("--workdir", default="work")
    ap.add_argument("--model", default="medium",
                    help="faster-whisper model: tiny/base/small/medium/large-v3")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                    help="Transcription device (auto selects CUDA when available, otherwise CPU)")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--cookies-from-browser", choices=["firefox", "chrome", "edge", "brave"],
                    help="Extract YouTube cookies from browser (firefox, chrome, edge, brave)")
    args = ap.parse_args()

    run(args.workdir, args.model, args.device, args.compute_type, args.url, args.local_video, args.cookies_from_browser)
    print("STAGE 1 COMPLETE", flush=True)


if __name__ == "__main__":
    main()
