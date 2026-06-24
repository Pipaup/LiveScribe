"""Main pipeline — coordinates Audio → VAD → ASR → Output across two threads.

Architecture:
  Main thread:  audio.read_chunk() → VAD state machine → segment assembly
  ASR thread:   audio_queue.get() → engine.transcribe() → result_queue
  Main thread:  result_queue.get_nowait() → output_manager.emit_all()
"""

import logging
import queue
import signal
import threading
from typing import Optional

import numpy as np

from livescribe.asr.base import BaseASREngine
from livescribe.asr.factory import create_asr_engine
from livescribe.audio.capture import AudioCapture
from livescribe.config.schema import LiveScribeConfig, RunMode
from livescribe.exceptions import AudioDeviceError, PipelineError
from livescribe.output.output_manager import OutputManager
from livescribe.vad.detector import VADDetector

logger = logging.getLogger(__name__)

# Sentinel value to tell the ASR thread to exit
_SENTINEL = "__STOP__"
# Max audio queue size in frames (~96 seconds at 32ms/frame)
_MAX_AUDIO_QUEUE = 3000


class Pipeline:
    """Orchestrates the full speech-to-text pipeline."""

    def __init__(self, config: LiveScribeConfig) -> None:
        self._config = config
        self._running = False
        self._audio: Optional[AudioCapture] = None
        self._vad: Optional[VADDetector] = None
        self._engine: Optional[BaseASREngine] = None
        self._output: Optional[OutputManager] = None
        self._audio_queue: queue.Queue = queue.Queue(maxsize=_MAX_AUDIO_QUEUE)
        self._result_queue: queue.Queue = queue.Queue()
        self._asr_thread: Optional[threading.Thread] = None

    def run(self) -> None:
        """Start the pipeline. Blocks until stop() is called or Ctrl+C."""
        logger.info("=" * 50)
        logger.info("LiveScribe starting — mode=%s, engine=%s",
                     self._config.mode.value, self._config.asr.engine.value)

        try:
            self._setup()
        except Exception:
            logger.exception("Failed to initialize pipeline")
            self._teardown()
            raise

        self._running = True

        # Start ASR worker thread
        self._asr_thread = threading.Thread(
            target=self._asr_worker,
            name="ASRWorker",
            daemon=True,
        )
        self._asr_thread.start()

        # Main thread: audio capture + VAD
        try:
            if self._config.mode == RunMode.VAD_SEGMENTED:
                self._run_segmented_capture()
            elif self._config.mode == RunMode.STREAMING:
                self._run_streaming_capture()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received, shutting down")
        except Exception:
            logger.exception("Pipeline error")
            raise
        finally:
            self._running = False
            # Send sentinel to ASR thread
            try:
                self._audio_queue.put(_SENTINEL, timeout=1)
            except queue.Full:
                pass
            # Wait for ASR thread
            if self._asr_thread and self._asr_thread.is_alive():
                self._asr_thread.join(timeout=30)
            self._teardown()

    def stop(self) -> None:
        """Signal the pipeline to stop gracefully."""
        self._running = False

    # ==================== Setup / Teardown ====================

    def _setup(self) -> None:
        cfg = self._config

        # Audio capture
        self._audio = AudioCapture(cfg.audio)
        self._audio.open()
        logger.info("Audio capture ready — chunk=%d samples @ %d Hz",
                     self._audio.chunk_samples, self._audio.sample_rate)

        # VAD (segmented mode only)
        if cfg.mode == RunMode.VAD_SEGMENTED and cfg.vad.enabled:
            self._vad = VADDetector(cfg.vad)
            logger.info("VAD detector ready")
        else:
            self._vad = None

        # ASR engine
        self._engine = create_asr_engine(cfg.asr)
        logger.info("ASR engine ready — %s", self._engine.engine_name)

        # Output
        self._output = OutputManager(cfg.output)
        logger.info("Output manager ready — %d backend(s)", self._output.active_count)

    def _teardown(self) -> None:
        # Output
        if self._output is not None:
            try:
                self._output.close()
            except Exception:
                logger.exception("Output close failed")
        # ASR engine (uses shutdown(), not close())
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception:
                logger.exception("ASR engine shutdown failed")
        # VAD
        if self._vad is not None:
            try:
                self._vad.close()
            except Exception:
                logger.exception("VAD close failed")
        # Audio
        if self._audio is not None:
            try:
                self._audio.close()
            except Exception:
                logger.exception("Audio close failed")

        logger.info("LiveScribe stopped")

    # ==================== Segmented mode ====================

    def _run_segmented_capture(self) -> None:
        """Main thread: read frames → VAD state machine → send segments to ASR thread."""
        buffer: list[np.ndarray] = []
        pre_speech: list[np.ndarray] = []  # Rolling window for speech_pad_ms before start
        is_speaking = False
        max_segment_frames = int(
            self._config.vad.max_segment_duration_s * 1000 / self._config.audio.chunk_duration_ms
        )
        pad_frames = int(self._config.vad.speech_pad_ms / self._config.audio.chunk_duration_ms)

        while self._running:
            try:
                chunk = self._audio.read_chunk()
            except AudioDeviceError:
                logger.exception("Audio read error, attempting to continue")
                continue

            if self._vad is not None:
                event = self._vad.process(chunk)

                if event == "start":
                    is_speaking = True
                    # Prepend pad frames captured before speech began
                    if pad_frames > 0 and pre_speech:
                        buffer = pre_speech[-pad_frames:]
                    pre_speech = []

                elif event == "end" and is_speaking:
                    if buffer:
                        self._send_segment(_concat(buffer))
                    buffer = []
                    is_speaking = False
                    self._vad.reset()

                # Forced cutoff (very long utterance)
                if is_speaking and len(buffer) >= max_segment_frames:
                    self._send_segment(_concat(buffer))
                    buffer = []
                    # Start fresh — no overlap to avoid duplicate transcription

            else:
                # Blind segmentation (no VAD): energy-based
                if is_speaking:
                    buffer.append(chunk)
                    if len(buffer) >= max_segment_frames:
                        self._send_segment(_concat(buffer))
                        buffer = []
                        is_speaking = False
                else:
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    if rms > 0.01:
                        is_speaking = True
                        buffer.append(chunk)

            if is_speaking:
                buffer.append(chunk)
            elif self._vad is not None:
                # Pre-speech: maintain rolling window for pad frames
                pre_speech.append(chunk)
                keep = max(pad_frames, 10)
                if len(pre_speech) > keep:
                    pre_speech = pre_speech[-keep:]

            # Drain results (non-blocking)
            self._drain_results()

        # Flush remaining buffer
        if buffer and is_speaking:
            self._send_segment(_concat(buffer))

    # ==================== Streaming mode ====================

    def _run_streaming_capture(self) -> None:
        """Main thread: read frames → feed directly to audio queue."""
        self._engine.start_stream()

        while self._running:
            try:
                chunk = self._audio.read_chunk()
            except AudioDeviceError:
                logger.exception("Audio read error, attempting to continue")
                continue

            try:
                self._audio_queue.put(chunk, timeout=0.1)
            except queue.Full:
                logger.warning("Audio queue full, dropping frame")

            self._drain_results()

        # Signal end of stream
        self._send_segment(None)  # None = end of streaming session

    # ==================== ASR worker thread ====================

    def _asr_worker(self) -> None:
        """Dedicated thread: consume audio from queue → run ASR → push results."""
        if self._config.mode == RunMode.STREAMING:
            self._asr_worker_streaming()
        else:
            self._asr_worker_segmented()

    def _asr_worker_segmented(self) -> None:
        """ASR thread for segmented mode: wait for complete audio segments."""
        while True:
            item = self._audio_queue.get()

            if item is _SENTINEL:
                break

            # item is (tag, data) tuple from _send_segment()
            tag, data = item

            if tag == "segment":
                try:
                    text = self._engine.transcribe(data)
                except Exception:
                    logger.exception("ASR transcribe failed for segment")
                    continue
                if text:
                    self._result_queue.put(("final", text))

            elif tag == "stop":
                break

    def _asr_worker_streaming(self) -> None:
        """ASR thread for streaming mode: process frame-by-frame."""
        while True:
            item = self._audio_queue.get()

            if item is _SENTINEL:
                break

            if isinstance(item, tuple):
                tag, data = item
                if tag == "stop":
                    # End of stream — flush
                    try:
                        final = self._engine.end_stream()
                        if final:
                            self._result_queue.put(("final", final))
                    except Exception:
                        logger.exception("ASR end_stream failed")
                    break
                continue

            # Regular audio chunk
            try:
                partial = self._engine.feed_chunk(item)
                if partial:
                    self._result_queue.put(("partial", partial))
            except Exception:
                logger.exception("ASR feed_chunk failed")

    # ==================== Inter-thread communication ====================

    def _send_segment(self, audio: np.ndarray | None) -> None:
        """Send a complete audio segment (or None=end-of-stream) to the ASR thread."""
        try:
            if audio is None:
                self._audio_queue.put(("stop", None), timeout=1)
            else:
                self._audio_queue.put(("segment", audio), timeout=1)
        except queue.Full:
            logger.warning("Audio queue full, dropping segment")

    def _drain_results(self) -> None:
        """Non-blocking: consume all pending ASR results and emit to output."""
        while True:
            try:
                tag, text = self._result_queue.get_nowait()
                if text:
                    self._output.emit_all(text, is_partial=(tag == "partial"))
            except queue.Empty:
                break


def _concat(frames: list[np.ndarray]) -> np.ndarray:
    """Concatenate audio frames into a single array."""
    if not frames:
        return np.array([], dtype=np.float32)
    if len(frames) == 1:
        return frames[0]
    return np.concatenate(frames)
