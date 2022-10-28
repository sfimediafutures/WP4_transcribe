#!/usr/bin/env python3
import contextlib
import sys
import wave
import json
import webrtcvad
import operator
from argparse import ArgumentParser
import tempfile
import os


ccmodule = {
    "description": "Detect voices, create an output file with start,end tags",
    "depends": [],
    "provides": [],
    "inputs": {
        "src": "Source file to detect from (WAVE file)",
        "dst": "Destination file, .json or .csv",
        "format": "Format for end file - autodetected if dst given, otherwise dst is named after format",
        "agressive": "How agressive to be (1-3), default 2",
        "max_pause": "What's the maximum pause between segments for it to be detected as two?",
        "max_segment_length": "Max lenght of detected segments",
        "output_dir": "Destination directory for wave segments - if blank no segments are stored"
    },
    "outputs": {
        "dst": "Output file"
    },
    "defaults": {
        "priority": 50,  # Normal
        "runOn": "success"
    },
    "status": {
        "progress": "Progress 0-100%",
        "state": "Current state of processing"
    }
}


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class VoiceDetector:

    def __init__(self, sourcefile, output_dir=None):

        self.is_tmp = False
        self.output_dir = output_dir

        if not sourcefile.endswith(".wav"):
            self.is_tmp = True
            self.sourcefile = self.convert(sourcefile)
        else:
            self.sourcefile = sourcefile

        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def __del__(self):
        if self.is_tmp:
            os.remove(self.sourcefile)

    def analyze(self, aggressive=2, max_segment_length=8, max_pause=0, framelen=30):
        audio, sample_rate = self.read_wave(self.sourcefile)
        vad = webrtcvad.Vad(int(aggressive))
        frames = self.frame_generator(framelen, audio, sample_rate)
        frames = list(frames)
        segments = self.vad_collector(sample_rate, framelen, 3, vad, frames,
                                      output_dir=self.output_dir,
                                      max_segment_length=max_segment_length,
                                      max_pause=max_pause)

        return segments

    def read_wave(self, path):
        """Reads a .wav file.

        Takes the path, and returns (PCM audio data, sample rate).
        """
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            # assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate in (8000, 16000, 32000, 48000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    def frame_generator(self, frame_duration_ms, audio, sample_rate):
        """Generates audio frames from PCM audio data.

        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.

        Yields Frames of the requested duration.
        """
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frame_duration_ms,
                      padding_frames, vad, frames,
                      output_dir=None, max_segment_length=None, max_pause=0):
        """
        If output dir is given, speech audio segments are saved there
        """
        triggered = False
        segments = []
        voiced_frames = []
        segment_data = []
        frames_speech = 0
        frames_audio = 0
        padding = start = end = 0
        for idx, frame in enumerate(frames):
            is_speech = vad.is_speech(frame.bytes, sample_rate)
            if is_speech:
                frames_speech += 1
            else:
                frames_audio += 1

            if is_speech:
                segment_data.append(frame)

            if not triggered and is_speech:
                triggered = True
                start = idx * frame_duration_ms
                # segment_data = [frame]
                # if start == 0:
                #    raise SystemExit("What, starts with voice (%d)?" % idx)

            elif triggered and (not is_speech or \
                  (idx * frame_duration_ms) - start > max_segment_length * 1000):
                if padding < padding_frames:
                    padding += 1
                    continue
                triggered = False
                end = idx * frame_duration_ms

                s = {"type": "voice", "start": start / 1000., "end": end / 1000., "idx": idx}

                if output_dir:
                    target = os.path.join(output_dir, "segment_%08d.wav" % idx)

                merged = False
                if max_pause and len(segments) > 0:
                    if s["start"] - segments[-1]["end"] < max_pause and \
                     s["end"] - segments[-1]["start"] < max_segment_length:

                        # Only merge if the last segment is too short
                        if segments[-1]["end"] - segments[-1]["start"] < 4.0:
                            merged = True
                            # MERGE
                            print("MERGING", segments[-1]["end"], s["start"], segments[-1]["idx"])
                            segments[-1]["end"] = s["end"]
                            # We should overwrite the last file if output_dir is given!
                            if output_dir:
                                target = segments[-1]["file"]
                            s = segments[-1]
                            if "data" in s:
                                segment_data = segments[-1]["data"] + segment_data

                # Save the audio segment if requested
                if output_dir:
                    # target = os.path.join(output_dir, "segment_%08d.wav" % idx)
                    with wave.open(target, "w") as target_f:
                        target_f.setnchannels(1)
                        target_f.setsampwidth(2)
                        target_f.setframerate(sample_rate)
                        for d in segment_data:
                            target_f.writeframes(d.bytes)
                    s["file"] = target
                s["data"] = segment_data

                if not merged:
                    segments.append(s)
                start = end = padding = 0
                segment_data = []
            elif triggered and is_speech:
                padding = 0

        if output_dir:
            for s in segments:
                del s["data"]
        return segments

    def convert(self, mp3file):

        import subprocess
        fd, tmpfile = tempfile.mkstemp(suffix=".wav")
        print("Extracting audio to", tmpfile)
        cmd = ["ffmpeg", "-i", mp3file, "-vn", "-ac", "1", "-y", tmpfile]
        print(" ".join(cmd))
        s = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        s.wait()
        print(s.poll())

        print("Analyzing")
        return tmpfile


def process_task(cc, task):

    args = task["args"]
    src = args["src"]
    dst = args.get("dst", None)
    if not dst:
        dst = os.path.splitext(src)[0] + "_segments." + args.get("format", "json")

    detector = VoiceDetector(src, output_dir=args.get("output_dir", None))
    segments = detector.analyze(aggressive=args.get("aggressive", 2),
                                max_pause=float(args.get("max_pause", 0)),
                                max_segment_length=float(args.get("max_segment_length", 30)))

    # Dump json
    if dst.endswith("json"):
        with open(dst, "w") as f:
            json.dump(segments, f, indent=" ")
    else:
        # Dump CSV
        with open(dst, "w") as f:
            f.write("speaker,start,end,duration,audio_path\n")
            for item in segments:
                if "who" not in item:
                    item["who"] = "unknown"
                f.write("%s,%f,%f,%f,%s\n" % (item["who"],
                                              item["start"],
                                              item["end"],
                                              item["end"] - item["start"],
                                              src))

    return 100, {"dst": dst}