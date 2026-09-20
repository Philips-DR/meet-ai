"""Layers 4-6: correction, notes, and the verification that makes them trustworthy.

This half of meet-ai consumes a timeline and produces notes. It never decodes audio and
never talks to a transcription model -- transcribe.py's timeline JSON is the whole
interface between the two halves, which is why that file is written on every run.
"""
