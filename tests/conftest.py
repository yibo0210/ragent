"""Test environment setup — ensures required env vars are present."""
import os

# Set before any app imports to satisfy config validation
os.environ.setdefault("JWT_SECRET", "test-secret-for-pytest")
