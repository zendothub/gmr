"""Encryption and hashing utilities (direct bcrypt, no passlib)."""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from jose import jwt


from app.config import get_settings

settings = get_settings()

# bcrypt operates on the first 72 bytes only
_BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    """Hash a plain-text password with bcrypt."""
    pw_bytes = password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain-text password against its bcrypt hash."""
    try:
        pw_bytes = plain_password.encode("utf-8")[:_BCRYPT_MAX_BYTES]
        return bcrypt.checkpw(pw_bytes, hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a short-lived JWT access token.

    Args:
        data: Payload data (should include 'sub' for user ID)
        expires_delta: Token expiration time delta

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode.update({"exp": expire, "type": "access", "jti": uuid.uuid4().hex})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def create_refresh_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a long-lived JWT refresh token.

    The refresh token carries ``type=refresh`` so it can never be used as an
    access token (and vice-versa).
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
    to_encode.update({"exp": expire, "type": "refresh", "jti": uuid.uuid4().hex})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)



def decode_token(token: str) -> dict:
    """Decode and validate a JWT, returning its claims. Raises JWTError on failure."""
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])

