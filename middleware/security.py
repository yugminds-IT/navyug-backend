"""
Security middleware for rate limiting, CORS, and other security features.
"""
from fastapi import Request, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
from collections import defaultdict
from typing import Dict, Tuple
import logging

from core.logger import logger


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(self, app, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            requests_per_minute: Max requests per minute per IP
            requests_per_hour: Max requests per hour per IP
        """
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.minute_requests: Dict[str, list] = defaultdict(list)
        self.hour_requests: Dict[str, list] = defaultdict(list)
        self.cleanup_interval = 300  # Clean up old entries every 5 minutes
        self.last_cleanup = time.time()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Clean up old entries periodically
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time
        
        # Check rate limits
        if not self._check_rate_limit(client_ip, current_time):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please try again later."
            )
        
        # Process request
        response = await call_next(request)
        return response
    
    def _check_rate_limit(self, client_ip: str, current_time: float) -> bool:
        """Check if request is within rate limits."""
        # Clean minute requests (older than 1 minute)
        self.minute_requests[client_ip] = [
            t for t in self.minute_requests[client_ip]
            if current_time - t < 60
        ]
        
        # Clean hour requests (older than 1 hour)
        self.hour_requests[client_ip] = [
            t for t in self.hour_requests[client_ip]
            if current_time - t < 3600
        ]
        
        # Check limits
        if len(self.minute_requests[client_ip]) >= self.requests_per_minute:
            return False
        if len(self.hour_requests[client_ip]) >= self.requests_per_hour:
            return False
        
        # Add current request
        self.minute_requests[client_ip].append(current_time)
        self.hour_requests[client_ip].append(current_time)
        
        return True
    
    def _cleanup_old_entries(self, current_time: float):
        """Clean up old rate limit entries."""
        # Clean minute requests
        for ip in list(self.minute_requests.keys()):
            self.minute_requests[ip] = [
                t for t in self.minute_requests[ip]
                if current_time - t < 60
            ]
            if not self.minute_requests[ip]:
                del self.minute_requests[ip]
        
        # Clean hour requests
        for ip in list(self.hour_requests.keys()):
            self.hour_requests[ip] = [
                t for t in self.hour_requests[ip]
                if current_time - t < 3600
            ]
            if not self.hour_requests[ip]:
                del self.hour_requests[ip]


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to responses."""
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers."""
        response = await call_next(request)
        
        # Security headers (CSP relaxed for API to allow cross-origin requests from frontend)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # Only add strict HSTS/CSP in production if needed; skip CSP for API to avoid blocking CORS
        
        return response


def setup_cors(app, allowed_origins: list[str], allowed_methods: list[str] = None):
    """
    Setup CORS middleware.
    
    Args:
        app: FastAPI application
        allowed_origins: List of allowed origins
        allowed_methods: List of allowed HTTP methods
    """
    if allowed_methods is None:
        allowed_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"]
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=allowed_methods,
        allow_headers=["*"],
        expose_headers=["*"],
    )


def setup_trusted_hosts(app, allowed_hosts: list[str]):
    """
    Setup trusted hosts middleware.
    
    Args:
        app: FastAPI application
        allowed_hosts: List of allowed hostnames
    """
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=allowed_hosts
    )
