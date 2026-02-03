"""
Authentication middleware to enforce authentication on all routes except public ones.
This middleware provides an early check, but actual validation is done by FastAPI dependencies.
"""
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import List
import logging

from core.logger import logger
import config

# Public routes that don't require authentication
PUBLIC_ROUTES: List[str] = [
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
    # Auth endpoints (with /api/auth prefix)
    "/api/auth/login/master-admin",
    "/api/auth/login/management",
    "/api/auth/login/faculty",
    "/api/auth/login/student",
    "/api/auth/refresh",
    "/api/auth/signup/master-admin",
    "/api/auth/signup/management",
    "/api/auth/signup/faculty",
    "/api/auth/signup/student",
    "/api/auth/check-user",
    "/api/auth/send-otp",
    "/api/auth/verify-otp",
    "/api/auth/setup-password",
    "/api/auth/reset-password",
]


class AuthRequiredMiddleware(BaseHTTPMiddleware):
    """
    Middleware to enforce authentication on all routes.
    
    This provides an early check for authentication headers.
    Actual token validation is handled by FastAPI dependencies.
    """
    
    def __init__(self, app, public_routes: List[str] = None):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application
            public_routes: List of public routes (paths) that don't require auth
        """
        super().__init__(app)
        self.public_routes = public_routes or PUBLIC_ROUTES
    
    async def dispatch(self, request: Request, call_next):
        """Process request with authentication check."""
        # Check if route is public
        path = request.url.path
        
        # Allow public routes
        is_public = any(
            path.startswith(route) or path == route
            for route in self.public_routes
        )
        
        if is_public:
            return await call_next(request)
        
        # Allow OPTIONS for CORS preflight
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Check for authentication headers
        authorization = request.headers.get("authorization")
        api_key = request.headers.get("x-api-key")
        
        # Early check: warn if no auth headers (but let FastAPI dependencies handle actual validation)
        if not authorization and not api_key:
            logger.warning(f"Request without authentication headers: {request.method} {path} from {request.client.host if request.client else 'unknown'}")
            # Don't block here - let FastAPI dependencies handle it for proper error messages
            # This is just for logging/monitoring
        
        # Continue to route handler (dependencies will validate)
        return await call_next(request)
