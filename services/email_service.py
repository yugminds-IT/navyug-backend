"""
Email service for sending OTPs and notifications.
Uses fastapi-mail for OTP and other async email sending.
"""
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, TYPE_CHECKING
from datetime import datetime, timedelta
import random
import string

from core.logger import logger
import config

if TYPE_CHECKING:
    from fastapi_mail import FastMail

# SMTP fallback config (used by sync send_email when FastMail not used)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USER)
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Campus Safety App")


class EmailService:
    """Service for sending emails via fastapi-mail (OTP) or SMTP fallback."""

    @staticmethod
    def generate_otp(length: int = 6) -> str:
        """
        Generate a random OTP.

        Args:
            length: Length of OTP (default 6)

        Returns:
            OTP string
        """
        return "".join(random.choices(string.digits, k=length))

    @staticmethod
    async def send_otp_email(to_email: str, otp: str, fm: "FastMail") -> bool:
        """
        Send OTP email using fastapi-mail.

        Args:
            to_email: Recipient email address
            otp: OTP code to send
            fm: FastMail instance (from request.app.state.mail)

        Returns:
            True if sent successfully, False otherwise
        """
        from fastapi_mail import MessageSchema, MessageType

        subject = "Your OTP Code - Campus Safety App"
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #4CAF50;">Campus Safety App</h2>
                <p>Your OTP code is:</p>
                <div style="background-color: #f4f4f4; padding: 20px; text-align: center; margin: 20px 0; border-radius: 5px;">
                    <h1 style="color: #4CAF50; font-size: 32px; margin: 0; letter-spacing: 5px;">{otp}</h1>
                </div>
                <p>This code will expire in 10 minutes.</p>
                <p style="color: #666; font-size: 12px;">If you didn't request this code, please ignore this email.</p>
            </div>
        </body>
        </html>
        """
        message = MessageSchema(
            subject=subject,
            recipients=[to_email],
            body=html_body,
            subtype=MessageType.html,
        )
        try:
            await fm.send_message(message)
            logger.info(f"OTP email sent successfully to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send OTP email to {to_email}: {e}", exc_info=True)
            return False

    @staticmethod
    def send_email(
        to_email: str,
        subject: str,
        html_body: Optional[str] = None,
        text_body: Optional[str] = None,
    ) -> bool:
        """
        Send an email via SMTP (fallback for non-OTP emails).

        Args:
            to_email: Recipient email address
            subject: Email subject
            html_body: HTML email body (optional)
            text_body: Plain text email body (optional, required if html_body not provided)

        Returns:
            True if sent successfully, False otherwise
        """
        if not SMTP_USER or not SMTP_PASSWORD:
            logger.error(
                "SMTP credentials not configured. Set SMTP_USER and SMTP_PASSWORD environment variables."
            )
            return False

        if not html_body and not text_body:
            logger.error("Either html_body or text_body must be provided")
            return False

        try:
            msg = MIMEMultipart("alternative")
            msg["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
            msg["To"] = to_email
            msg["Subject"] = subject

            if text_body:
                msg.attach(MIMEText(text_body, "plain"))
            if html_body:
                msg.attach(MIMEText(html_body, "html"))

            if SMTP_USE_SSL:
                with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg)
            else:
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                    if SMTP_USE_TLS:
                        server.starttls()
                    server.login(SMTP_USER, SMTP_PASSWORD)
                    server.send_message(msg)

            logger.info(f"Email sent successfully to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}", exc_info=True)
            return False

    @staticmethod
    def send_password_reset_email(
        to_email: str, reset_link: Optional[str] = None
    ) -> bool:
        """Send password reset email (uses SMTP fallback)."""
        subject = "Password Reset Request - Campus Safety App"
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #4CAF50;">Password Reset Request</h2>
                <p>You requested to reset your password for Campus Safety App.</p>
                {f'<p><a href="{reset_link}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">Reset Password</a></p>' if reset_link else ''}
                <p>If you didn't request this, please ignore this email.</p>
            </div>
        </body>
        </html>
        """
        text_body = f"""
Password Reset Request

You requested to reset your password for Campus Safety App.
{f'Reset link: {reset_link}' if reset_link else ''}

If you didn't request this, please ignore this email.
        """
        return EmailService.send_email(
            to_email=to_email,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
        )

    @staticmethod
    def send_welcome_email(to_email: str, name: str, role: str) -> bool:
        """Send welcome email to new user (uses SMTP fallback)."""
        subject = "Welcome to Campus Safety App"
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #4CAF50;">Welcome to Campus Safety App!</h2>
                <p>Hello {name},</p>
                <p>Your account has been created successfully with role: <strong>{role}</strong></p>
                <p>You can now log in and start using the Campus Safety App.</p>
                <p>Thank you for joining us!</p>
            </div>
        </body>
        </html>
        """
        text_body = f"""
Welcome to Campus Safety App!

Hello {name},

Your account has been created successfully with role: {role}

You can now log in and start using the Campus Safety App.

Thank you for joining us!
        """
        return EmailService.send_email(
            to_email=to_email,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
        )
