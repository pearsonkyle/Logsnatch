"""Tests for the redaction module."""

import os

import pytest

from logsnatch.redaction.anonymizer import Anonymizer
from logsnatch.redaction.secrets import redact_text, scan_text


def test_anthropic_api_key_detected():
    text = "key=sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789ABCDEF"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "anthropic_api_key" in names


def test_openai_api_key_detected():
    text = "Authorization: sk-abcdefghijklmnopqrstuvwx"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "openai_api_key" in names


def test_github_token_detected():
    text = "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "github_token" in names


def test_jwt_token_detected():
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    findings = scan_text(jwt)
    names = [f.pattern_name for f in findings]
    assert "jwt_token" in names


def test_email_detected():
    text = "Contact alice@somecompany.io for details"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "email" in names


def test_email_allowlist_skipped():
    text = "noreply@github.com should not be flagged"
    findings = scan_text(text)
    email_findings = [f for f in findings if f.pattern_name == "email"]
    assert len(email_findings) == 0


def test_pem_private_key_detected():
    key = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA1234567890abcdef\n"
        "-----END RSA PRIVATE KEY-----"
    )
    findings = scan_text(key)
    names = [f.pattern_name for f in findings]
    assert "private_key" in names


def test_redact_text_replaces_secrets():
    text = "My key is sk-ant-api03-abcdefghijklmnopqrstuvwxyz0123456789ABCDEF"
    redacted, count = redact_text(text)
    assert count >= 1
    assert "sk-ant-api03" not in redacted
    assert "[REDACTED_ANTHROPIC_API_KEY]" in redacted


def test_redact_text_no_secrets():
    text = "Hello world, this is a normal sentence."
    redacted, count = redact_text(text)
    assert count == 0
    assert redacted == text


def test_anonymizer_username_replacement(monkeypatch):
    monkeypatch.setenv("USER", "johndoe")
    anon = Anonymizer()
    result = anon.text("Hello from johndoe on this machine")
    assert "johndoe" not in result
    assert "user_" in result


def test_anonymizer_home_dir_replacement():
    home = str(__import__("pathlib").Path.home())
    anon = Anonymizer()
    result = anon.text(f"file is at {home}/config.yaml")
    assert home not in result
    assert "REDACTED_USER" in result


def test_anonymizer_short_username_not_replaced(monkeypatch):
    monkeypatch.setenv("USER", "ab")
    anon = Anonymizer()
    result = anon.text("user ab logged in")
    assert result == "user ab logged in"


def test_anonymizer_extra_usernames():
    anon = Anonymizer(extra_usernames=["secretuser"])
    result = anon.text("path /home/secretuser/data")
    assert "secretuser" not in result


def test_huggingface_token_detected():
    text = "HF_TOKEN=hf_abcdefghijklmnopqrstuvwxyz"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "huggingface_token" in names


def test_aws_access_key_detected():
    text = "aws_access_key_id = AKIAIOSFODNN7EXAMPLE"
    findings = scan_text(text)
    names = [f.pattern_name for f in findings]
    assert "aws_access_key" in names
