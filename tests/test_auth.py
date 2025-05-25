import pytest
from fastapi import status
import re

def test_healthcheck(client):
    """Test the healthcheck endpoint"""
    response = client.get("/healthcheck")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Smart Interview Companion API v2.0"
    assert data["database"] == "Supabase"

def test_signup_success(client, test_user_data):
    """Test successful user signup"""
    response = client.post("/auth/signup", json=test_user_data)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "user_id" in data
    assert data["email"] == test_user_data["email"]
    assert "requires_email_confirmation" in data

def test_signup_invalid_email(client, test_user_data):
    """Test signup with invalid email format"""
    invalid_data = test_user_data.copy()
    invalid_data["email"] = "invalid-email"
    response = client.post("/auth/signup", json=invalid_data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Invalid email format"

def test_signup_short_password(client, test_user_data):
    """Test signup with password shorter than 8 characters"""
    invalid_data = test_user_data.copy()
    invalid_data["password"] = "short"
    response = client.post("/auth/signup", json=invalid_data)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

def test_signup_duplicate_email(client, test_user_data):
    """Test signup with already registered email"""
    # First signup
    client.post("/auth/signup", json=test_user_data)
    # Try to signup again with same email
    response = client.post("/auth/signup", json=test_user_data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Email already registered"

def test_login_success(client, test_user_data, test_login_data):
    """Test successful login"""
    # First signup
    client.post("/auth/signup", json=test_user_data)
    # Then login
    response = client.post("/auth/login", json=test_login_data)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "user" in data
    assert "session" in data
    assert data["user"]["email"] == test_login_data["email"]
    assert "access_token" in data["session"]
    assert "refresh_token" in data["session"]

def test_login_invalid_credentials(client):
    """Test login with invalid credentials"""
    invalid_data = {
        "email": "wrong@example.com",
        "password": "wrongpassword"
    }
    response = client.post("/auth/login", json=invalid_data)
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Invalid credentials"

def test_login_invalid_email_format(client):
    """Test login with invalid email format"""
    invalid_data = {
        "email": "invalid-email",
        "password": "testpassword123"
    }
    response = client.post("/auth/login", json=invalid_data)
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"] == "Invalid email format"

def test_refresh_token(client, test_user_data, test_login_data):
    """Test token refresh"""
    # First signup and login
    client.post("/auth/signup", json=test_user_data)
    login_response = client.post("/auth/login", json=test_login_data)
    refresh_token = login_response.json()["session"]["refresh_token"]
    
    # Test refresh
    response = client.post("/auth/refresh", json={"refresh_token": refresh_token})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert "expires_at" in data

def test_refresh_token_invalid(client):
    """Test refresh token with invalid token"""
    response = client.post("/auth/refresh", json={"refresh_token": "invalid-token"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Invalid refresh token"

def test_logout(client, test_user_data, test_login_data):
    """Test logout"""
    # First signup and login
    client.post("/auth/signup", json=test_user_data)
    login_response = client.post("/auth/login", json=test_login_data)
    access_token = login_response.json()["session"]["access_token"]
    
    # Test logout
    response = client.post(
        "/auth/logout",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["message"] == "Logged out successfully"

def test_logout_unauthorized(client):
    """Test logout without authentication"""
    response = client.post("/auth/logout")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED 