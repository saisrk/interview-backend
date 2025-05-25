import pytest
from fastapi.testclient import TestClient
from app import app
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@pytest.fixture
def client():
    """Create a test client for the FastAPI application"""
    return TestClient(app)

@pytest.fixture
def test_user_data():
    """Test user data for authentication tests"""
    return {
        "email": "test@example.com",
        "password": "testpassword123",
        "full_name": "Test User",
        "role": "candidate"
    }

@pytest.fixture
def test_login_data():
    """Test login data"""
    return {
        "email": "test@example.com",
        "password": "testpassword123"
    } 