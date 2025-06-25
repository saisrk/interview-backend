"""
Smart Interview Companion - Backend APIs with FastAPI + Supabase + Perplexity Sonar Pro
Production-ready implementation with persistent storage
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import httpx
from datetime import datetime, timedelta
import os
from contextlib import asynccontextmanager
from supabase import create_client, Client
import logging
from dotenv import load_dotenv
import re, bcrypt
from prompt_generator import InterviewConfig, generate_prompt
from collections import Counter
from openai import OpenAI, AsyncOpenAI, OpenAIError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models
class InterviewDomain(str, Enum):
    SOFTWARE_ENGINEERING = "software_engineering"
    DATA_SCIENCE = "data_science"
    PRODUCT_MANAGEMENT = "product_management"
    DESIGN = "design"
    MARKETING = "marketing"
    CONSULTING = "consulting"

class ExperienceLevel(str, Enum):
    JUNIOR = "entry"
    MID_LEVEL = "mid"
    SENIOR = "senior"
    LEAD = "lead"

class InterviewType(str, Enum):
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    SYSTEM_DESIGN = "system_design"
    CASE_STUDY = "case_study"

class InterviewMode(str, Enum):
    PRACTISE = "practise"
    REAL_TIME = "interview"

class PromptType(str, Enum):
    INTIAL_QUESTION = "initial_question"
    INTIAL_QUESTION_JD = "initial_question_jd"
    NEXT_QUESTION = "next_question"
    FEEDBACK = "feedback"

class SessionRequest(BaseModel):
    user_id: Optional[str] = None
    # For config-based interview
    domain: Optional[InterviewDomain] = None
    experience_level: Optional[ExperienceLevel] = None
    interview_type: Optional[InterviewType] = None
    duration_minutes: int = Field(default=30, ge=15, le=120)
    mode: InterviewMode
    company_name: Optional[str] = None
    job_title: Optional[str] = None
    location: Optional[str] = None
    job_description: Optional[str] = None

class AnswerRequest(BaseModel):
    session_id: str
    question_id: str
    answer: str
    response_time_seconds: int

class QuestionResponse(BaseModel):
    question_id: int
    question: str
    difficulty: str
    context: Optional[str] = None
    hints: List[str] = []
    citations: Optional[List[str]] = []

class FeedbackResponse(BaseModel):
    score: float
    strengths: List[str]
    improvements: List[str]
    industry_insights: List[str]
    follow_up_question: Optional[str] = None
    citations: Optional[List[str]] = []

class SessionSummary(BaseModel):
    session_id: str
    domain: str
    experience_level: str
    interview_type: str
    company_name: Optional[str]
    questions_answered: int
    average_score: float
    duration_minutes: int
    created_at: datetime
    status: str

class AuthRequest(BaseModel):
    email: str = Field(..., description="User's email address")
    password: str = Field(..., min_length=8, description="User's password (min 8 characters)")
    full_name: Optional[str] = Field(None, description="User's full name")
    role: Optional[str] = Field(None, description="User's role (e.g., 'candidate', 'interviewer')")

class LoginRequest(BaseModel):
    email: str = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")

class RefreshTokenRequest(BaseModel):
    refresh_token: str

class ChangePasswordRequest(BaseModel):
    user_id: str
    old_password: str
    new_password: str = Field(..., min_length=8)

# Load environment variables from .env file
load_dotenv()

# Supabase Configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")  # Service role key for server-side
PERPLEXITY_API_KEY = os.environ.get("PERPLEXITY_API_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Initialize database tables if needed
    logger.info("🚀 Smart Interview Companion Backend Starting...")
    await initialize_database()
    yield
    # Shutdown
    logger.info("👋 Shutting down...")

app = FastAPI(
    title="Smart Interview Companion API",
    description="AI-powered interview preparation with Supabase + Perplexity Sonar Pro",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Operations
class DatabaseManager:
    def __init__(self):
        self.supabase = supabase

    async def create_session(self, session_data: dict) -> str:
        """Create a new interview session in Supabase"""
        try:
            result = self.supabase.table("interview_sessions").insert(session_data).execute()
            return result.data[0]["id"]
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail="Failed to create session")

    async def get_session(self, session_id: str) -> dict:
        """Get session data from Supabase"""
        try:
            result = self.supabase.table("interview_sessions").select("*").eq("id", session_id).execute()
            if not result.data:
                raise HTTPException(status_code=404, detail="Session not found")
            return result.data[0]
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            logger.error(f"Error getting session: {e}")
            raise HTTPException(status_code=500, detail="Failed to get session")

    async def update_session(self, session_id: str, updates: dict):
        """Update session data in Supabase"""
        try:
            updates["updated_at"] = datetime.utcnow().isoformat()
            self.supabase.table("interview_sessions").update(updates).eq("id", session_id).execute()
        except Exception as e:
            logger.error(f"Error updating session: {e}")
            raise HTTPException(status_code=500, detail="Failed to update session")

    async def save_question(self, question_data: dict):
        """Save question to database"""
        try:
            question = self.supabase.table("interview_questions").insert(question_data).execute()
            return question
        except Exception as e:
            logger.error(f"Error saving question: {e}")

    async def save_answer(self, answer_data: dict):
        """Save user answer and feedback to database"""
        try:
            self.supabase.table("interview_answers").insert(answer_data).execute()
        except Exception as e:
            logger.error(f"Error saving answer: {e}")

    async def get_session_questions(self, session_id: str) -> List[dict]:
        """Get all questions for a session"""
        try:
            result = self.supabase.table("interview_questions").select("*").eq("session_id", session_id).order("question_order").execute()
            return result.data
        except Exception as e:
            logger.error(f"Error getting questions: {e}")
            return []

    async def get_session_answers(self, session_id: str) -> List[dict]:
        """Get all answers for a session"""
        try:
            result = self.supabase.table("interview_answers").select("*").eq("session_id", session_id).order("created_at").execute()
            return result.data
        except Exception as e:
            logger.error(f"Error getting answers: {e}")
            return []

    async def get_user_sessions(self, user_id: str, limit: int = 10) -> List[dict]:
        """Get user's recent sessions"""
        try:
            result = (self.supabase.table("interview_sessions")
                     .select("*")
                     .eq("user_id", user_id)
                     .order("created_at", desc=True)
                     .limit(limit)
                     .execute())
            return result.data
        except Exception as e:
            logger.error(f"Error getting user sessions: {e}")
            return []

    async def get_all_user_sessions(self, user_id: str) -> List[dict]:
        """Get all of a user's sessions"""
        try:
            result = (self.supabase.table("interview_sessions")
                     .select("*")
                     .eq("user_id", user_id)
                     .order("created_at", desc=True)
                     .execute())
            return result.data
        except Exception as e:
            logger.error(f"Error getting all user sessions: {e}")
            return []

    async def save_company_research(self, session_id: str, research_data: dict):
        """Save company research data"""
        try:
            research_data["session_id"] = session_id
            research_data["created_at"] = datetime.utcnow().isoformat()
            self.supabase.table("company_research").insert(research_data).execute()
        except Exception as e:
            logger.error(f"Error saving company research: {e}")

db = DatabaseManager()

# Perplexity Sonar Pro Client
class SonarProClient:
    def __init__(self):
        self.api_key = PERPLEXITY_API_KEY
        self.base_url = "https://api.perplexity.ai"
        
    async def search_and_analyze(self, query: str, model: str = "llama-3.1-sonar-large-128k-online", return_citations: bool = True) -> Dict:
        """Enhanced Sonar Pro search with real-time web data"""
        try:
            # Use OpenAI's async client for Perplexity endpoint
            client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )

            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert interview coach with access to real-time industry data. Provide detailed, current, and actionable insights with proper citations."
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                max_tokens=2000,
                temperature=0.7,
                extra_body={
                    "return_citations": return_citations,
                    "search_domain_filter": [
                        "linkedin.com", "glassdoor.com", "stackoverflow.com",
                        "github.com", "medium.com", "dev.to"
                    ]
                },
                timeout=45.0
            )
            # OpenAI's response is an object, convert to dict for compatibility
            return response.model_dump() if hasattr(response, "model_dump") else response
        except OpenAIError as e:
            logger.error(f"Sonar API error: {e}")
            raise HTTPException(status_code=500, detail=f"Sonar API error: {e}")
        except Exception as e:
            logger.error(f"Sonar API unexpected error: {e}")
            raise HTTPException(status_code=500, detail="AI service error")

sonar_client = SonarProClient()

# Database Initialization
async def initialize_database():
    """Initialize database tables using Supabase SQL"""
    # This would typically be done via Supabase dashboard or migrations
    # Here's the SQL structure for reference:
    sql_schemas = """
    -- Interview Sessions Table
    CREATE TABLE IF NOT EXISTS interview_sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id VARCHAR,
        domain VARCHAR NOT NULL,
        experience_level VARCHAR NOT NULL,
        interview_type VARCHAR NOT NULL,
        duration_minutes INTEGER DEFAULT 30,
        company_name VARCHAR,
        job_title VARCHAR,
        status VARCHAR DEFAULT 'active',
        current_question_index INTEGER DEFAULT 0,
        total_questions INTEGER DEFAULT 0,
        average_score DECIMAL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        completed_at TIMESTAMP WITH TIME ZONE
    );

    -- Interview Questions Table
    CREATE TABLE IF NOT EXISTS interview_questions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id UUID REFERENCES interview_sessions(id) ON DELETE CASCADE,
        question_order INTEGER NOT NULL,
        question TEXT NOT NULL,
        difficulty VARCHAR DEFAULT 'medium',
        context TEXT,
        hints JSONB DEFAULT '[]',
        citations JSONB DEFAULT '[]',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Interview Answers Table
    CREATE TABLE IF NOT EXISTS interview_answers (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id UUID REFERENCES interview_sessions(id) ON DELETE CASCADE,
        question_id UUID REFERENCES interview_questions(id) ON DELETE CASCADE,
        question_order INTEGER NOT NULL,
        user_answer TEXT NOT NULL,
        response_time_seconds INTEGER,
        feedback_score DECIMAL,
        feedback_strengths JSONB DEFAULT '[]',
        feedback_improvements JSONB DEFAULT '[]',
        industry_insights JSONB DEFAULT '[]',
        follow_up_question TEXT,
        citations JSONB DEFAULT '[]',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Company Research Table
    CREATE TABLE IF NOT EXISTS company_research (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        session_id UUID REFERENCES interview_sessions(id) ON DELETE CASCADE,
        company_name VARCHAR NOT NULL,
        research_data JSONB NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );
    -- Auth Users Table
    CREATE TABLE IF NOT EXISTS auth_users (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        email VARCHAR NOT NULL UNIQUE,
        password_hash VARCHAR NOT NULL,
        full_name VARCHAR,
        role VARCHAR DEFAULT 'candidate',
        preferences JSONB DEFAULT '{}',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        last_login TIMESTAMP WITH TIME ZONE
    );

    -- Auth Sessions Table
    CREATE TABLE IF NOT EXISTS auth_sessions (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
        user_id UUID REFERENCES auth_users(id) ON DELETE CASCADE,
        access_token VARCHAR NOT NULL,
        refresh_token VARCHAR NOT NULL,
        expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
    );

    -- Create indexes for better performance
    CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON interview_sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON interview_sessions(created_at);
    CREATE INDEX IF NOT EXISTS idx_questions_session_id ON interview_questions(session_id);
    CREATE INDEX IF NOT EXISTS idx_answers_session_id ON interview_answers(session_id);
    CREATE INDEX IF NOT EXISTS idx_auth_users_email ON auth_users(email);
    CREATE INDEX IF NOT EXISTS idx_auth_sessions_user_id ON auth_sessions(user_id);
    CREATE INDEX IF NOT EXISTS idx_auth_sessions_access_token ON auth_sessions(access_token);
    """
    
    logger.info("Database schema initialized (run via Supabase dashboard)")


# ---------------------- Sonar Integration ----------------------
async def ask_sonar(system_prompt: str, user_prompt: str) -> str:
    messages = [
        {
            "role": "system", 
            "content": system_prompt
        },
        {
            "role": "user", 
            "content": user_prompt
        }
    ]
    client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(
        model="sonar-pro",
        messages=messages,
    )
    print(response)
    return response
    

def extract_weak_areas_from_feedback(answers: List[dict]) -> List[str]:
    all_issues = []
    for ans in answers:
        improvements = ans.get("feedback_improvements", [])
        for issue in improvements:
            tokens = [w.strip().lower() for w in re.split(r'[.,;:/()\n]', issue) if len(w.strip()) > 2]
            all_issues.extend(tokens)
    counts = Counter(all_issues)
    common = [word for word, freq in counts.most_common(5)]
    return common


async def get_current_user(authorization: str = Header(None)) -> dict:
    """Get current user from access token"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    try:
        token = authorization.split(" ")[1]
        user = supabase.auth.get_user(token)
        
        if not user:
            raise HTTPException(status_code=401, detail="Invalid token")
            
        return user
    except Exception as e:
        logger.error(f"Token validation error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")
    
# API Endpoints
@app.get("/healthcheck")
async def health_check():
    return {"status": "healthy", "service": "Smart Interview Companion API v2.0", "database": "Supabase"}

# ------------------------- 🔐 Auth Routes ----------------------------
@app.post("/auth/signup")
async def signup(req: AuthRequest):
    """Register a new user with email and password"""
    try:
        # Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", req.email):
            raise HTTPException(status_code=400, detail="Invalid email format")

        # Check if user already exists
        existing_user = supabase.table("auth_users").select("id").eq("email", req.email).execute()
        if existing_user.data:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create password hash
        password_hash = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

        # Create auth user
        auth_response = supabase.auth.sign_up({
            "email": req.email,
            "password": req.password,
            "options": {
                "data": {
                    "full_name": req.full_name,
                    "role": req.role or "candidate"
                }
            }
        })

        if not auth_response.user:
            raise HTTPException(status_code=500, detail="Failed to create user account")

        # Create user profile in profiles table
        profile_data = {
            "id": auth_response.user.id,
            "email": req.email,
            "password_hash": password_hash,
            "full_name": req.full_name,
            "role": req.role or "candidate",
            "created_at": datetime.utcnow().isoformat(),
            "last_login": datetime.utcnow().isoformat(),
            "preferences": {
                "notification_enabled": True,
                "email_notifications": True
            }
        }

        profile_result = supabase.table("auth_users").insert(profile_data).execute()
        
        if not profile_result.data:
            # Rollback auth user creation if profile creation fails
            supabase.auth.admin.delete_user(auth_response.user.id)
            raise HTTPException(status_code=500, detail="Failed to create user profile")

        # Get user preferences
        preferences = profile_data.get("preferences", {})

        return {
            "message": "User registered successfully",
            "user": {
                "id": auth_response.user.id,
                "email": auth_response.user.email,
                "full_name": req.full_name,
                "role": req.role or "candidate",
                "preferences": preferences
            },
            "session": {
                "access_token": auth_response.session.access_token,
                "refresh_token": auth_response.session.refresh_token,
                "expires_at": auth_response.session.expires_at
            },
            "requires_email_confirmation": auth_response.user.confirmed_at is None
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during signup")

@app.post("/auth/login")
async def login(req: LoginRequest):
    """Authenticate user and create session"""
    try:
        # Validate email format
        if not re.match(r"[^@]+@[^@]+\.[^@]+", req.email):
            raise HTTPException(status_code=400, detail="Invalid email format")

        # Attempt login
        auth_response = supabase.auth.sign_in_with_password({
            "email": req.email,
            "password": req.password
        })

        if not auth_response.user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Get user profile
        profile = supabase.table("auth_users").select("*").eq("id", auth_response.user.id).execute()
        
        if not profile.data:
            raise HTTPException(status_code=404, detail="User profile not found")

        # Update last login timestamp
        supabase.table("auth_users").update({
            "last_login": datetime.utcnow().isoformat()
        }).eq("id", auth_response.user.id).execute()

        # Get user preferences
        preferences = profile.data[0].get("preferences", {})

        return {
            "message": "Login successful",
            "user": {
                "id": auth_response.user.id,
                "email": auth_response.user.email,
                "full_name": profile.data[0].get("full_name"),
                "role": profile.data[0].get("role"),
                "preferences": preferences
            },
            "session": {
                "access_token": auth_response.session.access_token,
                "refresh_token": auth_response.session.refresh_token,
                "expires_at": auth_response.session.expires_at
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during login")


@app.post("/auth/refresh")
async def refresh_token(payload: RefreshTokenRequest):
    """Refresh access token using refresh token"""
    try:
        auth_response = supabase.auth.refresh_session(payload.refresh_token)
        
        return {
            "access_token": auth_response.session.access_token,
            "refresh_token": auth_response.session.refresh_token,
            "expires_at": auth_response.session.expires_at
        }
    except Exception as e:
        logger.error(f"Token refresh error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid refresh token")

@app.post("/auth/logout")
async def logout(current_user: dict = Depends(get_current_user)):
    """Logout user and invalidate session"""
    try:
        supabase.auth.sign_out()
        return {"message": "Logged out successfully"}
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to logout")

@app.post("/auth/change-password")
async def change_password(req: ChangePasswordRequest):
    """Change user password after validating the old password."""
    try:
        # Fetch user profile
        user_profile = supabase.table("auth_users").select("id, password_hash, email").eq("id", req.user_id).execute()
        if not user_profile.data:
            raise HTTPException(status_code=404, detail="User not found")
        user = user_profile.data[0]
        # Validate old password
        if not bcrypt.checkpw(req.old_password.encode("utf-8"), user["password_hash"].encode("utf-8")):
            raise HTTPException(status_code=401, detail="Old password is incorrect")
        # Hash new password
        new_password_hash = bcrypt.hashpw(req.new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        # Update password in auth_users table
        supabase.table("auth_users").update({"password_hash": new_password_hash}).eq("id", req.user_id).execute()
        # Optionally, update password in Supabase Auth
        try:
            supabase.auth.admin.update_user_by_id(req.user_id, {"password": req.new_password})
        except Exception as e:
            # Log but don't fail if Supabase Auth update fails
            logger.error(f"Supabase Auth password update failed: {e}")
        return {"message": "Password changed successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Change password error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error during password change")

# ------------------------- 🎯 Core Interview APIs ----------------------------

def parse_first_question(sonar_response: Dict) -> Dict:
    """Parse the first question response from Sonar Pro into introduction and main question"""
    content = sonar_response.choices[0].message.content
    citations = sonar_response.citations

    # The prompt is multi-line. Parse from Introduction till Question, then after everything is question.
    introduction = ""
    main_question = ""
    in_intro = False
    in_question = False
    intro_lines = []
    question_lines = []

    for line in content.splitlines():
        if '[INTRODUCTION]' in line:
            in_intro = True
            in_question = False
            # If there's text after the marker, add it
            after = line.split('[INTRODUCTION]', 1)[1].strip()
            if after:
                intro_lines.append(after)
            continue
        if '[QUESTION]' in line:
            in_intro = False
            in_question = True
            # If there's text after the marker, add it
            after = line.split('[QUESTION]', 1)[1].strip()
            if after:
                question_lines.append(after)
            continue
        if in_intro:
            intro_lines.append(line)
        elif in_question:
            question_lines.append(line)

    introduction = "\n".join(intro_lines).strip()
    main_question = "\n".join(question_lines).strip()

    # Clean up the question text
    question_text = re.sub(r'^#+\s*', '', main_question)
    question_text = re.sub(r'^[Qq]uestion:\s*', '', question_text)

    return {
        "introduction": introduction,
        "main_question": question_text.strip(),
        "citations": citations[:3] if citations else []
    }

@app.post("/api/sessions/start")
async def start_interview_session(request: SessionRequest, background_tasks: BackgroundTasks):
    """Start a new interview session with Supabase persistence"""
    # --- Job Description Support ---
    if request.job_description:
        # If job description is provided, only use mode, duration, job_title, and job_description
        # For config, only use mode and duration
        config = InterviewConfig(
            domain=None,
            difficulty=request.experience_level,
            duration=request.duration_minutes,
            session_type=request.interview_type,
            mode=request.mode,
            location=request.location,
            job_title=request.job_title,
            job_description=request.job_description
        )
        system_prompt = generate_prompt(config, PromptType.INTIAL_QUESTION_JD.value)
        personalized_prompt = "Please begin the interview with a challenging but fair opening question based on the job description."
    else:
        # Generate prompt with company focus (config-based)
        config = InterviewConfig(
            domain=request.domain,
            difficulty=request.experience_level,
            duration=request.duration_minutes,
            session_type=request.interview_type,
            mode=request.mode,
            location=request.location,
            job_title=None,
            job_description=None
        )
        # Retrieve previous feedback
        past = supabase.table("interview_sessions").select("id").eq("user_id", request.user_id).order("created_at", desc=True).limit(3).execute()
        user_sessions = past.data if past else []
        feedback_answers = []
        for s in user_sessions:
            a = supabase.table("interview_answers").select("feedback_improvements").eq("session_id", s['id']).execute()
            feedback_answers.extend(a.data if a else [])
        common_weaknesses = extract_weak_areas_from_feedback(feedback_answers)
        config.known_weak_areas = common_weaknesses
        system_prompt = generate_prompt(config, PromptType.INTIAL_QUESTION.value)
        personalized_prompt = "Please begin the interview with a challenging but fair opening question."

    sonar_response = await ask_sonar(system_prompt, personalized_prompt)
    
    # Parse the first question response
    first_question_data = parse_first_question(sonar_response)

    # Save session data
    session_data = {
        "user_id": request.user_id,
        "mode": request.mode,
        "duration_minutes": request.duration_minutes,
        "status": "active",
        "created_at": datetime.utcnow().isoformat(),
        "current_question_index": 0,
        "total_questions": 5,
        "job_title": request.job_title or None,
        "job_description": request.job_description or None
    }
    if not request.job_description:
        session_data.update({
            "domain": request.domain,
            "experience_level": request.experience_level,
            "interview_type": request.interview_type,
            "company_name": request.company_name or None,
            "location": request.location or None
        })
    
    interview_session = supabase.table("interview_sessions").insert(session_data).execute()

    # Save the first question in the interview_questions table
    question_data = {
        "session_id": interview_session.data[0]['id'],
        "question_order": 1,
        "question": first_question_data["main_question"],
        "difficulty": request.experience_level if not request.job_description else None,
        "hints": [],  # No hints for the first question by default
        "citations": first_question_data.get("citations", []),
        "context": None,
    }
    interview_question = supabase.table("interview_questions").insert(question_data).execute()

    return {
        "session_id": interview_session.data[0]['id'],
        "question_id": interview_question.data[0]['id'],
        "introduction": first_question_data["introduction"],
        "question": first_question_data["main_question"],
        "citations": first_question_data["citations"]
    }

@app.get("/api/sessions/{session_id}/question")
async def get_current_question(session_id: str):
    """Get the current question for the session"""
    session = await db.get_session(session_id)
    questions = await db.get_session_questions(session_id)
    
    current_index = session.get("current_question_index", 0)
    
    if current_index >= len(questions):
        raise HTTPException(status_code=400, detail="No more questions available")
    
    current_q = questions[current_index]
    
    return QuestionResponse(
        question_id=current_index,
        question=current_q["question"],
        difficulty=current_q.get("difficulty", "medium"),
        context=current_q.get("context"),
        hints=current_q.get("hints", []),
        citations=current_q.get("citations", [])
    )

@app.post("/api/sessions/{session_id}/answer")
async def submit_answer(session_id: str, answer_request: AnswerRequest):
    """Submit answer and get AI-powered feedback using Sonar Pro"""
    session = await db.get_session(session_id)
    questions = await db.get_session_questions(session_id)
    
    # Find the question with the matching UUID
    current_question = next((q for q in questions if str(q["id"]) == str(answer_request.question_id)), None)
    if not current_question:
        raise HTTPException(status_code=400, detail="Invalid question ID")
    
    # Determine max questions based on session duration
    duration = session.get("duration_minutes", 30)
    if duration <= 20: # short
        max_questions = 3
    elif duration <= 45: # medium
        max_questions = 5
    else: # long
        max_questions = 10
        
    # Check if max questions limit reached
    current_index = session.get("current_question_index", 0)
    
    # Generate comprehensive feedback using Sonar Pro
    feedback_prompt_template = """
    As an expert interview coach, analyze this {domain} interview answer for a {experience_level} role:
    
    Question: {question}
    Candidate Answer: {answer}
    
    Provide structured feedback with:
    1. Score (1-10) with justification
    2. 3 specific strengths demonstrated
    3. 3 concrete areas for improvement
    4. Current industry insights and best practices (2024-2025)
    {follow_up_section}
    
    Use real-time industry data, current hiring trends, and best practices.
    Be constructive and actionable in your feedback.
    """
    
    if current_index >= max_questions -1:
        follow_up_section = ""
    else:
        follow_up_section = "5. A natural, relevant follow-up question"
        
    feedback_query = feedback_prompt_template.format(
        domain=session['domain'],
        experience_level=session['experience_level'],
        question=current_question['question'],
        answer=answer_request.answer,
        follow_up_section=follow_up_section
    )
    
    try:
        sonar_response = await sonar_client.search_and_analyze(feedback_query)
        feedback = parse_feedback_response(sonar_response)
        
        # Save answer and feedback to database
        answer_data = {
            "session_id": session_id,
            "question_id": current_question["id"],
            "question_order": current_question["question_order"],
            "user_answer": answer_request.answer,
            "response_time_seconds": answer_request.response_time_seconds,
            "feedback_score": feedback.score,
            "feedback_strengths": feedback.strengths,
            "feedback_improvements": feedback.improvements,
            "industry_insights": feedback.industry_insights,
            "follow_up_question": feedback.follow_up_question,
            "citations": feedback.citations or [],
            "created_at": datetime.utcnow().isoformat()
        }
        
        await db.save_answer(answer_data)
        
        # If there's a follow-up, save it as the next question
        if feedback.follow_up_question and current_index < max_questions - 1:
            new_question_order = current_question["question_order"] + 1
            question_data = {
                "session_id": session_id,
                "question_order": new_question_order,
                "question": feedback.follow_up_question,
                "difficulty": "medium",  # Default difficulty
                "context": f"Follow-up to: {current_question['question']}",
                "hints": [],
                "citations": []
            }
            next_question = await db.save_question(question_data)
            answer_data["status"] = "active"
            answer_data["question_id"] = next_question.data[0]['id']
            answer_data["question_order"] = new_question_order
        else:
            # End of interview if no follow-up or max questions reached
            await db.update_session(session_id, {"status": "completed"})
            answer_data["status"] = "completed"
            
        # Move to next question
        new_index = session.get("current_question_index", 0) + 1
        await db.update_session(session_id, {"current_question_index": new_index})
        
        return answer_data
        
    except Exception as e:
        logger.error(f"Failed to generate feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate feedback")

@app.get("/api/sessions/{session_id}/next")
async def get_next_question(session_id: str):
    """
    Get the next question in the interview session.
    The model should be aware of the existing session and previous answers to maintain context.
    """
    # Fetch session details
    session = await db.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get all questions and answers for this session
    questions = await db.get_session_questions(session_id)
    answers = await db.get_session_answers(session_id)

    # Determine the next question index
    current_index = session.get("current_question_index", 0)
    total_questions = len(questions)

    if current_index >= total_questions:
        return {
            "message": "Interview session completed",
            "next_question": None,
            "completed": True
        }

    # Get the next question
    next_question = questions[current_index]

    # Optionally, you can include previous answers for model context
    previous_answers = [
        {
            "question": q["question"],
            "answer": a["user_answer"]
        }
        for q, a in zip(questions[:current_index], answers[:current_index])
        if a.get("user_answer")
    ]

    return {
        "session_id": session_id,
        "question_order": next_question["question_order"],
        "question_id": next_question["id"],
        "question": next_question["question"],
        "difficulty": next_question.get("difficulty"),
        "context": next_question.get("context"),
        "hints": next_question.get("hints", []),
        "previous_answers": previous_answers,
        "completed": False
    }

@app.get("/api/sessions/{session_id}/transcript")
async def get_session_transcript(session_id: str):
    """Get complete session transcript with analytics from Supabase"""
    session = await db.get_session(session_id)
    questions = await db.get_session_questions(session_id)
    answers = await db.get_session_answers(session_id)
    
    # Calculate performance metrics
    scores = [ans.get("feedback_score", 0) for ans in answers if ans.get("feedback_score")]
    avg_score = sum(scores) / len(scores) if scores else 0
    total_time = sum(ans.get("response_time_seconds", 0) for ans in answers)
    
    # Combine questions and answers
    qa_pairs = []
    for i, question in enumerate(questions):
        answer = next((ans for ans in answers if ans["question_order"] == i), None)
        qa_pairs.append({
            "question_order": i,
            "question": question["question"],
            "difficulty": question.get("difficulty"),
            "user_answer": answer.get("user_answer") if answer else None,
            "feedback_score": answer.get("feedback_score") if answer else None,
            "feedback_strengths": answer.get("feedback_strengths", []) if answer else [],
            "feedback_improvements": answer.get("feedback_improvements", []) if answer else [],
            "response_time_seconds": answer.get("response_time_seconds") if answer else None
        })
    
    return {
        "session_id": session_id,
        "domain": session["domain"],
        "experience_level": session["experience_level"],
        "interview_type": session["interview_type"],
        "company_name": session.get("company_name"),
        "questions_answered": len(answers),
        "total_questions": len(questions),
        "average_score": round(avg_score, 1),
        "total_time_minutes": round(total_time / 60, 1),
        "start_time": session["created_at"],
        "status": session["status"],
        "qa_pairs": qa_pairs
    }

@app.get("/api/sessions/{session_id}/company-insights")
async def get_company_insights(session_id: str):
    """Get company research data from Supabase"""
    try:
        result = supabase.table("company_research").select("*").eq("session_id", session_id).execute()
        if not result.data:
            raise HTTPException(status_code=404, detail="No company research available")
        
        return result.data[0]["research_data"]
    except Exception as e:
        logger.error(f"Error getting company insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to get company insights")

@app.get("/api/sessions")
async def get_sessions_by_user(user_id: str):
    """Get user's sessions, separating incomplete and complete ones."""
    sessions = await db.get_all_user_sessions(user_id)
    
    incomplete_session = None
    completed_sessions = []
    
    for session in sessions:
        # Get answer count and average score
        answers = await db.get_session_answers(session["id"])
        scores = [ans.get("feedback_score", 0) for ans in answers if ans.get("feedback_score")]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        summary = SessionSummary(
            session_id=session["id"],
            domain=session["domain"],
            experience_level=session["experience_level"],
            interview_type=session["interview_type"],
            company_name=session.get("company_name"),
            questions_answered=len(answers),
            average_score=round(avg_score, 1),
            duration_minutes=session.get("duration_minutes", 0),
            created_at=datetime.fromisoformat(session["created_at"]),
            status=session["status"]
        )
        
        if session["status"] != "completed" and incomplete_session is None:
            incomplete_session = summary
        elif session["status"] == "completed":
            completed_sessions.append(summary)
            
    return {"incomplete_session": incomplete_session, "completed_sessions": completed_sessions}

@app.get("/api/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Get interview session details with questions and answers."""
    session = await db.get_session(session_id)
    questions = await db.get_session_questions(session_id)
    answers = await db.get_session_answers(session_id)

    answers_by_q_order = {ans["question_order"]: ans for ans in answers}

    question_answers_list = []
    for q in questions:
        answer_data = answers_by_q_order.get(q["question_order"])
        
        answer_obj = None
        if answer_data:
            answer_obj = {
                "answer_id": answer_data["id"],
                "user_answer": answer_data["user_answer"],
                "response_time_seconds": answer_data["response_time_seconds"],
                "feedback_score": answer_data["feedback_score"],
                "feedback_strengths": answer_data["feedback_strengths"],
                "feedback_improvements": answer_data["feedback_improvements"],
                "industry_insights": answer_data["industry_insights"],
                "citations": answer_data.get("citations", []),
                "created_at": answer_data["created_at"]
            }

        question_obj = {
            "question_id": q["id"],
            "question_order": q["question_order"],
            "question": q["question"],
            "difficulty": q.get("difficulty"),
            "hints": q.get("hints", []),
            "citations": q.get("citations", []),
            "answer": answer_obj
        }
        
        question_answers_list.append({"question": question_obj})

    return {
        "session_id": session["id"],
        "user_id": session["user_id"],
        "experience_level": session["experience_level"],
        "interview_type": session["interview_type"],
        "mode": session["mode"],
        "company_name": session.get("company_name"),
        "job_title": session.get("job_title"),
        "duration_minutes": session["duration_minutes"],
        "status": session["status"],
        "created_at": session["created_at"],
        "current_question_index": session["current_question_index"],
        "total_questions": len(questions),
        "question_answers": question_answers_list
    }

@app.get("/api/users/{user_id}/sessions")
async def get_user_sessions(user_id: str, limit: int = 10):
    """Get user's session history"""
    sessions = await db.get_user_sessions(user_id, limit)
    
    session_summaries = []
    for session in sessions:
        # Get answer count and average score
        answers = await db.get_session_answers(session["id"])
        scores = [ans.get("feedback_score", 0) for ans in answers if ans.get("feedback_score")]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        session_summaries.append(SessionSummary(
            session_id=session["id"],
            domain=session["domain"],
            experience_level=session["experience_level"],
            interview_type=session["interview_type"],
            company_name=session.get("company_name"),
            questions_answered=len(answers),
            average_score=round(avg_score, 1),
            duration_minutes=session.get("duration_minutes", 0),
            created_at=datetime.fromisoformat(session["created_at"]),
            status=session["status"]
        ))
    
    return {"sessions": session_summaries}

@app.post("/api/sessions/{session_id}/end")
async def end_session(session_id: str):
    """End interview session and generate final report"""
    answers = await db.get_session_answers(session_id)
    
    # Calculate final metrics
    scores = [ans.get("feedback_score", 0) for ans in answers if ans.get("feedback_score")]
    avg_score = sum(scores) / len(scores) if scores else 0
    
    try:
        # Update session as completed
        await db.update_session(session_id, {
            "status": "completed",
            "completed_at": datetime.utcnow().isoformat(),
            "average_score": avg_score
        })
        
        return {
            "session_id": session_id,
            "status": "completed",
            "performance_summary": {
                "questions_answered": len(answers),
                "average_score": round(avg_score, 1),
                "total_sessions": len(answers)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate final report: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate final report")

@app.get("/api/sessions/{session_id}/industry-trends")
async def industry_trends(session_id: str):
    """Get industry trends and insights for the session's domain"""
    try:
        session = await db.get_session(session_id)
        
        # Generate industry trends using Sonar Pro
        trends_query = f"""
        Provide current industry trends and insights for {session['domain']} roles at {session['experience_level']} level:
        
        1. Top 5 in-demand technical skills (2024-2025)
        2. Emerging technologies and frameworks
        3. Industry best practices and methodologies
        4. Salary trends and market demand
        5. Career growth opportunities
        
        Focus on actionable insights and real-time market data.
        """
        
        sonar_response = await sonar_client.search_and_analyze(trends_query)
        trends_data = parse_trends_from_response(sonar_response)
        
        return {
            "domain": session["domain"],
            "experience_level": session["experience_level"],
            "trends": trends_data,
            "last_updated": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to generate industry trends: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate industry trends")

@app.get("/api/sessions/{session_id}/transcript")
async def get_transcript(session_id: str):
    """Get complete session transcript with questions, answers, and scores"""
    try:
        session = await db.get_session(session_id)
        questions = await db.get_session_questions(session_id)
        answers = await db.get_session_answers(session_id)
        
        # Calculate performance metrics
        scores = [ans.get("feedback_score", 0) for ans in answers if ans.get("feedback_score")]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Combine questions and answers chronologically
        qa_pairs = []
        for i, question in enumerate(questions):
            answer = next((ans for ans in answers if ans["question_order"] == i), None)
            qa_pairs.append({
                "question_order": i,
                "question": question["question"],
                "difficulty": question.get("difficulty", "medium"),
                "user_answer": answer.get("user_answer") if answer else None,
                "feedback_score": answer.get("feedback_score") if answer else None,
                "feedback_strengths": answer.get("feedback_strengths", []) if answer else [],
                "feedback_improvements": answer.get("feedback_improvements", []) if answer else [],
                "response_time_seconds": answer.get("response_time_seconds") if answer else None
            })
        
        return {
            "session_id": session_id,
            "domain": session["domain"],
            "experience_level": session["experience_level"],
            "interview_type": session["interview_type"],
            "company_name": session.get("company_name"),
            "start_time": session["created_at"],
            "status": session["status"],
            "performance_metrics": {
                "questions_answered": len(answers),
                "total_questions": len(questions),
                "average_score": round(avg_score, 1),
                "completion_percentage": round((len(answers) / len(questions)) * 100 if questions else 0, 1)
            },
            "qa_pairs": qa_pairs
        }
        
    except Exception as e:
        logger.error(f"Failed to get transcript: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session transcript")

# Weekly improvement digest
@app.get("/api/users/{user_id}/weekly-report")
async def weekly_report(user_id: str):
    one_week_ago = datetime.utcnow() - timedelta(days=7)
    sessions = await supabase.table("sessions").select("id, created_at").eq("user_id", user_id).gte("created_at", one_week_ago.isoformat()).execute()
    session_ids = [s['id'] for s in sessions.data]
    all_feedback = []
    for sid in session_ids:
        a = await supabase.table("session_answers").select("feedback_improvements").eq("session_id", sid).execute()
        all_feedback.extend(a.data)

    weaknesses = extract_weak_areas_from_feedback(all_feedback)
    return {
        "user_id": user_id,
        "sessions_count": len(session_ids),
        "common_weaknesses": weaknesses,
        "suggestions": [f"Focus on strengthening '{w}' in upcoming sessions." for w in weaknesses]
    }

# Background Tasks
async def research_company_context(session_id: str, company_name: str, job_title: Optional[str]):
    """Background task to research company using Sonar Pro and save to Supabase"""
    research_query = f"""
    Conduct comprehensive research on {company_name} for interview preparation:
    
    1. Company Overview & Recent Developments (2024-2025)
    2. Financial performance and market position
    3. Company culture, values, and work environment
    4. Technology stack and engineering practices
    5. Leadership team and organizational structure
    6. Recent news, product launches, or strategic initiatives
    7. Interview process and common questions
    8. Employee reviews and workplace insights
    {"9. Specific role insights for " + job_title if job_title else ""}
    
    Provide actionable talking points and intelligent questions to ask during the interview.
    Include salary ranges and growth opportunities if available.
    """
    
    try:
        sonar_response = await sonar_client.search_and_analyze(research_query)
        research_data = parse_company_research(sonar_response)
        
        # Save to Supabase
        await db.save_company_research(session_id, {
            "company_name": company_name,
            "research_data": research_data
        })
        
        logger.info(f"Company research completed for {company_name}")
        
    except Exception as e:
        logger.error(f"Company research failed for {company_name}: {e}")

# Helper Functions
async def parse_and_save_questions(sonar_response: Dict, session_id: str) -> List[Dict]:
    """Parse questions from Sonar Pro response and save to Supabase"""
    content = sonar_response.get("choices", [{}])[0].get("message", {}).get("content", "")
    citations = sonar_response.get("citations", [])
    
    questions = []
    lines = content.split('\n')
    
    current_question = {}
    question_order = 0
    
    for line in lines:
        line = line.strip()
        if line and any(line.startswith(f'{i}.') for i in range(1, 11)):
            if current_question:
                # Save previous question
                question_data = {
                    "session_id": session_id,
                    "question_order": question_order,
                    "question": current_question.get("question", ""),
                    "difficulty": current_question.get("difficulty", "medium"),
                    "context": current_question.get("context", ""),
                    "hints": current_question.get("hints", []),
                    "citations": citations[:3] if citations else []  # Limit citations
                }
                await db.save_question(question_data)
                questions.append(current_question)
                question_order += 1
            
            # Start new question
            current_question = {
                "question": line,
                "difficulty": extract_difficulty(line),
                "context": "",
                "hints": []
            }
    
    # Save last question
    if current_question:
        question_data = {
            "session_id": session_id,
            "question_order": question_order,
            "question": current_question.get("question", ""),
            "difficulty": current_question.get("difficulty", "medium"),
            "context": current_question.get("context", ""),
            "hints": current_question.get("hints", []),
            "citations": citations[:3] if citations else []
        }
        await db.save_question(question_data)
        questions.append(current_question)
    
    return questions[:10]  # Limit to 10 questions

def extract_difficulty(question_text: str) -> str:
    """Extract difficulty level from question text"""
    question_lower = question_text.lower()
    if any(word in question_lower for word in ['basic', 'simple', 'introduction', 'explain']):
        return 'easy'
    elif any(word in question_lower for word in ['complex', 'advanced', 'design', 'architect', 'scale']):
        return 'hard'
    else:
        return 'medium'

def parse_feedback_response(sonar_response: Dict) -> FeedbackResponse:
    """Parse feedback from Sonar Pro response into structured FeedbackResponse"""
    content = sonar_response.get("choices", [{}])[0].get("message", {}).get("content", "")
    citations = sonar_response.get("citations", [])
    
    # Initialize default values
    score = 7.0
    strengths = []
    improvements = []
    industry_insights = []
    follow_up_question = None
    
    # Split content into lines and parse by sections
    lines = content.split('\n')
    current_section = None
    current_content = []
    
    # Define main headers to look for
    main_headers = ['Score', 'Strengths', 'Areas for Improvement', 'Current Industry Insights', 'Industry Insights', 'Follow-Up Question']
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check if this line contains any of the main headers
        is_header_line = any(header in line for header in main_headers)
        
        if is_header_line:
            # Process previous section content
            if current_section and current_content:
                if current_section == 'strengths':
                    strengths = [item.strip() for item in current_content if item.strip() and len(item.strip()) > 10]
                elif current_section == 'improvements':
                    improvements = [item.strip() for item in current_content if item.strip() and len(item.strip()) > 10]
                elif current_section == 'industry_insights':
                    industry_insights = [item.strip() for item in current_content if item.strip() and len(item.strip()) > 10]
                elif current_section == 'follow_up':
                    follow_up_question = ' '.join(current_content).strip()
            
            # Start new section
            current_content = []
            
            # Determine section type
            if 'Score:' in line:
                # Extract score
                score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', line)
                if score_match:
                    score = float(score_match.group(1))
                    if score > 10:
                        score = score / 10
            elif 'Strengths' in line:
                current_section = 'strengths'
            elif 'Improvement' in line:
                current_section = 'improvements'
            elif 'Current Industry Insights' in line or 'Industry Insights' in line:
                current_section = 'industry_insights'
            elif 'Follow-Up Question' in line:
                current_section = 'follow_up'
            else:
                current_section = None
            
            # Iterate through lines until next header is found
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                if not next_line:
                    j += 1
                    continue
                
                # Check if next line is a header
                if any(header in next_line for header in main_headers):
                    break
                
                # Add content to current section
                if current_section and next_line:
                    # Skip sub-headers (###)
                    if not next_line.startswith('###'):
                        # Remove bullet points and numbering
                        clean_line = re.sub(r'^\d+\.\s*', '', next_line)
                        clean_line = re.sub(r'^\*\s*', '', clean_line)
                        clean_line = re.sub(r'^-\s*', '', clean_line)
                        clean_line = re.sub(r'^\*\*\s*', '', clean_line)
                        clean_line = re.sub(r'^\*\*\*\s*', '', clean_line)
                        
                        if clean_line and len(clean_line) > 5:  # Filter out very short lines
                            current_content.append(clean_line)
                
                j += 1
            
            # Skip the lines we've already processed
            i = j
        else:
            # Add content to current section for non-header lines
            if current_section and line:
                # Skip sub-headers (###)
                if not line.startswith('###'):
                    # Remove bullet points and numbering
                    clean_line = re.sub(r'^\d+\.\s*', '', line)
                    clean_line = re.sub(r'^\*\s*', '', clean_line)
                    clean_line = re.sub(r'^-\s*', '', clean_line)
                    clean_line = re.sub(r'^\*\*\s*', '', clean_line)
                    clean_line = re.sub(r'^\*\*\*\s*', '', clean_line)
                    
                    if clean_line and len(clean_line) > 5:  # Filter out very short lines
                        current_content.append(clean_line)
            i += 1
    
    # Process the last section
    if current_section and current_content:
        if current_section == 'strengths':
            strengths = [item.strip() for item in current_content if item.strip() and len(item.strip()) > 10]
        elif current_section == 'improvements':
            improvements = [item.strip() for item in current_content if item.strip() and len(item.strip()) > 10]
        elif current_section == 'industry_insights':
            industry_insights = [item.strip() for item in current_content if item.strip() and len(item.strip()) > 10]
        elif current_section == 'follow_up':
            follow_up_question = ' '.join(current_content).strip()
    
    # Clean up and limit the number of items
    strengths = strengths[:3]  # Limit to 3 strengths
    improvements = improvements[:3]  # Limit to 3 improvements
    industry_insights = industry_insights[:3]  # Limit to 3 insights
    
    return FeedbackResponse(
        score=score,
        strengths=strengths,
        improvements=improvements,
        industry_insights=industry_insights,
        follow_up_question=follow_up_question,
        citations=citations[:3] if citations else []
    )

def parse_trends_from_response(sonar_response: Dict) -> Dict:
    """Parse industry trends from Sonar Pro response"""
    content = sonar_response.get("choices", [{}])[0].get("message", {}).get("content", "")
    citations = sonar_response.get("citations", [])
    
    # Extract structured trends (implement proper parsing based on response format)
    lines = content.split('\n')
    
    trends = {
        "technical_skills": [],
        "emerging_technologies": [],
        "best_practices": [],
        "salary_trends": [],
        "career_opportunities": [],
        "citations": citations[:3] if citations else []
    }
    
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if "technical skill" in line.lower():
            current_section = "technical_skills"
        elif "emerging" in line.lower() or "technology" in line.lower():
            current_section = "emerging_technologies"
        elif "best practice" in line.lower() or "methodology" in line.lower():
            current_section = "best_practices"
        elif "salary" in line.lower() or "market" in line.lower():
            current_section = "salary_trends"
        elif "career" in line.lower() or "opportunity" in line.lower():
            current_section = "career_opportunities"
        elif current_section and line.startswith("- "):
            trends[current_section].append(line[2:])
    
    return trends