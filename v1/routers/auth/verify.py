from fastapi import APIRouter
from typing import Optional
from pydantic import BaseModel
import os
from supabase import create_client
from dotenv import load_dotenv
from supabase.lib.client_options import ClientOptions

# Load environment variables
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")

# Create Supabase client
supabase = create_client(
    SUPABASE_URL,
    SUPABASE_SERVICE_ROLE_KEY,
    options=ClientOptions(
        auto_refresh_token=False,
        persist_session=False,
    )
)

router = APIRouter()

class AuthRequest(BaseModel):
    token: str
    
class AuthResponse(BaseModel):
    is_valid: bool
    user_id: Optional[str] = None
    email: Optional[str] = None
    user_metadata: Optional[dict] = None
    message: str

@router.post("/verify", response_model=AuthResponse)
async def verify_token(request: AuthRequest):
    """
    Verify a JWT token provided by Supabase client.
    
    This endpoint:
    1. Takes a JWT token in the request body
    2. Verifies the token using Supabase
    3. Returns user information if the token is valid
    """
    try:
        # Get the JWT token from request
        token = request.token
        
        # Verify the token with Supabase's auth.getUser() 
        # This will throw an error if the token is invalid
        user = supabase.auth.get_user(token)
        
        # If we reach here, the token is valid
        return AuthResponse(
            is_valid=True,
            user_id=user.user.id,
            email=user.user.email,
            user_metadata=user.user.user_metadata,
            message="Token is valid"
        )
        
    except Exception as e:
        # Return failed authentication but don't expose details
        return AuthResponse(
            is_valid=False,
            message="Invalid or expired token"
        )