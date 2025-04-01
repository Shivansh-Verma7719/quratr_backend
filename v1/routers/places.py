from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import json
from dotenv import load_dotenv
from supabase import create_client
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI

# Import helpers from v1/helpers
from ..helpers.chains import (
    create_query_understanding_chain,
    create_response_chain,
    format_places_for_llm
)
from ..helpers.helpers import (
    multi_query_retrieval,
    get_place_details,
    QueryIntent,
)

router = APIRouter()

# Load environment variables
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY") 

# Create Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize OpenAI client

# openrouter.ai deepseek-chat-v3-0324 model - free version
# def get_llm():
#     return ChatOpenAI(
#         model="deepseek/deepseek-chat-v3-0324:free",
#         base_url="https://openrouter.ai/api/v1",
#         temperature=0.3,
#         api_key=OPENROUTER_API_KEY
#     )

# 4o-mini model - default
def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=OPENAI_API_KEY,
        temperature=0.4,
    )

# o3-mini model
# def get_llm():
#     return ChatOpenAI(
#         model="o3-mini",
#         api_key=OPENAI_API_KEY,
#     )

# Google Gemini model 2.0 flash
# def get_llm():
#     return ChatGoogleGenerativeAI(
#         model="gemini-2.0-flash",
#         temperature=0.4,
#         api_key=GOOGLE_API_KEY
#     )

# Google Gemini model - pro version
# def get_llm():
#     return ChatGoogleGenerativeAI(
#         model="gemini-2.5-pro-exp-03-25",
#         temperature=0.4,
#         api_key=GOOGLE_API_KEY
#     )        

# Define response models
class PlaceRanking(BaseModel):
    similarity_score: float

# Add this to your existing models
class SearchRequest(BaseModel):
    query: str
    limit: int = 30
    threshold: float = 0.45
    user_attributes: List[int] = [0, 0, 0, 0, 0]
    
    class Config:
        schema_extra = {
            "example": {
                "query": "Suggest some good coffee places in South Delhi",
                "limit": 30,
                "threshold": 0.45,
                "user_attributes": [1, 1, 0, 0, 0]
            }
        }

class PlaceResponse(BaseModel):
    id: int
    name: str
    address: Optional[str] = None
    city: Optional[str] = None
    cuisine: Optional[str] = None
    tags: Optional[str] = None
    rating: Optional[float] = None
    price: Optional[float] = None
    description: Optional[str] = None
    image: Optional[str] = None
    ranking: Optional[PlaceRanking] = None

class RecommendationItem(BaseModel):
    id: int
    name: str
    description: str
    match_reasons: List[str]
    highlights: List[str]
    cuisine: Optional[str] = None
    price_range: Optional[str] = None
    location: Optional[str] = None
    atmosphere: Optional[str] = None
    image_url: Optional[str] = None

class SearchResponse(BaseModel):
    query: str
    intent: Dict[str, Any]
    places: List[PlaceResponse]
    recommendations: List[RecommendationItem]
    summary: str

@router.post("/search", response_model=SearchResponse)
async def search_places(request: SearchRequest):
    """
    Search for places using natural language processing with personalization.
    
    This endpoint:
    1. Analyzes the query intent
    2. Performs vector search using multiple query variations  
    3. Retrieves place details
    4. Generates a detailed personalized response
    """
    try:
        llm = get_llm()
        
        # Process user attributes from request
        user_profile = {}
        try:
            attribute_values = request.user_attributes
            if len(attribute_values) == 5:
                attribute_names = [
                    "Nightlife Enthusiast", 
                    "Luxury-Seeking", 
                    "Solitary", 
                    "Adventurous", 
                    "Social"
                ]
                user_profile = {
                    name: bool(value) 
                    for name, value in zip(attribute_names, attribute_values)
                }
        except Exception as e:
            # Handle invalid input gracefully
            print(f"Invalid user_attributes format: {request.user_attributes}")
            # Continue with empty user profile
        
        # Step 1: Understand the query
        query_chain = create_query_understanding_chain(llm)
        intent_dict = query_chain({"query": request.query})
        print(f"Intent Dictionary: {intent_dict}")  # Debugging line
        
        # Add user attributes to intent dictionary
        intent_dict["user_attributes"] = user_profile
        
        # Convert to QueryIntent object
        intent = QueryIntent(
            original_query=intent_dict.get("original_query", request.query),
            cuisine_types=intent_dict.get("cuisine_types", []),
            locations=intent_dict.get("locations", []),
            price_range=intent_dict.get("price_range"),
            atmosphere=intent_dict.get("atmosphere"),
            occasion=intent_dict.get("occasion"),
            dietary_preferences=intent_dict.get("dietary_preferences", []),
            expanded_queries=intent_dict.get("expanded_queries", []),
            user_attributes=user_profile
        )
        
        # Step 2: Multi-Query Retrieval
        results = multi_query_retrieval(intent, supabase, request.threshold, request.limit)
        if not results:
            raise HTTPException(status_code=404, detail="No matching places found")
            
        # Step 3: Get Place Details
        place_details = get_place_details(supabase, results)
        if not place_details:
            raise HTTPException(status_code=404, detail="Could not retrieve place details")
            
        # Step 4: Format Places for Response (using vector similarity only)
        formatted_places = format_places_for_llm(place_details, results)
        
        # Step 5: Generate Enhanced Response
        response_chain = create_response_chain(llm)
        places_json = json.dumps(formatted_places, indent=2)
        
        # Format user preferences for the prompt
        user_preferences = []
        for attr, value in intent.user_attributes.items():
            if value:
                user_preferences.append(attr)
        
        user_input = f"""
User Query: "{request.query}"

Based on our analysis, the user is looking for:
- Cuisine Types: {', '.join(intent.cuisine_types) if intent.cuisine_types else 'Any'}
- Locations: {', '.join(intent.locations) if intent.locations else 'Any'}
- Price Range: {intent.price_range if intent.price_range else 'Any'}
- Atmosphere: {intent.atmosphere if intent.atmosphere else 'Not specified'}
- Occasion: {intent.occasion if intent.occasion else 'Not specified'}
- Dietary Preferences: {', '.join(intent.dietary_preferences) if intent.dietary_preferences else 'None'}

User Profile: {', '.join(user_preferences) if user_preferences else 'No specific preferences'}

Places data:
{places_json}

Provide personalized recommendations based on this data. Consider both the user's query and their profile preferences.
Focus only on generating id, name, description, match_reasons, highlights, and atmosphere for each place.
"""
        # Get the JSON response
        response_data = response_chain({"user_input": user_input})
        
        # Create a place lookup dictionary to easily find detailed data
        place_lookup = {place["id"]: place for place in formatted_places}
        
        # Merge AI-generated content with existing place data
        enriched_recommendations = []
        
        for rec in response_data.get("recommendations", []):
            place_id = rec.get("id")
            
            # Get existing place data if available
            place_data = place_lookup.get(place_id, {})
            
            # Create enriched recommendation with both AI-generated and existing data
            enriched_rec = {
                "id": place_id,
                "name": rec.get("name", place_data.get("name", "")),
                "description": rec.get("description", ""),
                "match_reasons": rec.get("match_reasons", []),
                "highlights": rec.get("highlights", []),
                "atmosphere": rec.get("atmosphere", ""),
                "cuisine": place_data.get("cuisine", ""),
                "price_range": f"₹{int(place_data.get('price'))}" if place_data.get('price') else "",
                "location": place_data.get("address", ""),
                "image_url": place_data.get("image", "")
            }
            
            enriched_recommendations.append(enriched_rec)
        
        # Prepare the final API response with structured data
        return SearchResponse(
            query=request.query,
            intent=intent_dict,
            places=formatted_places,
            recommendations=enriched_recommendations,
            summary=response_data.get("summary", "")
        )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}\n{error_details}")

# Optional: Add an endpoint for just retrieving place details
@router.get("/places/{place_id}", response_model=PlaceResponse)
async def get_place(place_id: int):
    """Get details for a specific place by ID"""
    try:
        response = supabase.table("placesv2").select("*").eq("id", place_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Place with ID {place_id} not found")
            
        place = response.data[0]
        
        # Format the response
        cuisine = ", ".join(place.get("cuisine", [])) if place.get("cuisine") else ""
        tags = ", ".join(place.get("tags", [])) if place.get("tags") else ""
        
        return PlaceResponse(
            id=place.get("id"),
            name=place.get("name", ""),
            address=place.get("address", ""),
            city=place.get("city", ""),
            cuisine=cuisine,
            tags=tags,
            rating=place.get("rating"),
            price=place.get("price"),
            description=place.get("description", ""),
            image=place.get("image", "")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving place: {str(e)}")