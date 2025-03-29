from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import os
import json
import re
from dotenv import load_dotenv
from supabase import create_client
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# Import helpers from v1/helpers
from ..helpers.chains import (
    create_query_understanding_chain,
    create_response_chain,
    format_places_for_llm
)
from ..helpers.helpers import (
    json_to_markdown,
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
    markdown_response: Optional[str] = None

@router.get("/search", response_model=SearchResponse)
async def search_places(
    query: str = Query(..., description="Natural language query to search for places"),
    limit: int = Query(15, description="Maximum number of results to return"),
    threshold: float = Query(0.45, description="Similarity threshold (0-1)"),
    format: str = Query("json", description="Response format (json, markdown, html, plain)"),
):
    """
    Search for places using natural language processing.
    
    This endpoint:
    1. Analyzes the query intent
    2. Performs vector search using multiple query variations
    3. Retrieves place details
    4. Ranks results based on relevance
    5. Generates a detailed response
    """
    try:
        llm = get_llm()
        
        # Step 1: Understand the query
        query_chain = create_query_understanding_chain(llm)
        intent_dict = query_chain({"query": query})
        print(f"Intent Dictionary: {intent_dict}")  # Debugging line
        
        # Convert to QueryIntent object
        intent = QueryIntent(
            original_query=intent_dict.get("original_query", query),
            cuisine_types=intent_dict.get("cuisine_types", []),
            locations=intent_dict.get("locations", []),
            price_range=intent_dict.get("price_range"),
            atmosphere=intent_dict.get("atmosphere"),
            occasion=intent_dict.get("occasion"),
            dietary_preferences=intent_dict.get("dietary_preferences", []),
            expanded_queries=intent_dict.get("expanded_queries", [])
        )
        
        # Step 2: Multi-Query Retrieval
        results = multi_query_retrieval(intent, supabase, threshold, limit)
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
        user_input = f"""
User Query: "{query}"

Based on our analysis, the user is looking for:
- Cuisine Types: {', '.join(intent.cuisine_types) if intent.cuisine_types else 'Any'}
- Locations: {', '.join(intent.locations) if intent.locations else 'Any'}
- Price Range: {intent.price_range if intent.price_range else 'Any'}
- Atmosphere: {intent.atmosphere if intent.atmosphere else 'Not specified'}
- Occasion: {intent.occasion if intent.occasion else 'Not specified'}
- Dietary Preferences: {', '.join(intent.dietary_preferences) if intent.dietary_preferences else 'None'}

Places data:
{places_json}

Provide personalized recommendations based solely on this data.
        """
        # Get the JSON response
        response_data = response_chain({"user_input": user_input})
        
        # Optional: Convert to markdown if requested
        markdown_response = None
        if format.lower() in ["markdown", "html", "plain"]:
            # Generate markdown from the JSON response
            markdown_response = json_to_markdown(response_data)
            
            # Convert to HTML if requested
            if format.lower() == "html":
                import markdown2
                markdown_response = markdown2.markdown(
                    markdown_response,
                    extras=["tables", "fenced-code-blocks"]
                )
            elif format.lower() == "plain":
                # Strip markdown formatting for plain text
                plain_response = re.sub(r'!\[.*?\]\(.*?\)', '', markdown_response)  # Remove images
                plain_response = re.sub(r'#{1,6}\s*(.*)', r'\1', plain_response)    # Remove headings
                plain_response = re.sub(r'\*\*(.*?)\*\*', r'\1', plain_response)    # Remove bold
                markdown_response = plain_response
        
        # Prepare the API response with structured data
        return SearchResponse(
            query=query,
            intent=intent_dict,
            places=formatted_places,
            recommendations=response_data.get("recommendations", []),
            summary=response_data.get("summary", ""),
            markdown_response=markdown_response
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