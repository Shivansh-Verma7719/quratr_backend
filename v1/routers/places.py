from fastapi import APIRouter, Query, HTTPException, Depends
import os
import openai
from dotenv import load_dotenv

from ..helpers.places import (
    RecommendationResponse, 
    load_place_data, format_place_data,
    format_places_for_response
)
load_dotenv()
router = APIRouter()

# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create dependency for getting place data
async def get_place_data():
    # Load data asynchronously
    return await load_place_data()

# Original endpoint that returns all places
@router.get("/places/")
async def read_places(df=Depends(get_place_data)):
    return format_places_for_response(df)

@router.get("/places/recommend", response_model=RecommendationResponse)
async def recommend_places(
    mood: str = Query(..., description="User's current mood (e.g., Relaxed, Excited)"),
    vibe: str = Query(..., description="Desired place vibe (e.g., Cozy, Nightlife)"),
    location: str = Query(..., description="Target location (e.g., Manhattan)"),
    df=Depends(get_place_data)
):
    """Get place recommendations based on mood, vibe and location"""
    try:
        # Format the place data for the assistant
        place_info = format_place_data(df)

        assistant = openai.beta.assistants.create(
        name="Mood-Based Place Recommender",
        instructions=f"""
        You are an AI travel assistant that suggests the best places based on user mood and preferences.
        Here is the place database:
        {place_info}
        Given a user's mood, vibe, and location, provide recommendations based on the following criteria:
        - Prioritize highly-rated places.
        - Consider mood, tags, and location.
        - Include images and addresses.
        Give at least 10 recommendations for each query, no less.
        If no relevant place is found, suggest exploring the web.
        """,
        model="gpt-4o-mini",
    )
        
        # Using a fixed assistant ID (assumed pre-created with your data context)
        assistant_id = "asst_LijyJeIVBHNEv61HzTe3hwHn"
        
        # Create a thread and generate a recommendation with an enhanced user query
        thread = openai.beta.threads.create()
        
        # Enhanced user query with detailed instructions
        user_query = (
            f"Please provide recommendations based solely on the provided place database. "
            f"I am looking for places that match the following criteria:\n"
            f"- **Mood:** {mood}\n"
            f"- **Vibe:** {vibe}\n"
            f"- **Location:** {location}\n\n"
            f"For each recommendation, include the following details:\n"
            f"1. The name of the place\n"
            f"2. Its verified address\n"
            f"3. A relevant image or image URL\n"
            f"4. Key features or tags from the database that align with the mood and vibe\n\n"
            f"**Important:** Only use the data available in the provided database and do not fabricate any details. "
            f"If there is insufficient information for a recommendation, please clearly indicate that no matching results "
            f"were found and suggest alternative ways to search."
        )
        
        # Add the user's message to the thread
        openai.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_query
        )

        # Run the assistant on the thread
        run = openai.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Wait for the run to complete (in production, use async patterns with timeouts and sleep intervals)
        while True:
            run_status = openai.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status in ["completed", "failed"]:
                break

        if run_status.status == "failed":
            raise HTTPException(status_code=500, detail="Recommendation generation failed")
            
        # Get the assistant's response
        messages = openai.beta.threads.messages.list(thread_id=thread.id)
        if not messages.data:
            raise HTTPException(status_code=500, detail="No recommendation generated")
            
        recommendation_text = messages.data[0].content[0].text.value
        return {"recommendations": recommendation_text, "assistant_id": assistant_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")
