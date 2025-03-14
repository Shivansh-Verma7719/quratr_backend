from fastapi import APIRouter, Query, HTTPException, Depends
from dotenv import load_dotenv
from typing import Optional

# LangChain imports
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

from ..helpers.places import (
    RecommendationResponse, 
    load_place_data, 
    format_place_data,
    format_places_for_response
)

load_dotenv()
router = APIRouter()

# Create dependency for getting place data
async def get_place_data():
    # Load data asynchronously
    return await load_place_data()

def get_llm(provider: str = "openai", model_name: Optional[str] = None):
    """Get LLM based on provider and model name"""
    if provider.lower() == "openai":
        model = model_name or "gpt-4-turbo"
        return ChatOpenAI(model=model, temperature=0.7)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

# Original endpoint that returns all places
@router.get("/places/")
async def read_places(df=Depends(get_place_data)):
    return format_places_for_response(df)

@router.get("/places/recommend", response_model=RecommendationResponse)
async def recommend_places(
    mood: str = Query(..., description="User's current mood (e.g., Relaxed, Excited)"),
    vibe: str = Query(..., description="Desired place vibe (e.g., Cozy, Nightlife)"),
    location: str = Query(..., description="Target location (e.g., Delhi)"),
    # provider: str = Query("openai", description="LLM provider (openai, anthropic)"),
    # model: Optional[str] = Query(None, description="Specific model to use"),
    df=Depends(get_place_data)
):
    """Get place recommendations based on mood, vibe and location using LangChain"""
    try:
        # Format the place data for the LLM
        place_info = format_place_data(df)
        
        # Create system and human message templates
        system_template = """
        You are an AI travel assistant that suggests the best places based on user mood and preferences.
        
        Here is the place database to use for your recommendations:
        
        {place_data}
        
        Important guidelines:
        - Only use the data available in the provided database
        - Do not fabricate any details not present in the database
        - Analyze the descriptions as well for finding the perfect matches.
        - Prioritize highly-rated places
        - Consider mood, vibe, and location when making recommendations
        - If there is insufficient information for a recommendation, clearly indicate that no matching results were found
        """
        
        human_template = """
        I'm looking for place recommendations with the following criteria:
        - Mood: {mood}
        - Vibe: {vibe}
        - Location: {location}
        
        Please provide at least 10 recommendations that match these criteria.
        
        For each recommendation, include:
        1. The name of the place
        2. Its verified address
        3. A relevant image or image URL
        4. Key features or tags that align with my mood and vibe
        """
        
        # Create messages from templates
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
        
        # Get the LLM model based on provider
        try:
            llm = get_llm("openai", "gpt-4o-mini")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Build the chain
        chain = (
            {"place_data": RunnablePassthrough(), 
             "mood": lambda _: mood, 
             "vibe": lambda _: vibe, 
             "location": lambda _: location}
            | prompt 
            | llm 
            | StrOutputParser()
        )
        
        # Execute the chain
        recommendation_text = chain.invoke(place_info)
        
        return {"recommendations": recommendation_text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")