from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseOutputParser
from google import genai
from google.genai import types
from v1.helpers.blog.helpers import (
    pcm_to_mp3_bytes, 
    upload_mp3_to_supabase, 
    get_audio_prompt, 
    get_speaker_characters_from_db,
    get_script_generation_prompt
)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

router = APIRouter()

class BlogToPodcastRequest(BaseModel):
    blog_content: str
    num_speakers: int = 2
    podcast_style: Optional[str] = "conversational"
    speaker_ids: list[str] = ["1", "2"]  # Speaker character IDs from database

    class Config:
        schema_extra = {
            "example": {
                "blog_content": "Artificial Intelligence is transforming the way we work and live...",
                "num_speakers": 2,
                "podcast_style": "conversational",
                "speaker_ids": ["1", "2"]
            }
        }

class PodcastScriptOutputParser(BaseOutputParser):
    """Custom parser to extract podcast script from LLM output"""
    
    def parse(self, text: str) -> str:
        # Clean up the text and return the script
        script = text.strip()
        # Remove any markdown formatting if present
        script = script.replace("```", "").strip()
        return script

def create_podcast_script_chain(speaker_characters):
    """Create a LangChain chain to convert blog content to podcast script with character-aware prompts"""
    
    # Initialize Gemini model for text generation
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=GOOGLE_API_KEY
    )
    
    # Get dynamic prompt based on speaker characters
    system_prompt = get_script_generation_prompt(speaker_characters)
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    # Create the chain
    chain = prompt | llm | PodcastScriptOutputParser()
    
    return chain

async def generate_audio_from_script(script: str, speaker_characters) -> bytes:
    """Generate audio from podcast script using Google GenAI SDK"""
    
    try:
        # Configure the client
        client = genai.Client(api_key=GOOGLE_API_KEY)
        
        # Get TTS prompt and voice configurations
        prompt, voice_configs = get_audio_prompt(script, speaker_characters)
        
        # Generate audio using Google GenAI SDK with dynamic voice configs
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                        speaker_voice_configs=voice_configs
                    )
                )
            )
        )
        
        # Extract audio data from response
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        
        if audio_data:
            mp3_data = pcm_to_mp3_bytes(audio_data, sample_rate=24000, channels=1, bitrate=192)
            # print(f"Audio converted successfully")

            return mp3_data
        else:
            raise HTTPException(status_code=500, detail="No audio data received from TTS service")
        
    except Exception as e:
        print(f"TTS Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Text-to-speech conversion failed: {str(e)}")

@router.post("/blog-to-podcast")
async def convert_blog_to_podcast(request: BlogToPodcastRequest):
    """
    Convert blog content to podcast audio file
    
    Workflow:
    1. Convert blog content to podcast script using Gemini
    2. Generate audio from script using Gemini TTS
    3. Return audio file as streaming response
    """
    
    try:
        # Validate number of speakers
        if request.num_speakers < 2 or request.num_speakers > 3:
            raise HTTPException(status_code=400, detail="Number of speakers must be 2 or 3")
        
        # Get speaker characters from database
        if request.speaker_ids:
            speaker_characters = await get_speaker_characters_from_db(request.speaker_ids[:request.num_speakers])
        else:
            # Throw error if no speaker IDs provided
            raise HTTPException(status_code=400, detail="Speaker IDs must be provided for podcast generation")
                
        # Step 1: Generate podcast script
        print("Generating podcast script...")
        script_chain = create_podcast_script_chain(speaker_characters)
        
        podcast_script = script_chain.invoke({
            "blog_content": request.blog_content,
            "num_speakers": request.num_speakers,
            "podcast_style": request.podcast_style,
        })
                
        # Step 2: Convert script to audio using Gemini TTS
        print("Converting script to audio...")
        audio_data = await generate_audio_from_script(podcast_script, speaker_characters)
        
        print("Audio generation completed")
        
        # Upload MP3 file to Supabase storage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"podcast_{timestamp}.mp3"
        public_url = upload_mp3_to_supabase(audio_data, filename)
                
        # Return JSON response with both download link and storage URL
        return {
            "message": "Podcast generated successfully",
            "storage_url": public_url,
            "filename": filename,
            "speakers": [{"name": char["name"], "description": char["description"]} for char in speaker_characters],
            "script_preview": podcast_script[:200] + "..." if len(podcast_script) > 200 else podcast_script
        }
        
    except Exception as e:
        print(f"Error in blog-to-podcast conversion: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to convert blog to podcast: {str(e)}"
        )

@router.post("/generate-script-only")
async def generate_podcast_script_only(request: BlogToPodcastRequest):
    """
    Generate only the podcast script without audio conversion
    Useful for testing and previewing the script
    """
    
    try:
        # Validate number of speakers
        if request.num_speakers < 2 or request.num_speakers > 3:
            raise HTTPException(status_code=400, detail="Number of speakers must be 2 or 3")
        
        # Get speaker characters from database
        if request.speaker_ids:
            speaker_characters = await get_speaker_characters_from_db(request.speaker_ids[:request.num_speakers])
        else:
            # Use default speaker IDs if none provided
            default_ids = ["1", "2", "3"][:request.num_speakers]
            speaker_characters = await get_speaker_characters_from_db(default_ids)
        
        print("Generating podcast script...")
        script_chain = create_podcast_script_chain(speaker_characters)
        
        podcast_script = script_chain.invoke({
            "blog_content": request.blog_content,
            "num_speakers": request.num_speakers,
            "podcast_style": request.podcast_style,
        })
        
        return {
            "script": podcast_script,
            "speaker_characters": [{"name": char["name"], "description": char["description"]} for char in speaker_characters],
            "message": "Podcast script generated successfully"
        }
        
    except Exception as e:
        print(f"Error generating script: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate podcast script: {str(e)}"
        )