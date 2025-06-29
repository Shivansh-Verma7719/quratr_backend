from typing import Dict, Any, List
import os
import lameenc
from datetime import datetime
from fastapi import HTTPException
from google.genai import types
from supabase import create_client
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# Create Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def pcm_to_mp3_bytes(pcm_data: bytes, sample_rate: int = 24000, channels: int = 1, bitrate: int = 192) -> bytes:
    """Helper function to convert PCM data directly to MP3 bytes using pure Python"""
    try:
        
        # Drop an incomplete trailing byte if any
        if len(pcm_data) % 2:
            pcm_data = pcm_data[:-1]
        
        # Initialize LAME encoder
        encoder = lameenc.Encoder()
        encoder.set_bit_rate(bitrate)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(channels)
        encoder.set_quality(2)  # 2 is high quality, 7 is fast
                
        # Encode PCM bytes directly to MP3
        mp3_data = encoder.encode(pcm_data)
        mp3_data += encoder.flush()

        return mp3_data
        
    except Exception as e:
        print(f"Error in MP3 conversion: {str(e)}")
        print(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to convert PCM to MP3: {str(e)}")

def upload_mp3_to_supabase(mp3_data, filename: str = None) -> str:
    """Upload MP3 data to Supabase storage and return the public URL"""
    try:
        # Convert to bytes if it's a bytearray
        if isinstance(mp3_data, bytearray):
            mp3_data = bytes(mp3_data)
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"podcast_{timestamp}.mp3"
        
        # Ensure filename has .mp3 extension
        if not filename.endswith('.mp3'):
            filename += '.mp3'
        
        # Upload file to Supabase storage in /episodes folder
        file_path = f"episodes/{filename}"
        
        # Upload the file to Supabase storage using bytes directly
        response = supabase.storage.from_("podcast").upload(
            path=file_path,
            file=mp3_data,
            file_options={
                "content-type": "audio/mpeg",
                "cache-control": "3600",
                "upsert": "true"  # Allow overwriting if file exists
            }
        )
        
        # Get the public URL for the uploaded file
        public_url = supabase.storage.from_("podcast").get_public_url(file_path)
        
        return public_url
        
    except Exception as e:
        print(f"Error uploading MP3 to Supabase storage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to upload MP3 to storage: {str(e)}")

def get_audio_prompt(script: str, speaker_characters: list[Dict[str, Any]]) -> tuple[str, list]:
    """Generate TTS prompt and voice configurations from speaker characters and script"""
    
    # Create voice configurations for TTS
    voice_configs = []
    speaker_info_lines = []
    
    for character in speaker_characters:
        # Create voice config
        voice_config = types.SpeakerVoiceConfig(
            speaker=character['name'],
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=character['voice_name'],
                )
            )
        )
        voice_configs.append(voice_config)
        
        # Create speaker info for the prompt
        speaker_info_lines.append(
            f"        - {character['name']}: {character['description']} (Tone: {character.get('tone', 'Natural')}, Pitch: {character.get('pitch', 'Middle')})"
        )
    
    # Create the TTS prompt
    speaker_info = "\n".join(speaker_info_lines)

    prompt = f"""You are an expert podcast speech generator. Make sure each speaker sounds natural and engaging, with appropriate pauses and emphasis. Keep the podcast length 4 minutes only.

Speaker Characteristics:
{speaker_info}

Generate natural speech that reflects each speaker's personality, tone, and style as described above. Make sure to:
- Match each speaker's described tone and personality
- Use appropriate pacing and emphasis for their character
- Maintain consistent voice characteristics throughout

TTS the following podcast conversation:

{script}"""
    
    return prompt, voice_configs

async def get_speaker_characters_from_db(speaker_ids: List[str]) -> List[Dict[str, Any]]:
    """Fetch speaker character information from Supabase blog.characters table based on speaker IDs"""
    try:
        print(f"Fetching speaker characters for IDs: {speaker_ids}")
        
        # Query the blog.characters table
        response = supabase.schema("blog").table("characters").select("*").in_("id", speaker_ids).execute()
        
        characters = []
        found_ids = set()
        
        # Process found characters
        for char_data in response.data:
            character = {
                "id": str(char_data["id"]),
                "name": char_data["name"],
                "voice_name": char_data["voice_name"], 
                "description": char_data["description"],
                "tone": char_data.get("tone", "Professional"),
                "pitch": char_data.get("pitch", "Middle Pitch")
            }
            characters.append(character)
            found_ids.add(str(char_data["id"]))
        
        # Sort characters to match the order of speaker_ids
        sorted_characters = []
        for speaker_id in speaker_ids:
            for char in characters:
                if char["id"] == speaker_id:
                    sorted_characters.append(char)
                    break
        
        return sorted_characters
        
    except Exception as e:
        print(f"Error fetching speaker characters from database: {str(e)}")
        # Throw 500 error if database query fails
        raise HTTPException(status_code=500, detail=f"Failed to fetch speaker characters from database: {str(e)}")

def get_script_generation_prompt(speaker_characters: list[Dict[str, Any]]) -> str:
    """Generate the script generation prompt incorporating speaker characteristics"""
    
    # Create speaker descriptions for the script prompt
    speaker_descriptions = []
    speaker_labels = []

    for _, character in enumerate(speaker_characters):
        speaker_labels.append(character['name'])
        speaker_descriptions.append(
            f"- {character['name']}: {character['description']} (Tone: {character.get('tone', 'Natural')}, Pitch: {character.get('pitch', 'Middle')})"
        )
    
    speaker_info = "\n".join(speaker_descriptions)
    speaker_format_examples = "\n".join([f"{name}: [dialogue]" for name in speaker_labels])
    
    prompt = f"""You are an expert podcast script writer. Convert the given blog content into an engaging podcast script with {{num_speakers}} speakers.

Speaker Personalities and Characteristics:
{speaker_info}

Guidelines:
- Create natural, conversational dialogue between speakers that reflects their unique personalities and expertise
- Make each speaker's dialogue authentic to their described character, tone, and background
- Include smooth transitions between topics
- Add appropriate pauses and emphasis markers
- Keep the tone {{podcast_style}} while maintaining each speaker's individual voice
- Aim for 4 minutes speaking time in podcast length
- Include natural interjections, questions, and responses that fit each speaker's personality
- Make sure all speakers contribute meaningfully based on their expertise and character
- Use each speaker's knowledge areas and personality traits to create authentic dialogue

Format the output as a clean script with clear speaker labels:
{speaker_format_examples}

Blog Content: {{blog_content}}"""

    return prompt
