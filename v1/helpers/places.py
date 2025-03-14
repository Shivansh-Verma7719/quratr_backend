import os
import pandas as pd
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import AsyncClient, acreate_client

load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

# Create supabase client asynchronously
async def create_supabase() -> AsyncClient:
    return await acreate_client(url, key)

# Define data models
class PlaceBase(BaseModel):
    name: str
    tags: Optional[str] = None
    rating: Optional[float] = None
    address: Optional[str] = None
    locality: Optional[str] = None
    city_name: Optional[str] = None
    group_experience: Optional[str] = None
    description: Optional[str] = None
    image: Optional[str] = None
    price: Optional[str] = None
    reservation: Optional[bool] = None
    likes: Optional[int] = None
    dislikes: Optional[int] = None
    created_at: Optional[str] = None

class Place(PlaceBase):
    place_id: int

class RecommendationResponse(BaseModel):
    recommendations: str

# Load the place data asynchronously from Supabase
async def load_place_data():
    try:
        # Get async client
        supabase = await create_supabase()
        
        # Execute query asynchronously only Delhi locations
        response = await supabase.table("places").select(
            "id, created_at, name, image, address, tags, rating, group_experience, locality, city_name, price, reservation, description, likes, dislikes"
        ).eq("city_name", "Delhi NCR").execute()
        
        # Convert to DataFrame for easier manipulation
        data = response.data
        df = pd.DataFrame(data)
        
        # Replace NaN values with None/appropriate defaults to avoid JSON serialization issues
        # For numeric columns
        numeric_cols = ['rating', 'likes', 'dislikes']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # For string columns
        string_cols = ['name', 'tags', 'address', 'locality', 'city_name', 
                      'group_experience', 'description', 'image', 'price', 'created_at']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
        
        # For boolean columns
        if 'reservation' in df.columns:
            df['reservation'] = df['reservation'].fillna(False)
        
        print(f"Loaded {len(df)} places from Supabase")
        return df
    except Exception as e:
        print(f"Error loading place data from Supabase: {e}")
        # Initialize with empty DataFrame if fetching fails
        return pd.DataFrame(columns=["id", "created_at", "name", "image", "address", "tags", 
                                  "rating", "group_experience", "locality", "city_name", 
                                  "price", "reservation", "description", "likes", "dislikes"])

def format_place_data(df):
    places = []
    for _, row in df.iterrows():
        location = f"{row.get('locality', '')}, {row.get('city_name', '')}".strip(", ")
        place_info = f"""
        Name: {row.get('name', '')}
        Tags: {row.get('tags', '')}
        Rating: {row.get('rating', '')}
        Address: {row.get('address', '')}
        Location: {location}
        Price: {row.get('price', '')}
        Reservation Required: {row.get('reservation', False)}
        Group Experience: {row.get('group_experience', '')}
        Description: {row.get('description', '')}
        Image: {row.get('image', '')}
        Likes: {row.get('likes', 0)}
        Dislikes: {row.get('dislikes', 0)}
        """
        places.append(place_info)
    return "\n".join(places)

def filter_places(df, mood: str, vibe: str, location: str):
    try:
        # Split location into potential locality and city_name components
        location_terms = location.lower().split()
        
        # Filter based on tags containing mood and vibe
        filtered_df = df[df['tags'].str.contains(mood, case=False, na=False) & 
                        df['tags'].str.contains(vibe, case=False, na=False)]
        
        # Further filter by location (check both locality and city_name)
        location_matches = pd.Series(False, index=filtered_df.index)
        for term in location_terms:
            locality_match = filtered_df['locality'].str.contains(term, case=False, na=False)
            city_match = filtered_df['city_name'].str.contains(term, case=False, na=False)
            location_matches = location_matches | locality_match | city_match
        
        final_results = filtered_df[location_matches]
        
        places = []
        for i, row in final_results.iterrows():
            location = f"{row.get('locality', '')}, {row.get('city_name', '')}".strip(", ")
            place = Place(
                place_id=row.get('id', i),
                name=row.get('name', ''),
                tags=row.get('tags', ''),
                rating=row.get('rating', 0),
                address=row.get('address', ''),
                locality=row.get('locality', ''),
                city_name=row.get('city_name', ''),
                price=row.get('price', ''),
                reservation=row.get('reservation', False),
                group_experience=row.get('group_experience', ''),
                description=row.get('description', ''),
                image=row.get('image', ''),
                likes=row.get('likes', 0),
                dislikes=row.get('dislikes', 0),
                created_at=row.get('created_at', '')
            )
            places.append(place)
        return places
    except Exception as e:
        print(f"Error filtering places: {e}")
        return []

def format_places_for_response(df):
    places = []
    for i, row in df.iterrows():
        location = f"{row.get('locality', '')}, {row.get('city_name', '')}".strip(", ")
        place = {
            "place_id": row.get('id', i),
            "name": row.get('name', ''),
            "tags": row.get('tags', ''),
            "rating": row.get('rating', 0),
            "address": row.get('address', ''),
            "locality": row.get('locality', ''),
            "city_name": row.get('city_name', ''),
            "location": location,
            "price": row.get('price', ''),
            "reservation": row.get('reservation', False),
            "description": row.get('description', ''),
            "image": row.get('image', ''),
            "likes": row.get('likes', 0),
            "dislikes": row.get('dislikes', 0),
            "created_at": row.get('created_at', '')
        }
        places.append(place)
    return places