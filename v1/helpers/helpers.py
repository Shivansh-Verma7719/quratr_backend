import os
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI  # Changed from Google import
from pydantic import BaseModel, Field

load_dotenv()

class QueryIntent(BaseModel):
    """Structured representation of the user's query intent."""
    original_query: str
    cuisine_types: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    price_range: Optional[str] = None
    atmosphere: Optional[str] = None
    occasion: Optional[str] = None
    dietary_preferences: List[str] = Field(default_factory=list)
    expanded_queries: List[str] = Field(default_factory=list)
    user_attributes: Dict[str, bool] = Field(default_factory=dict)  # Added user attributes

@dataclass
class SearchResult:
    """Individual search result with ranking info."""
    id: int
    name: str
    content: str
    similarity: float
    reranking_score: float = 0.0
    final_score: float = 0.0

def clean_ordering_output(raw_text: str) -> str:
    """
    Remove triple backticks and any "json " prefix (first 5 characters) from the raw LLM output.
    """
    text = raw_text.replace("```", "").strip()
    if text.lower().startswith("json "):
        text = text[5:].strip()
    return text

def extract_name_from_content(content: str) -> str:
    """
    Extract the place name from the content string.
    """
    if not content:
        return "Unknown Place"
    match = re.search(r"Name: ([^,]+)", content)
    return match.group(1).strip() if match else content[:30] + "..."

def get_embedding_client():
    """
    Initialize and return the OpenAI client.
    """
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_query_embedding(query_text: str) -> List[float]:
    """
    Generate embedding for the given text using OpenAI's text-embedding-3-small model.
    """
    client = get_embedding_client()
    response = client.embeddings.create(
        input=query_text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def vector_search(supabase, query_embedding: List[float], threshold: float = 0.5, limit: int = 30) -> List[Dict]:
    """
    Perform vector search in Supabase using the RPC function.
    """
    try:
        # Call the match_places RPC function
        response = supabase.rpc(
            'match_places',
            {
                'query_embedding': query_embedding,
                'match_threshold': threshold,
                'match_count': limit
            }
        ).execute()
        
        # Return the data or empty list
        return response.data if response.data else []
    except Exception as e:
        print(f"Error in vector search: {e}")
        return []

def vector_search_multiple(supabase, query_embeddings: List[List[float]], threshold: float = 0.5, limit: int = 30) -> List[Dict]:
    """
    Perform vector search in Supabase using the optimized RPC function that accepts multiple embeddings.
    """
    try:
        # PostgreSQL expects vectors in the format: '[0.1,0.2,0.3,...]'
        formatted_embeddings = []
        for embedding in query_embeddings:
            # Convert each embedding to a string in PostgreSQL vector format
            vector_str = f"[{','.join(str(val) for val in embedding)}]"
            formatted_embeddings.append(vector_str)
        
        print(f"Sending {len(formatted_embeddings)} embeddings to match_places_multiple...")
        
        response = supabase.rpc(
            'match_places_multiple',
            {
                'query_embeddings': formatted_embeddings,
                'match_threshold': threshold,
                'match_count': limit
            }
        ).execute()
        
        # Return the data or empty list
        return response.data if response.data else []
    except Exception as e:
        print(f"Error in vector search: {e}")
        if formatted_embeddings and len(formatted_embeddings) > 0:
            print(f"First formatted embedding sample (first 30 chars): {formatted_embeddings[0][:30]}...")
        return []

def multi_query_retrieval(intent: QueryIntent, supabase, threshold: float, limit: int) -> List[SearchResult]:
    """
    Retrieve results using multiple query variations in a single optimized database call.
    """
    # Ensure expanded_queries is a list, not None
    expanded_queries = intent.expanded_queries if intent.expanded_queries else []
    
    # Combine original query with expanded queries
    queries_to_try = [intent.original_query] + expanded_queries
    
    # Filter out empty queries
    queries_to_try = [q for q in queries_to_try if q and q.strip()]
    
    print(f"Retrieving results using {len(queries_to_try)} query variations...")
    
    # Generate embeddings for all queries first
    all_embeddings = []
    for query_text in queries_to_try:
        print(f"Processing query variation: '{query_text}'")
        query_embedding = generate_query_embedding(query_text)
        all_embeddings.append(query_embedding)
    
    # Single database call with all embeddings
    results = vector_search_multiple(supabase, all_embeddings, threshold, limit)
    
    print(f"Found {len(results)} total results")
    
    # Process and deduplicate results
    all_results = {}
    for result in results:
        rid = result["id"]
        if rid not in all_results:
            # Add new results to our collection (first time seeing this place)
            all_results[rid] = SearchResult(
                id=rid,
                name=extract_name_from_content(result["content"]),
                content=result["content"],
                similarity=result["similarity"]
            )
        else:
            # If we already have this result, keep only the highest similarity score
            all_results[rid].similarity = max(all_results[rid].similarity, result["similarity"])
    
    # Sort by similarity score and return top results
    results_list = list(all_results.values())
    results_list.sort(key=lambda x: x.similarity, reverse=True)
    return results_list[:limit]

def get_place_details(supabase, results: List[SearchResult]) -> List[Dict]:
    """
    Retrieve detailed information for places from Supabase.
    """
    if not results:
        return []
    place_ids = [result.id for result in results]
    details = supabase.table("placesv2").select("*").in_("id", place_ids).execute()
    return details.data if details.data else []
