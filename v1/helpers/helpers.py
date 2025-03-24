import os
import re
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI  # Changed from Google import
from pydantic import BaseModel, Field

# Load environment variables if needed
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

def multi_query_retrieval(intent: QueryIntent, supabase, threshold: float, limit: int) -> List[SearchResult]:
    """
    Retrieve results using multiple query variations.
    """
    all_results = {}
    
    # Ensure expanded_queries is a list, not None
    expanded_queries = intent.expanded_queries if intent.expanded_queries else []
    
    # Combine original query with expanded queries
    queries_to_try = [intent.original_query] + expanded_queries
    
    print(f"Retrieving results using {len(queries_to_try)} query variations...")
    
    for query_text in queries_to_try:
        # Skip empty queries
        if not query_text or not query_text.strip():
            continue
            
        print(f"Processing query variation: '{query_text}'")
        query_embedding = generate_query_embedding(query_text)
        results = vector_search(supabase, query_embedding, threshold, limit)
        print(f"  → Found {len(results)} results")
        
        # Process results...
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

def format_response_as_markdown(response_text):
    """
    Format the response as clean markdown, ensuring proper formatting.
    
    Args:
        response_text: The response from the LLM (can be string or dict)
        
    Returns:
        Properly formatted markdown response
    """
    # Handle dict responses (common with newer LLM output formats)
    if isinstance(response_text, dict):
        # Try common keys for the content
        if 'text' in response_text:
            content = response_text['text']
        elif 'content' in response_text:
            content = response_text['content']
        elif 'answer' in response_text:
            content = response_text['answer']
        else:
            # If we can't find a standard text field, convert the whole dict to string
            content = str(response_text)
    else:
        # If it's already a string, use it directly
        content = str(response_text)
    
    # Clean up markdown formatting and handle literal newlines
    clean_text = content.replace("```markdown", "").replace("```", "")
    # Replace literal \n with actual newlines
    clean_text = clean_text.replace("\\n", "\n")
    
    # Make the formatting cleaner and more consistent
    lines = clean_text.split('\n')
    formatted_lines = []
    
    inside_list = False
    for line in lines:
        line = line.strip()
        if not line:
            # Keep empty lines
            formatted_lines.append("")
            continue
            
        # Handle numbered lists
        if re.match(r'^\d+\.\s', line):
            inside_list = True
            formatted_lines.append(line)
            
        # Handle bold headings (often used instead of proper headings)
        elif line.startswith("**") and line.endswith("**") and "**  " in line:
            heading_text = line.replace("**", "").strip()
            formatted_lines.append(f"### {heading_text}")
            
        # Handle proper headings
        elif line.startswith("#"):
            inside_list = False
            # Don't change existing ### headings
            if line.startswith("### "):
                formatted_lines.append(line)
            # Convert other headings to standardized ### format
            else:
                text = re.sub(r'^#+\s*', '', line)
                if text:  # Only add if there's actual content
                    formatted_lines.append(f"### {text}")
                    
        # Handle image links
        elif line.startswith("!["):
            inside_list = False
            formatted_lines.append(line)
            
        # Normal text
        else:
            formatted_lines.append(line)
    
    # Join lines back together
    markdown = "\n\n".join([line for line in formatted_lines if line is not None])
    
    # Fix image formatting
    markdown = re.sub(r'\n{2,}(!\[)', '\n\n![', markdown)
    
    return markdown

def json_to_markdown(json_data):
    """
    Convert a structured JSON recommendation response to markdown format.
    """
    # Get recommendations and summary
    recommendations = json_data.get("recommendations", [])
    summary = json_data.get("summary", "")
    
    if not recommendations:
        return "No recommendations found."
    
    # Start with the summary
    markdown = f"{summary}\n\n"
    
    # Add each recommendation
    for i, place in enumerate(recommendations):
        # Place name as heading
        markdown += f"### {place.get('name')}\n"
        
        # Add image if available
        if place.get('image_url'):
            markdown += f"![{place.get('name')}]({place.get('image_url')})\n\n"
        
        # Add description
        markdown += f"{place.get('description', '')}\n\n"
        
        # Add key details as bullet points
        markdown += "**Key Details:**\n"
        if place.get('cuisine'):
            markdown += f"- **Cuisine:** {place.get('cuisine')}\n"
        if place.get('location'):
            markdown += f"- **Location:** {place.get('location')}\n"
        if place.get('price_range'):
            markdown += f"- **Price Range:** {place.get('price_range')}\n"
        if place.get('atmosphere'):
            markdown += f"- **Atmosphere:** {place.get('atmosphere')}\n"
        markdown += "\n"
        
        # Add why this matches the user's query
        if place.get('match_reasons'):
            markdown += "**Why This Matches Your Search:**\n"
            for reason in place.get('match_reasons'):
                markdown += f"- {reason}\n"
            markdown += "\n"
        
        # Add highlights
        if place.get('highlights'):
            markdown += "**Highlights:**\n"
            for highlight in place.get('highlights'):
                markdown += f"- {highlight}\n"
            markdown += "\n"
        
        # Add separator between places except for the last one
        if i < len(recommendations) - 1:
            markdown += "---\n\n"
    
    return markdown