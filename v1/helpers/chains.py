import re
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from .helpers import QueryIntent, clean_ordering_output, SearchResult
from typing import List, Dict
import json

def create_query_understanding_chain(llm: ChatOpenAI):
    """
    Create a chain to extract structured query intent using new LangChain patterns.
    """
    system_message = """
You are a search query analysis expert. Extract structured information from the user's query.
Identify cuisine types, locations, price preferences, atmosphere, occasions, and dietary needs.
Also, generate 2-3 expanded queries that might retrieve better search results.

Return a JSON object with these fields:
- original_query: the original query string
- cuisine_types: list of cuisines mentioned (or empty list)
- locations: list of locations/areas mentioned (or empty list)
- price_range: price indication (budget, mid-range, luxury) or null
- atmosphere: desired atmosphere (romantic, casual, etc.) or null
- occasion: any special occasion mentioned or null
- dietary_preferences: any dietary preferences (vegetarian, vegan, etc.) or empty list
- expanded_queries: 2-3 alternative ways to phrase this query

Format the response ONLY as a valid JSON object.
    """
    
    # Use the modern prompt | llm pattern
    prompt = ChatPromptTemplate.from_messages([
        {"role": "system", "content": system_message},
        {"role": "user", "content": "Please analyze this query: \"{query}\""}
    ])
    
    # Create a function that processes the response and ensures correct JSON parsing
    def parse_json_response(response):
        if hasattr(response, 'content'):
            content = response.content
        else:
            content = str(response)
            
        # Clean up the JSON response
        cleaned_content = content.replace("```json", "").replace("```", "").strip()
        
        try:
            # Parse JSON
            parsed_json = json.loads(cleaned_content)
            return parsed_json
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            # Return a basic fallback
            return {
                "original_query": "",
                "cuisine_types": [],
                "locations": [],
                "expanded_queries": []
            }
    
    # Chain components
    chain = prompt | llm | parse_json_response
    
    # Return a function that takes the query input and invokes the chain
    return lambda inputs: chain.invoke(inputs)

def create_ordering_chain(llm: ChatOpenAI) -> LLMChain:
    """
    Create a chain that outputs a comma-separated list of place IDs in descending order of relevance.
    """
    system_template = """
You are an expert ranking assistant. Given the following query analysis and details for several places, 
determine the order of relevance. Return ONLY a comma-separated list of place IDs in descending order (most relevant first).
Do not include any extra text.
    """
    human_template = "{user_input}"
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
    return LLMChain(llm=llm, prompt=prompt)

def create_response_chain(llm: ChatOpenAI) -> LLMChain:
    """
    Create a chain to generate a detailed, personalized recommendation response in JSON format.
    """
    system_template = """
You are a sophisticated travel and dining assistant that provides personalized recommendations.

IMPORTANT: You must return your response in valid JSON format with the following structure:
{{
  "recommendations": [
    {{
      "id": 123,
      "name": "Place Name",
      "description": "Brief engaging description of the place",
      "match_reasons": ["Why this place matches the query", "Another reason"],
      "highlights": ["Special feature 1", "Special feature 2"],
      "cuisine": "Type of cuisine",
      "price_range": "₹xxx",
      "location": "Area/neighborhood",
      "atmosphere": "Description of atmosphere",
      "image_url": "URL of the image"
    }},
     more places...
  ],
  "summary": "Brief overall summary of the recommendations"
}}

Guidelines:
1. Only include places from the provided data
2. Order places by relevance to the query
3. For each place, explain specifically why it matches the user's query
4. Include 3-5 highlight points for each place
5. The summary should be concise (2-3 sentences)
6. IMPORTANT: A rating of -1 indicates a newly opened place. Mentioning it's a "New" or "Recently opened" place in the description or highlights when applicable.
7. IMPORTANT: Only include ONE location from each chain restaurant (like Starbucks, McDonald's, KFC, Subway, etc.)
8. IMPORTANT: Only suggest chain restaurants if no other suitable options are available or if available options don't match the query well.
9. Prioritize local, unique establishments over chain restaurants when possible.

CRITICAL: Return ONLY valid JSON without explanation text, code blocks, or any other formatting.
    """
    human_template = "{user_input}"
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
    
    # Create a wrapper that ensures valid JSON output
    def ensure_json_response(chain_response):
        try:
            if hasattr(chain_response, 'content'):
                text = chain_response.content
            else:
                text = str(chain_response)
                
            # Clean up any markdown formatting or explanatory text
            json_text = re.sub(r'^```(json)?|```$', '', text, flags=re.MULTILINE).strip()
            
            # Find JSON content if mixed with text
            json_match = re.search(r'(\{[\s\S]*\})', json_text)
            if json_match:
                json_text = json_match.group(1)
                
            # Parse and validate
            response_data = json.loads(json_text)
            
            # Ensure expected structure
            if 'recommendations' not in response_data:
                response_data = {
                    'recommendations': response_data if isinstance(response_data, list) else [],
                    'summary': "Here are some personalized place recommendations."
                }
                
            return response_data
            
        except Exception as e:
            print(f"Error parsing JSON response: {e}")
            # Fallback to a minimal valid structure
            return {
                'recommendations': [],
                'summary': "Could not generate structured recommendations."
            }
    
    # Use the LangChain chain with our custom wrapper
    chain = prompt | llm 
    return lambda inputs: ensure_json_response(chain.invoke(inputs))

def format_places_for_llm(places: List[Dict], results: List[SearchResult]) -> List[Dict]:
    """
    Prepare place data for the LLM response using only similarity scores.
    """
    score_map = {result.id: result for result in results}
    formatted_places = []
    for place in places:
        pid = place.get("id")
        cuisine = ", ".join(place.get("cuisine", [])) if place.get("cuisine") else ""
        tags = ", ".join(place.get("tags", [])) if place.get("tags") else ""
        
        # Process rating - convert to float, -1 for "New", or None if invalid
        rating_value = place.get("rating")
        if rating_value is not None:
            try:
                if isinstance(rating_value, str) and rating_value.lower() == "new":
                    # Use -1 as a special value to represent "New" places
                    rating = -1.0
                else:
                    rating = float(rating_value) if rating_value != "" else None
            except (ValueError, TypeError):
                # Handle other non-numeric strings
                rating = None
        else:
            rating = None
            
        # Process price - convert to float or None if not a valid number
        price_value = place.get("price")
        if price_value is not None:
            try:
                price = float(price_value) if price_value != "" else None
            except (ValueError, TypeError):
                price = None
        else:
            price = None
            
        ranking_info = {}
        if pid in score_map:
            res = score_map[pid]
            ranking_info = {
                "similarity_score": round(res.similarity, 3)
            }
            
        formatted_place = {
            "id": pid,
            "name": place.get("name", ""),
            "address": place.get("address", ""),
            "city": place.get("city", ""),
            "cuisine": cuisine,
            "tags": tags,
            "rating": rating,
            "price": price,
            "description": place.get("description", ""),
            "image": place.get("image", ""),
            "ranking": ranking_info
        }
        formatted_places.append(formatted_place)
        
    # Sort by similarity score
    formatted_places.sort(key=lambda x: x.get("ranking", {}).get("similarity_score", 0), reverse=True)
    return formatted_places
