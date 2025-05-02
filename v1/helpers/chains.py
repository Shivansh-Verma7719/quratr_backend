import re
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from .helpers import SearchResult
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
    Only generates AI attributes (description, match_reasons, highlights, atmosphere) to improve performance.
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
      "atmosphere": "Description of atmosphere"
    }},
     more places...
  ],
  "summary": "Brief overall summary of the recommendations"
}}

Recommendation Guidelines:
1. ALWAYS include at least 5 places from the provided data if not more, even if they only partially match the user's query
2. The user's query is the PRIMARY filter - user profile attributes are SECONDARY preferences
3. User attributes should influence the ranking and explanation of places, but should NOT exclude places
4. Start with places that best match the query, then prioritize those that ALSO align with user attributes
5. Include 3-5 highlight points for each place that emphasize aspects relevant to the query and user preferences
6. The summary should be concise (2-3 sentences) and mention key themes in the recommendations

User Attribute Considerations (these enhance but don't limit recommendations):
- "Nightlife Enthusiast": Mention evening/night atmosphere features when present
- "Luxury-Seeking": Highlight premium features for higher-priced establishments
- "Solitary": Point out aspects good for solo dining when applicable
- "Adventurous": Emphasize unique and exotic elements when present
- "Social": Note features good for groups when applicable

Other Important Notes:
- Rating of -1 indicates a newly opened place - mention this as "New" or "Recently opened" when applicable
- Include only ONE location from each chain restaurant (Starbucks, McDonald's, KFC, etc.)
- Prioritize local, unique establishments over chains when available
- If a place doesn't perfectly match attributes but matches the query well, include it anyway

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
