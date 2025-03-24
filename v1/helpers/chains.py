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
      "price_range": "$$$",
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

def rerank_by_ordering(results: List[SearchResult], places: list, intent: QueryIntent, llm: ChatOpenAI, debug: bool = False) -> List[SearchResult]:
    """
    Rerank results using an ordering-based approach:
      - Compile a summary of each place.
      - Provide query analysis.
      - Ask the LLM for a comma-separated list of place IDs in descending order.
      - Parse the output and assign scores (trimmed to 3 decimals).
    """
    # Build a summary for each place.
    place_map = {place["id"]: place for place in places}
    details_list = []
    for result in results:
        if result.id in place_map:
            place = place_map[result.id]
            cuisine = ", ".join(place.get("cuisine", [])) if place.get("cuisine") else "Unknown"
            details = f"ID: {place.get('id')}, Name: {place.get('name','')}, Cuisine: {cuisine}, Price: {place.get('price','')}, Rating: {place.get('rating','')}, Location: {place.get('address','')} {place.get('city','')}"
            details_list.append(details)
    combined_details = "\n".join(details_list)
    query_analysis = (
        f"User Query: \"{intent.original_query}\"\n"
        f"Cuisine Preferences: {', '.join(intent.cuisine_types) if intent.cuisine_types else 'Any'}\n"
        f"Locations: {', '.join(intent.locations) if intent.locations else 'Any'}\n"
        f"Price Range: {intent.price_range if intent.price_range else 'Any'}\n"
        f"Atmosphere: {intent.atmosphere if intent.atmosphere else 'Any'}\n"
    )
    user_input = f"{query_analysis}\nPlace Details:\n{combined_details}\n\nOutput a comma-separated list of place IDs in descending order of relevance (most relevant first)."
    ordering_chain = create_ordering_chain(llm)
    raw_ordering_output = ordering_chain.invoke({"user_input": user_input})
    if not isinstance(raw_ordering_output, str):
        raw_ordering_output = str(raw_ordering_output)
    cleaned_output = clean_ordering_output(raw_ordering_output)
    if debug:
        print("Ordering chain raw output:", raw_ordering_output)
        print("Cleaned ordering output:", cleaned_output)
    id_list = re.findall(r'\d+', cleaned_output)
    id_list = [int(x) for x in id_list]
    if debug:
        print("Parsed ordering:", id_list)
    n = len(id_list)
    ranking_dict = {pid: round((n - rank + 1) / n * 10, 3) for rank, pid in enumerate(id_list, start=1)}
    for result in results:
        if result.id in ranking_dict:
            result.reranking_score = ranking_dict[result.id]
            result.final_score = round((result.similarity * 0.4) + ((result.reranking_score / 10) * 0.6), 3)
    results.sort(key=lambda x: x.final_score, reverse=True)
    return results

def format_places_for_llm(places: List[Dict], results: List[SearchResult]) -> List[Dict]:
    """
    Prepare place data for the LLM response. Relevance scores are trimmed to 3 decimals.
    """
    score_map = {result.id: result for result in results}
    formatted_places = []
    for place in places:
        pid = place.get("id")
        cuisine = ", ".join(place.get("cuisine", [])) if place.get("cuisine") else ""
        tags = ", ".join(place.get("tags", [])) if place.get("tags") else ""
        ranking_info = {}
        if pid in score_map:
            res = score_map[pid]
            ranking_info = {
                "similarity_score": round(res.similarity, 3),
                "relevance_score": round(res.reranking_score, 3),
                "final_score": round(res.final_score, 3)
            }
        formatted_place = {
            "id": pid,
            "name": place.get("name", ""),
            "address": place.get("address", ""),
            "city": place.get("city", ""),
            "cuisine": cuisine,
            "tags": tags,
            "rating": place.get("rating", ""),
            "price": place.get("price", ""),
            "description": place.get("description", ""),
            "image": place.get("image", ""),
            "ranking": ranking_info
        }
        formatted_places.append(formatted_place)
    formatted_places.sort(key=lambda x: x.get("ranking", {}).get("final_score", 0), reverse=True)
    return formatted_places
