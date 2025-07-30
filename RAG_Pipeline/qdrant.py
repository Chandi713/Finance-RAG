import os
import re
import ast
import json
import logging
import requests
import numpy as np
import pandas as pd
from openai import AzureOpenAI
from dataclasses import dataclass
import google.generativeai as genai
from collections import defaultdict
from qdrant_client import QdrantClient
from datasets import concatenate_datasets
from datasets import load_dataset, Dataset
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from sentence_transformers.util import cos_sim
from langchain.schema import HumanMessage, AIMessage
from typing import Dict, List, Optional, Any, Tuple, Union
from sentence_transformers import SentenceTransformer, util
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from langchain.memory import VectorStoreRetrieverMemory, ConversationBufferMemory
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny

class QdrantPipeline:
    def __init__(self, qdrant_url, api_key, collection_name="Finance_RAG_DB"):
        self.client = QdrantClient(url=qdrant_url, api_key=api_key)
        self.collection_name = collection_name
    
    def _ensure_indexes(self):
        """Ensure payload indexes exist after data upload"""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="company",
                field_schema="keyword"
            )
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="year",
                field_schema="keyword"  
            )
            
            print("Ensured payload indexes exist for company and year filtering")
            
        except Exception as e:
            print(f"Note: Indexes may already exist: {e}")
            print("Filtering will use fallback method if needed")
    
    def search(self, 
               query_vector: List[float], 
               limit: int = 5,
               company: Optional[Union[str, List[str]]] = None,
               year: Optional[Union[int, List[int]]] = None) -> List[Dict[str, Any]]:
        """
        Company and year filtering on payload + vector similarity search
        
        Args:
            query_vector: Vector embeddings for similarity search
            limit: Number of results to return
            company: Company ticker(s) to filter by
            year: Year(s) to filter by
            
        Returns:
            List of raw result dictionaries
        """
        
        # Build filter conditions for company and year only
        filter_conditions = []
        
        if company:
            if isinstance(company, str):
                filter_conditions.append(FieldCondition(key="company", match=MatchValue(value=company)))
            elif isinstance(company, list):
                filter_conditions.append(FieldCondition(key="company", match=MatchAny(any=company)))
        
        if year:
            if isinstance(year, (int, str)):
                filter_conditions.append(FieldCondition(key="year", match=MatchValue(value=str(year))))
            elif isinstance(year, list):
                filter_conditions.append(FieldCondition(key="year", match=MatchAny(any=[str(y) for y in year])))
        
        # Create filter for payload filtering
        search_filter = Filter(must=filter_conditions) if filter_conditions else None
        
        try:
            # Use Qdrant's native filtering + vector similarity
            if search_filter:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=search_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                ).points
            else:
                # No filtering - just vector similarity
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                ).points
            
            # Return raw results
            raw_results = []
            for result in results:
                raw_results.append({
                    'id': result.id,
                    'score': result.score,
                    'content': result.payload.get('chunks', ''),
                    'company': result.payload.get('company'),
                    'year': result.payload.get('year'),
                    'report': result.payload.get('report'),
                    'item': result.payload.get('item'),
                    'source_file': result.payload.get('source_file')
                })
            
            return raw_results
            
        except Exception as e:
            print(f"Native filtering failed: {e}")
            print("Falling back to manual filtering...")
            
            # Fallback: Manual filtering
            try:
                results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit * 3,
                    with_payload=True,
                    with_vectors=False
                ).points
                
                # Manual filtering for company and year only
                filtered_results = []
                for result in results:
                    if not result.payload:
                        continue
                    
                    match = True
                    
                    # Check company filter
                    if company:
                        payload_company = result.payload.get('company')
                        if isinstance(company, str) and payload_company != company:
                            match = False
                        elif isinstance(company, list) and payload_company not in company:
                            match = False
                    
                    # Check year filter
                    if year and match:
                        payload_year = result.payload.get('year')
                        if isinstance(year, int) and payload_year != year:
                            match = False
                        elif isinstance(year, list) and payload_year not in year:
                            match = False
                    
                    if match:
                        filtered_results.append({
                            'id': result.id,
                            'score': result.score,
                            'content': result.payload.get('chunks', ''),
                            'company': result.payload.get('company'),
                            'year': result.payload.get('year'),
                            'report': result.payload.get('report'),
                            'item': result.payload.get('item'),
                            'source_file': result.payload.get('source_file')
                        })
                        
                        if len(filtered_results) >= limit:
                            break
                
                return filtered_results
                
            except Exception as fallback_error:
                print(f"Fallback filtering also failed: {fallback_error}")
                return []

    def create_indexes_manually(self):
        """Create payload indexes manually for company and year filtering"""
        try:
            print("Creating payload indexes for company and year filtering...")
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="company",
                field_schema="keyword"
            )
            print("Created company index")
            
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="year",
                field_schema="keyword"
            )
            print("Created year index")
            
            print("All required payload indexes created successfully")
            
        except Exception as e:
            print(f"Error creating indexes: {e}")
            print("You may need to check your Qdrant permissions or collection status")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FinancialContext:
    """Data class to store financial context and metadata"""
    company: Optional[str] = None
    year: Optional[List[int]] = None
    context: str = ""
    
    def __post_init__(self):
        pass

class FinancialRAGSystem:
    """Complete Financial RAG system with Gemini 2.5 Flash integration"""
    
    def __init__(self, 
                 qdrant_pipeline, 
                 gemini_api_key: str,
                 encoder_model: str = "gte-finance-model"):
        """
        Initialize the Financial RAG System
        
        Args:
            qdrant_pipeline: Your existing QdrantPipeline instance
            gemini_api_key: Google Gemini API key
            encoder_model: Sentence transformer model for embeddings
        """
        self.qdrant_pipeline = qdrant_pipeline
        self.encoder = SentenceTransformer(encoder_model, trust_remote_code=True)
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Initialize context and memory
        self.current_context = FinancialContext()
        self.memory = ConversationBufferMemory(return_messages=True)
        
        # Dedicated metadata variable
        self.current_metadata = {}
    
        # Company ticker mapping
        self.company_tickers = {
            "UnitedHealth Group Incorporated": "UNH", "Pfizer Inc.": "PFE",
            "Johnson & Johnson": "JNJ", "AbbVie Inc.": "ABBV",
            "Eli Lilly and Company": "LLY",  "Apple Inc.": "AAPL",
            "Microsoft Corporation": "MSFT", "Alphabet Inc.": "GOOGL",
            "Meta Platforms Inc.": "META", "NVIDIA Corporation": "NVDA",
            "Caterpillar Inc.": "CAT", "General Electric Company": "GE",
            "3M Company": "MMM", "Honeywell International Inc.": "HON",
            "Illinois Tool Works Inc.": "ITW", "Lockheed Martin Corporation": "LMT",
            "Northrop Grumman Corporation": "NOC", "Raytheon Technologies Corporation": "RTX",
            "General Dynamics Corporation": "GD", "L3Harris Technologies Inc.": "LHX",
            "Capital One Financial Co.": "COF", "Bank of America Corporation": "BAC",
            "Wells Fargo & Company": "WFC", "Morgan Stanley": "MS"
        }

# ----------------------------------------------------------------------------------------------------------------------------------------

    def get_company_ticker(self, company_name: str) -> Optional[str]:
        """Get company ticker from company name"""
        company_lower = company_name.lower()
        for name, ticker in self.company_tickers.items():
            if name in company_lower:
                return ticker
        return None

# ----------------------------------------------------------------------------------------------------------------------------------------  

    def extract_metadata_with_gemini(self, query: str) -> Dict[str, Any]:
        """Extract metadata from user query using Gemini"""
        prompt = f"""
                **Instruction:**
                Extract specific keywords from the query limited to:
                • Company Name/Ticker (From Edgar)
                • Year
                
                **Expected Output Format:**
                {{
                  "Company": ["MSFT", "AAPL"],
                  "Year": [2024, 2025]
                }}
                
                If either field is not found, use null:
                {{
                  "Company": null,
                  "Year": [2020, 2025]
                }}
                
                **Year Logic:**
                - Extract explicit years as list of integers
                - For ranges (e.g., "2019–2022" or "2019 to 2022"), extract all years inclusively
                - If no year specified, return null
                
                **Company Logic:**
                - Convert company names to stock tickers
                - If company name cannot be identified, return null
                
                Return ONLY the JSON object.
                
                Query: {query}
                """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text.strip()
            
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                metadata = json.loads(json_str)
                
                # Validate and clean
                if 'Company' in metadata and metadata['Company']:
                    if isinstance(metadata['Company'], str):
                        metadata['Company'] = [metadata['Company']]
                    elif not isinstance(metadata['Company'], list):
                        metadata['Company'] = None
                
                if 'Year' in metadata and metadata['Year']:
                    if isinstance(metadata['Year'], int):
                        metadata['Year'] = [metadata['Year']]
                    elif not isinstance(metadata['Year'], list):
                        metadata['Year'] = None
                
                return metadata
            else:
                return {"Company": None, "Year": None}
                
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {"Company": None, "Year": None}

# ----------------------------------------------------------------------------------------------------------------------------------------    

    def refine_query_with_gemini(self, original_query: str, context: str, metadata: Dict[str, Any]) -> str:
        """Refine query for better semantic search"""
        prompt = f"""
                **Instruction:**
                Refine this financial query for better semantic search in 10-K reports.
                
                **Context:** {context}
                **Metadata:** {json.dumps(metadata, indent=2)}
                **Original Query:** {original_query}
                
                **Task:**
                Make the query more specific and optimized for semantic search while preserving intent.
                Add relevant financial terminology and company context if available.
                
                **Output:**
                Return only the refined query as a single string.
                """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            refined_query = response.text.strip()
            
            if refined_query.startswith('"') and refined_query.endswith('"'):
                refined_query = refined_query[1:-1]
            
            return refined_query
            
        except Exception as e:
            logger.error(f"Error refining query: {e}")
            return original_query

# ----------------------------------------------------------------------------------------------------------------------------------------  

    def check_context_alignment(self, new_query: str, current_context: str, new_metadata: Dict[str, Any]) -> bool:
        """Check if new query aligns with current context"""
        prompt = f"""
                **Current Context:** {current_context}
                **New Query:** {new_query}
                **New Metadata:** {json.dumps(new_metadata, indent=2)}
                
                Determine if the new query is a continuation of the current conversation topic.
                Return only "ALIGNED" or "DIFFERENT".
                """
        
        try:
            response = self.gemini_model.generate_content(prompt)
            result = response.text.strip().upper()
            return result == "ALIGNED"
            
        except Exception as e:
            logger.error(f"Error checking context alignment: {e}")
            return False

# ----------------------------------------------------------------------------------------------------------------------------------------  

    def generate_embeddings(self, query: str) -> List[float]:
        """Generate embeddings for the query"""
        return self.encoder.encode(query).tolist()

# ----------------------------------------------------------------------------------------------------------------------------------------

    def generate_content_summary(self, raw_results: List[Dict[str, Any]], query: str) -> str:
        """Generate LLM summary of fetched content"""
        if not raw_results:
            return "No relevant content found for the query."
        
        # Prepare content for summarization
        content_text = ""
        content_list = []
        for i, result in enumerate(raw_results[:5]):  # Top 5 results
            content_text += f"\n[Result {i+1}] Company: {result['company']}, Year: {result['year']}\n"
            content_text += f"Content: {result['content']}\n---\n"
            print(f"Result: {result}")
            content_list.append(result['content'])

        url = os.getenv("JINA_URL")
        # print('#'*130)
        # for i, val in enumerate(content_list):
        #     print(f"Content {i+1}: {val}")
        #     print('-'*130)
        # print('#'*130)

        headers = {
            "Authorization": f"Bearer {os.getenv('JINA_API')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": os.getenv("JINA_MODEL"),
            "query": query,
            "documents": content_list
        }

        response = requests.post(url, headers=headers, json=payload)
        reranked = response.json()

        # print(f"Content Text: {content_text}\n")
        # print('#'*130)
        # print(f"Result Length: {len(reranked['results'])}")
        
        # print(f"Reranked:")
        final_content = ""
        for val in reranked["results"]:
            final_content += val["document"]["text"] + "\n...\n"
        
        
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        model_name = "gpt-4o"
        deployment = "gpt-4o"
        
        subscription_key = os.getenv("AZURE_OPENAI_KEY")
        api_version = "2024-12-01-preview"
        
        client = AzureOpenAI(
            api_version=api_version,
            azure_endpoint=endpoint,
            api_key=subscription_key,
        )

        role = "You are a language model designed to generate highly focused, contextually rich, and semantically complete summaries from source material, with a specific emphasis on addressing the provided query."

        objective = "Generate a well-structured summary that directly answers the query while capturing all relevant and interpretable information from the given corpus. The summary must revolve around the query and ensure the answer to it is unmistakably present and well-supported."

        instructions = """
                        **Instructions:**
                                    
                                    - **Focus & Relevance:**  
                                        • The summary must revolve entirely around the query and address it explicitly and completely.
                                        • Answer each sub-part of the question in a clear and structured manner, no sub-question should be omitted.
                                        • Include only information that directly contributes to answering the query or offers necessary supporting context.
                                        • Do not include irrelevant background, tangential descriptions, or off-topic data.
                                        • Use exact numerical values from the corpus whenever available, no rounding or generalization.
                                        • Quantify and highlight year-over-year changes, emphasizing the magnitude and direction of variation, if required.
                                        • Focus exclusively on company-specific information *related to the query*.
                                    
                                    - **Content & Completeness:**  
                                        • Include all key facts, numbers, dates, entities, and drivers that contribute meaningfully to the query’s answer.
                                        • Capture both direct and inferable insights, as long as they are clearly grounded in the corpus.
                                        • Ensure the answer is fully traceable to the source content, do not introduce interpretations beyond what is supported.
                                        • Tables within the provided chunk may contain data spanning previous and subsequent years. When referencing numerical values, exercise careful judgment in selecting the appropriate figures, as numerical accuracy is critical, particularly in the financial domain where such data is highly sensitive and often subject to precise interpretation.
                                    
                                    - **Language & Style:**  
                                        • Use formal, clear, and professional language throughout.
                                        • Avoid any informal tone, conversational phrasing, emojis, or speculative language.
                                        • Ensure sentences are logically constructed, grammatically correct, and use cohesive transitions.
                                    
                                    - **Structure:**  
                                        • Present the summary as a single, unified narrative.
                                        • Use paragraphs or in-line bullet elements for clarity when helpful, but do not use section headers or labels.
                                        • Maintain consistency in tone, tense, and formatting. Avoid switching styles or perspectives.
                                    
                                    - **Formatting & Output Constraints:**  
                                        • Do not include the instructions, query, or any metadata in the final output.
                                        • The summary must be self-contained and understandable without requiring external context.
                                        • Be concise but thorough, avoid repetition, filler content, or vague generalizations.
                                        • Please provide the response in HTML format, ensuring that all elements—whether textual content, tables, mathematical expressions, or typographic styles such as bold, italic, or customized fonts—are appropriately structured and rendered using valid HTML syntax.
                        """
        
        try:
            result = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"Role:\n{role}\nObjective:\n{objective}\nInstructions:\n{instructions}",
                    },
                    {
                        "role": "user",
                        "content": f"{query}+{final_content}",
                    }
                ],
                max_tokens=4096,
                temperature=1.0,
                top_p=1.0,
                model=deployment
            )
            
            # print(f"Summary:\n {result.choices[0].message.content}")
            response = result.choices[0].message.content
            # response = self.gemini_model.generate_content(prompt)
            # return response.text.strip()
            return response
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Unable to generate content summary."

# ---------------------------------------------------------------------------------------------------------------------------------------- 

    def search_qdrant(self, query_vector: List[float], metadata_filter: Dict[str, Any], limit: int = 10) -> List[Dict[str, Any]]:
        """Company and year filtered search in Qdrant"""
        try:
            # Extract company and year filters only
            company = metadata_filter.get('company')
            year = metadata_filter.get('year')
            
            # Perform filtered search
            results = self.qdrant_pipeline.search(
                query_vector=query_vector,
                limit=limit,
                company=company,
                year=year
            )
            
            return results
                
        except Exception as e:
            logger.error(f"Error searching Qdrant: {e}")
            return []

# ----------------------------------------------------------------------------------------------------------------------------------------  

    def process_query(self, user_query: str) -> Dict[str, Any]:
        """Main method to process user queries"""
        logger.info(f"Processing query: {user_query}")
        
        # Extract metadata from query
        extracted_metadata = self.extract_metadata_with_gemini(user_query)
        logger.info(f"Extracted metadata: {extracted_metadata}")
        
        # Determine case and handle accordingly
        has_prior_context = bool(self.current_context.context)
        query_has_company = extracted_metadata.get('Company') is not None
        
        if not has_prior_context:
            return self._handle_no_prior_context(user_query, extracted_metadata)
        else:
            return self._handle_prior_context_exists(user_query, extracted_metadata)

# ----------------------------------------------------------------------------------------------------------------------------------------  

    def _handle_no_prior_context(self, user_query: str, extracted_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Case 1: No Prior Context"""
        
        if not extracted_metadata.get('Company'):
            return {
                "status": "prompt_needed",
                "message": "Please provide the Company Name to initiate the search.",
                "query": user_query,
                "metadata": extracted_metadata
            }
        
        return self._execute_standard_flow(user_query, extracted_metadata, is_new_context=True)

# ----------------------------------------------------------------------------------------------------------------------------------------   

    def _handle_prior_context_exists(self, user_query: str, extracted_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Case 2: Prior Context Exists"""
        
        if not extracted_metadata.get('Company'):
            return self._execute_standard_flow(user_query, extracted_metadata, is_new_context=False)
        
        # Check if company matches stored metadata
        stored_company = self.current_metadata.get('Company', [None])[0] if self.current_metadata.get('Company') else None
        new_company = extracted_metadata['Company'][0] if extracted_metadata['Company'] else None
        
        if stored_company == new_company:
            return self._execute_standard_flow(user_query, extracted_metadata, is_new_context=False)
        
        # Different company - check context alignment
        context_aligned = self.check_context_alignment(user_query, self.current_context.context, extracted_metadata)
        
        if context_aligned:
            return self._execute_standard_flow(user_query, extracted_metadata, is_new_context=False, update_company=True)
        else:
            return self._execute_standard_flow(user_query, extracted_metadata, is_new_context=True)

# ---------------------------------------------------------------------------------------------------------------------------------------- 

    def _execute_standard_flow(self, 
                              user_query: str, 
                              extracted_metadata: Dict[str, Any], 
                              is_new_context: bool = False,
                              update_company: bool = False,
                              limit: int = 5) -> Dict[str, Any]:
        """Execute the standard query refining and search flow"""
        
        try:
            # Update context and metadata handling
            if is_new_context:
                self.current_context = FinancialContext()
                self.current_metadata = {}
                self.memory.clear()
            
            # Update dedicated metadata variable
            if update_company or is_new_context:
                if extracted_metadata.get('Company'):
                    self.current_metadata['Company'] = extracted_metadata['Company']
                if extracted_metadata.get('Year'):
                    self.current_metadata['Year'] = extracted_metadata['Year']
            
            # Use stored metadata if not provided in query
            working_metadata = extracted_metadata.copy()
            if not working_metadata.get('Company') and self.current_metadata.get('Company'):
                working_metadata['Company'] = self.current_metadata['Company']
            if not working_metadata.get('Year') and self.current_metadata.get('Year'):
                working_metadata['Year'] = self.current_metadata['Year']
            
            # Refine query
            refined_query = self.refine_query_with_gemini(user_query, self.current_context.context, working_metadata)
            logger.info(f"Refined query: {refined_query}")
            
            # Generate embeddings
            query_vector = self.generate_embeddings(refined_query)
            
            # Prepare metadata filter for company and year only
            metadata_filter = {}
            if working_metadata.get('Company'):
                metadata_filter['company'] = working_metadata['Company'][0]
            if working_metadata.get('Year'):
                metadata_filter['year'] = working_metadata['Year']
            
            # Fast filtered search
            raw_results = self.search_qdrant(query_vector, metadata_filter, limit=limit)
            
            # Generate content summary
            content_summary = self.generate_content_summary(raw_results, refined_query)
            
            # Update context with generated summary
            self.current_context.context += f"\nQuery: {refined_query}\nSummary: {content_summary}"
            
            # Update metadata
            self.current_metadata.update({k: v for k, v in working_metadata.items() if v is not None})
            
            # Add to memory
            # self.memory.chat_memory.add_user_message(user_query)
            # self.memory.chat_memory.add_ai_message(f"Retrieved {len(raw_results)} results and generated summary")
            
            # Print raw data and summary
            # print(f"\nRAW DATA CHUNKS")
            # for i, result in enumerate(raw_results):
            #     print(f"Chunk {i+1}:")
            #     print(f"  ID: {result['id']}")
            #     print(f"  Score: {result['score']:.3f}")
            #     print(f"  Company: {result['company']}")
            #     print(f"  Year: {result['year']}")
            #     print(f"  Content: {result['content'][:200]}...")
            #     print("-" * 50)
            
            print(f"\nCONTENT SUMMARY")
            print(content_summary)
            print("=" * 80)
            
            # Prepare response
            response = {
                "status": "success",
                "original_query": user_query,
                "refined_query": refined_query,
                "current_metadata": self.current_metadata,
                "raw_results": raw_results,
                "content_summary": content_summary,
                "context": self.current_context.context,
                "num_results": len(raw_results)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in standard flow: {e}")
            return {
                "status": "error",
                "message": f"An error occurred: {str(e)}",
                "query": user_query,
                "metadata": extracted_metadata
            }

# ----------------------------------------------------------------------------------------------------------------------------------------  

    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the chat history"""
        history = []
        for message in self.memory.chat_memory.messages:
            if isinstance(message, HumanMessage):
                history.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                history.append({"role": "assistant", "content": message.content})
        return history

# ---------------------------------------------------------------------------------------------------------------------------------------- 

    def clear_context(self):
        """Clear current context and memory"""
        self.current_context = FinancialContext()
        self.current_metadata = {}
        self.memory.clear()
        logger.info("Context, metadata, and memory cleared")
