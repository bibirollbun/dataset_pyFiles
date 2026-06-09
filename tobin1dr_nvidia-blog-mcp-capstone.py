# Example: RAG Query Implementation
# From mcp/query_rag.py

import requests
from typing import List, Dict

class RAGQuery:
    """Enhanced RAG query interface with query transformation and answer grading."""
    
    def _retrieve_contexts(
        self,
        query_text: str,
        similarity_top_k: int = 10,
        vector_distance_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Retrieve contexts from RAG Corpus using Vertex AI API.
        """
        retrieve_url = (
            f"{self.base_url}/projects/{self.project_id}/"
            f"locations/{self.region}:retrieveContexts"
        )
        
        request_body = {
            "vertex_rag_store": {
                "rag_resources": {
                    "rag_corpus": self.rag_corpus_name
                },
                "vector_distance_threshold": vector_distance_threshold
            },
            "query": {
                "text": query_text,
                "similarity_top_k": similarity_top_k
            }
        }
        
        # Returns contexts with text, source_uri, and distance scores
        response = requests.post(retrieve_url, headers=headers, json=request_body)
        contexts = response.json().get("contexts", {}).get("contexts", [])
        return contexts


# Example: Query Transformation Implementation
# From mcp/rag_query_transformer.py

from vertexai.generative_models import GenerativeModel
import vertexai

class QueryTransformer:
    """Transforms user queries to improve retrieval quality."""
    
    def transform_query(self, original_query: str) -> str:
        """
        Transform a user query to improve retrieval quality.
        Uses Gemini 2.0 Flash to rewrite vague queries into specific ones.
        """
        transformation_prompt = f"""You are a query transformation expert specializing in NVIDIA developer documentation.

TRANSFORMATION GUIDELINES:
1. Expand abbreviations and acronyms (e.g., "CUDA" â†’ "CUDA parallel computing platform")
2. Add technical context when queries are vague
3. Include relevant NVIDIA technologies when implied
4. Preserve the user's intent while making the query more specific

ORIGINAL QUERY:
{original_query}

TRANSFORMED QUERY:"""
        
        response = self.model.generate_content(
            transformation_prompt,
            generation_config={
                "temperature": 0.3,  # Lower temperature for consistency
                "max_output_tokens": 200,
            }
        )
        
        return response.text.strip()

# Example transformations:
# "optimize CUDA" â†’ "CUDA kernel optimization techniques and best practices"
# "TensorRT" â†’ "TensorRT inference optimization and deployment"
# "GPU memory" â†’ "NVIDIA GPU memory management and optimization"


# Example: Answer Grading & Iterative Refinement
# From mcp/query_rag.py and mcp/rag_answer_grader.py

from pydantic import BaseModel, Field
import json

class AnswerGrade(BaseModel):
    """Structured grade for an answer or retrieved context."""
    score: float = Field(..., description="Quality score from 0.0 to 1.0")
    relevance: float = Field(..., description="Relevance to query from 0.0 to 1.0")
    completeness: float = Field(..., description="Completeness of answer from 0.0 to 1.0")
    grounded: bool = Field(..., description="Whether answer is grounded in retrieved contexts")
    reasoning: str = Field(..., description="Brief explanation of the grade")
    should_refine: bool = Field(..., description="Whether query should be refined and retried")

class AnswerGrader:
    """Grades retrieved contexts for quality and relevance."""
    
    def grade_contexts(
        self,
        query: str,
        contexts: List[Dict],
        min_acceptable_score: float = 0.6
    ) -> AnswerGrade:
        """
        Grade retrieved contexts using Gemini 2.0 Flash.
        Returns structured grade with scores and refinement recommendation.
        """
        grading_prompt = f"""Evaluate whether retrieved contexts are sufficient to answer the query.

EVALUATION CRITERIA:
1. Relevance (0.0-1.0): How well do contexts address the query?
2. Completeness (0.0-1.0): Do contexts provide enough information?
3. Grounded (true/false): Can answer be fully supported by contexts?
4. Should Refine (true/false): Should query be refined and retried?

USER QUERY: {query}
RETRIEVED CONTEXTS: {contexts_summary}

Provide evaluation in JSON format:"""
        
        response = self.model.generate_content(grading_prompt)
        grade = AnswerGrade(**json.loads(response.text))
        return grade

# Iterative refinement loop (from query_rag.py):
# NOTE: This is example code showing the refinement loop pattern
# In actual implementation, this code runs inside the RAGQuery class
# Commented out here to prevent execution errors in the notebook

# max_refinement_iterations = 2
# for iteration in range(max_refinement_iterations + 1):
#     contexts = self._retrieve_contexts(current_query)
#     grade = self.answer_grader.grade_contexts(query, contexts)
#     
#     if not grade.should_refine or iteration >= max_refinement_iterations:
#         break  # Quality acceptable or max iterations reached
#     
#     # Refine query for next iteration
#     current_query = self.query_transformer.transform_query(refinement_prompt)


# Example: MCP Server Implementation
# From mcp/mcp_server.py
# NOTE: This is example code showing the MCP server structure
# Commented out to prevent execution errors (requires MCP dependencies)

# from mcp.server.fastmcp import FastMCP
# from pydantic import BaseModel, Field

# # Initialize MCP server
# mcp = FastMCP(
#     "NVIDIA Developer Resources Search",
#     stateless_http=True,  # Required for Cloud Run
#     instructions="A read-only search tool for NVIDIA developer blog content."
# )

# # Define tool with structured output
# @mcp.tool()
# def search_nvidia_blogs(
#     query: str,
#     method: str = "rag",
#     top_k: int = 10
# ) -> dict:
#     """
#     Enhanced grounded search tool for NVIDIA developer resources.
#     
#     Args:
#         query: Search query (e.g., "How to optimize CUDA kernels")
#         method: "rag" (default) or "vector"
#         top_k: Number of results (1-20, default: 10)
#     
#     Returns:
#         SearchResult with contexts, source URIs, and quality grades
#     """
#     if method == "rag":
#         # Use RAG with transformation, grading, and refinement
#         rag_query = get_rag_query()
#         result = rag_query.query(
#             query_text=query,
#             similarity_top_k=top_k,
#             vector_distance_threshold=0.5
#         )
#         return RAGQueryResult(**result)
#     elif method == "vector":
#         # Use Vector Search for semantic similarity
#         vector_query = get_vector_query()
#         result = vector_query.query(query_text=query, num_neighbors=top_k)
#         return VectorSearchResult(**result)

print("MCP Server example code (commented out - see GitHub repo for full implementation)")


# Example: Vector Search Implementation
# From mcp/query_vector_search.py

from google.cloud import aiplatform
from google.cloud.aiplatform import matching_engine
from vertexai.language_models import TextEmbeddingModel

class VectorSearchQuery:
    """Read-only query interface for Vertex AI Vector Search."""
    
    def query(self, query_text: str, num_neighbors: int = 10) -> Dict:
        """
        Query Vector Search index for similar vectors.
        """
        # Generate query embedding using text-embedding-004
        query_embedding = self.embedding_model.get_embeddings([query_text])[0].values
        
        # Find nearest neighbors
        results = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_embedding],
            num_neighbors=num_neighbors
        )
        
        # Process results
        neighbors = [
            {
                "datapoint_id": neighbor.id,
                "distance": neighbor.distance,
                "feature_vector": neighbor.feature_vector[:10]  # Preview
            }
            for neighbor in results[0]
        ]
        
        return {
            "query": query_text,
            "neighbors": neighbors,
            "count": len(neighbors)
        }


# Create submission file for competition
# This file is required for Kaggle competition submission

import json

submission_data = {
    "track": "Agents for Good",
    "project_name": "NVIDIA Blog MCP Server",
    "concepts": [
        "RAG (Retrieval-Augmented Generation)",
        "Query Transformation",
        "Answer Grading & Iterative Refinement",
        "Model Context Protocol (MCP)",
        "Vector Search"
    ],
    "github_repo": "https://github.com/TomBombadil/nvidia-blog",
    "live_demo": "https://nvidia-blog-mcp-server-4vvir4xvda-ey.a.run.app",
    "description": "Production-ready MCP server providing grounded access to NVIDIA developer blog content"
}

# Save to /kaggle/working/ directory (required for competition)
with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(submission_data, f, indent=2)

print("âœ… Submission file created successfully!")
print("ğŸ“� File saved to: /kaggle/working/submission.json")
print(f"ğŸ“Š Track: {submission_data['track']}")
print(f"ğŸ”— GitHub: {submission_data['github_repo']}")
print(f"ğŸŒ� Demo: {submission_data['live_demo']}")

