from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Movie Recommendation API",
    description="A production-ready API for movie recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class MovieBase(BaseModel):
    movie_id: int
    title: str
    similarity_score: Optional[float] = None
    poster_path: Optional[str] = None
    overview: Optional[str] = None
    genres: Optional[List[str]] = None
    release_year: Optional[int] = None
    vote_average: Optional[float] = None

class MovieRecommendationRequest(BaseModel):
    title: str
    num_recommendations: Optional[int] = 5

class MovieRecommendationResponse(BaseModel):
    input_movie: MovieBase
    recommendations: List[MovieBase]
    status: str
    message: str

# Load the model and data
try:
    with open('movie_recommender_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    df = model_data['df']
    similarity = model_data['similarity']
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise Exception("Failed to load model")

def get_movie_details(movie_id: int, similarity_score: Optional[float] = None) -> MovieBase:
    """Get detailed movie information from movie ID"""
    try:
        movie = df[df['movie_id'] == movie_id].iloc[0]
        
        # Base movie details
        movie_details = MovieBase(
            movie_id=movie_id,
            title=movie['title'],
            similarity_score=similarity_score,
            poster_path=f"https://image.tmdb.org/t/p/w500/{movie.get('poster_path', '')}",
            overview=movie.get('overview', ''),
            genres=[genre['name'] for genre in eval(movie['genres'])] if isinstance(movie.get('genres'), str) else [],
            release_year=pd.to_datetime(movie.get('release_date')).year if pd.notnull(movie.get('release_date')) else None,
            vote_average=movie.get('vote_average')
        )
        return movie_details
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Movie details not found: {str(e)}")

def get_movie_recommendations(title: str, num_recommendations: int = 5) -> MovieRecommendationResponse:
    """Generate movie recommendations based on title"""
    try:
        # Find movie index
        movie_idx = df[df['title'].str.lower() == title.lower()].index[0]
        
        # Get similarity scores
        similarity_scores = list(enumerate(similarity[movie_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N recommendations
        top_movies = similarity_scores[1:num_recommendations+1]
        
        # Get input movie details
        input_movie = get_movie_details(df.iloc[movie_idx]['movie_id'])
        
        # Get recommendation details
        recommendations = []
        for idx, score in top_movies:
            movie_id = df.iloc[idx]['movie_id']
            movie_details = get_movie_details(movie_id, similarity_score=score)
            recommendations.append(movie_details)
        
        return MovieRecommendationResponse(
            input_movie=input_movie,
            recommendations=recommendations,
            status="success",
            message=f"Found {len(recommendations)} recommendations for '{title}'"
        )
    
    except IndexError:
        raise HTTPException(status_code=404, detail=f"Movie '{title}' not found in database")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# API endpoints
@app.get("/", tags=["Health Check"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Movie Recommendation API is running"
    }

@app.get("/movies/search/{query}", tags=["Movie Search"])
async def search_movies(query: str, limit: int = 10):
    """Search movies by title"""
    try:
        matches = df[df['title'].str.contains(query, case=False, na=False)]
        results = [
            get_movie_details(row['movie_id'])
            for _, row in matches.head(limit).iterrows()
        ]
        return {
            "status": "success",
            "count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching movies: {str(e)}")

@app.post("/movies/recommend", response_model=MovieRecommendationResponse, tags=["Movie Recommendations"])
async def recommend_movies(request: MovieRecommendationRequest):
    """Get movie recommendations"""
    return get_movie_recommendations(
        title=request.title,
        num_recommendations=request.num_recommendations
    )

@app.get("/movies/{movie_id}", response_model=MovieBase, tags=["Movie Details"])
async def get_movie(movie_id: int):
    """Get detailed information about a specific movie"""
    try:
        return get_movie_details(movie_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Movie not found: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)