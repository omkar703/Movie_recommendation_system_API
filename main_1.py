from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import pickle
import pandas as pd
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime
import re
from difflib import SequenceMatcher
import string
# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Movie Recommendation API",
    description="A production-ready API for movie recommendations with search suggestions",
    version="2.0.0"
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
    backdrop_path: Optional[str] = None
    overview: Optional[str] = None
    genres: Optional[List[str]] = None
    release_date: Optional[str] = None
    release_year: Optional[int] = None
    vote_average: Optional[float] = None
    vote_count: Optional[int] = None
    popularity: Optional[float] = None
    runtime: Optional[int] = None
    imdb_id: Optional[str] = None
    tmdb_url: Optional[str] = None

class SearchSuggestion(BaseModel):
    title: str
    movie_id: int
    release_year: Optional[int] = None
    poster_path: Optional[str] = None
    match_score: float

class SearchResponse(BaseModel):
    query: str
    suggestions: List[SearchSuggestion]
    exact_matches: List[MovieBase]
    total_found: int
    status: str

class MovieRecommendationRequest(BaseModel):
    title: str = Field(..., description="Movie title to get recommendations for")
    num_recommendations: Optional[int] = Field(default=10, ge=1, le=50, description="Number of recommendations (1-50)")
    include_similar_genres: Optional[bool] = Field(default=True, description="Include genre-based filtering")
    min_vote_average: Optional[float] = Field(default=0.0, ge=0.0, le=10.0, description="Minimum vote average filter")

class DetailedMovieRecommendationResponse(BaseModel):
    input_movie: MovieBase
    recommendations: List[MovieBase]
    recommendation_metadata: Dict[str, Any]
    status: str
    message: str

class MovieDetailsResponse(BaseModel):
    movie: MovieBase
    similar_movies: List[MovieBase]
    status: str

# Load the model and data
try:
    with open('movie_recommender_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    df = model_data['df']
    similarity = model_data['similarity']
    print("✅ Model loaded successfully")
    print(f"📊 Loaded {len(df)} movies")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    raise Exception("Failed to load model")

# Create search index for faster searching
def create_search_index():
    """Create an optimized search index"""
    search_terms = []
    for idx, row in df.iterrows():
        title = str(row['title']).lower()
        search_terms.append({
            'title': title,
            'original_title': row['title'],
            'movie_id': row['movie_id'],
            'index': idx,
            'release_year': pd.to_datetime(row.get('release_date')).year if pd.notnull(row.get('release_date')) else None
        })
    return search_terms

search_index = create_search_index()

def calculate_similarity(query: str, title: str) -> float:
    """Calculate similarity score between query and title"""
    return SequenceMatcher(None, query.lower(), title.lower()).ratio()

def get_enhanced_movie_details(movie_id: int, similarity_score: Optional[float] = None) -> MovieBase:
    """Get enhanced movie information with TMDB integration"""
    try:
        movie = df[df['movie_id'] == movie_id].iloc[0]
        
        # Parse genres safely
        genres = []
        try:
            if pd.notnull(movie.get('genres')):
                genres_data = eval(str(movie['genres']))
                if isinstance(genres_data, list):
                    genres = [genre['name'] for genre in genres_data if isinstance(genre, dict)]
        except:
            genres = []
        
        # Parse release date
        release_date = None
        release_year = None
        if pd.notnull(movie.get('release_date')):
            try:
                release_date = str(movie['release_date'])
                release_year = pd.to_datetime(release_date).year
            except:
                pass
        
        movie_details = MovieBase(
            movie_id=movie_id,
            title=movie['title'],
            similarity_score=similarity_score,
            poster_path=f"https://image.tmdb.org/t/p/w500{movie.get('poster_path', '')}" if pd.notnull(movie.get('poster_path')) else None,
            backdrop_path=f"https://image.tmdb.org/t/p/w1280{movie.get('backdrop_path', '')}" if pd.notnull(movie.get('backdrop_path')) else None,
            overview=movie.get('overview', ''),
            genres=genres,
            release_date=release_date,
            release_year=release_year,
            vote_average=float(movie.get('vote_average', 0)) if pd.notnull(movie.get('vote_average')) else None,
            vote_count=int(movie.get('vote_count', 0)) if pd.notnull(movie.get('vote_count')) else None,
            popularity=float(movie.get('popularity', 0)) if pd.notnull(movie.get('popularity')) else None,
            runtime=int(movie.get('runtime', 0)) if pd.notnull(movie.get('runtime')) else None,
            imdb_id=movie.get('imdb_id'),
            tmdb_url=f"https://www.themoviedb.org/movie/{movie_id}"
        )
        return movie_details
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Movie details not found: {str(e)}")

def smart_search_movies(query: str, limit: int = 10) -> SearchResponse:
    """Enhanced search with suggestions and smart matching"""
    query_lower = query.lower().strip()
    
    if len(query_lower) < 2:
        return SearchResponse(
            query=query,
            suggestions=[],
            exact_matches=[],
            total_found=0,
            status="error - query too short"
        )
    
    suggestions = []
    exact_matches = []
    
    # Search through index
    for item in search_index:
        title_lower = item['title']
        similarity_score = calculate_similarity(query_lower, title_lower)
        
        if similarity_score > 0.3:  # Threshold for suggestions
            suggestion = SearchSuggestion(
                title=item['original_title'],
                movie_id=item['movie_id'],
                release_year=item['release_year'],
                poster_path=f"https://image.tmdb.org/t/p/w200{df.iloc[item['index']].get('poster_path', '')}" if pd.notnull(df.iloc[item['index']].get('poster_path')) else None,
                match_score=round(similarity_score, 3)
            )
            
            # Check for exact or very close matches
            if similarity_score > 0.85 or query_lower in title_lower:
                try:
                    movie_details = get_enhanced_movie_details(item['movie_id'])
                    exact_matches.append(movie_details)
                except:
                    pass
            
            suggestions.append(suggestion)
    
    # Sort by similarity score
    suggestions.sort(key=lambda x: x.match_score, reverse=True)
    suggestions = suggestions[:limit]
    
    return SearchResponse(
        query=query,
        suggestions=suggestions,
        exact_matches=exact_matches[:5],  # Limit exact matches
        total_found=len(suggestions),
        status="success"
    )

def get_enhanced_recommendations(title: str, num_recommendations: int = 10, 
                               include_similar_genres: bool = True,
                               min_vote_average: float = 0.0) -> DetailedMovieRecommendationResponse:
    """Generate enhanced movie recommendations with metadata"""
    try:
        # Find movie index (case insensitive)
        movie_matches = df[df['title'].str.lower() == title.lower()]
        if movie_matches.empty:
            # Try partial matching
            movie_matches = df[df['title'].str.contains(title, case=False, na=False)]
            if movie_matches.empty:
                raise HTTPException(status_code=404, detail=f"Movie '{title}' not found in database")
        
        movie_idx = movie_matches.index[0]
        input_movie_id = df.iloc[movie_idx]['movie_id']
        
        # Get similarity scores
        similarity_scores = list(enumerate(similarity[movie_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get input movie details
        input_movie = get_enhanced_movie_details(input_movie_id)
        
        # Filter and enhance recommendations
        recommendations = []
        recommendation_count = 0
        
        for idx, score in similarity_scores[1:]:  # Skip the movie itself
            if recommendation_count >= num_recommendations:
                break
                
            try:
                movie_id = df.iloc[idx]['movie_id']
                movie_details = get_enhanced_movie_details(movie_id, similarity_score=round(score, 4))
                
                # Apply filters
                if min_vote_average > 0 and (not movie_details.vote_average or movie_details.vote_average < min_vote_average):
                    continue
                
                # Genre similarity filter (optional)
                if include_similar_genres and input_movie.genres and movie_details.genres:
                    common_genres = set(input_movie.genres) & set(movie_details.genres)
                    if not common_genres and score < 0.5:  # Skip if no common genres and low similarity
                        continue
                
                recommendations.append(movie_details)
                recommendation_count += 1
                
            except Exception as e:
                print(f"Error processing recommendation {idx}: {e}")
                continue
        
        # Recommendation metadata
        metadata = {
            "total_movies_analyzed": len(similarity_scores) - 1,
            "average_similarity_score": round(np.mean([r.similarity_score for r in recommendations if r.similarity_score]), 4),
            "genre_diversity": len(set([genre for r in recommendations for genre in (r.genres or [])])),
            "filters_applied": {
                "min_vote_average": min_vote_average,
                "include_similar_genres": include_similar_genres
            }
        }
        
        return DetailedMovieRecommendationResponse(
            input_movie=input_movie,
            recommendations=recommendations,
            recommendation_metadata=metadata,
            status="success",
            message=f"Found {len(recommendations)} high-quality recommendations for '{input_movie.title}'"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# API Endpoints
@app.get("/", tags=["Health Check"])
async def root():
    """Health check endpoint with API information"""
    return {
        "status": "healthy",
        "message": "Enhanced Movie Recommendation API is running",
        "version": "2.0.0",
        "features": [
            "Smart search suggestions",
            "Enhanced movie details with TMDB integration",
            "Advanced filtering options",
            "Detailed recommendation metadata"
        ],
        "endpoints": {
            "search": "/movies/search/{query}",
            "recommendations": "/movies/recommend",
            "movie_details": "/movies/{movie_id}"
        }
    }

@app.get("/movies/search/{query}", tags=["Movie Search"])
async def search_movies_enhanced(
    query: str, 
    limit: int = Query(default=10, ge=1, le=50, description="Maximum number of suggestions")
):
    """Enhanced search with smart suggestions and exact matches"""
    try:
        return smart_search_movies(query, limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching movies: {str(e)}")

@app.post("/movies/recommend", response_model=DetailedMovieRecommendationResponse, tags=["Movie Recommendations"])
async def recommend_movies_enhanced(request: MovieRecommendationRequest):
    """Get enhanced movie recommendations with advanced filtering"""
    return get_enhanced_recommendations(
        title=request.title,
        num_recommendations=request.num_recommendations,
        include_similar_genres=request.include_similar_genres,
        min_vote_average=request.min_vote_average
    )

@app.get("/movies/recommend/{title}", tags=["Movie Recommendations"])
async def recommend_movies_get(
    title: str,
    num_recommendations: int = Query(default=10, ge=1, le=50),
    include_similar_genres: bool = Query(default=True),
    min_vote_average: float = Query(default=0.0, ge=0.0, le=10.0)
):
    """Get movie recommendations via GET request"""
    return get_enhanced_recommendations(
        title=title,
        num_recommendations=num_recommendations,
        include_similar_genres=include_similar_genres,
        min_vote_average=min_vote_average
    )

@app.get("/movies/{movie_id}", response_model=MovieDetailsResponse, tags=["Movie Details"])
async def get_movie_details(
    movie_id: int,
    include_similar: bool = Query(default=True, description="Include similar movies")
):
    """Get detailed information about a specific movie with similar movies"""
    try:
        movie_details = get_enhanced_movie_details(movie_id)
        
        similar_movies = []
        if include_similar:
            try:
                # Find the movie in our dataset and get similar ones
                movie_row = df[df['movie_id'] == movie_id]
                if not movie_row.empty:
                    movie_idx = movie_row.index[0]
                    similarity_scores = list(enumerate(similarity[movie_idx]))
                    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
                    
                    # Get top 5 similar movies
                    for idx, score in similarity_scores[1:6]:
                        try:
                            similar_movie_id = df.iloc[idx]['movie_id']
                            similar_movie = get_enhanced_movie_details(similar_movie_id, similarity_score=round(score, 4))
                            similar_movies.append(similar_movie)
                        except:
                            continue
            except Exception as e:
                print(f"Error getting similar movies: {e}")
        
        return MovieDetailsResponse(
            movie=movie_details,
            similar_movies=similar_movies,
            status="success"
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Movie not found: {str(e)}")

@app.get("/movies/popular/trending", tags=["Movie Discovery"])
async def get_trending_movies(limit: int = Query(default=20, ge=1, le=100)):
    """Get trending movies based on popularity and vote average"""
    try:
        # Filter movies with good ratings and popularity
        trending = df[
            (df['vote_average'] >= 7.0) & 
            (df['vote_count'] >= 100) & 
            (df['popularity'] > 10)
        ].nlargest(limit, ['popularity', 'vote_average'])
        
        movies = []
        for _, row in trending.iterrows():
            try:
                movie = get_enhanced_movie_details(row['movie_id'])
                movies.append(movie)
            except:
                continue
        
        return {
            "status": "success",
            "count": len(movies),
            "trending_movies": movies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting trending movies: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)