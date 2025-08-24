# Movie Recommendation API Testing Guide

## 🚀 Start Your API Server

```bash
python your_api_file.py
```

Your API will be running at: `http://localhost:8000`

## 📋 Available Endpoints to Test

### 1. **Health Check**

```bash
curl -X GET "http://localhost:8000/"
```

### 2. **Smart Movie Search with Suggestions**

```bash
# Search for movies (returns suggestions + exact matches)
curl -X GET "http://localhost:8000/movies/search/batman?limit=10"
curl -X GET "http://localhost:8000/movies/search/avengers?limit=5"
curl -X GET "http://localhost:8000/movies/search/inception?limit=8"
```

### 3. **Get Movie Recommendations (POST)**

```bash
curl -X POST "http://localhost:8000/movies/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "The Dark Knight",
    "num_recommendations": 10,
    "include_similar_genres": true,
    "min_vote_average": 7.0
  }'
```

```bash
curl -X POST "http://localhost:8000/movies/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Avatar",
    "num_recommendations": 15,
    "include_similar_genres": false,
    "min_vote_average": 6.5
  }'
```

### 4. **Get Movie Recommendations (GET)**

```bash
curl -X GET "http://localhost:8000/movies/recommend/Interstellar?num_recommendations=12&min_vote_average=7.5&include_similar_genres=true"

curl -X GET "http://localhost:8000/movies/recommend/Titanic?num_recommendations=8&min_vote_average=6.0"
```

### 5. **Get Detailed Movie Information**

```bash
# Get movie details with similar movies
curl -X GET "http://localhost:8000/movies/155?include_similar=true"
curl -X GET "http://localhost:8000/movies/550?include_similar=true"
curl -X GET "http://localhost:8000/movies/27205?include_similar=false"
```

### 6. **Get Trending Movies**

```bash
curl -X GET "http://localhost:8000/movies/popular/trending?limit=20"
curl -X GET "http://localhost:8000/movies/popular/trending?limit=10"
```

## 🔧 Testing with Different Tools

### **Using Python Requests**

```python
import requests
import json

# Base URL
BASE_URL = "http://localhost:8000"

# Test search
response = requests.get(f"{BASE_URL}/movies/search/batman")
print("Search Results:", json.dumps(response.json(), indent=2))

# Test recommendations
payload = {
    "title": "The Matrix",
    "num_recommendations": 10,
    "min_vote_average": 7.0
}
response = requests.post(f"{BASE_URL}/movies/recommend", json=payload)
print("Recommendations:", json.dumps(response.json(), indent=2))
```

### **Using JavaScript/Fetch**

```javascript
// Test search
fetch("http://localhost:8000/movies/search/spiderman?limit=10")
  .then((response) => response.json())
  .then((data) => console.log("Search Results:", data));

// Test recommendations
fetch("http://localhost:8000/movies/recommend", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
  },
  body: JSON.stringify({
    title: "Iron Man",
    num_recommendations: 15,
    min_vote_average: 6.5,
  }),
})
  .then((response) => response.json())
  .then((data) => console.log("Recommendations:", data));
```

### **Using Postman**

1. **Import Collection**: Create a new collection in Postman
2. **Add these requests**:

**GET Search Movies**

- URL: `http://localhost:8000/movies/search/{{movie_query}}`
- Method: GET
- Params: `limit=10`

**POST Recommendations**

- URL: `http://localhost:8000/movies/recommend`
- Method: POST
- Body (JSON):

```json
{
  "title": "The Godfather",
  "num_recommendations": 10,
  "include_similar_genres": true,
  "min_vote_average": 8.0
}
```

**GET Movie Details**

- URL: `http://localhost:8000/movies/{{movie_id}}`
- Method: GET
- Params: `include_similar=true`

## 🧪 Test Scenarios

### **Scenario 1: Movie Discovery Flow**

```bash
# 1. Search for a movie
curl -X GET "http://localhost:8000/movies/search/godfather?limit=5"

# 2. Get recommendations based on search result
curl -X POST "http://localhost:8000/movies/recommend" \
  -H "Content-Type: application/json" \
  -d '{"title": "The Godfather", "num_recommendations": 10}'

# 3. Get details of a recommended movie
curl -X GET "http://localhost:8000/movies/238?include_similar=true"
```

### **Scenario 2: High-Quality Recommendations**

```bash
# Get only highly-rated recommendations
curl -X POST "http://localhost:8000/movies/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Pulp Fiction",
    "num_recommendations": 15,
    "min_vote_average": 8.0,
    "include_similar_genres": true
  }'
```

### **Scenario 3: Trending Discovery**

```bash
# Get trending movies
curl -X GET "http://localhost:8000/movies/popular/trending?limit=15"
```

## 📊 Expected Response Formats

### **Search Response**

```json
{
  "query": "batman",
  "suggestions": [
    {
      "title": "Batman Begins",
      "movie_id": 272,
      "release_year": 2005,
      "poster_path": "https://image.tmdb.org/t/p/w200/path.jpg",
      "match_score": 0.95
    }
  ],
  "exact_matches": [...],
  "total_found": 10,
  "status": "success"
}
```

### **Recommendation Response**

```json
{
  "input_movie": {
    "movie_id": 155,
    "title": "The Dark Knight",
    "poster_path": "https://image.tmdb.org/t/p/w500/path.jpg",
    "overview": "Movie description...",
    "genres": ["Action", "Crime", "Drama"],
    "release_year": 2008,
    "vote_average": 9.0
  },
  "recommendations": [...],
  "recommendation_metadata": {
    "total_movies_analyzed": 1500,
    "average_similarity_score": 0.75,
    "genre_diversity": 8
  },
  "status": "success"
}
```

## 🐛 Testing Error Cases

### **Test Invalid Movie**

```bash
curl -X POST "http://localhost:8000/movies/recommend" \
  -H "Content-Type: application/json" \
  -d '{"title": "NonExistentMovie123", "num_recommendations": 10}'
```

### **Test Invalid Movie ID**

```bash
curl -X GET "http://localhost:8000/movies/999999"
```

### **Test Invalid Parameters**

```bash
curl -X POST "http://localhost:8000/movies/recommend" \
  -H "Content-Type: application/json" \
  -d '{"title": "Batman", "num_recommendations": 100}'
```

## 📖 Interactive API Documentation

Once your server is running, visit:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

These provide interactive testing interfaces where you can:

- Try all endpoints directly in the browser
- See request/response schemas
- Test with different parameters
- Download API specifications

## 💡 Pro Testing Tips

1. **Start with Health Check** to ensure API is running
2. **Test Search First** to find valid movie titles in your dataset
3. **Use Search Results** as input for recommendation testing
4. **Test Edge Cases** like empty queries, invalid IDs
5. **Check Response Times** for performance optimization
6. **Validate Image URLs** in responses work correctly
7. **Test with Popular Movies** like "Avatar", "Titanic", "The Dark Knight"

## 🎬 Popular Movie Titles to Test With

- "The Dark Knight"
- "Avatar"
- "Titanic"
- "The Godfather"
- "Pulp Fiction"
- "Forrest Gump"
- "Inception"
- "The Matrix"
- "Iron Man"
- "Avengers"
- "Star Wars"
- "Jurassic Park"
