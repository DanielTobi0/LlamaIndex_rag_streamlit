from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from simple import generate_no_stream

app = FastAPI()

@app.get('/')
async def health():
    return {
        "application": "Simple LLM API",
        "message": "running succesfully"
    }


@app.post('/chat')
async def generate_chat(request: Request):
    query = await request.json()
    model = query["model"]
    
    try:
        llm_response = generate_no_stream(
            query['question'], model
        )
        return {
            'status': 'success',
            'response': llm_response
        }
    
    except Exception as e:
        return {
            "error": str(e),
            "status_code": 400
        }


if __name__ == "__main__":
    import uvicorn
    print("Starting LLM API")
    uvicorn.run(app, host="0.0.0.0", reload=True)