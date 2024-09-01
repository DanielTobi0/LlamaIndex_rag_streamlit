from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from simple import process_query
import traceback
from exceptions.operations_handler import system_logger

app = FastAPI()

'''
@app.get('/')
async def health():
    return {
        "application": "Simple LLM API",
        "message": "running succesfully"
    }
'''

@app.post('/chat')
async def generate_chat(request: Request):
    try:
        user_input = await request.json()
        query = user_input.get("query")
        model = user_input.get("model")
        temperature = user_input.get("temperature")
        
        if not query or not model or not temperature:
            raise HTTPException(status_code=400, detail="Missing parameters")
        
        llm_response = process_query(query=query, model=model, temperature=temperature)
        return JSONResponse(content={"status": "success", "response": llm_response}, status_code=200)

        
    except Exception as e:
        system_logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("Starting LLM API")
    uvicorn.run(app, host="0.0.0.0", reload=True)