from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from simple import process_query
import uvicorn

app = FastAPI()

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
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
