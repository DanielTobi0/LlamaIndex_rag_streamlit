# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Define environment variable
ENV STREAMLIT_SERVER_PORT=8501

# Run both FastAPI and Streamlit when the container launches    
# CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8505 & streamlit run streamlit_app.py"]
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port 8505 & streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501"]
