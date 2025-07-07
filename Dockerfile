# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set work directory inside the container
WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "spectre_web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
