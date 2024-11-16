# 1. Use a base image with Python installed
FROM python:3.12-slim

# 2. Set a working directory inside the container
WORKDIR /app

# 3. Copy your requirements.txt (or environment.yml) to install dependencies
COPY requirements.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK resources
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords punkt_tab

# 5. Copy the rest of your application code
COPY . .

# 6. Expose a port if your app runs on a web server (e.g., for Flask/ FastAPI)
EXPOSE 5000

# 7. Define the command to run your application (modify based on your app)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]