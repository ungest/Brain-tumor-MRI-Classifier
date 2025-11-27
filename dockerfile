# Use a lightweight Python image
FROM python:3.11-slim

# Set work directory inside the container
WORKDIR /app

# Install system deps if you need them (uncomment if PyTorch etc. complain)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#  && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (better for caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY streamlit_app ./streamlit_app
COPY Model_Evaluation.png ./Model_Evaluation.png
COPY README.md ./README.md


# Expose Streamlit's default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "streamlit_app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
