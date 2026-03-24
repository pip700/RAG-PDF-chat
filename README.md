# RAG-PDF-chat

1. sudo docker build -t pdf-rag-agent .

2. sudo docker run --rm -d -p 8501:8501 --env-file .env --name rag-app pdf-rag-agent

3. Go to web-browser and open link "http:localhost:8501"
