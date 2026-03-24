# RAG-PDF-chat
sudo docker build -t pdf-rag-agent .
sudo docker run --rm -d -p 8501:8501 --env-file .env --name rag-app pdf-rag-agent


Go to web-browser and open link "http:localhost:8501"
