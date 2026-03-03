import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What is RAG?", "top_k": 5}
)
report = response.json()
print(report["answer"])
print(report["citations"])
print(report["graph_execution"]["has_cycle"])