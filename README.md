Required Python Libraries/Models
1) pip install ollama
2) pip install streamlit
3) ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
4) ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
5) ollama pull 'hf.co/bartowski/Llama-3.2-3B-Instruct-GGUF'  #llama3.2 (LLM)
5) pip install faiss-cpu


pip install --upgrade -r requirements.txt
pip download -r requirements.txt -d packages/

Test_feedback.py:
============
Have a feedback loop. Lean from Json is not working. Need to fix. 


Test_rest.py:
============
Copy of Test_feedback.py:
Server run: uvicorn Test_rest:app --host 0.0.0.0 --port 8000 --reload 
uvicorn Test_rest:app --host 0.0.0.0 --port 8000 --reload --log-level info


Test_rest_session.py:
============
Copy of Test_feedback.py:
Server run: uvicorn Test_rest:app --host 0.0.0.0 --port 8000 --reload 
uvicorn Test_rest_session:app --host 0.0.0.0 --port 8000 --reload --log-level info
+ Session Management

Test_rest_session-strict.py:
============
Copy of Test_feedback.py:
uvicorn Test_rest_session-strict:app --host 0.0.0.0 --port 8000 --reload --log-level info
+ Session Management
