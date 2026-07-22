-Murex Knowledge Assistant

A retrieval-augmented generation (RAG) assistant that answers questions from Murex project documentation — configuration notes, incident tickets,\
 and functional specs — instead of requiring manual searches through folders and tribal knowledge.

Built to explore how a Murex functional consultant's domain expertise can combine with modern AI tooling to solve a real, everyday pain point in trade lifecycle support and operations.

-What it does
Ingests a folder of internal documents (.txt files)
Splits them into overlapping chunks and converts each into an embedding (a numeric representation of its meaning)
Stores those embeddings in a local vector database (Chroma)
When asked a question, finds the most relevant chunks using similarity search - not keyword matching
Uses an LLM to generate an answer grounded only in the retrieved chunks, citing which source document(s) it used
Refuses to answer (rather than guessing) when nothing relevant is found, using a distance-based confidence threshold


-Why this matters
Trade support and operations teams lose significant time reconstructing tribal knowledge about past incidents and configuration decisions. This assistant captures that knowledge and makes it instantly queryable, while being explicit about the limits of what it actually knows - an important property for anything used in a finance/ops context, where a confidently wrong answer is worse than no answer at all.

-Tech stack
Python
Chroma (vector database)
DeepSeek API (LLM for answer generation)
Streamlit (chat interface)
Running it locally
Clone this repository
Install dependencies:
   pip install -r requirements.txt
Create a .env file in the project root with:
   DEEPSEEK_API_KEY=your_key_here
Add your .txt documents to the documents folder
Run the app:
   python -m streamlit run MXRag.py


-Project background
This project was built as a hands-on exploration of RAG architecture, starting from a single-document Q&A script and incrementally adding multi-document support,\
 real vector search, confidence-based filtering, and a usable chat interface. Retrieval quality and the confidence threshold were tuned empirically against a stress-test\
 set of realistic (synthetic) Murex documentation covering settlements, confirmations, credit checks, market data, corporate actions, and more.

Note: All documents used for testing are synthetic - written to be realistic based on real Murex domain knowledge, but containing no real client or production data.