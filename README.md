---
# RAG-Powered First-Aid Chatbot for Diabetes, Cardiac & Renal Emergencies
---

## Objectives

1. **Triage / Diagnosis**  
   Automatically infer the most likely **medical condition** based on the user's free-text symptom query.
2. **Hybrid Retrieval**  
   Retrieve the most relevant medical knowledge using a multi-source fusion strategy:
   - **Semantic Search:** Local embedding-based search over 60 curated medical knowledge base snippets.
   - **Keyword Search:** Surface results with overlapping medical terms to improve accuracy and redundancy handling.
   - **Web Search (Serper.dev):** Query the web for up-to-date medical advice or emergency guidelines.

   > Sign up for an API key at [Serper.dev](https://serper.dev)

   The results from all retrieval components are fused and ranked based on relevance.

3. **Answer Generation**  
   Produce a well-structured, actionable medical response within 250 words, including:
   - **Identified Condition**  
   - **First-Aid Steps** 
   - **Key Medication(s)**  
   - **Cited Sources** 

   The response always begins with a medical disclaimer and is optimized for clarity, safety, and practical relevance.

---

## Repository Structure

```
project/
│
├── src/
│   ├── main.py          # Core chatbot implementation
│   └── adb.xlsx         # Local medical knowledge base
│
├── tests/
│   └── test.py          # Unit tests for chatbot behavior
│
├── requirements.txt     # Required Python libraries
├── README.md            # Project documentation
├── architecture.md      # System architecture and design rationale
```

---

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/first-aid-chatbot.git
cd first-aid-chatbot
```

2. **Install dependencies**

```bash
pip install -r requirements.txt

```

3. **Set up API keys**

Create a `.env` file in the root directory with the following:

```
GROQ_API_KEY=your_groq_api_key
SERPER_API_KEY=your_serper_api_key
```

You can get these keys from:
- https://console.groq.com
- https://serper.dev

---

## How to Use

Run the chatbot interactively from the command line:

```bash
python src/main.py
```

Enter a medical query such as:

```
Crushing chest pain and pain in left arm. Should I take aspirin or call an ambulance?
```

The chatbot will classify the condition, retrieve relevant guidance, and return structured first-aid steps with sources.

---

## Input
Query (User Input)

## Output

Each response includes:
- A disclaimer for medical safety
- Assessment (likely diagnosis e.g. "Myocardial Infarction")
- Immediate first-aid steps
- Sources (references from local knowledge or web)

---
---

## Technical Stack

### Core Technologies

- **Language:** Python 3.8+
- **ML Frameworks:** scikit-learn, sentence-transformers
- **Data Processing:** pandas, numpy
- **HTTP Client:** requests
- **Environment Management:** python-dotenv

---

### External Dependencies

- **Groq API:** For LLM-based inference and reasoning
- **Serper API:** For web search and real-time evidence retrieval
- **SentenceTransformer:** For semantic similarity computation
- **Excel Data File (`adb.xlsx`):** Local medical knowledge base

---

### Model Specifications

- **Embedding Model:** `all-MiniLM-L6-v2` (384-dimensional sentence embeddings)
- **LLM Model:** `Llama3-70b-8192` hosted on Groq
- **Similarity Metric:** Cosine similarity
- **Vector Storage:** In-memory numpy arrays

---

## Security & Safety Features

### Medical Safety

- **Disclaimer Integration:** Every response begins with a medical safety disclaimer
- **Emergency Prioritization:** Automatically identifies critical symptoms and recommends emergency action
- **Source Verification:** Information is cross-checked across local and web sources

### API Security

- **Environment Variables:** API keys are securely managed using `.env` files

### Data Privacy

- **No Persistence:** User queries are not stored
- **Stateless Design:** Each query is processed independently
- **Local Processing:** All semantic computations are handled locally

---

## Dependencies

- **Python Environment:** Use of virtual environments is recommended
- **Package Management:** Install via `pip` using `requirements.txt`
- **API Access:** Requires valid keys for Groq and Serper APIs
- **Data Files:** `adb.xlsx` must be available in `src/` directory

---

## Running Tests

To run unit tests on critical cases:

```bash
pytest tests/test.py
```

Each test checks that the chatbot:
- Correctly identifies the condition and domain
- Includes first-aid steps and source references
- Maintains response structure and safety disclaimer

---

## Design Choices

- Uses a local Excel knowledge base (`adb.xlsx`) for reliable medical sentences.
- Classifies symptoms using a combination of:
  - Groq LLM (primary method)
  - Keyword fallback (if API fails)
- Retrieves relevant data using:
  - Sentence similarity
  - Keyword overlap
  - External search (Serper.dev)
- Generates responses using the same LLM with a clear template and medical safety constraints

For more, see [`architecture.md`](architecture.md).

---

## Disclaimer

This chatbot is a prototype. It does **not replace professional medical advice**. Always consult a licensed doctor or emergency service in real-life situations.
