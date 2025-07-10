# First-Aid Chatbot

## Description

This project is a medical assistant chatbot that helps users with emergency first-aid advice for diabetes, cardiac, and renal-related conditions. It uses:

- A local medical knowledge base
- A language model (LLM) for symptom classification
- A hybrid retrieval system to gather relevant information
- A response generator to produce structured, helpful replies

The goal is to provide fast, reliable first-aid suggestions based on user symptom queries.

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

## Example Outputs

Each response includes:
- A disclaimer for medical safety
- A likely diagnosis (e.g. "Myocardial Infarction")
- Urgency level (e.g. "Critical")
- Actionable first-aid instructions
- Referenced sources from local knowledge or web


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
