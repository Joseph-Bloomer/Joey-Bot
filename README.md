# Joey-Bot
A custom LLM wrapper built with Flask, running local models via Ollama.

## Features
* **Local Intelligence:** Powered by `gemma3:4b` running on Ollama.
* **Persistent Memory:** Uses a local JSON-based semantic memory for context.
* **Web Interface:** Custom UI built with HTML, CSS, and JavaScript.

## Setup
1. **Install Ollama:** Download from [ollama.com](https://ollama.com).
2. **Pull the Model:** Run `ollama pull gemma3:4b` in your terminal.
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
4. Run the App:
  Bash
  python app.py

Project Structure
/static: UI styling and logic.
/templates: HTML layout.
/instance: Local data (ignored by Git).
