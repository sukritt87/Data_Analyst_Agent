# AI Data Analyst Agent (Google ADK)

## 📌 Project Overview

This project builds a **tool-augmented AI Data Analyst Agent** that can analyze small datasets (<1000 rows) using natural language queries.

Instead of relying entirely on an LLM, the system uses **modular analytical tools** for:

* Data inspection
* Exploratory Data Analysis (EDA)
* Visualization
* Machine Learning (Regression & Classification)

The LLM (Gemini) is used for:

* Understanding user intent
* Selecting tools
* Explaining results in natural language

---

## 🧠 Architecture

User Query → Agent (Google ADK) → Tool Selection → Tool Execution → LLM Explanation → Output

---

## 🛠️ Tech Stack

* Python 3.12+
* Pandas, NumPy
* Matplotlib
* Scikit-learn
* Google Gemini API (for LLM)
* Python-dotenv

---

## 📁 Project Structure

```
data-analyst-agent-adk/
│
├── app/                  # Main application logic
├── tools/                # Tool modules (EDA, ML, Visualization)
├── data/                 # Sample datasets
├── charts/               # Generated plots
├── requirements.txt
├── .env                  # API keys (not tracked in Git)
└── README.md
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/sukritt87/Data_Analyst_Agent.git
cd data-analyst-agent-adk
```

---

### 2. Activate Virtual Environment

If using the shared `.venv` in parent folder:

```powershell
& "..\.venv\Scripts\Activate.ps1"
```

OR if you created your own:

```powershell
.\.venv\Scripts\Activate.ps1
```

---

### 3. Install Dependencies

```powershell
python -m pip install -r requirements.txt
```

---

### 4. Create `.env` File

Create a file in the root folder:

```
.env
```

Add your Google API key:

```
GOOGLE_API_KEY=your_api_key_here
```

⚠️ Do NOT commit this file to GitHub.

---

## 🔑 How to Get Google API Key

1. Go to Google AI Studio
2. Generate API Key
3. Copy and paste into `.env` with this line GOOGLE_API_KEY=your_api_key

---

## ▶️ Running the Project

Run from project root:

```powershell
python -m app.main
```

---

## 🧪 Example Usage

* Load dataset
* Ask questions like:

  * "What columns are in this dataset?"
  * "Which region has highest sales?"
  * "Show correlation between variables"
  * "Train a regression model"

---

## 📌 Contribution Guidelines

* Each member should work inside the `tools/` directory
* Follow modular design (one tool per file)
* Do not hardcode paths
* Test tools independently before integration

---

## ⚠️ Common Issues

### 1. Python not found

Make sure Python is installed and added to PATH.

### 2. Virtual environment issues

Always activate `.venv` before running commands.

### 3. Import errors

Run project using:

```
python -m app.main
```

NOT:

```
python app/main.py
```

---

## 📈 Future Enhancements

* Full Google ADK integration
* Streamlit UI
* Advanced ML models
* Multi-dataset support

---

## 👥 Team Collaboration Notes

* Keep commits small and modular
* Push only working code
* Do not modify others’ tools without discussion

---

## 📄 License

For academic use only.
