# Automated-Job-Search-AI-Agent

An AI-powered job automation system that extracts and analyzes resume data, evaluates job-role suitability, and automatically navigates the National Career Service (NCS) portal using Playwright to identify and filter the most relevant job postings. The system integrates automation, AI-driven text analysis, and web scraping to streamline the job search process and deliver results through a structured, interactive Streamlit dashboard.

---
## Features

• AI Resume Parsing  
  - Extracts text from your resume (PDF format)  
  - Uses LLMs (OpenAI / Google GenAI) to analyze skills, keywords, and experience  
  - Generates structured summaries for job matching  

• Automated Job Search on NCS  
  - Uses Playwright to automate browser actions  
  - Logs into your NCS account  
  - Applies filters such as sector, qualification, job type, and location  
  - Scrapes and stores relevant job listings  

• Interactive Dashboard  
  - Built using Streamlit  
  - Displays all job results with filters  
  - Provides a clean interface to explore insights  

• Generative AI Integration  
  - LangChain integration with OpenAI and Google GenAI  
  - AI-based role suitability estimation  
  - Optional cover letter generation  

---

## Tech Stack

| Component | Technology |
|----------|------------|
| AI Models | OpenAI GPT, Google GenAI, LangChain |
| Automation | Playwright |
| Backend | Python |
| Dashboard | Streamlit |
| Parsing | PyPDF2 |
| Data Storage | CSV, Pandas |

---

## Project Structure

JobAutomationPortal/
│-- app_ncs_v2.py # Main automation script
│-- dashboard.py # Streamlit dashboard
│-- summarize.py # Resume summarization logic
│-- read_pdf.py # Resume text extraction
│-- cover_letter_generator.py # Optional cover letter generator
│-- requirements.txt # Dependencies
│-- .env (ignored) # API keys and credentials
│-- ncs_job_results.csv (generated)
│-- venv/ (ignored)


---

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/YeshwanthRajSelvaraj/Automated-Job-Search-AI-Agent.git
cd Automated-Job-Search-AI-Agent


### 2. Create a Virtual Environment

python -m venv venv
venv/Scripts/activate # Windows
source venv/bin/activate # MacOS/Linux


### 3. Install Dependencies

pip install -r requirements.txt
pip install streamlit pandas
python -m playwright install


### 4. Configure API Keys

Create a `.env` file:

OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key


### 5. Run the Automation Script

python app_ncs_v2.py


### 6. Launch the Dashboard

streamlit run dashboard.py


---

## Output

• AI-analyzed summary of your resume  
• Job listings scraped from the NCS portal  
• Skill–role match estimation  
• CSV export for all job results  
• Interactive exploration using the dashboard  

---

## Security Note

The project handles personal information such as:  
• API keys  
• Resume data  
• Login credentials  

Ensure that `.env` and personal documents are not pushed to GitHub.

---

## Contributing

Pull requests are welcome.  
For major changes, please open an issue beforehand for discussion.

---

## License

This project is licensed under the MIT License.

