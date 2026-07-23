import os
import re
import pdfplumber

def parse_pdf(path:str):
    """
    parse each pdf in simple text format
    """
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text


def extract_section(text, section_name, next_sections):
    pattern = rf"{section_name}\s*[:\n](.*?)(?=(?:{'|'.join(next_sections)})|$)"
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_resume(path):
    text = parse_pdf(path)

    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
    phone_match = re.search(r"(\+?\d[\d\s\-\(\)]{8,}\d)", text)

    sections = ["skills", "education", "experience", "projects", "certifications"]

    return {
        "file_name": os.path.basename(path),
        "name": text.strip().split("\n")[0].strip(),
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
        "skills": extract_section(text, "skills", sections),
        "education": extract_section(text, "education", sections),
        "experience": extract_section(text, "experience", sections),
    }


def parse_resumes_in_folder(folder_path):
    results = []
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            results.append(parse_resume(file_path))
    return results