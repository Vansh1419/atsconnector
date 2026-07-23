"""
Complete Resume Normalization Pipeline (Function-Based + RapidFuzz)

Handles messy PDF/OCR extractions for:
1. Skills Normalization & Unstructured Skill Chunk Extraction
2. Role & Seniority Normalization
3. Degree & Level Normalization
4. Complex Date Parsing & Experience Calculation
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from rapidfuzz import process, fuzz


# ============================================================================
# TAXONOMIES
# ============================================================================

DEFAULT_SKILL_TAXONOMY = {
    "programming_languages": {
        "python": ["py", "python", "python3", "python 3"],
        "javascript": ["js", "javascript", "ecmascript", "es6", "es2015"],
        "typescript": ["ts", "typescript"],
        "java": ["java", "j2ee"],
        "c++": ["cpp", "c++", "cplusplus", "c plus plus"],
        "c#": ["csharp", "c#", "c-sharp"],
        "php": ["php", "php7", "php8"],
        "ruby": ["ruby", "rails"],
        "go": ["go", "golang"],
        "rust": ["rust"],
        "sql": ["sql", "tsql", "plsql", "sql server"],
    },
    "frameworks": {
        "react": ["react", "reactjs", "react.js"],
        "react native": ["react native", "react-native", "rn"],
        "vue": ["vue", "vuejs"],
        "angular": ["angular", "angular.js", "angularjs"],
        "django": ["django"],
        "flask": ["flask"],
        "fastapi": ["fastapi"],
        "spring": ["spring", "spring boot"],
        "express": ["express", "express.js", "expressjs"],
        "nest.js": ["nest", "nestjs", "nest.js"],
        "next.js": ["next", "nextjs", "next.js"],
        "node.js": ["node", "nodejs", "node.js"],
        "langchain": ["langchain"],
        "langgraph": ["langgraph"],
        "redux": ["redux"],
        "websockets": ["websockets", "websocket"],
    },
    "databases": {
        "postgresql": ["postgres", "postgresql", "psql"],
        "mysql": ["mysql", "mysql5"],
        "mongodb": ["mongo", "mongodb"],
        "redis": ["redis"],
        "elasticsearch": ["elasticsearch", "elastic"],
        "dynamodb": ["dynamodb"],
        "firebase": ["firebase"],
    },
    "tools_and_devops": {
        "docker": ["docker", "containerization"],
        "kubernetes": ["k8s", "kubernetes", "k8"],
        "aws": ["aws", "amazon web services"],
        "gcp": ["gcp", "google cloud"],
        "azure": ["azure", "microsoft azure"],
        "git": ["git", "github", "gitlab", "bitbucket"],
        "ci_cd": ["ci/cd", "ci-cd", "continuous integration", "github actions"],
        "vercel": ["vercel"],
        "n8n": ["n8n"],
        "postman": ["postman"],
        "figma": ["figma"],
    },
    "ai_and_robotics": {
        "rag": ["rag", "retrieval augmented generation"],
        "llm": ["llm", "llm prompting", "large language models"],
        "mavlink": ["mavlink"],
        "qgroundcontrol": ["qgroundcontrol", "qgc"],
        "photogrammetry": ["photogrammetry"],
        "geospatial": ["geospatial", "mapbox", "rnmapbox"],
        "claude code": ["claude code"],
    }
}

DEFAULT_ROLE_TAXONOMY = {
    "frontend": {
        "canonical": "Frontend Engineer",
        "variants": ["frontend engineer", "frontend developer", "ui developer", "react developer", "front end engineer", "fe engineer", "ui/ux developer", "web frontend engineer"]
    },
    "backend": {
        "canonical": "Backend Engineer",
        "variants": ["backend engineer", "backend developer", "server-side developer", "api developer", "database developer", "sde"]
    },
    "fullstack": {
        "canonical": "Full Stack Engineer",
        "variants": ["full stack engineer", "full stack developer", "fullstack engineer", "full-stack developer"]
    },
    "devops": {
        "canonical": "DevOps Engineer",
        "variants": ["devops engineer", "infrastructure engineer", "sre", "site reliability engineer", "cloud engineer"]
    },
    "data_ai": {
        "canonical": "AI / Data Engineer",
        "variants": ["data engineer", "data scientist", "ml engineer", "machine learning engineer", "ai engineer", "ai & automation"]
    },
    "embedded_robotics": {
        "canonical": "Robotics / Firmware Engineer",
        "variants": ["robotics engineer", "firmware engineer", "fde", "field engineer", "hardware engineer"]
    },
    "intern": {
        "canonical": "Software Engineering Intern",
        "variants": ["software engineering intern", "sde intern", "intern", "developer intern", "trainee"]
    }
}

DEFAULT_DEGREE_TAXONOMY = {
    "high_school": {"canonical": "High School", "level": 0, "variants": ["high school", "hs", "12th grade", "secondary school", "board of secondary"]},
    "associate": {"canonical": "Associate's Degree", "level": 1, "variants": ["associate", "associates degree", "aa", "as"]},
    "bachelor": {"canonical": "Bachelor's Degree", "level": 2, "variants": ["bachelor", "bachelors", "b.tech", "btech", "b.e.", "be", "bsc", "b.a.", "b.s.", "bachelor of technology", "bachelor of science"]},
    "master": {"canonical": "Master's Degree", "level": 3, "variants": ["master", "masters", "m.tech", "mtech", "msc", "m.a.", "m.s.", "master of technology", "master of science"]},
    "phd": {"canonical": "PhD", "level": 4, "variants": ["phd", "ph.d.", "doctorate", "doctoral degree"]},
}

DATE_FORMATS = ["%b %Y", "%B %Y", "%m/%Y", "%Y-%m", "%Y", "%d %b %Y", "%d/%m/%Y", "%m/%d/%Y"]
CURRENT_KEYWORDS = ["present", "current", "now", "ongoing"]


# ============================================================================
# LOOKUP BUILDERS
# ============================================================================

def _build_skill_lookup(taxonomy: dict) -> Dict[str, Tuple[str, str, List[str]]]:
    lookup = {}
    for cat, skills in taxonomy.items():
        for canonical, variants in skills.items():
            all_vars = sorted(list(set([canonical] + variants)))
            lookup[canonical.lower()] = (canonical, cat, all_vars)
            for v in variants:
                lookup[v.lower()] = (canonical, cat, all_vars)
    return lookup

def _build_role_lookup(taxonomy: dict) -> Dict[str, Tuple[str, str]]:
    lookup = {}
    for cat, data in taxonomy.items():
        canonical = data["canonical"]
        lookup[canonical.lower()] = (canonical, cat)
        for v in data["variants"]:
            lookup[v.lower()] = (canonical, cat)
    return lookup

def _build_degree_lookup(taxonomy: dict) -> Dict[str, Tuple[str, int]]:
    lookup = {}
    for key, data in taxonomy.items():
        canonical = data["canonical"]
        level = data["level"]
        lookup[canonical.lower()] = (canonical, level)
        for v in data["variants"]:
            lookup[v.lower()] = (canonical, level)
    return lookup

SKILL_LOOKUP = _build_skill_lookup(DEFAULT_SKILL_TAXONOMY)
ROLE_LOOKUP = _build_role_lookup(DEFAULT_ROLE_TAXONOMY)
DEGREE_LOOKUP = _build_degree_lookup(DEFAULT_DEGREE_TAXONOMY)


# ============================================================================
# FUZZY NORMALIZATION FUNCTIONS (RapidFuzz Enabled)
# ============================================================================

def normalize_skill(skill: str, score_cutoff: float = 82.0) -> Optional[str]:
    """Fuzzy matching for individual skill strings."""
    if not skill or not isinstance(skill, str):
        return None
    cleaned = re.sub(r'\s*(developer|engineer|specialist)$', '', skill.strip().lower())
    
    # 1. Exact lookup
    if cleaned in SKILL_LOOKUP:
        return SKILL_LOOKUP[cleaned][0]
    
    # 2. Fuzzy match via RapidFuzz
    match = process.extractOne(cleaned, SKILL_LOOKUP.keys(), scorer=fuzz.token_sort_ratio, score_cutoff=score_cutoff)
    if match:
        return SKILL_LOOKUP[match[0]][0]
    return None


def normalize_skills(raw_input: Any) -> List[Dict[str, Any]]:
    """Handles both list of skills and unstructured string blocks with PDF formatting noise."""
    tokens = []
    if isinstance(raw_input, str):
        # Extract skills separated by bullets (●), commas, newlines, colons, or slashes
        raw_tokens = re.split(r'[\n●•:,/|\\]+', raw_input)
        for token in raw_tokens:
            sub_tokens = re.split(r'--', token)
            tokens.extend(sub_tokens)
    elif isinstance(raw_input, list):
        tokens = raw_input

    normalized = []
    seen = set()

    for token in tokens:
        clean_token = token.strip()
        if not clean_token or len(clean_token) < 2:
            continue
        
        canonical = normalize_skill(clean_token)
        if canonical and canonical not in seen:
            seen.add(canonical)
            _, category, variants = SKILL_LOOKUP.get(canonical.lower(), (canonical, "other", [canonical]))
            normalized.append({
                "canonical_name": canonical,
                "category": category,
                "variants": variants
            })
    return normalized


def normalize_role(title: str, score_cutoff: float = 75.0) -> Dict[str, Any]:
    """Fuzzy matches job roles against taxonomy and extracts level."""
    if not title:
        return {"canonical_name": None, "category": None, "level": "mid"}
    
    cleaned = title.strip().lower()
    canonical, category = None, None

    # 1. Exact Match
    if cleaned in ROLE_LOOKUP:
        canonical, category = ROLE_LOOKUP[cleaned]
    else:
        # 2. RapidFuzz Partial Match
        match = process.extractOne(cleaned, ROLE_LOOKUP.keys(), scorer=fuzz.partial_ratio, score_cutoff=score_cutoff)
        if match:
            canonical, category = ROLE_LOOKUP[match[0]]

    # Determine Seniority
    if any(w in cleaned for w in ['junior', 'jr', 'associate', 'intern', 'trainee']):
        level = 'junior'
    elif any(w in cleaned for w in ['senior', 'sr', 'principal', 'staff']):
        level = 'senior'
    elif any(w in cleaned for w in ['lead', 'manager', 'head', 'director']):
        level = 'lead'
    else:
        level = 'mid'

    return {"canonical_name": canonical, "category": category, "level": level}


def normalize_degree(degree_str: str, score_cutoff: float = 80.0) -> Dict[str, Any]:
    """Fuzzy matches academic degrees against degree taxonomy."""
    if not degree_str:
        return {"canonical_name": None, "level": -1}

    cleaned = degree_str.strip().lower()

    # Direct match
    if cleaned in DEGREE_LOOKUP:
        canonical, level = DEGREE_LOOKUP[cleaned]
        return {"canonical_name": canonical, "level": level}

    # Fuzzy match using RapidFuzz
    match = process.extractOne(cleaned, DEGREE_LOOKUP.keys(), scorer=fuzz.partial_ratio, score_cutoff=score_cutoff)
    if match:
        canonical, level = DEGREE_LOOKUP[match[0]]
        return {"canonical_name": canonical, "level": level}

    return {"canonical_name": None, "level": -1}


# ============================================================================
# DATE & EXPERIENCE PARSER
# ============================================================================

def parse_date(date_str: str) -> Optional[datetime]:
    if not date_str:
        return None
    cleaned = date_str.strip().lower()
    if any(k in cleaned for k in CURRENT_KEYWORDS):
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def parse_date_range(text: str) -> Dict[str, Any]:
    """Extracts date ranges from multiline blocks or structured text."""
    # Find patterns like "Oct 2021 – Jun 2025" or "July 2025 – Present" or "2021 – 2025"
    date_pattern = r'([A-Za-z]{3,9}\s+\d{4}|\d{4})\s*[\–\-–—]\s*([A-Za-z]{3,9}\s+\d{4}|\d{4}|Present|Current|Now)'
    match = re.search(date_pattern, text, re.IGNORECASE)
    
    if not match:
        return {"start_date": None, "end_date": None, "is_current": False, "duration_months": 0}

    start_str, end_str = match.group(1), match.group(2)
    start_date = parse_date(start_str)
    
    is_current = any(k in end_str.lower() for k in CURRENT_KEYWORDS)
    end_date = datetime.now() if is_current else parse_date(end_str)

    duration_months = 0
    if start_date and end_date:
        duration_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

    return {
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "is_current": is_current,
        "duration_months": max(0, duration_months)
    }


# ============================================================================
# MAIN BULK PIPELINE FUNCTION
# ============================================================================

def normalize_resume(resume: Dict[str, Any]) -> Dict[str, Any]:
    """Pipeline function to normalize a single resume dictionary."""
    
    # 1. Normalize Personal Details
    file_name = resume.get("file_name", "")
    name = resume.get("name", "")
    email = resume.get("email", "")
    phone = resume.get("phone", "")

    # 2. Normalize Skills
    normalized_skills = normalize_skills(resume.get("skills", ""))

    # 3. Normalize Education
    raw_edu = resume.get("education", "")
    degree_info = normalize_degree(raw_edu)
    edu_dates = parse_date_range(raw_edu)
    
    education_entry = {
        "raw_text": raw_edu,
        "degree": degree_info["canonical_name"],
        "degree_level": degree_info["level"],
        "start_date": edu_dates["start_date"],
        "graduation_date": edu_dates["end_date"],
    }

    # 4. Normalize Experience
    raw_exp = resume.get("experience", "")
    
    # Splitting unstructured experience block by job segments
    exp_entries = []
    total_experience_months = 0

    if isinstance(raw_exp, str):
        # Split jobs on line breaks preceding common title/date patterns
        jobs = re.split(r'\n(?=[A-Z][A-Za-z0-9\s\.\,]+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|20\d\d))', raw_exp)
        
        for job in jobs:
            if not job.strip():
                continue
            
            # Match role and dates
            role_info = normalize_role(job)
            dates = parse_date_range(job)
            job_skills = normalize_skills(job)

            total_experience_months += dates["duration_months"]

            exp_entries.append({
                "raw_text": job.strip(),
                "normalized_title": role_info["canonical_name"],
                "title_category": role_info["category"],
                "title_level": role_info["level"],
                "start_date": dates["start_date"],
                "end_date": dates["end_date"],
                "is_current": dates["is_current"],
                "duration_months": dates["duration_months"],
                "duration_years": round(dates["duration_months"] / 12, 1),
                "extracted_skills": [s["canonical_name"] for s in job_skills]
            })

    return {
        "file_name": file_name,
        "name": name,
        "email": email,
        "phone": phone,
        "education": [education_entry],
        "experience": exp_entries,
        "skills": normalized_skills,
        "total_experience_years": round(total_experience_months / 12, 1)
    }


def normalize_bulk_resumes(bulk_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Runs batch normalization on a list of extracted resume dicts."""
    return [normalize_resume(resume) for resume in bulk_data]