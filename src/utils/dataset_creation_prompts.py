"""LLM prompts to be used for creating the dataset"""

import json
import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")

with open(os.path.join(PROJECT_ROOT), "resume_json_schema.json") as f:
    json_schema = json.load(f)

SYSTEM_PROMPT = f"""You are an expert in converting CVs to JSON format based on the following JSON schema: {json_schema}

##### RULES:
- Always respond with a valid JSON; do not provide any extra information.
- 
"""

EXAMPLE_1 = """"""
RESPONSE_1 = {
  "personal_information": {
    "full_name": "Jessica Martinez",
    "title": "Community Outreach Coordinator",
    "email": "jessica.martinez@email.com",
    "phone": "+1-555-123-4567",
    "location": {
      "city": "Denver",
      "state": "CO",
      "country": "USA",
      "postal_code": "80203"
    },
    "linkedin_url": "https://www.linkedin.com/in/jessicamartinez",
    "website_or_portfolio": "https://jessicamartinezportfolio.com",
    "github_url": ""
  },
  "about_info": "A dedicated, results-driven, and resourceful professional with a strong focus on building positive community relationships and fostering program growth. Possesses excellent organizational, problem-solving, and interpersonal communication skills, supported by over 10 years of experience in client relations and team leadership. A passionate advocate for creating accessible educational opportunities and supportive community networks, adept at managing multiple projects and engaging effectively with diverse populations.",
  "work_experience": [
    {
      "job_title": "Substitute Teacher",
      "company": "Denver Public Schools",
      "employment_type": "Part-time",
      "location": "Denver, CO",
      "start_date": "2016-08",
      "end_date": "present",
      "achievements": [
        "Executed lesson plans provided by the regular teacher to ensure continuity of learning across various subjects and grade levels.",
        "Adapted teaching methods to meet the needs of diverse student populations, including those with special needs.",
        "Maintained a productive, safe, and respectful classroom environment through effective classroom management techniques.",
        "Communicated with faculty and administrators regarding student progress and classroom issues.",
        "Provided clear instruction, graded assignments, and offered constructive feedback to students."
      ]
    },
    {
      "job_title": "Community Outreach Assistant",
      "company": "Colorado Community Health Network",
      "employment_type": "Full-time",
      "location": "Denver, CO",
      "start_date": "2014-06",
      "end_date": "2016-07",
      "achievements": [
        "Developed and maintained positive relationships with community leaders, local businesses, and partner organizations.",
        "Coordinated and promoted community events, workshops, and health fairs to increase program visibility and participation.",
        "Served as the primary point of contact for public inquiries, providing information and resources effectively.",
        "Managed the organization's social media accounts and created content for monthly newsletters.",
        "Assisted in data collection and reporting to track program impact and identify areas for improvement."
      ]
    },
    {
      "job_title": "Client Relations Manager",
      "company": "BrightPath Solutions",
      "employment_type": "Full-time",
      "location": "Boulder, CO",
      "start_date": "2010-04",
      "end_date": "2014-05",
      "achievements": [
        "Managed a portfolio of over 50 key client accounts, ensuring high levels of satisfaction and retention.",
        "Acted as the main liaison between clients and internal teams to guarantee the timely and successful delivery of solutions.",
        "Analyzed client feedback and performance data to develop strategies for service improvement.",
        "Trained and supervised a team of 5 customer service representatives, focusing on performance and professional development.",
        "Resolved complex client issues and escalations with diplomacy and efficiency."
      ]
    },
    {
      "job_title": "Customer Service Supervisor",
      "company": "Summit Financial Services",
      "employment_type": "Full-time",
      "location": "Aurora, CO",
      "start_date": "2006-09",
      "end_date": "2010-04",
      "achievements": [
        "Supervised the daily operations of a 15-person customer service team.",
        "Monitored call metrics and individual performance to ensure quality and productivity standards were met.",
        "Addressed and resolved escalated customer complaints, turning negative experiences into positive outcomes.",
        "Created employee work schedules and managed shift assignments.",
        "Developed and delivered training on new products and customer service protocols."
      ]
    }
  ],
  "education": [
    {
      "degree": "Bachelor of Arts",
      "field_of_study": "Communications (Minor in Business Administration)",
      "institution": "University of Colorado Boulder",
      "location": "Boulder, CO",
      "start_date": "2006-08",
      "end_date": "2010-05",
      "grade_or_gpa": "3.7"
    },
    {
      "degree": "Associate of Arts",
      "field_of_study": "Liberal Arts",
      "institution": "Front Range Community College",
      "location": "Westminster, CO",
      "start_date": "2004-08",
      "end_date": "2007-05",
      "grade_or_gpa": "3.8"
    }
  ],
  "certifications": [],
  "skills": {
    "hard_skills": [
      "Strategic Planning",
      "Event Planning & Execution",
      "Project Management",
      "Grant Writing Basics",
      "Client Relationship Management (CRM) Software",
      "Social Media Management",
      "Data Entry",
      "Microsoft Office Suite",
      "Reporting"
    ],
    "soft_skills": [
      "Public Speaking",
      "Team Leadership",
      "Relationship Management",
      "Conflict Resolution",
      "Mentorship",
      "Written Communication",
      "Scheduling",
      "Customer Service",
      "Community Outreach"
    ],
    "languages": [
      {
        "language": "English",
        "proficiency": "Native or Bilingual"
      },
      {
        "language": "Spanish",
        "proficiency": "Professional Working Proficiency"
      }
    ],
    "tools_and_technologies": [
      "Salesforce",
      "Microsoft Office Suite",
      "Facebook",
      "Twitter",
      "LinkedIn"
    ]
  },
  "projects": [],
  "publications": [],
  "awards": [
    {
      "title": "Employee of the Year",
      "issuer": "BrightPath Solutions",
      "date": "2013-12",
      "description": "Awarded for outstanding client retention and team leadership."
    },
    {
      "title": "Dean's List Recognition",
      "issuer": "University of Colorado Boulder",
      "date": "2008-2010",
      "description": "Consistently named to the Dean's List during undergraduate studies."
    }
  ],
  "volunteer_experience": [],
  "preferences": {
    "desired_job_titles": ["Community Outreach Coordinator"],
    "desired_locations": ["Denver, CO", "Boulder, CO"],
    "remote_work": True,
    "salary_expectations": {
      "currency": "USD",
      "minimum": 60000,
      "maximum": 75000,
      "period": "year"
    },
    "notice_period": "2 weeks"
  }
}

EXAMPLE_2 = """"""
RESPONSE_2 = {}

