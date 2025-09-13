"""LLM prompts to be used for creating the dataset"""

import json
import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT")

with open(os.path.join(PROJECT_ROOT, "resume_json_schema.json"), "r") as f:
    json_schema = json.load(f)

SYSTEM_PROMPT = f"""You are an expert in extracting information from CVs to JSON format based on the following JSON schema: {json.dumps(json_schema)}

##### RULES:
- Always respond with a valid JSON; do not provide any extra information.
- Missing informations should be noted either as "" or [], depending on the JSON schema.
"""

EXAMPLE_1 = """COMMUNITY OUTREACH COORDINATOR

Professional Summary

A dedicated, results-driven, and resourceful professional with a strong focus on building positive community relationships and fostering program growth. Possesses excellent organizational, problem-solving, and interpersonal communication skills, supported by over 10 years of experience in client relations and team leadership. A passionate advocate for creating accessible educational opportunities and supportive community networks, adept at managing multiple projects and engaging effectively with diverse populations.

Core Qualifications

Strategic Planning

Public Speaking & Presentations

Relationship & Team Management

Event Planning & Execution

Conflict Resolution

Client Relationship Management (CRM) Software

Project Management

Grant Writing Basics

Bilingual (English/Spanish)

Maintains Strict Confidentiality

Accomplishments

Received "Employee of the Year" award for outstanding client retention and team leadership at BrightPath Solutions.

Consistently named to the Dean's List during undergraduate studies.

Selected to train and mentor new team members on company protocols and effective client communication strategies.

Experience

Substitute Teacher | Aug 2016 to Current
Denver Public Schools – Denver, CO

Execute lesson plans provided by the regular teacher to ensure continuity of learning across various subjects and grade levels.

Adapt teaching methods to meet the needs of diverse student populations, including those with special needs.

Maintain a productive, safe, and respectful classroom environment through effective classroom management techniques.

Communicate with faculty and administrators regarding student progress and any classroom issues.

Provide clear instruction, grade assignments, and offer constructive feedback to students.

Community Outreach Assistant | Jun 2014 to Jul 2016
Colorado Community Health Network – Denver, CO

Developed and maintained positive relationships with community leaders, local businesses, and partner organizations.

Coordinated and promoted community events, workshops, and health fairs to increase program visibility and participation.

Served as the primary point of contact for public inquiries, providing information and resources effectively.

Managed the organization's social media accounts and created content for monthly newsletters.

Assisted in data collection and reporting to track program impact and identify areas for improvement.

Client Relations Manager | Apr 2010 to May 2014
BrightPath Solutions – Boulder, CO

Managed a portfolio of over 50 key client accounts, ensuring high levels of satisfaction and retention.

Acted as the main liaison between clients and internal teams to guarantee the timely and successful delivery of our solutions.

Analyzed client feedback and performance data to develop strategies for service improvement.

Trained and supervised a team of 5 customer service representatives, focusing on performance and professional development.

Resolved complex client issues and escalations with diplomacy and efficiency.

Customer Service Supervisor | Sep 2006 to Apr 2010
Summit Financial Services – Aurora, CO

Supervised the daily operations of a 15-person customer service team.

Monitored call metrics and individual performance to ensure quality and productivity standards were met.

Addressed and resolved escalated customer complaints, turning negative experiences into positive outcomes.

Created employee work schedules and managed shift assignments.

Developed and delivered training on new products and customer service protocols.

Education

Bachelor of Arts, Communications (Minor in Business Administration) | 2010
University of Colorado Boulder – Boulder, CO

Associate of Arts, Liberal Arts | 2007
Front Range Community College – Westminster, CO

Skills
Public Speaking, Client Relations, Team Leadership, Event Planning, Project Management, CRM Software, Social Media Management, Conflict Resolution, Mentorship, Scheduling, Reporting, Customer Service, Microsoft Office Suite, Data Entry, Written Communication, Community Outreach, Teaching Support.
"""

RESPONSE_1 = {
  "personal_information": {
    "full_name": "",
    "title": "Community Outreach Coordinator",
    "email": "",
    "phone": "",
    "location": {
      "city": "",
      "state": "",
      "country": "",
    },
    "linkedin_url": "",
    "website_or_portfolio": "",
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
        "Execute lesson plans provided by the regular teacher to ensure continuity of learning across various subjects and grade levels.",
        "Adapt teaching methods to meet the needs of diverse student populations, including those with special needs.",
        "Maintain a productive, safe, and respectful classroom environment through effective classroom management techniques.",
        "Communicate with faculty and administrators regarding student progress and any classroom issues.",
        "Provide clear instruction, grade assignments, and offer constructive feedback to students."
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
        "Acted as the main liaison between clients and internal teams to guarantee the timely and successful delivery of our solutions.",
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
      "start_date": "",
      "end_date": "2010-01",
      "grade_or_gpa": ""
    },
    {
      "degree": "Associate of Arts",
      "field_of_study": "Liberal Arts",
      "institution": "Front Range Community College",
      "location": "Westminster, CO",
      "start_date": "",
      "end_date": "2007-05",
      "grade_or_gpa": ""
    }
  ],
  "certifications": [],
  "skills": {
    "hard_skills": [
      "Event Planning",
      "Project Management",
      "CRM Software",
      "Social Media Management",
      "Scheduling",
      "Reporting",
      "Microsoft Office Suite",
      "Data Entry"
    ],
    "soft_skills": [
      "Public Speaking",
      "Client Relations",
      "Team Leadership",
      "Conflict Resolution",
      "Mentorship",
      "Customer Service",
      "Written Communication",
      "Community Outreach",
      "Teaching Support"
    ],
    "languages": [
      {
        "language": "English",
        "proficiency": ""
      },
      {
        "language": "Spanish",
        "proficiency": ""
      }
    ],
    "tools_and_technologies": []
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
}

EXAMPLE_2 = """"""
RESPONSE_2 = {}

