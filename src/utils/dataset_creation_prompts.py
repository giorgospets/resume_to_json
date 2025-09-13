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
            "start_date": "Aug 2016",
            "end_date": "present",
            "details": [
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
            "start_date": "Jun 2014",
            "end_date": "Jul 2016",
            "details": [
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
            "start_date": "Apr 2010",
            "end_date": "May 2014",
            "details": [
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
            "start_date": "Sep 2006",
            "end_date": "Apr 2010",
            "details": [
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
            "end_date": "2010",
            "grade_or_gpa": ""
        },
        {
            "degree": "Associate of Arts",
            "field_of_study": "Liberal Arts",
            "institution": "Front Range Community College",
            "location": "Westminster, CO",
            "start_date": "",
            "end_date": "2007",
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

EXAMPLE_2 = """Olivia Chen\r\nSenior Graphic Designer\r\nCONTACT\r\n(987)-654-3210\r\nolivia.chen@email.com\r\nBrooklyn, NY
\r[www.oliviacreates.com](www.oliviacreates.com)\r\n\r\n
PROFILE\r\nA creative and detail-oriented Senior Graphic Designer with over 7 years of experience in developing engaging and innovative digital and print designs for a diverse range of clients. Proven expertise in brand identity development, UI/UX principles, and project management. Seeking to leverage artistic and technical skills to create compelling visual solutions.\r\n\r\n
EXPERIENCE\r\n\r\nSenior Graphic Designer | Chroma Vision Studios | New York, NY\r\nJune 2020 - Present
\r\n- Led the creative direction and execution for a complete rebranding of a major tech client, resulting in a 25% increase in their social media engagement.\r\n- Mentored and supervised a team of two junior designers, providing guidance on projects and fostering their professional growth.
\r\n- Managed multiple design projects simultaneously from concept to completion, ensuring adherence to deadlines and client specifications.\r\n- Collaborated with marketing teams to develop visual assets for digital campaigns, including social media graphics, email newsletters, and web banners.
\r\n\r\nGraphic Designer | Bright Spark Agency | Brooklyn, NY\r\nJuly 2017 - May 2020 (Part-time)
\r\n- Developed branding packages for new startups, including logos, color palettes, typography, and style guides.
\r\n- Designed print materials such as brochures, posters, and trade show displays.
\r\n- Created wireframes and user interface mockups for mobile and web applications using Figma.
\r\n- Worked closely with clients to understand their needs and translate them into effective visual designs.
\r\n\r\nEDUCATION\r\n\r\nBachelor of Fine Arts in Graphic Design | 2017\r\nPratt Institute - Brooklyn, NY
\r\n\r\nSKILLS\r\n- Technical Skills: Adobe Creative Suite (Photoshop, Illustrator, InDesign), Figma, Sketch, Typography, UI/UX Principles, Digital Illustration.
\r\n- Soft Skills: Creativity, Collaboration, Project Management, Client Communication, Attention to Detail.
\r\n\r\nLANGUAGES\r\nEnglish (Native)\r\nMandarin (Conversational)\r\n\r\nPROJECTS
\r\n\r\nProject: "Verdant" Mobile App UI/UX Design
\r\n- Role: Lead UI/UX Designer
\r\n- Description: Conceptualized and designed the complete user interface and user experience for a new gardening app, "Verdant." This included user flow diagrams, wireframing, prototyping, and creating a full design system. The project focused on creating an intuitive and visually appealing interface to help users manage their plants.\r\n- Technologies Used: Figma, Adobe Illustrator\r\n\r\nAWARDS\r\n\r\n"Creative Innovator" Award, 2022 - Chroma Vision Studios\r\n- Awarded for outstanding performance and creative contributions to the tech client rebranding project.\r\n\r\nPUBLICATIONS\r\n\r\n"The Psychology of Color in Branding" | Design Forward Magazine | March 2023\r\n- An article exploring the impact of color theory on consumer perception and brand identity.\r\n- Available at: www.designforwardmag.com/color-psychology\r\n\r\nVOLUNTEER EXPERIENCE\r\n\r\nDesign Mentor | Creative Futures Youth Program | Brooklyn, NY\r\nAugust 2021 - Present\r\n- Volunteered weekly to mentor aspiring high school students in graphic design, providing portfolio reviews and career advice."""

RESPONSE_2 = {
    "personal_information": {
        "full_name": "Olivia Chen",
        "title": "Senior Graphic Designer",
        "email": "olivia.chen@email.com",
        "phone": "(987)-654-3210",
        "location": {
            "city": "Brooklyn",
            "state": "NY",
            "country": ""
        },
        "urls": [
            "www.oliviacreates.com"
        ]
    },
    "about_info": "A creative and detail-oriented Senior Graphic Designer with over 7 years of experience in developing engaging and innovative digital and print designs for a diverse range of clients. Proven expertise in brand identity development, UI/UX principles, and project management. Seeking to leverage artistic and technical skills to create compelling visual solutions.",
    "work_experience": [
        {
            "job_title": "Senior Graphic Designer",
            "company": "Chroma Vision Studios",
            "employment_type": "",
            "location": "New York, NY",
            "start_date": "June 2020",
            "end_date": "Present",
            "details": [
                "Led the creative direction and execution for a complete rebranding of a major tech client, resulting in a 25% increase in their social media engagement.",
                "Mentored and supervised a team of two junior designers, providing guidance on projects and fostering their professional growth.",
                "Managed multiple design projects simultaneously from concept to completion, ensuring adherence to deadlines and client specifications.",
                "Collaborated with marketing teams to develop visual assets for digital campaigns, including social media graphics, email newsletters, and web banners."
            ]
        },
        {
            "job_title": "Graphic Designer",
            "company": "Bright Spark Agency",
            "employment_type": "Part-time",
            "location": "Brooklyn, NY",
            "start_date": "July 2017",
            "end_date": "May 2020",
            "details": [
                "Developed branding packages for new startups, including logos, color palettes, typography, and style guides.",
                "Designed print materials such as brochures, posters, and trade show displays.",
                "Created wireframes and user interface mockups for mobile and web applications using Figma.",
                "Worked closely with clients to understand their needs and translate them into effective visual designs."
            ]
        }
    ],
    "education": [
        {
            "degree": "Bachelor of Fine Arts in Graphic Design",
            "field_of_study": "Graphic Design",
            "institution": "Pratt Institute",
            "location": "Brooklyn, NY",
            "start_date": "",
            "end_date": "2017",
            "grade_or_gpa": ""
        }
    ],
    "certifications": [],
    "skills": {
        "hard_skills": [
            "Typography",
            "UI/UX Principles",
            "Digital Illustration",
            "Branding"
        ],
        "soft_skills": [
            "Creativity",
            "Collaboration",
            "Project Management",
            "Client Communication",
            "Attention to Detail"
        ],
        "languages": [
            {
                "language": "English",
                "proficiency": "Native"
            },
            {
                "language": "Mandarin",
                "proficiency": "Conversational"
            }
        ],
        "tools_and_technologies": [
            "Adobe Creative Suite",
            "Photoshop",
            "Illustrator",
            "InDesign",
            "Figma",
            "Sketch"
        ]
    },
    "projects": [
        {
            "name": '"Verdant" Mobile App UI/UX Design',
            "description": 'Conceptualized and designed the complete user interface and user experience for a new gardening app, "Verdant." This included user flow diagrams, wireframing, prototyping, and creating a full design system. The project focused on creating an intuitive and visually appealing interface to help users manage their plants.',
            "technologies_used": [
                "Figma",
                "Adobe Illustrator"
            ],
            "role": "Lead UI/UX Designer",
            "start_date": "",
            "end_date": "",
            "project_url": ""
        }
    ],
    "publications": [
        {
            "title": "The Psychology of Color in Branding",
            "publisher": "Design Forward Magazine",
            "publication_date": "March 2023",
            "url": "www.designforwardmag.com/color-psychology",
            "description": "An article exploring the impact of color theory on consumer perception and brand identity."
        },
    ],
    "awards": [
        {
            "title": '"Creative Innovator" Award',
            "issuer": "Chroma Vision Studios",
            "date": "2022",
            "description": "Awarded for outstanding performance and creative contributions to the tech client rebranding project."
        }
    ],
    "volunteer_experience": [
        {
            "role": "Design Mentor",
            "organization": "Creative Futures Youth Program",
            "location": "Brooklyn, NY",
            "start_date": "August 2021",
            "end_date": "Present",
            "description": "Volunteered weekly to mentor aspiring high school students in graphic design, providing portfolio reviews and career advice."
        }
    ]
}