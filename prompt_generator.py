# prompt_generator.py
from typing import Literal
from dataclasses import dataclass

@dataclass
class InterviewConfig:
    domain: str
    difficulty: Literal['entry', 'mid', 'senior', 'lead']
    duration: int  # in minutes
    session_type: Literal['technical', 'behavioral']
    mode: Literal['practice', 'interview']
    location: str  # New field for location
    job_title: str
    job_description: str

@dataclass
class AnswerConfig:
    session_id: str
    question_id: str
    answer: str
    response_time_seconds: int

# Global variables for common data
DOMAINS = {
    'software-engineering': 'Software Engineering',
    'data-science': 'Data Science & Analytics',
    'product-management': 'Product Management',
    'marketing': 'Digital Marketing',
    'finance': 'Finance & Accounting',
    'sales': 'Sales & Business Development',
    'design': 'UI/UX Design',
    'operations': 'Operations & Strategy'
}

DIFFICULTIES = {
    'entry': 'Entry Level (0-2 years)',
    'mid': 'Mid Level (2-5 years)',
    'senior': 'Senior Level (5-10 years)',
    'lead': 'Lead/Principal (10+ years)'
}

LOCATION_CONTEXT = {
    'north-america': {
        'focus': 'Emphasis on practical problem-solving and system design',
        'communication': 'Clear and direct communication style',
        'cultural': 'Focus on individual contributions and innovation'
    },
    'europe': {
        'focus': 'Strong emphasis on theoretical knowledge and best practices',
        'communication': 'Structured and methodical communication',
        'cultural': 'Balance of individual and team contributions'
    },
    'asia-pacific': {
        'focus': 'Technical depth and attention to detail',
        'communication': 'Respectful and hierarchical communication style',
        'cultural': 'Strong emphasis on team harmony and collaboration'
    },
    'india': {
        'focus': 'Strong theoretical foundation and problem-solving',
        'communication': 'Clear technical communication with cultural context',
        'cultural': 'Balance of individual excellence and team collaboration'
    }
}

LOCATION_QUESTION_TYPES = {
    'north-america': {
        'technical': ['System design', 'Scalability', 'Cloud architecture', 'Agile practices'],
        'behavioral': ['Leadership style', 'Conflict resolution', 'Innovation mindset'],
        'case_study': ['Business growth case', 'Market entry strategy', 'Product launch scenario', 'Operational scaling', 'Turnaround strategy']
    },
    'europe': {
        'technical': ['Software architecture', 'Design patterns', 'Testing methodologies', 'Security practices'],
        'behavioral': ['Team collaboration', 'Process improvement', 'Quality focus'],
        'case_study': ['Regulatory compliance scenario', 'Market expansion case', 'Sustainability challenge', 'Digital transformation', 'Cross-border operations']
    },
    'asia-pacific': {
        'technical': ['Technical depth', 'Implementation details', 'Performance optimization'],
        'behavioral': ['Team dynamics', 'Cultural awareness', 'Process adherence'],
        'case_study': ['Emerging market entry', 'Cost optimization', 'Localization strategy', 'Supply chain challenge', 'Growth hacking scenario']
    },
    'india': {
        'technical': ['Algorithm complexity', 'System design', 'Implementation details'],
        'behavioral': ['Team leadership', 'Cross-functional collaboration', 'Process improvement'],
        'case_study': ['Startup scaling', 'Market penetration', 'Resource allocation', 'Digital adoption', 'Customer acquisition case']
    }
}

DOMAIN_QUESTION_TYPES = {
    'software_engineering': {
        'technical': ['Coding problems', 'System design', 'Data structures & algorithms', 'Architecture patterns', 'Debugging scenarios'],
        'behavioral': ['Technical leadership', 'Code review processes', 'Team collaboration', 'Project management'],
        'case_study': ['System outage response', 'Legacy system migration', 'Tech debt prioritization', 'Scaling architecture', 'DevOps transformation']
    },
    'data_science': {
        'technical': ['Statistical analysis', 'Machine learning algorithms', 'Data manipulation', 'Model evaluation', 'A/B testing'],
        'behavioral': ['Data storytelling', 'Stakeholder communication', 'Project prioritization', 'Cross-functional collaboration'],
        'case_study': ['Churn prediction project', 'Fraud detection scenario', 'Personalization engine', 'Data pipeline optimization', 'Business impact analysis']
    },
    'product_management': {
        'technical': ['Product strategy', 'Feature prioritization', 'Metrics & KPIs', 'User research', 'Roadmap planning'],
        'behavioral': ['Stakeholder management', 'Cross-team coordination', 'Decision making', 'Conflict resolution'],
        'case_study': ['Go-to-market plan', 'Feature launch', 'User adoption challenge', 'Pivot decision', 'Competitive analysis']
    },
    'marketing': {
        'technical': ['Campaign strategy', 'Analytics & attribution', 'Growth hacking', 'Content strategy', 'SEO/SEM'],
        'behavioral': ['Creative collaboration', 'Brand management', 'Budget planning', 'Team leadership'],
        'case_study': ['Brand repositioning', 'Market entry campaign', 'Crisis communication', 'Viral growth case', 'Multi-channel strategy']
    },
    'finance': {
        'technical': ['Financial modeling', 'Risk assessment', 'Valuation methods', 'Investment analysis', 'Regulatory compliance'],
        'behavioral': ['Client relationship management', 'Team leadership', 'Presentation skills', 'Ethical decision making'],
        'case_study': ['M&A scenario', 'Cost reduction plan', 'IPO readiness', 'Investment portfolio review', 'Financial turnaround']
    },
    'sales': {
        'technical': ['Sales methodology', 'Pipeline management', 'CRM usage', 'Forecasting', 'Deal negotiation'],
        'behavioral': ['Relationship building', 'Team motivation', 'Goal setting', 'Customer success'],
        'case_study': ['Enterprise deal close', 'Sales process redesign', 'Territory expansion', 'Quota recovery', 'Channel conflict resolution']
    },
    'design': {
        'technical': ['Design systems', 'User research', 'Prototyping', 'Usability testing', 'Design thinking'],
        'behavioral': ['Creative collaboration', 'Feedback incorporation', 'Design leadership', 'Stakeholder presentation'],
        'case_study': ['Product redesign', 'Accessibility challenge', 'Brand refresh', 'User journey mapping', 'Design sprint case']
    },
    'operations': {
        'technical': ['Process optimization', 'Supply chain management', 'Quality control', 'Data analysis', 'Strategic planning'],
        'behavioral': ['Change management', 'Team coordination', 'Crisis management', 'Performance improvement'],
        'case_study': ['Logistics bottleneck', 'Cost optimization', 'Business continuity', 'Vendor management', 'Operational scaling']
    }
}

DIFFICULTY_CONTEXT = {
    'entry': {
        'expectation': 'foundational knowledge and basic problem-solving',
        'complexity': 'straightforward scenarios with guided thinking',
        'evaluation': 'learning potential and fundamental understanding'
    },
    'mid': {
        'expectation': 'practical experience and intermediate problem-solving',
        'complexity': 'realistic scenarios requiring analytical thinking',
        'evaluation': 'hands-on experience and solution methodology'
    },
    'senior': {
        'expectation': 'deep expertise and advanced problem-solving',
        'complexity': 'complex, multi-faceted challenges',
        'evaluation': 'strategic thinking and technical depth'
    },
    'lead': {
        'expectation': 'thought leadership and architectural thinking',
        'complexity': 'ambiguous, high-level strategic problems',
        'evaluation': 'vision, mentorship, and system-level thinking'
    }
}

MODE_INSTRUCTIONS = {
    'practise': {
        'pacing': 'Allow flexible pacing and provide hints when the candidate is stuck',
        'feedback': 'Provide immediate feedback after each question',
        'assistance': 'Offer guidance and alternative approaches when needed',
        'atmosphere': 'Create a supportive, learning-focused environment'
    },
    'interview': {
        'pacing': 'Maintain realistic interview timing and pressure',
        'feedback': 'Save detailed feedback for the end of the session',
        'assistance': 'Provide minimal hints, simulating real interview conditions',
        'atmosphere': 'Create authentic interview pressure while remaining professional'
    }
}

SESSION_TYPE_INSTRUCTIONS = {
    'technical': 'Focus on technical competency, problem-solving methodology, and domain expertise',
    'behavioral': 'Focus on behavioral scenarios, soft skills, cultural fit, and interpersonal abilities'
}

def generate_initial_prompt(config: InterviewConfig) -> str:
    # Convert Enum values to their string representation for dictionary lookup
    domain_key = config.domain.value if hasattr(config.domain, 'value') else str(config.domain)
    session_type_key = config.session_type.value if hasattr(config.session_type, 'value') else str(config.session_type)
    difficulty_key = config.difficulty.value if hasattr(config.difficulty, 'value') else str(config.difficulty)

    domain = DOMAINS.get(domain_key, domain_key)
    difficulty = DIFFICULTIES.get(difficulty_key, difficulty_key)
    question_types = DOMAIN_QUESTION_TYPES[domain_key][session_type_key]

    # Get location-specific context
    location_key = config.location.lower().replace(' ', '-')
    location_info = LOCATION_CONTEXT.get(location_key, LOCATION_CONTEXT['north-america'])
    location_questions = LOCATION_QUESTION_TYPES.get(location_key, LOCATION_QUESTION_TYPES['north-america'])

    technical_criteria = '''- Technical accuracy and depth of knowledge
- Problem-solving approach and methodology
- Code quality and best practices (if applicable)
- Communication of technical concepts
- Handling of edge cases and trade-offs'''

    hr_criteria = '''- Communication and articulation skills
- Leadership and teamwork examples
- Problem-solving in interpersonal contexts
- Cultural fit and values alignment
- Growth mindset and adaptability'''

    practice_guidance = 'Provide helpful guidance and create a learning environment'
    realtime_guidance = 'Maintain professional interview standards and realistic pressure'
    practice_feedback = 'Give immediate feedback after each major question'
    realtime_feedback = 'Provide comprehensive feedback at the end'

    evaluation_criteria = technical_criteria if config.session_type == 'technical' else hr_criteria
    guidance_text = practice_guidance if config.mode == 'practice' else realtime_guidance
    feedback_text = practice_feedback if config.mode == 'practice' else realtime_feedback

    prompt = f"""# Interview Session Configuration

## Role & Context
You are an experienced {domain} interviewer with 180IQ conducting a {config.session_type.value} interview. This is a {config.mode.value} session designed to evaluate a {difficulty.lower()} candidate in the {config.location} region.

## Session Parameters
- **Domain**: {domain}
- **Difficulty Level**: {difficulty}
- **Duration**: {config.duration} minutes
- **Session Type**: {config.session_type.value.capitalize()}
- **Mode**: {config.mode.value.capitalize()}
- **Location**: {config.location}

## Regional Context
- **Focus**: {location_info['focus']}
- **Communication Style**: {location_info['communication']}
- **Cultural Considerations**: {location_info['cultural']}

## Candidate Expectations
Evaluate candidates based on {DIFFICULTY_CONTEXT[config.difficulty.value]['expectation']}. Present {DIFFICULTY_CONTEXT[config.difficulty.value]['complexity']} and assess their {DIFFICULTY_CONTEXT[config.difficulty.value]['evaluation']}.

## Question Categories to Cover
{chr(10).join([f'- {qt}' for qt in question_types])}

## Location-Specific Focus Areas
{chr(10).join([f'- {qt}' for qt in location_questions[config.session_type.value]])}

## Interview Approach
{SESSION_TYPE_INSTRUCTIONS[config.session_type.value]}

### Mode-Specific Instructions
- **Pacing**: {MODE_INSTRUCTIONS[config.mode.value]['pacing']}
- **Feedback**: {MODE_INSTRUCTIONS[config.mode.value]['feedback']}
- **Assistance**: {MODE_INSTRUCTIONS[config.mode.value]['assistance']}
- **Atmosphere**: {MODE_INSTRUCTIONS[config.mode.value]['atmosphere']}

## Session Structure
1. **Opening (2-3 minutes)**: Brief introduction and candidate background
2. **Core Questions ({int(config.duration * 0.7)} minutes)**: {max(1, round(config.duration / 15))} main questions covering the categories above
3. **Deep Dive ({int(config.duration * 0.2)} minutes)**: Follow-up questions on 1-2 areas
4. **Closing ({int(config.duration * 0.1)} minutes)**: Candidate questions and next steps

## Evaluation Criteria
{evaluation_criteria}

## Instructions
1. Start by greeting the candidate and explaining the session format
2. Ask questions progressively, building on their responses
3. {guidance_text}
4. Take notes on their responses for final evaluation
5. {feedback_text}
6. When starting an interview, always provide an introduction followed by the first question. Format your response as follows:\n\n---\n[INTRODUCTION]\nYour intro text here.\n\n[QUESTION]\nYour first interview question here.\n---

Begin the interview when ready. Remember to adapt your questions based on the candidate's responses and maintain the specified difficulty level throughout the session."""

    return prompt


def generate_initial_prompt_jd(config: InterviewConfig) -> str:
    system_prompt = f"""You are an experienced {config.job_title} interviewer with 180IQ conducting a {config.session_type.value} interview. This is a {config.mode.value} session designed to evaluate a candidate in the {config.location} region. 

## Session Parameters
- **Domain**: {config.job_title}
- **Difficulty Level**: {config.difficulty}
- **Duration**: {config.duration} minutes
- **Session Type**: {config.session_type.value.capitalize()}
- **Mode**: {config.mode.value.capitalize()}
- **Location**: {config.location}

## Candidate Expectations
Evaluate candidates based on {DIFFICULTY_CONTEXT[config.difficulty]['expectation']}. Present {DIFFICULTY_CONTEXT[config.difficulty]['complexity']} and assess their {DIFFICULTY_CONTEXT[config.difficulty]['evaluation']}.

### Mode-Specific Instructions
- **Pacing**: {MODE_INSTRUCTIONS[config.mode.value]['pacing']}
- **Feedback**: {MODE_INSTRUCTIONS[config.mode.value]['feedback']}
- **Assistance**: {MODE_INSTRUCTIONS[config.mode.value]['assistance']}
- **Atmosphere**: {MODE_INSTRUCTIONS[config.mode.value]['atmosphere']}

Prepare an interview for the following job description. Focus all questions and context on the requirements, responsibilities, and skills mentioned.\n\nJob Description:\n{config.job_description}\n\nInterview Mode: {config.mode.value}\nDuration: {config.duration} minutes\nJob Title: {config.job_title or 'N/A'}\n\nStart with a challenging but fair opening question that is highly relevant to the job description.
Format your response as follows:\n\n---\n[INTRODUCTION]\nYour intro text here.\n\n[QUESTION]\nYour first interview question here.\n---"""

    return system_prompt


def generate_next_prompt(config: InterviewConfig, previous_question: str, candidate_response: str) -> str:
    # Convert Enum values to their string representation for dictionary lookup
    domain_key = config.domain.value if hasattr(config.domain, 'value') else str(config.domain)
    session_type_key = config.session_type.value if hasattr(config.session_type, 'value') else str(config.session_type)
    difficulty_key = config.difficulty.value if hasattr(config.difficulty, 'value') else str(config.difficulty)

    domain = DOMAINS.get(domain_key, domain_key)
    difficulty = DIFFICULTIES.get(difficulty_key, difficulty_key)

    # Get location-specific context
    location_key = config.location.lower().replace(' ', '-')
    location_info = LOCATION_CONTEXT.get(location_key, LOCATION_CONTEXT['north-america'])

    prompt = f"""# Interview Session Configuration

## Role & Context
You are an experienced {domain} interviewer conducting a {config.session_type.value} interview for a {difficulty.lower()} candidate in {config.location}. This is a {config.mode.value} session.

## Session Parameters
- **Domain**: {domain}
- **Difficulty Level**: {difficulty}
- **Session Type**: {config.session_type.value.capitalize()}
- **Mode**: {config.mode.value.capitalize()}
- **Region**: {config.location}

## Interview Guidelines
- Ask questions that test {location_info['focus']}
- Provide a clear, structured question, and follow up based on the candidate's previous response
- Be culturally aware and maintain a {MODE_INSTRUCTIONS[config.mode.value]['atmosphere']}
- {MODE_INSTRUCTIONS[config.mode.value]['assistance']}
- Use the candidate's prior responses to determine the most effective follow-up or next topic

## Question History

### Previous Question:
{previous_question}

### Candidate Response:
{candidate_response}

## Instructions
1. Based on the previous exchange, either ask a follow-up to probe deeper or move to a related topic.
2. Frame only one question at a time.
3. Ensure it fits the session's difficulty level and scope.
4. Maintain clarity and relevance to the current session's progression.
5. Respond using the following format:

---
[NEXT QUESTION]
Your question here.
---

Generate the next interview question now."""

    return prompt


def generate_prompt(config: InterviewConfig | AnswerConfig, type: str, previous_question: str = None, candidate_response: str = None) -> str:
    if type == 'initial_question':
        return generate_initial_prompt(config)
    elif type == 'initial_question_jd':
        return generate_initial_prompt_jd(config)
    elif type == 'next_question':
        if not previous_question or not candidate_response:
            raise ValueError("Previous question and candidate response are required for next question generation")
        return generate_next_prompt(config, previous_question, candidate_response)
    else:
        raise ValueError(f"Unknown prompt type: {type}") 