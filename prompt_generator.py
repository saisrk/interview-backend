# prompt_generator.py
from typing import Literal
from dataclasses import dataclass

@dataclass
class InterviewConfig:
    domain: str
    difficulty: Literal['entry', 'mid', 'senior', 'lead']
    duration: int  # in minutes
    session_type: Literal['technical', 'hr']
    mode: Literal['practice', 'real-time']

def generate_prompt(config: InterviewConfig) -> str:
    domains = {
        'software-engineering': 'Software Engineering',
        'data-science': 'Data Science & Analytics',
        'product-management': 'Product Management',
        'marketing': 'Digital Marketing',
        'finance': 'Finance & Accounting',
        'sales': 'Sales & Business Development',
        'design': 'UI/UX Design',
        'operations': 'Operations & Strategy'
    }

    difficulties = {
        'entry': 'Entry Level (0-2 years)',
        'mid': 'Mid Level (2-5 years)',
        'senior': 'Senior Level (5-10 years)',
        'lead': 'Lead/Principal (10+ years)'
    }

    domain_question_types = {
        'software-engineering': {
            'technical': ['Coding problems', 'System design', 'Data structures & algorithms', 'Architecture patterns', 'Debugging scenarios'],
            'hr': ['Technical leadership', 'Code review processes', 'Team collaboration', 'Project management']
        },
        'data-science': {
            'technical': ['Statistical analysis', 'Machine learning algorithms', 'Data manipulation', 'Model evaluation', 'A/B testing'],
            'hr': ['Data storytelling', 'Stakeholder communication', 'Project prioritization', 'Cross-functional collaboration']
        },
        'product-management': {
            'technical': ['Product strategy', 'Feature prioritization', 'Metrics & KPIs', 'User research', 'Roadmap planning'],
            'hr': ['Stakeholder management', 'Cross-team coordination', 'Decision making', 'Conflict resolution']
        },
        'marketing': {
            'technical': ['Campaign strategy', 'Analytics & attribution', 'Growth hacking', 'Content strategy', 'SEO/SEM'],
            'hr': ['Creative collaboration', 'Brand management', 'Budget planning', 'Team leadership']
        },
        'finance': {
            'technical': ['Financial modeling', 'Risk assessment', 'Valuation methods', 'Investment analysis', 'Regulatory compliance'],
            'hr': ['Client relationship management', 'Team leadership', 'Presentation skills', 'Ethical decision making']
        },
        'sales': {
            'technical': ['Sales methodology', 'Pipeline management', 'CRM usage', 'Forecasting', 'Deal negotiation'],
            'hr': ['Relationship building', 'Team motivation', 'Goal setting', 'Customer success']
        },
        'design': {
            'technical': ['Design systems', 'User research', 'Prototyping', 'Usability testing', 'Design thinking'],
            'hr': ['Creative collaboration', 'Feedback incorporation', 'Design leadership', 'Stakeholder presentation']
        },
        'operations': {
            'technical': ['Process optimization', 'Supply chain management', 'Quality control', 'Data analysis', 'Strategic planning'],
            'hr': ['Change management', 'Team coordination', 'Crisis management', 'Performance improvement']
        }
    }

    difficulty_context = {
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

    mode_instructions = {
        'practice': {
            'pacing': 'Allow flexible pacing and provide hints when the candidate is stuck',
            'feedback': 'Provide immediate feedback after each question',
            'assistance': 'Offer guidance and alternative approaches when needed',
            'atmosphere': 'Create a supportive, learning-focused environment'
        },
        'real-time': {
            'pacing': 'Maintain realistic interview timing and pressure',
            'feedback': 'Save detailed feedback for the end of the session',
            'assistance': 'Provide minimal hints, simulating real interview conditions',
            'atmosphere': 'Create authentic interview pressure while remaining professional'
        }
    }

    session_type_instructions = {
        'technical': 'Focus on technical competency, problem-solving methodology, and domain expertise',
        'hr': 'Focus on behavioral scenarios, soft skills, cultural fit, and interpersonal abilities'
    }

    domain = domains.get(config.domain, config.domain)
    difficulty = difficulties.get(config.difficulty, config.difficulty)
    question_types = domain_question_types[config.domain][config.session_type]

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
You are an experienced {domain} interviewer conducting a {config.session_type} interview. This is a {config.mode} session designed to evaluate a {difficulty.lower()} candidate.

## Session Parameters
- **Domain**: {domain}
- **Difficulty Level**: {difficulty}
- **Duration**: {config.duration} minutes
- **Session Type**: {config.session_type.capitalize()}
- **Mode**: {config.mode.capitalize()}

## Candidate Expectations
Evaluate candidates based on {difficulty_context[config.difficulty]['expectation']}. Present {difficulty_context[config.difficulty]['complexity']} and assess their {difficulty_context[config.difficulty]['evaluation']}.

## Question Categories to Cover
{chr(10).join([f'- {qt}' for qt in question_types])}

## Interview Approach
{session_type_instructions[config.session_type]}

### Mode-Specific Instructions
- **Pacing**: {mode_instructions[config.mode]['pacing']}
- **Feedback**: {mode_instructions[config.mode]['feedback']}
- **Assistance**: {mode_instructions[config.mode]['assistance']}
- **Atmosphere**: {mode_instructions[config.mode]['atmosphere']}

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

Begin the interview when ready. Remember to adapt your questions based on the candidate's responses and maintain the specified difficulty level throughout the session."""

    return prompt
