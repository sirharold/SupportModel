"""
Advanced RAG metrics for comprehensive evaluation.
Includes hallucination detection, context utilization, completeness, and user satisfaction proxies.
"""

import re
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)

def calculate_hallucination_score(
    answer: str, 
    context_docs: List[Dict], 
    question: str = None
) -> Dict:
    """
    Detect potential hallucinations in the generated answer.
    
    Args:
        answer: Generated answer text
        context_docs: Retrieved documents used as context
        question: Original question (optional)
    
    Returns:
        Dict with hallucination metrics
    """
    if not answer or not context_docs:
        return {"hallucination_score": 1.0, "confidence": "low", "details": "No answer or context"}
    
    # Extract all text from context documents
    context_text = ""
    for doc in context_docs:
        title = doc.get('title', '')
        content = doc.get('content', '')
        context_text += f"{title} {content} "
    
    context_text = context_text.lower()
    answer_lower = answer.lower()
    
    # 1. Entity and Fact Extraction from Answer
    answer_entities = extract_entities_and_facts(answer_lower)
    context_entities = extract_entities_and_facts(context_text)
    
    # 2. Check for unsupported claims
    unsupported_claims = []
    supported_claims = []
    
    for entity in answer_entities:
        if any(entity in context_entity for context_entity in context_entities):
            supported_claims.append(entity)
        else:
            # Check for partial matches or synonyms
            if not check_entity_support(entity, context_text):
                unsupported_claims.append(entity)
            else:
                supported_claims.append(entity)
    
    # 3. Calculate hallucination score
    total_claims = len(answer_entities)
    if total_claims == 0:
        hallucination_score = 0.0
        confidence = "low"
    else:
        hallucination_score = len(unsupported_claims) / total_claims
        confidence = "high" if total_claims >= 3 else "medium" if total_claims >= 1 else "low"
    
    # 4. Additional checks
    numeric_hallucinations = check_numeric_consistency(answer, context_text)
    temporal_hallucinations = check_temporal_consistency(answer, context_text)
    
    return {
        "hallucination_score": hallucination_score,  # 0 = no hallucinations, 1 = all hallucinated
        "confidence": confidence,
        "total_claims": total_claims,
        "unsupported_claims": len(unsupported_claims),
        "supported_claims": len(supported_claims),
        "numeric_inconsistencies": numeric_hallucinations,
        "temporal_inconsistencies": temporal_hallucinations,
        "details": {
            "unsupported": unsupported_claims[:3],  # Show first 3
            "supported": supported_claims[:3]
        }
    }

def calculate_context_utilization(
    answer: str, 
    context_docs: List[Dict], 
    question: str
) -> Dict:
    """
    Measure how well the context is utilized in the answer.
    
    Args:
        answer: Generated answer
        context_docs: Retrieved documents
        question: Original question
    
    Returns:
        Dict with context utilization metrics
    """
    if not answer or not context_docs:
        return {"utilization_score": 0.0, "coverage_score": 0.0}
    
    answer_lower = answer.lower()
    
    # 1. Extract key information from each document
    doc_info = []
    total_context_length = 0
    
    for i, doc in enumerate(context_docs):
        title = doc.get('title', '')
        content = doc.get('content', '')
        doc_text = f"{title} {content}".lower()
        
        # Extract key phrases (3+ char words, excluding common words)
        key_phrases = extract_key_phrases(doc_text)
        doc_info.append({
            'doc_id': i,
            'text': doc_text,
            'key_phrases': key_phrases,
            'length': len(doc_text.split())
        })
        total_context_length += len(doc_text.split())
    
    # 2. Calculate utilization per document
    utilized_docs = 0
    total_overlap = 0
    doc_utilizations = []
    
    for doc in doc_info:
        # Check how many key phrases from this doc appear in the answer
        phrases_used = 0
        for phrase in doc['key_phrases']:
            if phrase in answer_lower:
                phrases_used += 1
        
        doc_utilization = phrases_used / len(doc['key_phrases']) if doc['key_phrases'] else 0
        doc_utilizations.append(doc_utilization)
        
        if doc_utilization > 0.1:  # At least 10% of phrases used
            utilized_docs += 1
        
        total_overlap += phrases_used
    
    # 3. Calculate overall scores
    utilization_score = utilized_docs / len(context_docs) if context_docs else 0
    coverage_score = np.mean(doc_utilizations) if doc_utilizations else 0
    
    # 4. Calculate information density
    answer_length = len(answer.split())
    context_length = sum(doc['length'] for doc in doc_info)
    compression_ratio = answer_length / context_length if context_length > 0 else 0
    
    return {
        "utilization_score": utilization_score,  # 0-1, how many docs were used
        "coverage_score": coverage_score,  # 0-1, average utilization per doc
        "compression_ratio": compression_ratio,  # answer_len / context_len
        "docs_utilized": utilized_docs,
        "total_docs": len(context_docs),
        "doc_utilizations": doc_utilizations,
        "total_key_phrases_used": total_overlap
    }

def calculate_answer_completeness(
    answer: str, 
    question: str, 
    context_docs: List[Dict]
) -> Dict:
    """
    Evaluate the completeness of the generated answer.
    
    Args:
        answer: Generated answer
        question: Original question
        context_docs: Retrieved documents
    
    Returns:
        Dict with completeness metrics
    """
    if not answer or not question:
        return {"completeness_score": 0.0, "missing_aspects": []}
    
    # 1. Identify question type and expected components
    question_type = classify_question_type(question)
    expected_components = get_expected_components(question_type, question)
    
    # 2. Check for presence of expected components
    present_components = []
    missing_components = []
    
    answer_lower = answer.lower()
    
    for component in expected_components:
        if check_component_presence(component, answer_lower, context_docs):
            present_components.append(component)
        else:
            missing_components.append(component)
    
    # 3. Calculate completeness score
    completeness_score = len(present_components) / len(expected_components) if expected_components else 1.0
    
    # 4. Additional completeness checks
    has_examples = check_for_examples(answer)
    has_steps = check_for_steps(answer, question)
    has_caveats = check_for_caveats(answer)
    
    # 5. Length and detail analysis
    answer_length = len(answer.split())
    detail_score = min(1.0, answer_length / 50)  # Normalize to 50 words as baseline
    
    return {
        "completeness_score": completeness_score,
        "missing_components": missing_components,
        "present_components": present_components,
        "question_type": question_type,
        "has_examples": has_examples,
        "has_steps": has_steps,
        "has_caveats": has_caveats,
        "detail_score": detail_score,
        "answer_length": answer_length
    }

def calculate_user_satisfaction_proxy(
    answer: str, 
    question: str, 
    context_docs: List[Dict],
    response_time: float = None
) -> Dict:
    """
    Calculate proxy metrics for user satisfaction.
    
    Args:
        answer: Generated answer
        question: Original question
        context_docs: Retrieved documents
        response_time: Time taken to generate response
    
    Returns:
        Dict with user satisfaction proxy metrics
    """
    if not answer or not question:
        return {"satisfaction_score": 0.0}
    
    # 1. Clarity and readability
    clarity_score = calculate_clarity_score(answer)
    
    # 2. Directness (answers question directly)
    directness_score = calculate_directness_score(answer, question)
    
    # 3. Actionability (provides actionable information)
    actionability_score = calculate_actionability_score(answer)
    
    # 4. Confidence indicators
    confidence_score = calculate_confidence_indicators(answer)
    
    # 5. Response time factor
    time_penalty = 0
    if response_time:
        if response_time > 10:  # Very slow
            time_penalty = 0.3
        elif response_time > 5:  # Slow
            time_penalty = 0.1
    
    # 6. Combine scores
    base_satisfaction = np.mean([
        clarity_score,
        directness_score, 
        actionability_score,
        confidence_score
    ])
    
    final_satisfaction = max(0, base_satisfaction - time_penalty)
    
    return {
        "satisfaction_score": final_satisfaction,
        "clarity_score": clarity_score,
        "directness_score": directness_score,
        "actionability_score": actionability_score,
        "confidence_score": confidence_score,
        "time_penalty": time_penalty,
        "response_time": response_time
    }

# Helper functions

def extract_entities_and_facts(text: str) -> List[str]:
    """Extract potential entities and facts from text."""
    # Extract numbers, dates, technical terms, etc.
    entities = []
    
    # Numbers and versions
    numbers = re.findall(r'\b\d+(?:\.\d+)*\b', text)
    entities.extend(numbers)
    
    # Technical terms (capitalized words, acronyms)
    tech_terms = re.findall(r'\b[A-Z][a-zA-Z]*\b|\b[A-Z]{2,}\b', text)
    entities.extend(tech_terms)
    
    # Azure services and features
    azure_terms = re.findall(r'\bazure\s+\w+|\b\w+\s+service\b|\b\w+\s+api\b', text)
    entities.extend(azure_terms)
    
    return list(set(entities))

def check_entity_support(entity: str, context: str) -> bool:
    """Check if an entity is supported by context."""
    # Simple substring check with some fuzzy matching
    entity_lower = entity.lower()
    
    # Direct match
    if entity_lower in context:
        return True
    
    # Partial match for compound terms
    words = entity_lower.split()
    if len(words) > 1:
        return all(word in context for word in words)
    
    return False

def check_numeric_consistency(answer: str, context: str) -> int:
    """Check for numeric inconsistencies."""
    answer_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))
    context_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', context))
    
    inconsistent = answer_numbers - context_numbers
    return len(inconsistent)

def check_temporal_consistency(answer: str, context: str) -> int:
    """Check for temporal inconsistencies."""
    answer_dates = set(re.findall(r'\b20\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', answer))
    context_dates = set(re.findall(r'\b20\d{2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b', context))
    
    inconsistent = answer_dates - context_dates
    return len(inconsistent)

def extract_key_phrases(text: str) -> List[str]:
    """Extract key phrases from text."""
    # Remove common words and extract meaningful phrases
    common_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
    
    words = re.findall(r'\b\w{3,}\b', text.lower())
    key_words = [word for word in words if word not in common_words]
    
    # Also extract bigrams
    bigrams = [f"{key_words[i]} {key_words[i+1]}" for i in range(len(key_words)-1)]
    
    return list(set(key_words + bigrams))

def classify_question_type(question: str) -> str:
    """Classify the type of question to determine expected components."""
    question_lower = question.lower()
    
    if any(word in question_lower for word in ['how', 'steps', 'process', 'tutorial']):
        return 'how-to'
    elif any(word in question_lower for word in ['what', 'define', 'definition']):
        return 'definition'
    elif any(word in question_lower for word in ['why', 'reason', 'cause']):
        return 'explanation'
    elif any(word in question_lower for word in ['when', 'time', 'schedule']):
        return 'temporal'
    elif any(word in question_lower for word in ['where', 'location']):
        return 'location'
    elif any(word in question_lower for word in ['compare', 'difference', 'vs', 'versus']):
        return 'comparison'
    else:
        return 'general'

def get_expected_components(question_type: str, question: str) -> List[str]:
    """Get expected components based on question type."""
    components = {
        'how-to': ['steps', 'prerequisites', 'examples', 'tools'],
        'definition': ['definition', 'purpose', 'examples', 'related_concepts'],
        'explanation': ['reasons', 'causes', 'effects', 'context'],
        'temporal': ['timeframe', 'schedule', 'deadlines'],
        'location': ['location', 'access_method'],
        'comparison': ['similarities', 'differences', 'recommendations'],
        'general': ['main_answer', 'context', 'examples']
    }
    
    return components.get(question_type, ['main_answer'])

def check_component_presence(component: str, answer: str, context_docs: List[Dict]) -> bool:
    """Check if a component is present in the answer."""
    component_indicators = {
        'steps': ['step', 'first', 'second', 'then', 'next', 'finally', '1.', '2.'],
        'prerequisites': ['require', 'need', 'prerequisite', 'before'],
        'examples': ['example', 'for instance', 'such as', 'like'],
        'definition': ['is', 'means', 'refers to', 'defined as'],
        'reasons': ['because', 'due to', 'reason', 'since'],
        'tools': ['tool', 'use', 'utility', 'application']
    }
    
    indicators = component_indicators.get(component, [component])
    return any(indicator in answer for indicator in indicators)

def check_for_examples(answer: str) -> bool:
    """Check if answer contains examples."""
    example_indicators = ['example', 'for instance', 'such as', 'like', 'e.g.', 'for example']
    return any(indicator in answer.lower() for indicator in example_indicators)

def check_for_steps(answer: str, question: str) -> bool:
    """Check if answer contains steps when appropriate."""
    if 'how' not in question.lower():
        return True  # Not applicable
    
    step_indicators = ['step', 'first', 'second', 'then', 'next', 'finally', '1.', '2.', '3.']
    return any(indicator in answer.lower() for indicator in step_indicators)

def check_for_caveats(answer: str) -> bool:
    """Check if answer includes appropriate caveats."""
    caveat_indicators = ['however', 'but', 'note that', 'important', 'warning', 'caution', 'limitation']
    return any(indicator in answer.lower() for indicator in caveat_indicators)

def calculate_clarity_score(answer: str) -> float:
    """Calculate clarity score based on readability."""
    sentences = answer.split('.')
    if not sentences:
        return 0.0
    
    # Average sentence length (shorter is clearer)
    avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
    sentence_clarity = max(0, 1 - (avg_sentence_length - 15) / 20)  # Optimal ~15 words
    
    # Check for clear structure
    has_structure = any(indicator in answer.lower() for indicator in ['first', 'second', 'then', 'finally', '1.', '2.'])
    structure_bonus = 0.2 if has_structure else 0
    
    return min(1.0, sentence_clarity + structure_bonus)

def calculate_directness_score(answer: str, question: str) -> float:
    """Calculate how directly the answer addresses the question."""
    question_words = set(re.findall(r'\b\w{3,}\b', question.lower()))
    answer_words = set(re.findall(r'\b\w{3,}\b', answer.lower()))
    
    # Overlap between question and answer
    overlap = len(question_words.intersection(answer_words))
    directness = overlap / len(question_words) if question_words else 0
    
    # Check if answer starts directly
    starts_directly = not answer.lower().startswith(('well', 'so', 'actually', 'in general'))
    directness_bonus = 0.2 if starts_directly else 0
    
    return min(1.0, directness + directness_bonus)

def calculate_actionability_score(answer: str) -> float:
    """Calculate how actionable the answer is."""
    action_indicators = [
        'click', 'select', 'choose', 'configure', 'set', 'create', 'install',
        'run', 'execute', 'follow', 'go to', 'navigate', 'open'
    ]
    
    action_count = sum(1 for indicator in action_indicators if indicator in answer.lower())
    actionability = min(1.0, action_count / 3)  # Normalize to 3 actions
    
    return actionability

def calculate_confidence_indicators(answer: str) -> float:
    """Calculate confidence based on language used."""
    uncertain_phrases = ['might', 'maybe', 'possibly', 'could be', 'not sure', 'unclear']
    confident_phrases = ['will', 'is', 'are', 'definitely', 'always', 'must']
    
    uncertain_count = sum(1 for phrase in uncertain_phrases if phrase in answer.lower())
    confident_count = sum(1 for phrase in confident_phrases if phrase in answer.lower())
    
    total_phrases = uncertain_count + confident_count
    if total_phrases == 0:
        return 0.5  # Neutral
    
    confidence = confident_count / total_phrases
    return confidence