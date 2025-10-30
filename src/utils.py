"""
Utility functions for Pest Advisory System
"""

import json
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
import os


def sanitize_filename(text: str) -> str:
    """
    Convert text to valid filename
    
    Example: "Multan District" -> "multan_district"
    """
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    return text


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from LLM response that might have extra text
    
    Handles cases like:
    "Here's the analysis: {json...} Hope this helps!"
    """
    # Try to find JSON block
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    if matches:
        # Try each match
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
    
    # If no valid JSON found, try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def create_fallback_analysis(tehsil: str, severity_level: str, 
                             threat_pests: list) -> Dict[str, Any]:
    """
    Create fallback analysis when LLM doesn't return valid JSON
    """
    urgency_map = {
        'LOW': 'MONITOR',
        'MEDIUM': 'THIS_WEEK',
        'HIGH': 'IMMEDIATE'
    }
    
    impact_map = {
        'LOW': '5-10%',
        'MEDIUM': '15-25%',
        'HIGH': '30-50%'
    }
    
    action_urgency = urgency_map.get(severity_level, 'THIS_WEEK')

    fallback = {
        'risk_level': severity_level,
        'primary_threats': [p['name'] for p in threat_pests[:2]] if threat_pests else [],
        'secondary_concerns': [p['name'] for p in threat_pests[2:4]] if len(threat_pests) > 2 else [],
        'action_urgency': action_urgency,
        'economic_impact': impact_map.get(severity_level, '10-20%'),
        'key_insights': [
            f"Analysis for {tehsil} tehsil",
            f"Overall severity: {severity_level}",
            f"{len(threat_pests)} pest(s) above critical thresholds" if threat_pests else "No critical threats detected"
        ]
    }

    fallback['farmer_advice'] = build_default_farmer_advice({
        'severity_level': severity_level,
        'action_urgency': action_urgency,
        'threat_pests': threat_pests
    })

    return fallback


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary as formatted JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_timestamp() -> str:
    """Get formatted timestamp for filenames"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def format_pest_name(column_name: str) -> str:
    """
    Convert column name to readable pest name
    
    Example: "W. FLY(ABOVE ETL)" -> "Whitefly"
    """
    # Remove (ABOVE ETL) and clean
    name = column_name.replace('(ABOVE ETL)', '').strip()
    
    # Map abbreviations to full names
    name_map = {
        'W. FLY': 'Whitefly',
        'JASSID': 'Jassid',
        'THRIPS': 'Thrips',
        'M.BUG': 'Mealy Bug',
        'MITES': 'Mites',
        'APHIDS': 'Aphids',
        'DUSKY COTTON BUG': 'Dusky Cotton Bug',
        'SBW': 'Spotted Bollworm',
        'PBW': 'Pink Bollworm',
        'ABW': 'American Bollworm',
        'ARMY WORM': 'Army Worm'
    }
    
    return name_map.get(name, name.title())


def calculate_percentage_above_etl(count: float, threshold: float) -> float:
    """Calculate how much above ETL (as percentage)"""
    if threshold == 0:
        return 0.0
    return max(0, ((count - threshold) / threshold) * 100)


def format_number(num: float, decimals: int = 2) -> str:
    """Format number for display"""
    if num >= 1000:
        return f"{num:,.{decimals}f}"
    return f"{num:.{decimals}f}"


def build_default_farmer_advice(context: Dict[str, Any]) -> List[str]:
    """
    Build English farmer advice when LLM guidance is unavailable.
    """
    advice: List[str] = []

    severity_level = str(context.get('severity_level', 'UNKNOWN')).upper()
    action_urgency = str(context.get('action_urgency', 'MONITOR')).upper()
    threat_pests = context.get('threat_pests', []) or []

    severity_guidance = {
        'LOW': "Maintain weekly scouting and keep photographic records to track emerging pest hotspots.",
        'MEDIUM': "Increase scouting to every 3-4 days, especially along field borders and previous infestation zones.",
        'HIGH': "Coordinate immediate field walks with the agriculture extension team and prepare integrated control actions.",
    }

    urgency_guidance = {
        'MONITOR': "Hold insecticide sprays until two consecutive counts cross ETL; keep scouting notes up to date.",
        'THIS_WEEK': "Schedule recommended pesticides this week and coordinate timings with neighbouring farms.",
        'WITHIN_48H': "Mobilise spray teams within 48 hours and ensure equipment is calibrated for uniform coverage.",
        'IMMEDIATE': "Begin emergency control today, remove heavily infested plants, and seek official guidance for chemicals.",
    }

    if severity_level in severity_guidance:
        advice.append(severity_guidance[severity_level])

    if action_urgency in urgency_guidance:
        advice.append(urgency_guidance[action_urgency])

    for pest in threat_pests[:3]:
        name = pest.get('name', 'Key pest')
        advice.append(f"Prioritise targeted control for {name}; follow ETL-based spray schedules and remove infested plant parts.")

    if not advice:
        advice.append("Current conditions are stable—keep routine scouting and share any sudden pest rise with the extension officer.")

    advice.append("Consult the local agriculture department before applying chemicals and record every intervention for compliance.")

    return advice
