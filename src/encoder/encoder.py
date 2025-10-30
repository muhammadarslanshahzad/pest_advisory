"""
Encoder Module: Analyzes pest data and generates structured insights
Model: LLaMA 3.2 (3B) via Ollama
"""

import requests
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from src.config.config import (
    OLLAMA_BASE_URL, ENCODER_MODEL, ENCODER_TEMPERATURE, 
    ENCODER_TIMEOUT, ETL_THRESHOLDS, SEVERITY_THRESHOLDS,
    CRITICAL_MULTIPLIER, PEST_INFO, ENCODER_OUTPUT_DIR,
    SAVE_INDIVIDUAL_RESULTS
)
from src.utils import (
    extract_json_from_text, create_fallback_analysis,
    save_json, sanitize_filename, format_pest_name,
    calculate_percentage_above_etl, get_timestamp,
    build_default_farmer_advice
)
from src.visualization.visualizer import PestVisualizer
from src.reporting.report_generator import PestReportGenerator


class PestDataEncoder:
    """
    Encoder component for pest data analysis
    
    Responsibilities:
    - Calculate severity scores
    - Identify threat pests
    - Call LLaMA for structured analysis
    - Validate and parse LLM outputs
    """
    
    def __init__(
        self,
        ollama_url: str = OLLAMA_BASE_URL,
        visualizer: Optional[PestVisualizer] = None,
        report_generator: Optional[PestReportGenerator] = None,
        auto_visualize: bool = True,
        auto_report: bool = True,
    ):
        self.ollama_url = ollama_url
        self.model = ENCODER_MODEL
        self.temperature = ENCODER_TEMPERATURE
        self.timeout = ENCODER_TIMEOUT
        self.visualizer = visualizer if visualizer else (PestVisualizer() if auto_visualize else None)
        self.report_generator = report_generator if report_generator else (PestReportGenerator() if auto_report else None)
        self.auto_visualize = auto_visualize and self.visualizer is not None
        self.auto_report = auto_report and self.report_generator is not None
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test if Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            response.raise_for_status()
            
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if self.model not in model_names:
                print(f"⚠️  Warning: {self.model} not found in Ollama")
                print(f"   Available models: {model_names}")
                print(f"   Run: ollama pull {self.model}")
            else:
                print(f"✅ Ollama connected - {self.model} ready")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Cannot connect to Ollama at {self.ollama_url}")
            print(f"   Error: {e}")
            print(f"   Make sure Ollama is running: ollama serve")
            raise
    
    def calculate_severity(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate pest severity score and identify threats
        
        Returns:
            {
                'severity_score': float,
                'severity_level': 'LOW'|'MEDIUM'|'HIGH',
                'threat_pests': [{'name': str, 'count': float, ...}, ...]
            }
        """
        severity_score = 0.0
        threat_pests = []
        
        for pest_col, threshold in ETL_THRESHOLDS.items():
            if pest_col not in row:
                continue
            
            pest_count = float(row.get(pest_col, 0))
            
            if pest_count > threshold:
                # Calculate severity contribution
                # Score = (count / threshold - 1) * weight
                # Higher weight for more dangerous pests
                pest_weight = 2.0  # Default weight
                
                # Bollworms are more critical
                if 'BW' in pest_col:
                    pest_weight = 3.0
                
                pest_severity = ((pest_count / threshold) - 1) * pest_weight
                severity_score += min(pest_severity, 10)  # Cap individual pest at 10
                
                # Mark as threat if significantly above ETL
                if pest_count > threshold * 2:
                    threat_level = 'CRITICAL' if pest_count > threshold * CRITICAL_MULTIPLIER else 'HIGH'
                    
                    threat_pests.append({
                        'name': format_pest_name(pest_col),
                        'column': pest_col,
                        'count': pest_count,
                        'threshold': threshold,
                        'multiplier': round(pest_count / threshold, 2),
                        'percentage_above': calculate_percentage_above_etl(pest_count, threshold),
                        'severity': threat_level
                    })
        
        # Sort threats by severity
        threat_pests.sort(key=lambda x: x['multiplier'], reverse=True)
        
        # Determine overall severity level
        if severity_score < SEVERITY_THRESHOLDS['LOW']:
            severity_level = 'LOW'
        elif severity_score < SEVERITY_THRESHOLDS['MEDIUM']:
            severity_level = 'MEDIUM'
        else:
            severity_level = 'HIGH'
        
        return {
            'severity_score': round(severity_score, 2),
            'severity_level': severity_level,
            'threat_pests': threat_pests
        }
    
    def prepare_context(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare structured context for LLM from data row
        """
        severity_info = self.calculate_severity(row)
        
        context = {
            'tehsil': row.get('TEHSILS', 'Unknown'),
            'timeframe': row.get('TIMEFRAME', 'Unknown'),
            'total_spots': int(row.get('TOTAL SPOTS VISITED', 0)),
            'total_area': round(float(row.get('TOTAL AREA VISITED', 0)), 2),
            'severity_score': severity_info['severity_score'],
            'severity_level': severity_info['severity_level'],
            'threat_pests': severity_info['threat_pests'],
            'pest_pressure_per_acre': round(float(row.get('PEST_PRESSURE_PER_ACRE', 0)), 2),
            'pest_data': {},
            'disease_data': {}
        }
        
        # Add all pest counts
        for pest_col, threshold in ETL_THRESHOLDS.items():
            if pest_col in row:
                count = float(row.get(pest_col, 0))
                context['pest_data'][format_pest_name(pest_col)] = {
                    'count': round(count, 2),
                    'threshold': threshold,
                    'status': 'ABOVE' if count > threshold else 'BELOW',
                    'percentage_above': calculate_percentage_above_etl(count, threshold) if count > threshold else 0
                }
        
        # Add disease data (CLCuV and Wilt)
        if 'CLCV(%SPOT)' in row:
            context['disease_data']['CLCuV'] = {
                'spot_infection': round(float(row.get('CLCV(%SPOT)', 0)), 2),
                'area_infection': round(float(row.get('CLCV(%AREA)', 0)), 2)
            }
        
        if 'WILT(%SPOT)' in row:
            context['disease_data']['Wilt'] = {
                'spot_infection': round(float(row.get('WILT(%SPOT)', 0)), 2),
                'area_infection': round(float(row.get('WILT(%AREA)', 0)), 2)
            }
        
        return context
    
    def build_encoder_prompt(self, context: Dict[str, Any]) -> str:
        """
        Build structured prompt for LLaMA encoder
        """
        prompt = f"""You are an expert cotton pest analyst for Punjab Agriculture Department. Analyze this survey data and provide structured insights.

SURVEY INFORMATION:
- Tehsil: {context['tehsil']}
- Timeframe: {context['timeframe']}
- Survey Coverage: {context['total_spots']} spots covering {context['total_area']} acres
- Overall Severity: {context['severity_level']} (Score: {context['severity_score']}/30)
- Pest Pressure: {context['pest_pressure_per_acre']} pests per acre

PEST POPULATIONS vs ETL THRESHOLDS:
"""
        
        # Add pest data
        for pest_name, pest_info in context['pest_data'].items():
            status_emoji = "🔴" if pest_info['status'] == 'ABOVE' else "🟢"
            prompt += f"\n{status_emoji} {pest_name}: {pest_info['count']} (ETL: {pest_info['threshold']}) - {pest_info['status']}"
            if pest_info['status'] == 'ABOVE':
                prompt += f" [{pest_info['percentage_above']:.0f}% above ETL]"
        
        # Add critical threats
        if context['threat_pests']:
            prompt += f"\n\nCRITICAL THREATS ({len(context['threat_pests'])} pests):"
            for pest in context['threat_pests']:
                prompt += f"\n• {pest['name']}: {pest['count']} ({pest['severity']} - {pest['multiplier']}x above ETL)"
        
        # Add disease data
        if context['disease_data']:
            prompt += f"\n\nDISEASE STATUS:"
            for disease, data in context['disease_data'].items():
                prompt += f"\n• {disease}: {data['spot_infection']}% spots infected, {data['area_infection']}% area infected"
        
        # Request structured output
        prompt += """

TASK: Provide analysis in valid JSON format with these fields:

{
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "primary_threats": ["pest1", "pest2"],
  "secondary_concerns": ["pest3", "pest4"],
  "action_urgency": "MONITOR|THIS_WEEK|WITHIN_48H|IMMEDIATE",
  "economic_impact": "estimated yield loss percentage (e.g., 15-25%)",
  "key_insights": [
    "insight about pest population trends",
    "insight about disease pressure",
    "insight about economic risk"
  ],
  "farmer_advice": [
    "Action 1 in English referencing threats/severity",
    "Action 2 in English with clear IPM or chemical guidance",
    "Action 3 in English about monitoring or coordination"
  ]
}

IMPORTANT:
- Output ONLY valid JSON, no additional text
- Be specific about which pests are most threatening
- Consider both pest counts AND disease infection rates
- Base urgency on actual ETL violations
- Farmer advice must be expert, concise, actionable, in English, and reference survey data where relevant"""

        return prompt
    
    def call_llama(self, prompt: str) -> str:
        """
        Call Ollama API with LLaMA model
        
        Returns: Raw text response from LLM
        """
        url = f"{self.ollama_url}/api/generate"
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 512  # Limit response length
            }
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            elapsed = time.time() - start_time
            
            result = response.json()
            llm_response = result.get('response', '')
            
            print(f"   ⏱️  LLM response time: {elapsed:.2f}s")
            
            return llm_response
            
        except requests.exceptions.Timeout:
            print(f"   ❌ LLM timeout after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            print(f"   ❌ LLM API error: {e}")
            raise
    
    def analyze(self, row: Dict[str, Any], save_output: bool = True) -> Dict[str, Any]:
        """
        Complete encoding pipeline for one data row
        
        Args:
            row: Dictionary containing pest survey data
            save_output: Whether to save results to file
        
        Returns:
            {
                'tehsil': str,
                'context': dict,
                'encoder_analysis': dict,
                'metadata': dict
            }
        """
        print(f"\n{'='*60}")
        print(f"🔍 Analyzing: {row.get('TEHSILS', 'Unknown')} - {row.get('TIMEFRAME', 'Unknown')}")
        print(f"{'='*60}")
        
        # Step 1: Prepare context
        print("📊 Preparing context...")
        context = self.prepare_context(row)
        print(f"   ✅ Severity: {context['severity_level']} ({context['severity_score']}/30)")
        print(f"   ✅ Threat pests: {len(context['threat_pests'])}")
        
        # Step 2: Build prompt
        print("📝 Building encoder prompt...")
        prompt = self.build_encoder_prompt(context)
        
        # Step 3: Call LLM
        print(f"🤖 Calling {self.model}...")
        try:
            llm_response = self.call_llama(prompt)
        except Exception as e:
            print(f"   ❌ LLM call failed: {e}")
            print("   🔄 Using fallback analysis...")
            encoder_analysis = create_fallback_analysis(
                context['tehsil'],
                context['severity_level'],
                context['threat_pests']
            )
            encoder_analysis['_fallback'] = True
        else:
            # Step 4: Parse JSON
            print("🔍 Parsing LLM response...")
            encoder_analysis = extract_json_from_text(llm_response)
            
            if encoder_analysis is None:
                print("   ⚠️  Failed to parse JSON, using fallback...")
                encoder_analysis = create_fallback_analysis(
                    context['tehsil'],
                    context['severity_level'],
                    context['threat_pests']
                )
                encoder_analysis['_fallback'] = True
                encoder_analysis['_raw_response'] = llm_response[:200]  # Save snippet
            else:
                print("   ✅ JSON parsed successfully")
                encoder_analysis['_fallback'] = False
        
        # Step 5: Compile results
        result = {
            'tehsil': context['tehsil'],
            'timeframe': context['timeframe'],
            'context': context,
            'encoder_analysis': encoder_analysis,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': self.model,
                'temperature': self.temperature,
                'total_spots': context['total_spots'],
                'total_area': context['total_area'],
                'severity_score': context['severity_score'],
                'severity_level': context['severity_level']
            }
        }
        
        # Step 6: Enrich analysis with farmer advice (LLM or fallback)
        context_for_advice = dict(context)
        context_for_advice['action_urgency'] = encoder_analysis.get('action_urgency')
        encoder_analysis['farmer_advice'] = self._normalize_farmer_advice(
            encoder_analysis.get('farmer_advice'),
            context_for_advice
        )

        chart_paths: Dict[str, str] = {}
        if self.auto_visualize and self.visualizer:
            print("🖼️  Generating charts via visualizer...")
            try:
                chart_paths = self.visualizer.generate_all_charts(result)
                result['charts'] = chart_paths
            except Exception as exc:
                print(f"   ⚠️  Chart generation failed: {exc}")
                result['charts_error'] = str(exc)

        if self.auto_report and self.report_generator:
            print("📝 Building PDF report...")
            report_input = chart_paths or result.get('charts', {})
            try:
                report_path = self.report_generator.generate_report(result, report_input)
                result['report_path'] = report_path
            except Exception as exc:
                print(f"   ⚠️  Report generation failed: {exc}")
                result['report_error'] = str(exc)

        # Step 7: Save output (after artifacts are attached)
        if save_output and SAVE_INDIVIDUAL_RESULTS:
            filename = f"{sanitize_filename(context['tehsil'])}_{sanitize_filename(context['timeframe'])}_{get_timestamp()}.json"
            filepath = f"{ENCODER_OUTPUT_DIR}/{filename}"
            save_json(result, filepath)
            print(f"💾 Saved: {filepath}")
        
        print(f"✅ Analysis complete for {context['tehsil']}")
        
        return result

    @staticmethod
    def _ensure_list_of_strings(value: Any) -> List[str]:
        """Force arbitrary value into list of non-empty strings."""
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            return [text] if text else []
        if isinstance(value, (list, tuple, set)):
            items: List[str] = []
            for item in value:
                if item is None:
                    continue
                text = str(item).strip()
                if text:
                    items.append(text)
            return items
        text = str(value).strip()
        return [text] if text else []

    def _normalize_farmer_advice(self, advice: Any, context: Dict[str, Any]) -> List[str]:
        """
        Ensure farmer advice is a non-empty list of English guidance strings.
        """
        advice_list = self._ensure_list_of_strings(advice if not isinstance(advice, dict) else advice.get('english'))

        if not advice_list:
            advice_list = build_default_farmer_advice(context)

        return advice_list


def test_encoder_with_sample():
    """
    Test encoder with sample data
    """
    # Sample data row (adjust based on your actual data)
    sample_row = {
        'TEHSILS': 'Multan City',
        'TIMEFRAME': '2024-W42',
        'TOTAL SPOTS VISITED': 50,
        'TOTAL AREA VISITED': 250.5,
        'W. FLY(ABOVE ETL)': 25.5,
        'JASSID(ABOVE ETL)': 12.3,
        'THRIPS(ABOVE ETL)': 5.0,
        'M.BUG(ABOVE ETL)': 1.5,
        'MITES(ABOVE ETL)': 0.0,
        'APHIDS(ABOVE ETL)': 0.0,
        'DUSKY COTTON BUG(ABOVE ETL)': 0.0,
        'SBW(ABOVE ETL)': 0.0,
        'PBW(ABOVE ETL)': 8.5,
        'ABW(ABOVE ETL)': 3.2,
        'ARMY WORM(ABOVE ETL)': 0.0,
        'CLCV(%SPOT)': 15.5,
        'CLCV(%AREA)': 12.3,
        'WILT(%SPOT)': 2.1,
        'WILT(%AREA)': 1.8,
        'TOTAL_PESTS_ABOVE_ETL': 56.0,
        'PEST_PRESSURE_PER_ACRE': 0.224
    }
    
    # Initialize encoder
    encoder = PestDataEncoder()
    
    # Run analysis
    result = encoder.analyze(sample_row, save_output=True)
    
    # Print results
    print("\n" + "="*60)
    print("ENCODER ANALYSIS RESULT")
    print("="*60)
    print(json.dumps(result['encoder_analysis'], indent=2))
    
    return result


if __name__ == "__main__":
    # Run test
    test_encoder_with_sample()
