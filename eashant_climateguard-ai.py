import os
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient
# Get Gemini API key from Kaggle Secrets
user_secrets = UserSecretsClient()
GOOGLE_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

# Configure Gemini with latest model
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# Test Gemini connection
print("ğŸ”„ Testing Gemini API connection...")
response = model.generate_content("Respond with: 'Climate Fact-Checker AI is online and ready!'")
print(response.text)
print("âœ… Gemini API connected successfully!")



# Climate Misinformation Combat Agent(Climate Guard AI) - Setup
# Install required packages for fact-checking system

!pip install -q reportlab
print("report lab is isntalled")
!pip install -q google-generativeai requests beautifulsoup4 pandas
import json
import re
import requests
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
from bs4 import BeautifulSoup
import json as json_lib
import time


print("âœ… All dependencies installed successfully!")
print("ğŸ“¦ Packages: Gemini AI, Web Scraping, Data Processing")


# Configuration for Climate Fact-Checking System
CONFIG = {
    'scientific_sources': {
        'NASA_GISS': 'https://climate.nasa.gov/vital-signs/global-temperature/',
        'IPCC_AR6': 'https://www.ipcc.ch/report/ar6/wg1/',
        'NOAA_Climate': 'https://www.climate.gov/news-features/understanding-climate',
    },
    'consensus_threshold': 97,  # 97% of climate scientists agree
    'credibility_weights': {
        'peer_reviewed': 0.4,
        'government_source': 0.3,
        'scientific_consensus': 0.3
    },
    'claim_types': [
        'climate_denial',
        'exaggeration',
        'factual',
        'misleading_data',
        'cherry_picking'
    ]
}
# Sample climate claims for testing (you can replace with real data)
SAMPLE_CLAIMS = [
    "Electric vehicles are worse for the environment than gas cars",
    "Solar panels create more pollution than they prevent",
    "Nuclear power plants cause more deaths than coal",
    "Planting trees is enough to stop climate change",
    "The Arctic ice is actually increasing, not melting",
    "Climate models have never been accurate",
    "Volcanoes emit more CO2 than humans",
    "A few degrees of warming won't make a difference",
    "CO2  is a green house gas",
]


print(f"âœ… Configuration loaded!")
print(f"ğŸ“Š Scientific sources: {len(CONFIG['scientific_sources'])} databases")
print(f"ğŸ�¯ Consensus threshold: {CONFIG['consensus_threshold']}%")
print(f"ğŸ“‹ Sample claims for testing: {len(SAMPLE_CLAIMS)}")


# ============================================================================
# AGENT 1 - CLASSIFIES CLAIMS INTO CATEGORIES 
# ============================================================================
class ClaimDetectorAgent:
    """Agent 1: Detects and classifies climate-related claims"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.detected_claims = []
    
    def detect_claim(self, text: str) -> Dict[str, Any]:
        """Analyze text and detect climate claims"""
        try:
            prompt = f"""You are a climate science fact-checker. Analyze this statement:

"{text}"

Classify this claim in JSON format:
{{
    "is_climate_related": <true/false>,
    "claim_type": "<one of: climate_denial, exaggeration, factual, misleading_data, cherry_picking, neutral>",
    "main_claim": "<extract the core scientific claim>",
    "confidence": <0.0 to 1.0>,
    "keywords": ["keyword1", "keyword2"],
    "needs_verification": <true/false>,
    "reasoning": "<brief explanation>"
}}

CRITICAL INSTRUCTION FOR needs_verification:
- Set to TRUE for: conspiracy theories, false claims, misleading statements, denialism, exaggerations
- Set to FALSE for: neutral statements, well-established scientific facts
- Examples that need verification: "hoax", "not a greenhouse gas", "temperatures haven't increased"
- Examples that don't: "97% of scientists agree" (already documented fact)
"""

            response = self.model.generate_content(prompt)
            text_response = response.text
            
            # Remove markdown fences if present
            text_response = text_response.replace('``````', '').strip()
            
            # Extract JSON
            json_start = text_response.find('{')
            json_end = text_response.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No valid JSON found in response")
            
            result = json.loads(text_response[json_start:json_end])
            
            result['original_text'] = text
            result['status'] = 'success'
            
            return result
            
        except Exception as e:
            return {
                'original_text': text,
                'status': 'error',
                'error_message': str(e),
                'needs_verification': False
            }
    
    def execute(self, claims: List[str]) -> List[Dict[str, Any]]:
        """Process multiple claims"""
        print(f"ğŸ”� Agent 1: Detecting claims in {len(claims)} statements...")
        
        results = []
        for i, claim in enumerate(claims, 1):
            print(f"  â†’ Processing claim {i}/{len(claims)}...")
            result = self.detect_claim(claim)
            results.append(result)
        
        successful = sum(1 for r in results if r['status'] == 'success')
        verified_needed = sum(1 for r in results if r.get('needs_verification', False))
        
        print(f"âœ… Agent 1 Complete: {successful}/{len(claims)} claims analyzed")
        print(f"ğŸ�¯ Claims needing verification: {verified_needed}")
        
        self.detected_claims = results
        return results

# Test Agent 1 (FIXED VERSION)
agent1 = ClaimDetectorAgent(model, CONFIG)
detected_claims = agent1.execute(SAMPLE_CLAIMS)

# Display results
print("\nğŸ“Š CLAIM DETECTION RESULTS:")
print("=" * 70)
for i, claim in enumerate(detected_claims, 1):
    if claim['status'] == 'success':
        print(f"\n{i}. Original: {claim['original_text'][:60]}...")
        print(f"   Type: {claim.get('claim_type', 'Unknown')}")
        print(f"   Main Claim: {claim.get('main_claim', 'N/A')}")
        print(f"   Needs Verification: {'Yes âš ï¸�' if claim.get('needs_verification') else 'No âœ“'}")
        print(f"   Confidence: {claim.get('confidence', 0):.2f}")
    else:
        print(f"\n{i}. â�Œ ERROR: {claim.get('error_message', 'Unknown error')}")



# ============================================================================
# AGENT 2 - SOURCE VERIFIER 
# ============================================================================
class SourceVerifierAgent:
    """Agent 2: Verifies claims against scientific databases"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.scientific_facts = self._load_scientific_facts()
    
    def _load_scientific_facts(self) -> Dict[str, str]:
        """Load verified scientific facts about climate"""
        return {
            'global_warming': 'Global average temperature has increased by approximately 1.1Â°C since pre-industrial times (1850-1900). Source: IPCC AR6 2021',
            'scientific_consensus': '97% of actively publishing climate scientists agree that humans are causing global warming. Source: Cook et al. 2013',
            'co2_greenhouse': 'CO2 is a greenhouse gas that traps heat in the atmosphere. This is established physics since the 1800s. Source: NASA',
            'human_cause': 'Human activities (fossil fuels, deforestation) are the primary cause of recent warming. Source: IPCC AR6 WG1',
            'temperature_trend': 'The past decade (2011-2020) was the warmest on record. Source: NOAA, NASA GISS',
            'sea_level_rise': 'Global sea level has risen about 8-9 inches since 1880. Source: NASA',
            'ice_melt': 'Arctic sea ice is declining at a rate of 13% per decade. Source: NSIDC',
            'extreme_weather': 'Climate change is increasing the frequency and intensity of extreme weather events. Source: IPCC AR6 WG1'
        }
    
    def verify_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Verify a claim against scientific sources"""
        try:
            main_claim = claim_data.get('main_claim', '')
            claim_type = claim_data.get('claim_type', 'unknown').lower()
            
            # Flexible verification triggers - checks for keywords instead of exact matches
            should_verify = any([
                'false' in claim_type,
                'conspiracy' in claim_type,
                'mislead' in claim_type,
                'denial' in claim_type,
                'hoax' in claim_type,
                'factual' in claim_type,  # Added: factual claims need verification
                claim_data.get('confidence', 1.0) < 0.5,  # Low confidence claims
                claim_data.get('needs_verification', False)
            ])
            
            if not should_verify:
                return {
                    'claim': main_claim,
                    'verification_needed': False,
                    'status': 'skipped',
                    'reason': f'Claim type "{claim_type}" does not require verification'
                }
            
            prompt = f"""You are a climate scientist fact-checker. Verify this claim:

CLAIM: "{main_claim}"
CLAIM TYPE: {claim_type}

VERIFIED SCIENTIFIC FACTS:
{json.dumps(self.scientific_facts, indent=2)}

Analyze the claim against scientific consensus. Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "verdict": "TRUE/FALSE/MISLEADING/UNVERIFIABLE",
    "scientific_consensus": <0-100 percentage of scientists who agree with the CORRECT position>,
    "supporting_evidence": ["evidence that supports the CORRECT position"],
    "contradicting_evidence": ["evidence that contradicts the CLAIM if it's false"],
    "authoritative_sources": ["NASA", "IPCC", "NOAA"],
    "explanation": "Clear 2-3 sentence explanation of why the claim is true/false",
    "confidence": <0.0 to 1.0>
}}"""

            response = self.model.generate_content(prompt)
            text_response = response.text.strip()
            
            # Remove markdown code fences if present
            text_response = text_response.replace('``````', '').strip()
            
            # Extract JSON robustly
            json_start = text_response.find('{')
            json_end = text_response.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No JSON found in response")
            
            result = json.loads(text_response[json_start:json_end])
            
            result['original_claim'] = main_claim
            result['claim_type'] = claim_type
            result['status'] = 'success'
            
            return result
            
        except json.JSONDecodeError as e:
            return {
                'original_claim': claim_data.get('main_claim', ''),
                'status': 'error',
                'error_type': 'JSON_PARSE_ERROR',
                'error_message': f"Could not parse JSON: {str(e)}"
            }
        except Exception as e:
            return {
                'original_claim': claim_data.get('main_claim', ''),
                'status': 'error',
                'error_type': type(e).__name__,
                'error_message': str(e)
            }
    
    def execute(self, detected_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Verify all detected claims"""
        print(f"\nğŸ”¬ Agent 2: Verifying claims against scientific sources...")
        
        results = []
        verification_count = 0
        
        for i, claim in enumerate(detected_claims, 1):
            if claim.get('status') == 'success':
                print(f"  â†’ Processing claim {i}/{len(detected_claims)}...")
                result = self.verify_claim(claim)
                if result.get('status') == 'success':
                    verification_count += 1
                results.append(result)
            else:
                results.append({'status': 'skipped', 'reason': 'Claim detection failed'})
        
        successful = sum(1 for r in results if r.get('status') == 'success')
        print(f"âœ… Agent 2 Complete: {successful}/{len(detected_claims)} claims verified\n")
        
        return results

# Test Agent 2
agent2 = SourceVerifierAgent(model, CONFIG)
verified_claims = agent2.execute(detected_claims)

# Display results
print("ğŸ”¬ VERIFICATION RESULTS:")
print("=" * 80)
for i, result in enumerate(verified_claims, 1):
    if result.get('status') == 'success':
        print(f"\n{i}. CLAIM: {result.get('original_claim', '')[:70]}...")
        print(f"   ğŸ“‹ Type: {result.get('claim_type', 'unknown')}")
        print(f"   âœ“ VERDICT: {result.get('verdict', 'N/A').upper()}")
        print(f"   ğŸ“Š Scientific Consensus: {result.get('scientific_consensus', 0)}%")
        print(f"   ğŸ”— Sources: {', '.join(result.get('authoritative_sources', []))}")
        print(f"   ğŸ’¡ Explanation: {result.get('explanation', '')}")
        print(f"   ğŸ�¯ Confidence: {result.get('confidence', 0):.2f}")
    elif result.get('status') == 'error':
        print(f"\n{i}. â�Œ ERROR: {result.get('error_message', 'Unknown error')}")
    else:
        print(f"\n{i}. â�­ï¸�  SKIPPED: {result.get('reason', 'Unknown reason' )}")



# ============================================================================
# AGENT 3 - EVIDENCE SYNTHESIZER 
# ============================================================================
class EvidenceSynthesizerAgent:
    """Agent 3: Synthesizes evidence into clear fact-checks"""
    
    def __init__(self, model):
        self.model = model
    
    def synthesize_evidence(self, verification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create clear fact-check from verification data"""
        try:
            if verification_data['status'] != 'success':
                return {'status': 'skipped'}
            
            claim = verification_data.get('original_claim', '')
            verdict = verification_data.get('verdict', '')
            evidence = verification_data.get('supporting_evidence', [])
            sources = verification_data.get('authoritative_sources', [])
            
            prompt = f"""Create a clear, accessible fact-check for the public:

CLAIM: "{claim}"
VERDICT: {verdict}
EVIDENCE: {json.dumps(evidence)}
SOURCES: {json.dumps(sources)}

Generate a fact-check in JSON format:
{{
    "headline": "",
    "summary": "<2-3 sentence summary for general public>",
    "key_evidence": ["point1", "point2", "point3"],
    "what_science_says": "",
    "citations": [
        {{"source": "NASA", "fact": "specific fact", "url": "https://climate.nasa.gov/..."}},
        {{"source": "IPCC", "fact": "specific fact", "url": "https://ipcc.ch/..."}}
    ],
    "bottom_line": ""
}}"""

            response = self.model.generate_content(prompt)
            text_response = response.text
            
            # Extract JSON
            json_start = text_response.find('{')
            json_end = text_response.rfind('}') + 1
            result = json.loads(text_response[json_start:json_end])
            
            result['original_claim'] = claim
            result['verdict'] = verdict
            result['status'] = 'success'
            
            return result
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    def execute(self, verified_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synthesize evidence for all verified claims"""
        print(f"ğŸ“š Agent 3: Synthesizing evidence into fact-checks...")
        
        results = []
        for i, claim in enumerate(verified_claims, 1):
            if claim['status'] == 'success':
                print(f"  â†’ Creating fact-check {i}...")
                result = self.synthesize_evidence(claim)
                results.append(result)
            else:
                results.append({'status': 'skipped'})
        
        successful = sum(1 for r in results if r['status'] == 'success')
        print(f"âœ… Agent 3 Complete: {successful} fact-checks created")
        
        return results

# Test Agent 3
agent3 = EvidenceSynthesizerAgent(model)
synthesized_evidence = agent3.execute(verified_claims)

# Display results
print("\nğŸ“š FACT-CHECK SUMMARIES:")
print("=" * 70)
for i, result in enumerate(synthesized_evidence, 1):
    if result['status'] == 'success':
        print(f"\n{i}. {result.get('headline', 'N/A')}")
        print(f"   Verdict: {result.get('verdict', 'N/A')}")
        print(f"   Summary: {result.get('summary', '')}")
        print(f"   Bottom Line: {result.get('bottom_line', '')}")
        print(f"   Citations: {len(result.get('citations', []))} sources")


# ============================================================================
# AGENT 4 - GIVE CREDEBILITY SCORE 
# ============================================================================
class CredibilityScorerAgent:
    """Agent 4: Calculates credibility scores for claims"""
    
    def __init__(self, config):
        self.config = config
        self.weights = config['credibility_weights']
    
    def calculate_score(self, verification_data: Dict[str, Any], 
                       evidence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive credibility score"""
        try:
            if verification_data['status'] != 'success':
                return {'status': 'skipped'}
            
            # Extract data
            verdict = verification_data.get('verdict', '')
            consensus = verification_data.get('scientific_consensus', 0)
            sources = verification_data.get('authoritative_sources', [])
            citations = evidence_data.get('citations', [])
            
            # Calculate component scores
            # 1. Peer Review Score (0-100)
            peer_review_score = min(len(citations) * 20, 100)  # Max 5 citations
            
            # 2. Source Quality Score (0-100)
            trusted_sources = {'NASA', 'IPCC', 'NOAA', 'NSIDC'}
            source_quality = (len([s for s in sources if s in trusted_sources]) / 
                            max(len(sources), 1) * 100) if sources else 0
            
            # 3. Scientific Consensus Score (0-100)
            consensus_score = consensus
            
            # Weighted final score
            final_score = (
                peer_review_score * self.weights['peer_reviewed'] +
                source_quality * self.weights['government_source'] +
                consensus_score * self.weights['scientific_consensus']
            )
            
            # Determine rating
            if final_score >= 80:
                rating = 'Highly Credible'
            elif final_score >= 60:
                rating = 'Credible'
            elif final_score >= 40:
                rating = 'Questionable'
            else:
                rating = 'Not Credible'
            
            return {
                'claim': verification_data.get('original_claim', ''),
                'verdict': verdict,
                'credibility_score': round(final_score, 1),
                'rating': rating,
                'breakdown': {
                    'peer_review': round(peer_review_score, 1),
                    'source_quality': round(source_quality, 1),
                    'scientific_consensus': round(consensus_score, 1)
                },
                'citations_count': len(citations),
                'trusted_sources': len([s for s in sources if s in trusted_sources]),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error_message': str(e)
            }
    
    def execute(self, verified_claims: List[Dict[str, Any]], 
                synthesized_evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score all claims"""
        print(f"â­� Agent 4: Calculating credibility scores...")
        
        results = []
        for i, (verification, evidence) in enumerate(zip(verified_claims, synthesized_evidence), 1):
            if verification['status'] == 'success' and evidence['status'] == 'success':
                print(f"  â†’ Scoring claim {i}...")
                result = self.calculate_score(verification, evidence)
                results.append(result)
            else:
                results.append({'status': 'skipped'})
        
        successful = sum(1 for r in results if r['status'] == 'success')
        print(f"âœ… Agent 4 Complete: {successful} scores calculated")
        
        return results

# Test Agent 4
agent4 = CredibilityScorerAgent(CONFIG)
credibility_scores = agent4.execute(verified_claims, synthesized_evidence)

# Display results
print("\nâ­� CREDIBILITY SCORES:")
print("=" * 70)
for i, result in enumerate(credibility_scores, 1):
    if result['status'] == 'success':
        print(f"\n{i}. Claim: {result.get('claim', '')[:50]}...")
        print(f"   Verdict: {result.get('verdict', 'N/A')}")
        print(f"   Credibility: {result.get('credibility_score', 0)}/100 ({result.get('rating', '')})")
        print(f"   Breakdown:")
        breakdown = result.get('breakdown', {})
        print(f"     - Peer Review: {breakdown.get('peer_review', 0)}/100")
        print(f"     - Source Quality: {breakdown.get('source_quality', 0)}/100")
        print(f"     - Scientific Consensus: {breakdown.get('scientific_consensus', 0)}/100")


# ============================================================================
# AGENT 5 - COUNTER NARRATIVE GENERATOR
# ============================================================================

class CounterNarrativeGeneratorAgent:
    """Agent 5: Generates detailed, shareable counter-narratives with visual elements"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.counter_narratives = []
    
    def generate_narrative(self, verified_claim: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive counter-narrative"""
        try:
            original_claim = verified_claim.get('original_claim', '')
            verdict = verified_claim.get('verdict', 'UNKNOWN')
            explanation = verified_claim.get('explanation', '')
            sources = verified_claim.get('authoritative_sources', [])
            evidence = verified_claim.get('supporting_evidence', [])
            
            prompt = f"""You are an expert science communicator creating compelling counter-narratives to climate misinformation.

MISINFORMATION: "{original_claim}"
VERDICT: {verdict}
SCIENTIFIC EXPLANATION: {explanation}
AUTHORITATIVE SOURCES: {', '.join(sources)}
SUPPORTING EVIDENCE: {', '.join(evidence)}

Create a comprehensive, engaging counter-narrative package in JSON format:
{{
    "verdict": "{verdict}",
    "short_summary": "<1-sentence summary for busy readers>",
    
    "social_media": {{
        "twitter_280": "<Tweet under 280 chars with emojis, fact, and #FactCheck>",
        "twitter_extended": "<280-500 char version with more details and credibility markers>",
        "facebook_long": "<800-1200 char detailed post with opening hook, explanation, and call-to-action>",
        "instagram_caption": "<Caption with emojis, key takeaway, and save/share CTA>",
        "linkedin_professional": "<Professional post for workplace/academic audience with sources>",
        "tiktok_script": "<15-30 second video script with hook, main point, and CTA>"
    }},
    
    "infographic_text": {{
        "headline": "<Bold eye-catching headline>",
        "claim_vs_reality": [
            {{"false_claim": "...", "scientific_reality": "..."}},
            {{"false_claim": "...", "scientific_reality": "..."}}
        ],
        "key_numbers": ["stat 1", "stat 2", "stat 3"],
        "visual_elements": ["emoji suggestion 1", "emoji suggestion 2"]
    }},
    
    "detailed_explanation": {{
        "why_its_wrong": "<2-3 paragraphs explaining why the claim is false/misleading with concrete examples>",
        "scientific_consensus": "<What the scientific community actually agrees on with percentages>",
        "real_world_examples": ["example 1 with specifics", "example 2 with specifics"],
        "common_misconceptions": ["misconception 1 addressed", "misconception 2 addressed"]
    }},
    
    "credibility_markers": [
        "peer-reviewed research showing...",
        "official data from [SOURCE] demonstrates...",
        "consensus among X% of scientists that..."
    ],
    
    "engagement_hooks": [
        "<Question to spark conversation>",
        "<Surprising fact that engages audience>",
        "<Personal impact statement>"
    ],
    
    "hashtags": ["#FactCheck", "#ClimateScience", "#ScienceMatters", "<topic-specific tags>"],
    
    "call_to_action": "<Persuasive CTA encouraging shares, learning, or action>",
    
    "sources_cited": [
        {{"source": "source name", "link": "url", "why_credible": "explanation"}}
    ]
}}
"""

            response = self.model.generate_content(prompt)
            text_response = response.text.strip()
            
            # Remove markdown fences
            text_response = text_response.replace('```json', '').replace('```', '').strip()
            
            # Extract JSON
            json_start = text_response.find('{')
            json_end = text_response.rfind('}') + 1
            
            if json_start == -1 or json_end <= json_start:
                raise ValueError("No JSON found in response")
            
            result = json.loads(text_response[json_start:json_end])
            result['original_misinformation'] = original_claim
            result['status'] = 'success'
            
            return result
            
        except Exception as e:
            return {
                'original_misinformation': original_claim,
                'status': 'error',
                'error_message': str(e)
            }
    
    def execute(self, verified_claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate counter-narratives for all verified claims"""
        print(f"\nâœ�ï¸�  Agent 5: Generating detailed counter-narratives...")
        
        results = []
        successful = 0
        
        for i, claim in enumerate(verified_claims, 1):
            if claim.get('status') == 'success':
                print(f"  â†’ Creating counter-narrative {i}...")
                narrative = self.generate_narrative(claim)
                if narrative.get('status') == 'success':
                    successful += 1
                results.append(narrative)
            else:
                results.append({'status': 'skipped', 'reason': 'Verification failed'})
        
        print(f"âœ… Agent 5 Complete: {successful} counter-narratives created\n")
        self.counter_narratives = results
        return results
    print("âœ… Agent 5 (Counter-Narrative Generator) class defined successfully!")



# ============================================================================
# ORCHESTRATOR 
# ============================================================================

class ClimateFactCheckOrchestrator:
    """Orchestrates all 5 agents for complete fact-checking pipeline"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Initialize all agents
        self.agent1 = ClaimDetectorAgent(model, config)
        self.agent2 = SourceVerifierAgent(model, config)
        self.agent3 = EvidenceSynthesizerAgent(model)
        self.agent4 = CredibilityScorerAgent(config)
        self.agent5 = CounterNarrativeGeneratorAgent(model, config)
    
    def execute(self, claims: List[str]) -> Dict[str, Any]:
        """Execute complete fact-checking pipeline"""
        print("=" * 80)
        print("ğŸŒ� CLIMATE MISINFORMATION COMBAT AGENT - FULL PIPELINE")
        print("=" * 80)
        print(f"Processing {len(claims)} claims...")
        print("=" * 80)
        print()
        
        # Sequential agent execution
        detected = self.agent1.execute(claims)
        print()
        
        verified = self.agent2.execute(detected)
        print()
        
        synthesized = self.agent3.execute(verified)
        print()
        
        scored = self.agent4.execute(verified, synthesized)
        print()
        
        # CORRECT usage of Agent 5: only pass verified claims
        counter_narratives = self.agent5.execute(verified)
        print()
        
        # Compile final report preserving detailed outputs
        report = self._compile_report(claims, detected, verified, synthesized, scored, counter_narratives)
        
        print("=" * 80)
        print("âœ… FACT-CHECKING PIPELINE COMPLETE!")
        print("=" * 80)
        
        return report
    
    def _compile_report(self, claims, detected, verified, synthesized, scored, counter_narratives) -> Dict[str, Any]:
        """Compile comprehensive report preserving full agent outputs"""
        fact_checks = []
        
        for i in range(len(claims)):
            if (detected[i]['status'] == 'success' and 
                verified[i]['status'] == 'success' and
                synthesized[i]['status'] == 'success' and
                scored[i]['status'] == 'success' and
                counter_narratives[i]['status'] == 'success'):
                
                fact_checks.append({
                    'original_claim': claims[i],
                    'agent1_detection': detected[i],        # Full agent 1 output
                    'agent2_verification': verified[i],    # Full agent 2 output
                    'agent3_synthesis': synthesized[i],    # Full agent 3 output
                    'agent4_credibility': scored[i],       # Full agent 4 output
                    'agent5_counter_narrative': counter_narratives[i]  # Full agent 5 output
                })
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'total_claims_analyzed': len(claims),
            'successful_fact_checks': len(fact_checks),
            'system_config': {
                'consensus_threshold': self.config['consensus_threshold'],
                'scientific_sources': list(self.config['scientific_sources'].keys())
            },
            'fact_checks': fact_checks,
            'summary': {
                'false_claims': sum(1 for fc in fact_checks if 'FALSE' in fc['agent2_verification']['verdict']),
                'true_claims': sum(1 for fc in fact_checks if 'TRUE' in fc['agent2_verification']['verdict']),
                'avg_credibility': sum(fc['agent4_credibility']['credibility_score'] for fc in fact_checks) / len(fact_checks) if fact_checks else 0
            }
        }
    
    def save_report(self, report: Dict[str, Any], filename='climate_fact_check_report.json'):
        """Save report to JSON file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"ğŸ’¾ Report saved to {filename}")
        return filename
    
    def display_summary(self, report: Dict[str, Any]):
        """Display human-readable summary"""
        print("\n" + "=" * 80)
        print("ğŸ“Š FACT-CHECK SUMMARY REPORT")
        print("=" * 80)
        
        print(f"\nğŸ“… Report Date: {report['report_timestamp']}")
        print(f"ğŸ“‹ Claims Analyzed: {report['total_claims_analyzed']}")
        print(f"âœ… Successful Fact-Checks: {report['successful_fact_checks']}")
        
        summary = report['summary']
        print(f"\nğŸ“Š RESULTS BREAKDOWN:")
        print(f"  â�Œ False Claims: {summary['false_claims']}")
        print(f"  âœ… True Claims: {summary['true_claims']}")
        print(f"  ğŸ“ˆ Avg Credibility Score: {summary['avg_credibility']:.1f}/100")
        
        print(f"\nğŸ”¬ DETAILED FACT-CHECKS:")
        print("=" * 80)
        
        for i, fc in enumerate(report['fact_checks'], 1):
            print(f"\n{i}. CLAIM: {fc['original_claim']}")
            print(f"   ğŸ“Œ Detection: {fc['agent1_detection']}")
            print(f"   ğŸ”� Verification: {fc['agent2_verification']}")
            print(f"   ğŸ“� Synthesis: {fc['agent3_synthesis']}")
            print(f"   â­� Credibility: {fc['agent4_credibility']}")
            print(f"   ğŸ“± Counter-Narrative: {fc['agent5_counter_narrative']}")
            
print("âœ… Orchestrator class defined successfully!")



# ============================================================================
# WEB SCRAPPER
# ============================================================================
class LiveClaimScraper:
    """Scrapes climate-related claims from various sources with fallbacks"""
    
    def __init__(self, config):
        self.config = config
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
    
    def scrape_climate_news_rss(self, query='climate change', num_results=5):
        """Scrape climate news from Google News RSS"""
        print(f"\nğŸ“° Scraping climate news for '{query}'...")
        
        try:
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                print(f"âš ï¸�  News request failed: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')[:num_results]
            
            claims = []
            for item in items:
                title = item.find('title')
                link = item.find('link')
                
                if title:
                    claims.append({
                        'text': title.text.strip(),
                        'source': 'Google News',
                        'url': link.text if link else ''
                    })
            
            print(f"âœ… Found {len(claims)} news articles")
            return claims
            
        except Exception as e:
            print(f"âš ï¸�  News scraping error: {e}")
            return []
    
    def execute(self, sources=['news'], claims_per_source=5):
        """Scrape claims from sources"""
        print("=" * 80)
        print("ğŸŒ� LIVE WEB SCRAPING INITIATED")
        print("=" * 80)
        
        all_claims: List[str] = []
        
        if 'news' in sources:
            news_claims = self.scrape_climate_news_rss(num_results=claims_per_source)
            if news_claims:
                all_claims.extend([c['text'] for c in news_claims])
            time.sleep(1)
        
        print(f"\nâœ… Total claims collected: {len(all_claims)}")
        print("=" * 80)
        
        return all_claims

# Initialize scraper and fetch live claims
print("\nğŸ�¯ FETCHING LIVE CLIMATE CLAIMS...")
scraper = LiveClaimScraper({'model': None})
live_claims = scraper.execute(sources=['news'], claims_per_source=5)

# Decide which claims to use
if live_claims and len(live_claims) > 0:
    CLAIMS_TO_ANALYZE = live_claims
    print("\nâœ… Using LIVE CLAIMS from Google News!")
    print(f"ğŸ“Š Total: {len(CLAIMS_TO_ANALYZE)} real-time headlines")
else:
    CLAIMS_TO_ANALYZE = SAMPLE_CLAIMS
    print("\nâš ï¸�  Using SAMPLE_CLAIMS as fallback")
    print(f"ğŸ“Š Total: {len(CLAIMS_TO_ANALYZE)} sample claims")

# Display claims (for quick confirmation)
print("\nğŸ“‹ CLAIMS TO BE FACT-CHECKED:")
print("=" * 80)
for i, claim in enumerate(CLAIMS_TO_ANALYZE, 1):
    preview = claim[:80] + "..." if len(claim) > 80 else claim
    print(f"{i}. {preview}")
print("=" * 80)


# ============================================================================
# FINAL PIPELINE
# ============================================================================

print("\n" + "=" * 80)
print("ğŸ�¯ RUNNING FACT-CHECKING PIPELINE")
print("=" * 80)
print(f"Processing {len(CLAIMS_TO_ANALYZE)} claims through 5-agent system...")
print("=" * 80)

# Create orchestrator and run pipeline
orchestrator = ClimateFactCheckOrchestrator(model, CONFIG)
final_report = orchestrator.execute(CLAIMS_TO_ANALYZE)

# Save and display results
orchestrator.save_report(final_report)
orchestrator.display_summary(final_report)

print("\n" + "=" * 80)
print("âœ… COMPLETE! Check climate_fact_check_report.json for full results")
print("=" * 80)



# ============================================================================
# ENHANCED PDF GENERATOR - VERBOSE OUTPUT WITH VALIDATION
# ============================================================================

!pip install -q reportlab

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from datetime import datetime
import json

class EnhancedPDFGenerator:
    """Enhanced PDF with full agent details and validation"""
    
    def __init__(self, filename='Climate_FactCheck_Report_COMPLETE.pdf'):
        self.filename = filename
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles"""
        if 'ReportTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ReportTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1F4788'),
                spaceAfter=12,
                alignment=TA_CENTER,
                bold=True
            ))
        
        if 'ClaimNumber' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ClaimNumber',
                parent=self.styles['Heading2'],
                fontSize=14,
                textColor=colors.HexColor('#FFFFFF'),
                backColor=colors.HexColor('#2A5CAE'),
                spaceAfter=12,
                spaceBefore=12,
                bold=True,
                leftIndent=10,
                rightIndent=10
            ))
        
        if 'AgentHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='AgentHeader',
                parent=self.styles['Heading3'],
                fontSize=11,
                textColor=colors.HexColor('#FFFFFF'),
                backColor=colors.HexColor('#3D5A80'),
                spaceAfter=8,
                spaceBefore=10,
                bold=True,
                leftIndent=8
            ))
        
        if 'DetailText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='DetailText',
                parent=self.styles['Normal'],
                fontSize=9,
                leading=11,
                spaceBefore=4,
                spaceAfter=4,
                alignment=TA_LEFT
            ))
    
    def clean_text(self, text, max_len=None):
        """Clean text for PDF"""
        if text is None:
            return "N/A"
        
        text_str = str(text).strip()
        
        # Escape HTML special chars
        text_str = text_str.replace('&', '&amp;')
        text_str = text_str.replace('<', '&lt;')
        text_str = text_str.replace('>', '&gt;')
        text_str = text_str.replace('"', '&quot;')
        
        if max_len and len(text_str) > max_len:
            text_str = text_str[:max_len] + "..."
        
        return text_str if text_str else "N/A"
    
    def generate_pdf(self, report_data):
        """Generate comprehensive PDF"""
        doc = SimpleDocTemplate(
            self.filename,
            pagesize=letter,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch,
            leftMargin=0.6*inch,
            rightMargin=0.6*inch
        )
        
        story = []
        
        # ===== TITLE PAGE =====
        story.append(Paragraph("ğŸŒ� CLIMATE FACT-CHECK REPORT", self.styles['ReportTitle']))
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}", self.styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        summary_text = f"""
        <b>Total Claims Analyzed:</b> {report_data['total_claims_analyzed']}<br/>
        <b>Successfully Processed:</b> {report_data['successful_fact_checks']}<br/>
        <b>True Claims:</b> {report_data['summary']['true_claims']}<br/>
        <b>False Claims:</b> {report_data['summary']['false_claims']}<br/>
        <b>Average Credibility Score:</b> {report_data['summary']['avg_credibility']:.1f}/100
        """
        story.append(Paragraph(summary_text, self.styles['DetailText']))
        story.append(PageBreak())
        
        # ===== PROCESS EACH CLAIM WITH FULL DETAIL =====
        for idx, fact_check in enumerate(report_data['fact_checks'], 1):
            # Claim header
            story.append(Paragraph(f"CLAIM #{idx}", self.styles['ClaimNumber']))
            story.append(Spacer(1, 0.08*inch))
            
            # Original claim
            original = self.clean_text(fact_check.get('original_claim', 'N/A'))
            story.append(Paragraph(f"<b>Original Claim:</b><br/>{original}", self.styles['DetailText']))
            story.append(Spacer(1, 0.1*inch))
            
            # ===== AGENT 1: DETECTION =====
            agent1 = fact_check.get('agent1_detection', {})
            story.append(Paragraph("ğŸ”� AGENT 1: CLAIM DETECTION & ANALYSIS", self.styles['AgentHeader']))
            
            a1_details = f"""
            <b>Claim Type:</b> {self.clean_text(agent1.get('claim_type'))}<br/>
            <b>Climate Related:</b> {self.clean_text(agent1.get('is_climate_related'))}<br/>
            <b>Confidence:</b> {agent1.get('confidence', 0):.2f}<br/>
            <b>Needs Verification:</b> {self.clean_text(agent1.get('needs_verification'))}<br/>
            <b>Keywords Detected:</b> {', '.join(agent1.get('keywords', [])) or 'N/A'}<br/>
            <b>Reasoning:</b><br/>{self.clean_text(agent1.get('reasoning'), max_len=800)}
            """
            story.append(Paragraph(a1_details, self.styles['DetailText']))
            story.append(Spacer(1, 0.12*inch))
            
            # ===== AGENT 2: VERIFICATION =====
            agent2 = fact_check.get('agent2_verification', {})
            story.append(Paragraph("âœ“ AGENT 2: SOURCE VERIFICATION & FACT-CHECK", self.styles['AgentHeader']))
            
            a2_details = f"""
            <b>Verdict:</b> {self.clean_text(agent2.get('verdict'))}<br/>
            <b>Scientific Consensus:</b> {agent2.get('scientific_consensus', 0)}%<br/>
            <b>Verification Confidence:</b> {agent2.get('confidence', 0):.2f}<br/>
            <b>Authoritative Sources Used:</b> {', '.join(agent2.get('authoritative_sources', [])) or 'N/A'}<br/>
            <b>Explanation:</b><br/>{self.clean_text(agent2.get('explanation'), max_len=1000)}<br/><br/>
            <b>Supporting Evidence:</b><br/>
            """
            for evidence in agent2.get('supporting_evidence', []):
                a2_details += f"â€¢ {self.clean_text(evidence, max_len=400)}<br/>"
            
            a2_details += "<br/><b>Contradicting Evidence:</b><br/>"
            for evidence in agent2.get('contradicting_evidence', []):
                a2_details += f"â€¢ {self.clean_text(evidence, max_len=400)}<br/>"
            
            story.append(Paragraph(a2_details, self.styles['DetailText']))
            story.append(Spacer(1, 0.12*inch))
            
            # ===== AGENT 3: SYNTHESIS =====
            agent3 = fact_check.get('agent3_synthesis', {})
            story.append(Paragraph("ğŸ“š AGENT 3: EVIDENCE SYNTHESIS & SUMMARY", self.styles['AgentHeader']))
            
            a3_details = f"""
            <b>Headline:</b> {self.clean_text(agent3.get('headline'), max_len=200)}<br/><br/>
            <b>Summary:</b><br/>{self.clean_text(agent3.get('summary'), max_len=1200)}<br/><br/>
            <b>Bottom Line:</b><br/>{self.clean_text(agent3.get('bottom_line'), max_len=600)}<br/><br/>
            <b>Citations & References:</b><br/>
            """
            for idx_cite, citation in enumerate(agent3.get('citations', []), 1):
                if isinstance(citation, dict):
                    a3_details += f"{idx_cite}. <b>{self.clean_text(citation.get('source'))}</b>: {self.clean_text(citation.get('fact'), max_len=300)}<br/>"
                else:
                    a3_details += f"{idx_cite}. {self.clean_text(citation, max_len=300)}<br/>"
            
            story.append(Paragraph(a3_details, self.styles['DetailText']))
            story.append(Spacer(1, 0.12*inch))
            
            # ===== AGENT 4: CREDIBILITY =====
            agent4 = fact_check.get('agent4_credibility', {})
            story.append(Paragraph("â­� AGENT 4: CREDIBILITY ASSESSMENT", self.styles['AgentHeader']))
            
            score = agent4.get('credibility_score', 0)
            
            a4_details = f"""
            <b>Credibility Score:</b> {score}/100<br/>
            <b>Rating:</b> {self.clean_text(agent4.get('rating'))}<br/><br/>
            <b>Scoring Breakdown:</b><br/>
            """
            for key, val in agent4.get('breakdown', {}).items():
                a4_details += f"â€¢ {key}: {val}<br/>"
            
            a4_details += f"<br/><b>Assessment Justification:</b><br/>{self.clean_text(agent4.get('justification'), max_len=800)}"
            
            story.append(Paragraph(a4_details, self.styles['DetailText']))
            story.append(Spacer(1, 0.12*inch))
            
            # ===== AGENT 5: COUNTER-NARRATIVE =====
            agent5 = fact_check.get('agent5_counter_narrative', {})
            story.append(Paragraph("ğŸ“± AGENT 5: COUNTER-NARRATIVE & SOCIAL MEDIA", self.styles['AgentHeader']))
            
            social = agent5.get('social_media', {})
            
            a5_details = f"""
            <b>Summary:</b><br/>{self.clean_text(agent5.get('short_summary'), max_len=500)}<br/><br/>
            
            <b>Twitter (280 chars):</b><br/>{self.clean_text(social.get('twitter_280'), max_len=300)}<br/><br/>
            
            <b>Twitter Extended:</b><br/>{self.clean_text(social.get('twitter_extended'), max_len=400)}<br/><br/>
            
            <b>Facebook Post:</b><br/>{self.clean_text(social.get('facebook_long'), max_len=500)}<br/><br/>
            
            <b>Instagram Caption:</b><br/>{self.clean_text(social.get('instagram_caption'), max_len=400)}<br/><br/>
            
            <b>LinkedIn Professional:</b><br/>{self.clean_text(social.get('linkedin_professional'), max_len=500)}<br/><br/>
            
            <b>TikTok Script:</b><br/>{self.clean_text(social.get('tiktok_script'), max_len=400)}<br/><br/>
            
            <b>Hashtags:</b> {' '.join(agent5.get('hashtags', [])) or 'N/A'}<br/>
            <b>Call-to-Action:</b> {self.clean_text(agent5.get('call_to_action'), max_len=200)}
            """
            
            story.append(Paragraph(a5_details, self.styles['DetailText']))
            story.append(PageBreak())
        
        # Build PDF
        try:
            doc.build(story)
            print(f"âœ… PDF GENERATED: {self.filename}")
            print(f"ğŸ“Š Claims included: {len(report_data['fact_checks'])}")
            return True
        except Exception as e:
            print(f"â�Œ PDF Generation Error: {str(e)}")
            return False


# ============================================================================
# EXECUTE WITH VALIDATION
# ============================================================================

print("=" * 80)
print("LOADING JSON REPORT & GENERATING ENHANCED PDF")
print("=" * 80)

try:
    with open('climate_fact_check_report.json', 'r') as f:
        report_data = json.load(f)
    
    print(f"âœ… JSON loaded - {report_data['total_claims_analyzed']} claims found")
    
    # Generate PDF
    pdf_gen = EnhancedPDFGenerator()
    success = pdf_gen.generate_pdf(report_data)
    
    if success:
        import os
        size_kb = os.path.getsize('Climate_FactCheck_Report_COMPLETE.pdf') / 1024
        print(f"âœ… File size: {size_kb:.1f} KB")
        print("\nâœ¨ PDF WITH FULL AGENT DETAILS IS READY!")
    
except Exception as e:
    print(f"â�Œ Error: {str(e)}")
    import traceback
    traceback.print_exc()



# ============================================================================
# CREATE SUBMISSION OUTPUT FILES
# ============================================================================

import os
import json
import csv
from datetime import datetime

print("\n" + "=" * 80)
print("ğŸ“¦ CREATING SUBMISSION OUTPUT FILES")
print("=" * 80)

# Verify final_report exists
if 'final_report' not in globals():
    print("â�Œ ERROR: final_report not found! Run all prior cells first.")
else:
    try:
        # 1. Save JSON Report
        report_json_path = 'climate_fact_check_report.json'
        with open(report_json_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        if os.path.exists(report_json_path):
            size_kb = os.path.getsize(report_json_path) / 1024
            print(f"âœ… JSON Report saved: {report_json_path} ({size_kb:.1f} KB)")
        else:
            print(f"â�Œ JSON save failed!")
        
        # 2. Generate PDF
        pdf_path = 'Climate_FactCheck_Report_COMPLETE.pdf'
        pdf_gen = ProductionPDFGenerator(pdf_path)
        pdf_gen.generate_pdf(final_report)
        
        if os.path.exists(pdf_path):
            size_kb = os.path.getsize(pdf_path) / 1024
            print(f"âœ… PDF Report generated: {pdf_path} ({size_kb:.1f} KB)")
        else:
            print(f"â�Œ PDF save failed!")
        
        # 3. Create Summary CSV
        csv_path = 'climate_fact_check_summary.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Claim_Number', 'Claim', 'Type', 'Verdict', 'Credibility_Score', 'Rating', 'Confidence']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for idx, fc in enumerate(final_report['fact_checks'], 1):
                try:
                    writer.writerow({
                        'Claim_Number': idx,
                        'Claim': fc['original_claim'][:100],
                        'Type': fc['agent1_detection'].get('claim_type', 'N/A'),
                        'Verdict': fc['agent2_verification'].get('verdict', 'N/A'),
                        'Credibility_Score': round(fc['agent4_credibility'].get('credibility_score', 0), 1),
                        'Rating': fc['agent4_credibility'].get('rating', 'N/A'),
                        'Confidence': round(fc['agent2_verification'].get('confidence', 0), 2)
                    })
                except Exception as e:
                    print(f"âš ï¸�  Skipped row {idx}: {str(e)}")
        
        if os.path.exists(csv_path):
            size_kb = os.path.getsize(csv_path) / 1024
            print(f"âœ… Summary CSV saved: {csv_path} ({size_kb:.1f} KB)")
        else:
            print(f"â�Œ CSV save failed!")
        
        # 4. Create Text Summary Report
        txt_path = 'climate_fact_check_summary.txt'
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ğŸŒ� CLIMATE FACT-CHECK REPORT SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Claims: {final_report['total_claims_analyzed']}\n")
            f.write(f"Successfully Processed: {final_report['successful_fact_checks']}\n")
            f.write(f"True Claims: {final_report['summary']['true_claims']}\n")
            f.write(f"False Claims: {final_report['summary']['false_claims']}\n")
            f.write(f"Average Credibility: {final_report['summary']['avg_credibility']:.1f}/100\n\n")
            
            for idx, fc in enumerate(final_report['fact_checks'], 1):
                f.write(f"\nCLAIM {idx}:\n")
                f.write(f"  Original: {fc['original_claim'][:150]}\n")
                f.write(f"  Verdict: {fc['agent2_verification'].get('verdict', 'N/A')}\n")
                f.write(f"  Credibility: {fc['agent4_credibility'].get('credibility_score', 'N/A')}/100\n")
                f.write(f"  Rating: {fc['agent4_credibility'].get('rating', 'N/A')}\n")
        
        if os.path.exists(txt_path):
            size_kb = os.path.getsize(txt_path) / 1024
            print(f"âœ… Text Summary saved: {txt_path} ({size_kb:.1f} KB)")
        else:
            print(f"â�Œ Text save failed!")
        
        # Final confirmation
        print("\n" + "=" * 80)
        print("ğŸ“¦ SUBMISSION FILES READY:")
        output_files = []
        for fname in [report_json_path, pdf_path, csv_path, txt_path]:
            if os.path.exists(fname):
                size = os.path.getsize(fname)
                output_files.append(f"   âœ… {fname} ({size:,} bytes)")
        
        print("\n".join(output_files))
        print(f"\nğŸ�¯ Total Output Files: {len(output_files)}/4")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nâ�Œ ERROR during file creation: {str(e)}")
        import traceback
        traceback.print_exc()


