%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
    !pip install --no-deps unsloth


%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps git+https://github.com/huggingface/transformers.git # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N
!pip install transformers -U


from unsloth import FastModel
import torch
from transformers import TextStreamer
import gc
import json
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Union
import requests  # Add this if not already there
from IPython.display import display, HTML  # Add these
import base64  # Add this
from PIL import Image  # Add this if not already there
from io import BytesIO  # Add this if not already there

print("ğŸŒ‰ MindBridge Accessibility Assistant")
print("ğŸš€ Powered by Gemma 3n Multimodal AI")
print("=" * 60)


fourbit_models = [
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
]

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E4B-it",
    dtype = None,
    max_seq_length = 1024,
    load_in_4bit = True,
    full_finetuning = False,
)

print("âœ… Gemma 3n model loaded successfully!")


@dataclass
class AccessibilityResponse:
    scene_type: str
    immediate_area: str
    navigation: str
    safety_alerts: List[str]
    audio_summary: str
    confidence: float


def do_gemma_3n_inference_simple(model, messages, max_new_tokens=128):
    """Original simple inference with streaming output"""
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    
    _ = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0, 
        top_p=0.95, 
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    
    del inputs
    torch.cuda.empty_cache()
    gc.collect()

def do_gemma_3n_inference_capture(model, messages, max_new_tokens=256):
    """Inference that captures response text for processing"""
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=1.0, 
        top_p=0.95, 
        top_k=64,
        do_sample=True,
    )
    
    generated_tokens = outputs[0][len(inputs["input_ids"][0]):]
    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    print(response_text)  # Show to user
    
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    return response_text


def create_accessibility_prompt(mode="navigation"):
    """Create mode-specific prompts for accessibility"""
    base_prompt = "You are MindBridge, an accessibility assistant for visually impaired users.\n\n"
    
    mode_instructions = {
        "navigation": "Focus on spatial navigation: pathways, obstacles, directions, and safe movement.",
        "reading": "Focus on text and signs: read all visible text, identify important information.",
        "shopping": "Focus on products and items: identify objects, brands, prices, locations.",
        "safety": "Focus on hazards and safety: identify dangers, risks, and safety concerns."
    }
    
    instruction = mode_instructions.get(mode, mode_instructions["navigation"])
    
    return f"""{base_prompt}MODE: {mode.upper()}
{instruction}

Respond in this exact JSON format:
{{
    "scene_type": "brief environment description",
    "immediate_area": "what's within 3 feet of user",
    "navigation": "step-by-step movement guidance",
    "safety_alerts": ["list", "of", "safety", "concerns"],
    "audio_summary": "2-3 sentence audio description",
    "confidence": 0.8
}}

IMPORTANT: Respond ONLY with valid JSON, no other text."""

def create_simple_prompt(request="Describe this image for accessibility"):
    """Simple prompt for basic analysis"""
    return f"""You are MindBridge, an accessibility assistant.

{request}

Provide:
- SCENE: What type of environment this is
- IMMEDIATE: What's directly around the user
- NAVIGATION: How to move safely
- SAFETY: Any hazards to note
- SUMMARY: Brief audio description"""


def parse_accessibility_response(response_text):
    """Parse Gemma 3n response into structured format"""
    try:
        # Look for JSON in the response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_data = json.loads(json_match.group())
            return AccessibilityResponse(
                scene_type=json_data.get("scene_type", "Unknown"),
                immediate_area=json_data.get("immediate_area", "Analysis in progress"),
                navigation=json_data.get("navigation", "Navigation guidance available"),
                safety_alerts=json_data.get("safety_alerts", ["General awareness recommended"]),
                audio_summary=json_data.get("audio_summary", "Accessibility analysis complete"),
                confidence=json_data.get("confidence", 0.7)
            )
    except:
        pass
    
    # Fallback if JSON parsing fails
    return AccessibilityResponse(
        scene_type="Analyzed scene",
        immediate_area=response_text[:100] + "...",
        navigation="Review full response for guidance",
        safety_alerts=["Review response for safety information"],
        audio_summary=response_text[:150] + "...",
        confidence=0.6
    )


def analyze_scene(image_url, mode="navigation", show_details=True):
    """Analyze scene with structured output"""
    print(f"ğŸ”� Analyzing: {image_url}")
    print(f"ğŸ�¯ Mode: {mode.upper()}")
    print("-" * 50)
    
    accessibility_prompt = create_accessibility_prompt(mode)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_url},
            {"type": "text", "text": accessibility_prompt}
        ]
    }]
    
    print("ğŸ§  Gemma 3n Response:")
    response_text = do_gemma_3n_inference_capture(model, messages, max_new_tokens=300)
    
    structured_response = parse_accessibility_response(response_text)
    
    if show_details:
        print(f"\nğŸ“‹ Structured Analysis:")
        print(f"ğŸ�—ï¸�  Scene: {structured_response.scene_type}")
        print(f"ğŸ“� Immediate: {structured_response.immediate_area}")
        print(f"ğŸ§­ Navigation: {structured_response.navigation}")
        print(f"âš ï¸�  Safety: {'; '.join(structured_response.safety_alerts)}")
        print(f"ğŸ”Š Audio: {structured_response.audio_summary}")
        print(f"ğŸ“Š Confidence: {structured_response.confidence:.1%}")
    
    print("\n" + "="*60)
    return structured_response

def analyze_simple(image_url, request="Provide accessibility guidance"):
    """Simple analysis with basic prompt"""
    print(f"ğŸ”� Simple Analysis: {image_url}")
    print(f"ğŸ“� Request: {request}")
    print("-" * 40)
    
    simple_prompt = create_simple_prompt(request)
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_url},
            {"type": "text", "text": simple_prompt}
        ]
    }]
    
    print("ğŸ§  Response:")
    do_gemma_3n_inference_simple(model, messages, max_new_tokens=200)
    print("\n" + "="*40)

# =============================================================================
# AUDIO ANALYSIS FUNCTIONS (Add these to Part 7)
# =============================================================================

def analyze_with_audio(image_url: str, audio_file: str, mode: str = "navigation"):
    """Analyze scene with both image and audio input"""
    print(f"ğŸ”� Multimodal Analysis: Image + Audio")
    print(f"ğŸ�¯ Mode: {mode.upper()}")
    print(f"ğŸ“¸ Image: {image_url}")
    print(f"ğŸ�µ Audio: {audio_file}")
    print("-" * 50)
    
    # Display image
    display_image_inline(image_url, width=400, title=f"Multimodal Analysis: {mode.upper()}")
    
    # Create multimodal prompt
    accessibility_prompt = create_accessibility_prompt(mode)
    multimodal_prompt = f"""{accessibility_prompt}

IMPORTANT: You have both visual and audio information. Consider:
- What you SEE in the image
- What you HEAR in the audio
- How the audio context affects the visual scene
- Combined guidance using both inputs

Provide enhanced accessibility guidance that uses both visual and audio cues."""

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_url},
            {"type": "audio", "audio": audio_file},
            {"type": "text", "text": multimodal_prompt}
        ]
    }]
    
    print("ğŸ§  Gemma 3n Multimodal Response:")
    response_text = do_gemma_3n_inference_capture(model, messages, max_new_tokens=350)
    
    structured_response = parse_accessibility_response(response_text)
    
    print(f"\nğŸ“‹ Multimodal Analysis:")
    print(f"ğŸ�—ï¸�  Scene: {structured_response.scene_type}")
    print(f"ğŸ“� Immediate: {structured_response.immediate_area}")
    print(f"ğŸ§­ Navigation: {structured_response.navigation}")
    print(f"âš ï¸� Safety: {'; '.join(structured_response.safety_alerts)}")
    print(f"ğŸ”Š Audio Summary: {structured_response.audio_summary}")
    
    print("\n" + "="*60)
    return structured_response

def analyze_audio_only(audio_file: str, mode: str = "navigation"):
    """Analyze audio-only input for accessibility guidance"""
    print(f"ğŸ�µ Audio-Only Analysis")
    print(f"ğŸ�¯ Mode: {mode.upper()}")
    print(f"ğŸ�§ Audio: {audio_file}")
    print("-" * 40)
    
    # Create audio-specific accessibility prompt
    audio_prompt = f"""You are MindBridge accessibility assistant analyzing audio content.

MODE: {mode.upper()}

Listen to this audio and provide accessibility guidance based on what you hear.

Respond in JSON format:
{{
    "scene_type": "environment type based on audio",
    "immediate_area": "what sounds suggest about immediate surroundings",
    "navigation": "movement guidance based on audio cues",
    "safety_alerts": ["safety concerns identified from audio"],
    "assistance_tips": ["how to use audio cues for accessibility"],
    "next_actions": "recommended actions based on sounds",
    "audio_summary": "accessibility guidance based on what you hear"
}}

Focus on:
- Spatial awareness from sound
- Safety hazards detected through audio
- Navigation cues from environmental sounds
- Accessibility guidance using audio information"""

    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_file},
            {"type": "text", "text": audio_prompt}
        ]
    }]
    
    print("ğŸ§  Audio Analysis:")
    response_text = do_gemma_3n_inference_capture(model, messages, max_new_tokens=250)
    
    # Parse the audio response
    structured_response = parse_accessibility_response(response_text)
    
    print(f"\nğŸ“‹ Audio Analysis Results:")
    print(f"ğŸ�µ Scene: {structured_response.scene_type}")
    print(f"ğŸ“� Surroundings: {structured_response.immediate_area}")
    print(f"ğŸ§­ Audio Navigation: {structured_response.navigation}")
    print(f"âš ï¸� Audio Safety: {'; '.join(structured_response.safety_alerts)}")
    print(f"ğŸ”Š Summary: {structured_response.audio_summary}")
    
    print("\n" + "="*40)
    return structured_response

def quick_audio_test(audio_file: str, request: str = "Describe what you hear for accessibility"):
    """Quick audio test with minimal output"""
    print(f"âš¡ Quick Audio Test: {audio_file}")
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_file},
            {"type": "text", "text": f"Briefly {request}"}
        ]
    }]
    
    print("ğŸ§  Quick Audio Analysis:")
    do_gemma_3n_inference_simple(model, messages, max_new_tokens=150)
    print("\n" + "="*40)



class MindBridgeAssistant:
    def __init__(self):
        self.session_history = []
        self.available_modes = ["navigation", "reading", "shopping", "safety"]
        print("âœ… MindBridge Assistant initialized")
        print(f"ğŸ�¯ Available modes: {', '.join(self.available_modes)}")
    
    def analyze(self, image_url: str, mode: str = "navigation", context: str = None):
        """Main analysis function"""
        print(f"\nğŸŒ‰ MindBridge Analysis")
        print(f"ğŸš€ Powered by Gemma 3n")
        print(f"ğŸ�¯ Mode: {mode.upper()}")
        if context:
            print(f"ğŸ“� Context: {context}")
        print("=" * 50)
        
        start_time = time.time()
        
        if mode not in self.available_modes:
            print(f"âš ï¸� Unknown mode '{mode}', using 'navigation'")
            mode = "navigation"
        
        try:
            result = analyze_scene(image_url, mode=mode, show_details=True)
            processing_time = time.time() - start_time
            
            session_entry = {
                "timestamp": time.time(),
                "image_url": image_url,
                "mode": mode,
                "context": context,
                "result": result,
                "processing_time": processing_time
            }
            self.session_history.append(session_entry)
            
            print(f"âš¡ Processed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"â�Œ Analysis failed: {e}")
            return None
    
    def simple_analyze(self, image_url: str, request: str = "Provide accessibility guidance"):
        """Simple analysis without structured parsing"""
        return analyze_simple(image_url, request)
    
    def get_session_summary(self):
        """Show session history"""
        if not self.session_history:
            print("ğŸ“Š No analyses performed yet")
            return
        
        print(f"\nğŸ“Š Session Summary:")
        print(f"   â€¢ Total analyses: {len(self.session_history)}")
        print(f"   â€¢ Modes used: {set(item['mode'] for item in self.session_history)}")
        print(f"   â€¢ Average processing time: {sum(item['processing_time'] for item in self.session_history) / len(self.session_history):.2f}s")
        
        # print(f"\nğŸ“� Recent analyses:")
        # for i, entry in enumerate(self.session_history[-3:], 1):
        #     print(f"   {i}. {entry['mode']} - {entry['image_url'][:50]}...")
    
    def compare_modes(self, image_url: str, modes: List[str] = None):
        """Compare different modes on same image"""
        if modes is None:
            modes = ["navigation", "safety"]
        
        print(f"\nğŸ”� Mode Comparison: {image_url}")
        print("=" * 60)
        
        results = {}
        for mode in modes:
            print(f"\nğŸ“Š {mode.upper()} Analysis:")
            result = self.analyze(image_url, mode=mode)
            results[mode] = result
            time.sleep(1)
        
        print(f"\nğŸ“‹ Comparison Summary:")
        for mode, result in results.items():
            if result:
                print(f"\nğŸ�¯ {mode.upper()}:")
                print(f"   Scene: {result.scene_type}")
                print(f"   Safety: {len(result.safety_alerts)} alerts")
                print(f"   Confidence: {result.confidence:.1%}")
        
        return results

        


# =============================================================================
# PART 8: COMPLETE ASSISTANT CLASS (with Audio Integration)
# =============================================================================

class MindBridgeAssistant:
    """Main accessibility assistant interface with full multimodal capabilities"""
    
    def __init__(self):
        self.session_history = []
        self.available_modes = ["navigation", "reading", "shopping", "safety"]
        self.user_preferences = {
            "audio_detail_level": "detailed",
            "spatial_precision": "high",
            "safety_sensitivity": "high",
            "language": "en"
        }
        print("âœ… MindBridge Assistant initialized")
        print(f"ğŸ�¯ Available modes: {', '.join(self.available_modes)}")
        print("ğŸ�µ Audio capabilities: Enabled")
        print("ğŸ–¼ï¸� Visual analysis: Enabled")
        print("ğŸ�­ Multimodal analysis: Enabled")
    
    def analyze(self, image_url: str, mode: str = "navigation", context: str = None):
        """Main visual analysis function"""
        print(f"\nğŸŒ‰ MindBridge Visual Analysis")
        print(f"ğŸš€ Powered by Gemma 3n")
        print(f"ğŸ�¯ Mode: {mode.upper()}")
        if context:
            print(f"ğŸ“� Context: {context}")
        print("=" * 50)
        
        start_time = time.time()
        
        if mode not in self.available_modes:
            print(f"âš ï¸� Unknown mode '{mode}', using 'navigation'")
            mode = "navigation"
        
        try:
            result = analyze_scene(image_url, mode=mode, show_details=True)
            processing_time = time.time() - start_time
            
            session_entry = {
                "timestamp": time.time(),
                "image_url": image_url,
                "mode": mode,
                "context": context,
                "result": result,
                "processing_time": processing_time,
                "type": "visual_only"
            }
            self.session_history.append(session_entry)
            
            print(f"âš¡ Processed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"â�Œ Visual analysis failed: {e}")
            return None
    
    def analyze_multimodal(self, image_url: str, audio_file: str, mode: str = "navigation", context: str = None):
        """Enhanced analyze with image + audio input"""
        print(f"\nğŸŒ‰ MindBridge Multimodal Analysis")
        print(f"ğŸš€ Powered by Gemma 3n")
        print(f"ğŸ�¯ Mode: {mode.upper()}")
        print(f"ğŸ“¸ Image: {image_url}")
        print(f"ğŸ�µ Audio: {audio_file}")
        if context:
            print(f"ğŸ“� Context: {context}")
        print("=" * 50)
        
        start_time = time.time()
        
        # Validate mode
        if mode not in self.available_modes:
            print(f"âš ï¸� Unknown mode '{mode}', using 'navigation'")
            mode = "navigation"
        
        try:
            result = analyze_with_audio(image_url, audio_file, mode)
            processing_time = time.time() - start_time
            
            # Add to session history
            session_entry = {
                "timestamp": time.time(),
                "image_url": image_url,
                "audio_file": audio_file,
                "mode": mode,
                "context": context,
                "result": result,
                "processing_time": processing_time,
                "type": "multimodal"
            }
            self.session_history.append(session_entry)
            
            print(f"âš¡ Processed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"â�Œ Multimodal analysis failed: {e}")
            return None
    
    def analyze_audio_only(self, audio_file: str, mode: str = "navigation", context: str = None):
        """Analyze audio-only input"""
        print(f"\nğŸŒ‰ MindBridge Audio Analysis")
        print(f"ğŸš€ Powered by Gemma 3n")
        print(f"ğŸ�¯ Mode: {mode.upper()}")
        print(f"ğŸ�µ Audio: {audio_file}")
        if context:
            print(f"ğŸ“� Context: {context}")
        print("=" * 50)
        
        start_time = time.time()
        
        # Validate mode
        if mode not in self.available_modes:
            print(f"âš ï¸� Unknown mode '{mode}', using 'navigation'")
            mode = "navigation"
        
        try:
            result = analyze_audio_only(audio_file, mode)
            processing_time = time.time() - start_time
            
            # Add to session history
            session_entry = {
                "timestamp": time.time(),
                "audio_file": audio_file,
                "mode": mode,
                "context": context,
                "result": result,
                "processing_time": processing_time,
                "type": "audio_only"
            }
            self.session_history.append(session_entry)
            
            print(f"âš¡ Processed in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            print(f"â�Œ Audio analysis failed: {e}")
            return None
    
    def simple_analyze(self, image_url: str, request: str = "Provide accessibility guidance"):
        """Simple visual analysis without structured parsing"""
        return analyze_simple(image_url, request)
    
    def quick_audio_analyze(self, audio_file: str, request: str = "Describe what you hear for accessibility"):
        """Quick audio analysis without structured parsing"""
        return quick_audio_test(audio_file, request)
    
    def get_session_summary(self):
        """Show comprehensive session history including all analysis types"""
        if not self.session_history:
            print("ğŸ“Š No analyses performed yet")
            return
        
        # Count different types of analyses
        visual_count = len([item for item in self.session_history if item.get('type') == 'visual_only'])
        audio_count = len([item for item in self.session_history if item.get('type') == 'audio_only'])
        multimodal_count = len([item for item in self.session_history if item.get('type') == 'multimodal'])
        
        print(f"\nğŸ“Š Session Summary:")
        print(f"   â€¢ Total analyses: {len(self.session_history)}")
        print(f"   â€¢ Visual analyses: {visual_count}")
        print(f"   â€¢ Audio analyses: {audio_count}")
        print(f"   â€¢ Multimodal analyses: {multimodal_count}")
        print(f"   â€¢ Modes used: {set(item['mode'] for item in self.session_history)}")
        print(f"   â€¢ Average processing time: {sum(item['processing_time'] for item in self.session_history) / len(self.session_history):.2f}s")
        
        print(f"\nğŸ“� Recent analyses:")
        for i, entry in enumerate(self.session_history[-5:], 1):  # Show last 5 instead of 3
            analysis_type = entry.get('type', 'visual_only')
            if analysis_type == 'visual_only':
                print(f"   {i}. ğŸ“¸ {entry['mode']} - {entry.get('image_url', 'N/A')[:50]}...")
            elif analysis_type == 'audio_only':
                print(f"   {i}. ğŸ�µ {entry['mode']} - {entry.get('audio_file', 'N/A')}")
            elif analysis_type == 'multimodal':
                print(f"   {i}. ğŸ�­ {entry['mode']} - Image + Audio")
    
    def compare_modes(self, image_url: str, modes: List[str] = None):
        """Compare different modes on same image"""
        if modes is None:
            modes = ["navigation", "safety"]
        
        print(f"\nğŸ”� Mode Comparison: {image_url}")
        print("=" * 60)
        
        results = {}
        for mode in modes:
            print(f"\nğŸ“Š {mode.upper()} Analysis:")
            result = self.analyze(image_url, mode=mode)
            results[mode] = result
            time.sleep(1)
        
        print(f"\nğŸ“‹ Comparison Summary:")
        for mode, result in results.items():
            if result:
                print(f"\nğŸ�¯ {mode.upper()}:")
                print(f"   Scene: {result.scene_type}")
                print(f"   Safety: {len(result.safety_alerts)} alerts")
                print(f"   Confidence: {result.confidence:.1%}")
        
        return results
    
    def compare_modalities(self, image_url: str, audio_file: str, mode: str = "navigation"):
        """Compare visual-only vs audio-only vs multimodal on same content"""
        print(f"\nğŸ”� Modality Comparison")
        print(f"ğŸ�¯ Mode: {mode.upper()}")
        print(f"ğŸ“¸ Image: {image_url}")
        print(f"ğŸ�µ Audio: {audio_file}")
        print("=" * 60)
        
        results = {}
        
        # Visual only
        print(f"\nğŸ“¸ Visual-Only Analysis:")
        visual_result = self.analyze(image_url, mode=mode, context="Visual comparison")
        results['visual'] = visual_result
        time.sleep(1)
        
        # Audio only  
        print(f"\nğŸ�µ Audio-Only Analysis:")
        audio_result = self.analyze_audio_only(audio_file, mode=mode, context="Audio comparison")
        results['audio'] = audio_result
        time.sleep(1)
        
        # Multimodal
        print(f"\nğŸ�­ Multimodal Analysis:")
        multimodal_result = self.analyze_multimodal(image_url, audio_file, mode=mode, context="Multimodal comparison")
        results['multimodal'] = multimodal_result
        
        # Comparison summary
        print(f"\nğŸ“‹ Modality Comparison Summary:")
        for modality, result in results.items():
            if result:
                print(f"\nğŸ�¯ {modality.upper()}:")
                print(f"   Scene: {result.scene_type}")
                print(f"   Safety Alerts: {len(result.safety_alerts)}")
                print(f"   Confidence: {result.confidence:.1%}")
        
        return results
    
    def batch_analyze(self, inputs: List[Dict], mode: str = "navigation"):
        """Batch process multiple inputs (images, audio, or both)"""
        print(f"\nğŸ“¦ Batch Analysis")
        print(f"ğŸ�¯ Mode: {mode.upper()}")
        print(f"ğŸ“Š Processing {len(inputs)} items")
        print("=" * 50)
        
        results = []
        
        for i, input_item in enumerate(inputs, 1):
            print(f"\nğŸ”„ Processing {i}/{len(inputs)}")
            
            if 'image_url' in input_item and 'audio_file' in input_item:
                # Multimodal
                result = self.analyze_multimodal(
                    input_item['image_url'], 
                    input_item['audio_file'], 
                    mode, 
                    f"Batch item {i}"
                )
            elif 'image_url' in input_item:
                # Visual only
                result = self.analyze(
                    input_item['image_url'], 
                    mode, 
                    f"Batch item {i}"
                )
            elif 'audio_file' in input_item:
                # Audio only
                result = self.analyze_audio_only(
                    input_item['audio_file'], 
                    mode, 
                    f"Batch item {i}"
                )
            else:
                print(f"âš ï¸� Invalid input format for item {i}")
                result = None
            
            results.append(result)
            time.sleep(0.5)  # Brief pause between items
        
        # Batch summary
        successful = len([r for r in results if r is not None])
        print(f"\nğŸ“Š Batch Summary:")
        print(f"   â€¢ Items processed: {len(inputs)}")
        print(f"   â€¢ Successful: {successful}")
        print(f"   â€¢ Failed: {len(inputs) - successful}")
        
        return results
    
    def update_preferences(self, **kwargs):
        """Update user preferences"""
        for key, value in kwargs.items():
            if key in self.user_preferences:
                self.user_preferences[key] = value
                print(f"âœ… Updated {key}: {value}")
            else:
                print(f"âš ï¸� Unknown preference: {key}")
        
        print(f"ğŸ“‹ Current preferences: {self.user_preferences}")
    
    def get_capabilities(self):
        """Show all available capabilities"""
        print(f"\nğŸŒŸ MindBridge Capabilities:")
        print(f"ğŸ“¸ Visual Analysis: {self.available_modes}")
        print(f"ğŸ�µ Audio Analysis: {self.available_modes}")
        print(f"ğŸ�­ Multimodal Analysis: {self.available_modes}")
        print(f"ğŸ”„ Batch Processing: Supported")
        print(f"ğŸ“Š Mode Comparison: Supported")
        print(f"ğŸ”� Modality Comparison: Supported")
        print(f"ğŸ“ˆ Session Tracking: Enabled")
        print(f"âš™ï¸� User Preferences: Customizable")
        
        return {
            "visual_modes": self.available_modes,
            "audio_modes": self.available_modes,
            "multimodal_modes": self.available_modes,
            "batch_processing": True,
            "mode_comparison": True,
            "modality_comparison": True,
            "session_tracking": True,
            "preferences": self.user_preferences
        }


def display_image_inline(image_url, width=400, title=None):
    """Display image inline in notebook"""
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        
        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Create HTML with optional title
        title_html = f"<h4 style='color: #4285f4; margin: 5px 0;'>{title}</h4>" if title else ""
        
        html = f'''
        <div style="border: 2px solid #4285f4; padding: 10px; margin: 10px 0; border-radius: 8px; background: #f8f9ff;">
            {title_html}
            <img src="data:image/png;base64,{img_b64}" width="{width}px" 
                 style="border-radius: 5px; display: block; box-shadow: 0 2px 6px rgba(0,0,0,0.1);"/>
            <p style="margin: 5px 0 0 0; font-size: 11px; color: #666;">
                ğŸ“� Size: {image.width} x {image.height} pixels
            </p>
        </div>
        '''
        display(HTML(html))
        return True
    except Exception as e:
        print(f"âš ï¸� Could not display image: {e}")
        return False

def run_demo():
    """Run comprehensive demo with new example images"""
    demo_images = [
        {
            "name": "Safety Sign",
            "url": "https://images.unsplash.com/photo-1557911049-cf8be4d2ad79?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "mode": "reading",
            "context": "Construction/safety signage"
        },
        {
            "name": "Busy Street",
            "url": "https://images.unsplash.com/photo-1622189449143-828363e13de3?q=80&w=2138&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "mode": "safety",
            "context": "Urban street navigation"
        },
        {
            "name": "Fruit Market with Prices",
            "url": "https://images.unsplash.com/photo-1653903057504-7417e3cd9aa1?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "mode": "shopping",
            "context": "Grocery shopping assistance"
        },
        {
            "name": "Power Switches (Unsafe)",
            "url": "https://images.unsplash.com/photo-1566048908540-2d898ad550c2?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "mode": "safety",
            "context": "Electrical hazard assessment"
        }
    ]
    
    print("\nğŸ�¬ MindBridge Comprehensive Demo")
    print("=" * 60)
    print("ğŸ�¯ Testing all accessibility modes with real-world scenarios")
    
    for i, demo in enumerate(demo_images, 1):
        print(f"\nğŸ“¸ Demo {i}: {demo['name']}")
        print(f"ğŸ�¯ Testing {demo['mode'].upper()} mode")
        print(f"ğŸ“� Context: {demo['context']}")
        
        # Display image inline
        display_image_inline(demo['url'], width=400, title=f"Demo {i}: {demo['name']}")
        
        assistant.analyze(demo['url'], mode=demo['mode'], context=demo['context'])
        
        if i < len(demo_images):
            time.sleep(2)  # Brief pause between demos
    
    print("\nğŸ�† Demo Complete - All Accessibility Modes Tested!")
    assistant.get_session_summary()

def run_mode_comparison_demo():
    """Compare different modes on the same challenging image"""
    print("\nğŸ”� Mode Comparison Demo")
    print("=" * 60)
    
    # Use the busy street for comprehensive mode comparison
    busy_street_url = "https://images.unsplash.com/photo-1622189449143-828363e13de3?q=80&w=2138&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
    
    print("ğŸ“¸ Testing: Busy Street Scene")
    print("ğŸ�¯ Comparing: Navigation vs Safety vs Reading modes")
    
    # Display the image once
    display_image_inline(busy_street_url, width=500, title="Mode Comparison: Busy Street Scene")
    
    results = assistant.compare_modes(
        busy_street_url, 
        modes=['navigation', 'safety', 'reading']
    )
    
    return results

def run_targeted_tests():
    """Run tests targeting specific accessibility scenarios"""
    test_scenarios = [
        {
            "name": "Sign Reading Test",
            "url": "https://images.unsplash.com/photo-1557911049-cf8be4d2ad79?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "test": "Read all visible text and safety warnings"
        },
        {
            "name": "Price Recognition Test", 
            "url": "https://images.unsplash.com/photo-1653903057504-7417e3cd9aa1?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "test": "Identify products and their prices"
        },
        {
            "name": "Hazard Detection Test",
            "url": "https://images.unsplash.com/photo-1600298881974-6be191ceeda1?q=80&w=2126&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "test": "Identify electrical safety hazards"
        }
    ]
    
    print("\nğŸ§ª Targeted Accessibility Tests")
    print("=" * 60)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nâš¡ Test {i}: {scenario['name']}")
        
        # Display image for each test
        display_image_inline(scenario['url'], width=350, title=f"Test {i}: {scenario['name']}")
        
        assistant.simple_analyze(scenario['url'], scenario['test'])
        time.sleep(1)

def quick_test(image_url: str, request: str = "Describe this scene for accessibility", show_image: bool = True):
    """Quick test function with optional image display"""
    print(f"âš¡ Quick Test: {image_url[:50]}...")
    
    if show_image:
        display_image_inline(image_url, width=300, title="Quick Test Image")
    
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image_url},
            {"type": "text", "text": f"Briefly {request}"}
        ]
    }]
    
    do_gemma_3n_inference_simple(model, messages, max_new_tokens=100)

def test_all_modes():
    """Test all modes with appropriate images"""
    mode_tests = [
        {
            "mode": "reading",
            "url": "https://images.unsplash.com/photo-1557911049-cf8be4d2ad79?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "description": "Safety signs and text"
        },
        {
            "mode": "safety", 
            "url": "https://images.unsplash.com/photo-1622189449143-828363e13de3?q=80&w=2138&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "description": "Busy street traffic"
        },
        {
            "mode": "shopping",
            "url": "https://images.unsplash.com/photo-1653903057504-7417e3cd9aa1?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "description": "Fruit market with prices"
        },
        {
            "mode": "navigation",
            "url": "https://images.unsplash.com/photo-1566048908540-2d898ad550c2?q=80&w=3087&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
            "description": "Electrical panel area"
        }
    ]
    
    print("\nğŸ�¯ Testing All Accessibility Modes")
    print("=" * 60)
    
    for test in mode_tests:
        print(f"\nğŸ”¬ Testing {test['mode'].upper()} mode")
        print(f"ğŸ“� Scenario: {test['description']}")
        
        # Display image for each mode test
        display_image_inline(test['url'], width=350, title=f"{test['mode'].upper()} Mode Test")
        
        assistant.analyze(test['url'], mode=test['mode'], context=f"{test['mode']} mode test")
        time.sleep(1)


# Initialize the assistant
assistant = MindBridgeAssistant()

print("\nğŸ“± Available Commands:")
print("â€¢ assistant.analyze(image_url, mode='navigation', context='description')")
print("â€¢ assistant.simple_analyze(image_url, request='custom request')")
print("â€¢ assistant.get_session_summary()")
print("â€¢ assistant.compare_modes(image_url, ['mode1', 'mode2'])")

print("\nğŸ�¬ Demo Functions:")
print("â€¢ run_demo() - Full demo with all new examples")
print("â€¢ run_mode_comparison_demo() - Compare modes on busy street")
print("â€¢ run_targeted_tests() - Specific accessibility tests")
print("â€¢ test_all_modes() - Test each mode with optimal image")
print("â€¢ quick_test(image_url, 'request') - Quick analysis")

print("\nğŸ”§ Direct Functions:")
print("â€¢ analyze_scene(image_url, mode='navigation')")
print("â€¢ analyze_simple(image_url, 'request')")

print("\nğŸ�¯ Example Usage:")
print("assistant.analyze('safety_sign_url', mode='reading')")
print("assistant.analyze('busy_street_url', mode='safety')")
print("assistant.analyze('fruit_market_url', mode='shopping')")

print("\nğŸŒŸ New Example Images Loaded:")
print("ğŸ“‹ Safety Sign - Perfect for READING mode")
print("ğŸš¦ Busy Street - Ideal for SAFETY mode") 
print("ğŸ�� Fruit Market - Great for SHOPPING mode")
print("âš¡ Power Switches - Excellent for SAFETY/NAVIGATION modes")

print("\nâœ… MindBridge Ready with Real-World Examples!")
print("ğŸš€ Try: run_demo() to see all modes in action!")


run_demo()




