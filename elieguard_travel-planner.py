import os
import sys
import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

class LLMWrapper:
    def __init__(self, gemini_api_key_env='GEMINI_API_KEY'):
        """LLM wrapper that ONLY uses Gemini API.
        Fallbacks: local flan‑t5 → dummy.
        """
        self.gemini_key = os.environ.get(gemini_api_key_env)
        self.backend = None

        # Try Gemini only
        try:
            from google import genai
            self.genai = genai
            if self.gemini_key:
                try:
                    genai.configure(api_key=self.gemini_key)
                except Exception:
                    pass
            try:
                self.genai_client = genai.Client()
            except Exception:
                self.genai_client = None
            self.backend = 'gemini'
            print('LLM backend: Gemini (Google Gen AI)')
        except Exception:
            try:
                import google.generativeai as genai_old
                self.genai = genai_old
                self.genai_client = None
                self.backend = 'gemini'
                print('LLM backend: Gemini (google.generativeai)')
            except Exception as e:
                print('Gemini failed, switching to local fallback:', e)
                self._init_local()

        if self.backend is None:
            self._init_local()

    def _init_local(self):
        try:
            from transformers import pipeline
            self.pipeline = pipeline('text2text-generation', model='google/flan-t5-small')
            self.backend = 'local'
            print('LLM backend: local (flan‑t5-small)')
        except Exception:
            self.backend = 'dummy'
            print('LLM backend: dummy fallback')

    def generate(self, prompt: str, temperature=0.2, max_tokens=256) -> str:
        if self.backend == 'gemini':
            # Try modern genai client
            try:
                if self.genai_client:
                    resp = self.genai_client.models.generate_content(
                        model='gemini-1.5-flash', contents=prompt
                    )
                    if hasattr(resp, 'text'):
                        return resp.text
                    return str(resp)
            except Exception:
                pass
            # Try older API
            try:
                if hasattr(self.genai, 'generate'):
                    resp = self.genai.generate(model='gemini-1.5-flash', prompt=prompt)
                    if hasattr(resp, 'text'):
                        return resp.text
                    return str(resp)
            except Exception:
                pass
            return self._fallback_generate(prompt)

        if self.backend == 'local':
            try:
                return self.pipeline(prompt, max_length=max_tokens)[0]['generated_text']
            except Exception:
                self.backend = 'dummy'
                return self._dummy_response(prompt)

        return self._dummy_response(prompt)

    def _fallback_generate(self, prompt):
        if hasattr(self, 'pipeline'):
            try:
                return self.pipeline(prompt, max_length=256)[0]['generated_text']
            except Exception:
                pass
        return self._dummy_response(prompt)

    def _dummy_response(self, prompt):
        if 'plan' in prompt.lower():
            return 'Day 1: Arrival → sightseeing. Day 2: Main attractions. Day 3: Markets → departure.'
        return 'Gemini key missing. Add GEMINI_API_KEY in Kaggle environment.'

# End of notebook





