from cog import BasePredictor, Input

import os
import sys
import json
import gc

import torch
from transformers import pipeline


TRANSLATION_MODELS = {}
MODEL_ID = os.environ.get("MODEL_ID", "facebook/nllb-200-distilled-600M")


def unload_all_translation_models():
    print(f"Unloading all translation models.")
    TRANSLATION_MODELS.clear()
    torch.cuda.empty_cache()
    gc.collect()


def load_model(model_id):
    unload_all_translation_models()
    translator = pipeline("translation", model=model_id)
    TRANSLATION_MODELS[model_id] = translator


class Predictor(BasePredictor):
    def setup(self):
        load_model(MODEL_ID)

    def predict(self,
                text: str = Input(description="text to translate"),
                src_lang: str = Input(description="Source lang ID"),
                tgt_lang: str = Input(description="Source lang ID"),
                model_id: str = Input(description="huggingface model ID, default: " + MODEL_ID, default=MODEL_ID)
                ) -> str:
        print("Number of lines in data: " + str(text.count('\n')))
        translator = TRANSLATION_MODELS.get(model_id)
        if not translator:
            load_model(model_id)
            translator = TRANSLATION_MODELS.get(model_id)
        response_text = translator(text, src_lang=src_lang, tgt_lang=tgt_lang)[0].get('translation_text')
        return response_text
