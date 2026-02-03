# audio_analyzer.py
from audio_decoder import decode_base64_audio
from feature_extractor import extract_voice_features
from language_handler import validate_language


def analyze_audio(audio_base64: str, language: str, audio_format: str = "mp3"):
    """
    Person 2 main function.
    Inputs:
      - audio_base64: base64 string of MP3 (or supported) audio
      - language: request-provided language (validated, preserved)
      - audio_format: 'mp3' (default) or 'webm' if needed; used by decoder fallback

    Returns:
      {
        "language": <language preserved>,
        "features": { ... }   # numeric feature dict
      }

    Raises ValueError on bad input.
    """
    lang = validate_language(language)  # will raise if unsupported
    y, sr = decode_base64_audio(audio_base64, audio_format)
    features = extract_voice_features(y, sr)
    return {
        "language": lang,
        "features": features
    }
