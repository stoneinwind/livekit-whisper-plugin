# LiveKit Whisper STT Plugin

A high-performance speech-to-text plugin for LiveKit agents using OpenAI Whisper with faster-whisper implementation for accurate and efficient speech recognition.

## Features

- **Faster-Whisper Implementation**: Optimized inference using faster-whisper for improved performance
- **High Accuracy**: State-of-the-art speech recognition using OpenAI Whisper models
- **Local Processing**: On-device inference with no external API dependencies
- **Multi-Language Support**: Support for 90+ languages with configurable language detection
- **Warmup Support**: Optional model warmup for consistent performance
- **LiveKit Integration**: Seamless integration with LiveKit agents framework

## Requirements

- LiveKit Agents v1.2 or higher
- NVIDIA GPU (recommended for optimal performance)
- Python 3.8+
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) library

## Performance

| Model | Hardware | Latency | Use Case |
|-------|----------|-------------|----------|
| **Large-v3-Turbo** | RTX 4090 | <180ms | Real-time applications |

## Installation

1. Clone or download this plugin into your LiveKit-based agents project root directory
2. Install required dependencies:
   ```bash
   pip install faster-whisper soundfile numpy
   ```
3. Ensure you have adequate storage for model downloads (models are cached locally)

## Usage

Initialize your agent session with the WhisperSTT plugin:

```python
from whisper_plugin import WhisperSTT

session = AgentSession(
    # ... other configuration
    stt=WhisperSTT(
            model="<local path>/faster-whisper-large-v3-turbo-ct2", # where the model's downloaded
            language="zh", # zh/ja/.etc also supports auto
            device="cuda",
            compute_type="int8_float16", # best perf on GPU
            model_cache_directory=False,
            zh_lang=True,
            init_prompt="以下是普通话的内容，请使用简体中文，并正确添加标点。" # optional in ZH case
        )
)
```

### Language Support

The plugin supports 90+ languages. Common language codes:

```python
# English
stt = WhisperSTT(language="en")

# Spanish
stt = WhisperSTT(language="es")

# French
stt = WhisperSTT(language="fr")

# German
stt = WhisperSTT(language="de")

# Japanese
stt = WhisperSTT(language="ja")

# Auto-detect language
stt = WhisperSTT(language=None)  # Will auto-detect
```

### Device Configuration

```python
# GPU acceleration (recommended)
stt = WhisperSTT(device="cuda", compute_type="float16")

# CPU processing
stt = WhisperSTT(device="cpu", compute_type="float32")

# Auto-select best device
stt = WhisperSTT(device="auto")
```

### Model Warmup

```python
# Enable warmup for consistent performance
stt = WhisperSTT(
    warmup_audio="./sample_audio.wav",  # 5-10 second audio clip
    device="cuda"
)
```
