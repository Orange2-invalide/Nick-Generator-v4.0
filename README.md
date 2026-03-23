# NickGen-v4
PyTorch implementation of GPT-style transformer for procedural gamer tag generation. 
Pure Python, no external dependencies beyond PyTorch.

## Features
- **DNA Crossover**: Breed two existing tags using 4 genetic algorithms (interleave, splice, prefix-inheritance)
- **Popularity Forecast**: Platform-specific success prediction (YouTube/Twitch/Steam/TikTok) based on phonetics and leetspeak density
- **Compatibility Matrix**: Analyze synergy between two tags (shared entropy, leet balance, visual harmony)
- **Lore Generator**: Procedural backstory generation based on archetype classification (Dark/Cyber/Warrior/Beast/Elite)
- **Pronounceability Filter**: Vowel-consonant ratio validation to ensure human-readable output

## Architecture
- Transformer decoder-only, 2 layers, 4 heads, 32 dim embedding
- RMSNorm, GeLU activations, causal self-attention
- Trains from scratch on 90+ curated gaming tags
- ~28k parameters, trains in <1s on modern GPU

## Requirements
- Python 3.8+
- PyTorch 2.0+ with CUDA support
- GPU recommended (GTX 1660 or higher)

## Installation
```bash
git clone https://github.com/username/nickgen-v4.git
cd nickgen-v4
pip install torch --index-url https://download.pytorch.org/whl/cu121
python nickgen.py
