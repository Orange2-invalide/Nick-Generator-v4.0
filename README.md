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

## Usage
Modify DOCS list in source to customize training corpus, then run:

# Generate single tag
nickname = gen(temperature=1.3)

# DNA crossover
children = dna("shadow", "h4x0r")

# Get metrics
print(score(nickname))        # 0-100 rating
print(category(nickname))     # Dark/Cyber/Leet/etc
print(popularity(nickname))   # Platform dict
print(backstory(nickname))    # Flavor text
print(compatibility("tag1", "tag2"))  # Synergy score

## Sample Output
NICK            CAT        RATING              VERDICT
sn3ke           Leet       [#######---] 76/100  EPIC
133ke           Leet       [########--] 92/100  LEGENDARY
megon           Gamer      [######----] 65/100  COOL

DNA CROSS [sn3ke] x [133ke]:
  Interleave: 1n3ke         [#######---] 73/100  EPIC
  Splice:     sn33k         [#########-] 88/100  LEGENDARY

BACKSTORY: 133ke
"Wrote his first script at 12. Now he's on the other side. 
Devs found a bug he left. Still don't know how."

## Training
Model trains 500 steps with AdamW + cosine scheduling.
Loss converges to ~1.8 on default dataset.

## Credits
Architecture inspired by Karpathy's minGPT and llm.c.
Implementation optimized for single-file deployment and educational clarity.

## Installation
```bash
git clone https://github.com/username/nickgen-v4.git
cd nickgen-v4
pip install torch --index-url https://download.pytorch.org/whl/cu121
python nickgen.py
