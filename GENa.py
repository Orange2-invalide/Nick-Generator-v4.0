"""
Nick Generator v4.0
@Orange
"""

import torch, random, time
import torch.nn as nn
import torch.nn.functional as F
random.seed(1337)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
B,T,E,H,L = 16,20,32,2,2

DOCS = [
    "shadow","darkside","killer","pro100","n1ghtmare","bl4ckout","z3us",
    "viper","phantom","ghost","sniper","dragon","r3kt","h4x0r","cr4zy",
    "l33t","sk1ll","predator","terminator","overlord","darknight","master",
    "legend","god","demon","angel","storm","thunder","wolf","bear","eagle",
    "snake","tiger","cyber","ultra","mega","neon","ninja","samurai","warrior",
    "knight","hunter","reaper","doom","chaos","toxic","venom","royal","prime",
    "elite","alpha","omega","inferno","crystal","diamond","gold","silver","iron",
    "bl4ck","wh1te","r3d","gr33n","blu3","zer0","n30n","l4ser","pl4sma",
    "qu4ntum","turb0","cryp7","ph4ntom","sh4dow","d4rkness","l1ght","f1re",
    "1ce","bl00d","d34th","p41n","r4ge","fury","p0wer","str3ngth","sp33d",
]
random.shuffle(DOCS)

CH = sorted(set(''.join(DOCS)))
BOS = len(CH)
VOC = len(CH)+1
S2I = {c:i for i,c in enumerate(CH)}

def tok(d): return [BOS]+[S2I[c] for c in d]+[BOS]

def batch():
    ds = [DOCS[random.randint(0,len(DOCS)-1)] for _ in range(B)]
    ts = [tok(d) for d in ds]
    ml = min(max(len(t) for t in ts), T+1)
    x = torch.zeros(B,ml-1,dtype=torch.long,device=DEV)
    y = torch.zeros(B,ml-1,dtype=torch.long,device=DEV)
    m = torch.zeros(B,ml-1,dtype=torch.bool,device=DEV)
    for i,t in enumerate(ts):
        l = min(len(t)-1,ml-1)
        x[i,:l]=torch.tensor(t[:l],device=DEV)
        y[i,:l]=torch.tensor(t[1:l+1],device=DEV)
        m[i,:l]=True
    return x,y,m

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(VOC,E)
        self.wpe = nn.Embedding(T,E)
        self.drop = nn.Dropout(0.3)
        self.blocks = nn.ModuleList([nn.ModuleDict({
            'ln1': nn.LayerNorm(E),
            'attn': nn.MultiheadAttention(E,H,batch_first=True,dropout=0.1),
            'ln2': nn.LayerNorm(E),
            'mlp': nn.Sequential(nn.Linear(E,4*E),nn.GELU(),nn.Dropout(0.1),nn.Linear(4*E,E))
        }) for _ in range(L)])
        self.ln = nn.LayerNorm(E)
        self.head = nn.Linear(E,VOC,bias=False)

    def forward(self,x):
        _,T_ = x.shape
        h = self.drop(self.wte(x)+self.wpe(torch.arange(T_,device=DEV)))
        mask = nn.Transformer.generate_square_subsequent_mask(T_,device=DEV)
        for b in self.blocks:
            h2 = b['ln1'](h)
            a,_ = b['attn'](h2,h2,h2,attn_mask=mask,is_causal=True)
            h = h+a
            h = h+b['mlp'](b['ln2'](h))
        return self.head(self.ln(h))

model = GPT().to(DEV)
if torch.cuda.device_count()>1: model=nn.DataParallel(model)
print(f"device: {DEV} | params: {sum(p.numel() for p in model.parameters()):,}")

opt = torch.optim.AdamW(model.parameters(),lr=3e-3,weight_decay=0.01)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=500)

t0 = time.time()
for step in range(500):
    x,y,m = batch()
    loss = F.cross_entropy(model(x)[m].view(-1,VOC),y[m].view(-1))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    opt.step(); sch.step()
    if (step+1)%100==0: print(f"step {step+1}/500 | loss {loss.item():.4f}")
print(f"done in {time.time()-t0:.2f}s\n")

def gen(temp=1.3,maxl=20,prefix=None):
    m = model.module if hasattr(model,'module') else model
    m.eval()
    with torch.no_grad():
        toks = [BOS] if not prefix else [BOS]+[S2I[c] for c in prefix if c in S2I]
        for _ in range(maxl):
            x = torch.tensor([toks[-T:]],device=DEV)
            p = F.softmax(m(x)[0,-1]/temp,dim=-1)
            t = torch.multinomial(p,1).item()
            if t==BOS: break
            toks.append(t)
    return ''.join(CH[t] for t in toks[1:])

def pronounceable(n):
    v = set('aeiou0134')
    c = n.replace('_','')
    return len(c)>=2 and all(any(ch in v for ch in c[i:i+3]) for i in range(0,len(c),3))

def score(n):
    s = 50
    s += 15 if 5<=len(n)<=10 else (-20 if len(n)<3 else -10 if len(n)>15 else 0)
    lt = sum(1 for c in n if c in '0134')
    s += lt*8-(10 if lt>4 else 0)
    s += 10 if '_' in n else 0
    s += sum(12 for p in ['dark','bl4','ph4','n1','x_','_x','z3','k1ll','n30','sh4'] if p in n)
    s += sum(3 for c in n if c in 'xzqkj')
    s -= 15 if not pronounceable(n) else 0
    return max(0,min(100,s))

def category(n):
    cats = {
        'Dark':   (['dark','bl4','sh4','d34','bl00','doom','shadow'],2),
        'Cyber':  (['cyber','n30','l4s','pl4','qu4','turb'],2),
        'Warrior':(['kill','war','slay','rage','fury','str3'],2),
        'Leet':   (['4','3','1','0','7'],3),
        'Beast':  (['wolf','bear','eagle','snak','tiger','drag'],1),
        'Elite':  (['master','legend','god','royal','prime','alpha'],1),
    }
    sc = {cat:sum(1 for p in ps if p in n) for cat,(ps,th) in cats.items()
          if sum(1 for p in ps if p in n)>=th}
    return max(sc,key=sc.get) if sc else 'Gamer'

def verdict(s):
    return ('LEGENDARY' if s>=85 else 'EPIC' if s>=70 else
            'COOL' if s>=55 else 'AVERAGE' if s>=40 else 'MEH')

def bar(s): return f"[{'#'*(s//10)}{'-'*(10-s//10)}] {s}/100"

def popularity(n):
    base = score(n)*0.6
    return {p:min(95,int(base+random.randint(a,b))) for p,a,b in [
        ('YouTube',0,20),('Twitch',-5,15),('Steam',-10,10),('TikTok',5,25)]}

def backstory(n):
    nl = n.lower()
    arch = ('dark' if any(p in nl for p in ['dark','doom','death','shadow','bl4']) else
            'cyber' if any(p in nl for p in ['cyber','n30','l4s','qu4','turb']) else
            'beast' if any(p in nl for p in ['wolf','bear','eagle','tiger','snake','dragon']) else
            'legend' if any(p in nl for p in ['god','legend','master','elite','alpha']) else 'default')
    stories = {
        'dark':   [f"{n} had one bad night. Never lost again.",
                   f"Nobody knows where {n} came from. The ranked queue does.",
                   f"{n} queued at 3am. Something changed."],
        'cyber':  [f"{n} wrote his first script at 12. Now he's on the other side.",
                   f"Devs found a bug {n} left. Still don't know how.",
                   f"{n} sees the game engine. Others see pixels."],
        'beast':  [f"{n} uses a trackpad. Still top-1.",
                   f"First tournament. One hand. {n} won.",
                   f"16 hours grinding. 8 hours watching replays. Repeat."],
        'legend': [f"Three world records. Two bans. One {n}.",
                   f"Pro team offer at 15. {n} said no. Twice.",
                   f"People talk about {n}. Most of it is true."],
        'default':[f"{n} appeared. Disappeared. The record stayed.",
                   f"Nobody knows who {n} really is. Better that way.",
                   f"Bronze to Diamond in one evening. Screenshots exist."],
    }
    ends = ["Stats don't lie. They just can't keep up.",
            "Legends don't explain themselves.",
            "Next match will tell.",
            "Enemies fear him. Teammates too."]
    return f"{random.choice(stories[arch])} {random.choice(ends)}"

def compatibility(n1,n2):
    s = 20
    reasons = []
    if abs(len(n1)-len(n2))<=2: s+=20; reasons.append("similar length")
    c1,c2 = category(n1),category(n2)
    if c1==c2: s+=25; reasons.append(f"same category: {c1}")
    else: reasons.append(f"different styles: {c1} vs {c2}")
    common = set(n1)&set(n2)
    if len(common)>=3: s+=15; reasons.append(f"shared chars: {''.join(sorted(common)[:4])}")
    l1,l2 = sum(1 for c in n1 if c in '0134'),sum(1 for c in n2 if c in '0134')
    if abs(l1-l2)<=1: s+=15; reasons.append("same leet level")
    if len(n1)+len(n2)<=16: s+=10; reasons.append("looks good together")
    s+=((sum(ord(c) for c in n1+n2))%20)
    s=max(0,min(100,s))
    v=('PERFECT DUO' if s>=80 else 'GOOD TEAM' if s>=60 else 'FINE' if s>=40 else 'RIVALS')
    return s,v,reasons[:3]

def dna(n1,n2):
    c1=''.join(a if i%2==0 else b for i,(a,b) in enumerate(zip(n1,n2)))
    c2=n1[:len(n1)//2]+n2[len(n2)//2:]
    c3=gen(1.1,prefix=n1[:3])
    c4=gen(1.1,prefix=n2[:3])
    return [c for c in [c1,c2,c3,c4] if len(c)>=3]

# --- main ---
nicks,seen,tries = [],[],0
while len(nicks)<12 and tries<300:
    n=gen(); tries+=1
    if len(n)>=3 and n not in seen and n not in DOCS and pronounceable(n):
        seen.append(n); nicks.append(n)
nicks.sort(key=score,reverse=True)

print("="*65)
print("NICK GENERATOR v4.0")
print("="*65)
print(f"\n{'NICK':<18}{'CAT':<12}{'RATING':<24}{'VERDICT'}")
print("-"*65)
for n in nicks:
    print(f"{n:<18}{category(n):<12}{bar(score(n)):<24}{verdict(score(n))}")

best=nicks[0]
print(f"\nBest nick: {best.upper()}")

if len(nicks)>=2:
    p1,p2=nicks[0],nicks[1]
    print(f"\n{'='*65}")
    print(f"DNA: [{p1}] x [{p2}]")
    print(f"{'='*65}")
    methods=['Interleave','Splice','Prefix-1','Prefix-2']
    children=dna(p1,p2)
    for i,c in enumerate(children):
        print(f"  {methods[i]:<12} {c:<15} {bar(score(c))} {verdict(score(c))}")
    best_dna=max(children,key=score)

print(f"\n{'='*65}")
print(f"POPULARITY FORECAST - top 3")
print(f"{'='*65}")
for n in nicks[:3]:
    print(f"\n  {n.upper()}")
    for platform,pct in popularity(n).items():
        print(f"    {platform:<10} [{'#'*(pct//10)}{'-'*(10-pct//10)}] {pct}%")

print(f"\n{'='*65}")
print(f"BACKSTORY: {best.upper()}")
print(f"{'='*65}")
print(f"\n  {backstory(best)}")
if len(nicks)>=2:
    print(f"\n  DNA pick: {best_dna.upper()}")
    print(f"  {backstory(best_dna)}")

if len(nicks)>=3:
    print(f"\n{'='*65}")
    print(f"COMPATIBILITY")
    print(f"{'='*65}")
    for n1,n2 in [(nicks[0],nicks[1]),(nicks[0],nicks[2]),(nicks[1],nicks[2])]:
        s,v,r=compatibility(n1,n2)
        print(f"\n  {n1} + {n2}")
        print(f"  {bar(s)} {v}")
        for reason in r: print(f"  - {reason}")

print(f"\n{'='*65}")
print("done.")
print(f"{'='*65}")
