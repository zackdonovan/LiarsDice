"""Render saved benchmark values to a dependency-free SVG."""
import json
from pathlib import Path
root = Path(__file__).resolve().parents[1]
data = json.loads((root/'results/benchmark.json').read_text())
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="340" viewBox="0 0 900 340">',
       '<rect width="900" height="340" fill="#0e1828"/>',
       '<g font-family="Arial,sans-serif" fill="#edf4ff">',
       '<text x="32" y="42" font-size="25" font-weight="bold">Liar’s Dice · fixed baseline check</text>',
       '<text x="32" y="72" font-size="15" fill="#b7c9e5">2 players · 2 dice · wild ones · 2,000 games per matchup</text>']
for i,m in enumerate(data['matchups']):
    y=126+i*80; x=270; scale=5.4
    svg += [f'<text x="32" y="{y+19}" font-size="17">{m["label"]}</text>',
            f'<rect x="{x}" y="{y}" width="540" height="28" rx="5" fill="#24354d"/>',
            f'<rect x="{x}" y="{y}" width="{100*m["win_rate"]*scale:.2f}" height="28" rx="5" fill="{["#80a6db","#5ad3bd"][i]}"/>',
            f'<line x1="{x+100*m["interval"][0]*scale:.2f}" x2="{x+100*m["interval"][1]*scale:.2f}" y1="{y+14}" y2="{y+14}" stroke="#102035" stroke-width="3"/>',
            f'<text x="{x}" y="{y+50}" font-size="14">{m["wins"]}/{m["games"]} wins · {100*m["win_rate"]:.2f}% · Wilson 95% interval {100*m["interval"][0]:.2f}–{100*m["interval"][1]:.2f}%</text>']
svg += ['<text x="32" y="313" font-size="14" fill="#b7c9e5">Synthetic play against random actions. Not human-level skill, learning, or an optimal strategy.</text>', '</g></svg>']
(root/'results/benchmark.svg').write_text('\n'.join(svg)+'\n')
