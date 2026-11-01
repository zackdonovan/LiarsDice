"""Run a reproducible demonstration or benchmark, with no runtime dependencies."""
import argparse
import json
from pathlib import Path
from .engine import Rules
from .simulation import benchmark, run_game


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='one fully auditable two-player game')
    demo.add_argument('--seed', type=int, default=42)
    demo.add_argument('--dice', type=int, default=2)
    bench = sub.add_parser('benchmark', help='balanced-seat baseline comparison')
    bench.add_argument('--games', type=int, default=2000)
    bench.add_argument('--seed', type=int, default=20261101)
    bench.add_argument('--dice', type=int, default=2)
    bench.add_argument('--output', type=Path, default=Path('results/benchmark.json'))
    args = parser.parse_args()
    try:
        if args.command == 'demo':
            data = run_game(Rules(2, args.dice, True), ('probability', 'random'), args.seed, trace=True)
            print(json.dumps(data, indent=2))
        else:
            if args.output.exists():
                parser.error(f'{args.output} exists; choose a new --output path')
            data = benchmark(args.games, args.seed, args.dice)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x') as out:
                out.write(json.dumps(data, indent=2) + '\n')
            for match in data['matchups']:
                lo, hi = match['interval']
                print(f"{match['label']}: {match['wins']}/{match['games']} "
                      f"({100*match['win_rate']:.1f}%, 95% CI {100*lo:.1f}–{100*hi:.1f}%)")
            print(f'Saved complete per-game records to {args.output}')
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
