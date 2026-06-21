"""Quick training monitor — prints win-rate trend from progress.csv."""
import sys, csv
path = sys.argv[1] if len(sys.argv) > 1 else 'runs/RIR_50M/progress.csv'
rows = [r for r in csv.DictReader(open(path)) if r.get('combat/win_rate')]
print(f"rows with combat data: {len(rows)}")
print(f"{'step':>11}  {'win%':>6}  {'loss%':>6}  {'timeout%':>8}  {'ret':>8}  {'best':>7}")
# print every Nth row to show the trend
step = max(1, len(rows)//15)
for r in rows[::step] + [rows[-1]]:
    s = int(float(r['time/total_timesteps']))
    w = float(r['combat/win_rate'])*100
    l = float(r.get('combat/loss_rate') or 0)*100
    t = float(r.get('combat/timeout_rate') or 0)*100
    ret = float(r['combat/ep_return_mean'])
    best = r.get('combat/best_return') or ''
    print(f"{s:>11,}  {w:>5.0f}%  {l:>5.0f}%  {t:>7.0f}%  {ret:>8.1f}  {best[:7]:>7}")
