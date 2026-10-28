"""Writes the two report figures (mAP@10 vs D, query time vs D) as static SVG, from results/test.csv."""
import csv, math, sys
from collections import defaultdict

repo = sys.argv[1]
rows = list(csv.DictReader(open(f"{repo}/results/test.csv")))
data = defaultdict(dict)                # method -> D -> row (last run wins)
for r in rows:
    data[r["method"]][int(r["D"])] = r

DS = [1000, 5000, 10000, 25000]
DX = {1000: 1000, 5000: 5000, 10000: 10000, 25000: 24985}   # real D for the log axis
SERIES = [  # fixed categorical order, light palette slots 1..5, plus a marker shape each
    ("exact", "Exact", "#2a78d6", "circle"),
    ("lsh", "LSH", "#eb6834", "square"),
    ("hypercube", "Hypercube", "#1baf7a", "triangle"),
    ("ivfflat", "IVFFlat", "#eda100", "diamond"),
    ("ivfpq", "IVFPQ", "#e87ba4", "cross"),
]
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
W, H = 760, 400
L, R, T, B = 64, 168, 70, 52      # plot margins; right margin holds direct labels


def marker(shape, x, y, c):
    s = 4.5
    ring = f'stroke="{SURF}" stroke-width="2" paint-order="stroke"'
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{s}" fill="{c}" {ring}/>'
    if shape == "square":
        return f'<rect x="{x-s:.1f}" y="{y-s:.1f}" width="{2*s}" height="{2*s}" rx="1" fill="{c}" {ring}/>'
    if shape == "triangle":
        return f'<path d="M{x:.1f},{y-s-1:.1f} L{x+s+1:.1f},{y+s:.1f} L{x-s-1:.1f},{y+s:.1f}Z" fill="{c}" {ring}/>'
    if shape == "diamond":
        return f'<path d="M{x:.1f},{y-s-1:.1f} L{x+s+1:.1f},{y:.1f} L{x:.1f},{y+s+1:.1f} L{x-s-1:.1f},{y:.1f}Z" fill="{c}" {ring}/>'
    return (f'<path d="M{x-s:.1f},{y-s:.1f} L{x+s:.1f},{y+s:.1f} M{x+s:.1f},{y-s:.1f} L{x-s:.1f},{y+s:.1f}" '
            f'stroke="{c}" stroke-width="2.5" stroke-linecap="round"/>')


def chart(path, title, subtitle, field, ylog, yticks, fmt, endfmt):
    px0, px1, py0, py1 = L, W - R, H - B, T
    lx0, lx1 = math.log10(800), math.log10(32000)
    xs = lambda d: px0 + (math.log10(d) - lx0) / (lx1 - lx0) * (px1 - px0)
    lo, hi = yticks[0], yticks[-1]
    f = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    ys = lambda v: py0 - (f(v) - f(lo)) / (f(hi) - f(lo)) * (py0 - py1)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Helvetica, Arial, sans-serif" role="img" aria-label="{title}">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="{L}" y="24" font-size="15" font-weight="600" fill="{INK}">{title}</text>',
         f'<text x="{L}" y="42" font-size="12" fill="{INK2}">{subtitle}</text>']
    # legend, one row under the subtitle
    lx = L
    for _, name, c, shape in SERIES:
        o.append(marker(shape, lx + 5, 58, c))
        o.append(f'<text x="{lx + 14}" y="62" font-size="12" fill="{INK2}">{name}</text>')
        lx += 14 + 6.6 * len(name) + 18
    for v in yticks:   # recessive grid + y labels
        y = ys(v)
        o.append(f'<line x1="{px0}" x2="{px1}" y1="{y:.1f}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1"/>')
        o.append(f'<text x="{px0 - 8}" y="{y + 4:.1f}" font-size="11" fill="{MUTED}" text-anchor="end">{fmt(v)}</text>')
    o.append(f'<line x1="{px0}" x2="{px1}" y1="{py0}" y2="{py0}" stroke="{AXIS}" stroke-width="1"/>')
    for d in DS:
        x = xs(DX[d])
        o.append(f'<text x="{x:.1f}" y="{py0 + 18}" font-size="11" fill="{MUTED}" text-anchor="middle">{DX[d]:,}</text>')
    o.append(f'<text x="{(px0 + px1) / 2}" y="{H - 10}" font-size="12" fill="{INK2}" text-anchor="middle">'
             f'MIRFlickr distractors D (log scale)</text>')

    ends = []
    for key, name, c, shape in SERIES:
        pts = [(xs(DX[d]), ys(float(data[key][d][field]))) for d in DS]
        o.append('<polyline fill="none" stroke="{}" stroke-width="2" stroke-linejoin="round" points="{}"/>'
                 .format(c, " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)))
        for x, y in pts:
            o.append(marker(shape, x, y, c))
        ends.append([pts[-1][1], name, float(data[key][25000][field])])
    # direct labels at the right end, nudged apart so they never collide (14px min gap)
    ends.sort()
    groups = []
    for e in ends:   # merge series whose end values are identical into one label
        if groups and abs(groups[-1][2] - e[2]) < 5e-4:
            groups[-1][1] += ", " + e[1]
        else:
            groups.append(list(e))
    for i in range(1, len(groups)):
        groups[i][0] = max(groups[i][0], groups[i - 1][0] + 14)
    for y, names, v in groups:
        o.append(f'<text x="{px1 + 10}" y="{y + 4:.1f}" font-size="11" fill="{INK2}">{names} {endfmt(v)}</text>')
    o.append('</svg>')
    open(path, "w").write("\n".join(o) + "\n")


chart(f"{repo}/report/figures/map_vs_d.svg", "mAP@10 as the database grows",
      "Test set, 23 queries, chosen parameters fixed for every D", "mAP@10", False,
      [0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8], lambda v: f"{v:.2f}", lambda v: f"{v:.3f}")
chart(f"{repo}/report/figures/time_vs_d.svg", "Average query time as the database grows",
      "Test set, ms per query (log scale), index construction not included", "avg_ms", True,
      [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5], lambda v: f"{v:g}", lambda v: f"{v:.2f} ms")
print("ok")
