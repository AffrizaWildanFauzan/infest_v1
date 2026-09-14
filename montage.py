import glob, os, sys, json
from PIL import Image, ImageDraw, ImageFont

FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
H = 64          # target line height
MAXW = 1500     # max width per line
GUT = 90        # left gutter for index
PER = 16        # lines per sheet

files = sorted(glob.glob('data/test/*.png'))
os.makedirs('sheets', exist_ok=True)
index = {}

for s in range(0, len(files), PER):
    chunk = files[s:s+PER]
    rows = []
    for f in chunk:
        im = Image.open(f).convert('RGB')
        w, h = im.size
        nw = max(1, int(w * H / h))
        im = im.resize((min(nw, MAXW), H), Image.LANCZOS)
        rows.append(im)
    W = GUT + max(r.width for r in rows) + 10
    Ht = sum(r.height + 14 for r in rows) + 14
    sheet = Image.new('RGB', (W, Ht), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    y = 10
    sid = s // PER
    names = []
    for i, (r, f) in enumerate(zip(rows, chunk)):
        d.rectangle([0, y - 4, W, y + H + 4], outline=(200, 200, 200))
        d.text((8, y + 16), f"{i+1:02d}", fill=(200, 0, 0), font=FONT)
        sheet.paste(r, (GUT, y))
        names.append(os.path.basename(f))
        y += H + 14
    sheet.save(f'sheets/s{sid:03d}.png')
    index[f's{sid:03d}'] = names

json.dump(index, open('sheet_index.json', 'w'), indent=0)
print("sheets:", len(index), "images:", len(files))
