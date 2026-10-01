"""Save the prompt UI as an SVG: panels, buttons and text as vectors, the images (camera views, prompt image, samples)
embedded as JPEGs from a screenshot at the page's device scale. scripts/record_demo.py calls snapshot() mid-recording.

  .venv/bin/python scripts/ui_svg.py [--url http://localhost:8088] [--out docs/ui/prompt_ui.svg]

Text gets textLength, so it keeps its width in whatever font the viewer has.
"""
import argparse
import base64
import html
import io

from PIL import Image

# Every visible box (background, border, radius), text line (per line box, via Range rects), input value and image.
WALK = r"""() => {
  const out = [], vw = document.documentElement.scrollWidth, vh = document.documentElement.scrollHeight;
  const rgba = (c) => { const m = c.match(/[\d.]+/g); return m ? m.map(Number) : [0, 0, 0, 0]; };
  const alpha = (c) => { const m = rgba(c); return m.length > 3 ? m[3] : 1; };
  const opacity = (el) => { let o = 1; for (let e = el; e && e.nodeType === 1; e = e.parentElement)
                              o *= parseFloat(getComputedStyle(e).opacity); return o; };
  const clipOf = (el) => { let r = {left: 0, top: 0, right: vw, bottom: vh};
    for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
      const s = getComputedStyle(e);
      if (s.overflow !== 'visible' || s.overflowY !== 'visible') { const b = e.getBoundingClientRect();
        r = {left: Math.max(r.left, b.left), top: Math.max(r.top, b.top),
             right: Math.min(r.right, b.right), bottom: Math.min(r.bottom, b.bottom)}; } }
    return r; };
  const font = (s) => ({family: s.fontFamily, size: parseFloat(s.fontSize), weight: s.fontWeight,
                        spacing: s.letterSpacing === 'normal' ? 0 : parseFloat(s.letterSpacing), color: s.color});
  const visible = (el) => { const s = getComputedStyle(el);
    return s.display !== 'none' && s.visibility !== 'hidden' && el.getClientRects().length > 0; };
  const walk = (el) => {
    if (!visible(el)) return;
    const s = getComputedStyle(el), b = el.getBoundingClientRect(), op = opacity(el);
    const bw = ['Top', 'Right', 'Bottom', 'Left'].map((k) => parseFloat(s['border' + k + 'Width']));
    const bc = ['Top', 'Right', 'Bottom', 'Left'].map((k) => s['border' + k + 'Color']);
    if (b.width > 0 && b.height > 0 && (alpha(s.backgroundColor) > 0 || bw.some((w) => w > 0)))
      out.push({k: 'box', x: b.left, y: b.top, w: b.width, h: b.height, r: parseFloat(s.borderTopLeftRadius) || 0,
                fill: s.backgroundColor, bw, bc, dashed: s.borderTopStyle === 'dashed', op});
    if (el.tagName === 'IMG' || el.tagName === 'CANVAS') {
      const pad = ['Top', 'Right', 'Bottom', 'Left'].map((k) => parseFloat(s['padding' + k]));
      let x = b.left + bw[3] + pad[3], y = b.top + bw[0] + pad[0];
      let w = b.width - bw[1] - bw[3] - pad[1] - pad[3], h = b.height - bw[0] - bw[2] - pad[0] - pad[2];
      if (el.tagName === 'IMG' && el.naturalWidth && s.objectFit === 'contain') {   // the drawn image, not the box
        const k = Math.min(w / el.naturalWidth, h / el.naturalHeight), iw = el.naturalWidth * k, ih = el.naturalHeight * k;
        x += (w - iw) / 2; y += (h - ih) / 2; w = iw; h = ih; }
      out.push({k: 'img', x, y, w, h, r: parseFloat(s.borderTopLeftRadius) || 0, op});
      return;
    }
    if (el.tagName === 'INPUT') {
      const v = el.value || el.placeholder, f = font(s);
      if (!el.value) f.color = getComputedStyle(el, '::placeholder').color;
      const c = document.createElement('canvas').getContext('2d');
      c.font = s.fontWeight + ' ' + s.fontSize + ' ' + s.fontFamily;
      out.push({k: 'text', t: v, x: b.left + bw[3] + parseFloat(s.paddingLeft), y: b.top, h: b.height,
                w: c.measureText(v).width, f, op});
      return;
    }
    for (const n of el.childNodes) {
      if (n.nodeType === 1) { walk(n); continue; }
      if (n.nodeType !== 3 || !n.textContent.trim()) continue;
      const clip = clipOf(el), f = font(s), up = s.textTransform === 'uppercase', r = document.createRange();
      let line = null;
      const flush = () => { if (line && line.t.trim() && line.top >= clip.top - 1 && line.bottom <= clip.bottom + 1)
                              out.push({k: 'text', t: line.t.replace(/\s+$/, ''), x: line.x, y: line.top,
                                        h: line.bottom - line.top, w: line.right - line.x, f, op});
                            line = null; };
      for (let i = 0; i < n.textContent.length; i++) {
        r.setStart(n, i); r.setEnd(n, i + 1);
        const q = r.getClientRects()[0], ch = n.textContent[i];
        if (!q) continue;
        if (line && Math.abs(q.top - line.top) > 2) flush();
        if (!line) { if (!ch.trim()) continue; line = {t: '', x: q.left, top: q.top, bottom: q.bottom, right: q.right}; }
        line.t += up ? ch.toUpperCase() : ch;
        if (ch.trim()) line.right = q.right;
      }
      flush();
    }
  };
  walk(document.body);
  return {w: vw, h: vh, bg: getComputedStyle(document.body).backgroundColor, items: out};
}"""


def color(css):
    """CSS rgb()/rgba() -> (svg colour, alpha)."""
    v = [float(x) for x in css[css.index('(') + 1:css.index(')')].replace('/', ',').split(',')]
    return 'rgb(%d,%d,%d)' % tuple(v[:3]), (v[3] if len(v) > 3 else 1.0)


def paint(attr, css, op=1.0):
    c, a = color(css)
    return '%s="%s"%s' % (attr, c, ' %s-opacity="%.3g"' % (attr, a * op) if a * op < 1 else '')


def to_svg(page, quality=88):
    d = page.evaluate(WALK)
    scale = page.evaluate('window.devicePixelRatio')
    shot = Image.open(io.BytesIO(page.screenshot(full_page=True, type='png'))).convert('RGB')
    e = []
    for n, it in enumerate(d['items']):
        x, y, w, h = it['x'], it['y'], it.get('w', 0), it.get('h', 0)
        if 'r' in it:
            it['r'] = min(it['r'], w / 2, h / 2)                           # CSS clamps a 999px radius, SVG makes an ellipse
        op = ' opacity="%.3g"' % it['op'] if it['op'] < 1 else ''
        if it['k'] == 'box':
            bw, bc = it['bw'], it['bc']
            fill = paint('fill', it['fill']) if color(it['fill'])[1] > 0 else 'fill="none"'
            if len(set(bw)) == 1 and len(set(bc)) == 1 and bw[0] > 0:     # one border all round: a stroked rect
                s = bw[0]
                e.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f" %s %s stroke-width="%g"%s%s/>' % (
                    x + s / 2, y + s / 2, w - s, h - s, max(it['r'] - s / 2, 0), fill, paint('stroke', bc[0]), s,
                    ' stroke-dasharray="6 4"' if it['dashed'] else '', op))
            else:
                e.append('<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f" %s%s/>' % (
                    x, y, w, h, it['r'], fill, op))
                if bw[3] > 0 and color(bc[3])[1] > 0:                      # a left accent bar (the result box)
                    e.append('<clipPath id="c%d"><rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f"/></clipPath>'
                             '<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" %s clip-path="url(#c%d)"%s/>' % (
                                 n, x, y, w, h, it['r'], x, y, bw[3], h, paint('fill', bc[3]), n, op))
        elif it['k'] == 'img':
            if w < 1 or h < 1:
                continue
            crop = shot.crop(tuple(round(v * scale) for v in (x, y, x + w, y + h)))
            buf = io.BytesIO()
            crop.save(buf, 'JPEG', quality=quality)
            clip = ''
            if it['r'] > 0:
                e.append('<clipPath id="c%d"><rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="%.1f"/></clipPath>' % (
                    n, x, y, w, h, it['r']))
                clip = ' clip-path="url(#c%d)"' % n
            e.append('<image x="%.1f" y="%.1f" width="%.1f" height="%.1f" preserveAspectRatio="none"%s%s '
                     'href="data:image/jpeg;base64,%s"/>' % (x, y, w, h, clip, op,
                                                             base64.b64encode(buf.getvalue()).decode()))
        else:
            f = it['f']
            e.append('<text x="%.1f" y="%.1f" dominant-baseline="central" font-family="%s" font-size="%g" '
                     'font-weight="%s" %s%s textLength="%.1f" lengthAdjust="spacingAndGlyphs"%s xml:space="preserve">'
                     '%s</text>' % (x, y + h / 2, html.escape(f['family']), f['size'], f['weight'],
                                    paint('fill', f['color']),
                                    ' letter-spacing="%g"' % f['spacing'] if f['spacing'] else '', w, op,
                                    html.escape(it['t'])))
    height = min(d['h'], max(it['y'] + it.get('h', 0) for it in d['items']) + 22)   # down to the content
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d">\n'
            '<rect width="100%%" height="100%%" %s/>\n%s\n</svg>\n' % (
                d['w'], height, d['w'], height, paint('fill', d['bg']), '\n'.join(e)))


def snapshot(page, path):
    with open(path, 'w') as f:
        f.write(to_svg(page))
    print('[ui_svg] %s' % path, flush=True)


def main():
    from playwright.sync_api import sync_playwright
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--url', default='http://localhost:8088')
    ap.add_argument('--out', default='docs/ui/prompt_ui.svg')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=780)
    ap.add_argument('--scale', type=float, default=2)
    args = ap.parse_args()
    with sync_playwright() as p:
        browser = p.chromium.launch(channel='chrome', headless=True)
        page = browser.new_page(viewport={'width': args.width, 'height': args.height}, device_scale_factor=args.scale)
        page.goto(args.url, wait_until='domcontentloaded')
        page.wait_for_timeout(3000)
        snapshot(page, args.out)
        browser.close()


if __name__ == '__main__':
    main()
