"""Build the editable Figure 1 model, export SVG, then PDF and PNG.

Run: conda run -n pyg python scripts/build_aligned_architecture.py
Uses installed matplotlib, Pillow and CairoSVG; no experiment files are changed.
"""
from __future__ import annotations

import base64
import io
import math as mathlib
from pathlib import Path
import xml.etree.ElementTree as ET

import cairosvg
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures' / 'generated'
NAME = 'gnn_architecture_aligned_v3'
WIDTH, HEIGHT = 2250, 920
BLUE, GREEN, INK = '#1689c9', '#49a16a', '#182c3a'
model = ET.Element('mxGraphModel', page='1', pageWidth=str(WIDTH), pageHeight=str(HEIGHT))
root = ET.SubElement(model, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', parent='0')
counter = 1


def cell(x, y, w, h, style, value='', **extra):
    global counter
    counter += 1
    c = ET.SubElement(root, 'mxCell', id=str(counter), parent='1', vertex='1',
                      value=value, style=style, **extra)
    ET.SubElement(c, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as': 'geometry'})
    return c


def rect(x, y, w, h, fill='white', stroke='#a7bbc8', radius=12):
    cell(x, y, w, h, f'rounded={int(radius>0)};fillColor={fill};strokeColor={stroke};strokeWidth=1.7;')


def text(x, y, w, h, value, size=26, bold=False, color=INK, align='center'):
    cell(x, y, w, h, f'text;html=0;align={align};verticalAlign=middle;fontFamily=Arial;fontSize={size};fontColor={color};fontStyle={int(bold)};strokeColor=none;fillColor=none;', value)


def ellipse(x, y, w, h, fill, stroke=INK):
    cell(x, y, w, h, f'ellipse;fillColor={fill};strokeColor={stroke};strokeWidth=2;')


def star(x,y):
    cell(x-13,y-13,26,26,'shape=star;fillColor=#54ac78;strokeColor=#286345;strokeWidth=1.5;')


def arrow(points, color=INK, dashed=False, head=True, width=3):
    global counter
    counter += 1
    c = ET.SubElement(root, 'mxCell', id=str(counter), parent='1', edge='1',
                      style=f'endArrow={"block" if head else "none"};strokeColor={color};strokeWidth={width};dashed={int(dashed)};')
    g = ET.SubElement(c, 'mxGeometry', relative='1', **{'as': 'geometry'})
    for p, role in [(points[0], 'sourcePoint'), (points[-1], 'targetPoint')]:
        ET.SubElement(g, 'mxPoint', x=str(p[0]), y=str(p[1]), **{'as': role})
    arr = ET.SubElement(g, 'Array', **{'as': 'points'})
    for x, y in points[1:-1]:
        ET.SubElement(arr, 'mxPoint', x=str(x), y=str(y))


def math(x, y, w, h, latex, size=29):
    buf = io.BytesIO()
    math_to_image(f'${latex}$', buf, prop=FontProperties(size=size), format='svg', color=INK)
    raw = buf.getvalue()
    svg = ET.fromstring(raw)
    _, _, sw, sh = map(float, svg.attrib['viewBox'].split())
    scale = min(w/sw, h/sh, 1.30)
    iw, ih = sw*scale, sh*scale
    uri = 'data:image/svg+xml;base64,' + base64.b64encode(raw).decode()
    cell(x+(w-iw)/2, y+(h-ih)/2, iw, ih,
         'shape=image;imageAspect=0;aspect=fixed;image='+uri.replace(';base64,', ',')+';',
         latex=latex, imageData=uri)


def panel(x, w, number, title, color, fill):
    rect(x, 20, w, 790, fill, color, 15)
    ellipse(x+12, 34, 42, 42, color, color)
    text(x+12, 34, 42, 42, str(number), 27, True, 'white')
    text(x+59, 29, w-69, 70, title, 27, True)


def card(x,y,w,h,title,lines=()):
    rect(x,y,w,h)
    text(x+5,y+8,w-10,34,title,25,True)
    for i,line in enumerate(lines):
        text(x+8,y+47+i*30,w-16,31,line,23)


def scene():
    x=20
    rect(x+14,113,242,414,'#f5fbff')
    # Neighbor links are illustrative local communication edges.
    pts=[(x+65,225),(x+195,271),(x+81,403),(x+204,451)]
    for a,b in [(0,1),(0,2),(1,2),(1,3),(2,3)]:
        arrow([pts[a],pts[b]], '#738995', True, False, 2)
    rect(x+125,145,58,45,'#89949a',INK,0)
    rect(x+20,297,36,51,'#89949a',INK,0)
    ellipse(x+126,333,42,42,'#89949a')
    rect(x+171,353,54,50,'#89949a',INK,0)
    for n,((px,py),color) in enumerate(zip(pts,['#df6972','#4b9cd3','#57ae85','#dfc650']),1):
        ellipse(px-19,py-15,38,34,color)
        rect(px-24,py-5,7,19,INK,INK,0)
        rect(px+17,py-5,7,19,INK,INK,0)
        arrow([(px,py-7),(px+7,py-43)],INK,False,True,2)
        text(px-28,py+22,56,28,f'R{n}',23)
    for gx,gy in [(x+50,154),(x+214,193),(x+48,460),(x+145,488)]:
        star(gx,gy)
    card(x+14,548,242,224,'Observed state',('Robot positions','Current velocities','Assigned goals','Static obstacles'))


def build():
    panel(20,270,1,'Multi-robot\nscene',BLUE,'#e8f5ff')
    panel(306,310,2,'Robot\nobservations',GREEN,'#eef9ef')
    panel(632,330,3,'Dynamic graph\ninput',BLUE,'#edf7ff')
    panel(978,300,4,'Feature\nencoding',GREEN,'#eff9f0')
    panel(1294,590,5,'Shared GNN policy',BLUE,'#eaf5ff')
    panel(1900,330,6,'Velocity\nexecution',GREEN,'#eef9ef')
    scene()
    x=320
    text(x,109,282,36,'Node features · 8D',24,True)
    items=[('Goal displacement · 2D',r'\gamma_i=(g_i-p_i)\oslash s'),
           ('Current velocity · 2D',r'v_i/v_{\max}'),
           ('Obstacle direction · 2D',r'n_i^{\mathrm{obs}}'),
           ('Obstacle distance · 1D',r'\eta_i=\mathrm{clip}(d_i^{\mathrm{obs}}/4,0,1)'),
           ('Constant · 1D',r'1')]
    for k,(title,eq) in enumerate(items):
        y=155+k*113
        card(x,y,282,101,title)
        math(x+8,y+44,266,48,eq,25)
    math(x,735,282,48,r'x_i=[\gamma_i,\ v_i/v_{\max},\ n_i^{\mathrm{obs}},\eta_i,1]',23)
    x=646
    card(x,113,302,151,'Radius adjacency')
    math(x+10,157,282,53,r'A_{ij}=\mathbf{1}\{0<\|r_{ij}\|_2\leq R_c\}',24)
    text(x+5,218,292,32,'Sender j → receiver i',24)
    card(x,280,302,284,'Relative-motion edges')
    math(x+8,324,286,48,r'r_{ij}=p_i-p_j',28)
    math(x+8,374,286,48,r'\Delta v_{ij}=v_i-v_j',28)
    math(x+8,424,286,57,r'n_{ij}=\frac{r_{ij}}{\max(\|r_{ij}\|_2,\epsilon)}',27)
    math(x+8,492,286,48,r'c_{ij}=-\Delta v_{ij}^{\mathsf{T}}n_{ij}',28)
    card(x,580,302,193,'Edge feature: 5D')
    math(x+8,626,286,44,r'e_{ij}=A_{ij}[r_{ij}\oslash s,',26)
    math(x+8,670,286,44,r'\Delta v_{ij}/v_{\max},\ c_{ij}/v_{\max}]',26)
    text(x+5,723,292,35,'Inputs: X, A, E',26,True)
    x=992
    card(x,113,272,156,'Node encoder')
    text(x+5,157,262,38,'Linear + ReLU',25)
    math(x+8,205,256,51,r'h_i^{(0)}=\mathrm{ReLU}(W_xx_i+b_x)',24)
    card(x,310,272,218,'Mean edge feature')
    math(x+8,357,256,58,r'\bar e_i=\frac{\sum_jA_{ij}e_{ij}}{D_i}',28)
    text(x+5,424,262,38,'Linear edge encoder',24,True)
    math(x+8,466,256,48,r'z_i=W_e\bar e_i+b_e',27)
    card(x,574,272,199,'Shared edge encoder',('Same parameters','in all three layers'))
    math(x+8,692,256,63,r'D_i=\max(1,\sum_j A_{ij})',24)
    x=1308
    text(x,113,562,34,'3 layers · 128 hidden units',27,True)
    # Three miniature graph layers, each is a full aggregation/update layer.
    for k in range(3):
        cx=x+90+k*188
        for dx,dy in [(-43,-22),(40,-27),(-36,31)]:
            arrow([(cx+dx,202+dy),(cx,214)],'#567588',False,True,2)
            ellipse(cx+dx-8,202+dy-8,16,16,['#5da5d4','#69ad7e','#e0bd57'][k])
        ellipse(cx-12,202,24,24,'#388ec2')
        text(cx-72,245,144,32,f'Layer {k+1}',25,True)
        if k<2: arrow([(cx+60,211),(cx+117,211)],BLUE,False,True,3)
    rect(x,292,562,331,'#ffffff',BLUE)
    text(x+8,301,546,35,'Each layer: aggregate + update',27,True)
    math(x+13,347,536,63,r'\bar h_i^{(l)}=\frac{\sum_jA_{ij}h_j^{(l)}}{D_i}',29)
    math(x+13,426,536,58,r'm_i^{(l)}=\mathrm{ReLU}(\mathrm{Linear}_{m,l}(\bar h_i^{(l)}+z_i))',28)
    math(x+13,501,536,57,r'h_i^{(l+1)}=\mathrm{ReLU}(\mathrm{Linear}_{u,l}([h_i^{(l)}\Vert m_i^{(l)}]))',27)
    text(x+8,575,546,35,'Repeat for l = 0, 1, 2',25)
    arrow([(x+281,627),(x+281,652)],BLUE)
    card(x,660,562,113,'Shared action head: Linear → ReLU → Linear')
    math(x+10,707,542,48,r'a_i=f_{\mathrm{head}}(h_i^{(3)})',29)
    # Explicit two-path wiring into the graph policy.
    arrow([(1264,234),(1286,234),(1286,282),(1318,282)],BLUE,False,True,3)
    arrow([(1264,489),(1298,489),(1298,454),(1318,454)],GREEN,False,True,3)
    x=1914
    card(x,113,302,131,'Raw velocity')
    math(x+8,165,286,61,r'u_i^{\mathrm{raw}}=v_{\max}a_i',30)
    arrow([(x+151,248),(x+151,282)],GREEN)
    rect(x,291,302,172,'#fff5ce','#b39328')
    text(x+8,300,286,65,'Optional pairwise\ncorrection',26,True)
    text(x+8,373,286,70,'Sequential updates\nfor approaching pairs',23)
    arrow([(x+151,467),(x+151,500)],GREEN)
    card(x,508,302,125,'Norm clipping')
    math(x+8,557,286,57,r'\|u_i\|_2\leq v_{\max}',30)
    arrow([(x+151,637),(x+151,665)],GREEN)
    card(x,673,302,100,'Executed velocity',('Robot motion',))
    # Pipeline arrows and observation feedback.
    for xa,xb in [(290,306),(616,632),(962,978),(1884,1900)]:
        arrow([(xa,402),(xb,402)],INK,False,True,3)
    arrow([(2065,778),(2065,853),(155,853),(155,815)],INK,False,True,4)
    text(400,858,1470,36,'Next observation: rebuild graph and recompute features',27,True)
    text(400,895,1470,24,'s = [W, H] · epsilon = 10^-6 · Linear includes bias · parameters shared across robots',20)
    OUT.mkdir(parents=True,exist_ok=True)
    ET.indent(model)
    path=OUT/f'{NAME}.drawio'
    ET.ElementTree(model).write(path,encoding='utf-8',xml_declaration=True)
    export_svg(path,OUT/f'{NAME}.svg')
    # SVG first; PDF and preview PNG are derived from the same exported SVG.
    cairosvg.svg2pdf(url=str(OUT/f'{NAME}.svg'),write_to=str(OUT/f'{NAME}.pdf'))
    cairosvg.svg2png(url=str(OUT/f'{NAME}.svg'),write_to=str(OUT/f'{NAME}.png'),output_width=4500)
    print(f'Built {NAME}: editable Draw.io, SVG, PDF, and PNG.')


def export_svg(source: Path, destination: Path):
    """Render the native subset used by this Draw.io model without a GUI."""
    ns='http://www.w3.org/2000/svg'
    ET.register_namespace('',ns)
    # At 0.15 mm per design unit, the standalone page has 1 mm margins.
    margin=1/0.15
    svg=ET.Element('svg',xmlns=ns,width=f'{WIDTH*.15+2}mm',height=f'{HEIGHT*.15+2}mm',
                   viewBox=f'{-margin} {-margin} {WIDTH+2*margin} {HEIGHT+2*margin}')
    ET.SubElement(svg,'rect',x=str(-margin),y=str(-margin),width=str(WIDTH+2*margin),height=str(HEIGHT+2*margin),fill='white')
    defs=ET.SubElement(svg,'defs')
    marker=ET.SubElement(defs,'marker',id='arrow',markerWidth='8',markerHeight='8',refX='7',refY='4',orient='auto',markerUnits='userSpaceOnUse',viewBox='0 0 8 8')
    ET.SubElement(marker,'path',d='M0 0 L8 4 L0 8 Z',fill=INK)
    for c in ET.parse(source).getroot().findall('./root/mxCell'):
        if 'style' not in c.attrib: continue
        st=dict(p.split('=',1) if '=' in p else (p,'1') for p in c.attrib['style'].split(';') if p)
        g=c.find('mxGeometry')
        if c.get('edge')=='1':
            src=g.find("mxPoint[@as='sourcePoint']")
            dst=g.find("mxPoint[@as='targetPoint']")
            points=[src]+list(g.findall('./Array/mxPoint'))+[dst]
            attrs={'points':' '.join(f'{p.get("x")},{p.get("y")}' for p in points),'fill':'none','stroke':st['strokeColor'],'stroke-width':st['strokeWidth'],'stroke-linejoin':'round'}
            if st.get('dashed')=='1': attrs['stroke-dasharray']='8 6'
            if st.get('endArrow')!='none':attrs['marker-end']='url(#arrow)'
            ET.SubElement(svg,'polyline',attrs)
            continue
        x,y,w,h=(float(g.get(k,'0')) for k in ('x','y','width','height'))
        if c.get('imageData'):
            ET.SubElement(svg,'image',x=str(x),y=str(y),width=str(w),height=str(h),href=c.get('imageData'))
        elif st.get('text'):
            size=float(st['fontSize']); lines=c.get('value','').split('\n'); step=size*1.16
            cy=y+h/2-(len(lines)-1)*step/2
            for i,line in enumerate(lines):
                t=ET.SubElement(svg,'text',x=str(x+w/2),y=str(cy+i*step),fill=st['fontColor'],**{'font-family':'Arial, sans-serif','font-size':str(size),'font-weight':'bold' if st['fontStyle']=='1' else 'normal','text-anchor':'middle','dominant-baseline':'central'})
                t.text=line
        elif st.get('shape')=='star':
            pts=[]
            for k in range(10):
                ang=-mathlib.pi/2+k*mathlib.pi/5
                radius=w/2 if k%2==0 else w*.22
                pts.append(f'{x+w/2+radius*mathlib.cos(ang)},{y+h/2+radius*mathlib.sin(ang)}')
            ET.SubElement(svg,'polygon',points=' '.join(pts),fill=st['fillColor'],stroke=st['strokeColor'],**{'stroke-width':st['strokeWidth']})
        elif st.get('ellipse'):
            ET.SubElement(svg,'ellipse',cx=str(x+w/2),cy=str(y+h/2),rx=str(w/2),ry=str(h/2),fill=st['fillColor'],stroke=st['strokeColor'],**{'stroke-width':st['strokeWidth']})
        else:
            ET.SubElement(svg,'rect',x=str(x),y=str(y),width=str(w),height=str(h),rx='12' if st.get('rounded')=='1' else '0',fill=st['fillColor'],stroke=st['strokeColor'],**{'stroke-width':st['strokeWidth']})
    ET.ElementTree(svg).write(destination,encoding='utf-8',xml_declaration=True)


if __name__=='__main__':
    build()
