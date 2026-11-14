"""Reproducible visual-review sheets. Coordinates in annotations use the 0..100 axes.

Manual reconstructions are estimates, not organizer ground truth.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[4]
D=ROOT/'work/Detector-Error-Discovery/data/raw'
OUT=ROOT/'outputs/manual_bbox_review'
OUT.mkdir(exist_ok=True)
SHEETS=OUT/'sheets'; SHEETS.mkdir(exist_ok=True)
FONT=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',14)
SMALL=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',11)
BOX=['x1','y1','x2','y2']

def window(r):
    x1,y1,x2,y2=[float(r[k]) for k in BOX]
    w,h=x2-x1,y2-y1
    side=max(w,h)*(1.9 if r.class_name=='person' else 3.0)
    side=max(side,240 if r.class_name=='person' else 70)
    side=min(side,720)
    cx=(x1+x2)/2; cy=(y1+y2)/2+(h*.18 if r.class_name=='person' else 0)
    a=int(np.clip(cx-side/2,0,1280-side)); b=int(np.clip(cy-side/2,0,720-side))
    side=int(side)
    return [a,b,a+side,b+side]

def panel(r,gt=None,show_label=False):
    im=Image.new('RGB',(440,460),'#202126'); draw=ImageDraw.Draw(im)
    crop=window(r); a,b,c,d=crop
    raw=Image.open(D/f'{r.frame_id}.jpg').convert('RGB')
    im.paste(raw.crop(crop).resize((400,400)),(32,35))
    draw=ImageDraw.Draw(im)
    draw.text((4,2),f'{r.review_number:03} {r.class_name} {r.frame_id[-13:]}',font=FONT,fill='white')
    def xy(bb): return [(bb[0]-a)*400/(c-a)+32,(bb[1]-b)*400/(d-b)+35,(bb[2]-a)*400/(c-a)+32,(bb[3]-b)*400/(d-b)+35]
    for k in range(0,101,10):
        p=k*4
        draw.text((28+p,20),str(k),font=SMALL,fill='#ffeb8a')
        draw.text((1,30+p),str(k),font=SMALL,fill='#ffeb8a')
        # Short ticks avoid obscuring tiny hands and cup boundaries.
        draw.line((32+p,35,32+p,42),fill='#ffeb8a')
        draw.line((32,35+p,39,35+p),fill='#ffeb8a')
    draw.rectangle(xy([r[k] for k in BOX]),outline='#38ff48',width=1)
    if gt is not None:
        for bb in gt: draw.rectangle(xy(bb),outline='#ff4bff',width=2)
    text=f'crop {a},{b},{c},{d}'
    if show_label: text+=f' error={r.error}'
    draw.text((5,440),text,font=FONT,fill='white')
    return im,crop

def generate():
    test=pd.read_csv(ROOT/'work/offline_review/index.csv')
    coords=[]
    for page in range((len(test)+5)//6):
        sheet=Image.new('RGB',(1320,920))
        for j,(_,r) in enumerate(test.iloc[page*6:page*6+6].iterrows()):
            p,crop=panel(r); sheet.paste(p,((j%3)*440,(j//3)*460))
            coords.append(dict(review_number=int(r.review_number),crop=crop))
        sheet.save(SHEETS/f'test_{page:02}.jpg',quality=94)
    (OUT/'crop_coordinates.json').write_text(json.dumps(coords,indent=2))
    train=pd.read_csv(D/'train.csv'); train=train[train.class_name.isin(['person','nude_hands','cup','tray'])]
    gt=pd.read_csv(D/'human_boxes_train.csv'); gt.class_name=gt.class_name.replace({'nude_hand':'nude_hands'})
    gt['x1']=(gt.cx-gt.w/2)*1280;gt['x2']=(gt.cx+gt.w/2)*1280
    gt['y1']=(gt.cy-gt.h/2)*720;gt['y2']=(gt.cy+gt.h/2)*720
    samples=pd.concat([q.sample(min(4,len(q)),random_state=42) for _,q in train.groupby(['class_name','error'])]).reset_index(drop=True)
    samples['review_number']=np.arange(1,len(samples)+1)
    samples.to_csv(OUT/'calibration_samples.csv',index=False)
    for page in range((len(samples)+5)//6):
        sheet=Image.new('RGB',(1320,920))
        for j,(_,r) in enumerate(samples.iloc[page*6:page*6+6].iterrows()):
            g=gt[(gt.frame_id==r.frame_id)&(gt.class_name==r.class_name)][BOX].values
            p,_=panel(r,g,True);sheet.paste(p,((j%3)*440,(j//3)*460))
        sheet.save(SHEETS/f'calibration_{page:02}.jpg',quality=94)
    print('test panels',len(test),'calibration panels',len(samples),flush=True)

if __name__=='__main__': generate()
