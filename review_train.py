"""Render TRAIN predictions for coordinate/preprocessing inspection only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pipeline import ROOT,COORDS,choose_box


def main():
    train=pd.read_csv(ROOT.parent/'data/csv/train.csv',dtype={'id':str})
    examples=['783306ebdb135ea7','ef2b5697003f3385','bd84fbc2dccb1310',
              '551de8deea29de5d','6c2455147f4579be','9ba000a9790dff61']
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    for ax,identifier in zip(axes.flat,examples):
        row=train[train.id.eq(identifier)].iloc[0]
        box=row[COORDS].to_numpy(float)
        detector=json.loads((ROOT/'outputs/cache/detector'/f'{row.frame_id}.json').read_text())
        pose=json.loads((ROOT/'outputs/cache/pose'/f'{row.id}.json').read_text())
        match,overlap,confidence,_=choose_box(box,detector)
        image=Image.open(ROOT.parent/'data/imgs'/f'{row.frame_id}.jpg')
        ax.imshow(image)
        ax.add_patch(Rectangle(box[:2],*(box[2:]-box[:2]),fill=False,edgecolor='red',linewidth=2,label='original'))
        if match is not None:
            ax.add_patch(Rectangle(match[:2],*(match[2:]-match[:2]),fill=False,edgecolor='lime',linewidth=2,label='DEIM'))
        points=np.asarray(pose['keypoints'])
        valid=np.asarray(pose['scores'])>=.3
        ax.scatter(points[valid,0],points[valid,1],c='cyan',s=12,label='pose >= 0.3')
        window=np.array(pose['window'])
        ax.set_xlim(max(0,window[0]-20),min(image.width,window[2]+20))
        ax.set_ylim(min(image.height,window[3]+20),max(0,window[1]-20))
        ax.set_title(f'TRAIN error={row.error}; agreement IoU={overlap:.3f}\nDEIM confidence={confidence:.3f}')
        ax.axis('off')
    axes[0,0].legend(loc='lower left',fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/'outputs/train_coordinate_review.png',dpi=130)
    plt.close(fig)


if __name__=='__main__':
    main()
