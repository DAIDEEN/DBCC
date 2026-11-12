"""Frozen candidate calibration and point evaluation."""
import numpy as np
from scipy import ndimage, spatial, special

def proposals(prob,gen):
 if gen.startswith('cc'):
  t={'cc025':.25,'cc050':.5}.get(gen,float(gen[2:])/100 if gen[2:].isdigit() else .5);pl,n=ndimage.label(prob>=t,np.ones((3,3)));result=[]
  for j,b in enumerate(ndimage.find_objects(pl),1):
   if b is None:continue
   local=pl[b]==j;yy,xx=np.where(local);result.append((float(yy.mean()+b[0].start),float(xx.mean()+b[1].start),float(prob[b][local].max()),int(local.sum()),float(local.mean())))
  return result
 size=int(gen[-1]);mask=(prob==ndimage.maximum_filter(prob,size=size,mode='nearest'))&(prob>=.05);pl,n=ndimage.label(mask,np.ones((3,3)));result=[]
 for j,b in enumerate(ndimage.find_objects(pl),1):
  if b is None:continue
  ys,xs=np.where(pl[b]==j);jj=int(np.argmin((ys-ys.mean())**2+(xs-xs.mean())**2));y=int(ys[jj]+b[0].start);x=int(xs[jj]+b[1].start);s=float(prob[y,x]);p=prob[max(0,y-4):y+5,max(0,x-4):x+5];support=p>=max(.05,s*.5);sl,sn=ndimage.label(support,np.ones((3,3)));cy=y-max(0,y-4);cx=x-max(0,x-4);lab=sl[cy,cx];local=sl==lab if lab else support;yy,xx=np.where(local);area=int(local.sum());fill=area/max((int(yy.max()-yy.min()+1)*int(xx.max()-xx.min()+1)),1) if len(yy) else 0.
  result.append((float(y),float(x),s,area,fill))
 return result

def adjacency(coords,gtcoords):
 if not len(coords):return []
 return [[int(g)+1 for g in gs] for gs in spatial.cKDTree(gtcoords).query_ball_point(np.asarray(coords),3.)] if len(gtcoords) else [[] for _ in coords]

def predict(images,m):
 result=[]
 for im in images:
  if not im['items']:result.append(np.array([]));continue
  X=np.asarray(im['features'])[:,m['columns']];X=np.c_[np.ones(len(X)),(X-np.asarray(m['mean']))/np.asarray(m['std'])];result.append(special.expit(X.dot(m['coef'])))
 return result

def extract_features(prob,image,border,gtcoords,gen):
 h,w=prob.shape;candidates=proposals(prob,gen);coords=[(x,y) for y,x,*_ in candidates];adj=adjacency(coords,gtcoords);tree=spatial.cKDTree(coords) if coords else None;neighbors=tree.query_ball_point(coords,8.) if tree is not None else [];items=[];features=[]
 for j,((yf,xf,s,area,fill),edges) in enumerate(zip(candidates,adj)):
  y=int(np.clip(round(yf),0,h-1));x=int(np.clip(round(xf),0,w-1));y0=max(0,y-6);y1=min(h,y+7);x0=max(0,x-6);x1=min(w,x+7);p=image[y0:y1,x0:x1].astype(float);yy,xx=np.mgrid[y0:y1,x0:x1];ring=(abs(yy-y)>2)|(abs(xx-x)>2);rv=p[ring];med=float(np.median(rv)) if rv.size else float(p.mean());mad=max(float(np.median(abs(rv-med)))*1.4826,3.) if rv.size else 3.;contrast=min(20.,abs(float(image[max(0,y-1):y+2,max(0,x-1):x+2].mean())-med)/mad);ps=prob[y0:y1,x0:x1];background=float(np.median(ps[ring])) if rv.size else float(ps.mean());prominence=s-background;sharpness=s-float(prob[max(0,y-1):y+2,max(0,x-1):x+2].mean());signed=(yf-float(border[int(np.clip(round(xf),0,w-1))]))/h
  items.append(dict(score=s,overlap=edges));features.append([float(special.logit(np.clip(s,1e-5,1-1e-5))),float(np.log1p(area)),contrast,fill,prominence,float(np.log1p(max(0,len(neighbors[j])-1))),sharpness,float(yf/h),float(signed<=0),signed,float(np.exp(-abs(signed)/.03))])
 return items,features

def apply(image,score,selection):
 assert image.shape==score.shape and np.isfinite(score).all()
 if 'fallback' in selection:
  gen='cc050';candidates=proposals(score,gen);return [dict(x=p[1],y=p[0],confidence=p[2]) for p in candidates]
 gen=selection['generator'];m=selection['model'];assert not set(m['columns'])&{7,8,9,10},'This entry point supports annotation-free mixed models only'
 items,features=extract_features(score,image,np.zeros(score.shape[1]),np.empty((0,2)),gen)
 for f in features:f.extend([f[0]*f[5],f[2]*f[5]])
 prob=predict([dict(items=items,features=features)],m)[0];candidates=proposals(score,gen)
 return [dict(x=p[1],y=p[0],confidence=float(q),detector_score=p[2]) for p,q in zip(candidates,prob) if q>=selection['threshold']]

def match(items,keep):
    accepted=[i for i,k in enumerate(keep) if k];matched={}
    def augment(i,seen):
        for g in items[i]['overlap']:
            if g in seen:continue
            seen.add(g)
            if g not in matched or augment(matched[g],seen):matched[g]=i;return True
        return False
    for i in accepted:augment(i,set())
    return set(matched.values())

def curve(images,preds):
 events=[];matched=[{} for _ in images];tp=0;count=0;gt=sum(im['gt_count'] for im in images)
 for j,(im,pred) in enumerate(zip(images,preds)):
  events.extend((float(s),j,i) for i,s in enumerate(pred))
 events.sort(reverse=True);xs=[0];ys=[0];thresholds=[None]
 def augment(j,i,seen):
  for g in images[j]['items'][i]['overlap']:
   if g in seen:continue
   seen.add(g)
   if g not in matched[j] or augment(j,matched[j][g],seen):matched[j][g]=i;return True
  return False
 for k,(s,j,i) in enumerate(events):
  count+=1;tp+=int(augment(j,i,set()))
  if k==len(events)-1 or events[k+1][0]!=s:xs.append(count-tp);ys.append(tp);thresholds.append(s)
 return dict(FP=xs,TP=ys,thresholds=thresholds,GT=gt,images=len(images),Pd=[a/max(gt,1) for a in ys],FP_per_image=[a/max(len(images),1) for a in xs])
