"""Transfer the folded textile by pinching four measured layers on opposite sides near its center of mass."""
import os
import numpy as np
from scipy.spatial.distance import pdist
from fold import Fold
from arm import limited,tf

class CarryFold(Fold):
    def __init__(self,*args):
        super().__init__(*args)
        self.prefix=self.phases.index('above_stack')
        self.phases=self.phases[:self.prefix]+['survey_stack','touch_layers','grip_folded_stack','test_lift']
        self.phases+=['carry_'+str(i) for i in range(1,11)]+['lower_stack','release_in_storage','rise_clear','inspect']
        self.storage=np.array([.43,.116])
        self.picks=None
        self.high=.020 if self.variant=='carry' else .035
        if self.variant.startswith('carry_corners'):
            self.storage=np.array([.419,.095])
            self.high=.024
    def select(self,p):
        if self.variant.startswith('carry_corners'):
            # Keep the four free corners together while supporting the diagonally opposite hinge.
            return np.array([0,14,210,224]),np.array([6*15+6,6*15+8,8*15+6,8*15+8])
        def candidates(xset):
            out=[]
            for i in xset:
                for j in (3,4,5):
                    ids=np.array([i*15+j,i*15+14-j,(14-i)*15+j,(14-i)*15+14-j])
                    a=p[ids];d=float(np.max(pdist(a)));c=a.mean(axis=0)
                    out.append((d,ids,c))
            return out
        pairs=[]
        for a in candidates((0,1)):
            for b in candidates((5,6)):
                dist=np.linalg.norm(a[2][:2]-b[2][:2])
                if dist<.094:continue
                cost=max(a[0],b[0])+.22*abs(a[2][1]-b[2][1])+.25*abs((a[2][1]+b[2][1])/2-p[:,1].mean())
                pairs.append((cost,a,b))
        if not pairs:raise RuntimeError('folded layers lack opposing grasp sites')
        _,a,b=min(pairs,key=lambda v:v[0]);return a[1],b[1]
    def goals(self,row,wrists,t,props):
        if self.ix<self.prefix:return super().goals(row,wrists,t,props)
        name=self.phases[min(self.ix,len(self.phases)-1)]
        p=np.asarray(props['material_grid'])
        if name!=self.seen:
            self.seen=name;self.age=0.;self.steady=0.
            if name=='survey_stack':
                self.picks=self.select(p);self.grip=np.stack([p[ids].mean(axis=0) for ids in self.picks])
                self.events.append(dict(phase=name,physics_s=t,material_points=[v.tolist() for v in self.picks],layer_spread_mm=[float(1000*pdist(p[v]).max()) for v in self.picks],outcome='selected'))
            if name=='test_lift':
                self.grip=np.stack([wrists[k][:3,3].copy() for k in ('left','right')]);self.grip[:,2]-=.1605
                self.center_at_pick=p.mean(axis=0)
                self.delta=self.storage-self.center_at_pick[:2]
        loc=self.grip.copy();gap=.002;h=0.;dwell=.25;tol=.006
        if name=='survey_stack':gap=.024;h=.082
        elif name in ('touch_layers','grip_folded_stack'):
            loc=np.stack([p[ids].mean(axis=0) for ids in self.picks])
            if name=='touch_layers':gap=.017;h=.0006;dwell=.40
            else:dwell=.72
        elif name=='test_lift':h=self.high;dwell=.48
        elif name.startswith('carry_'):
            i=int(name.split('_')[1]);h=self.high;loc[:,:2]+=self.delta*i/10;dwell=.16
        else:
            loc[:,:2]+=self.delta
            if name=='lower_stack':h=.0006;dwell=.60
            elif name=='release_in_storage':gap=.021;h=.002;dwell=.55
            elif name=='rise_clear':gap=.020;h=.028;dwell=.3
            else:gap=.027;h=.12;dwell=2.05
        residual=limited(row[10:13]*.1,.00015)
        goals={}
        for i,(k,R) in enumerate((('left',self.L),('right',self.R))):
            q=loc[i].copy();q[2]+=h+.1605
            if k=='right':q+=residual
            goals[k]=(tf(q,R),gap)
        error=max(np.linalg.norm(wrists[k][:3,3]-v[0][:3,3]) for k,v in goals.items())
        self.age+=.04;self.steady=self.steady+.04 if error<tol else 0.
        if self.steady>=dwell or self.age>2.8:
            self.events.append(dict(phase=name,physics_s=t,position_error_m=float(error),outcome='arrived' if self.steady>=dwell else 'timeout'));self.ix+=1
        return goals,name,residual
