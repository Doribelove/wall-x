"""Fold a measured deformable surface toward the shared arm workspace and deliver it with two contacts."""
import numpy as np
import os
from dexterous import DOWN,zrot
from arm import tf,limited

class Fold:
    def __init__(self,task,props,home):
        self.variant=os.getenv("FOLD_CASE","reference")
        self.ix=0;self.phases=[];self.events=[];self.steady=0.;self.age=0.;self.seen=''
        self.saved={};self.reach_fail=False;self.storage=np.array([.435,.14])
        self.p0=self.points(props)
        self.L=zrot(np.deg2rad(0))@DOWN;self.R=zrot(np.deg2rad(78))@DOWN
        self.phases+=['find_tag','touch_first','pinch_first_edge']+['fold_x_'+str(i) for i in range(1,13)]+['release_first_fold','clear_first']
        self.phases+=['above_second','touch_second','pinch_second_edge']+['fold_y_'+str(i) for i in range(1,13)]+['release_second_fold','clear_second']
        self.phases+=['above_stack','touch_stack','grip_folded_stack','lift_stack']+['transfer_stack_'+str(i) for i in range(1,7)]+['set_stack','release_in_storage','inspect']
    @staticmethod
    def points(props):return {int(k):np.array(v)for k,v in props["material_landmarks"].items()}
    def goals(self,row,wrists,t,props):
        pts=self.points(props);name=self.phases[min(self.ix,len(self.phases)-1)]
        if name!=self.seen:
            self.age=0.;self.steady=0.;self.seen=name
            if name=='find_tag':
                self.first=np.stack([pts[224],pts[210]]);self.first_end=np.stack([pts[14],pts[0]])
            if name=='above_second':
                self.second=np.stack([.5*(pts[0]+pts[210]),pts[105]])
                self.second_end=np.stack([.5*(pts[14]+pts[224]),pts[119]])
            if name=='above_stack':
                self.stack=np.stack([.5*(pts[7]+pts[217]),pts[112]])
                self.delta=self.storage-np.array(props['cloth_center_world_m'])[:2]
        first=name in ('find_tag','touch_first','pinch_first_edge','release_first_fold','clear_first') or name.startswith('fold_x_')
        second=name in ('above_second','touch_second','pinch_second_edge','release_second_fold','clear_second') or name.startswith('fold_y_')
        gap=.002;dwell=.18;tol=.007;h=0.
        if first or second:
            start,end=(self.first,self.first_end) if first else (self.second,self.second_end); loc=start.copy()
            if name in ('find_tag','above_second'):gap=.024;h=.065
            elif name in ('touch_first','touch_second'):gap=.019;h=.001;dwell=.28
            elif name in ('pinch_first_edge','pinch_second_edge'):h=.001;dwell=.65
            elif name.startswith(('fold_x_','fold_y_')):
                i=int(name.split('_')[-1]);angle=np.pi*i/12
                loc=start+(end-start)*(.5-.5*np.cos(angle))
                h=(.105 if first else .110)*np.sin(angle)+.003*(i/12);dwell=.11;tol=.008
                if self.variant!="reference":
                    dwell=.20
                    if second:
                        # Follow the actual stationary hem and hinge edge as the cloth slips.
                        live=np.stack([.5*(pts[14]+pts[224]),pts[119]])
                        drift=np.clip(live-end,-.030,.030)
                        loc+=drift*(i/12)
                        h=.083*np.sin(angle)+.002*(i/12)

            elif name.startswith('release'):loc=end.copy();gap=.024;h=.004;dwell=.70
            else:loc=end.copy();gap=.026;h=.090
        else:
            loc=self.stack.copy()
            if name=='above_stack':gap=.026;h=.075
            elif name=='touch_stack':gap=.019;h=.001;dwell=.36
            elif name=='grip_folded_stack':h=.001;dwell=.65
            elif name=='lift_stack':h=.008;dwell=.36
            elif name.startswith('transfer_stack_'):
                i=int(name.split('_')[-1]);loc[:,:2]+=self.delta*(i/6);h=.008;dwell=.22
            else:
                loc[:,:2]+=self.delta
                if name=='set_stack':h=.001;dwell=.45
                elif name=='release_in_storage':gap=.025;h=.003;dwell=.60
                else:gap=.026;h=.080;dwell=1.4
        residual=limited(row[10:13]*.10,.0002)
        goals={}
        for i,(s,R) in enumerate((('left',self.L),('right',self.R))):
            p=loc[i].copy();p[2]+=h+.1624
            if s=='right':p+=residual
            goals[s]=(tf(p,R),gap)
        err=max(np.linalg.norm(W[:3,3]-goals[s][0][:3,3]) for s,W in wrists.items())
        self.age+=.04;self.steady=self.steady+.04 if err<tol else 0.
        if self.steady>=dwell or self.age>2.6:
            self.events.append(dict(phase=name,physics_s=t,position_error_m=float(err),outcome='arrived' if self.steady>=dwell else 'timeout'))
            self.ix+=1
        return goals,name,residual
