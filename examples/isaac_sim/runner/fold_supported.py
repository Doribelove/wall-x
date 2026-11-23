"""Use one hand as a physical brace while the other folds each end of the second hem."""
import numpy as np
from fold import Fold
from arm import tf,limited
from dexterous import DOWN,zrot

class SupportedFold(Fold):
    def __init__(self,*args):
        super().__init__(*args)
        self.prefix=self.phases.index('above_second')
        self.phases=self.phases[:self.prefix]+['find_support','touch_support','pinch_supported']
        self.phases+=['seam_arc_'+str(i) for i in range(1,15)]
        self.phases+=['secure_corners','release_brace','find_hinge','touch_hinge','pinch_hinge']
        self.phases+=['hinge_arc_'+str(i) for i in range(1,15)]
        self.phases+=['settle_second','grip_folded_stack','raise_folded_stack']
        self.phases+=['store_'+str(i) for i in range(1,9)]+['set_stack','release_in_storage','inspect']
        self.storage=np.array([.42,-.01])
    def goals(self,row,wrists,t,props):
        if self.ix<self.prefix:return super().goals(row,wrists,t,props)
        pts=self.points(props);name=self.phases[min(self.ix,len(self.phases)-1)]
        def pair(i,j):return .5*(pts[i]+pts[j])
        if name!=self.seen:
            self.seen=name;self.age=0.;self.steady=0.
            if name=='find_support':
                self.start_seam=pair(0,210);self.brace=pts[119].copy();self.end_seam=pair(14,224)
            if name=='seam_arc_1':self.start_seam=pair(0,210);self.end_seam=pair(14,224)
            if name=='secure_corners':self.held_seam=pair(14,224)
            if name=='find_hinge':self.move_hinge=pts[105].copy();self.end_hinge=pts[119].copy()
            if name=='hinge_arc_1':self.move_hinge=pts[105].copy();self.end_hinge=pts[119].copy()
            if name=='settle_second':
                self.bundle=np.stack([self.held_seam,self.end_hinge])
            if name=='raise_folded_stack':
                self.delta=self.storage-np.array(props['cloth_center_world_m'])[:2]
        gl=.002;gr=.002;dwell=.21;tol=.006
        L=self.start_seam.copy();R=self.brace.copy()
        def arc(a,b,i):
            angle=i*np.pi/14
            q=a+(b-a)*(.5-.5*np.cos(angle));q[2]+=.085*np.sin(angle)+.0025*(i/14)
            return q
        if name in ('find_support','touch_support','pinch_supported'):
            if name=='find_support':gl=gr=.024;L[2]+=.07;R[2]+=.07
            elif name=='touch_support':gl=gr=.018;L[2]+=.001;R[2]+=.001;dwell=.32
            else:dwell=.72
        elif name.startswith('seam_arc_'):
            i=int(name.split('_')[-1]);dest=pair(14,224)
            # Follow the ungrasped target edge, which can slide on the tabletop.
            dest=np.minimum(np.maximum(dest,self.end_seam-[.05,.05,.006]),self.end_seam+[.05,.05,.012])
            L=arc(self.start_seam,dest,i);dwell=.16
        else:
            L=self.held_seam.copy()
            if name=='secure_corners':dwell=.70
            elif name=='release_brace':gr=.024;R[2]+=.042;dwell=.36
            elif name in ('find_hinge','touch_hinge','pinch_hinge'):
                R=self.move_hinge.copy()
                if name=='find_hinge':gr=.025;R[2]+=.065
                elif name=='touch_hinge':gr=.018;R[2]+=.001;dwell=.35
                else:dwell=.70
            elif name.startswith('hinge_arc_'):
                i=int(name.split('_')[-1]);R=arc(self.move_hinge,self.end_hinge,i);dwell=.16
            else:
                loc=self.bundle.copy();h=.003
                if name in ('settle_second','grip_folded_stack'):dwell=.75
                elif name=='raise_folded_stack':h=.012;dwell=.4
                elif name.startswith('store_'):
                    i=int(name.split('_')[-1]);loc[:,:2]+=self.delta*i/8;h=.012;dwell=.20
                else:
                    loc[:,:2]+=self.delta
                    if name=='set_stack':dwell=.62
                    elif name=='release_in_storage':gl=gr=.025;h=.008;dwell=.65
                    else:gl=gr=.026;h=.095;dwell=1.55
                L,R=loc;L[2]+=h;R[2]+=h
        g={};residual=limited(row[10:13]*.1,.00018)
        for side,p,rot,gap in [('left',L,self.L,gl),('right',R,self.R,gr)]:
            p=p.copy();p[2]+=.1624
            if side=='right':p+=residual
            g[side]=(tf(p,rot),gap)
        error=max(np.linalg.norm(wrists[s][:3,3]-v[0][:3,3]) for s,v in g.items())
        self.age+=.04;self.steady=self.steady+.04 if error<tol else 0.
        if self.steady>=dwell or self.age>2.6:
            self.events.append(dict(phase=name,physics_s=t,position_error_m=float(error),outcome='arrived' if self.steady>=dwell else 'timeout'))
            self.ix+=1
        return g,name,residual
