"""Audit the continuous Isaac recording, every model row, and the measured terminal state."""
from pathlib import Path
import argparse, gzip, hashlib, io, json, math, subprocess, tarfile
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull, distance

def bytes_hash(x):return hashlib.sha256(x).hexdigest()
def pose(obj):
    p,q=obj; t=np.eye(4);t[:3,3]=p;t[:3,:3]=Rotation.from_quat(np.roll(q,-1)).as_matrix();return t
def rect(points):
    xy=points[:,:2];p=xy[ConvexHull(xy).vertices];edge=np.roll(p,-1,axis=0)-p
    angles=np.arctan2(edge[:,1],edge[:,0]);best=float('inf');sides=None
    for a in angles:
        c,s=np.cos(a),np.sin(a)
        ext=np.ptp(p@np.array([[c,-s],[s,c]]),axis=0);area=ext.prod()
        if area<best:sides=np.sort(ext);best=area
    return sides
def verify(tag,rawbase):
    p = rawbase / tag
    result = json.loads((p / "result.json").read_text())
    plain = p / "trace.jsonl"
    with (plain.open() if plain.is_file() else gzip.open(p / "trace.jsonl.gz", "rt")) as trace:
        rows = [json.loads(line) for line in trace]
    vid=p/'trial.mp4'; meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,codec_name,r_frame_rate,nb_read_frames,duration','-of','json',vid]))['streams'][0]
    assert result['status']=='horizon_reached' and result['encoder_rc']==0 and result['frames']==len(rows)+1==int(meta['nb_read_frames'])
    assert meta['codec_name']=='h264' and meta['r_frame_rate']=='25/1' and (meta['width'],meta['height'])==(1280,540)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(vid),'-f','null','-'],check=True)
    k=result['model_calls'];assert len(rows)==k*8
    with tarfile.open(p/'model_actions.tgz') as ar:
        for i in range(k):
            get=lambda n:ar.extractfile(f'{i:03d}'+n).read()
            data=get('.action.npy');act=np.load(io.BytesIO(data),allow_pickle=False);reply=json.loads(get('.reply'));req=json.loads(get('.request'))
            assert act.shape==(32,26) and np.isfinite(act).all() and reply['action_sha256']==bytes_hash(data)
            for j in range(8):
                idx=i*8+j;r=rows[idx];assert r['chunk']==i and r['row']==j and np.array_equal(act[j],r['row26'])
                assert r['chunk_sha256']==reply['action_sha256'] and r['observation_sha256']==reply['observation_sha256']
                steps=round(.04/result['dt']);assert r['tick']==result['sim_start_tick']+(idx+1)*steps and abs(r['time']-(idx+1)*.04)<1.e-6
                if j==0:assert req['physics_tick']==r['tick']-steps
                assert all(np.isfinite(r[s]).all() for s in ('target','measured_after','measured_before','row26'))
            if i in (0,k-1):
                src=p/f'chunk_{i:03d}.npz';assert bytes_hash(src.read_bytes())==reply['observation_sha256']
                with np.load(src,allow_pickle=False) as obs:
                    assert obs['state'].shape==(26,) and all(obs[c].shape==(480,640,3) and obs[c].std()>1 for c in ['face_view','right_wrist_view'])
    task=result['task'];o=result['final_props'];first=result['initial_props']
    stat={'task':task,'run':tag,'sim_s':rows[-1]['time'],'model_calls':k,'executed_action_rows':len(rows),'frames':len(rows)+1,'video_sha256':bytes_hash(vid.read_bytes()),'video':meta,'all_action_rows_exact':True}
    if task=='D1':
        pairs=[('red2','yellow'),('red0','yellow'),('blue4','mint')]
        lift=[next(x['time'] for x in rows if x['props'][n][0][2]-first[n][0][2]>.03)for n,b in pairs]
        err=lambda o,n,b:np.array(o[n][0])-np.array(o[b][0])
        last=rows[-10:];correct=all(np.linalg.norm(err(x['props'],n,b)[:2])<.029 and .009<err(x['props'],n,b)[2]<.026 for x in last for n,b in pairs)
        bowl=np.linalg.norm(np.array(o['mint'][0][:2])-[.54,-.408])*1000;other=max(abs(o[n][0][2]-.642)for n in ('red1','blue3','green5'))
        stat.update(lift_s=lift,targets_in_correct_bowls=sum(np.linalg.norm(err(o,n,b)[:2])<.029 for n,b in pairs),bowl_position_error_mm=bowl,remaining_cube_z_error_mm=other*1000,sim_pass=bool(correct and sorted(lift)==lift and bowl<30 and other<.001))
    elif task=='D2':
        gone=o['erased_mark_ids'];allgone=set().union(*(set(x['props']['erased_mark_ids']) for x in rows));drift=lambda key:1000*np.linalg.norm(np.array(o[key][0])-first[key][0])
        dock=drift('EraserProxy');stat.update(targets_erased=len(set(gone)&set(range(8))),protected_remaining=6-len(allgone&set(range(8,14))),tool_dock_error_mm=dock,board_shift_mm=drift('Board'),peak_eraser_board_contact_N=max(x['props']['eraser_board_contact_n']for x in rows),sim_pass=bool(set(gone)==set(range(8)) and not allgone&set(range(8,14)) and dock<10))
    elif task=='D3':
        B,C=pose(o['Bottle']),pose(o['Cap']);T=np.linalg.inv(B)@C
        upright=np.degrees(np.arccos(np.clip(B[2,2],-1,1)));tilt=np.arccos(np.clip(T[2,2],-1,1));height_err=1000*abs(T[2,3]-.0547)
        xy=1000*np.linalg.norm(B[:2,3]-[.60,.116]);shoulder=1000*.042*abs(np.sin(tilt))
        times=[ev['time'] for ev in result['grasp_events'] if ev['event']=='release' and ev['side']=='left'];released=rows[-1]['time']-max(times) if times else 0.
        stat.update(turns=result['physical_turns'],seat_detected=result['seat_detected'],height_error_mm=height_err,opposed_lip_height_difference_mm=shoulder,upright_deg=upright,delivery_error_mm=xy,after_release_s=released,sim_pass=bool(result['seat_detected'] and upright<5 and xy<30 and height_err<1.5 and shoulder<2 and released>=1.))
    elif task=='D4':
        with np.load(p/'cloth_nodes.npz') as a:pts=a['samples'][-1].astype(float)
        sides=rect(pts)*1000;corners=pts[[0,14,210,224],:2];seam=float(distance.pdist(corners).max()*1000);center=np.mean(pts,axis=0);zone=result.get('storage_center_xy_m',[.435,.14]);region=bool(np.all(np.abs(center[:2]-zone)<.065));events=result['grasp_events']
        pinch={v.get('phase')for v in events if v['event']=='pinch_nodes'};bothfold='pinch_first_edge' in pinch and ('pinch_second_edge' in pinch or {'pinch_supported','pinch_hinge'}<=pinch)
        def stage(name):
            return next(x['props'] for x in rows if x['phase']==name)
        def landmarks(props,indices):
            return np.mean([props['material_landmarks'][str(i)][:2] for i in indices],axis=0)
        half=stage('above_second');quarter=stage('release_second_fold')
        first_dim=1000*rect(np.asarray(half['material_grid']))
        v1=landmarks(half,[210,224])-landmarks(first,[210,224])
        v2=landmarks(quarter,[0,105,210])-landmarks(half,[0,105,210])
        angle=float(np.rad2deg(np.arccos(np.clip(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)),-1,1))))
        fold2seam=1000*distance.pdist(np.asarray(quarter['cloth_corners_world_m'])[:,:2]).max()
        releases={side:max(e['time'] for e in events if e['event']=='release_nodes' and e['side']==side) for side in ['left','right']}
        free=rows[-1]['time']-max(releases.values())
        tail=[x for x in rows if x['time']>=rows[-1]['time']-2]
        last_seams=[1000*distance.pdist(np.asarray(x['props']['cloth_corners_world_m'])[:,:2]).max() for x in tail]
        tail_dims=np.array([1000*rect(np.asarray(x['props']['material_grid'])) for x in tail])
        position_error=float(1000*np.linalg.norm(center[:2]-zone))
        transported=float(1000*np.linalg.norm(center[:2]-np.asarray(stage('survey_stack')['cloth_center_world_m'])[:2]))
        align_ok=bool(130<=first_dim[0]<=170 and 280<=first_dim[1]<=320 and 80<=angle<=100)
        stable=bool(np.all((tail_dims>=130)&(tail_dims<=170)) and max(last_seams)<=15)
        stat.update(nodes=len(pts),first_fold_mm=first_dim.tolist(),fold_angle_deg=angle,second_fold_corner_spread_mm=float(fold2seam),footprint_mm=sides.tolist(),original_corner_spread_mm=seam,center_m=center.tolist(),storage_center_inside=region,storage_center_xy_m=zone,storage_center_error_mm=position_error,transported_mm=transported,both_released_s=free,last_two_seconds_max_corner_spread_mm=max(last_seams),last_two_seconds_side_range_mm=[tail_dims.min(axis=0).tolist(),tail_dims.max(axis=0).tolist()],pinch_events=sorted(pinch),sim_pass=bool(len(pts)==225 and free>=2 and bothfold and align_ok and stable and position_error<=30 and region and 'grip_folded_stack' in pinch))
    else:
        def met(x):
            o=x['props'];f=pose(o['BenchAssembly/FemaleShell']);m=pose(o['BenchAssembly/ServiceDrive']);r=np.linalg.inv(f)@m
            tip=r[:3,3]+r[:3,2]*.006;shoulders=np.array([tip[2]+sign*.0062*r[2,0] for sign in (-1,1)])*1000
            return tip*1000,shoulders,np.degrees(np.arccos(np.clip(r[2,2],-1,1)))
        released=max(ev['time'] for ev in result['grasp_events'] if ev['event']=='release')
        free=[x for x in rows if x['time']>released]; m=met(rows[-1]);depths=[met(x)[0][2] for x in free]
        hold=rows[-1]['time']-released
        rel={s:max(e['time'] for e in result['grasp_events'] if e['event']=='release' and e['side']==s) for s in ('left','right')}
        one=rel['left']-rel['right']
        min_one=min(met(x)[0][2] for x in rows if rel['right']<x['time']<=rel['left'])
        stat.update(engaged=result['seat_detected'],final_depth_mm=m[0][2],terminal_radial_error_mm=float(np.linalg.norm(m[0][:2])),axes_tilt_deg=float(m[2]),sim_shoulder_deviation_from_nominal_mm=(m[1]-9.5).tolist(),right_release_s=rel['right'],left_release_s=rel['left'],one_hand_hold_s=one,min_one_hand_depth_mm=min_one,both_released_s=hold,minimum_depth_after_hand_release_mm=min(depths),sim_pass=bool(result['seat_detected'] and min(min(depths),min_one)>=9.2 and max(abs(m[1]-9.5))<=1 and hold>=2.0 and one>=2.0))
    return stat
if __name__=='__main__':
    c=argparse.ArgumentParser();c.add_argument('runs',nargs='+');c.add_argument('--raw',type=Path,default=Path(__file__).resolve().parent/'evidence');a=c.parse_args()
    for run in a.runs:
        x=verify(run,a.raw);print(json.dumps(x,ensure_ascii=False,default=lambda a:a.item(),indent=2))
