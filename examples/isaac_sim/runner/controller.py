"""Record an EX001 trial using direct model actions or a pose-assisted controller."""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from arm import Arm, action_rotation, tf
from scene_improved import scene_file, configure_stage, film_camera, camera_rgb

root = Path(__file__).resolve().parent
sim = root.parent
tasks = json.loads((root/"assets/tasks.json").read_text())


def save(p,data):
    p.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def run(a):
    run_dir = a.work_dir.resolve()
    runs = run_dir / "runs"
    ipc = run_dir / "ipc"
    runs.mkdir(parents=True, exist_ok=True)
    ipc.mkdir(parents=True, exist_ok=True)
    out = runs / a.uid
    if any(ipc.glob(f"{a.uid}-[0-9]*.request")):
        raise FileExistsError(f"IPC already contains requests for {a.uid}. Use a fresh run ID.")
    out.mkdir(exist_ok=False)
    import faulthandler
    faulthandler.dump_traceback_later(45,repeat=True)
    from lens_5756_rtx import add_startup_extension_args
    sys.argv=add_startup_extension_args([sys.argv[0]])
    from isaacsim import SimulationApp
    app=SimulationApp({"headless":True,"renderer":"RealTimePathTracing","multi_gpu":False,"fast_shutdown":True,
                       "limit_cpu_threads":6,"disable_viewport_updates":True,"create_new_stage":False})
    writer=None
    log=(out/"trace.jsonl").open("w",buffering=1)
    task=a.task
    spec=dict(tasks[task])
    mode=a.mode
    aligned=mode=="aligned"
    if aligned and task=="D1":
        spec["prompt"]="Use the right hand to pick up the red cube directly below the right wrist. Lift it and place it in the yellow bowl."

    assist=mode=="assisted"
    if assist and task=="D1":
        spec["prompt"]="Put the red cube on the right and then the red cube on the left in the yellow bowl. Put the blue cube on the right in the mint bowl and deliver it."
    if assist and task=="D2":
        spec["prompt"]="Use the orange eraser to wipe the black marks on the left. Keep the colored marks and park the eraser."

    reported={"task":task,"title":spec["title"],"prompt":spec["prompt"],"run_id":a.uid,"model_calls":0,"frames":0,
              "control":"Bimanual stock WALL-OSS-0.5 output, capped differential IK with force-limited position drives. Fixed base, lift and head.",
              "physics_engine":"PhysX","physical_scene":"EX001 V6 collision/visual proxy, native task props",
              "control_period_s":.04,"execution_rows_per_chunk":8,"max_chunks":a.chunks,
              "camera_inputs":["face_view","right_wrist_view"],"unverified_official_score":True}
    try:
        from isaacsim.core.experimental.utils.app import enable_extension
        for extension in ("omni.physx","isaacsim.core.api","isaacsim.core.prims","isaacsim.core.version",
                          "isaacsim.sensors.experimental.rtx","isaacsim.core.rendering_manager","omni.usd.schema.omni_lens_distortion"):
            enable_extension(extension)
        app.update()
        import omni.usd
        from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,PhysxSchema
        from isaacsim.core.version import get_version
        from isaacsim.core.api import World
        from isaacsim.core.prims import Articulation, RigidPrim
        from isaacsim.core.rendering_manager import RenderingManager
        from isaacsim.core.simulation_manager import SimulationManager
        from camera_capture import Capture,pose
        from sim_relative_h_pick import attach_home,state
        from PIL import Image,ImageDraw,ImageFont
        reported["isaac_version"]=list(get_version())
        stage_path=(root/"assets/D1_A_fixtures.usda") if assist and task=="D1" else scene_file(task,root/"assets",out)
        reported["mode"]=mode
        if task=="D4":
            reported["fold_case"]=os.getenv("FOLD_CASE","reference")
            reported["storage_center_xy_m"]=[.42,-.01] if os.getenv("FOLD_CASE") in ("supported","crease") else [.419,.095] if os.getenv("FOLD_CASE","").startswith("carry_corners") else [.43,.116] if os.getenv("FOLD_CASE","").startswith("carry") else [.435,.14]
        if assist:
            reported["grasp_mode"]={"D1":"opposed physical friction pads", "D2":"contact-region breakable grasp","D3":"contact-region breakable grasp","D4":"solver node grip within closed finger region","D5":"contact-region breakable grasp"}[task]
        if assist:
            reported["control"]="Task-space servo using simulated initial object coordinates, gripper and wrist feedback, plus capped WALL-OSS right-hand position residual."
        elif aligned:
            reported["control"]="Single right-hand pick test. Absolute lift-local proprioception, blend overlapping world-frame model targets, physics-rate reference interpolation. Other arm holds its initial pose."
        elif mode=="hold":
            reported["control"]="Fixed joint-position hold diagnostic. WALL-OSS proposals recorded but motor targets held."

        ctx=omni.usd.get_context()
        print("INIT open task stage", flush=True)
        ctx.open_stage(str(stage_path))
        for _ in range(50):
            app.update()
            if ctx.get_stage() and ctx.get_stage().GetRootLayer().realPath==str(stage_path):
                break
        st=ctx.get_stage();assert st and st.GetRootLayer().realPath==str(stage_path)
        st.SetEditTarget(st.GetSessionLayer())
        print("INIT scene loaded", flush=True)
        names=configure_stage(st,task)
        if assist and task=="D3":
            st.GetPrimAtPath("/World/Bottle").GetAttribute("xformOp:translate").Set(Gf.Vec3d(.48,-.005,.684))
            st.GetPrimAtPath("/World/Cap").GetAttribute("xformOp:translate").Set(Gf.Vec3d(.415,-.235,.651))
        audit_dir=sim/"robot"
        audit=json.loads((audit_dir/"model.json").read_text())
        assert audit["physx_fixed_base"] and audit["runtime_complete"]
        robot=st.DefinePrim("/World/EX001")
        robot.GetReferences().AddReference(str(audit_dir/audit["usd"]),Sdf.Path(audit["default_prim"]))
        print("INIT robot referenced", flush=True)
        all_prims=list(Usd.PrimRange(robot,Usd.TraverseInstanceProxies()))
        art_root=next(p for p in all_prims if p.HasAPI(UsdPhysics.ArticulationRootAPI))
        joints={p.GetName():p for p in all_prims if p.IsA(UsdPhysics.Joint)}
        bodies={p.GetName():p for p in all_prims if p.HasAPI(UsdPhysics.RigidBodyAPI)}
        from collision_filters import apply_measured_ancestor_filters
        urdf = audit_dir / "urdf/ex001_d1_collision_proxy.urdf"
        apply_measured_ancestor_filters(st, bodies, robot.GetPath(), urdf,
                                        audit_dir / "collision_pairs.json", UsdPhysics)
        PhysxSchema.PhysxArticulationAPI.Apply(art_root).CreateEnabledSelfCollisionsAttr(True)
        arms={side:Arm(urdf,side) for side in ("left","right")}
        finger={s:[f"{s}_arm_gripper_{j}_joint" for j in ("left","right")] for s in arms}
        arm_names=arms["left"].names+arms["right"].names
        fingers=finger["left"]+finger["right"]
        used=arm_names+fingers+["lift_joint","head_yaw_joint","head_pitch_joint"]
        caps={n:(30. if n[-1] in "123" else 9., "angular") for n in arm_names}
        caps.update({n:(10.,"linear") for n in fingers})
        caps.update(lift_joint=(2000.,"linear"),head_yaw_joint=(9.,"angular"),head_pitch_joint=(9.,"angular"))
        for n,(force,typ) in caps.items():
            drive=UsdPhysics.DriveAPI.Apply(joints[n],typ)
            drive.CreateTypeAttr("force");drive.CreateMaxForceAttr(force)
            drive.CreateStiffnessAttr(15000. if n=="lift_joint" else 2200. if n in fingers else 100.)
            drive.CreateDampingAttr(1200. if n=="lift_joint" else 55. if n in fingers else 12.)
        if assist:
            from pxr import UsdShade
            material=UsdShade.Material.Define(st, "/World/GripPadMaterial")
            pa=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
            pa.CreateStaticFrictionAttr(1.35);pa.CreateDynamicFrictionAttr(1.2);pa.CreateRestitutionAttr(0.)
            for side in arms:
                for jaw in ("left","right"):
                    body=bodies[f"{side}_arm_gripper_{jaw}_link"]
                    for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
                        if prim.HasAPI(UsdPhysics.CollisionAPI):
                            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.strongerThanDescendants,"physics")
        for n in s1.DRIVER:
            drive=UsdPhysics.DriveAPI.Apply(joints[n],"angular")
            drive.CreateMaxForceAttr(0.);drive.CreateStiffnessAttr(0.);drive.CreateDampingAttr(0.)
        print("INIT drives authored", flush=True)
        dt=.0008 if task=="D5" else .004
        scene_path="/World/PhysicsScene" if task=="D4" else "/World/Physics"
        world=World(physics_dt=dt,rendering_dt=dt,stage_units_in_meters=1.,
                    physics_prim_path=scene_path,backend="torch" if task=="D4" else "numpy",device="cuda:0" if task=="D4" else "cpu")
        if task=="D4":
            SimulationManager.set_broadphase_type("GPU",physics_scene=scene_path)
            SimulationManager.enable_gpu_dynamics(True,physics_scene=scene_path)
            assert SimulationManager.is_gpu_dynamics_enabled(physics_scene=scene_path)
            reported["gpu_deformable_scene_enabled"] = True
        art=world.scene.add(Articulation(prim_paths_expr=str(art_root.GetPath()),name="ex001",reset_xform_properties=False))
        if task=="D4":
            from physics_io import DeviceArticulation
            art=DeviceArticulation(art)
        eraser_contact = None
        if task == "D2":
            eraser_contact = world.scene.add(RigidPrim(prim_paths_expr="/World/EraserProxy", name="eraser_probe",
                reset_xform_properties=False, track_contact_forces=True, prepare_contact_sensors=True,
                disable_stablization=False, contact_filter_prim_paths_expr=["/World/Board"], max_contact_count=128))
        print("INIT resetting scene", flush=True)
        world.reset();RenderingManager.set_dt(dt)
        assert SimulationManager.get_active_physics_engine()=="physx"
        index=np.array([art.get_dof_index(n) for n in used],dtype=np.int64)
        ix={n:i for i,n in enumerate(used)}
        ai=index[:12]
        kp=np.array([[1400.,1400.,1400.,850.,850.,850.]*2],np.float32)
        kd=np.array([[90.,90.,90.,58.,58.,58.]*2],np.float32)
        art.set_gains(kps=kp,kds=kd,joint_indices=ai)
        art.set_max_efforts(np.array([[caps[n][0] for n in used]]),joint_indices=index)
        art.set_max_joint_velocities(np.array([[4.18]*12+[.08]*4+[.2,.8,.8]]),joint_indices=index)
        q0=np.array([.21268986,1.27888565,-.38970036,.68161984,-.00000104,1.78348334])
        poses={"D1":((.38,.13,.92),(.333,-.233,.85)),"D2":((.44,.13,.92),(.40,-.15,.93)),
               "D3":((.42,.13,.91),(.45,-.18,.84)),"D4":((.41,.12,.89),(.40,-.16,.89)),
               "D5":((.43,.14,.89),(.42,-.11,.85))}
        target=np.zeros(len(used))
        for i,(s,rig) in enumerate(arms.items()):
            target[i*6:i*6+6]=rig.initial(q0,.38,np.asarray(poses[task][i]))
            target[[ix[n] for n in finger[s]]]=[.027,-.027]
        target[-3:]=[.38,-.4 if task=="D1" else -.16,.86]
        pack=lambda:np.array([target],np.float32)
        art.set_joint_positions(pack(),joint_indices=index)
        liftidx=np.array([index[ix["lift_joint"]]])
        grav=float(np.ravel(art.get_generalized_gravity_forces(joint_indices=liftidx))[0])
        liftkp=float(np.ravel(art.get_gains(joint_indices=liftidx)[0])[0])
        target[-3]=.38+grav/liftkp
        art.set_joint_position_targets(pack(),joint_indices=index)
        print("INIT creating sensors", flush=True)
        cap=Capture(st,robot,world,art)
        ext=film_camera(st,task)
        print("INIT sensors initialized", flush=True)
        world.play()
        layer_friction=None
        def step(render=False):
            if layer_friction is not None:layer_friction.step(dt)
            world.step(render=render,update_fabric=True)
            if render and task=="D4":
                import omni.physx
                omni.physx.get_physx_interface().update_transformations(False,True)
        for i in range(round(.48/dt)):
            step(render=i%max(1,round(.04/dt))==0)
        for _ in range(5):step(True)
        def raw_state():
            tick=int(world.current_time_step_index)
            views,rec=cap.capture(tick)
            if task=="D4":
                # Fabric GPU transforms are kept off USD. Use the measured articulation encoders for each wrist.
                q=np.ravel(art.get_joint_positions(joint_indices=index));lv=q[ix["lift_joint"]]
                for i,(side,arm) in enumerate(arms.items()):
                    rec["chassis_T_gripper_base_candidate"][side]=arm.kinematics(q[i*6:i*6+6],lv)[0].tolist()
                rec["chassis_T_lift_native"][2][3]+=(lv-.38)
            return views,rec
        print("INIT capture home", flush=True)
        views,home=raw_state()
        attach_home(home,home)
        begin=int(world.current_time_step_index)
        actualtime=lambda:(int(world.current_time_step_index)-begin)*dt
        reported.update(dt=dt,fixture_sha256=hashlib.sha256(stage_path.read_bytes()).hexdigest(),robot_usd_sha256=audit["usd_entry_sha256"],
                        start_joints=np.ravel(art.get_joint_positions(joint_indices=index)).tolist(),
                        sim_start_tick=begin, native_5756_wrist=True, command_channels=16,
                        scene_asset=str(stage_path), joint_names=used)
        writer=subprocess.Popen(["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-y","-f","rawvideo","-pix_fmt","rgb24",
                                 "-s","1280x540","-r","25","-i","pipe:0","-an","-c:v","libx264","-preset","veryfast","-crf","19",
                                 "-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(out/"trial.mp4")],stdin=subprocess.PIPE)
        font=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",18)
        small=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",15)
        labels={"D1":"Sort the ordered cubes","D2":"Erase the selected column","D3":"Seal the empty bottle","D4":"Fold and store the towel","D5":"Mate the unpowered USB connector"}
        props={n:st.GetPrimAtPath("/World/"+n) for n in names if st.GetPrimAtPath("/World/"+n)}
        node_log=[];a_record_nodes=task=="D4"
        wiped = set()
        pad_in_board_last = None
        def measure():
            nonlocal pad_in_board_last
            ob={}
            for n,p in props.items():
                tr=UsdGeom.XformCache().GetLocalToWorldTransform(p)
                pos=np.asarray(tr.ExtractTranslation(),float); rot=tr.ExtractRotationQuat()
                ob[n]=[pos.tolist(),[rot.GetReal(),*list(rot.GetImaginary())]]
                if task=="D4":
                    v=cloth_view.get_simulation_nodal_positions()
                    if hasattr(v,"detach"): v=v.detach().cpu()
                    pts=np.asarray(v.numpy() if hasattr(v,"numpy") else v,dtype=float).reshape(-1,3)
                    if len(pts)!=225 or not np.isfinite(pts).all():
                        raise RuntimeError("invalid solver node array " + str(pts.shape))
                    ob["cloth_solver_nodes"]=len(pts)
                    ob["cloth_extent_world_m"]=np.ptp(pts,axis=0).tolist()
                    ob["cloth_center_world_m"]=np.mean(pts,axis=0).tolist()
                    ob["cloth_samples_world_m"]=pts[::28].tolist()
                    if os.getenv("FOLD_CASE","").startswith("carry"):
                        ob["material_grid"]=pts.tolist()
                    ob["material_landmarks"]={k:pts[k].tolist() for k in (0,7,14,105,112,119,210,217,224)}
                    ob["cloth_corners_world_m"]=pts[[0,14,210,224]].tolist()
                    ob["cloth_layer_quadrants_m"]=[c.mean(axis=0).tolist() for c in [pts.reshape(15,15,3)[a,b].reshape(-1,3) for a in [slice(0,7),slice(8,15)] for b in [slice(0,7),slice(8,15)]]]
                    if len(out.name)>0 and a_record_nodes:
                        node_log.append(pts.astype(np.float32))
            if eraser_contact is not None:
                board_tf=pose(props["Board"])
                pad_tf=pose(props["EraserProxy"])
                rel=np.linalg.inv(board_tf) @ pad_tf
                force=float(np.linalg.norm(np.asarray(eraser_contact.get_contact_force_matrix(dt=dt))))
                if pad_in_board_last is not None and np.linalg.norm(rel[:2,3]-pad_in_board_last)>.00015 and force>.12:
                    for j in range(14):
                        mark=st.GetPrimAtPath(f"/World/Board/Mark{j:02d}")
                        if j in wiped or not mark:continue
                        local=(np.linalg.inv(pad_tf)@np.r_[np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mark).ExtractTranslation()),1.])[:3]
                        if np.all(np.abs(local)<[.016,.022,.009]):
                            UsdGeom.Imageable(mark).MakeInvisible()
                            wiped.add(j)
                pad_in_board_last=rel[:2,3].copy()
                ob.update(eraser_board_contact_n=force, erased_mark_ids=sorted(wiped))
            return ob
        cloth_view = None
        if task=="D4":
            view = SimulationManager.get_physics_simulation_view()
            cloth_view = view.create_surface_deformable_body_view("/World/D4Coupon/Cloth")
            if not cloth_view or cloth_view.count != 1:
                raise RuntimeError("native surface solver did not register the towel")
        print("INIT reading props", flush=True)
        observed0=measure()
        reported["initial_props"]=observed0
        hold=target.copy()
        from chunk_harness import ChunkHarness,measured_lift_state
        window=ChunkHarness()
        servo=None
        qref=target[:12].copy()
        if assist:
            from task_improved import TaskServo
            from grasp import Grasp
            grips=Grasp(st,bodies,props,task,cloth_view)
            if task=="D4" and os.getenv("FOLD_CASE","").endswith("friction"):
                from layer_friction import LayerFriction
                layer_friction=LayerFriction(cloth_view)
            home_p=np.array(home["chassis_T_gripper_base_candidate"]["left"])[:3,3]
            if task in ("D3","D5"):
                from mating import Assembly
                servo=Assembly(task,observed0,home_p,st,props)
            elif task=="D4":
                from fold import Fold
                from fold_supported import SupportedFold
                from fold_carry import CarryFold
                servo=(CarryFold if os.getenv("FOLD_CASE","").startswith("carry") else SupportedFold if os.getenv("FOLD_CASE") in ("supported","crease") else Fold)(task,observed0,home_p)
            else:
                servo=TaskServo(task,observed0,home_p)
        stage_label=mode
        residual=np.zeros(3)
        def frame(qi):
            w=camera_rgb(ext)
            if w.std()<2: raise RuntimeError("blank external view")
            im=Image.new("RGB",(1280,540),(21,28,36));im.paste(Image.fromarray(w),(0,0))
            draw=ImageDraw.Draw(im)
            for y,key in [(0,"face_view"),(240,"right_wrist_view")]:
                pic,_=cap.views[key].sensor.get_data("rgb")
                arr=np.asarray(pic.numpy() if hasattr(pic,"numpy") else pic)[:,:,:3]
                im.paste(Image.fromarray(arr).resize((320,240)),(906,y))
                draw.rectangle((906,y,1045,y+20),fill=(21,28,36))
                draw.text((913,y+1),"Head RGB" if key=="face_view" else "Right wrist RGB",fill=(229,234,238),font=small)
            draw.text((18,489),task+"   "+labels[task],font=font,fill=(238,244,249))
            draw.text((18,514),"Isaac Sim 6.0.1  |  EX001  |  WALL-OSS-0.5  |  "+("pose-assisted  |  "+stage_label if assist else mode),font=small,fill=(158,184,198))
            draw.text((883,496),f"Physics {actualtime():05.2f} s     Model chunk {qi:03d}",font=font,fill=(219,233,239))
            writer.stdin.write(np.ascontiguousarray(im).tobytes()); reported["frames"]+=1
        print("INIT encoding first frame", flush=True)
        frame(0)
        faulthandler.cancel_dump_traceback_later()
        arm_travel={s:0. for s in arms}
        prev_xyz={s:np.array(home["chassis_T_gripper_base_candidate"][s])[:3,3] for s in arms}
        for qi in range(a.chunks):
            views,rec=raw_state();attach_home(home,rec)
            x,_=state(rec)
            if aligned:
                x=measured_lift_state(x,rec)
            prefix=ipc/f"{a.uid}-{qi:03d}"
            np.savez_compressed(str(prefix)+".npz",state=x,face_view=views["face_view"],right_wrist_view=views["right_wrist_view"])
            path=prefix.with_suffix(".request")
            tmp=prefix.with_suffix(".tmp")
            save(tmp,{"prompt":spec["prompt"],"seed":39311+qi,"physics_tick":rec["physics_tick"]})
            tmp.replace(path)
            deadline=time.monotonic()+150
            while not prefix.with_suffix(".reply").exists():
                if prefix.with_suffix(".error").exists():raise RuntimeError(prefix.with_suffix(".error").read_text()[-1600:])
                if time.monotonic()>deadline:raise TimeoutError("model response")
                time.sleep(.025)
            act=np.load(prefix.with_suffix(".action.npy"),allow_pickle=False)
            assert act.shape==(32,26) and np.isfinite(act).all()
            resp=json.loads(prefix.with_suffix(".reply").read_text())
            if qi==0 or qi==a.chunks-1:
                for extn in [".npz",".action.npy",".reply",".request"]:
                    import shutil
                    shutil.copyfile(str(prefix)+extn,out/f"chunk_{qi:03d}{extn}")
            # Each action row is relative to this chunk's measured pose, never to the previous row.
            anchor={s:np.array(rec["chassis_T_gripper_base_candidate"][s]) for s in arms}
            liftT=np.array(rec["chassis_T_lift_native"]);rl=liftT[:3,:3]
            if aligned:
                window.add(8*qi,act,anchor,liftT)
            reported["model_calls"]+=1
            for ri,row in enumerate(act[:8]):
                measured=np.ravel(art.get_joint_positions(joint_indices=index)).astype(float)
                prev_target=target.copy()
                if mode=="hold":
                    target[:] = hold
                elif aligned:
                    desired,jaw=window.target(8*qi+ri,"right")
                    desired[:3,3]=np.clip(desired[:3,3],[.255,-.49,.781],[.69,.035,1.04])
                    qref[6:12]=arms["right"].step(qref[6:12],measured[ix["lift_joint"]],desired,.04)
                    ff=np.ravel(art.get_generalized_gravity_forces(joint_indices=ai))/kp[0]
                    target[6:12]=qref[6:12]+np.clip(ff[6:12],-.012,.012)
                    half=np.clip(jaw*.00852694,0.,.033)
                    j=[ix[n] for n in finger["right"]]
                    target[j]+=np.clip(np.array([half,-half])-target[j],-.0016,.0016)
                elif assist:
                    wrists={side:arms[side].kinematics(measured[i*6:i*6+6],measured[ix["lift_joint"]])[0] for i,side in enumerate(arms)}
                    goals,stage_label,residual=servo.goals(row,wrists,actualtime(),measure())
                    grips.apply(stage_label,goals,wrists,measured,actualtime())
                    if layer_friction is not None:layer_friction.stage(stage_label,grips.nodes)
                    gravity=np.ravel(art.get_generalized_gravity_forces(joint_indices=ai))/kp[0]
                    for i,(side,rig) in enumerate(arms.items()):
                        sl=slice(i*6,i*6+6)
                        goal,gap=goals[side]
                        qref[sl]=rig.step(qref[sl], measured[ix["lift_joint"]], goal, .04)
                        target[sl]=qref[sl]+np.clip(gravity[sl],-.012,.012)
                        fj=[ix[n] for n in finger[side]]
                        target[fj]+=np.clip(np.array([gap,-gap])-target[fj],-.0024,.0024)
                else:
                    for i,(s,rig) in enumerate(arms.items()):
                        off=i*10;ab=anchor[s]
                        desired=tf(ab[:3,3]+rl@row[off:off+3],rl@action_rotation(row[off+3:off+9])@rl.T@ab[:3,:3])
                        sl=slice(i*6,i*6+6)
                        target[sl]=rig.step(measured[sl],measured[ix["lift_joint"]],desired,.04)
                        fj=[ix[n] for n in finger[s]];half=np.clip(row[off+9]*.00852694,0.,.04875)
                        target[fj]=measured[fj]+np.clip(np.array([half,-half])-measured[fj],-.0032,.0032)
                        target[fj[0]]=np.clip(target[fj[0]],0.,.04875)
                        target[fj[1]]=np.clip(target[fj[1]],-.04875,0.)
                ticks=round(.04/dt)
                if aligned or assist:
                    for ti in range(ticks):
                        alpha=(ti+1)/ticks
                        point=(prev_target+alpha*(target-prev_target)).astype(np.float32)[None]
                        art.set_joint_position_targets(point,joint_indices=index)
                        step(render=ti==ticks-1)
                else:
                    art.set_joint_position_targets(pack(),joint_indices=index)
                    for ti in range(ticks):step(render=ti==ticks-1)
                qafter=np.ravel(art.get_joint_positions(joint_indices=index)).astype(float)
                if not np.isfinite(qafter).all():raise ValueError("nonfinite joint feedback")
                frame(qi+1)
                current={s:pose(cap.links[s+"_arm_gripper_base_link"])[:3,3] for s in arms}
                if task=="D4":
                    current={side:arm.kinematics(qafter[i*6:i*6+6],qafter[ix["lift_joint"]])[0][:3,3] for i,(side,arm) in enumerate(arms.items())}
                for s in arms:arm_travel[s]+=float(np.linalg.norm(current[s]-prev_xyz[s]));prev_xyz[s]=current[s]
                log.write(json.dumps({"tick":int(world.current_time_step_index),"time":actualtime(),"chunk":qi,"row":ri,"controller_mode":mode,"phase":stage_label,"model_residual_m":residual.tolist(),
                         "observation_sha256":resp["observation_sha256"],"chunk_sha256":resp["action_sha256"],"row26":row.tolist(),
                         "measured_before":measured.tolist(),"target":target.tolist(),"measured_after":qafter.tolist(),
                         "wrists":{s:v.tolist() for s,v in current.items()},"props":measure()},allow_nan=False)+"\n")
            print("CONTROL",task,qi+1,"sim_s",round(actualtime(),3),flush=True)
            if assist and servo.ix>=len(servo.phases):
                break
        if assist:
            reported["phase_events"]=servo.events
            reported["grasp_events"]=grips.events
            if layer_friction is not None:
                reported["layer_friction"]={"physics_steps":layer_friction.steps,"contact_pairs":len(layer_friction.pairs) if layer_friction.pairs is not None else 0,"peak_tangential_dv_per_tick":layer_friction.peak_delta,"peak_vector_sum_of_dv":layer_friction.imbalance}
            if task in ("D3","D5"):
                reported["connector_events"]=servo.log;reported["seat_detected"]=servo.seated
                reported["physical_turns"]=-servo.spin/(2*math.pi)
        reported.update(status="horizon_reached", simulated_seconds=actualtime(), final_props=measure(),
                        wrist_travel_m=arm_travel, final_joints=np.ravel(art.get_joint_positions(joint_indices=index)).tolist())
    except Exception:
        reported.update(status="error",error=traceback.format_exc())
        print("TRIAL_ERROR",reported["error"],flush=True)
    finally:
        if writer and writer.stdin:
            writer.stdin.close();reported["encoder_rc"]=writer.wait()
        if task=="D4" and "node_log" in locals() and node_log:
            np.savez_compressed(out/"cloth_nodes.npz", samples=np.array(node_log))
        log.close();save(out/"result.json",reported)
        print("TRIAL_RESULT",task,reported["status"],reported["frames"],flush=True)
        app.close()
    return int(reported["status"]=="error")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--task",choices=list(tasks),required=True)
    p.add_argument("--uid",required=True);p.add_argument("--chunks",type=int,default=50)
    p.add_argument("--mode",choices=("native","hold","assisted","aligned"),default="native")
    p.add_argument("--work-dir", type=Path, default=sim / ".runtime",
                   help="Shared policy queue and recording directory")
    args=p.parse_args()
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.uid):
        p.error("--uid must contain letters, digits, underscores or hyphens")
    if args.chunks < 1:
        p.error("--chunks must be positive")
    raise SystemExit(run(args))
