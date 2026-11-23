"""Re-render the recorded EX001 joint encoders and surface nodes beside their original Isaac RGB."""
import argparse, gzip, json, subprocess, xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import numpy as np
import vtk
from scipy.spatial.transform import Rotation
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from PIL import Image, ImageDraw, ImageFont

def urdf_transform(origin):
    T=np.eye(4)
    if origin is not None:
        T[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        T[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return T

def phase_zh(s):
    if s in ('find_tag','touch_first','pinch_first_edge'):return '抓取同一侧的两个布角'
    if s.startswith('fold_x') or s in ('release_first_fold','clear_first'):return '第一折'
    if s in ('above_second','touch_second','pinch_second_edge'):return '换边夹取双层布'
    if s.startswith('fold_y') or s in ('release_second_fold','clear_second'):return '第二折'
    if s in ('survey_stack','touch_layers','grip_folded_stack','test_lift'):return '两侧夹持折件'
    if s.startswith('carry_'):return '双臂搬运'
    if s=='lower_stack':return '放入收纳区'
    return '松手并观察折件'

def main():
    arg=argparse.ArgumentParser();arg.add_argument('--run',type=Path,required=True);arg.add_argument('--model',type=Path,required=True);arg.add_argument('--out',type=Path,required=True);arg.add_argument('--samples',type=Path)
    a=arg.parse_args()
    result=json.loads((a.run/'result.json').read_text())
    plain = a.run / "trace.jsonl"
    with (plain.open() if plain.is_file() else gzip.open(a.run / "trace.jsonl.gz", "rt")) as trace:
        rows = [json.loads(line) for line in trace]
    urdf=ET.parse(a.model/'ex001_d1_collision_proxy.urdf').getroot()
    actor_map={};joints=[];key_map={k:i for i,k in enumerate(result['joint_names'])}
    ren=vtk.vtkRenderer();ren.SetBackground(.91,.94,.95);ren.SetBackground2(.80,.85,.87);ren.GradientBackgroundOn()
    window=vtk.vtkRenderWindow();window.SetOffScreenRendering(1);window.SetSize(880,480);window.SetMultiSamples(0);window.AddRenderer(ren)
    def actor(data,color):
        mp=vtk.vtkPolyDataMapper(); mp.SetInputData(data);mp.ScalarVisibilityOff()
        obj=vtk.vtkActor();obj.SetMapper(mp);o=obj.GetProperty();o.SetColor(*color);o.SetAmbient(.30);o.SetDiffuse(.68);o.SetSpecular(.14);o.SetSpecularPower(22);ren.AddActor(obj);return obj,mp
    def box(center,size,color):
        cube=vtk.vtkCubeSource();cube.SetCenter(*center);cube.SetXLength(size[0]);cube.SetYLength(size[1]);cube.SetZLength(size[2]);cube.Update();return actor(cube.GetOutput(),color)[0]
    box((.47,0,.633),(.66,.65,.054),(.70,.70,.68))
    for x in (.37,.60):
        for y in (-.27,.21):box((x,y,.2665),(.022,.022,.673),(.33,.36,.38))
    zx,zy=result['storage_center_xy_m']
    for sign in (-1,1):
        box((zx+sign*.065,zy,.66015),(.003,.133,.0002),(.28,.76,.57))
        box((zx,zy+sign*.065,.66015),(.133,.003,.0002),(.28,.76,.57))
    box((.1,0,-.125),(4,4,.05),(.72,.76,.77))
    for link in urdf.findall('link'):
        name=link.get('name'); group=[]
        for vis in link.findall('visual'):
            mesh=vis.find('geometry/mesh')
            if mesh is None:continue
            rd=vtk.vtkSTLReader();rd.SetFileName(str(a.model/mesh.get('filename')));rd.Update()
            dec=vtk.vtkQuadricDecimation();dec.SetInputConnection(rd.GetOutputPort());dec.SetTargetReduction(.72);dec.Update()
            norm=vtk.vtkPolyDataNormals();norm.SetInputConnection(dec.GetOutputPort());norm.SetFeatureAngle(48);norm.Update()
            attr=vis.find('material/color');color=np.array([.74,.76,.76]) if attr is None else np.fromstring(attr.get('rgba'),sep=' ')[:3]
            obj,_=actor(norm.GetOutput(),color);mat=vtk.vtkMatrix4x4();obj.SetUserMatrix(mat)
            group.append((obj,mat,urdf_transform(vis.find('origin'))))
        if group:actor_map[name]=group
    # URDF XML order need not be topological.
    pending=list(urdf.findall('joint'));seen={'base_link'}
    while pending:
        for j in list(pending):
            parent=j.find('parent').get('link')
            if parent not in seen:continue
            child=j.find('child').get('link');axis=j.find('axis')
            axis=np.fromstring(axis.get('xyz'),sep=' ') if axis is not None else np.zeros(3)
            joints.append((parent,child,j.get('type'),urdf_transform(j.find('origin')),axis,key_map.get(j.get('name'))))
            seen.add(child);pending.remove(j)
    cloth=vtk.vtkPolyData(); points=vtk.vtkPoints()
    pos=np.array(result['initial_props']['material_grid'],dtype=np.float32);p_vtk=numpy_to_vtk(pos,deep=False);points.SetData(p_vtk);cloth.SetPoints(points)
    polys=vtk.vtkCellArray()
    for i in range(14):
        for j in range(14):
            k=i*15+j
            for tri in ((k,k+15,k+1),(k+15,k+16,k+1)):
                polys.InsertNextCell(3)
                for v in tri:polys.InsertCellPoint(v)
    cloth.SetPolys(polys)
    colors=np.tile([45,137,221],(225,1)).astype(np.uint8)
    colors[(np.arange(225)//15>=13)&(np.arange(225)%15<2)]=[254,136,49]
    cloth.GetPointData().SetScalars(numpy_to_vtk(colors,deep=True,array_type=vtk.VTK_UNSIGNED_CHAR))
    ca,cm=actor(cloth,(.2,.55,.85));cm.SetColorModeToDirectScalars();cm.ScalarVisibilityOn()
    ca.GetProperty().SetInterpolationToFlat();ca.GetProperty().SetAmbient(.42);ca.GetProperty().SetSpecular(.04)
    cam=ren.GetActiveCamera();cam.SetPosition(1.60,.85,1.55);cam.SetFocalPoint(.31,.005,.88);cam.SetViewUp(0,0,1);cam.SetViewAngle(32)
    for xyz,strength in [((1.2,-1.8,2.4),.90),((1.6,1.1,1.75),.68),((-.8,.7,2.1),.40)]:
        light=vtk.vtkLight();light.SetLightTypeToSceneLight();light.SetPosition(*xyz);light.SetFocalPoint(.35,0,.75);light.SetIntensity(strength);ren.AddLight(light)
    ren.ResetCameraClippingRange()
    cap=cv2.VideoCapture(str(a.run/'trial.mp4'))
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font=ImageFont.truetype(font_path,21);small=ImageFont.truetype(font_path,15)
    labels={}
    def draw(idx):
        row=rows[max(0,idx-1)];q=row['measured_after'] if idx else result['start_joints']
        transforms={'base_link':np.eye(4)}
        for parent,child,typ,M,axis,index in joints:
            T=transforms[parent]@M
            if index is not None:
                T=T.copy()
                if typ in ('revolute','continuous'):T[:3,:3]=T[:3,:3]@Rotation.from_rotvec(axis*q[index]).as_matrix()
                if typ=='prismatic':T[:3,3]+=T[:3,:3]@axis*q[index]
            transforms[child]=T
        for name,group in actor_map.items():
            for obj,mat,offset in group:mat.DeepCopy((transforms[name]@offset).ravel())
        if idx:
            for s in ('left','right'):
                distance=np.linalg.norm(transforms[s+'_arm_gripper_base_link'][:3,3]-row['wrists'][s])
                assert distance<.0015,(idx,s,distance)
        pos[:]=row['props']['material_grid'] if idx else result['initial_props']['material_grid'];p_vtk.Modified();points.Modified();cloth.Modified()
        window.Render()
        rgba=vtk.vtkUnsignedCharArray();window.GetRGBACharPixelData(0,0,879,479,0,rgba)
        view=vtk_to_numpy(rgba).reshape(480,880,4)[::-1,:,:3]
        if a.samples:cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
        ok,native=cap.read();assert ok,idx
        image=cv2.cvtColor(native,cv2.COLOR_BGR2RGB);image[:480,:880]=view
        for side,letter,color in [('left','L',(30,126,116)),('right','R',(136,80,42))]:
            arm=transforms.get(side+'_arm_link3')
            if arm is None:continue
            ren.SetWorldPoint(*arm[:3,3],1.);ren.WorldToDisplay();x,y,_=ren.GetDisplayPoint()
            tip=(int(x),int(479-y));offset=(-34,-16) if side=='right' else (34,-16)
            ctr=(tip[0]+offset[0],tip[1]+offset[1]);cv2.line(image,tip,ctr,color,2,cv2.LINE_AA)
            cv2.circle(image,ctr,11,color,-1,cv2.LINE_AA)
            cv2.putText(image,letter,(ctr[0]-5,ctr[1]+5),cv2.FONT_HERSHEY_SIMPLEX,.47,(255,255,255),1,cv2.LINE_AA)
        action=phase_zh(row['phase'])
        if action not in labels:
            footer=Image.new('RGB',(850,60),(21,28,36));w=ImageDraw.Draw(footer)
            w.text((15,1),'D4  方巾双折收纳   ·   '+action,font=font,fill=(242,246,248))
            w.text((15,33),'左侧为实测关节和布面节点的同步回放，右侧为 Isaac Sim 原始机头和右腕画面',font=small,fill=(172,192,202))
            labels[action]=np.asarray(footer)
        image[480:,:850]=labels[action]
        return image
    if a.samples:
        a.samples.mkdir(exist_ok=True,parents=True)
        for i in (0,90,325,605,800,1264):
            im=draw(i);Image.fromarray(im).resize((640,270)).save(a.samples/f'view-{i}.jpg')
        return
    tmp=a.out.with_suffix('.part.mp4')
    enc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x540','-r','25','-i','-','-an','-c:v','libx264','-preset','veryfast','-threads','5','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],stdin=subprocess.PIPE)
    for i in range(len(rows)+1):
        enc.stdin.write(draw(i).tobytes())
        if i%250==0: print('Rendered',i,flush=True)
    enc.stdin.close();assert enc.wait()==0
    tmp.replace(a.out)
if __name__=='__main__':main()
