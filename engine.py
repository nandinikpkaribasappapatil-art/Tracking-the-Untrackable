from __future__ import annotations
import csv, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import cv2
import numpy as np
from ultralytics import YOLO

@dataclass
class Identity:
    gid: int
    descriptor: list
    last_seen: int
    last_box: list
    hits: int = 1

def descriptor(frame, box):
    """Compact appearance signature: HSV histogram + normalized colour statistics.
    This is a lightweight baseline, not a learned person-ReID embedding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, box)
    x1, y1, x2, y2 = max(0,x1), max(0,y1), min(w,x2), min(h,y2)
    if x2-x1 < 8 or y2-y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    # Reduce background contribution by taking the inner crop.
    crop = crop[int(ch*.08):max(int(ch*.08)+1,int(ch*.92)),
                int(cw*.12):max(int(cw*.12)+1,int(cw*.88))]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [24,16], [0,180,0,256])
    cv2.normalize(hist, hist)
    hist = hist.flatten().astype(np.float32)
    # Normalized mean HSV is a coarse complementary cue.
    mean = cv2.mean(hsv)[:3]
    mean = np.asarray([mean[0]/180., mean[1]/255., mean[2]/255.], np.float32)
    vec = np.concatenate([hist, mean])
    n = np.linalg.norm(vec)
    return vec/(n+1e-8)

def appearance_similarity(a, b):
    if a is None or b is None or len(a) != len(b):
        return 0.0
    # Histogram portion uses Bhattacharyya distance.
    ah, bh = np.asarray(a[:384],np.float32), np.asarray(b[:384],np.float32)
    d = cv2.compareHist(ah, bh, cv2.HISTCMP_BHATTACHARYYA)
    hist_sim = max(0., min(1., 1.-float(d)))
    # Mean colour similarity is intentionally weakly weighted.
    am, bm = np.asarray(a[384:]), np.asarray(b[384:])
    mean_sim = max(0., 1.-float(np.linalg.norm(am-bm))/1.2)
    return .82*hist_sim + .18*mean_sim

class GlobalIdentityMemory:
    """Maps short-term tracker IDs to long-term IDs.
    Only dormant identities are candidates; uncertain matches are rejected."""
    def __init__(self, max_age_frames=900, threshold=.76, margin=.07):
        self.max_age = max_age_frames
        self.threshold = threshold
        self.margin = margin
        self.next_gid = 1
        self.local_to_global: Dict[int,int] = {}
        self.identities: Dict[int,Identity] = {}

    def update_or_assign(self, local_id, desc, box, frame_no, active_gids):
        local_id = int(local_id)
        if local_id in self.local_to_global:
            gid = self.local_to_global[local_id]
            item = self.identities.get(gid)
            if item:
                if desc is not None and len(item.descriptor) == len(desc):
                    item.descriptor = (.90*np.asarray(item.descriptor)+.10*desc).tolist()
                item.last_seen = frame_no
                item.last_box = list(map(float,box))
                item.hits += 1
                return gid, 1.0, "TRACKING"
        candidates = []
        for gid, item in self.identities.items():
            gap = frame_no-item.last_seen
            if gid in active_gids or gap <= 0 or gap > self.max_age:
                continue
            sim = appearance_similarity(desc, np.asarray(item.descriptor,np.float32))
            # Long gaps are less trustworthy. Keep penalty modest but nonzero.
            score = sim - min(.10, .10*gap/self.max_age)
            candidates.append((score,sim,gid))
        candidates.sort(reverse=True)
        if candidates:
            best_score,best_sim,best_gid = candidates[0]
            second = candidates[1][0] if len(candidates)>1 else -1.
            if best_sim >= self.threshold and best_score-second >= self.margin:
                self.local_to_global[local_id] = best_gid
                item = self.identities[best_gid]
                if desc is not None and len(item.descriptor)==len(desc):
                    item.descriptor = (.80*np.asarray(item.descriptor)+.20*desc).tolist()
                item.last_seen=frame_no
                item.last_box=list(map(float,box))
                item.hits += 1
                return best_gid,float(best_sim),"RE-IDENTIFIED"
        gid=self.next_gid
        self.next_gid+=1
        self.local_to_global[local_id]=gid
        self.identities[gid]=Identity(gid, (desc.tolist() if desc is not None else []),
                                      frame_no,list(map(float,box)))
        return gid,0.0,"NEW ID"

    def prune(self, frame_no):
        expired=[gid for gid,v in self.identities.items() if frame_no-v.last_seen>self.max_age]
        for gid in expired:
            del self.identities[gid]
        valid=set(self.identities)
        self.local_to_global={lid:gid for lid,gid in self.local_to_global.items() if gid in valid}

def resize_for_cpu(frame, max_width=960):
    h,w=frame.shape[:2]
    if w>max_width:
        nh=int(h*max_width/w)
        return cv2.resize(frame,(max_width,nh),interpolation=cv2.INTER_AREA)
    return frame

def process_video(source, output_video, output_csv, conf=.30, imgsz=640,
                  max_width=960, memory_seconds=30, reid_threshold=.76,
                  frame_stride=1, show=False):
    model=YOLO("yolo11n.pt")
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {source}")
    fps=cap.get(cv2.CAP_PROP_FPS) or 25.
    original_w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if original_w<=0 or original_h<=0:
        cap.release()
        raise RuntimeError("Invalid video dimensions.")
    # Process resized frames and write output at the resized dimensions.
    first_ok, first=cap.read()
    if not first_ok:
        cap.release()
        raise RuntimeError("Video contains no readable frames.")
    first=resize_for_cpu(first,max_width)
    out_h,out_w=first.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    Path(output_video).parent.mkdir(parents=True,exist_ok=True)
    writer=cv2.VideoWriter(output_video,cv2.VideoWriter_fourcc(*"mp4v"),max(1.,fps/frame_stride),(out_w,out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create MP4 output.")
    memory=GlobalIdentityMemory(max_age_frames=max(1,int(fps*memory_seconds)),
                                threshold=reid_threshold)
    rows=[]
    frame_no=0
    written=0
    active_global_ids=set()
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            frame_no+=1
            if frame_stride>1 and (frame_no-1)%frame_stride:
                continue
            frame=resize_for_cpu(frame,max_width)
            result=model.track(frame, persist=True, tracker="botsort_argus.yaml",
                               conf=conf, imgsz=imgsz, classes=[0], device="cpu",
                               verbose=False)[0]
            current=[]
            if result.boxes is not None and len(result.boxes)>0 and result.boxes.id is not None:
                xyxy=result.boxes.xyxy.cpu().numpy()
                lids=result.boxes.id.int().cpu().tolist()
                scores=result.boxes.conf.cpu().numpy()
                for box,lid,score in zip(xyxy,lids,scores):
                    current.append((int(lid),box,float(score)))
            # IDs seen in this frame should not be recycled for another detection.
            current_local={x[0] for x in current}
            active_global_ids=set()
            for lid in current_local:
                if lid in memory.local_to_global:
                    active_global_ids.add(memory.local_to_global[lid])
            for lid,box,score in current:
                desc=descriptor(frame,box)
                gid,sim,status=memory.update_or_assign(lid,desc,box,frame_no,active_global_ids)
                active_global_ids.add(gid)
                x1,y1,x2,y2=map(int,box)
                color=(int((gid*67)%200+30),int((gid*37)%200+30),int((gid*97)%200+30))
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                label=f"ID {gid} | {status}"
                cv2.rectangle(frame,(x1,max(0,y1-24)),(min(out_w,x1+max(125,len(label)*8)),y1),color,-1)
                cv2.putText(frame,label,(x1+4,max(15,y1-7)),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1,cv2.LINE_AA)
                rows.append({
                    "frame":frame_no,"time_sec":round(frame_no/fps,3),
                    "global_id":gid,"tracker_id":lid,
                    "detection_confidence":round(score,4),
                    "appearance_similarity":round(sim,4),"identity_event":status,
                    "x1":round(float(box[0]),2),"y1":round(float(box[1]),2),
                    "x2":round(float(box[2]),2),"y2":round(float(box[3]),2)
                })
            memory.prune(frame_no)
            cv2.putText(frame,f"ARGUS-X | Frame {frame_no} | IDs {len(active_global_ids)}",
                        (12,26),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
            writer.write(frame)
            written+=1
            if show:
                cv2.imshow("ARGUS-X",frame)
                if cv2.waitKey(1)&0xff==ord("q"): break
    finally:
        cap.release()
        writer.release()
        if show: cv2.destroyAllWindows()
    fields=["frame","time_sec","global_id","tracker_id","detection_confidence",
            "appearance_similarity","identity_event","x1","y1","x2","y2"]
    with open(output_csv,"w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    return {"frames_processed":written,"detections":len(rows),
            "unique_global_ids":len(set(r["global_id"] for r in rows)),
            "video":output_video,"csv":output_csv}

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--source",required=True)
    p.add_argument("--conf",type=float,default=.30)
    p.add_argument("--imgsz",type=int,default=640)
    p.add_argument("--memory-seconds",type=int,default=30)
    p.add_argument("--threshold",type=float,default=.76)
    p.add_argument("--stride",type=int,default=1)
    a=p.parse_args()
    print(process_video(a.source,"output/argus_output.mp4","output/tracks.csv",
                        a.conf,a.imgsz,memory_seconds=a.memory_seconds,
                        reid_threshold=a.threshold,frame_stride=a.stride))