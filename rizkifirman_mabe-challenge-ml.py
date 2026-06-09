import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from pathlib import Path
from tqdm import tqdm
from IPython.display import Video, display

dataset_path = Path('/kaggle/input/MABe-mouse-behavior-detection')
train = pd.read_csv(dataset_path / 'train.csv')


# Load train metadata
train_csv_path = "/kaggle/input/MABe-mouse-behavior-detection/train.csv"
train_df = pd.read_csv(train_csv_path)

# Filter out MABe22 if needed (as per baseline)
train_df = train_df[~train_df['lab_id'].str.startswith('MABe22')].reset_index(drop=True)

print(f"Loaded {len(train_df)} training episodes.")
train_df.head(10)


import os, json, math, tempfile, subprocess, base64
from functools import lru_cache
from typing import Optional, List, Tuple, Dict
import numpy as np, pandas as pd, cv2
from IPython.display import Video
from tqdm.auto import tqdm

def distinct_colors_bgr(n, sat=200, val=235, hue_offset=0):
    if n <= 0: return []
    hues = (np.linspace(0,179,n,endpoint=False)+hue_offset)%180
    hsv  = np.stack([hues, np.full(n,sat), np.full(n,val)],1).astype(np.uint8)[None,...]
    return [tuple(map(int,c)) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0]]

def _extract_bps(meta: Optional[pd.DataFrame]):
    if meta is None or meta.empty: return None
    row = meta.iloc[0]
    for col in ["body_parts_tracked","bodyparts_tracked","body_parts","tracked_body_parts"]:
        if col in meta.columns:
            v = row[col]
            if isinstance(v,list): return v
            if isinstance(v,str):
                try:
                    p = json.loads(v)
                    if isinstance(p,list): return p
                except: pass
    return None

def _edges(available: List[str]):
    cand=[("nose","neck"),("neck","ear_left"),("neck","ear_right"),
          ("neck","hip_left"),("neck","hip_right"),
          ("hip_left","tail_base"),("hip_right","tail_base"),
          ("body_center","neck"),("body_center","nose"),("body_center","tail_base")]
    seen=set(); out=[]
    for a,b in cand:
        if a in available and b in available:
            k=tuple(sorted((a,b)))
            if k not in seen: seen.add(k); out.append((a,b))
    return out

def export_square_nocrop_mp4_embed(
    df_episode: pd.DataFrame,
    df_episode_meta: Optional[pd.DataFrame]=None,
    df_annotation: Optional[pd.DataFrame]=None,   # <— NEW
    *,
    size_px:int=640, out_fps:Optional[int]=None,
    frame_start:Optional[int]=None, frame_end:Optional[int]=None, frame_stride:int=1,
    show_skeleton:bool=True, show_trails:bool=False, trail_len:int=25,
    bodyparts:Optional[List[str]]=None, marker_radius:int=3, border_thickness:int=2, line_thickness:int=2,
    # action caption options (NEW)
    show_actions: bool=True,
    action_anchor_priority: Tuple[str,...]=("nose","neck","body_center"),  # where to place label
    action_font_scale: float=0.5,
    action_thickness: int=1,
    action_text_color: Tuple[int,int,int]=(0,0,0),        # BGR
    action_box_bg: Tuple[int,int,int]=(255,255,255),      # BGR
    action_box_pad: int=3,
    action_box_border: int=1,
    action_box_max_width_ratio: float=0.9,                # clamp box within canvas
    action_joiner: str=" | ",                             # if multiple concurrent actions
    # output / encoding
    out_path:str="episode_square.mp4",
    ffmpeg_preset:str="veryfast", ffmpeg_crf:int=23,
    return_embed:bool=True, max_embed_mb:int=120, display_width:int=720
):
    # basic validation
    need={"video_frame","mouse_id","bodypart","x","y"}
    miss=need-set(df_episode.columns)
    if miss: raise ValueError(f"df_episode missing: {sorted(miss)}")
    df=df_episode.sort_values(["video_frame","mouse_id","bodypart"]).reset_index(drop=True)

    # source plane (no crop)
    if df_episode_meta is not None and not df_episode_meta.empty:
        r=df_episode_meta.iloc[0]
        W=r.get("video_width_pix",np.nan); H=r.get("video_height_pix",np.nan)
        if np.isfinite(W) and np.isfinite(H):
            xmin=ymin=0.0; src_w=float(W); src_h=float(H)
        else:
            xmin, xmax = float(df.x.min()), float(df.x.max())
            ymin, ymax = float(df.y.min()), float(df.y.max())
            src_w=max(1.0,xmax-xmin); src_h=max(1.0,ymax-ymin)
    else:
        xmin, xmax = float(df.x.min()), float(df.x.max())
        ymin, ymax = float(df.y.min()), float(df.y.max())
        src_w=max(1.0,xmax-xmin); src_h=max(1.0,ymax-ymin); xmin=ymin=0.0

    N=int(size_px); s=min(N/src_w, N/src_h); pad_x=0.5*(N-s*src_w); pad_y=0.5*(N-s*src_h)
    def map_xy(x,y):
        X=pad_x + s*(float(x)-xmin); Y=pad_y + s*(float(y)-ymin)
        return max(0,min(N-1,int(round(X)))), max(0,min(N-1,int(round(Y))))

    # fps
    if out_fps is None:
        v = df_episode_meta.iloc[0]["frames_per_second"] if (df_episode_meta is not None and not df_episode_meta.empty and "frames_per_second" in df_episode_meta.columns) else None
        out_fps = int(v) if (v is not None and pd.notna(v)) else 12

    # frames
    all_frames=np.sort(df.video_frame.dropna().unique())
    if all_frames.size==0: raise ValueError("No frames.")
    if frame_start is None: frame_start=int(all_frames[0])
    if frame_end   is None: frame_end  =int(all_frames[-1])
    frames=[int(f) for f in all_frames if frame_start<=f<=frame_end][::max(1,int(frame_stride))]
    if not frames: raise ValueError("No frames after filtering.")
    frames_np = np.array(frames, dtype=int)

    # bodyparts/colors
    data_bps=sorted(df.bodypart.dropna().unique().tolist())
    meta_bps=_extract_bps(df_episode_meta)
    available_bps=meta_bps if meta_bps else data_bps
    if bodyparts is None: bodyparts=available_bps
    else: bodyparts=[bp for bp in bodyparts if bp in available_bps]
    edges=_edges(bodyparts) if show_skeleton else []
    bp_cols=distinct_colors_bgr(len(available_bps), sat=200, val=235, hue_offset=0)
    bp_to_bgr={bp: bp_cols[i] for i,bp in enumerate(available_bps)}
    mice=sorted(df.mouse_id.dropna().unique().tolist())
    mouse_cols=distinct_colors_bgr(len(mice), sat=240, val=220, hue_offset=17)
    mouse_to_bgr={mid: mouse_cols[i] for i,mid in enumerate(mice)}

    # --- Build fast lookup for annotations: (frame_idx, agent_id) -> [actions]
    ann_map: Dict[Tuple[int,int], List[str]] = {}
    
    if show_actions and df_annotation is not None and not df_annotation.empty:
        req_cols = {"agent_id","action","start_frame","stop_frame"}
        miss = req_cols - set(df_annotation.columns)
        if miss:
            raise ValueError(f"df_annotation missing: {sorted(miss)}")
        # keep only agents in this episode
        ann = df_annotation.copy()
        ann = ann[ann["agent_id"].isin(mice)]
        # normalize numeric
        for c in ("start_frame","stop_frame"):
            ann[c] = pd.to_numeric(ann[c], errors="coerce").astype("Int64")
        ann = ann.dropna(subset=["start_frame","stop_frame","action","agent_id"])
        if not ann.empty:
            ann["start_frame"] = ann["start_frame"].astype(int)
            ann["stop_frame"]  = ann["stop_frame"].astype(int)
            if ann["start_frame"].gt(ann["stop_frame"]).any():
                # swap where needed
                bad = ann["start_frame"] > ann["stop_frame"]
                tmp = ann.loc[bad, "start_frame"].values
                ann.loc[bad, "start_frame"] = ann.loc[bad, "stop_frame"].values
                ann.loc[bad, "stop_frame"]  = tmp
            # For each annotation row, add entries for intersecting frames (use binary search into frames)
            for row in ann.itertuples(index=False):
                a0, a1 = int(row.start_frame), int(row.stop_frame)
                # overlap with rendered frames
                i0 = int(np.searchsorted(frames_np, a0, side="left"))
                i1 = int(np.searchsorted(frames_np, a1, side="right"))
                if i0 >= i1: 
                    continue
                agent = int(row.agent_id) if pd.api.types.is_numeric_dtype(type(row.agent_id)) or isinstance(row.agent_id, (int,np.integer)) else row.agent_id
                act   = str(row.action)
                for f in frames_np[i0:i1]:
                    ann_map.setdefault((int(f), agent), []).append(act)

    @lru_cache(None)
    def fm(frame, mid):
        sub=df[(df.video_frame==frame)&(df.mouse_id==mid)]
        return sub[sub.bodypart.isin(bodyparts)]

    trails={mid:[] for mid in mice}

    # write tmp with OpenCV, then ffmpeg → H.264
    tmp_output_path = "tmp_" + out_path
    vw=cv2.VideoWriter(tmp_output_path, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (N,N))
    if not vw.isOpened(): raise RuntimeError("OpenCV VideoWriter failed.")

    try:
        for f in tqdm(frames, total=len(frames), desc="Rendering frames", unit="f",
              smoothing=0.1, mininterval=0.1, leave=False):
            canvas=np.full((N,N,3),(255,255,255),dtype=np.uint8)  # white background
            for mid in mice:
                sub=fm(f,mid)
                if sub.empty: 
                    # even if no body part this frame, we might still want to write the action near last trail point (optional).
                    continue
                bp_map={}
                for r in sub.itertuples(index=False):
                    x,y=map_xy(r.x,r.y); bp_map[r.bodypart]=(x,y)
                # skeleton
                if show_skeleton:
                    for p1,p2 in edges:
                        if p1 in bp_map and p2 in bp_map:
                            x1,y1=bp_map[p1]; x2,y2=bp_map[p2]
                            cv2.line(canvas,(x1,y1),(x2,y2), bp_to_bgr.get(p1,(0,0,0)), thickness=line_thickness, lineType=cv2.LINE_AA)
                # joints
                for r in sub.itertuples(index=False):
                    x,y=bp_map[r.bodypart]
                    cv2.circle(canvas,(x,y),marker_radius, bp_to_bgr.get(r.bodypart,(0,0,0)), -1, cv2.LINE_AA)
                    cv2.circle(canvas,(x,y),marker_radius, mouse_to_bgr.get(mid,(0,0,0)), border_thickness, cv2.LINE_AA)

                # trails
                ref = None
                for k in action_anchor_priority:
                    if k in bp_map: 
                        ref = bp_map[k]; break
                if ref is None and bp_map:
                    xs,ys=zip(*bp_map.values()); ref=(int(round(np.mean(xs))), int(round(np.mean(ys))))

                if show_trails and ref is not None:
                    t=trails[mid]; t.append(ref); 
                    if len(t)>trail_len: trails[mid]=t[-trail_len:]
                    pts=np.array(trails[mid],np.int32).reshape(-1,1,2)
                    cv2.polylines(canvas,[pts],False, mouse_to_bgr[mid], 1, cv2.LINE_AA)

                # action captions
                if show_actions and ref is not None:
                    acts = ann_map.get((int(f), mid))
                    if acts:
                        # make unique but stable
                        uniq = list(dict.fromkeys(a.strip() for a in acts if str(a).strip()))
                        if uniq:
                            label = action_joiner.join(uniq)
                            # measure text
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            (tw, th), baseline = cv2.getTextSize(label, font, action_font_scale, action_thickness)
                            pad = int(action_box_pad)
                            box_w = min(tw + 2*pad, int(action_box_max_width_ratio * N))
                            # position the box above the anchor, clamp within canvas
                            x0 = int(ref[0] - box_w//2)
                            y0 = int(ref[1] - marker_radius - 6 - th - baseline - 2*pad)
                            x0 = max(0, min(N - box_w, x0))
                            y0 = max(0, y0)
                            # background box
                            x1 = x0 + box_w
                            y1 = y0 + th + baseline + 2*pad
                            cv2.rectangle(canvas, (x0,y0), (x1,y1), action_box_bg, thickness=-1)
                            if action_box_border > 0:
                                cv2.rectangle(canvas, (x0,y0), (x1,y1), mouse_to_bgr.get(mid,(0,0,0)), thickness=action_box_border, lineType=cv2.LINE_AA)
                            # text (left padded)
                            tx = x0 + pad
                            ty = y1 - baseline - pad
                            cv2.putText(canvas, label, (tx,ty), font, action_font_scale, action_text_color, action_thickness, cv2.LINE_AA)

            # Draw frame number (top-left corner)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Frame {f}"
            scale = 0.5
            thickness = 1
            color = (0, 0, 0)   # black
            margin = 50
            
            (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
            x = margin
            y = margin
            
            cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

            vw.write(canvas)
    finally:
        vw.release()

    # Re-encode with libx264 baseline/yuv420p-ish quality settings
    subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_output_path, "-crf", str(ffmpeg_crf), "-preset", ffmpeg_preset, "-vcodec", "libx264", "-pix_fmt", "yuv420p", out_path],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    try:
        os.remove(tmp_output_path)
    except OSError:
        pass

    return out_path


from IPython.display import Video, display

def create_video(video_idx):
    if video_idx is None or not (0 <= int(video_idx) < len(train)):
        raise IndexError(f"video_idx must be in [0, {len(train)-1}]")
        
    # Example: iterate over *all* episodes described in the meta dataframe
    row = train.iloc[int(video_idx)]
            
    lab = str(row["lab_id"])
    vid = str(int(row["video_id"]))  # cast to int then str for clean filename

    # Build the parquet file path
    pathto_tracking_data = os.path.join(
        dataset_path,
        "train_tracking",
        lab,
        f"{vid}.parquet"
    )

    print(f"Episode Tracking {video_idx} → {pathto_tracking_data}")

    # Load the tracking dataframe
    df_tracking = pd.read_parquet(pathto_tracking_data)

    # Build the parquet file path
    pathto_annotation_data = os.path.join(
        dataset_path,
        "train_annotation",
        lab,
        f"{vid}.parquet"
    )

    print(f"Episode Annotation {video_idx} → {pathto_annotation_data}")

    # Load the tracking dataframe
    df_annotation = pd.read_parquet(pathto_annotation_data)    

    vid = export_square_nocrop_mp4_embed(
        df_tracking, train.iloc[[video_idx]], df_annotation,
        size_px=640, frame_stride=2, show_trails=False,
        out_path=f"episode-{video_idx}.mp4",  # you also keep a file
        return_embed=True, max_embed_mb=120, display_width=720
    )

    display(Video(data=vid,
              embed=True,
              height=int(640),
              width=int(640))
       )

    return df_annotation # For debugging purposes


index = 1
annot = create_video(index)


index = 0
annot = create_video(index)

