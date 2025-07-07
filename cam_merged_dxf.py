#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rhino & Illustrator 両対応 DXF → パス連結スクリプト
  - Rhino ネイティブ曲線（ARC/CIRCLE/ELLIPSE/SPLINE）は各 discretize_* で分割
  - Illustrator の LWPOLYLINE の bulge も正確に分割
  - 全線分を connect_paths で連結
  - 閉じたパスは close=True、開いたパスは close=False で出力
  - POINT はそのまま出力
Usage:
  %run cam_merged_dxf.py input.dxf output_merged.dxf
"""

import sys, math
import ezdxf
import numpy as np
from ezdxf.math import bulge_to_arc
from shapely.geometry import LineString, MultiLineString
from shapely.ops import unary_union, linemerge

# 分割解像度
SEGMENTS = 120
DRILL_RADIUS_THRESHOLD = 2

def discretize_circle(center, radius, segments=SEGMENTS):
    angles = np.linspace(0, 2*math.pi, segments, endpoint=False)
    pts = [(center[0] + radius*math.cos(a), center[1] + radius*math.sin(a))
           for a in angles]
    pts.append(pts[0])
    return pts

def discretize_arc(center, radius, start_angle, end_angle, segments=SEGMENTS):
    if end_angle < start_angle:
        end_angle += 360.0
    angles = np.linspace(math.radians(start_angle), math.radians(end_angle), segments)
    return [(center[0] + radius*math.cos(a), center[1] + radius*math.sin(a))
            for a in angles]

def discretize_spline(entity, segments=SEGMENTS):
    # virtual_entities() があればまず試す
    if hasattr(entity, 'virtual_entities'):
        pts = []
        for v in entity.virtual_entities():
            if v.dxftype() == 'LINE':
                pts.append((v.dxf.start.x, v.dxf.start.y))
                pts.append((v.dxf.end.x,   v.dxf.end.y))
        if pts:
            return pts
    # control_points を試す
    try:
        ctrl = entity.control_points  # numpy array
        return [(pt[0], pt[1]) for pt in ctrl]
    except Exception:
        print("⚠️ Spline 分割失敗:", entity)
        return []

def discretize_ellipse(entity, segments=SEGMENTS):
    try:
        cx, cy = entity.dxf.center.x, entity.dxf.center.y
        major = entity.dxf.major_axis
        ratio = entity.dxf.ratio
        a = math.hypot(major.x, major.y)
        b = a * ratio
        phi = math.atan2(major.y, major.x)
        angles = np.linspace(0, 2*math.pi, segments, endpoint=False)
        pts = []
        for t in angles:
            x = a*math.cos(t); y = b*math.sin(t)
            xr = x*math.cos(phi) - y*math.sin(phi)
            yr = x*math.sin(phi) + y*math.cos(phi)
            pts.append((cx + xr, cy + yr))
        pts.append(pts[0])
        return pts
    except Exception:
        print("⚠️ Ellipse 分割失敗:", entity)
        return []

def to_lines_from_lwpoly(e, lines, segments=SEGMENTS):
    pts = e.get_points('xyb')  # (x, y, bulge)
    n = len(pts)
    # 各セグメント
    for i in range(n-1):
        x0,y0,b0 = pts[i]
        x1,y1,_  = pts[i+1]
        if abs(b0) < 1e-6:
            lines.append(LineString([(x0,y0),(x1,y1)]))
        else:
            ctr, r, sa, ea = bulge_to_arc((x0,y0),(x1,y1),b0)
            angs = np.linspace(sa, ea, segments+1)
            arc = [(ctr[0]+r*math.cos(a), ctr[1]+r*math.sin(a)) for a in angs]
            for j in range(len(arc)-1):
                lines.append(LineString([arc[j], arc[j+1]]))
    # 閉じ要素
    if getattr(e, 'closed', False):
        x0,y0,b0 = pts[-1]
        x1,y1,_  = pts[0]
        if abs(b0) < 1e-6:
            lines.append(LineString([(x0,y0),(x1,y1)]))
        else:
            ctr, r, sa, ea = bulge_to_arc((x0,y0),(x1,y1),b0)
            angs = np.linspace(sa, ea, segments+1)
            arc = [(ctr[0]+r*math.cos(a), ctr[1]+r*math.sin(a)) for a in angs]
            for j in range(len(arc)-1):
                lines.append(LineString([arc[j], arc[j+1]]))

def load_all_paths_from_dxf(fn):
    doc = ezdxf.readfile(fn)
    msp = doc.modelspace()
    lines, points = [], []
    for e in msp:
        t = e.dxftype()
        if t == 'LINE':
            x1,y1 = e.dxf.start.x, e.dxf.start.y
            x2,y2 = e.dxf.end.x,   e.dxf.end.y
            lines.append(LineString([(x1,y1),(x2,y2)]))
        elif t == 'LWPOLYLINE':
            to_lines_from_lwpoly(e, lines, SEGMENTS)
        elif t == 'POLYLINE':
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in e.vertices]
            for i in range(len(pts)-1):
                lines.append(LineString([pts[i], pts[i+1]]))
            if e.is_closed:
                lines.append(LineString([pts[-1], pts[0]]))
        elif t == 'CIRCLE':
            cx, cy = e.dxf.center.x, e.dxf.center.y
            r = float(e.dxf.radius)
            if r <= DRILL_RADIUS_THRESHOLD:
                # ビアとして登録
                points.append((cx, cy))
            else:
                # 通常の円分割
                arc = discretize_circle((cx, cy), r, SEGMENTS)
                for i in range(len(arc) - 1):
                    lines.append(LineString([arc[i], arc[i+1]]))
        elif t == 'ARC':
            arc = discretize_arc((e.dxf.center.x,e.dxf.center.y),
                                  float(e.dxf.radius),
                                  e.dxf.start_angle, e.dxf.end_angle,
                                  SEGMENTS)
            for i in range(len(arc)-1):
                lines.append(LineString([arc[i],arc[i+1]]))
        elif t == 'ELLIPSE':
            arc = discretize_ellipse(e, SEGMENTS)
            for i in range(len(arc)-1):
                lines.append(LineString([arc[i],arc[i+1]]))
        elif t == 'SPLINE':
            pts_s = discretize_spline(e, SEGMENTS)
            for i in range(len(pts_s)-1):
                lines.append(LineString([pts_s[i],pts_s[i+1]]))
        elif t == 'POINT':
            points.append((e.dxf.location.x, e.dxf.location.y))
        else:
            if hasattr(e, 'virtual_entities'):
                for v in e.virtual_entities():
                    if v.dxftype() == 'LINE':
                        lines.append(LineString([
                            (v.dxf.start.x, v.dxf.start.y),
                            (v.dxf.end.x,   v.dxf.end.y)
                        ]))
    return lines, points


def detect_via_lines(lines, length_threshold=0.1):
    """
    端点を近傍クラスタリングし、真に孤立した短線分だけピアス認定。
    """
    import numpy as np
    from collections import Counter
    from scipy.spatial import cKDTree

    # すべての端点を集める
    all_points = []
    for ln in lines:
        all_points.append(ln.coords[0])
        all_points.append(ln.coords[-1])
    all_points = np.array(all_points)
    # 端点クラスタリング
    tree = cKDTree(all_points)
    groups = tree.query_ball_tree(tree, r=1e-2)  # 0.01mm以内
    label_map = {}
    label = 0
    for i, group in enumerate(groups):
        if i in label_map: continue
        for j in group:
            label_map[j] = label
        label += 1
    # クラスタ番号で端点を代表点化
    cluster_id_to_coord = {}
    for idx, pt in enumerate(all_points):
        cluster_id = label_map[idx]
        if cluster_id not in cluster_id_to_coord:
            cluster_id_to_coord[cluster_id] = pt

    # 各線分のクラスタID
    def get_cluster_id(pt):
        idx = np.where((all_points == pt).all(axis=1))[0][0]
        return label_map[idx]
    cluster_end_counts = Counter()
    for ln in lines:
        st_cid = get_cluster_id(np.array(ln.coords[0]))
        ed_cid = get_cluster_id(np.array(ln.coords[-1]))
        cluster_end_counts[st_cid] += 1
        cluster_end_counts[ed_cid] += 1

    via_pts, rem = [], []
    for ln in lines:
        if ln.length <= length_threshold:
            st_cid = get_cluster_id(np.array(ln.coords[0]))
            ed_cid = get_cluster_id(np.array(ln.coords[-1]))
            # 両端点がそれぞれ1回ずつしか使われていなければ孤立（ピアス認定）
            if cluster_end_counts[st_cid] == 1 and cluster_end_counts[ed_cid] == 1:
                pt_st = cluster_id_to_coord[st_cid]
                pt_ed = cluster_id_to_coord[ed_cid]
                via_pts.append(((pt_st[0]+pt_ed[0])/2.0, (pt_st[1]+pt_ed[1])/2.0))
            else:
                rem.append(ln)
        else:
            rem.append(ln)
    return via_pts, rem


def snap_endpoints(lines, snap_threshold=0.2):
    """全端点をクラスタリングして吸着し、ズレ・隙間を除去する。"""
    from scipy.spatial import cKDTree
    if not lines:
        return lines
    # 端点収集
    endpoints = []
    for ln in lines:
        endpoints.append(ln.coords[0])
        endpoints.append(ln.coords[-1])
    endpoints = np.array(endpoints)
    # クラスタリング
    tree = cKDTree(endpoints)
    groups = {}
    for i, pt in enumerate(endpoints):
        idxs = tree.query_ball_point(pt, snap_threshold)
        rep = min(idxs)
        groups.setdefault(rep, []).append(i)
    # 各グループを代表点に吸着
    new_points = endpoints.copy()
    for rep, idxs in groups.items():
        mean_pt = np.mean(endpoints[idxs], axis=0)
        for i in idxs:
            new_points[i] = mean_pt
    # 再構成
    new_lines = []
    for i, ln in enumerate(lines):
        st = tuple(new_points[2*i])
        ed = tuple(new_points[2*i+1])
        new_lines.append(LineString([st, ed]))
    return new_lines


def connect_paths(lines, tolerance=0.3):
    """線分を最短接続で連結し、リスト化"""
    lines = snap_endpoints(lines, snap_threshold=tolerance)
    merged = linemerge(unary_union(lines))
    # MultiLineString の場合は .geoms を使って各 LineString を取り出す
    if isinstance(merged, MultiLineString):
        segs = list(merged.geoms)
    elif isinstance(merged, LineString):
        segs = [merged]
    else:
        segs = []
    paths = []
    used = [False] * len(segs)
    for idx in range(len(segs)):
        if used[idx]: continue
        path = list(segs[idx].coords)
        used[idx] = True
        extended = True
        while extended:
            extended = False
            for j, seg in enumerate(segs):
                if used[j]: continue
                s0,e0 = seg.coords[0], seg.coords[-1]
                if np.allclose(path[-1], s0, atol=tolerance):
                    path.extend(seg.coords[1:]); used[j]=True; extended=True; break
                if np.allclose(path[-1], e0, atol=tolerance):
                    path.extend(reversed(seg.coords[:-1])); used[j]=True; extended=True; break
                if np.allclose(path[0], e0, atol=tolerance):
                    path = list(seg.coords[:-1]) + path; used[j]=True; extended=True; break
                if np.allclose(path[0], s0, atol=tolerance):
                    path = list(reversed(seg.coords[1:])) + path; used[j]=True; extended=True; break
        paths.append(path)
    return paths

def export_to_dxf(paths, points, fn):
    doc = ezdxf.new(dxfversion='R2010')
    msp = doc.modelspace()
    for path in paths:
        close_flag = np.allclose(path[0], path[-1], atol=1e-6)
        msp.add_lwpolyline(path, close=bool(close_flag))
    for x,y in points:
        msp.add_point((x,y))
    doc.saveas(fn)

def main():
    src = sys.argv[1] if len(sys.argv)>1 else 'input.dxf'
    dst = sys.argv[2] if len(sys.argv)>2 else 'output.dxf'
    print(f"🔍 Loading: {src}")
    lines, pts = load_all_paths_from_dxf(src)
    print(f"  Loaded segments: {len(lines)}, points: {len(pts)}")
    paths = connect_paths(lines)
    print(f"  Connected paths: {len(paths)}")
    export_to_dxf(paths, pts, dst)
    print(f"✅ Saved: {dst}")

if __name__ == '__main__':
    main()
