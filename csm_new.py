#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
import ezdxf
import numpy as np

from shapely.geometry import LineString, Polygon, MultiPolygon, MultiLineString
from shapely.ops import unary_union, linemerge, polygonize

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

# ========= 設定 ==========
INPUT_FILE   = '/Users/tsukudayuuga/Downloads/rect_merged.dxf'
OUTPUT_GCODE = INPUT_FILE.replace('.dxf', '.gcode')

Z_CUT       = 0
Z_MOVE      = 20.0
FEED_RATE   = 500
SAFE_FEED   = 1000
SEGMENTS    = 120
OFFSET_MM   = 1.7 / 2.0  # レーザー幅の半分

print(f"[INIT] SEGMENTS = {SEGMENTS} (type: {type(SEGMENTS)})")
# ==========================

class PathObject:
    def __init__(self, polygon, index):
        self.polygon = polygon
        self.index = index
        self.children = []
        self.parent = None
        self.depth = 0
        self.offset_paths = []

def visualize_hierarchy(path_objs):
    """
    深さごとに色分けしてポリゴン階層を表示
    """
    patches = []
    depths = []
    for obj in path_objs:
        coords = list(obj.polygon.exterior.coords)
        patches.append(MplPolygon(coords, True))
        depths.append(obj.depth)
    fig, ax = plt.subplots()
    coll = PatchCollection(patches, cmap='viridis', alpha=0.6)
    coll.set_array(np.array(depths))
    ax.add_collection(coll)
    ax.autoscale_view()
    cbar = plt.colorbar(coll, ax=ax)
    cbar.set_label('Depth')
    ax.set_aspect('equal')
    plt.title('Polygon Hierarchy Visualization')
    plt.show()

def visualize_cut_paths(ordered_paths):
    """
    ordered_paths: [(start, end, is_inner), ...] のリスト
    切削順と内/外フラグを色分けしてプロットする
    """
    fig, ax = plt.subplots()
    for idx, (st, ed, is_inner) in enumerate(ordered_paths):
        x_vals = [st[0], ed[0]]
        y_vals = [st[1], ed[1]]
        color = '#1f77b4' if is_inner else '#ff7f0e'
        ax.plot(x_vals, y_vals, color=color, linewidth=1)
        mid_x = (st[0] + ed[0]) / 2.0
        mid_y = (st[1] + ed[1]) / 2.0
        ax.text(mid_x, mid_y, str(idx+1), color=color, fontsize=6, ha='center', va='center')
    ax.set_aspect('equal')
    ax.set_title('切削順序と内/外フラグ (赤=内周, 青=外周)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.show()

# ============================================================
# （以下、DXF から線分とビアポイントを取得するヘルパー群は省略）
# ここでの extract_paths_from_dxf() は previous code と同じ動作をします
# ============================================================
def extract_paths_from_dxf(filename):
    doc = ezdxf.readfile(filename)
    msp = doc.modelspace()
    paths = []
    points = []

    def transform_point(pt, insert):
        x, y = pt
        dx, dy = insert.dxf.insert[:2]
        return (x+dx, y+dy)

    def process_entity(e, insert=None):
        result = []
        tp = lambda p: transform_point(p, insert) if insert else p
        try:
            if e.dxftype() == 'LINE':
                start = e.dxf.start
                end = e.dxf.end
                result.append((tp((start[0], start[1])), tp((end[0], end[1]))))

            elif e.dxftype() == 'LWPOLYLINE':
                pts = [tp(p) for p in e.get_points('xy')]
                for i in range(len(pts)-1):
                    result.append((pts[i], pts[i+1]))
                if e.closed:
                    result.append((pts[-1], pts[0]))

            elif e.dxftype() == 'POLYLINE':
                pts = [tp((v.dxf.location[0], v.dxf.location[1])) for v in e.vertices]
                for i in range(len(pts)-1):
                    result.append((pts[i], pts[i+1]))
                if e.is_closed:
                    result.append((pts[-1], pts[0]))

            elif e.dxftype() == 'CIRCLE':
                center = e.dxf.center
                radius = float(e.dxf.radius)
                if radius <= 2:  # DRILL_RADIUS_THRESHOLD と同じ
                    points.append((center[0], center[1]))
                else:
                    segs = int(float(SEGMENTS))
                    pts = [(center[0]+radius*math.cos(a), center[1]+radius*math.sin(a))
                           for a in np.linspace(0, 2*math.pi, segs+1)]
                    for i in range(len(pts)-1):
                        result.append((tp(pts[i]), tp(pts[i+1])))

            elif e.dxftype() == 'ARC':
                pts = [(center[0] + radius*math.cos(a), center[1] + radius*math.sin(a))
                       for a in np.linspace(math.radians(e.dxf.start_angle),
                                            math.radians(e.dxf.end_angle),
                                            int(SEGMENTS)+1)]
                for i in range(len(pts)-1):
                    result.append((tp(pts[i]), tp(pts[i+1])))

            elif e.dxftype() == 'ELLIPSE':
                pts = []
                try:
                    cx, cy = e.dxf.center.x, e.dxf.center.y
                    major = e.dxf.major_axis
                    ratio = e.dxf.ratio
                    a = math.hypot(major.x, major.y)
                    b = a * ratio
                    phi = math.atan2(major.y, major.x)
                    for t in np.linspace(0, 2*math.pi, int(SEGMENTS), endpoint=False):
                        x = a*math.cos(t); y = b*math.sin(t)
                        xr = x*math.cos(phi) - y*math.sin(phi)
                        yr = x*math.sin(phi) + y*math.cos(phi)
                        pts.append((cx + xr, cy + yr))
                    pts.append(pts[0])
                    for i in range(len(pts)-1):
                        result.append((tp(pts[i]), tp(pts[i+1])))
                except Exception:
                    pass

            elif e.dxftype() == 'SPLINE':
                try:
                    ctrl = e.control_points
                    spline_pts = [(pt[0], pt[1]) for pt in ctrl]
                    for i in range(len(spline_pts)-1):
                        result.append((tp(spline_pts[i]), tp(spline_pts[i+1])))
                except Exception:
                    pass

            elif e.dxftype() == 'POINT':
                x, y = e.dxf.location.x, e.dxf.location.y
                points.append((x, y))

        except Exception as ex:
            print(f"[ERROR] {e.dxftype()} 処理エラー: {ex}")

        return result

    for e in msp:
        t = e.dxftype()
        if t == 'POINT':
            x, y = e.dxf.location.x, e.dxf.location.y
            points.append((x, y))
        else:
            for st, ed in process_entity(e):
                paths.append((st, ed))

    return paths, points

# ============================================================
# ここからが「改修された build_path_objects()」です
# ============================================================
def build_path_objects(raw_paths):
    """
    raw_paths: [(start, end), ...] の線分リスト
    → polygonize でセル単位に分割し、
      (1) 面積最大セルを外枠 (ext_poly)
      (2) 残りセルを ext_poly 内にあるものとして hole_cells
      (3) ext_poly 差分 hole_union を計算して「外枠＋穴」を一つの Polygon (or MultiPolygon) にする
    → 最終的に外枠 + 各穴を PathObject として返す
    """
    # 1) LineString を作って polygonize でセルを得る
    lines = [LineString([st, ed]) for st, ed in raw_paths]
    merged = unary_union(lines)
    # polygonize 用に MultiLineString にする
    if isinstance(merged, LineString):
        mls = [merged]
    elif isinstance(merged, MultiLineString):
        mls = list(merged.geoms)
    else:
        # たとえば GeometryCollection など
        mls = []
        try:
            for geom in merged.geoms:
                if isinstance(geom, (LineString, MultiLineString)):
                    mls.append(geom)
        except Exception:
            pass

    cell_polys = list(polygonize(mls))
    print(f"[DEBUG] polygonize でセルが {len(cell_polys)} 件できた")

    if not cell_polys:
        return []  # まったくセルがないなら空を返す

    # 2) セルを面積でソートし、一番大きいセルを ext_poly とみなす
    cell_polys.sort(key=lambda p: p.area, reverse=True)
    ext_poly = cell_polys[0]

    # 3) ext_poly 内にあるセルを hole_cells とする (面積小さいものを穴とみなす)
    hole_cells = []
    for cell in cell_polys[1:]:
        if cell.within(ext_poly):
            hole_cells.append(cell)

    # 4) hole_cells を一つのジオメトリ袋 (hole_union) にまとめる
    if hole_cells:
        hole_union = unary_union(hole_cells).buffer(0)
    else:
        hole_union = Polygon()  # 空のポリゴン

    # 5) ext_poly から hole_union を差分して、外枠＋穴をもつ Polygon を得る
    outer_with_holes = ext_poly.difference(hole_union)

    # outer_with_holes が Polygon or MultiPolygon になる
    holey_polys = []
    if isinstance(outer_with_holes, Polygon):
        holey_polys = [outer_with_holes]
    elif isinstance(outer_with_holes, MultiPolygon):
        holey_polys = list(outer_with_holes.geoms)
    else:
        holey_polys = []

    print(f"[DEBUG] holey_polys 件数: {len(holey_polys)}")

    # 6) 得られた Polygon に対して、外枠と穴を PathObject にする
    path_objs = []
    idx_counter = 0

    for poly in holey_polys:
        # 外枠 (exterior) を取り出す
        ext_coords = list(poly.exterior.coords)
        ext_polygon = Polygon(ext_coords)
        if not ext_polygon.is_valid or ext_polygon.area <= 0 or len(ext_coords) < 3:
            continue
        ext_obj = PathObject(ext_polygon, idx_counter)
        ext_obj.depth = 0
        path_objs.append(ext_obj)
        idx_counter += 1

        # interior (holes) を取り出す
        for hole_ring in poly.interiors:
            hole_coords = list(hole_ring.coords)
            hole_poly = Polygon(hole_coords)
            if not hole_poly.is_valid or hole_poly.area <= 0 or len(hole_coords) < 3:
                continue
            hole_obj = PathObject(hole_poly, idx_counter)
            hole_obj.parent = ext_obj
            hole_obj.depth = 1
            ext_obj.children.append(hole_obj)
            path_objs.append(hole_obj)
            idx_counter += 1

    print(f"[DEBUG] PathObject 件数: {len(path_objs)}")
    return path_objs

# ============================================================
# 以下は元のまま（offset_paths_within_analysis, reorder_paths, generate_gcode など）
# ============================================================
def offset_paths_within_analysis(path_objs, offset_val):
    segs = []
    for root in path_objs:
        # 全てにoffset_val方向でオフセット
        buf = root.polygon.buffer(offset_val, join_style=2)
        if buf.geom_type == 'Polygon':
            coords = list(buf.exterior.coords)
            for i in range(len(coords)-1):
                # inner属性はバッファの正負で判断（正:外周、負:内周/穴）
                is_inner = offset_val < 0
                segs.append((coords[i], coords[i+1], is_inner))
    return segs




def reorder_paths(paths):
    def greedy(segs):
        ordered = []
        if not segs:
            return ordered
        current = segs[0][0]
        used = [False]*len(segs)
        for _ in range(len(segs)):
            best_i, best_d, rev = None, float('inf'), False
            for i, (st, ed, flag) in enumerate(segs):
                if used[i]:
                    continue
                d0 = math.hypot(current[0]-st[0], current[1]-st[1])
                d1 = math.hypot(current[0]-ed[0],   current[1]-ed[1])
                if d0 < best_d:
                    best_i, best_d, rev = i, d0, False
                if d1 < best_d:
                    best_i, best_d, rev = i, d1, True
            if best_i is None:
                break
            used[best_i] = True
            st, ed, fl = segs[best_i]
            if rev:
                st, ed = ed, st
            ordered.append((st, ed, fl))
            current = ed
        return ordered

    inner_segs = [seg for seg in paths if seg[2]]
    outer_segs = [seg for seg in paths if not seg[2]]
    ordered_inner = greedy(inner_segs)
    ordered_outer = greedy(outer_segs)
    return ordered_inner + ordered_outer


# --- cam_new.py の write_lead_in と同じロジックから開始点だけ返す ---
def compute_lead_in_start(x0, y0, x1, y1, is_inner,
                          lead_in_inner=10.0, lead_in_outer=10.0,
                          poly_center=(0.0,0.0)):
    # 切削方向単位ベクトル
    dx, dy = x1-x0, y1-y0
    d = math.hypot(dx, dy)
    ux, uy = (dx/d, dy/d) if d>1e-6 else (0,0)
    # 法線
    nx1, ny1 = -uy, ux
    nx2, ny2 =  uy, -ux

    if is_inner:
        # 中心側を向く法線を選ぶ
        mx, my = (x0+x1)/2, (y0+y1)/2
        cx, cy = poly_center
        dot = (cx-mx)*nx1 + (cy-my)*ny1
        vx, vy = (nx1,ny1) if dot>0 else (nx2,ny2)
        dist = lead_in_inner
        # start から法線の逆向きにオフセット
        xi, yi = x0 - vx*dist, y0 - vy*dist
    else:
        # 外周は切削線の逆方向にオフセット
        dist = lead_in_outer
        xi, yi = x0 - ux*dist, y0 - uy*dist

    return xi, yi

# --- 新たに追加するヘルパー関数 ---
def write_lead_in(
    f,
    x0: float, y0: float,
    x1: float, y1: float,
    is_inner: bool,
    lead_in_dist_inner: float = 10.0,
    lead_in_dist_outer: float = 10.0,
    poly_center: tuple[float, float] = (0.0, 0.0),
):
    """
    start=(x0,y0), next=(x1,y1) に対して、
      is_inner=True → 内周 (法線方向) からリードイン
      is_inner=False → 外周 (切削線の延長方向) からリードイン
    lead_in_dist_inner/outer: 内外でリードイン長さを分ける (mm)
    poly_center: 内周判定用の親ポリゴン重心 (x, y)
    """

    # 1) 切削方向の単位ベクトル
    dx, dy = x1 - x0, y1 - y0
    d = math.hypot(dx, dy)
    if d < 1e-6:
        ux, uy = 0.0, 0.0
    else:
        ux, uy = dx / d, dy / d

    # 2) 法線ベクトル候補 (左回り・右回り)
    px1, py1 = -uy, ux
    px2, py2 =  uy, -ux

    if is_inner:
        # ── 内周: 「中心側」を向く法線を選ぶ ──
        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        cx, cy = poly_center
        dot = (cx - mx) * px1 + (cy - my) * py1
        if dot > 0:
            vx, vy = px1, py1
        else:
            vx, vy = px2, py2
        dist = lead_in_dist_inner
    else:
        # ── 外周: 切削線の逆方向 (延長方向) を使う ──
        vx, vy = -ux, -uy
        dist = lead_in_dist_outer

    # 3) リードイン方向を単位化
    lv = math.hypot(vx, vy)
    if lv < 1e-6:
        ivx, ivy = ux, uy
    else:
        ivx, ivy = vx / lv, vy / lv

    # 4) リードイン開始点 (start から逆向きに dist だけオフセット)
    if is_inner:
        xi = x0 - ivx * dist
        yi = y0 - ivy * dist
    else:
        xi = x0 + ivx * dist
        yi = y0 + ivy * dist

    # 5) G‑Code 出力
    f.write("\n; Lead-in to new start\n")
    f.write(f"G0 X{xi:.3f} Y{yi:.3f} F{SAFE_FEED}\n")
    f.write(f"G1 Z{Z_CUT:.3f} F{FEED_RATE}\n")
    f.write("M3 ; プラズマ ON\n")
    f.write("G4 P1.2 ; 安定待機\n")
    f.write(f"G1 X{x0:.3f} Y{y0:.3f} F{FEED_RATE}\n")


def generate_gcode(paths, points, output_file):
    with open(output_file, 'w') as f:
        f.write("G21 ; 単位: mm\n")
        f.write("G90 ; 絶対座標\n")
        f.write(f"G0 Z{Z_MOVE:.3f} F{SAFE_FEED}\n")

        # === STEP 1: ビア処理（最優先） ===
        for pt in points:
            x, y = pt
            f.write(f"\n; Drill point at ({x:.3f}, {y:.3f})\n")
            f.write(f"G0 X{x:.3f} Y{y:.3f} F{SAFE_FEED}\n")
            f.write(f"G1 Z{Z_CUT:.3f} F{FEED_RATE}\n")
            f.write("M3 ; プラズマ開始\n")
            f.write("G4 P1.2 ; 射出安定待機\n")
            f.write("G4 P0.1 ; ビア照射\n")
            f.write("M5 ; プラズマ停止\n")
            f.write(f"G1 Z{Z_MOVE:.3f} F{SAFE_FEED}\n")

        # === STEP 2: 通常パスの処理 ===
        last_end = None
        pen_down = False
        for segment in paths:
            if len(segment) == 3:
                start, end, is_inner = segment
            else:
                start, end = segment
                is_inner = False

            x0, y0 = start
            x1, y1 = end

            if start == end:
                continue

            if last_end != start:
                if pen_down:
                    f.write("M5 ; プラズマ停止\n")
                    f.write("G4 P0.05 ; 停止まで待機\n")
                    f.write(f"G1 Z{Z_MOVE:.3f} F{SAFE_FEED}\n")
                    pen_down = False
                write_lead_in(f, x0, y0, x1, y1, is_inner)
                # f.write("\n; Move to new start\n")
                # f.write(f"G0 X{x0:.3f} Y{y0:.3f} F{SAFE_FEED}\n")
                # f.write(f"G1 Z{Z_CUT:.3f} F{FEED_RATE}\n")
                # f.write("M3 ; プラズマ開始\n")
                # f.write("G4 P1.2 ; 射出安定待機\n")
                pen_down = True

            f.write(f"G1 X{x1:.3f} Y{y1:.3f} F{FEED_RATE}\n")
            # if is_inner:
            #     f.write("G4 P0.05 ; test\n")

            last_end = end

        if pen_down:
            f.write("M5 ; プラズマ停止\n")
            f.write(f"G1 Z{Z_MOVE:.3f} F{SAFE_FEED}\n")

        f.write("\nM2 ; プログラム終了\n")
    print(f"G-code を出力しました: {output_file}")

# ============================================================
# main 部分
# ============================================================
if __name__ == '__main__':
    # 1) DXF から raw_paths とビアポイントを取得
    raw_paths, drill_points = extract_paths_from_dxf(INPUT_FILE)
    print(f"[DEBUG] raw_paths 件数: {len(raw_paths)}, drill_points 件数: {len(drill_points)}")

    # 2) Polygon 化 & PathObject 化
    objs = build_path_objects(raw_paths)

    # 3) デバッグ: PathObject info 出力
    for o in objs:
        b = o.polygon.bounds
        print(f"[DEBUG] Index={o.index}, area={o.polygon.area:.2f}, bounds=({b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}), depth={o.depth}, parent_index={(o.parent.index if o.parent else None)}")
    visualize_hierarchy(objs)  # 必要なら

    # 4) オフセットパスを作成 (穴→外枠)
    offset_segs = offset_paths_within_analysis(objs)

    # 5) 切削順を決定
    ordered = reorder_paths(offset_segs)
    visualize_cut_paths(ordered)

    # 6) G-code を出力
    generate_gcode(ordered, drill_points, OUTPUT_GCODE)
