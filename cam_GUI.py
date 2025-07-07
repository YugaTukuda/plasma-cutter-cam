#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import math
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QGroupBox, QFormLayout
)
from PySide6.QtCore import Qt

# matplotlib integration
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
)
import matplotlib.pyplot as plt

# shapely for geometry buffer
from shapely.geometry import Polygon

# --- User modules ---
import csm_new      # place csm_new.py alongside
import cam_merged_dxf  # place cam_merged_dxf.py alongside

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("G-code Generator GUI")
        self.raw_segments = []
        self.drill_points = []
        self.paths = []
        self.merged = []
        self.poly_center = (0.0, 0.0)
        # edit modes: 'none', 'offset', 'leadin'
        self.edit_mode = False
        self.edit_target = 'none'
        self.line_artists = []
        self.lead_in_artists = []
        self.user_lead_overrides = {}
        self.dragging_idx = None
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        main_layout = QHBoxLayout(central)
        # Control panel
        ctrl_widget = QWidget()
        ctrl_layout = QVBoxLayout(ctrl_widget)
        # File selection
        hfile = QHBoxLayout()
        hfile.addWidget(QLabel("ファイルを読み込み"))
        self.file_edit = QLineEdit()
        hfile.addWidget(self.file_edit)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self.browse_dxf)
        hfile.addWidget(btn_browse)
        ctrl_layout.addLayout(hfile)
        # Parameters
        param_grp = QGroupBox("Parameters")
        form = QFormLayout()
        self.z_cut_edit     = QLineEdit(str(csm_new.Z_CUT))
        self.z_move_edit    = QLineEdit(str(csm_new.Z_MOVE))
        self.feed_rate_edit = QLineEdit(str(csm_new.FEED_RATE))
        self.safe_feed_edit = QLineEdit(str(csm_new.SAFE_FEED))
        self.segs_edit      = QLineEdit(str(csm_new.SEGMENTS))
        form.addRow("Z_CUT:", self.z_cut_edit)
        form.addRow("Z_MOVE:", self.z_move_edit)
        form.addRow("FEED_RATE:", self.feed_rate_edit)
        form.addRow("SAFE_FEED:", self.safe_feed_edit)
        form.addRow("SEGMENTS:", self.segs_edit)
        self.offset_outer_edit = QLineEdit("1")
        self.offset_inner_edit = QLineEdit("0.5")
        form.addRow("Offset Outer (外側):", self.offset_outer_edit)
        form.addRow("Offset Inner (内側):", self.offset_inner_edit)
        self.leadin_inner_edit = QLineEdit("10.0")
        self.leadin_outer_edit = QLineEdit("10.0")
        form.addRow("Lead-in Inner:", self.leadin_inner_edit)
        form.addRow("Lead-in Outer:", self.leadin_outer_edit)
        self.grid_major_edit = QLineEdit("10")
        self.grid_minor_edit = QLineEdit("5")
        form.addRow("Grid Major:", self.grid_major_edit)
        form.addRow("Grid Minor:", self.grid_minor_edit)
        param_grp.setLayout(form)
        ctrl_layout.addWidget(param_grp)
        # Buttons
        btn_load = QPushButton("Load")
        btn_load.clicked.connect(self.on_load)
        btn_calc = QPushButton("Calculate")
        btn_calc.clicked.connect(self.on_calc)
        btn_offset = QPushButton("Edit Offset")
        btn_offset.setCheckable(True)
        btn_offset.toggled.connect(self.on_offset_edit_toggle)
        btn_leadin = QPushButton("Edit Lead-in")
        btn_leadin.setCheckable(True)
        btn_leadin.toggled.connect(self.on_leadin_edit_toggle)
        btn_gen = QPushButton("Generate G-code")
        btn_gen.clicked.connect(self.on_generate)
        for btn in (btn_load, btn_calc, btn_offset, btn_leadin, btn_gen):
            ctrl_layout.addWidget(btn)
        ctrl_layout.addStretch(1)
        main_layout.addWidget(ctrl_widget, 0)
        # Plot area
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        toolbar = NavigationToolbar(self.canvas, self)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas)
        main_layout.addWidget(plot_widget, 1)
        self.setCentralWidget(central)
        for wid in (self.grid_major_edit, self.grid_minor_edit,
                    self.leadin_inner_edit, self.leadin_outer_edit):
            wid.editingFinished.connect(self._refresh_grid)

    def on_offset_edit_toggle(self, checked):
        # Toggle offset-edit mode
        self.edit_mode = checked
        self.edit_target = 'offset' if checked else 'none'
        if checked:
            # connect only pick for offset toggling
            self.cid_offset_pick = self.canvas.mpl_connect('pick_event', self.on_pick)
        else:
            if hasattr(self, 'cid_offset_pick'):
                self.canvas.mpl_disconnect(self.cid_offset_pick)

    def on_leadin_edit_toggle(self, checked):
        # Toggle lead-in-edit mode
        self.edit_mode = checked
        self.edit_target = 'leadin' if checked else 'none'
        if checked:
            self.cid_lead_pick = self.canvas.mpl_connect('pick_event', self.on_pick)
            self.cid_lead_motion = self.canvas.mpl_connect('motion_notify_event', self.on_motion)
            self.cid_lead_release = self.canvas.mpl_connect('button_release_event', self.on_release)
        else:
            for cid in ('cid_lead_pick', 'cid_lead_motion', 'cid_lead_release'):
                if hasattr(self, cid):
                    self.canvas.mpl_disconnect(getattr(self, cid))

    def on_pick(self, event):
        artist = event.artist
        if self.edit_target == 'offset':
            for art, idx in self.line_artists:
                if art == artist:
                    group = self._get_group_indices(idx)
                    # どのループか？先頭インデックスでループ決定
                    loop_idx = self.loop_indices[idx]
                    orig_coords, orig_inner = self.base_loops[loop_idx]
                    # 必ず始点/終点を揃えてpolygon化
                    coords = list(orig_coords)
                    if coords[0] != coords[-1]:
                        coords.append(coords[0])
                    poly = Polygon(coords)
                    # 反転
                    new_inner = not orig_inner
                    outer_offset = float(self.offset_outer_edit.text())
                    inner_offset = float(self.offset_inner_edit.text())
                    offset_val = abs(inner_offset) if new_inner else -abs(outer_offset)
                    new_poly = poly.buffer(offset_val, join_style=2)
                    new_coords = list(new_poly.exterior.coords)
                    new_segments = [(new_coords[i], new_coords[i+1], new_inner)
                                    for i in range(len(new_coords)-1)]
                    start = group[0]
                    self.paths[start:start+len(group)] = new_segments
                    for i in group:
                        self.user_lead_overrides.pop(i, None)
                    self._plot_processed(self.paths)
                    return
        elif self.edit_target == 'leadin':
            for art, idx in self.lead_in_artists:
                if art == artist:
                    self.dragging_idx = idx
                    return
                
    def _find_base_loop_idx(self, group):
    # ループのインデックス特定方法（必要なら改善）
        return group[0]  # paths/groupの先頭index＝base_loopsのindex、と仮定

    def on_motion(self, event):
        if self.edit_mode and self.edit_target == 'leadin' and self.dragging_idx is not None and event.inaxes:
            self.user_lead_overrides[self.dragging_idx] = (event.xdata, event.ydata)
            self._plot_processed(self.paths)

    def on_release(self, event):
        self.dragging_idx = None

    def _get_group_indices(self, clicked_idx):
        if not self.paths: return [clicked_idx]
        groups, group = [], [0]
        for i in range(1, len(self.paths)):
            if self.paths[i-1][1] == self.paths[i][0]: group.append(i)
            else:
                groups.append(group)
                group = [i]
        groups.append(group)
        for g in groups:
            if clicked_idx in g: return g
        return [clicked_idx]

    def compute_lead_in_start(self, x0, y0, x1, y1, is_inner,
                              lead_in_dist_inner=10.0, lead_in_dist_outer=10.0):
        dx, dy = x1 - x0, y1 - y0
        d = math.hypot(dx, dy)
        if d < 1e-6:
            ux, uy = 0.0, 0.0
        else:
            ux, uy = dx / d, dy / d
        px1, py1 = -uy, ux
        px2, py2 = uy, -ux
        if is_inner:
            mx, my = (x0 + x1)/2.0, (y0 + y1)/2.0
            cx, cy = self.poly_center
            dot = (cx - mx)*px1 + (cy - my)*py1
            vx, vy = (px1, py1) if dot > 0 else (px2, py2)
            dist = lead_in_dist_inner
        else:
            vx, vy = -ux, -uy
            dist = lead_in_dist_outer
        lv = math.hypot(vx, vy)
        if lv < 1e-6:
            ivx, ivy = ux, uy
        else:
            ivx, ivy = vx / lv, vy / lv
        if is_inner:
            xi = x0 + ivx * dist
            yi = y0 + ivy * dist
        else:
            xi = x0 + ivx * dist
            yi = y0 + ivy * dist
        return xi, yi


    def browse_dxf(self):  # ... same
        path, _ = QFileDialog.getOpenFileName(self, "Select DXF file",
                                               os.getcwd(), "DXF Files (*.dxf)")
        if path: self.file_edit.setText(path)
        
    from shapely.geometry import Polygon


    

    def on_load(self):  # ... same
        fx = self.file_edit.text().strip()
        if not fx.lower().endswith('.dxf') or not os.path.exists(fx):
            QMessageBox.warning(self, "File Error", "有効な .dxf ファイルを選択してください。")
            return
        try:
            csm_new.Z_CUT     = float(self.z_cut_edit.text())
            csm_new.Z_MOVE    = float(self.z_move_edit.text())
            csm_new.FEED_RATE = float(self.feed_rate_edit.text())
            csm_new.SAFE_FEED = float(self.safe_feed_edit.text())
            csm_new.SEGMENTS  = int(float(self.segs_edit.text()))
            # csm_new.OFFSET_MM = float(self.offset_edit.text())
            cam_merged_dxf.SEGMENTS = csm_new.SEGMENTS
            lines, pts = cam_merged_dxf.load_all_paths_from_dxf(fx)
            print(f"Loaded segments: {len(lines)}, points: {len(pts)}")  # ←ここ!!
            via_pts, rem = cam_merged_dxf.detect_via_lines(lines)
            merged = cam_merged_dxf.connect_paths(rem)
            segs = [(p[i], p[i+1]) for p in merged for i in range(len(p)-1)]
            self.raw_segments, self.drill_points, self.merged = segs, pts+via_pts, merged
            self.user_lead_overrides.clear()
            self._plot_merged(merged)
            QMessageBox.information(self, "Loaded", "マージされたパスを表示しました。")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def on_calc(self):
        """
        Calculate offset considering parent-child relationships,
        using independent GUI input for outer/inner offsets.
        """
        if not self.raw_segments:
            QMessageBox.warning(self, "No Data", 'まず "Load" を実行してください。')
            return
        contours = self.merged
        from shapely.geometry import Polygon
    
        # Filter contours that can form a polygon (at least 4 points)
        valid_idxs = [i for i, c in enumerate(contours) if len(c) >= 4]
        if not valid_idxs:
            QMessageBox.warning(self, "Error", "有効な図形が見つかりませんでした。")
            return
        open_idxs = [i for i, c in enumerate(contours) if len(c) < 4]
    
        # Build polygon objects for valid contours
        polys = [Polygon(contours[i]) for i in valid_idxs]
        # Map from parent index to children indices
        child_map = {i: [] for i in valid_idxs}
        # Determine parent-child using covers (including boundaries)
        for pi, i in enumerate(valid_idxs):
            for pj, j in enumerate(valid_idxs):
                if i != j and polys[pj].covers(polys[pi]):
                    child_map[j].append(i)
        # Identify parents: valid_idxs not covered by any other
        parents = [i for i in valid_idxs
                   if not any(polys[pj].covers(polys[valid_idxs.index(i)])
                              for pj in range(len(polys)) if valid_idxs[pj] != i)]
    
        # --- ここでGUIから直接オフセット値取得 ---
        try:
            outer_offset = float(self.offset_outer_edit.text())
            inner_offset = float(self.offset_inner_edit.text())
        except Exception as e:
            QMessageBox.warning(self, "Error", f"オフセット値が数値ではありません: {e}")
            return
    
        def enforce_loop_winding_order(loop, ccw=True):
            """loop: [(st, ed, inner), ...]。stだけを集めて向きを判定・修正"""
            pts = [seg[0] for seg in loop] + [loop[-1][1]]
            poly = Polygon(pts)
            is_ccw = poly.exterior.is_ccw
            # ccw=Trueなら反時計回りにしたい、Falseなら時計回り
            if is_ccw == ccw:
                return loop
            else:
                # 向きを反転（start, end, inner を逆順）
                return [(ed, st, inner) for (st, ed, inner) in reversed(loop)]
    
        self.paths.clear()
        self.base_loops = []
        self.loop_indices = []
        
        loop_id = 0
        for p_idx in parents:
            pts_loop = contours[p_idx]
            # --- offset後のパスを作る ---
            seg_pairs = [(pts_loop[k], pts_loop[k+1]) for k in range(len(pts_loop)-1)]
            objs = csm_new.build_path_objects(seg_pairs)
            segs = csm_new.offset_paths_within_analysis(objs, abs(outer_offset))
            ordered = csm_new.reorder_paths(segs)
            loop = [(st, ed, False) for (st, ed, inner) in ordered]
            # --- windingもここで ---
            loop = enforce_loop_winding_order(loop, ccw=False)
            self.paths.extend(loop)
            self.base_loops.append((pts_loop, False))
            self.loop_indices.extend([loop_id] * len(loop))  # ← loopごとに何セグメント分も拡張
            loop_id += 1
            # 子
            for c_idx in child_map.get(p_idx, []):
                hole_pts = contours[c_idx]
                seg_pairs_hole = [(hole_pts[k], hole_pts[k+1]) for k in range(len(hole_pts)-1)]
                objs_hole = csm_new.build_path_objects(seg_pairs_hole)
                segs_hole = csm_new.offset_paths_within_analysis(objs_hole, -abs(inner_offset))
                ordered_hole = csm_new.reorder_paths(segs_hole)
                loop = [(st, ed, True) for (st, ed, inner) in ordered_hole]
                loop = enforce_loop_winding_order(loop, ccw=True)
                self.paths.extend(loop)
                self.base_loops.append((hole_pts, True))
                self.loop_indices.extend([loop_id] * len(loop))
                loop_id += 1
                
        # ----(開いた図形)----
        for i in open_idxs:
            pts = contours[i]
            # n=2なら直線, n>2なら折れ線
            for k in range(len(pts)-1):
                self.paths.append((pts[k], pts[k+1], False))  # inner=False扱いで
            self.base_loops.append((pts, False))
            self.loop_indices.extend([loop_id] * (len(pts)-1))
            loop_id += 1
    
        # Use first parent centroid for lead-in reference
        if parents:
            self.poly_center = Polygon(contours[parents[0]]).centroid.coords[0]
        # Refresh display
        self.user_lead_overrides.clear()
        self._plot_processed(self.paths)
        QMessageBox.information(self, "Calculated", "親子関係に基づくオフセット処理が完了しました。")

    
    def on_generate(self):
        if not self.paths:
            QMessageBox.warning(self, "No Data", 'まず "Calculate" を実行してください。')
            return
        
        # ここで必ず最新値を取得してcsm_newに反映する
        csm_new.Z_CUT     = float(self.z_cut_edit.text())
        csm_new.Z_MOVE    = float(self.z_move_edit.text())
        csm_new.FEED_RATE = float(self.feed_rate_edit.text())
        csm_new.SAFE_FEED = float(self.safe_feed_edit.text())
        csm_new.SEGMENTS  = int(float(self.segs_edit.text()))
        # オフセットはここでは不要（on_calcで使うので）
    
        self._plot_processed(self.paths)
    
        inner_dist = float(self.leadin_inner_edit.text())
        outer_dist = float(self.leadin_outer_edit.text())
        od = QFileDialog.getExistingDirectory(self, "Select output folder", os.getcwd())
        if not od:
            return
        base = os.path.basename(self.file_edit.text()).replace('.dxf','')
    
        def write_header(f):
            f.write("G21 ; 単位: mm\n")
            f.write("G90 ; 絶対座標\n")
            f.write(f"G0 Z{csm_new.Z_MOVE} F{csm_new.SAFE_FEED}\n")
    
        lead_map = {
            idx: (artist.get_xdata()[0], artist.get_ydata()[0])
            for artist, idx in self.lead_in_artists
        }
    
        eps = 1e-6
        def same(a, b): return abs(a[0]-b[0])<eps and abs(a[1]-b[1])<eps
    
        # ループ単位に分割
        loops = []
        cur = []
        for idx, (st, ed, inner) in enumerate(self.paths):
            if not cur:
                cur = [(st, ed, inner, idx)]
            elif same(cur[-1][1], st):
                cur.append((st, ed, inner, idx))
            else:
                loops.append(cur)
                cur = [(st, ed, inner, idx)]
        if cur: loops.append(cur)
    
        all_loops = {'inner':[], 'outer':[]}
        for loop in loops:
            is_inner = loop[0][2]
            (all_loops['inner'] if is_inner else all_loops['outer']).append(loop)
    
        # --- すべて一つのファイルに出力 ---
        cut_file = os.path.join(od, f"{base}.gcode")
        with open(cut_file, 'w') as cf:
            write_header(cf)
    
            # ピアス
            for i, (x, y) in enumerate(self.drill_points):
                cf.write(f"; Pierce at ({x:.3f}, {y:.3f})\n")
                cf.write(f"G0 X{x:.3f} Y{y:.3f} F{csm_new.SAFE_FEED}\n")
                cf.write(f"G1 Z{csm_new.Z_CUT} F{csm_new.FEED_RATE}\n")
                cf.write("M3 ; プラズマ ON\n")
                if i == 0:
                    cf.write("G4 P1 ; ピアス待機\n")  # 最初だけ1.5秒
                else:
                    cf.write("G4 P0.5 ; ピアス待機\n")  # 2個目以降は0.7秒
                cf.write("M5 ; プラズマ OFF\n")
                cf.write(f"G0 Z{csm_new.Z_MOVE} F{csm_new.SAFE_FEED}\n")

    
            # 一時停止（オペレータに指示するコメントも）
            cf.write("M0 ; === ピアス完了・刃交換や位置調整などあればここで一時停止 ===\n")
    
            # 切削（従来通り）
            for category in ('inner','outer'):
                for loop in all_loops[category]:
                    st0, ed0, inner0, idx0 = loop[0]
                    st_last, ed_last, inner_last, idx_last = loop[-1]
    
                    # リードイン位置
                    if idx0 in self.user_lead_overrides:
                        xi, yi = self.user_lead_overrides[idx0]
                    else:
                        xi, yi = lead_map[idx0]
    
                    # ①リードイン位置まで移動
                    cf.write(f"; Lead-in for loop start at segment #{idx0}\n")
                    cf.write(f"G0 X{xi:.3f} Y{yi:.3f} F{csm_new.SAFE_FEED}\n")
                    # ②Z軸下げ
                    cf.write(f"G1 Z{csm_new.Z_CUT} F{csm_new.FEED_RATE}\n")
                    # ③プラズマON・安定待機
                    cf.write("M3 ; プラズマ ON\n")
                    cf.write("G4 P1 ; 安定待機\n")
                    # ④リードインでパス始点まで移動
                    cf.write(f"G1 X{st0[0]:.3f} Y{st0[1]:.3f} F{csm_new.FEED_RATE}\n")
    
                    # ⑤ループ本体
                    for st, ed, inner, idx in loop:
                        cf.write(f"G1 X{ed[0]:.3f} Y{ed[1]:.3f} F{csm_new.FEED_RATE}\n")
                    # 必要なら明示的に始点に戻す（閉ループ保証）
                    if not same(loop[0][0], loop[-1][1]):
                        cf.write(f"G1 X{loop[0][0]:.3f} Y{loop[0][1]:.3f} F{csm_new.FEED_RATE}\n")
    
                    # ⑥Zアップ・プラズマOFF
                    cf.write("M5 ; プラズマ OFF\n")
                    cf.write(f"G0 Z{csm_new.Z_MOVE} F{csm_new.SAFE_FEED}\n")
    
            cf.write("M2 ; プログラム終了\n")
    
        QMessageBox.information(
            self, "Saved",
            f"出力ファイル: {cut_file}"
        )


    def _refresh_grid(self):
        if self.paths: self._plot_processed(self.paths)
        elif self.merged: self._plot_merged(self.merged)

    def _setup_axes(self, ax):
        ax.axhline(0, color='red', lw=1.0, zorder=3)
        ax.axvline(0, color='green', lw=1.0, zorder=3)
        for spine in ax.spines.values(): spine.set_color('black'); spine.set_linewidth(1.0)
        ax.tick_params(axis='x', colors='black'); ax.tick_params(axis='y', colors='black')

    def _apply_grid(self, ax):
        try:
            gm, mi = float(self.grid_major_edit.text()), float(self.grid_minor_edit.text())
            xmin, xmax = ax.get_xlim(); ymin, ymax = ax.get_ylim()
            if gm > 0:
                ax.set_xticks(np.arange(math.floor(xmin/gm)*gm, math.ceil(xmax/gm)*gm+gm, gm))
                ax.set_yticks(np.arange(math.floor(ymin/gm)*gm, math.ceil(ymax/gm)*gm+gm, gm))
            if mi > 0:
                ax.set_xticks(np.arange(math.floor(xmin/mi)*mi, math.ceil(xmax/mi)*mi+mi, mi), minor=True)
                ax.set_yticks(np.arange(math.floor(ymin/mi)*mi, math.ceil(ymax/mi)*mi+mi, mi), minor=True)
            ax.grid(which='major', lw=0.2, alpha=0.4); ax.grid(which='minor', lw=0.1, alpha=0.3)
        except:
            ax.grid(True)

    def _plot_merged(self, merged):
        self.figure.clear(); ax = self.figure.add_subplot(111)
        for p in merged: xs, ys = zip(*p); ax.plot(xs, ys, color='#444444', linewidth=0.7)
        if self.drill_points:
            xs, ys = zip(*self.drill_points)
            ax.scatter(xs, ys, s=5, color='#8B008B', marker='o', zorder=10, label='Pierce (Drill)')
        self._apply_grid(ax); self._setup_axes(ax)
        ax.set_aspect('equal', 'datalim'); ax.set_xlabel('X'); ax.set_ylabel('Y')
        self.figure.tight_layout(); self.canvas.draw()

    def _plot_processed(self, paths):
        self.figure.clear(); ax = self.figure.add_subplot(111)
        self.line_artists.clear(); self.lead_in_artists.clear()
        for idx, (st, ed, inner) in enumerate(paths):
            color = '#3399FF' if not inner else '#FF9933'
            artist, = ax.plot([st[0], ed[0]], [st[1], ed[1]], color=color, linestyle='-', linewidth=0.7, picker=5)
            self.line_artists.append((artist, idx))
        inner_dist = float(self.leadin_inner_edit.text()); outer_dist = float(self.leadin_outer_edit.text())
        last_start = None
        for idx, (st, ed, inner) in enumerate(paths):
            if last_start != st:
                xi, yi = self.user_lead_overrides.get(idx,
                    self.compute_lead_in_start(st[0], st[1], ed[0], ed[1], inner, inner_dist, outer_dist))
                lead_artist, = ax.plot([xi, st[0]], [yi, st[1]], linestyle='--', linewidth=0.7, picker=5)
                self.lead_in_artists.append((lead_artist, idx))
            last_start = ed
        if self.drill_points:
            xs, ys = zip(*self.drill_points)
            ax.scatter(xs, ys, s=5, color='#8B008B', marker='o', zorder=10, label='Pierce (Drill)')
        self._apply_grid(ax); self._setup_axes(ax)
        ax.set_aspect('equal', 'datalim'); ax.set_xlabel('X'); ax.set_ylabel('Y')
        self.figure.tight_layout(); self.canvas.draw()
        

if __name__ == '__main__':
    app = QApplication(sys.argv); w = MainWindow(); w.show(); sys.exit(app.exec())
