"""Local desktop preview/control panel (tkinter). Not a web app."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
import numpy as np

from .calibration import detect_markers, draw_detections, make_detector
from .replay import depth_to_preview


class OperatorUI:
    def __init__(self, rec, allow_record: bool = True, scene_defaults: dict | None = None, task: str = "unnamed",
                 instruction: str = "", pilot: dict | None = None):
        self.rec, self.allow_record = rec, allow_record
        self.task, self.instruction, self.pilot = task, instruction, pilot
        self.root = tk.Tk()
        self.root.title("Robot data collector" + ("  [MOCK DATA]" if rec.is_mock else ""))
        self.img = tk.Label(self.root)
        self.img.grid(row=0, column=0, columnspan=2)
        self.info = tk.Label(self.root, justify="left", anchor="w", font=("monospace", 9), width=90)
        self.info.grid(row=1, column=0, columnspan=2, sticky="w")
        bar = ttk.Frame(self.root)
        bar.grid(row=2, column=0, columnspan=2, sticky="w")
        self.notes = ""
        self.outcome = tk.StringVar(value="unlabelled")
        self.show_depth = tk.BooleanVar(value=False)
        b = lambda t, c: ttk.Button(bar, text=t, command=c).pack(side="left", padx=2)
        if allow_record:
            b("Start episode", self.start)
            b("Finish episode", self.finish)
            b("Abort...", self.abort)
            for o in ("success", "failure", "unlabelled"):
                ttk.Radiobutton(bar, text=o, variable=self.outcome, value=o).pack(side="left")
            b("Notes...", self.enter_notes)
            self.fail_cat = ""
        ttk.Checkbutton(bar, text="depth view", variable=self.show_depth).pack(side="left")
        b("Quit", self.quit)
        self.detector = None
        m = rec.cfg["markers"]
        self.expected_ids = tuple(c["id"] for c in m["corners"].values() if c["id"] is not None)
        if m["dictionary"]:
            try:
                self.detector = make_detector(m["dictionary"])
            except Exception:
                self.detector = None
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self._tick()

    def start(self):
        try:
            self.rec.start_episode(self.task, self.instruction, pilot=self.pilot)
        except Exception as e:
            messagebox.showerror("Cannot start", str(e))

    def finish(self):
        try:
            if self.outcome.get() == "failure":
                self.fail_cat = simpledialog.askstring("Failure category", "Failure category (free text):") or ""
            src = "operator_visual" if self.outcome.get() in ("success", "failure") else None  # the operator selected it
            p = self.rec.finish_episode(self.outcome.get(), getattr(self, "fail_cat", None) or None, self.notes, src)
            messagebox.showinfo("Saved", str(p))
            self.notes = ""
        except Exception as e:
            messagebox.showerror("Cannot finish", str(e))

    def abort(self):
        r = simpledialog.askstring("Abort", "Abort reason (required):")
        if r:
            try:
                self.rec.abort_episode(r)
            except Exception as e:
                messagebox.showerror("Cannot abort", str(e))

    def enter_notes(self):
        self.notes = simpledialog.askstring("Notes", "Episode notes:", initialvalue=self.notes) or self.notes

    def quit(self):
        if self.rec.stats()["recording"] and not messagebox.askyesno(
                "Recording", "An episode is recording. Quitting aborts it (the robot is NOT affected). Quit?"):
            return
        self.rec.shutdown()  # releases ports only; no torque/mode change
        self.root.destroy()

    def _tick(self):
        st = self.rec.stats()
        reason = self.rec.should_abort()
        if reason and st["recording"]:
            self.rec.abort_episode(reason)
        frames = [(n, f) for n, f in list(self.rec.latest.items()) if hasattr(f, "color_rgb")]
        if frames:
            name, fr = frames[0]
            bgr = cv2.cvtColor(fr.color_rgb, cv2.COLOR_RGB2BGR)
            if self.show_depth.get() and fr.depth_aligned is not None:
                bgr = depth_to_preview(fr.depth_aligned)
            elif self.detector is not None:
                bgr = draw_detections(bgr, detect_markers(bgr, self.detector), self.expected_ids)
            h, w = bgr.shape[:2]
            s = min(1.0, 640 / w)
            bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
            ok, png = cv2.imencode(".png", bgr)
            import base64
            self._photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
            self.img.configure(image=self._photo)
        lines = [("RECORDING " + st["episode_id"]) if st["recording"] else "idle",
                 f"calibration: {st['calibration'] or 'NONE (diagnostic only)'}   mode: {self.rec.cfg['mode']}"
                 + ("   *** MOCK DATA ***" if st["mock"] else "")]
        for n, s in st["streams"].items():
            lines.append(f"{n:14s} {'OK ' if s['connected'] else 'DOWN'} {s['rate_hz']:6.1f} Hz n={s['count']} {s['error'] or ''}")
        if st["recording"]:
            q = st["queue"]
            lines.append(f"queue {q['size']}/{q['max']} (peak {q['peak']})  writer {st['writer_rate_hz']:.1f} items/s  "
                         f"drops {st['dropped'] or 'none'}")
        for w in self.rec.warnings:
            lines.append("warn: " + w)
        self.info.configure(text="\n".join(lines))
        self.root.after(100, self._tick)

    def run(self):
        self.root.mainloop()
