from __future__ import annotations
import requests
import time
from datetime import datetime, timezone, timedelta


class ApiError(RuntimeError):
    pass


class Client:
    CAMERAS = {"rgbd_left", "rgbd_right", "rgbd_hand"}
    DBS = {"dual_rgb", "rgbd_hand"}
    DONE = {"succeeded", "failed", "cancelled"}

    def __init__(self, url="http://127.0.0.1:8000", timeout=60):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.s = requests.Session()

    def close(self):
        self.s.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _call(self, method, path, *, raw=False, **kwargs):
        r = self.s.request(method, self.url + path, timeout=self.timeout, **kwargs)
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text
            raise ApiError(f"{method} {path} -> {r.status_code}: {detail}")
        if raw:
            return r.content
        return r.json() if r.content else None

    @staticmethod
    def _check(name, allowed):
        if name not in allowed:
            raise ValueError(f"{name!r} must be one of {sorted(allowed)}")
        return name

    # ---------- basic ----------

    def refresh(self):
        return self._call("GET", "/refresh")

    def open_dual_rgb(self,json={
            "rgb_size": [3872,3008],
            "stereo_size": [1280,800],
            "mjpeg_quality": 95,"fps": 10,
            "max_exposure_us": 16667,
            "timeout_s": 20
        }):
        return self._call("POST", "/open_dual_rgb", json=json)

    def open_hand(self,json={
            "rgb_size": [3872,3008],
            "stereo_size": [1280,800],
            "mjpeg_quality": 95,"fps": 10,
            "max_exposure_us": 16667,
            "timeout_s": 20
        }):
        return self._call("POST", "/open_hand", json=json)

    def close_cams(self):
        return self._call("GET", "/close_cams")

    def capture_dual(self, meta=None):
        return self._call("POST", "/capture_dual_rgb", json={"meta": meta or {}})

    def capture_hand(self, meta=None):
        return self._call("POST", "/capture_hand", json={"meta": meta or {}})

    # ---------- database ----------

    def db_find(self, db, selector, fields=("_id", "_attachments"), section="null"):
        self._check(db, self.DBS)
        return self._call(
            "POST", f"/db/{db}/_find",
            json={"selector": selector, "fields": list(fields), "section": section},
        )

    def db_get(self, db, doc_id):
        self._check(db, self.DBS)
        return self._call("GET", f"/db/{db}/{doc_id}")

    def db_attachment(self, db, doc_id, name):
        self._check(db, self.DBS)
        return self._call("GET", f"/db/{db}/{doc_id}/{name}", raw=True)

    def db_add_arm(self, db, doc_id, run_id, data, kind="arm_result"):
        self._check(db, self.DBS)
        return self._call(
            "POST", f"/db_record/{db}/{doc_id}/add_arm",
            params={"run_id": run_id, "kind": kind},
            json=data,
        )

    def yolo_set_model(self,json:dict=dict(
            model_name="yolo11l-seg.pt",
            tile_batch_size=6,
            tile_overlap=416,
            
            imgsz=1280, confidence=0.25, iou=0.45,
            max_detections=100)
        ):
        return self._call("POST", "/controllers/yolo/set_model", json=json)
    
    def pcd_set_backend(self,json:dict=dict(
            backend="dnn",
            max_depth_m=2.0)
        ):
        return self._call("POST", "/controllers/pcd/set_backend", json=json)

JST = timezone(timedelta(hours=9))
def current_jst_string() -> str:
    ns = time.time_ns()
    sec = ns // 1_000_000_000
    nanosecond = ns % 1_000_000_000
    dt = datetime.fromtimestamp(sec, JST)
    return dt.strftime("%Y-%m-%d:%H%M%S.") + f"{nanosecond:09d}JST"

  

if __name__ == "__main__":
    with Client(url="http://127.0.0.1:8000") as api:
        print("init",api.refresh(),
                    api.yolo_set_model({"model_name":"yolo11l-seg.pt"}),
                    api.yolo_set_model({"backend":"sgbm","max_depth_m":2.0}))
        
        print("close:", api.close_cams())
        print("dual:", api.open_dual_rgb())

        start = current_jst_string()
        for i in range(10):
            print("dual:", api.capture_dual(meta={
                            "gnss":{"the_data":"xxxxxxxxx"},
                            "arm":{"run_id":"UUIDXXXX","data":{"pose":"xxxxxxxxx"}}}))
        end = current_jst_string()

        class_name,confidence="tie",0.01
        print(f"search {start}->{end}",
              api.db_find(db="dual_rgb",
                    selector={
                        "_id": {"$gte": start.replace(":",":field_all:"),
                                "$lt":  end.replace(":",":field_all:"),
                                "$regex": ":yolo:"},
                        "detections": {
                            "$elemMatch": {
                                "class_name": class_name, "confidence": {"$gte": confidence}
                            }}},
                    fields=["_id"],
                    section="gnss"
              ))
        
        print("close:", api.close_cams())
        print("hand:", api.open_hand())
        print("hand:", api.capture_hand(meta={
                        "gnss":{"the_data":"xxxxxxxxx"},
                        "arm":{"run_id":"UUIDXXXX","data":{"pose":"xxxxxxxxx"}}}))
        
        print("close:", api.close_cams())
