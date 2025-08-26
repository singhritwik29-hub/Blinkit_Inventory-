#!/usr/bin/env python3

import json
import urllib.parse
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT_DIR = Path(__file__).parent
PUBLIC_DIR = ROOT_DIR / "public"
PRED_JSON_PATH = ROOT_DIR / "predictions_baseline.json"

# Load predictions into memory
if not PRED_JSON_PATH.exists():
	raise SystemExit(
		"predictions_baseline.json not found. Run predict_baseline.py first to generate predictions."
	)

with PRED_JSON_PATH.open("r", encoding="utf-8") as f:
	PREDICTION_GROUPS = json.load(f)

# Build quick lookup indices
GROUP_INDEX = {}
GROUP_LIST = []
for g in PREDICTION_GROUPS:
	key = (g.get("location"), g.get("sku_id"))
	GROUP_INDEX[key] = g
	GROUP_LIST.append({
		"location": g.get("location"),
		"sku_id": g.get("sku_id"),
		"sku_name": g.get("sku_name"),
	})


def compute_summary(groups):
	summary = []
	for g in groups:
		first_stockout_date = None
		for p in g.get("predictions", []):
			if p.get("stockout_risk") == 1:
				first_stockout_date = p.get("date")
				break
		summary.append({
			"location": g.get("location"),
			"sku_id": g.get("sku_id"),
			"sku_name": g.get("sku_name"),
			"first_stockout_date": first_stockout_date,
		})
	return summary

SUMMARY = compute_summary(PREDICTION_GROUPS)


class Handler(SimpleHTTPRequestHandler):
	def __init__(self, *args, **kwargs):
		# Serve files from PUBLIC_DIR by default
		super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

	def _send_json(self, payload, status=HTTPStatus.OK):
		data = json.dumps(payload).encode("utf-8")
		self.send_response(status)
		self.send_header("Content-Type", "application/json; charset=utf-8")
		self.send_header("Content-Length", str(len(data)))
		self.end_headers()
		self.wfile.write(data)

	def do_GET(self):
		parsed = urllib.parse.urlparse(self.path)
		if parsed.path == "/api/groups":
			return self._send_json({"groups": GROUP_LIST})
		elif parsed.path == "/api/predictions":
			qs = urllib.parse.parse_qs(parsed.query)
			location = (qs.get("location") or [None])[0]
			sku_id = (qs.get("sku_id") or [None])[0]
			key = (location, sku_id)
			group = GROUP_INDEX.get(key)
			if not group:
				return self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
			return self._send_json({
				"location": group.get("location"),
				"sku_id": group.get("sku_id"),
				"sku_name": group.get("sku_name"),
				"predictions": group.get("predictions", []),
			})
		elif parsed.path == "/api/summary":
			return self._send_json({"summary": SUMMARY})
		# Serve raw prediction files for download
		elif parsed.path == "/predictions.json":
			return self._send_json(PREDICTION_GROUPS)
		elif parsed.path == "/predictions.csv":
			# Stream the CSV file from root dir
			csv_path = ROOT_DIR / "predictions_baseline.csv"
			if not csv_path.exists():
				self.send_error(HTTPStatus.NOT_FOUND, "CSV not found")
				return
			data = csv_path.read_bytes()
			self.send_response(HTTPStatus.OK)
			self.send_header("Content-Type", "text/csv; charset=utf-8")
			self.send_header("Content-Length", str(len(data)))
			self.send_header("Content-Disposition", "attachment; filename=predictions_baseline.csv")
			self.end_headers()
			self.wfile.write(data)
		else:
			# Fallback to static serving
			return super().do_GET()


def run(host="0.0.0.0", port=8000):
	server = ThreadingHTTPServer((host, port), Handler)
	print(f"Serving on http://{host}:{port}")
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		pass
	finally:
		server.server_close()


if __name__ == "__main__":
	run()