#!/usr/bin/env python3

import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
from statistics import mean

DATA_PATH = "blinkit_full_dataset.csv"
OUTPUT_CSV = "predictions_baseline.csv"
OUTPUT_JSON = "predictions_baseline.json"

# Forecast horizon in days per city-SKU
HORIZON_DAYS = 30

# Seasonal/event boosts (very naive): if a known event date exists, carry a simple uplift
EVENT_UPLIFT_FACTOR = 1.35


def parse_date(date_str):
	return datetime.strptime(date_str, "%Y-%m-%d").date()


def load_rows(csv_path):
	with open(csv_path, newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			# Normalize fields
			row["date"] = parse_date(row["date"]) if row.get("date") else None
			row["inventory_level"] = int(row["inventory_level"]) if row.get("inventory_level") else 0
			row["orders"] = int(row["orders"]) if row.get("orders") else 0
			row["stockout_flag"] = int(row["stockout_flag"]) if row.get("stockout_flag") else 0
			yield row


def group_by_city_sku(rows):
	groups = defaultdict(list)
	for r in rows:
		key = (r["location"], r["sku_id"], r["sku_name"])  # keep name for readability
		groups[key].append(r)
	# sort each time series by date
	for key in groups:
		groups[key].sort(key=lambda r: r["date"])
	return groups


def naive_baseline_forecast(series_rows, horizon_days):
	# Use a simple moving average of last k observed orders as demand baseline
	if not series_rows:
		return []

	window = 3
	orders_history = [r["orders"] for r in series_rows]
	baseline = mean(orders_history[-window:]) if len(orders_history) >= window else mean(orders_history)

	last_date = series_rows[-1]["date"]

	# Extract simple event dates for naive uplift
	event_dates = {r["date"] for r in series_rows if r.get("event") and r["event"].strip()}

	preds = []
	for i in range(1, horizon_days + 1):
		future_date = last_date + timedelta(days=i)
		pred_orders = baseline
		# naive uplift if same day-month seen historically as an event
		if any((e.month, e.day) == (future_date.month, future_date.day) for e in event_dates):
			pred_orders = pred_orders * EVENT_UPLIFT_FACTOR

		preds.append({
			"date": future_date.isoformat(),
			"predicted_orders": round(pred_orders, 2)
		})
	return preds


def predict_stockouts(series_rows, preds):
	# Very naive: if last known inventory minus cumulative predicted orders goes below zero, mark stockout
	if not series_rows:
		return preds
	last_inventory = series_rows[-1]["inventory_level"]
	cum_orders = 0.0
	for p in preds:
		cum_orders += p["predicted_orders"]
		p["projected_inventory"] = round(last_inventory - cum_orders, 2)
		p["stockout_risk"] = 1 if p["projected_inventory"] <= 0 else 0
	return preds


def main():
	rows = list(load_rows(DATA_PATH))
	groups = group_by_city_sku(rows)

	output_rows = []
	output_json = []

	for (city, sku_id, sku_name), series_rows in groups.items():
		preds = naive_baseline_forecast(series_rows, HORIZON_DAYS)
		preds = predict_stockouts(series_rows, preds)
		for p in preds:
			output_rows.append({
				"location": city,
				"sku_id": sku_id,
				"sku_name": sku_name,
				"date": p["date"],
				"predicted_orders": p["predicted_orders"],
				"projected_inventory": p["projected_inventory"],
				"stockout_risk": p["stockout_risk"],
			})
		output_json.append({
			"location": city,
			"sku_id": sku_id,
			"sku_name": sku_name,
			"predictions": preds,
		})

	# Write CSV
	fieldnames = [
		"location",
		"sku_id",
		"sku_name",
		"date",
		"predicted_orders",
		"projected_inventory",
		"stockout_risk",
	]
	with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for r in output_rows:
			writer.writerow(r)

	# Write JSON
	with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
		json.dump(output_json, f, indent=2)

	print(f"Wrote {len(output_rows)} rows to {OUTPUT_CSV}")
	print(f"Wrote JSON groups to {OUTPUT_JSON}")


if __name__ == "__main__":
	main()