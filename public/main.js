async function fetchJSON(url) {
	const res = await fetch(url);
	if (!res.ok) throw new Error(`HTTP ${res.status}`);
	return res.json();
}

function groupLabel(g) {
	return `${g.location} — ${g.sku_name} (${g.sku_id})`;
}

function toSeries(predictions) {
	const dates = predictions.map(p => p.date);
	const orders = predictions.map(p => p.predicted_orders);
	const inventory = predictions.map(p => p.projected_inventory);
	const risk = predictions.map(p => p.stockout_risk);
	return { dates, orders, inventory, risk };
}

function renderChart(title, series) {
	const traces = [
		{
			x: series.dates,
			y: series.orders,
			type: 'scatter', mode: 'lines+markers', name: 'Predicted Orders',
			line: { color: '#2563eb' }
		},
		{
			x: series.dates,
			y: series.inventory,
			type: 'scatter', mode: 'lines+markers', name: 'Projected Inventory',
			line: { color: '#059669' }
		},
		{
			x: series.dates,
			y: series.risk.map(v => v ? 1 : 0),
			type: 'bar', name: 'Stockout Risk', yaxis: 'y2', marker: { color: '#ef4444' }, opacity: 0.3
		}
	];

	const layout = {
		title,
		plot_bgcolor: '#fff', paper_bgcolor: '#fff',
		yaxis: { title: 'Units' },
		yaxis2: { overlaying: 'y', side: 'right', range: [0, 1], showgrid: false, title: 'Risk' },
		margin: { l: 48, r: 48, t: 48, b: 48 }
	};

	Plotly.newPlot('chart', traces, layout, { responsive: true, displayModeBar: false });
}

async function populateSummary() {
	const tbody = document.querySelector('#summaryTable tbody');
	tbody.innerHTML = '';
	const { summary } = await fetchJSON('/api/summary');
	summary.forEach(row => {
		const tr = document.createElement('tr');
		tr.innerHTML = `
			<td>${row.location}</td>
			<td>${row.sku_name} (${row.sku_id})</td>
			<td>${row.first_stockout_date || '—'}</td>
		`;
		tbody.appendChild(tr);
	});
}

async function init() {
	const groupSelect = document.getElementById('groupSelect');
	const { groups } = await fetchJSON('/api/groups');
	groups.forEach(g => {
		const opt = document.createElement('option');
		opt.value = `${g.location}|||${g.sku_id}`;
		opt.textContent = groupLabel(g);
		groupSelect.appendChild(opt);
	});

	async function loadSelection() {
		const [location, sku_id] = groupSelect.value.split('|||');
		const data = await fetchJSON(`/api/predictions?location=${encodeURIComponent(location)}&sku_id=${encodeURIComponent(sku_id)}`);
		const series = toSeries(data.predictions);
		renderChart(`${data.location} — ${data.sku_name}`, series);
	}

	groupSelect.addEventListener('change', loadSelection);
	if (groupSelect.options.length) {
		groupSelect.selectedIndex = 0;
		await loadSelection();
	}

	await populateSummary();

	document.getElementById('exportPng').addEventListener('click', () => {
		Plotly.downloadImage('chart', { format: 'png', width: 1200, height: 700, filename: 'blinkit-inventory-forecast' });
	});
}

window.addEventListener('DOMContentLoaded', init);