// Energy Forecasting Dashboard - Main Application

const API_BASE = window.location.origin;

// Global state
let forecastData = null;
let charts = {};

// Initialize app
document.addEventListener("DOMContentLoaded", () => {
  initializeEventListeners();
  checkSystemStatus();
});

// Event Listeners
function initializeEventListeners() {
  // Run forecast button
  document
    .getElementById("runForecastBtn")
    .addEventListener("click", runForecast);

  // Optimize bid button
  document
    .getElementById("optimizeBidBtn")
    .addEventListener("click", optimizeBidding);

  // Tab navigation
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });
}

// API Functions
async function checkSystemStatus() {
  try {
    const response = await fetch(`${API_BASE}/status`);
    const data = await response.json();
    updateStatusIndicator(data.status === "online");
  } catch (error) {
    updateStatusIndicator(false);
  }
}

async function runForecast() {
  showLoading(true);

  try {
    const response = await fetch(`${API_BASE}/forecast/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        train_models: false,
        include_market: true,
        capacity_mw: 500,
        strategy: "optimal",
      }),
    });

    if (!response.ok) throw new Error("Forecast failed");

    const result = await response.json();
    forecastData = result.data;

    console.log("Forecast data received:", forecastData);

    // Update UI
    updateSummaryCards();

    // Small delay to ensure DOM is ready
    setTimeout(() => {
      updateAllCharts();
    }, 100);

    showToast("Forecast completed successfully!", "success");
  } catch (error) {
    showToast("Forecast failed: " + error.message, "error");
  } finally {
    showLoading(false);
  }
}

async function optimizeBidding() {
  if (!forecastData || !forecastData.price) {
    showToast("Please run forecast first", "error");
    return;
  }

  showLoading(true);

  try {
    const capacity = document.getElementById("capacityInput").value;
    const strategy = document.getElementById("strategySelect").value;

    const response = await fetch(`${API_BASE}/bidding/optimize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ capacity_mw: parseFloat(capacity), strategy }),
    });

    if (!response.ok) throw new Error("Optimization failed");

    const result = await response.json();
    forecastData.bid_schedule = result.data.bid_schedule;

    updateBiddingCharts();
    updateBidTable();

    showToast("Bidding optimization completed!", "success");
  } catch (error) {
    showToast("Optimization failed: " + error.message, "error");
  } finally {
    showLoading(false);
  }
}

// UI Update Functions
function updateSummaryCards() {
  if (!forecastData) return;

  // Peak Load
  if (forecastData.load) {
    const peakLoad = Math.max(...forecastData.load);
    document.getElementById("peakLoad").textContent = formatNumber(peakLoad, 0);
  }

  // Renewable Share
  if (forecastData.renewable && forecastData.load) {
    const totalRenewable = forecastData.renewable.reduce(
      (sum, r) => sum + r.total_renewable_mw,
      0,
    );
    const totalLoad = forecastData.load.reduce((sum, l) => sum + l, 0);
    const share = (totalRenewable / totalLoad) * 100;
    document.getElementById("renewableShare").textContent = formatNumber(
      share,
      1,
    );
  }

  // Average Price
  if (forecastData.price) {
    const avgPrice =
      forecastData.price.reduce((sum, p) => sum + p, 0) /
      forecastData.price.length;
    document.getElementById("avgPrice").textContent = formatNumber(avgPrice, 0);
  }

  // Expected Profit
  if (forecastData.bid_schedule) {
    const totalRevenue = forecastData.bid_schedule.reduce(
      (sum, b) => sum + b.expected_revenue,
      0,
    );
    document.getElementById("expectedProfit").textContent = formatNumber(
      totalRevenue,
      0,
    );
  }
}

function updateAllCharts() {
  updateLoadChart();
  updateRenewableChart();
  updatePriceChart();
  updateMCPChart();
  updateWeatherCharts();
  updateBiddingCharts();
  updatePerformanceTab();
}

function updateLoadChart() {
  if (!forecastData || !forecastData.load) return;

  const ctx = document.getElementById("loadChart");
  const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);

  if (charts.load) charts.load.destroy();

  charts.load = new Chart(ctx, {
    type: "line",
    data: {
      labels: hours,
      datasets: [
        {
          label: "Load Forecast (MW)",
          data: forecastData.load,
          borderColor: "#4F46E5",
          backgroundColor: "rgba(79, 70, 229, 0.1)",
          fill: true,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          beginAtZero: false,
          ticks: { callback: (value) => formatNumber(value, 0) },
        },
      },
    },
  });
}

function updateRenewableChart() {
  if (!forecastData || !forecastData.renewable) return;

  const ctx = document.getElementById("renewableChart");
  const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);

  if (charts.renewable) charts.renewable.destroy();

  charts.renewable = new Chart(ctx, {
    type: "bar",
    data: {
      labels: hours,
      datasets: [
        {
          label: "Solar (MW)",
          data: forecastData.renewable.map((r) => r.solar_mw),
          backgroundColor: "#FCD34D",
          stack: "renewable",
        },
        {
          label: "Wind (MW)",
          data: forecastData.renewable.map((r) => r.wind_mw),
          backgroundColor: "#3B82F6",
          stack: "renewable",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        y: {
          stacked: true,
          beginAtZero: true,
          ticks: { callback: (value) => formatNumber(value, 0) },
        },
        x: { stacked: true },
      },
    },
  });
}

function updatePriceChart() {
  if (!forecastData || !forecastData.price) return;

  const ctx = document.getElementById("priceChart");
  const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);

  if (charts.price) charts.price.destroy();

  // Calculate net demand
  const netDemand = forecastData.load.map((load, i) => {
    const renewable = forecastData.renewable[i].total_renewable_mw;
    return load - renewable;
  });

  charts.price = new Chart(ctx, {
    type: "line",
    data: {
      labels: hours,
      datasets: [
        {
          label: "Price (₹/MWh)",
          data: forecastData.price,
          borderColor: "#10B981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          yAxisID: "y",
          tension: 0.4,
        },
        {
          label: "Net Demand (MW)",
          data: netDemand,
          borderColor: "#F59E0B",
          backgroundColor: "rgba(245, 158, 11, 0.1)",
          yAxisID: "y1",
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        y: {
          type: "linear",
          display: true,
          position: "left",
          title: { display: true, text: "Price (₹/MWh)" },
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          title: { display: true, text: "Net Demand (MW)" },
          grid: { drawOnChartArea: false },
        },
      },
    },
  });

  // Update market stats
  updateMarketStats();
}

function updateWeatherCharts() {
  if (!forecastData || !forecastData.weather) return;

  const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);

  // Temperature
  createLineChart("tempChart", hours, [
    {
      label: "Temperature (°C)",
      data: forecastData.weather.map((w) => w.temperature_c),
      borderColor: "#EF4444",
      backgroundColor: "rgba(239, 68, 68, 0.1)",
    },
  ]);

  // Solar Radiation
  createLineChart("solarRadChart", hours, [
    {
      label: "Solar Radiation (W/m²)",
      data: forecastData.weather.map((w) => w.solar_radiation_wm2),
      borderColor: "#FCD34D",
      backgroundColor: "rgba(252, 211, 77, 0.1)",
    },
  ]);

  // Wind Speed
  createLineChart("windChart", hours, [
    {
      label: "Wind Speed (km/h)",
      data: forecastData.weather.map((w) => w.wind_speed_kmh),
      borderColor: "#3B82F6",
      backgroundColor: "rgba(59, 130, 246, 0.1)",
    },
  ]);

  // Humidity & Cloud Cover
  createLineChart("humidityChart", hours, [
    {
      label: "Humidity (%)",
      data: forecastData.weather.map((w) => w.humidity_percent),
      borderColor: "#10B981",
      backgroundColor: "rgba(16, 185, 129, 0.1)",
    },
    {
      label: "Cloud Cover (%)",
      data: forecastData.weather.map((w) => w.cloud_cover_percent),
      borderColor: "#6B7280",
      backgroundColor: "rgba(107, 114, 128, 0.1)",
    },
  ]);
}

function updateBiddingCharts() {
  if (!forecastData || !forecastData.bid_schedule) return;

  const ctx = document.getElementById("bidChart");
  const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);

  if (charts.bid) charts.bid.destroy();

  charts.bid = new Chart(ctx, {
    type: "bar",
    data: {
      labels: hours,
      datasets: [
        {
          label: "Forecast Price",
          data: forecastData.bid_schedule.map((b) => b.forecast_price),
          type: "line",
          borderColor: "#6B7280",
          backgroundColor: "transparent",
          yAxisID: "y",
        },
        {
          label: "Bid Price",
          data: forecastData.bid_schedule.map((b) => b.bid_price),
          backgroundColor: "#4F46E5",
          yAxisID: "y",
        },
        {
          label: "Bid Volume (MW)",
          data: forecastData.bid_schedule.map((b) => b.bid_volume),
          type: "line",
          borderColor: "#10B981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          yAxisID: "y1",
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
      },
      scales: {
        y: {
          type: "linear",
          display: true,
          position: "left",
          title: { display: true, text: "Price (₹/MWh)" },
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          title: { display: true, text: "Volume (MW)" },
          grid: { drawOnChartArea: false },
        },
      },
    },
  });
}

function updateBidTable() {
  if (!forecastData || !forecastData.bid_schedule) return;

  const tbody = document.querySelector("#bidTable tbody");
  tbody.innerHTML = "";

  forecastData.bid_schedule.forEach((bid) => {
    const row = tbody.insertRow();
    row.innerHTML = `
            <td>${bid.hour}:00</td>
            <td>₹${formatNumber(bid.forecast_price, 0)}</td>
            <td>₹${formatNumber(bid.bid_price, 0)}</td>
            <td>${formatNumber(bid.bid_volume, 0)}</td>
            <td>₹${formatNumber(bid.expected_revenue, 0)}</td>
            <td>${formatNumber(bid.acceptance_prob * 100, 1)}%</td>
        `;
  });
}

function updateMCPChart() {
  if (!forecastData || !forecastData.price) return;

  const ctx = document.getElementById("mcpChart");
  if (!ctx) return;

  const hours = Array.from({ length: 24 }, (_, i) => `${i}:00`);

  if (charts.mcp) charts.mcp.destroy();

  charts.mcp = new Chart(ctx, {
    type: "line",
    data: {
      labels: hours,
      datasets: [
        {
          label: "Market Clearing Price (₹/MWh)",
          data: forecastData.price,
          borderColor: "#10B981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          fill: true,
          tension: 0.4,
          borderWidth: 2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          position: "top",
        },
        tooltip: {
          callbacks: {
            label: function (context) {
              return "Price: ₹" + formatNumber(context.parsed.y, 0) + "/MWh";
            },
          },
        },
      },
      scales: {
        y: {
          beginAtZero: false,
          title: {
            display: true,
            text: "Price (₹/MWh)",
          },
          ticks: {
            callback: (value) => "₹" + formatNumber(value, 0),
          },
        },
        x: {
          title: {
            display: true,
            text: "Hour of Day",
          },
        },
      },
    },
  });
}

function updateMarketStats() {
  if (!forecastData || !forecastData.price) return;

  const prices = forecastData.price;
  const peakPrice = Math.max(...prices);
  const minPrice = Math.min(...prices);
  const avgPrice = prices.reduce((sum, p) => sum + p, 0) / prices.length;
  const volatility = calculateStdDev(prices);

  const statsHtml = `
        <div class="stat-item">
            <div class="stat-label">Peak Price</div>
            <div class="stat-value">₹${formatNumber(peakPrice, 0)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Min Price</div>
            <div class="stat-value">₹${formatNumber(minPrice, 0)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Average</div>
            <div class="stat-value">₹${formatNumber(avgPrice, 0)}</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Volatility</div>
            <div class="stat-value">₹${formatNumber(volatility, 0)}</div>
        </div>
    `;

  document.getElementById("priceStats").innerHTML = statsHtml;

  // Net demand stats
  const netDemand = forecastData.load.map((load, i) => {
    const renewable = forecastData.renewable[i].total_renewable_mw;
    return load - renewable;
  });

  const peakNetDemand = Math.max(...netDemand);
  const minNetDemand = Math.min(...netDemand);
  const avgNetDemand =
    netDemand.reduce((sum, d) => sum + d, 0) / netDemand.length;
  const totalRenewable = forecastData.renewable.reduce(
    (sum, r) => sum + r.total_renewable_mw,
    0,
  );
  const totalLoad = forecastData.load.reduce((sum, l) => sum + l, 0);
  const rePenetration = (totalRenewable / totalLoad) * 100;

  const netDemandHtml = `
        <div class="stat-item">
            <div class="stat-label">Peak Net Demand</div>
            <div class="stat-value">${formatNumber(peakNetDemand, 0)} MW</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Min Net Demand</div>
            <div class="stat-value">${formatNumber(minNetDemand, 0)} MW</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">Average</div>
            <div class="stat-value">${formatNumber(avgNetDemand, 0)} MW</div>
        </div>
        <div class="stat-item">
            <div class="stat-label">RE Penetration</div>
            <div class="stat-value">${formatNumber(rePenetration, 1)}%</div>
        </div>
    `;

  document.getElementById("netDemandStats").innerHTML = netDemandHtml;
}

function updatePerformanceTab() {
  if (!forecastData || !forecastData.simulation) return;

  const simulation = forecastData.simulation;

  // Update strategy comparison chart
  if (simulation.strategy_comparison) {
    updateStrategyChart(simulation.strategy_comparison);
  }

  // Update performance metrics
  if (simulation.strategy_comparison) {
    updatePerformanceMetrics(simulation.strategy_comparison);
  }
}

function updateStrategyChart(strategyData) {
  const ctx = document.getElementById("strategyChart");
  if (!ctx) return;

  if (charts.strategy) charts.strategy.destroy();

  const strategies = strategyData.map((s) => s.strategy.toUpperCase());
  const profits = strategyData.map((s) => s.total_profit);
  const acceptanceRates = strategyData.map((s) => s.avg_acceptance_rate * 100);

  charts.strategy = new Chart(ctx, {
    type: "bar",
    data: {
      labels: strategies,
      datasets: [
        {
          label: "Total Profit (₹)",
          data: profits,
          backgroundColor: "rgba(79, 70, 229, 0.8)",
          yAxisID: "y",
        },
        {
          label: "Acceptance Rate (%)",
          data: acceptanceRates,
          type: "line",
          borderColor: "#10B981",
          backgroundColor: "rgba(16, 185, 129, 0.1)",
          yAxisID: "y1",
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
        tooltip: {
          callbacks: {
            label: function (context) {
              if (context.dataset.label.includes("Profit")) {
                return "Profit: ₹" + formatNumber(context.parsed.y, 0);
              } else {
                return "Acceptance: " + formatNumber(context.parsed.y, 1) + "%";
              }
            },
          },
        },
      },
      scales: {
        y: {
          type: "linear",
          display: true,
          position: "left",
          title: { display: true, text: "Profit (₹)" },
          ticks: {
            callback: (value) => "₹" + formatNumber(value / 1000000, 0) + "M",
          },
        },
        y1: {
          type: "linear",
          display: true,
          position: "right",
          title: { display: true, text: "Acceptance Rate (%)" },
          grid: { drawOnChartArea: false },
          min: 0,
          max: 100,
        },
      },
    },
  });
}

function updatePerformanceMetrics(strategyData) {
  const container = document.getElementById("performanceMetrics");
  if (!container) return;

  let html = '<div class="metrics-container">';

  strategyData.forEach((strategy, index) => {
    const isWinner = index === 0;
    const borderColor = isWinner ? "#10B981" : "#4F46E5";

    html += `
      <div class="metric-card" style="border-left-color: ${borderColor}">
        <div class="metric-label">${strategy.strategy.toUpperCase()}${isWinner ? " 🏆" : ""}</div>
        <div class="metric-value">₹${formatNumber(strategy.total_profit / 10000000, 2)}Cr</div>
        <div class="metric-change">
          <div style="font-size: 13px; color: #6B7280; margin-top: 8px;">
            <div>Acceptance: ${formatNumber(strategy.avg_acceptance_rate * 100, 1)}%</div>
            <div>Energy: ${formatNumber(strategy.total_energy / 1000, 0)}k MWh</div>
            <div>Profit/MWh: ₹${formatNumber(strategy.profit_per_mwh, 0)}</div>
          </div>
        </div>
      </div>
    `;
  });

  html += "</div>";

  // Add best strategy recommendation
  const best = strategyData[0];
  html += `
    <div style="margin-top: 24px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; color: white;">
      <h3 style="margin: 0 0 12px 0; font-size: 18px;">Recommended Strategy</h3>
      <p style="margin: 0; font-size: 16px; opacity: 0.9;">
        <strong>${best.strategy.toUpperCase()}</strong> strategy is optimal with 
        <strong>₹${formatNumber(best.total_profit / 10000000, 2)} Crore</strong> expected profit 
        and <strong>${formatNumber(best.avg_acceptance_rate * 100, 1)}%</strong> acceptance rate.
      </p>
    </div>
  `;

  container.innerHTML = html;
}

// Helper Functions
function createLineChart(canvasId, labels, datasets) {
  const ctx = document.getElementById(canvasId);
  if (charts[canvasId]) charts[canvasId].destroy();

  charts[canvasId] = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: datasets.map((ds) => ({ ...ds, fill: true, tension: 0.4 })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { position: "top" } },
      scales: { y: { beginAtZero: false } },
    },
  });
}

function switchTab(tabName) {
  // Update tab buttons
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.tab === tabName);
  });

  // Update tab panes
  document.querySelectorAll(".tab-pane").forEach((pane) => {
    pane.classList.toggle("active", pane.id === tabName);
  });
}

function showLoading(show) {
  document.getElementById("loadingOverlay").classList.toggle("active", show);
}

function updateStatusIndicator(online) {
  const indicator = document.getElementById("statusIndicator");
  const dot = indicator.querySelector(".status-dot");
  const text = indicator.querySelector(".status-text");

  if (online) {
    dot.style.background = "#10B981";
    text.textContent = "Online";
  } else {
    dot.style.background = "#EF4444";
    text.textContent = "Offline";
  }
}

function showToast(message, type = "info") {
  const container = document.getElementById("toastContainer");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.innerHTML = `<div class="toast-message">${message}</div>`;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = "slideOut 0.3s ease-out";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

function formatNumber(value, decimals = 0) {
  return new Intl.NumberFormat("en-IN", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
}

function calculateStdDev(values) {
  const avg = values.reduce((sum, v) => sum + v, 0) / values.length;
  const squareDiffs = values.map((v) => Math.pow(v - avg, 2));
  const avgSquareDiff =
    squareDiffs.reduce((sum, v) => sum + v, 0) / values.length;
  return Math.sqrt(avgSquareDiff);
}
