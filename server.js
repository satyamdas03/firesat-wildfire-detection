const express = require('express');
const cors = require('cors');
const http = require('http');
const { Server } = require("socket.io");
const path = require('path');

// ------------------------------------------------------------------
// SATELLITE GROUND STATION: WILDFIRE RECEPTION DESK
// ------------------------------------------------------------------

const app = express();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

let latestAlerts = [];

// This is the webhook endpoint the Edge AI (Python Simulator) hits
app.post('/api/alert', (req, res) => {
    const data = req.body;
    
    const country = data.country || 'Unknown';
    const frp = data.fire_intensity || '?';
    
    console.log(`\n[GROUND STATION] 🚨 ALERT | ${country} | FRP: ${frp}MW | Conf: ${(data.confidence * 100).toFixed(1)}%`);

    // Keep buffer of last 500 alerts
    latestAlerts.unshift(data);
    if(latestAlerts.length > 500) latestAlerts.pop();

    // Broadcast the alert immediately to all connected browsers
    io.emit('new_fire_alert', data);

    res.status(200).json({ status: "Signal Received. Resources dispatched." });
});

// Return alert history for fresh page loads
app.get('/api/history', (req, res) => {
    res.json(latestAlerts);
});

// Regional stats for the sidebar panel
app.get('/api/stats', (req, res) => {
    const regionMap = {};
    latestAlerts.forEach(a => {
        const cc = (a.country || 'Unknown').split(',').pop().trim();
        regionMap[cc] = (regionMap[cc] || 0) + 1;
    });

    const sorted = Object.entries(regionMap)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10)
        .map(([region, count]) => ({ region, count }));

    res.json({ total: latestAlerts.length, regions: sorted });
});

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

io.on('connection', (socket) => {
    console.log('[DASHBOARD] New Command Center connected.');
    // Send recent history on connect (last 50 to avoid overwhelming)
    socket.emit('alert_history', latestAlerts.slice(0, 50));
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`📡 Ground Station active on port ${PORT}`);
    console.log(`🌐 Dashboard: http://localhost:${PORT}`);
});
