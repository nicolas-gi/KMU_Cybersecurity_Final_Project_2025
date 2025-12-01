/* eslint-disable @typescript-eslint/no-explicit-any */
'use client';

import { ResponsiveBar } from '@nivo/bar';
import { ResponsiveLine } from '@nivo/line';
import { useCallback, useEffect, useState } from 'react';
interface TrafficSample {
    timestamp: number;
    duration: number;
    src_bytes: number;
    dst_bytes: number;
    count: number;
    srv_count: number;
    serror_rate: number;
    rerror_rate: number;
    same_srv_rate: number;
    diff_srv_rate: number;
}

interface PredictionResult {
    is_anomaly: boolean;
    confidence: number;
    threat_level: 'normal' | 'medium' | 'high' | 'critical';
    prediction: string;
}

interface Alert {
    id: string;
    timestamp: number;
    threat_level: string;
    confidence: number;
    message: string;
}

export default function MonitoringDashboard() {
    const [isMonitoring, setIsMonitoring] = useState(false);
    const [mlServiceStatus, setMlServiceStatus] = useState<'checking' | 'healthy' | 'unhealthy'>('checking');
    const [trafficData, setTrafficData] = useState<TrafficSample[]>([]);
    const [alerts, setAlerts] = useState<Alert[]>([]);
    const [stats, setStats] = useState({
        totalSamples: 0,
        normalTraffic: 0,
        anomalies: 0,
        criticalAlerts: 0,
    });

    // Check ML service health
    useEffect(() => {
        const checkHealth = async () => {
            try {
                const response = await fetch('/api/ml-health');
                const data = await response.json();
                setMlServiceStatus(data.status === 'healthy' ? 'healthy' : 'unhealthy');
            } catch {
                setMlServiceStatus('unhealthy');
            }
        };

        checkHealth();
        const interval = setInterval(checkHealth, 30000); // Check every 30s
        return () => clearInterval(interval);
    }, []);

    // Generate mock traffic sample
    const generateTrafficSample = useCallback((): TrafficSample => {
        const isAnomaly = Math.random() < 0.15; // 15% anomaly rate

        if (isAnomaly) {
            // Generate attack-like traffic
            const attackType = Math.random();
            if (attackType < 0.4) {
                // DDoS pattern
                return {
                    timestamp: Date.now(),
                    duration: Math.random() * 0.5,
                    src_bytes: 50 + Math.random() * 100,
                    dst_bytes: 20 + Math.random() * 60,
                    count: 40 + Math.random() * 80,
                    srv_count: 30 + Math.random() * 60,
                    serror_rate: 0.6 + Math.random() * 0.3,
                    rerror_rate: 0.1 + Math.random() * 0.2,
                    same_srv_rate: 0.1 + Math.random() * 0.3,
                    diff_srv_rate: 0.6 + Math.random() * 0.3,
                };
            } else {
                // Port scan pattern
                return {
                    timestamp: Date.now(),
                    duration: Math.random() * 2,
                    src_bytes: 30 + Math.random() * 50,
                    dst_bytes: 10 + Math.random() * 30,
                    count: 80 + Math.random() * 100,
                    srv_count: 60 + Math.random() * 80,
                    serror_rate: 0.3 + Math.random() * 0.4,
                    rerror_rate: 0.3 + Math.random() * 0.4,
                    same_srv_rate: 0.05 + Math.random() * 0.15,
                    diff_srv_rate: 0.7 + Math.random() * 0.25,
                };
            }
        } else {
            // Normal traffic
            return {
                timestamp: Date.now(),
                duration: 1 + Math.random() * 3,
                src_bytes: 300 + Math.random() * 400,
                dst_bytes: 200 + Math.random() * 300,
                count: 3 + Math.random() * 7,
                srv_count: 2 + Math.random() * 4,
                serror_rate: Math.random() * 0.2,
                rerror_rate: Math.random() * 0.15,
                same_srv_rate: 0.6 + Math.random() * 0.3,
                diff_srv_rate: 0.1 + Math.random() * 0.2,
            };
        }
    }, []);

    // Monitor traffic
    useEffect(() => {
        if (!isMonitoring || mlServiceStatus !== 'healthy') return;

        const interval = setInterval(async () => {
            const sample = generateTrafficSample();

            try {
                // Get prediction from ML service
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(sample),
                });

                const prediction: PredictionResult = await response.json();

                // Update stats
                setStats((prev) => ({
                    totalSamples: prev.totalSamples + 1,
                    normalTraffic: prev.normalTraffic + (prediction.is_anomaly ? 0 : 1),
                    anomalies: prev.anomalies + (prediction.is_anomaly ? 1 : 0),
                    criticalAlerts: prev.criticalAlerts + (prediction.threat_level === 'critical' ? 1 : 0),
                }));

                // Add to traffic data (keep last 50 samples)
                setTrafficData((prev) => [...prev.slice(-49), sample]);

                // Generate alert if anomaly detected
                if (prediction.is_anomaly) {
                    const alert: Alert = {
                        id: `${Date.now()}-${Math.random()}`,
                        timestamp: Date.now(),
                        threat_level: prediction.threat_level,
                        confidence: prediction.confidence,
                        message: `${prediction.threat_level.toUpperCase()} threat detected with ${(prediction.confidence * 100).toFixed(1)}% confidence`,
                    };
                    setAlerts((prev) => [alert, ...prev.slice(0, 19)]); // Keep last 20 alerts
                }
            } catch (error) {
                console.error('Prediction error:', error);
            }
        }, 2000); // Check every 2 seconds

        return () => clearInterval(interval);
    }, [isMonitoring, mlServiceStatus, generateTrafficSample]);

    // Prepare chart data
    const threatDistribution = [
        { category: 'Normal', count: stats.normalTraffic, color: '#10b981' },
        { category: 'Anomalies', count: stats.anomalies, color: '#ef4444' },
    ];

    const timeSeriesData = [
        {
            id: 'Traffic Volume',
            data: trafficData.slice(-20).map((sample, idx) => ({
                x: idx,
                y: sample.count,
            })),
        },
    ];

    return (
        <main className="min-h-screen bg-zinc-50 dark:bg-black text-zinc-900 dark:text-zinc-50 p-4 md:p-8">
            <div className="max-w-7xl mx-auto space-y-6">
                {/* Header */}
                <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
                    <div>
                        <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent">
                            Network Anomaly Detection
                        </h1>
                        <p className="text-zinc-600 dark:text-zinc-400 mt-2">
                            Real-time ML-powered intrusion detection system
                        </p>
                    </div>

                    <div className="flex items-center gap-4">
                        {/* ML Service Status */}
                        <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
                            <div
                                className={`w-2 h-2 rounded-full ${mlServiceStatus === 'healthy'
                                        ? 'bg-green-500'
                                        : mlServiceStatus === 'unhealthy'
                                            ? 'bg-red-500'
                                            : 'bg-yellow-500'
                                    }`}
                            />
                            <span className="text-sm font-medium">
                                ML Service: {mlServiceStatus}
                            </span>
                        </div>

                        {/* Monitoring Toggle */}
                        <button
                            onClick={() => setIsMonitoring(!isMonitoring)}
                            disabled={mlServiceStatus !== 'healthy'}
                            className={`px-6 py-2 rounded-lg font-semibold transition-colors ${isMonitoring
                                    ? 'bg-red-500 hover:bg-red-600 text-white'
                                    : 'bg-green-500 hover:bg-green-600 text-white'
                                } disabled:opacity-50 disabled:cursor-not-allowed`}
                        >
                            {isMonitoring ? '⏸ Stop Monitoring' : '▶ Start Monitoring'}
                        </button>
                    </div>
                </div>

                {/* Stats Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="p-6 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
                        <div className="text-sm text-zinc-600 dark:text-zinc-400 mb-1">
                            Total Samples
                        </div>
                        <div className="text-3xl font-bold text-blue-500">
                            {stats.totalSamples.toLocaleString()}
                        </div>
                    </div>

                    <div className="p-6 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
                        <div className="text-sm text-zinc-600 dark:text-zinc-400 mb-1">
                            Normal Traffic
                        </div>
                        <div className="text-3xl font-bold text-green-500">
                            {stats.normalTraffic.toLocaleString()}
                        </div>
                    </div>

                    <div className="p-6 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
                        <div className="text-sm text-zinc-600 dark:text-zinc-400 mb-1">
                            Anomalies Detected
                        </div>
                        <div className="text-3xl font-bold text-orange-500">
                            {stats.anomalies.toLocaleString()}
                        </div>
                    </div>

                    <div className="p-6 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
                        <div className="text-sm text-zinc-600 dark:text-zinc-400 mb-1">
                            Critical Alerts
                        </div>
                        <div className="text-3xl font-bold text-red-500">
                            {stats.criticalAlerts.toLocaleString()}
                        </div>
                    </div>
                </div>

                {/* Charts */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Traffic Distribution */}
                    <div className="p-6 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
                        <h2 className="text-xl font-semibold mb-4">Traffic Distribution</h2>
                        <div style={{ height: '300px' }}>
                            <ResponsiveBar
                                data={threatDistribution as any}
                                keys={['count']}
                                indexBy="category"
                                margin={{ top: 20, right: 20, bottom: 50, left: 60 }}
                                padding={0.3}
                                colors={({ data }) => (data as any).color}
                                axisBottom={{
                                    tickSize: 5,
                                    tickPadding: 5,
                                    tickRotation: 0,
                                }}
                                axisLeft={{
                                    tickSize: 5,
                                    tickPadding: 5,
                                    tickRotation: 0,
                                }}
                                labelTextColor="#ffffff"
                                theme={{
                                    axis: {
                                        ticks: {
                                            text: { fill: '#94a3b8' },
                                        },
                                    },
                                    grid: {
                                        line: { stroke: '#334155', strokeOpacity: 0.2 },
                                    },
                                }}
                            />
                        </div>
                    </div>

                    {/* Real-time Traffic Volume */}
                    <div className="p-6 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
                        <h2 className="text-xl font-semibold mb-4">Real-time Traffic Volume</h2>
                        <div style={{ height: '300px' }}>
                            <ResponsiveLine
                                data={timeSeriesData as any}
                                margin={{ top: 20, right: 20, bottom: 50, left: 60 }}
                                xScale={{ type: 'linear' }}
                                yScale={{ type: 'linear', min: 0, max: 'auto' }}
                                axisBottom={{
                                    tickSize: 5,
                                    tickPadding: 5,
                                    tickRotation: 0,
                                    legend: 'Time',
                                    legendOffset: 36,
                                    legendPosition: 'middle',
                                }}
                                axisLeft={{
                                    tickSize: 5,
                                    tickPadding: 5,
                                    tickRotation: 0,
                                    legend: 'Connections',
                                    legendOffset: -50,
                                    legendPosition: 'middle',
                                }}
                                colors={['#3b82f6']}
                                pointSize={8}
                                pointColor={{ theme: 'background' }}
                                pointBorderWidth={2}
                                pointBorderColor={{ from: 'serieColor' }}
                                useMesh={true}
                                enableArea={true}
                                areaOpacity={0.1}
                                theme={{
                                    axis: {
                                        ticks: {
                                            text: { fill: '#94a3b8' },
                                        },
                                        legend: {
                                            text: { fill: '#94a3b8' },
                                        },
                                    },
                                    grid: {
                                        line: { stroke: '#334155', strokeOpacity: 0.2 },
                                    },
                                }}
                            />
                        </div>
                    </div>
                </div>

                {/* Alerts Feed */}
                <div className="p-6 rounded-xl bg-white dark:bg-zinc-900 border border-zinc-200 dark:border-zinc-800">
                    <h2 className="text-xl font-semibold mb-4">Security Alerts</h2>
                    <div className="space-y-2 max-h-96 overflow-y-auto">
                        {alerts.length === 0 ? (
                            <div className="text-center py-8 text-zinc-500">
                                No alerts yet. Start monitoring to detect anomalies.
                            </div>
                        ) : (
                            alerts.map((alert) => (
                                <div
                                    key={alert.id}
                                    className={`p-4 rounded-lg border-l-4 ${alert.threat_level === 'critical'
                                            ? 'bg-red-500/10 border-red-500'
                                            : alert.threat_level === 'high'
                                                ? 'bg-orange-500/10 border-orange-500'
                                                : 'bg-yellow-500/10 border-yellow-500'
                                        }`}
                                >
                                    <div className="flex justify-between items-start">
                                        <div>
                                            <div className="font-semibold text-sm mb-1">
                                                {alert.message}
                                            </div>
                                            <div className="text-xs text-zinc-500">
                                                {new Date(alert.timestamp).toLocaleTimeString()}
                                            </div>
                                        </div>
                                        <div
                                            className={`px-2 py-1 rounded text-xs font-semibold ${alert.threat_level === 'critical'
                                                    ? 'bg-red-500 text-white'
                                                    : alert.threat_level === 'high'
                                                        ? 'bg-orange-500 text-white'
                                                        : 'bg-yellow-500 text-black'
                                                }`}
                                        >
                                            {alert.threat_level.toUpperCase()}
                                        </div>
                                    </div>
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>
        </main>
    );
}
