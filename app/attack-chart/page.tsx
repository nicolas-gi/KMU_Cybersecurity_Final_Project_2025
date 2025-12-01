'use client';

import { ResponsiveBar } from '@nivo/bar';
import type { CSSProperties } from 'react';

const attackData = [
  { attack: 'DDoS', count: 120 },
  { attack: 'Port Scan', count: 185 },
  { attack: 'Brute Force', count: 50 },
];

const tooltipStyles: CSSProperties = {
  background: '#0f172a',
  color: '#f8fafc',
  padding: '0.4rem 0.75rem',
  borderRadius: '999px',
  fontSize: '0.85rem',
  boxShadow: '0 10px 25px rgba(15, 23, 42, 0.35)',
};

const AttackBarChart = () => (
  <div style={{ height: '360px' }}>
    <ResponsiveBar
      data={attackData}
      keys={['count']}
      indexBy="attack"
      margin={{ top: 30, right: 60, bottom: 70, left: 65 }}
      padding={0.3}
      axisBottom={{
        legend: 'Attack Type',
        legendOffset: 50,
        legendPosition: 'middle',
        tickPadding: 12,
      }}
      axisLeft={{
        legend: 'Detections',
        legendOffset: -55,
        legendPosition: 'middle',
        tickPadding: 8,
      }}
      theme={{
        axis: {
          domain: { line: { stroke: '#475569' } },
          ticks: {
            line: { stroke: '#475569', strokeWidth: 1 },
            text: { fill: '#f8fafc', fontSize: 12 },
          },
          legend: {
            text: { fill: '#f8fafc', fontSize: 13 },
          },
        },
        grid: {
          line: { stroke: '#334155', strokeOpacity: 0.35 },
        },
        tooltip: {
          container: tooltipStyles,
        },
      }}
      colors={['#f97316']}
      borderRadius={4}
      enableLabel={false}
      enableGridY
      motionConfig="gentle"
      tooltip={({ value, indexValue }) => (
        <div style={tooltipStyles}>
          <div style={{ fontWeight: 600 }}>{indexValue}</div>
          <div>{value} detections</div>
        </div>
      )}
    />
  </div>
);

export default function AttackChartPage() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-zinc-50 p-8 text-zinc-900 dark:bg-black dark:text-zinc-50">
      <div className="w-full max-w-3xl rounded-2xl bg-white p-8 shadow dark:bg-zinc-900">
        <h1 className="mb-6 text-3xl font-semibold">Attack Volume Overview</h1>
        <p className="mb-8 text-zinc-600 dark:text-zinc-400">
          This chart shows the volume of common attack types logged by the monitoring pipeline.
        </p>
        <AttackBarChart />
      </div>
    </main>
  );
}
