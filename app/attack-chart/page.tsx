'use client';

import type { CSSProperties } from 'react';
import { ResponsiveLine } from '@nivo/line';

const attackData = [
  {
    id: 'Attacks',
    data: [
      { x: 'DDoS', y: 120 },
      { x: 'Port Scan', y: 185 },
      { x: 'Brute Force', y: 50 },
    ],
  },
];

const tooltipStyles: CSSProperties = {
  background: '#0f172a',
  color: '#f8fafc',
  padding: '0.4rem 0.75rem',
  borderRadius: '999px',
  fontSize: '0.85rem',
  boxShadow: '0 10px 25px rgba(15, 23, 42, 0.35)',
};

const AttackLineChart = () => (
  <div style={{ height: '360px' }}>
    <ResponsiveLine
      data={attackData}
      margin={{ top: 30, right: 60, bottom: 70, left: 65 }}
      xScale={{ type: 'point' }}
      yScale={{ type: 'linear', min: 0, max: 'auto', stacked: false, reverse: false }}
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
      enableArea
      areaOpacity={0.15}
      defs={[
        {
          id: 'attackGradient',
          type: 'linearGradient',
          colors: [
            { offset: 0, color: '#f97316' },
            { offset: 100, color: '#fb923c' },
          ],
        },
      ]}
      fill={[{ match: { id: 'Attacks' }, id: 'attackGradient' }]}
      pointSize={10}
      pointColor={{ theme: 'background' }}
      pointBorderWidth={3}
      pointBorderColor={{ from: 'serieColor' }}
      pointLabelYOffset={-12}
      enableGridX={false}
      enableGridY
      crosshairType="x"
      enableSlices="x"
      useMesh
      curve="monotoneX"
      motionConfig="gentle"
      tooltip={({ point }) => {
        const category = (point.data.xFormatted ?? point.data.x) as string;
        const count = point.data.yFormatted ?? point.data.y;
        return (
          <div style={tooltipStyles}>
            <div style={{ fontWeight: 600 }}>{category}</div>
            <div>{count} detections</div>
          </div>
        );
      }}
      legends={[
        {
          anchor: 'bottom-right',
          direction: 'column',
          justify: false,
          translateX: 80,
          translateY: 0,
          itemsSpacing: 8,
          itemDirection: 'left-to-right',
          itemWidth: 80,
          itemHeight: 20,
          itemOpacity: 0.75,
          symbolSize: 12,
          symbolShape: 'circle',
          symbolBorderColor: 'rgba(0, 0, 0, .5)',
          effects: [
            {
              on: 'hover',
              style: {
                itemBackground: 'rgba(0, 0, 0, .03)',
                itemOpacity: 1,
              },
            },
          ],
        },
      ]}
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
        <AttackLineChart />
      </div>
    </main>
  );
}
