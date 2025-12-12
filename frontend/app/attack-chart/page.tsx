'use client';

import { ResponsiveBar } from '@nivo/bar';
import type { CSSProperties } from 'react';
import { useState } from 'react';

// Attack types categorized from the CSV
const attackCategories = {
  dos: ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop'],
  probe: ['ipsweep', 'nmap', 'portsweep', 'satan'],
  r2l: ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezclient', 'warezmaster'],
  u2r: ['buffer_overflow', 'loadmodule', 'perl', 'rootkit'],
  normal: ['normal'],
};

// Mock detection counts (in production, this would come from real monitoring data)
const generateAttackData = () => {
  const data = [
    { category: 'DoS', count: 3420, color: '#ef4444', attacks: attackCategories.dos },
    { category: 'Probe', count: 1850, color: '#f59e0b', attacks: attackCategories.probe },
    { category: 'R2L', count: 980, color: '#8b5cf6', attacks: attackCategories.r2l },
    { category: 'U2R', count: 340, color: '#ec4899', attacks: attackCategories.u2r },
    { category: 'Normal', count: 15200, color: '#10b981', attacks: attackCategories.normal },
  ];
  return data;
};

const tooltipStyles: CSSProperties = {
  background: '#0f172a',
  color: '#f8fafc',
  padding: '0.75rem 1rem',
  borderRadius: '0.5rem',
  fontSize: '0.875rem',
  boxShadow: '0 10px 25px rgba(15, 23, 42, 0.35)',
  maxWidth: '300px',
};

type AttackDatum = {
  category: string;
  count: number;
  color: string;
  attacks: string[];
  [key: string]: string | number | string[];
};

const AttackBarChart = ({ data }: { data: AttackDatum[] }) => (
  <div style={{ height: '400px' }}>
    <ResponsiveBar
      data={data as never}
      keys={['count']}
      indexBy="category"
      margin={{ top: 30, right: 60, bottom: 70, left: 80 }}
      padding={0.3}
      axisBottom={{
        legend: 'Attack Category',
        legendOffset: 50,
        legendPosition: 'middle',
        tickPadding: 12,
      }}
      axisLeft={{
        legend: 'Detection Count',
        legendOffset: -65,
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
            text: { fill: '#f8fafc', fontSize: 14, fontWeight: 600 },
          },
        },
        grid: {
          line: { stroke: '#334155', strokeOpacity: 0.35 },
        },
      }}
      colors={({ data }) => (data as unknown as AttackDatum).color}
      borderRadius={6}
      enableLabel={true}
      label={(d) => `${(d.value ?? 0).toLocaleString()}`}
      labelSkipWidth={12}
      labelSkipHeight={12}
      labelTextColor="#ffffff"
      enableGridY
      motionConfig="gentle"
      tooltip={({ data, value }) => {
        const attackData = data as unknown as AttackDatum;
        return (
          <div style={tooltipStyles}>
            <div style={{ fontWeight: 700, fontSize: '1rem', marginBottom: '0.5rem' }}>
              {attackData.category}
            </div>
            <div style={{ marginBottom: '0.5rem', color: '#94a3b8' }}>
              <strong>{value.toLocaleString()}</strong> detections
            </div>
            <div style={{ fontSize: '0.75rem', color: '#cbd5e1' }}>
              <strong>Attack Types:</strong>
              <div style={{ marginTop: '0.25rem', lineHeight: '1.4' }}>
                {attackData.attacks.join(', ')}
              </div>
            </div>
          </div>
        );
      }}
    />
  </div>
);

const CategoryLegend = () => (
  <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
    <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-3 h-3 rounded-full bg-red-500"></div>
        <h3 className="font-semibold text-sm">DoS</h3>
      </div>
      <p className="text-xs text-zinc-600 dark:text-zinc-400">Denial of Service attacks</p>
    </div>
    <div className="p-4 rounded-lg bg-amber-500/10 border border-amber-500/20">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-3 h-3 rounded-full bg-amber-500"></div>
        <h3 className="font-semibold text-sm">Probe</h3>
      </div>
      <p className="text-xs text-zinc-600 dark:text-zinc-400">Surveillance & reconnaissance</p>
    </div>
    <div className="p-4 rounded-lg bg-violet-500/10 border border-violet-500/20">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-3 h-3 rounded-full bg-violet-500"></div>
        <h3 className="font-semibold text-sm">R2L</h3>
      </div>
      <p className="text-xs text-zinc-600 dark:text-zinc-400">Remote to Local access</p>
    </div>
    <div className="p-4 rounded-lg bg-pink-500/10 border border-pink-500/20">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-3 h-3 rounded-full bg-pink-500"></div>
        <h3 className="font-semibold text-sm">U2R</h3>
      </div>
      <p className="text-xs text-zinc-600 dark:text-zinc-400">User to Root privilege escalation</p>
    </div>
    <div className="p-4 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
        <h3 className="font-semibold text-sm">Normal</h3>
      </div>
      <p className="text-xs text-zinc-600 dark:text-zinc-400">Legitimate traffic</p>
    </div>
  </div>
);

export default function AttackChartPage() {
  const [attackData] = useState(generateAttackData());

  return (
    <main className="flex min-h-screen flex-col items-center justify-start bg-zinc-50 p-8 text-zinc-900 dark:bg-black dark:text-zinc-50">
      <div className="w-full max-w-6xl rounded-2xl bg-white p-8 shadow-lg dark:bg-zinc-900">
        <div className="mb-8">
          <h1 className="mb-3 text-4xl font-bold bg-gradient-to-r from-red-500 to-pink-500 bg-clip-text text-transparent">
            Cybersecurity Threat Analysis
          </h1>
          <p className="text-lg text-zinc-600 dark:text-zinc-400">
            Network intrusion detection system visualization showing attack patterns across multiple threat categories.
          </p>
        </div>

        <CategoryLegend />

        <div className="mt-8 p-6 rounded-xl bg-zinc-50 dark:bg-zinc-950">
          <h2 className="mb-4 text-xl font-semibold">Detection Volume by Attack Category</h2>
          <AttackBarChart data={attackData} />
        </div>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-5 rounded-lg bg-gradient-to-br from-red-500/10 to-orange-500/10 border border-red-500/20">
            <div className="text-3xl font-bold text-red-500 mb-1">
              {attackData.reduce((sum, item) => sum + (item.category !== 'Normal' ? item.count : 0), 0).toLocaleString()}
            </div>
            <div className="text-sm font-medium text-zinc-600 dark:text-zinc-400">Total Attack Detections</div>
          </div>
          <div className="p-5 rounded-lg bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border border-emerald-500/20">
            <div className="text-3xl font-bold text-emerald-500 mb-1">
              {attackData.find(item => item.category === 'Normal')?.count.toLocaleString()}
            </div>
            <div className="text-sm font-medium text-zinc-600 dark:text-zinc-400">Normal Traffic Events</div>
          </div>
          <div className="p-5 rounded-lg bg-gradient-to-br from-violet-500/10 to-purple-500/10 border border-violet-500/20">
            <div className="text-3xl font-bold text-violet-500 mb-1">
              {Object.keys(attackCategories).length - 1}
            </div>
            <div className="text-sm font-medium text-zinc-600 dark:text-zinc-400">Attack Categories Monitored</div>
          </div>
        </div>

        <div className="mt-8 p-6 rounded-xl bg-blue-500/5 border border-blue-500/20">
          <h3 className="font-semibold mb-3 text-blue-600 dark:text-blue-400">📊 Dataset Information</h3>
          <p className="text-sm text-zinc-600 dark:text-zinc-400 leading-relaxed">
            This visualization is based on network intrusion detection data containing <strong>{Object.values(attackCategories).flat().length} distinct attack types</strong> across 4 major threat categories.
            The data helps security teams identify patterns, prioritize responses, and understand the threat landscape.
          </p>
        </div>
      </div>
    </main>
  );
}
