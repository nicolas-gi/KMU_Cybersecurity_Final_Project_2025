# KMU Cybersecurity Final Project 2025

## Network Anomaly Detection Using AI

A Next.js web application for visualizing network security attack data and monitoring cybersecurity threats in real-time.

---

## 📋 Project Overview

This project provides a dashboard for monitoring and analyzing network security threats. It visualizes attack patterns and volumes through interactive charts, helping security teams identify and respond to potential threats.

### Features

- **Attack Volume Visualization**: Interactive bar charts showing the frequency of different attack types
- **Real-time Monitoring**: Dashboard for tracking DDoS, Port Scans, Brute Force attacks, and more
- **Dark Mode Support**: Modern UI with automatic dark/light theme switching
- **Responsive Design**: Mobile-friendly interface built with Tailwind CSS

### Tech Stack

- **Frontend Framework**: [Next.js 16](https://nextjs.org/) (React 19)
- **Language**: TypeScript
- **Styling**: Tailwind CSS 4
- **Charts**: [Nivo](https://nivo.rocks/) - Powerful React data visualization library
- **Deployment**: Optimized for Vercel

---

## 🚀 Getting Started

### Prerequisites

- Node.js 18+ installed on your machine
- npm, yarn, or pnpm package manager

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/nicolas-gi/KMU_Cybersecurity_Final_Project_2025.git
   cd final_proj
   ```

2. **Install dependencies**

   ```bash
   npm install
   ```

### Running the Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to see the application.

### Building for Production

```bash
npm run build
npm start
```

This creates an optimized production build and starts the production server.

---

## 📁 Project Structure

```shell
final_proj/
├── app/
│   ├── page.tsx              # Home page
│   ├── layout.tsx            # Root layout with metadata
│   ├── globals.css           # Global styles
│   └── attack-chart/
│       └── page.tsx          # Attack visualization dashboard
├── public/                   # Static assets
├── package.json              # Dependencies and scripts
├── tsconfig.json             # TypeScript configuration
├── tailwind.config.ts        # Tailwind CSS configuration
└── next.config.ts            # Next.js configuration
```

---

## 🎯 Available Routes

- **`/`** - Home page with project introduction
- **`/attack-chart`** - Interactive attack volume visualization dashboard

---

## 🛠️ Development

### Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server on port 3000 |
| `npm run build` | Create production build |
| `npm start` | Start production server |
| `npm run lint` | Run ESLint for code quality checks |

### Code Quality

The project uses:

- **ESLint** for code linting
- **TypeScript** for type safety
- **SonarQube** for code quality analysis (see `sonar-project.properties`)

---

## 📊 Data Visualization

The attack chart displays mock data for demonstration purposes. In a production environment, this would be connected to:

- Real-time network monitoring tools
- SIEM (Security Information and Event Management) systems
- Intrusion Detection Systems (IDS)
- Log aggregation platforms

### Current Attack Types Monitored

1. **DDoS** (Distributed Denial of Service)
2. **Port Scan** (Network reconnaissance)
3. **Brute Force** (Authentication attacks)

---

## 🔧 Customization

### Adding New Attack Types

Edit `app/attack-chart/page.tsx`:

```typescript
const attackData = [
  { attack: 'DDoS', count: 120 },
  { attack: 'Port Scan', count: 185 },
  { attack: 'Brute Force', count: 50 },
  { attack: 'SQL Injection', count: 75 }, // Add new types here
];
```

### Changing Chart Colors

Modify the `colors` prop in the `ResponsiveBar` component:

```typescript
colors={['#f97316']} // Change to your preferred color scheme
```

---

## 📝 License

This project is part of the KMU Cybersecurity Final Project 2025.

---

## 👥 Contributors

- **Repository**: [nicolas-gi/KMU_Cybersecurity_Final_Project_2025](https://github.com/nicolas-gi/KMU_Cybersecurity_Final_Project_2025)

---

## 🔗 Resources

- [Next.js Documentation](https://nextjs.org/docs)
- [Nivo Chart Library](https://nivo.rocks/)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
