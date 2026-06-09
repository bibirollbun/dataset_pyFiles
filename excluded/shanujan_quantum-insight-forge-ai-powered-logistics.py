import os
import json
from kaggle_secrets import UserSecretsClient

# 1. SETUP API KEY
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GEMINI_API_KEY")
except:
    api_key = "YOUR_API_KEY_HERE"

# 2. HELPER TO WRITE FILES
def write_file(path, content):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)

# 3. CONFIGURATION
package_json = {
    "name": "quantum-insight-forge",
    "version": "1.0.0",
    "type": "module",
    "scripts": { "dev": "vite", "build": "vite build" },
    "dependencies": {
        "react": "^18.2.0",
        "react-dom": "^18.2.0",
        "lucide-react": "^0.263.1",
        "leaflet": "^1.9.4",
        "react-markdown": "^8.0.7",
        "papaparse": "^5.4.1",
        "jspdf": "^2.5.1",
        "jspdf-autotable": "^3.5.31",
        "@google/genai": "*",
        "clsx": "^2.0.0",
        "tailwind-merge": "^1.14.0"
    },
    "devDependencies": {
        "@types/react": "^18.2.15",
        "@types/react-dom": "^18.2.7",
        "@types/leaflet": "^1.9.3",
        "@vitejs/plugin-react": "^4.0.3",
        "typescript": "^5.0.2",
        "vite": "^5.0.0",
        "tailwindcss": "^3.3.3",
        "postcss": "^8.4.27",
        "autoprefixer": "^10.4.14"
    }
}

# VITE CONFIG
vite_config = f"""
import {{ defineConfig }} from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({{
  plugins: [react()],
  define: {{
    'process.env.API_KEY': JSON.stringify("{api_key}") 
  }},
  server: {{ 
    host: true, 
    port: 8000,
    allowedHosts: true
  }}
}});
"""

index_html = """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Quantum Insight Forge</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" crossorigin=""/>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
      ::-webkit-scrollbar { width: 8px; height: 8px; }
      ::-webkit-scrollbar-track { background: #0f172a; }
      ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
      .leaflet-container { background: #0f172a; }
    </style>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/index.tsx"></script>
  </body>
</html>
"""

write_file("package.json", json.dumps(package_json, indent=2))
write_file("vite.config.ts", vite_config)
write_file("index.html", index_html)
print("✅ Project Environment Configured (Host Blocks Removed)")


# 1. DATA SERVICE
data_service = """
import Papa from 'papaparse';
import { City } from '../types';

const FALLBACK_CITIES: City[] = [
  { city: "Tokyo", lat: 35.6897, lng: 139.6922, country: "Japan", population: 37000000, id: "jp-1" },
  { city: "New York", lat: 40.6943, lng: -73.9249, country: "United States", population: 18000000, id: "us-1" },
  { city: "Berlin", lat: 52.5200, lng: 13.4050, country: "Germany", population: 3600000, id: "de-1" },
  { city: "São Paulo", lat: -23.5505, lng: -46.6333, country: "Brazil", population: 22000000, id: "br-1" }
];
export const AVAILABLE_COUNTRIES = ["United States", "Japan", "Germany", "Brazil", "China", "India", "France", "United Kingdom"];

interface CSVRow { city_ascii: string; lat: string; lng: string; country: string; population: string; id: string; }

export const fetchCities = async (country: string, limit: number): Promise<City[]> => {
  try {
    const response = await fetch('https://simplemaps.com/static/data/world-cities/worldcities.csv');
    if (!response.ok) throw new Error("Network response was not ok");
    const csvText = await response.text();
    return new Promise((resolve) => {
      Papa.parse<CSVRow>(csvText, {
        header: true,
        skipEmptyLines: true,
        complete: (results) => {
          const parsedCities: City[] = results.data
            .filter(row => row.country === country && row.lat && row.lng && row.population)
            .map(row => ({
              city: row.city_ascii,
              lat: parseFloat(row.lat),
              lng: parseFloat(row.lng),
              country: row.country,
              population: parseFloat(row.population) || 0,
              id: row.id || Math.random().toString(36)
            }))
            .sort((a, b) => b.population - a.population)
            .slice(0, limit);
          resolve(parsedCities.length > 0 ? parsedCities : generateSyntheticCities(country, limit));
        },
        error: () => resolve(generateSyntheticCities(country, limit))
      });
    });
  } catch (error) { return generateSyntheticCities(country, limit); }
};

const generateSyntheticCities = (country: string, limit: number): City[] => {
  const centers: any = { "China": {lat: 35, lng: 105}, "India": {lat: 20, lng: 77} };
  const center = centers[country] || {lat: 51, lng: 10}; 
  const synthetic: City[] = [];
  for(let i=0; i<limit; i++) {
    synthetic.push({
      city: `${country} City ${i+1}`, country, population: 1000000,
      lat: center.lat + (Math.random() - 0.5) * 10,
      lng: center.lng + (Math.random() - 0.5) * 10,
      id: `syn-${i}`
    });
  }
  return synthetic;
};
"""

# 2. TSP SOLVER
tsp_solver = """
import { City, SolverResult } from '../types';
const toRad = (value: number) => (value * Math.PI) / 180;
export const calculateDistance = (cityA: City, cityB: City): number => {
  const R = 6371; 
  const dLat = toRad(cityB.lat - cityA.lat);
  const dLon = toRad(cityB.lng - cityA.lng);
  const a = Math.sin(dLat/2)**2 + Math.cos(toRad(cityA.lat)) * Math.cos(toRad(cityB.lat)) * Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
};
const calculateRouteDistance = (route: City[]): number => {
  let dist = 0;
  for (let i = 0; i < route.length - 1; i++) dist += calculateDistance(route[i], route[i + 1]);
  if (route.length > 0) dist += calculateDistance(route[route.length - 1], route[0]);
  return dist;
};
const shuffle = <T,>(array: T[]): T[] => {
  const newArray = [...array];
  for (let i = newArray.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [newArray[i], newArray[j]] = [newArray[j], newArray[i]];
  }
  return newArray;
};
const clusterCities = (cities: City[], k: number): City[][] => {
  if (k <= 1) return [cities];
  const points = cities.slice(1);
  if (points.length < k) return [cities];
  let centroids = shuffle(points).slice(0, k);
  let clusters: City[][] = Array.from({ length: k }, () => []);
  for (let iter = 0; iter < 10; iter++) {
    clusters = Array.from({ length: k }, () => []);
    points.forEach(p => {
      let minDist = Infinity, clusterIndex = 0;
      centroids.forEach((c, idx) => {
        const d = calculateDistance(p, c);
        if (d < minDist) { minDist = d; clusterIndex = idx; }
      });
      clusters[clusterIndex].push(p);
    });
    centroids = clusters.map((cluster, idx) => {
      if (cluster.length === 0) return centroids[idx];
      return { ...centroids[idx], lat: cluster.reduce((s, c) => s + c.lat, 0) / cluster.length, lng: cluster.reduce((s, c) => s + c.lng, 0) / cluster.length };
    });
  }
  return clusters.map(cluster => [cities[0], ...cluster]);
};
const optimizeCluster = (cluster: City[]): { route: City[], distance: number } => {
  if (cluster.length <= 2) return { route: cluster, distance: calculateRouteDistance(cluster) };
  let currentRoute = [cluster[0], ...shuffle(cluster.slice(1))];
  let currentDistance = calculateRouteDistance(currentRoute);
  let bestRoute = [...currentRoute], bestDistance = currentDistance;
  let temperature = 10000;
  while (temperature > 0.1) {
    const newRoute = [...currentRoute];
    const pos1 = Math.floor(Math.random() * (newRoute.length - 1)) + 1;
    const pos2 = Math.floor(Math.random() * (newRoute.length - 1)) + 1;
    [newRoute[pos1], newRoute[pos2]] = [newRoute[pos2], newRoute[pos1]];
    const newDistance = calculateRouteDistance(newRoute);
    if (newDistance < currentDistance || Math.random() < Math.exp(-(newDistance - currentDistance) / temperature)) {
      currentRoute = newRoute; currentDistance = newDistance;
      if (currentDistance < bestDistance) { bestRoute = [...currentRoute]; bestDistance = currentDistance; }
    }
    temperature *= 0.95;
  }
  return { route: bestRoute, distance: bestDistance };
};
export const solveVRP = async (cities: City[], vehicleCount: number): Promise<SolverResult> => {
  await new Promise(resolve => setTimeout(resolve, 500));
  if (cities.length < 2) return { routes: [cities], totalDistance: 0, maxRouteDistance: 0, iterations: 0, balanceScore: 10 };
  const effectiveVehicles = Math.min(vehicleCount, cities.length - 1);
  const optimizedResults = clusterCities(cities, effectiveVehicles).map(cluster => optimizeCluster(cluster));
  const distances = optimizedResults.map(r => r.distance);
  const totalDistance = distances.reduce((a, b) => a + b, 0);
  const avgDist = totalDistance / distances.length;
  const deviation = distances.reduce((a, d) => a + Math.abs(d - avgDist), 0) / distances.length;
  return {
    routes: optimizedResults.map(r => r.route),
    totalDistance,
    maxRouteDistance: Math.max(...distances),
    iterations: 1000 * effectiveVehicles,
    balanceScore: Math.max(0, 10 - (deviation / avgDist) * 10)
  };
};
"""

# 3. GEMINI SERVICE
gemini_service = """
import { GoogleGenAI } from "@google/genai";
import { City, AnalysisReport, VoiceCommandResult, DataValidationResult, MaintenanceAnalysis, Scenario, ScenarioAnalysis, ConstraintAnalysis, DisruptionEvent, DisruptionAnalysis } from '../types';

const getClient = () => {
    const apiKey = process.env.API_KEY; 
    if (!apiKey) throw new Error("API Key missing");
    return new GoogleGenAI({ apiKey });
};

export const generateReports = async (routes: City[][], totalDistance: number, vehicleCount: number): Promise<AnalysisReport> => {
    try {
        const ai = getClient();
        const res = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: `Logistics VRP Analysis. ${vehicleCount} vehicles, ${totalDistance.toFixed(0)}km total. Return 2 distinct sections: Mathematical Model (Latex) and Executive Judge Report (Markdown).`
        });
        return { 
            mathModel: "## VRP Formulation\\nMin $Z = \\sum c_{ij} x_{ij}$", 
            judgeReport: res.text || "Analysis generated." 
        };
    } catch (e) { return { mathModel: "Error", judgeReport: "AI Service Unavailable. Check API Key." }; }
};

export const chatWithAssistant = async (history: any[], message: string) => {
    try {
        const ai = getClient();
        const chat = ai.chats.create({ model: "gemini-2.5-flash", history });
        const res = await chat.sendMessage({ message });
        return res.text || "";
    } catch { return "Service Offline"; }
};

export const processVoiceCommand = async (transcript: string) => {
    try {
        const ai = getClient();
        const res = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: `Extract logic: "${transcript}". JSON {country, cityCount, extraStops:[{name,lat,lng}]}`,
            config: { responseMimeType: "application/json" }
        });
        return JSON.parse(res.text || "{}");
    } catch { return {}; }
};

export const validateDataset = async (csv: string) => {
    try {
        const ai = getClient();
        const res = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: `Validate CSV: ${csv}. JSON {isValid, issues:[], summary, confidenceScore}`,
            config: { responseMimeType: "application/json" }
        });
        return JSON.parse(res.text || "{}");
    } catch { return { isValid: false, issues: ["Error"], summary: "Fail", confidenceScore: 0 }; }
};

export const analyzeMaintenance = async (desc: string) => {
    try {
        const ai = getClient();
        const res = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: `Diagnose issue: "${desc}". JSON {diagnosis, recommendedAction, partsList:[], estimatedDowntime, priority}`,
            config: { responseMimeType: "application/json" }
        });
        return JSON.parse(res.text || "{}");
    } catch { return { diagnosis: "Error", recommendedAction: "", partsList: [], estimatedDowntime: "", priority: "HIGH" }; }
};

export const analyzeScenarios = async (scenarios: Scenario[]) => {
    try {
        const ai = getClient();
        const res = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: `Compare scenarios: ${JSON.stringify(scenarios)}. JSON {recommendedScenarioId, reasoning, forecast}`,
            config: { responseMimeType: "application/json" }
        });
        return JSON.parse(res.text || "{}");
    } catch { return { recommendedScenarioId: "", reasoning: "Error", forecast: "" }; }
};

export const parseConstraints = async (rule: string) => {
    try {
        const ai = getClient();
        const res = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: `Rule to math: "${rule}". JSON {penaltyFunction, impactScore, mathematicalForm}`,
            config: { responseMimeType: "application/json" }
        });
        return JSON.parse(res.text || "{}");
    } catch { return { penaltyFunction: "Error", impactScore: 0, mathematicalForm: "" }; }
};

export const assessDisruptions = async (events: DisruptionEvent[]) => {
    try {
        const ai = getClient();
        const res = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: `Analyze disruptions: ${JSON.stringify(events)}. JSON {riskLevel, impactSummary, rerouteSuggestion}`,
            config: { responseMimeType: "application/json" }
        });
        return JSON.parse(res.text || "{}");
    } catch { return { riskLevel: "LOW", impactSummary: "Error", rerouteSuggestion: "" }; }
};
"""

# 4. PDF SERVICE
pdf_service = """
import jsPDF from 'jspdf';
import 'jspdf-autotable';
import { SolverResult, CostMetrics, AnalysisReport } from '../types';
export const generatePDFReport = (res: SolverResult, cost: CostMetrics, analysis: AnalysisReport, country: string) => {
    const doc = new jsPDF();
    doc.text(`Quantum Insight Forge - ${country}`, 20, 20);
    doc.text(`Total Distance: ${cost.totalDistance.toFixed(1)} km`, 20, 30);
    doc.save('report.pdf');
};
"""

write_file("src/services/dataService.ts", data_service)
write_file("src/services/tspSolver.ts", tsp_solver)
write_file("src/services/geminiService.ts", gemini_service)
write_file("src/services/pdfService.ts", pdf_service)
print("✅ Services Written")


# MAP VISUALIZATION
map_vis = """
import React, { useEffect, useRef } from 'react';
import L from 'leaflet';
import { City } from '../types';

const ROUTE_COLORS = ['#06b6d4', '#a855f7', '#10b981', '#f59e0b', '#ec4899', '#ef4444'];

interface MapProps {
    routes: City[][]; 
    onMapClick?: (lat: number, lng: number) => void;
    interactiveMode: boolean;
    showZones?: boolean;
    liveDispatchMode?: boolean;
}

const MapVisualization: React.FC<MapProps> = ({ routes, onMapClick, interactiveMode, showZones, liveDispatchMode }) => {
    const mapContainerRef = useRef<HTMLDivElement>(null);
    const mapInstanceRef = useRef<L.Map | null>(null);
    const layerGroupRef = useRef<L.LayerGroup | null>(null);

    useEffect(() => {
        if (!mapContainerRef.current) return;
        if (mapInstanceRef.current) { (mapInstanceRef.current as any).remove(); mapInstanceRef.current = null; }

        const map = L.map(mapContainerRef.current, { zoomControl: false }).setView([20, 0], 2);
        mapInstanceRef.current = map;
        L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { maxZoom: 19 }).addTo(map);
        layerGroupRef.current = L.layerGroup([]).addTo(map);

        return () => { if (mapInstanceRef.current) { mapInstanceRef.current.remove(); mapInstanceRef.current = null; } };
    }, []); 

    useEffect(() => {
        if (!mapInstanceRef.current || !layerGroupRef.current) return;
        const layerGroup = layerGroupRef.current;
        const map = mapInstanceRef.current;
        layerGroup.clearLayers();
        if (!routes || routes.length === 0) return;

        const allLatLngs: L.LatLngExpression[] = [];
        routes.forEach((route, i) => {
            if (route.length === 0) return;
            const latLngs = route.map(c => [c.lat, c.lng] as [number, number]);
            allLatLngs.push(...latLngs);
            const color = ROUTE_COLORS[i % ROUTE_COLORS.length];
            L.polyline([...latLngs, latLngs[0]], { color, weight: 3, dashArray: '10, 10' }).addTo(layerGroup);
            if (showZones) L.circle([route[0].lat, route[0].lng], { radius: 300000, color, fillOpacity: 0.1 }).addTo(layerGroup);

            route.forEach((city, idx) => {
                L.circleMarker([city.lat, city.lng], { radius: idx===0?8:5, color: idx===0?'#ef4444':color, fillColor: idx===0?'#fff':color, fillOpacity: 1 }).addTo(layerGroup)
                .bindPopup(`<b>${city.city}</b>`);
            });
        });

        if (allLatLngs.length > 0) map.flyToBounds(L.latLngBounds(allLatLngs), { padding: [50, 50], duration: 1.5 });
    }, [routes, showZones]);

    return <div ref={mapContainerRef} className="w-full h-full min-h-[400px] bg-slate-900 rounded-xl" />;
};
export default MapVisualization;
"""

# --- 2. MAIN APP ---
app_tsx = """
import React, { useState } from 'react';
import { Truck, Activity, Play, BrainCircuit, LayoutDashboard, Database, Shield, Key, CreditCard, Menu, Wrench, GitBranch, Share2, Sliders, Radio, Info } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import MapVisualization from './components/MapVisualization';
import IntroScreen from './components/IntroScreen';
import FleetHub from './components/FleetHub';
import SecurityConsole from './components/SecurityConsole';
import ScenarioLab from './components/ScenarioLab';
import DataAssimilation from './components/DataAssimilation';
import MaintenanceBay from './components/MaintenanceBay';
import ConstraintEngine from './components/ConstraintEngine';
import EntangledOptimization from './components/EntangledOptimization';
import AboutUs from './components/AboutUs';
import { AVAILABLE_COUNTRIES, fetchCities } from './services/dataService';
import { solveVRP } from './services/tspSolver';
import { generateReports } from './services/geminiService';
import { AppStatus } from './types';

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState('HOME');
  const [introFinished, setIntroFinished] = useState(false);
  const [routes, setRoutes] = useState<any[]>([]);
  const [status, setStatus] = useState<AppStatus>(AppStatus.IDLE);
  const [selectedCountry, setSelectedCountry] = useState(AVAILABLE_COUNTRIES[0]);
  const [vehicleCount, setVehicleCount] = useState(3);
  const [aiReport, setAiReport] = useState<any>(null);

  const handleForge = async () => {
      setStatus(AppStatus.SOLVING);
      const data = await fetchCities(selectedCountry, 15);
      const result = await solveVRP(data, vehicleCount);
      setRoutes(result.routes);
      setStatus(AppStatus.GENERATING_AI);
      const report = await generateReports(result.routes, result.totalDistance, vehicleCount);
      setAiReport(report);
      setStatus(AppStatus.COMPLETE);
  };

  const NavItem = ({ view, icon: Icon, label }: any) => (
      <button onClick={() => setCurrentView(view)} className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all ${currentView === view ? 'bg-cyan-900/20 text-cyan-400 border border-cyan-500/30' : 'text-slate-400 hover:text-white hover:bg-slate-800'}`}>
          <Icon className="w-5 h-5" /> <span className="font-medium text-sm">{label}</span>
      </button>
  );

  if (!introFinished) return <IntroScreen onComplete={() => setIntroFinished(true)} />;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans flex overflow-hidden">
      <aside className="w-64 bg-slate-900 border-r border-slate-800 flex flex-col shrink-0 overflow-y-auto">
          <div className="p-6 flex items-center gap-3 border-b border-slate-800">
              <div className="p-1.5 bg-cyan-500 rounded"><BrainCircuit className="w-5 h-5 text-white" /></div>
              <h1 className="font-bold text-white">QIF <span className="text-cyan-400 font-light">HUB</span></h1>
          </div>
          <div className="p-4 space-y-1">
              <NavItem view="HOME" icon={LayoutDashboard} label="Operations Center" />
              <NavItem view="FLEET" icon={Truck} label="Fleet Hub" />
              <NavItem view="MAINTENANCE" icon={Wrench} label="Maintenance Bay" />
              <NavItem view="SCENARIO" icon={GitBranch} label="Scenario Lab" />
              <NavItem view="ENTANGLED" icon={Share2} label="Entangled Opt." />
              <NavItem view="CONSTRAINTS" icon={Sliders} label="Constraint Engine" />
              <NavItem view="DATA_ASSIM" icon={Radio} label="Live Assimilation" />
              <NavItem view="ABOUT" icon={Info} label="About" />
          </div>
      </aside>

      <main className="flex-1 h-screen overflow-y-auto p-6">
         {currentView === 'HOME' ? (
             <div className="grid grid-cols-12 gap-6">
                 <div className="col-span-4 space-y-4">
                     <div className="bg-slate-900 border border-slate-800 rounded-xl p-6">
                         <h2 className="font-bold text-white mb-4">Mission Control</h2>
                         <div className="space-y-4">
                             <select value={selectedCountry} onChange={e=>setSelectedCountry(e.target.value)} className="w-full bg-slate-950 border border-slate-700 rounded p-2 text-sm">
                                {AVAILABLE_COUNTRIES.map(c => <option key={c} value={c}>{c}</option>)}
                             </select>
                             <div className="text-sm">Vehicles: {vehicleCount}</div>
                             <input type="range" min="1" max="5" value={vehicleCount} onChange={e=>setVehicleCount(parseInt(e.target.value))} className="w-full" />
                             <button onClick={handleForge} disabled={status === AppStatus.SOLVING} className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 text-white rounded font-bold flex items-center justify-center gap-2">
                                 {status === AppStatus.SOLVING ? <Activity className="animate-spin w-4 h-4"/> : <Play className="w-4 h-4"/>} FORGE ROUTE
                             </button>
                         </div>
                     </div>
                     {aiReport && (
                         <div className="bg-slate-900 border border-slate-800 rounded-xl p-4 text-xs">
                             <h3 className="text-cyan-400 font-bold mb-2">Gemini Analysis</h3>
                             <ReactMarkdown>{aiReport.judgeReport}</ReactMarkdown>
                         </div>
                     )}
                 </div>
                 <div className="col-span-8 h-[600px] bg-slate-900 rounded-xl border border-slate-800 overflow-hidden relative">
                     <MapVisualization routes={routes} interactiveMode={false} />
                 </div>
             </div>
         ) : currentView === 'FLEET' ? <FleetHub />
           : currentView === 'MAINTENANCE' ? <MaintenanceBay />
           : currentView === 'SCENARIO' ? <ScenarioLab />
           : currentView === 'ENTANGLED' ? <EntangledOptimization />
           : currentView === 'CONSTRAINTS' ? <ConstraintEngine />
           : currentView === 'DATA_ASSIM' ? <DataAssimilation />
           : currentView === 'ABOUT' ? <AboutUs />
           : null
         }
      </main>
    </div>
  );
};
export default App;
"""

index_tsx = """
import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
"""
index_css = "@tailwind base; @tailwind components; @tailwind utilities;"

write_file("src/components/MapVisualization.tsx", map_vis)
write_file("src/App.tsx", app_tsx)
write_file("src/index.tsx", index_tsx)
write_file("src/index.css", index_css)
print("✅ Core Components Written")


import os

# Helper to write the file
def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


# Define content for src/types.ts
types_ts_content = """
export interface City {
  city: string;
  lat: number;
  lng: number;
  country: string;
  population: number;
  id: string;
  isEmergency?: boolean;
}

export interface SolverResult {
  routes: City[][];
  totalDistance: number;
  maxRouteDistance: number;
  iterations: number;
  balanceScore: number;
}

export interface AnalysisReport {
  mathModel: string;
  judgeReport: string;
}

export interface CostMetrics {
  totalDistance: number;
  fuelCost: number;
  co2Emissions: number;
  driverHours: number;
}

export enum AppStatus {
  IDLE = 'IDLE',
  LOADING_DATA = 'LOADING_DATA',
  SOLVING = 'SOLVING',
  GENERATING_AI = 'GENERATING_AI',
  COMPLETE = 'COMPLETE',
  ERROR = 'ERROR'
}

export type Role = 'ADMIN' | 'MEMBER';
export type PlanTier = 'FREE' | 'PRO' | 'ENTERPRISE';
export type AppView = 'HOME' | 'DATA' | 'ADMIN' | 'API' | 'BILLING' | 'FLEET' | 'PERFORMANCE' | 'KNOWLEDGE' | 'MAINTENANCE' | 'SCENARIO' | 'ENTANGLED' | 'CONSTRAINTS' | 'DATA_ASSIM' | 'ABOUT';

export interface User {
  id: string;
  name: string;
  email: string;
  isGuest: boolean;
  role: Role;
  plan: PlanTier;
  avatar?: string;
  organization?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'model';
  text: string;
  timestamp: number;
}

export interface HistoryEntry {
  id: string;
  timestamp: number;
  country: string;
  cityCount: number;
  vehicleCount: number;
  solverResult: SolverResult;
  costMetrics: CostMetrics;
}

export interface VoiceCommandResult {
  country?: string;
  cityCount?: number;
  extraStops?: { name: string; lat: number; lng: number }[];
  intent?: string;
}

export interface AuditLogEntry {
  id: string;
  timestamp: number;
  userId: string;
  userEmail: string;
  action: string;
  details: string;
  severity: 'INFO' | 'WARNING' | 'CRITICAL';
}

export interface Dataset {
  id: string;
  name: string;
  uploadDate: number;
  rowCount: number;
  status: 'VALIDATED' | 'PENDING' | 'REJECTED';
  data: City[];
}

export interface DataValidationResult {
  isValid: boolean;
  issues: string[];
  summary: string;
  confidenceScore: number;
}

// --- NEW MODULE TYPES ---

export interface Vehicle {
  id: string;
  name: string;
  type: 'TRUCK' | 'VAN' | 'DRONE';
  status: 'IDLE' | 'IN_TRANSIT' | 'MAINTENANCE' | 'OFFLINE';
  fuelLevel: number;
  location: string;
  nextService: string;
  driverId?: string;
}

export interface SecurityEvent {
  id: string;
  timestamp: number;
  type: 'LOGIN_SUCCESS' | 'LOGIN_FAILED' | 'SUSPICIOUS_IP' | 'API_ABUSE';
  ip: string;
  location: { lat: number; lng: number; city: string };
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  user?: string;
}

export interface Badge {
  id: string;
  name: string;
  icon: string;
  description: string;
  unlocked: boolean;
}

export interface DriverProfile {
  id: string;
  name: string;
  rank: number;
  score: number;
  deliveries: number;
  onTimeRate: number;
  badges: Badge[];
}

export interface KnowledgeArticle {
  id: string;
  title: string;
  category: 'OPERATIONS' | 'TECHNICAL' | 'STRATEGY';
  content: string;
  views: number;
}

export interface MaintenanceIssue {
  id: string;
  vehicleId: string;
  description: string;
  reportedAt: number;
  status: 'OPEN' | 'DIAGNOSED' | 'REPAIRING' | 'RESOLVED';
  diagnosis?: string;
  partsRequired?: string[];
  priority: 'LOW' | 'MEDIUM' | 'HIGH';
}

export interface MaintenanceAnalysis {
  diagnosis: string;
  recommendedAction: string;
  partsList: string[];
  estimatedDowntime: string;
  priority: 'LOW' | 'MEDIUM' | 'HIGH';
}

export interface Scenario {
  id: string;
  name: string;
  description: string;
  parameters: { speed: number; cost: number; eco: number };
}

export interface ScenarioAnalysis {
  recommendedScenarioId: string;
  reasoning: string;
  forecast: string;
}

export interface ConstraintRule {
  id: string;
  text: string;
  isActive: boolean;
}

export interface ConstraintAnalysis {
  penaltyFunction: string;
  impactScore: number; 
  mathematicalForm: string;
}

export interface DisruptionEvent {
  id: string;
  source: 'WEATHER' | 'TRAFFIC' | 'SOCIAL';
  description: string;
  location: string;
  timestamp: number;
  severity: 'LOW' | 'MEDIUM' | 'HIGH';
}

export interface DisruptionAnalysis {
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  impactSummary: string;
  rerouteSuggestion: string;
}
"""


# INTRO SCREEN
intro_screen = """
import React, { useState, useEffect } from 'react';
import { BrainCircuit } from 'lucide-react';

interface IntroScreenProps {
    onComplete: () => void;
}

const IntroScreen: React.FC<IntroScreenProps> = ({ onComplete }) => {
    const [progress, setProgress] = useState(0);
    const [statusText, setStatusText] = useState("Initializing Quantum Core...");

    useEffect(() => {
        const interval = setInterval(() => {
            setProgress(prev => {
                if (prev >= 100) {
                    clearInterval(interval);
                    return 100;
                }
                return prev + 1;
            });
        }, 30);

        setTimeout(() => setStatusText("Loading Geospatial Datasets..."), 800);
        setTimeout(() => setStatusText("Calibrating VRP Solver Engines..."), 1800);
        setTimeout(() => setStatusText("Establishing Secure Uplink..."), 2500);
        setTimeout(() => setStatusText("Ready."), 3200);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="fixed inset-0 bg-slate-950 flex flex-col items-center justify-center z-[100] overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-b from-transparent via-slate-900/50 to-slate-950 pointer-events-none"></div>
            <div className="relative z-10 flex flex-col items-center">
                <div className="w-24 h-24 bg-cyan-500/10 rounded-2xl border border-cyan-500/30 flex items-center justify-center mb-8 relative shadow-[0_0_50px_rgba(6,182,212,0.3)] animate-pulse">
                    <BrainCircuit className="w-12 h-12 text-cyan-400" />
                </div>
                <h1 className="text-4xl font-black text-white tracking-[0.2em] mb-2 text-center">
                    QUANTUM<br/><span className="text-cyan-500 font-light">FORGE</span>
                </h1>
                <p className="text-xs text-cyan-500/70 font-mono tracking-widest uppercase mb-4">{statusText}</p>
                <div className="w-64 h-1 bg-slate-800 rounded-full overflow-hidden mb-8">
                    <div className="h-full bg-cyan-500 shadow-[0_0_10px_rgba(6,182,212,0.8)] transition-all duration-75 ease-out" style={{width: `${progress}%`}}></div>
                </div>
                <button onClick={onComplete} disabled={progress < 100} className={`px-10 py-4 bg-cyan-600 text-white font-bold tracking-[0.2em] transition-all duration-500 ${progress < 100 ? 'opacity-50 cursor-wait' : 'opacity-100 hover:bg-cyan-500'}`}>
                    ENTER SYSTEM
                </button>
            </div>
        </div>
    );
};
export default IntroScreen;
"""

# FLEET HUB
fleet_hub = """
import React, { useState } from 'react';
import { Truck, Activity, Wrench, MapPin, Battery, Search } from 'lucide-react';
import { Vehicle } from '../types';

const MOCK_FLEET: Vehicle[] = [
    { id: 'V-101', name: 'Optimus Prime', type: 'TRUCK', status: 'IN_TRANSIT', fuelLevel: 78, location: 'En route to Berlin', nextService: '2025-11-01', driverId: 'D-01' },
    { id: 'V-102', name: 'Iron Hide', type: 'VAN', status: 'IDLE', fuelLevel: 45, location: 'Hamburg Depot', nextService: '2025-10-15', driverId: 'D-02' },
    { id: 'V-103', name: 'Bumblebee', type: 'DRONE', status: 'MAINTENANCE', fuelLevel: 12, location: 'Repair Bay 4', nextService: '2025-10-05' },
    { id: 'V-104', name: 'Megatron', type: 'TRUCK', status: 'IN_TRANSIT', fuelLevel: 92, location: 'Autobahn A9', nextService: '2025-12-01', driverId: 'D-03' },
    { id: 'V-105', name: 'Starscream', type: 'DRONE', status: 'OFFLINE', fuelLevel: 0, location: 'Charging Station', nextService: '2025-10-02' },
];

const FleetHub: React.FC = () => {
    const [filter, setFilter] = useState('');
    const filtered = MOCK_FLEET.filter(v => v.name.toLowerCase().includes(filter.toLowerCase()));
    
    return (
        <div className="space-y-6 animate-in fade-in">
            <div className="flex justify-between items-center">
                <h2 className="text-xl font-bold text-white">Fleet Registry</h2>
                <div className="relative">
                    <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-500" />
                    <input type="text" placeholder="Search..." value={filter} onChange={e=>setFilter(e.target.value)} className="bg-slate-900 border border-slate-700 rounded-lg py-2 pl-9 pr-4 text-sm text-white" />
                </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {filtered.map(vehicle => (
                    <div key={vehicle.id} className="bg-slate-900 border border-slate-800 rounded-xl p-6 hover:border-cyan-500/50 transition-all group">
                        <div className="flex items-center gap-4 mb-4">
                            <div className="w-12 h-12 bg-slate-950 rounded-full flex items-center justify-center border border-slate-800 group-hover:border-cyan-500/50">
                                <Truck className="w-6 h-6 text-slate-400 group-hover:text-cyan-400" />
                            </div>
                            <div>
                                <h3 className="font-bold text-white text-lg">{vehicle.name}</h3>
                                <div className="text-xs text-slate-500 font-mono">{vehicle.id}</div>
                            </div>
                        </div>
                        <div className="space-y-2 text-sm text-slate-400">
                             <div className="flex justify-between"><span>Status</span><span className="text-white">{vehicle.status}</span></div>
                             <div className="flex justify-between"><span>Fuel</span><span className="text-white">{vehicle.fuelLevel}%</span></div>
                             <div className="flex justify-between"><span>Location</span><span className="text-white">{vehicle.location}</span></div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};
export default FleetHub;
"""

# SCENARIO LAB
scenario_lab = """
import React, { useState } from 'react';
import { GitBranch, Play, Zap, DollarSign, Leaf, BarChart2 } from 'lucide-react';
import { Scenario, ScenarioAnalysis } from '../types';
import { analyzeScenarios } from '../services/geminiService';

const ScenarioLab: React.FC = () => {
    const [scenarios] = useState<Scenario[]>([
        { id: '1', name: 'Velocity Max', description: 'Maximize speed', parameters: { speed: 100, cost: 0, eco: 20 } },
        { id: '2', name: 'Eco Saver', description: 'Minimize CO2', parameters: { speed: 40, cost: 60, eco: 100 } },
    ]);
    const [result, setResult] = useState<ScenarioAnalysis | null>(null);
    const [isSimulating, setIsSimulating] = useState(false);

    const handleSimulation = async () => {
        setIsSimulating(true);
        const analysis = await analyzeScenarios(scenarios);
        setResult(analysis);
        setIsSimulating(false);
    };

    return (
        <div className="space-y-6 animate-in fade-in">
             <div className="bg-slate-900 border border-slate-800 p-8 rounded-2xl flex justify-between items-center">
                 <div>
                    <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-3"><GitBranch className="w-8 h-8 text-cyan-400" /> Scenario Lab</h2>
                    <p className="text-slate-400">Simulate multiple strategies in parallel.</p>
                 </div>
                 <button onClick={handleSimulation} disabled={isSimulating} className="px-8 py-4 bg-cyan-600 hover:bg-cyan-500 text-white font-bold rounded-xl flex items-center gap-3">
                    {isSimulating ? 'SIMULATING...' : 'RUN SIMULATION'}
                 </button>
             </div>
             {result && (
                 <div className="bg-slate-900 border border-slate-800 rounded-xl p-8">
                     <h3 className="text-xl font-bold text-white mb-4">Analysis Result</h3>
                     <p className="text-slate-300">{result.reasoning}</p>
                 </div>
             )}
        </div>
    );
};
export default ScenarioLab;
"""

# DATA ASSIMILATION
data_assim = """
import React, { useState, useEffect } from 'react';
import { Radio, Car, CloudRain, RefreshCw } from 'lucide-react';
import { assessDisruptions } from '../services/geminiService';
import { DisruptionEvent, DisruptionAnalysis } from '../types';

const MOCK_EVENTS: DisruptionEvent[] = [
    { id: '1', source: 'TRAFFIC', description: 'Accident on A4', location: 'Sector 7', timestamp: Date.now(), severity: 'HIGH' },
    { id: '2', source: 'WEATHER', description: 'Heavy Fog', location: 'North', timestamp: Date.now(), severity: 'MEDIUM' },
];

const DataAssimilation: React.FC = () => {
    const [analysis, setAnalysis] = useState<DisruptionAnalysis | null>(null);

    useEffect(() => { assessDisruptions(MOCK_EVENTS).then(setAnalysis); }, []);

    return (
        <div className="space-y-6 animate-in fade-in">
             <div className="bg-slate-900 border border-slate-800 p-8 rounded-2xl">
                 <h2 className="text-2xl font-bold text-white mb-2 flex items-center gap-3"><Radio className="w-8 h-8 text-blue-500" /> Live Assimilation</h2>
                 <p className="text-slate-400">Ingesting real-time traffic and weather data.</p>
             </div>
             <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                 <div className="space-y-4">
                     {MOCK_EVENTS.map(ev => (
                         <div key={ev.id} className="bg-slate-900 border border-slate-800 p-4 rounded-xl flex gap-4">
                             <div className="p-3 bg-slate-800 rounded-lg"><Car className="w-5 h-5 text-slate-400"/></div>
                             <div>
                                 <div className="font-bold text-white">{ev.description}</div>
                                 <div className="text-xs text-red-400 font-bold">{ev.severity}</div>
                             </div>
                         </div>
                     ))}
                 </div>
                 <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                     <h3 className="font-bold text-white mb-4">AI Risk Assessment</h3>
                     {analysis ? <p className="text-slate-300">{analysis.impactSummary}</p> : <p className="text-slate-500">Analyzing...</p>}
                 </div>
             </div>
        </div>
    );
};
export default DataAssimilation;
"""

# MAINTENANCE BAY
maintenance_bay = """
import React, { useState } from 'react';
import { Wrench, Activity } from 'lucide-react';
import { analyzeMaintenance } from '../services/geminiService';
import { MaintenanceAnalysis } from '../types';

const MaintenanceBay: React.FC = () => {
    const [desc, setDesc] = useState('');
    const [analysis, setAnalysis] = useState<MaintenanceAnalysis | null>(null);
    const [loading, setLoading] = useState(false);

    const run = async () => {
        setLoading(true);
        setAnalysis(await analyzeMaintenance(desc));
        setLoading(false);
    };

    return (
        <div className="space-y-6 animate-in fade-in">
            <div className="bg-slate-900 border border-slate-800 p-8 rounded-2xl">
                <h2 className="text-2xl font-bold text-white mb-4"><Wrench className="w-6 h-6 inline mr-2 text-amber-500"/> Maintenance Bay</h2>
                <textarea className="w-full bg-slate-950 border border-slate-700 rounded-lg p-4 text-white" value={desc} onChange={e=>setDesc(e.target.value)} placeholder="Describe vehicle issue..." />
                <button onClick={run} className="mt-4 px-6 py-2 bg-amber-600 rounded text-white font-bold">{loading ? 'Diagnosing...' : 'Run Diagnostics'}</button>
            </div>
            {analysis && (
                <div className="bg-slate-900 border border-slate-800 p-6 rounded-xl">
                    <h3 className="font-bold text-white mb-2">Diagnosis: {analysis.diagnosis}</h3>
                    <p className="text-slate-300">{analysis.recommendedAction}</p>
                </div>
            )}
        </div>
    );
};
export default MaintenanceBay;
"""

# OTHER COMPONENTS
about_us = """import React from 'react'; import { Info } from 'lucide-react'; 
const AboutUs = () => <div className="p-8 text-center text-white"><Info className="w-12 h-12 mx-auto text-cyan-500 mb-4"/><h1 className="text-2xl font-bold">About Quantum Insight Forge</h1><p className="text-slate-400 mt-2">Next-gen logistics optimization.</p></div>; 
export default AboutUs;"""

constraint_engine = """import React from 'react'; import { Sliders } from 'lucide-react';
const ConstraintEngine = () => <div className="p-8 text-center text-white"><Sliders className="w-12 h-12 mx-auto text-pink-500 mb-4"/><h1 className="text-2xl font-bold">Constraint Engine</h1><p className="text-slate-400">Define business rules.</p></div>;
export default ConstraintEngine;"""

entangled_opt = """import React from 'react'; import { Share2 } from 'lucide-react';
const EntangledOptimization = () => <div className="p-8 text-center text-white"><Share2 className="w-12 h-12 mx-auto text-purple-500 mb-4"/><h1 className="text-2xl font-bold">Entangled Optimization</h1><p className="text-slate-400">Quantum state synchronization.</p></div>;
export default EntangledOptimization;"""

# WRITE FILES
write_file("src/components/IntroScreen.tsx", intro_screen)
write_file("src/components/FleetHub.tsx", fleet_hub)
write_file("src/components/ScenarioLab.tsx", scenario_lab)
write_file("src/components/DataAssimilation.tsx", data_assim)
write_file("src/components/MaintenanceBay.tsx", maintenance_bay)
write_file("src/components/AboutUs.tsx", about_us)
write_file("src/components/ConstraintEngine.tsx", constraint_engine)
write_file("src/components/EntangledOptimization.tsx", entangled_opt)

# Mock remaining dashboards to prevent crashes
mock_comp = "import React from 'react'; export default () => <div className='p-8 text-white'>Component Placeholder</div>;"
write_file("src/components/SecurityConsole.tsx", mock_comp)
write_file("src/components/PerformanceCenter.tsx", mock_comp)
write_file("src/components/KnowledgeBase.tsx", mock_comp)
write_file("src/components/AdminDashboard.tsx", mock_comp)
write_file("src/components/ApiPortal.tsx", mock_comp)
write_file("src/components/BillingDashboard.tsx", mock_comp)
write_file("src/components/DataFoundry.tsx", mock_comp)
write_file("src/components/VoiceControl.tsx", mock_comp)
write_file("src/components/AnalyticsDashboard.tsx", mock_comp)
write_file("src/components/AuthModal.tsx", mock_comp)
write_file("src/components/AIAssistant.tsx", mock_comp)

# Write the file
write_file("src/types.ts", types_ts_content)
print("✅ Fixed: src/types.ts created successfully.")
print("✅ All Components Repaired")


# GET & PRINT PASSWORD (Public IP)
# Remove upcoming only "#" in every upcoming lines in this cell] (I did for privacy concerns you can get a copy and run on you copied notebook is recommended)
#import urllib.request
#print("THIS PASSWORD IS FOR YOUR LOCALTUNNEL")
#try:
    #ip = urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip()
    #print(f"{ip}")
#except:
    #print("Could not fetch IP. Try '!curl ifconfig.me'")


# Install Dependencies
!npm install

# Start Server & Tunnel
import subprocess
import time

print("Starting Vite Server in background.....")
# Start Vite (silently in background)
process = subprocess.Popen(["npm", "run", "dev"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(10)

# Remove upcoming only "#" in this 3 lines in this cell for endless running for this full process]
#print("Starting LocalTunnel...")
#!npm install -g localtunnel
#!lt --port 8000

