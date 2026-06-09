import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}")


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


import React, { useState, useEffect } from 'react';
import { 
  Car, Shield, DollarSign, ChevronRight, Check, 
  X, Fuel, Users, Gauge, ArrowRightLeft,
  Star, Sparkles, Loader
} from 'lucide-react';

// --- Configuration ---
// GET YOUR KEY: https://aistudio.google.com/app/apikey
const apiKey = ""; 

// --- Mock Database ---
const CAR_DATABASE = [
  {
    id: 1,
    make: "Toyota",
    model: "RAV4 Hybrid",
    year: 2024,
    price: 32000,
    type: "SUV",
    fuel: "Hybrid",
    mpg: 41,
    safetyRating: 5,
    seats: 5,
    features: ["Toyota Safety Sense 2.5", "AWD", "CarPlay"],
    safetyFeatures: { aeb: true, lka: true, bsm: true, ncap: 5 },
  },
  {
    id: 2,
    make: "Honda",
    model: "CR-V",
    year: 2024,
    price: 30000,
    type: "SUV",
    fuel: "Gas",
    mpg: 28,
    safetyRating: 5,
    seats: 5,
    features: ["Honda Sensing", "Spacious Interior", "Turbo Engine"],
    safetyFeatures: { aeb: true, lka: true, bsm: true, ncap: 5 },
  },
  {
    id: 3,
    make: "Tesla",
    model: "Model 3",
    year: 2024,
    price: 40000,
    type: "Sedan",
    fuel: "Electric",
    mpg: 130, // MPGe
    range: 272,
    safetyRating: 5,
    seats: 5,
    features: ["Autopilot Basic", "Glass Roof", "Supercharging"],
    safetyFeatures: { aeb: true, lka: true, bsm: true, ncap: 5 },
  },
  {
    id: 4,
    make: "Ford",
    model: "F-150 Lightning",
    year: 2024,
    price: 55000,
    type: "Truck",
    fuel: "Electric",
    mpg: 70, // MPGe
    range: 240,
    safetyRating: 5,
    seats: 5,
    features: ["Pro Power Onboard", "Frunk", "Towing"],
    safetyFeatures: { aeb: true, lka: true, bsm: true, ncap: 5 },
  },
  {
    id: 5,
    make: "Mazda",
    model: "CX-5",
    year: 2024,
    price: 29000,
    type: "SUV",
    fuel: "Gas",
    mpg: 26,
    safetyRating: 5,
    seats: 5,
    features: ["Premium Interior", "AWD Standard", "Quiet Ride"],
    safetyFeatures: { aeb: true, lka: true, bsm: true, ncap: 5 },
  },
  {
    id: 6,
    make: "Kia",
    model: "Telluride",
    year: 2024,
    price: 36000,
    type: "SUV",
    fuel: "Gas",
    mpg: 20,
    safetyRating: 4, 
    seats: 8,
    features: ["3rd Row Seating", "Highway Drive Assist", "Warranty"],
    safetyFeatures: { aeb: true, lka: true, bsm: true, ncap: 4 },
  },
  {
    id: 7,
    make: "Subaru",
    model: "Outback",
    year: 2024,
    price: 28895,
    type: "Wagon",
    fuel: "Gas",
    mpg: 26,
    safetyRating: 5,
    seats: 5,
    features: ["EyeSight Technology", "Symmetrical AWD", "Roof Rails"],
    safetyFeatures: { aeb: true, lka: true, bsm: true, ncap: 5 },
  },
  {
    id: 8,
    make: "Hyundai",
    model: "Elantra Hybrid",
    year: 2024,
    price: 26000,
    type: "Sedan",
    fuel: "Hybrid",
    mpg: 54,
    safetyRating: 5,
    seats: 5,
    features: ["Digital Key", "Bluelink", "Hands-free trunk"],
    safetyFeatures: { aeb: true, lka: true, bsm: true, ncap: 5 },
  },
  {
    id: 9,
    make: "Volvo",
    model: "XC90",
    year: 2024,
    price: 56000,
    type: "SUV",
    fuel: "Hybrid",
    mpg: 27,
    safetyRating: 5,
    seats: 7,
    features: ["Pilot Assist", "Built-in Booster Seat", "Luxury Interior"],
    safetyFeatures: { aeb: true, lka: true, bsm: true, ncap: 5 },
  },
  {
    id: 10,
    make: "Nissan",
    model: "Leaf",
    year: 2024,
    price: 28000,
    type: "Hatchback",
    fuel: "Electric",
    mpg: 111, // MPGe
    range: 149,
    safetyRating: 4,
    seats: 5,
    features: ["e-Pedal", "ProPILOT Assist", "Affordable EV"],
    safetyFeatures: { aeb: true, lka: true, bsm: true, ncap: 5 },
  },
];

// --- Utilities ---
const calculateMonthlyPayment = (price, downPayment, rate, term) => {
  const principal = price - downPayment;
  if (principal <= 0) return 0;
  const monthlyRate = rate / 100 / 12;
  return (principal * monthlyRate * Math.pow(1 + monthlyRate, term)) / (Math.pow(1 + monthlyRate, term) - 1);
};

// --- Components ---

const StepIndicator = ({ step }) => (
  <div className="flex items-center justify-center mb-8 space-x-2">
    {[1, 2, 3].map((s) => (
      <div 
        key={s} 
        className={`h-2 rounded-full transition-all duration-300 ${s === step ? 'w-8 bg-blue-600' : 'w-2 bg-gray-300'}`}
      />
    ))}
  </div>
);

const CarCard = ({ car, score, reason, onCompare, isSelected }) => {
  // Use a reliable placeholder image URL
  const carImage = car.make === "Toyota" && car.model === "RAV4 Hybrid" 
    ? "https://placehold.co/400x200/4F46E5/FFFFFF?text=Toyota+RAV4+Hybrid"
    : `https://placehold.co/400x200/F9FAFB/9CA3AF?text=${car.make}+${car.model}`;
    
  return (
    <div className={`relative bg-white rounded-xl shadow-md overflow-hidden border transition-all hover:shadow-lg ${isSelected ? 'border-blue-500 ring-2 ring-blue-100' : 'border-gray-100'}`}>
      <div className="h-32 flex items-center justify-center relative bg-gray-100">
        <img 
          src={carImage} 
          alt={`${car.make} ${car.model}`}
          className="w-full h-full object-cover"
          onError={(e) => { e.target.onerror = null; e.target.src="https://placehold.co/400x200/F9FAFB/9CA3AF?text=Car+Image"; }}
        />
        <div className="absolute top-3 right-3 bg-white/90 backdrop-blur-sm px-2 py-1 rounded-md text-xs font-bold text-gray-700 shadow-sm flex items-center">
          <Star size={12} className="text-yellow-400 mr-1 fill-yellow-400" />
          {car.safetyRating}/5 Safety
        </div>
        {score && (
          <div className="absolute top-3 left-3 bg-blue-600 text-white px-2 py-1 rounded-md text-xs font-bold shadow-sm flex items-center gap-1">
             {score}% Match
          </div>
        )}
      </div>
      
      <div className="p-4">
        <div className="flex justify-between items-start mb-2">
          <div>
            <h3 className="font-bold text-lg text-gray-900 leading-tight">{car.make} {car.model}</h3>
            {/* Using standard vertical bar | to prevent SyntaxError */}
            <p className="text-sm text-gray-500">{car.year} | {car.type}</p>
          </div>
          <div className="text-right">
            <span className="block font-bold text-blue-600">${car.price.toLocaleString()}</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-sm text-gray-600 mb-4">
          <div className="flex items-center gap-1"><Fuel size={14} /> {car.fuel}</div>
          <div className="flex items-center gap-1"><Gauge size={14} /> {car.mpg} {car.fuel === 'Electric' ? 'MPGe' : 'MPG'}</div>
          <div className="flex items-center gap-1"><Users size={14} /> {car.seats} Seats</div>
          <div className="flex items-center gap-1"><Shield size={14} /> {car.safetyFeatures.ncap} Star</div>
        </div>

        {reason && (
          <div className="bg-blue-50 p-3 rounded-lg text-xs text-blue-800 mb-4 border border-blue-100 flex gap-2 items-start">
             <Sparkles size={14} className="mt-0.5 flex-shrink-0 text-blue-500" />
             <span><span className="font-bold">AI Insight:</span> {reason}</span>
          </div>
        )}

        <button 
          onClick={() => onCompare(car)}
          className={`w-full py-2 rounded-lg font-medium text-sm transition-colors flex items-center justify-center gap-2
            ${isSelected 
              ? 'bg-red-50 text-red-600 hover:bg-red-100 border border-red-200' 
              : 'bg-white text-blue-600 hover:bg-blue-50 border border-blue-200'}`}
        >
          {isSelected ? <><X size={16} /> Remove Compare</> : <><ArrowRightLeft size={16} /> Compare</>}
        </button>
      </div>
    </div>
  );
};

export default function App() {
  const [step, setStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  
  const [preferences, setPreferences] = useState({
    budget: 35000,
    type: [],
    fuel: [],
    usage: 'commute',
    safetyPriority: 'high',
    passengers: 4
  });
  
  const [results, setResults] = useState([]);
  const [comparisonList, setComparisonList] = useState([]);
  const [financials, setFinancials] = useState({
    downPayment: 5000,
    creditScore: 'good',
    term: 60
  });

  const handlePrefChange = (key, value) => {
    setPreferences(prev => ({ ...prev, [key]: value }));
  };

  const toggleArrayPref = (key, value) => {
    setPreferences(prev => {
      const current = prev[key];
      if (current.includes(value)) {
        return { ...prev, [key]: current.filter(item => item !== value) };
      } else {
        return { ...prev, [key]: [...current, value] };
      }
    });
  };

  const getGeminiRecommendations = async () => {
    setIsLoading(true);
    setError(null);

    // Prompt Engineering
    const systemPrompt = "You are an expert car buying assistant. Analyze user preferences against the provided inventory and select the best matches.";
    const userPrompt = `
      User Preferences:
      - Max Budget: $${preferences.budget}
      - Preferred Types: ${preferences.type.length > 0 ? preferences.type.join(', ') : 'Any'}
      - Preferred Fuel: ${preferences.fuel.length > 0 ? preferences.fuel.join(', ') : 'Any'}
      - Primary Usage: ${preferences.usage}
      - Safety Importance: ${preferences.safetyPriority}

      Inventory Data (JSON):
      ${JSON.stringify(CAR_DATABASE.map(c => ({
        id: c.id, 
        make: c.make, 
        model: c.model, 
        price: c.price, 
        type: c.type, 
        fuel: c.fuel, 
        mpg: c.mpg,
        safety: c.safetyRating,
        seats: c.seats,
        features: c.features
      })))}

      Task:
      1. Filter and score the inventory based on how well it fits the User Preferences.
      2. Return the top 6 cars.
      3. For each car, provide:
         - "id": (number) matching the inventory id
         - "matchScore": (number) 0-100 score
         - "aiReason": (string) A persuasive, personalized 1-sentence reason why this specific car fits their usage (e.g., "Great for commuting due to 50 MPG" or "Perfect for family with 5-star safety").
      
      Return STRICT JSON format:
      [
        { "id": 1, "matchScore": 95, "aiReason": "..." },
        ...
      ]
    `;

    try {
      if (!apiKey) throw new Error("API Key missing");

      // Corrected API URL construction to use simple concatenation for stability
      const apiUrl = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key=" + apiKey;

      const response = await fetch(
        apiUrl,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            contents: [{ parts: [{ text: userPrompt }] }],
            systemInstruction: { parts: [{ text: systemPrompt }] },
            generationConfig: { responseMimeType: "application/json" }
          }),
        }
      );

      if (!response.ok) throw new Error("Gemini API call failed");

      const data = await response.json();
      const aiText = data.candidates?.[0]?.content?.parts?.[0]?.text;
      const aiResults = JSON.parse(aiText);

      // Merge AI results with full car objects
      const finalResults = aiResults.map(aiItem => {
        const fullCar = CAR_DATABASE.find(c => c.id === aiItem.id);
        if (!fullCar) return null;
        return { ...fullCar, matchScore: aiItem.matchScore, aiReason: aiItem.aiReason };
      }).filter(Boolean).sort((a, b) => b.matchScore - a.matchScore);

      setResults(finalResults);
      setStep(2);
      window.scrollTo(0, 0);

    } catch (err) {
      console.error("AI Error, falling back to heuristic:", err);
      // Fallback to local heuristic if API fails or no key
      fallbackRecommendations();
    } finally {
      setIsLoading(false);
    }
  };

  const fallbackRecommendations = () => {
    const scoredCars = CAR_DATABASE.map(car => {
      let score = 0;
      if (car.price <= preferences.budget) score += 30;
      if (preferences.type.includes(car.type)) score += 20;
      if (preferences.fuel.includes(car.fuel)) score += 15;
      return { 
        ...car, 
        matchScore: Math.min(100, Math.max(0, score + 20)), // Baseline padding 
        aiReason: "Standard match based on your criteria (Offline Mode)." 
      };
    }).sort((a, b) => b.matchScore - a.matchScore);

    setResults(scoredCars);
    setStep(2);
    window.scrollTo(0, 0);
  };

  const toggleCompare = (car) => {
    if (comparisonList.find(c => c.id === car.id)) {
      setComparisonList(comparisonList.filter(c => c.id !== car.id));
    } else {
      if (comparisonList.length < 3) {
        setComparisonList([...comparisonList, car]);
      }
    }
  };

  // --- Render Steps ---

  const renderStep1 = () => (
    <div className="max-w-2xl mx-auto space-y-8 animate-fade-in">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-extrabold text-gray-900 mb-2">Find Your Perfect Car</h1>
        <p className="text-gray-600">
           {apiKey ? "Powered by Google Gemini AI" : "Enter API Key in code to enable AI features"}
        </p>
      </div>

      <div className="bg-white p-6 rounded-2xl shadow-sm border border-gray-100 space-y-6">
        
        {/* Budget */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-2 flex justify-between">
            <span>Max Budget</span>
            <span className="text-blue-600 font-bold">${preferences.budget.toLocaleString()}</span>
          </label>
          <input 
            type="range" 
            min="15000" 
            max="100000" 
            step="1000" 
            value={preferences.budget} 
            onChange={(e) => handlePrefChange('budget', parseInt(e.target.value))}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
          />
          <div className="flex justify-between text-xs text-gray-400 mt-1">
            <span>{'$15k'}</span>
            <span>{'$100k+'}</span>
          </div>
        </div>

        {/* Usage */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">Primary Usage</label>
          <div className="grid grid-cols-3 gap-3">
            {['commute', 'family', 'adventure'].map((u) => (
              <button
                key={u}
                onClick={() => handlePrefChange('usage', u)}
                className={`py-3 px-4 rounded-xl border font-medium text-sm transition-all capitalize
                  ${preferences.usage === u 
                    ? 'bg-blue-600 text-white border-blue-600 shadow-md transform scale-105' 
                    : 'bg-white text-gray-600 border-gray-200 hover:border-blue-300'}`}
              >
                {u}
              </button>
            ))}
          </div>
        </div>

        {/* Types */}
        <div>
          <label className="block text-sm font-semibold text-gray-700 mb-3">Preferred Types</label>
          <div className="flex flex-wrap gap-2">
            {['Sedan', 'SUV', 'Truck', 'Hatchback', 'Electric'].map((type) => (
              <button
                key={type}
                onClick={() => toggleArrayPref('type', type)}
                className={`px-4 py-2 rounded-full text-xs font-semibold border transition-colors
                  ${preferences.type.includes(type)
                    ? 'bg-blue-100 text-blue-800 border-blue-200'
                    : 'bg-gray-50 text-gray-600 border-gray-200 hover:bg-gray-100'}`}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* Safety Priority */}
        <div className="bg-orange-50 p-4 rounded-xl border border-orange-100">
          <label className="block text-sm font-bold text-orange-900 mb-2 flex items-center gap-2">
            <Shield size={16} />
            Safety Importance
          </label>
          <div className="flex gap-4">
             <label className="flex items-center gap-2 cursor-pointer">
                <input 
                  type="radio" 
                  name="safety" 
                  checked={preferences.safetyPriority === 'standard'} 
                  onChange={() => handlePrefChange('safetyPriority', 'standard')}
                  className="text-orange-600 focus:ring-orange-500"
                />
                <span className="text-sm text-gray-700">Standard</span>
             </label>
             <label className="flex items-center gap-2 cursor-pointer">
                <input 
                  type="radio" 
                  name="safety" 
                  checked={preferences.safetyPriority === 'high'} 
                  onChange={() => handlePrefChange('safetyPriority', 'high')}
                  className="text-orange-600 focus:ring-orange-500"
                />
                <span className="text-sm text-gray-700 font-medium">High Priority</span>
             </label>
          </div>
        </div>

        <button 
          onClick={getGeminiRecommendations}
          disabled={isLoading}
          className={`w-full py-4 rounded-xl font-bold text-lg shadow-lg transition-all flex items-center justify-center gap-2
            ${isLoading ? 'bg-blue-400 cursor-not-allowed' : 'bg-blue-600 hover:bg-blue-700 text-white hover:shadow-xl'}`}
        >
          {isLoading ? (
            <><Loader className="animate-spin" /> Analyzing with AI...</>
          ) : (
            <>Find My Car <ChevronRight size={20} /></>
          )}
        </button>

      </div>
    </div>
  );

  const renderStep2 = () => (
    <div className="max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <div>
           <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
             <Sparkles className="text-blue-500" /> Top Recommendations
           </h2>
           <p className="text-gray-500 text-sm">AI-selected based on your needs.</p>
        </div>
        <button onClick={() => setStep(1)} className="text-blue-600 font-medium text-sm hover:underline">
          Edit Preferences
        </button>
      </div>

      <div className="flex flex-col lg:flex-row gap-6">
        {/* Results Grid */}
        <div className="flex-1 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {results.slice(0, 6).map(car => (
             <CarCard 
               key={car.id} 
               car={car} 
               score={car.matchScore} 
               reason={car.aiReason}
               onCompare={toggleCompare}
               isSelected={!!comparisonList.find(c => c.id === car.id)}
             />
          ))}
        </div>

        {/* Comparison Dock */}
        <div className="lg:w-80 flex-shrink-0">
           <div className="bg-white p-6 rounded-2xl shadow-lg border border-gray-100 sticky top-4">
             <h3 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
               <ArrowRightLeft size={18} /> 
               Compare ({comparisonList.length}/3)
             </h3>
             
             {comparisonList.length === 0 ? (
               <div className="text-center py-8 text-gray-400 border-2 border-dashed border-gray-200 rounded-xl">
                 <Car size={32} className="mx-auto mb-2 opacity-30" />
                 <p className="text-sm">Select cars to compare cost & safety.</p>
               </div>
             ) : (
               <div className="space-y-4">
                 {comparisonList.map(car => (
                   <div key={car.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                     <div className="text-sm font-medium">
                       {car.make} {car.model}
                     </div>
                     <button onClick={() => toggleCompare(car)} className="text-gray-400 hover:text-red-500">
                       <X size={16} />
                     </button>
                   </div>
                 ))}
                 <button 
                   onClick={() => setStep(3)}
                   className="w-full bg-indigo-600 hover:bg-indigo-700 text-white py-3 rounded-xl font-bold text-sm shadow-md transition-all mt-4"
                 >
                   Analyze & Compare
                 </button>
               </div>
             )}
           </div>
        </div>
      </div>
    </div>
  );

  const renderStep3 = () => {
    const rate = financials.creditScore === 'excellent' ? 4.5 : financials.creditScore === 'good' ? 6.5 : 9.0;
    
    return (
      <div className="max-w-6xl mx-auto animate-fade-in">
        <button onClick={() => setStep(2)} className="mb-6 flex items-center text-gray-500 hover:text-gray-900">
           <ChevronRight className="rotate-180 mr-1" size={16}/> Back to Results
        </button>
        
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Deep Dive Comparison</h2>

        <div className="bg-white p-4 rounded-xl shadow-sm border border-gray-200 mb-8">
          <h3 className="text-sm font-bold text-gray-700 uppercase tracking-wide mb-4 flex items-center gap-2">
            <DollarSign size={16} /> Cost Parameters
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div>
               <label className="text-xs font-semibold text-gray-500 block mb-1">Down Payment ($)</label>
               <input 
                 type="number" 
                 value={financials.downPayment} 
                 onChange={(e) => setFinancials({...financials, downPayment: Number(e.target.value)})}
                 className="w-full p-2 border rounded-lg text-sm"
               />
            </div>
            <div>
               <label className="text-xs font-semibold text-gray-500 block mb-1">Credit Score</label>
               <select 
                 value={financials.creditScore}
                 onChange={(e) => setFinancials({...financials, creditScore: e.target.value})}
                 className="w-full p-2 border rounded-lg text-sm bg-white"
               >
                 <option value="excellent">Excellent (750+)</option>
                 <option value="good">Good (650-749)</option>
                 <option value="fair">Fair (600-649)</option>
               </select>
            </div>
            <div>
               <label className="text-xs font-semibold text-gray-500 block mb-1">Loan Term</label>
               <select 
                 value={financials.term}
                 onChange={(e) => setFinancials({...financials, term: Number(e.target.value)})}
                 className="w-full p-2 border rounded-lg text-sm bg-white"
               >
                 <option value={36}>36 Months</option>
                 <option value={48}>48 Months</option>
                 <option value={60}>60 Months</option>
                 <option value={72}>72 Months</option>
               </select>
            </div>
          </div>
        </div>

        <div className="overflow-x-auto bg-white rounded-xl shadow-sm border border-gray-200">
          <table className="w-full min-w-[800px]">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="p-4 text-left w-48 text-gray-500 font-medium">Feature</th>
                {comparisonList.map(car => (
                  <th key={car.id} className="p-4 text-left min-w-[200px]">
                    <div className="font-bold text-lg text-gray-900">{car.make}</div>
                    <div className="text-sm text-gray-500">{car.model}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              <tr>
                <td className="p-4 font-semibold text-gray-700 bg-gray-50/50">Sticker Price</td>
                {comparisonList.map(car => (
                  <td key={car.id} className="p-4 font-bold text-gray-900">
                    ${car.price.toLocaleString()}
                  </td>
                ))}
              </tr>
              <tr>
                <td className="p-4 font-semibold text-blue-800 bg-blue-50/30">
                  <div className="flex flex-col">
                    <span>Est. Monthly Payment</span>
                    <span className="text-xs font-normal text-gray-500">Based on {rate}% APR</span>
                  </div>
                </td>
                {comparisonList.map(car => {
                  const monthly = calculateMonthlyPayment(car.price, financials.downPayment, rate, financials.term);
                  return (
                    <td key={car.id} className="p-4">
                      <span className="block font-bold text-blue-600 text-lg">
                        ${monthly.toFixed(0)}<span className="text-sm text-gray-500">/mo</span>
                      </span>
                    </td>
                  );
                })}
              </tr>
              <tr>
                <td className="p-4 font-semibold text-gray-700 bg-gray-50/50 align-top pt-6">Safety Profile</td>
                {comparisonList.map(car => (
                  <td key={car.id} className="p-4 align-top">
                    <div className="flex items-center gap-1 mb-2">
                       {Array.from({length: 5}).map((_, i) => (
                         <Star 
                           key={i} 
                           size={16} 
                           className={i < car.safetyRating ? "fill-yellow-400 text-yellow-400" : "text-gray-300"} 
                         />
                       ))}
                    </div>
                    <ul className="text-sm space-y-2">
                      <li className="flex items-center gap-2">
                        {car.safetyFeatures.aeb ? <Check size={14} className="text-green-500"/> : <X size={14} className="text-red-300"/>}
                        <span className="text-gray-600">Auto Emergency Braking</span>
                      </li>
                      <li className="flex items-center gap-2">
                        {car.safetyFeatures.lka ? <Check size={14} className="text-green-500"/> : <X size={14} className="text-red-300"/>}
                        <span className="text-gray-600">Lane Keep Assist</span>
                      </li>
                    </ul>
                  </td>
                ))}
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 font-sans text-gray-800">
      <header className="bg-white border-b border-gray-200 sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 text-blue-700">
            <Car size={28} />
            <span className="text-xl font-extrabold tracking-tight">Car Hunt <span className="text-blue-500 font-light">Agent</span></span>
          </div>
          {step > 1 && (
             <div className="text-sm font-medium text-gray-500">
                {comparisonList.length} selected for compare
             </div>
          )}
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8">
        <StepIndicator step={step} />
        {step === 1 && renderStep1()}
        {step === 2 && renderStep2()}
        {step === 3 && renderStep3()}
      </main>

      <footer className="mt-12 border-t border-gray-200 py-8 text-center text-gray-500 text-sm">
        <p>&copy; 2024 Car Hunt Agent. Estimates for demonstration only.</p>
      </footer>
    </div>
  );
}

