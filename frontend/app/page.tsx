'use client';

import { useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';

interface HRVFeatures {
  [key: string]: number;
}

interface Prediction {
  prediction: number;
  prediction_label: string;
  probabilities?: {
    no_stress: number;
    stress: number;
  };
}

interface PredictionResult {
  success: boolean;
  features: HRVFeatures;
  prediction: Prediction;
  model_used: string;
  signal_info: {
    length: number;
    sampling_rate: 700;
    duration_seconds: number;
  };
}

export default function ECGAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string>('');
  const [ecgData, setEcgData] = useState<any[]>([]);
  const [selectedModel, setSelectedModel] = useState<'random_forest' | 'svc'>('random_forest');

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError('');

      const text = await selectedFile.text();
      const lines = text.split('\n').filter(line => line.trim());
      const headers = lines[0].split(',').map(h => h.trim());

      const ecgColumnIndex = headers.findIndex(h =>
        h.toUpperCase() === 'ECG' || h.toLowerCase() === 'ecg'
      );
      const columnIndex = ecgColumnIndex !== -1 ? ecgColumnIndex : 0;

      const data = lines.slice(1, Math.min(1001, lines.length)).map((line, index) => {
        const values = line.split(',');
        return {
          index,
          ecg: parseFloat(values[columnIndex]) || 0
        };
      });

      setEcgData(data);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!file) {
      setError('Please select a CSV file');
      return;
    }

    setLoading(true);
    setError('');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', selectedModel);
    formData.append('sampling_rate', '700');

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

      const data: PredictionResult = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const prepareFeatureData = () => {
    if (!result) return [];

    return Object.entries(result.features).map(([key, value]) => ({
      feature: key.replace('HRV_', ''),
      value: value
    }));
  };

  return (
    <div className="min-h-screen bg-black text-gray-100">
      <div className="max-w-7xl mx-auto p-6 lg:p-8">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-semibold text-white mb-2">
            ECG Stress Detection
          </h1>
          <p className="text-gray-400 text-sm">
            Analyze heart rate variability and predict stress levels
          </p>
        </div>

        {/* Upload Card */}
        <div className="bg-black border border-zinc-800 rounded-lg p-6 mb-6">
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Model Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Model Selection
              </label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value as 'random_forest' | 'svc')}
                className="w-full bg-black border border-zinc-700 text-white rounded-md px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
              >
                <option value="random_forest">Random Forest</option>
                <option value="svc">Support Vector Classifier</option>
              </select>
            </div>

            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Upload ECG Data
              </label>
              <div className="border-2 border-dashed border-zinc-700 rounded-lg p-8 text-center hover:border-zinc-600 transition-colors">
                <input
                  type="file"
                  accept=".csv"
                  onChange={handleFileChange}
                  className="hidden"
                  id="file-upload"
                />
                <label htmlFor="file-upload" className="cursor-pointer block">
                  <svg
                    className="w-10 h-10 text-zinc-500 mx-auto mb-3"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                    />
                  </svg>
                  <span className="text-white font-medium">
                    {file ? file.name : 'Click to upload CSV file'}
                  </span>
                  <p className="text-sm text-gray-500 mt-1">
                    CSV file containing ECG signal data
                  </p>
                </label>
              </div>
            </div>

            {error && (
              <div className="bg-red-950/30 border border-red-900/50 p-4 rounded-lg">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            <button
              type="submit"
              disabled={!file || loading}
              className="w-full bg-cyan-500 text-black py-3 rounded-md font-bold hover:bg-cyan-400 disabled:bg-zinc-900 disabled:text-zinc-700 disabled:cursor-not-allowed transition-colors shadow-[0_0_15px_rgba(6,182,212,0.5)]"
            >
              {loading ? 'Analyzing...' : 'Analyze ECG Data'}
            </button>
          </form>
        </div>

        {/* ECG Signal Preview */}
        {ecgData.length > 0 && (
          <div className="bg-black border border-zinc-800 rounded-lg p-6 mb-6">
            <h2 className="text-lg font-medium text-white mb-4">
              ECG Signal Preview
            </h2>
            <div className="bg-black rounded-lg p-4">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={ecgData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                  <XAxis
                    dataKey="index"
                    stroke="#52525b"
                    tick={{ fill: '#71717a', fontSize: 12 }}
                  />
                  <YAxis
                    stroke="#52525b"
                    tick={{ fill: '#71717a', fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#18181b',
                      border: '1px solid #27272a',
                      borderRadius: '0.375rem'
                    }}
                    labelStyle={{ color: '#e4e4e7' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="ecg"
                    stroke="#22d3ee"
                    dot={false}
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Results */}
        {result && (
          <>
            {/* Prediction Result */}
            <div className="bg-black border border-zinc-800 rounded-lg p-6 mb-6">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-lg font-medium text-white">
                  Prediction Result
                </h2>
                <span className="text-xs text-gray-500 bg-zinc-800 px-3 py-1 rounded-full">
                  {result.model_used === 'random_forest' ? 'Random Forest' : 'Support Vector Classifier'}
                </span>
              </div>

              <div className="text-center py-8">
                <div className={`text-5xl font-semibold mb-4 ${result.prediction.prediction === 1 ? 'text-red-500' : 'text-green-500'
                  }`}>
                  {result.prediction.prediction_label}
                </div>

                {result.prediction.probabilities && (
                  <div className="max-w-md mx-auto space-y-4 mt-8">
                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-gray-400">No Stress</span>
                        <span className="text-white font-medium">
                          {(result.prediction.probabilities.no_stress * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-zinc-800 rounded-full h-2">
                        <div
                          className="bg-green-500 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${result.prediction.probabilities.no_stress * 100}%` }}
                        />
                      </div>
                    </div>

                    <div>
                      <div className="flex justify-between text-sm mb-2">
                        <span className="text-gray-400">Stress</span>
                        <span className="text-white font-medium">
                          {(result.prediction.probabilities.stress * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-zinc-800 rounded-full h-2">
                        <div
                          className="bg-red-500 h-2 rounded-full transition-all duration-500"
                          style={{ width: `${result.prediction.probabilities.stress * 100}%` }}
                        />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Signal Information */}
            <div className="bg-black border border-zinc-800 rounded-lg p-6 mb-6">
              <h2 className="text-lg font-medium text-white mb-4">
                Signal Information
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-zinc-900/30 border border-zinc-800 p-4 rounded-lg">
                  <p className="text-xs text-gray-500 mb-1">Signal Length</p>
                  <p className="text-2xl font-semibold text-white">
                    {result.signal_info.length.toLocaleString()}
                  </p>
                  <p className="text-xs text-gray-600 mt-1">samples</p>
                </div>
                <div className="bg-zinc-900/30 border border-zinc-800 p-4 rounded-lg">
                  <p className="text-xs text-gray-500 mb-1">Sampling Rate</p>
                  <p className="text-2xl font-semibold text-white">
                    {result.signal_info.sampling_rate}
                  </p>
                  <p className="text-xs text-gray-600 mt-1">Hz</p>
                </div>
                <div className="bg-zinc-900/30 border border-zinc-800 p-4 rounded-lg">
                  <p className="text-xs text-gray-500 mb-1">Duration</p>
                  <p className="text-2xl font-semibold text-white">
                    {result.signal_info.duration_seconds.toFixed(2)}
                  </p>
                  <p className="text-xs text-gray-600 mt-1">seconds</p>
                </div>
              </div>
            </div>

            {/* HRV Features */}
            <div className="bg-black border border-zinc-800 rounded-lg p-6">
              <h2 className="text-lg font-medium text-white mb-4">
                Extracted HRV Features
              </h2>
              <div className="bg-black rounded-lg p-4 mb-6">
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={prepareFeatureData()}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                    <XAxis
                      dataKey="feature"
                      angle={-45}
                      textAnchor="end"
                      height={100}
                      interval={0}
                      tick={{ fontSize: 10, fill: '#71717a' }}
                      stroke="#52525b"
                    />
                    <YAxis
                      stroke="#52525b"
                      tick={{ fill: '#71717a', fontSize: 12 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#18181b',
                        border: '1px solid #27272a',
                        borderRadius: '0.375rem'
                      }}
                      labelStyle={{ color: '#e4e4e7' }}
                    />
                    <Bar dataKey="value" fill="#818cf8" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                {Object.entries(result.features).map(([key, value]) => (
                  <div key={key} className="bg-zinc-900/30 border border-zinc-800 p-3 rounded-lg">
                    <p className="text-xs text-gray-500 mb-1">{key}</p>
                    <p className="text-sm font-medium text-white">
                      {typeof value === 'number' ? value.toFixed(3) : value}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}