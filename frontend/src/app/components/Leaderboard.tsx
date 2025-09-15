'use client';

import { useEffect, useState } from 'react';

interface LeaderboardEntry {
    id: number;
    name: string;
    rank: number;
    score: number;
    maxScore: number;
    accuracy: number;
    avgTime: number;
    cost: number;
    efficiency: number;
    trend: 'up' | 'down' | 'stable';
}

const sortOptions = [
    { name: "Rank", key: "rank" },
    { name: "Score", key: "score" },
    { name: "Accuracy", key: "accuracy" },
    { name: "Speed", key: "avgTime" },
    { name: "Cost", key: "cost" },
    { name: "Efficiency", key: "efficiency" }
];

export default function Leaderboard() {
    const [leaderboardData, setLeaderboardData] = useState<LeaderboardEntry[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [refreshing, setRefreshing] = useState(false);

    const fetchLeaderboard = async () => {
        try {
            setError(null);
            const response = await fetch('/api/leaderboard');
            
            if (!response.ok) {
                throw new Error(`Failed to fetch leaderboard: ${response.status}`);
            }
            
            const data: LeaderboardEntry[] = await response.json();
            setLeaderboardData(data);
            
        } catch (err) {
            console.error('Error fetching leaderboard:', err);
            setError(err instanceof Error ? err.message : 'Failed to load leaderboard data');
        } finally {
            setLoading(false);
            setRefreshing(false);
        }
    };

    useEffect(() => {
        fetchLeaderboard();
    }, []);

    const handleRefresh = () => {
        setRefreshing(true);
        fetchLeaderboard();
    };

    const getMedalIcon = (rank: number) => {
        if (rank === 1) return "🥇";
        if (rank === 2) return "🥈";
        if (rank === 3) return "🥉";
        return rank.toString();
    };

    const getTrendIcon = (trend: string) => {
        if (trend === "up") return "📈";
        if (trend === "down") return "📉";
        return "➖"; // stable
    };

    // Loading skeleton component
    const LoadingSkeleton = () => (
        <div className="space-y-4">
            {[1, 2, 3, 4, 5].map((i) => (
                <div key={i} className="bg-slate-700 rounded-xl p-4 border border-slate-600">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-4">
                            <div className="animate-pulse bg-slate-600 rounded w-10 h-10"></div>
                            <div className="flex flex-col space-y-2">
                                <div className="animate-pulse bg-slate-600 rounded h-5 w-24"></div>
                                <div className="animate-pulse bg-slate-600 rounded h-4 w-16"></div>
                            </div>
                        </div>
                        <div className="flex items-center gap-6">
                            {[1, 2, 3, 4, 5].map((j) => (
                                <div key={j} className="text-center min-w-[80px]">
                                    <div className="animate-pulse bg-slate-600 rounded h-6 w-12 mb-1"></div>
                                    <div className="animate-pulse bg-slate-600 rounded h-3 w-16"></div>
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            ))}
        </div>
    );

    return (
        <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 my-12 max-w-5xl mx-auto">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    <span className="text-2xl">🏆</span>
                    <h2 className="text-xl font-semibold text-white">Championship Standings</h2>
                </div>
                <button
                    onClick={handleRefresh}
                    disabled={refreshing}
                    className="bg-yellow-600 hover:bg-yellow-700 disabled:bg-yellow-800 text-black font-medium px-4 py-2 rounded-lg transition-colors flex items-center gap-2"
                >
                    {refreshing ? (
                        <>
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-black border-t-transparent"></div>
                            Refreshing...
                        </>
                    ) : (
                        'Live Results'
                    )}
                </button>
            </div>

            {error && (
                <div className="mb-6 p-4 bg-red-900/20 border border-red-800 text-red-400 rounded-lg">
                    <p>{error}</p>
                    <button
                        onClick={handleRefresh}
                        className="mt-2 text-sm text-red-300 hover:text-red-200 underline"
                    >
                        Try again
                    </button>
                </div>
            )}

            <div className="flex gap-3 mb-6 flex-wrap">
                {sortOptions.map((option) => (
                    <button
                        key={option.key}
                        className="bg-slate-700 hover:bg-slate-600 text-white px-4 py-2 rounded-lg border border-slate-600 transition-colors flex items-center gap-1"
                    >
                        {option.name}
                        <span className="text-slate-400">↕</span>
                    </button>
                ))}
            </div>

            {loading ? (
                <LoadingSkeleton />
            ) : leaderboardData.length === 0 ? (
                <div className="text-center py-12 text-slate-400">
                    <span className="text-4xl mb-4 block">🏁</span>
                    <p>No benchmark data available yet.</p>
                    <p className="text-sm mt-2">Run some benchmarks to see the leaderboard!</p>
                </div>
            ) : (
                <div className="space-y-4">
                    {leaderboardData.map((entry) => (
                        <div
                            key={entry.id}
                            className="bg-slate-700 rounded-xl p-4 border border-slate-600 hover:border-slate-500 transition-colors"
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-4">
                                    {/* Medal/Rank */}
                                    <div className="text-2xl min-w-[40px] text-center">
                                        {entry.rank <= 3 ? getMedalIcon(entry.rank) : (
                                            <span className="text-white font-bold text-lg">{entry.rank}</span>
                                        )}
                                    </div>

                                    {/* Model Name and Details */}
                                    <div className="flex flex-col">
                                        <div className="flex items-center gap-3">
                                            <h3 className="text-white font-semibold text-lg">{entry.name}</h3>
                                            <span className="text-lg">{getTrendIcon(entry.trend)}</span>
                                        </div>
                                        <div className="flex items-center gap-2 mt-1">
                                            <span className="bg-slate-600 text-slate-300 px-2 py-1 rounded text-sm">
                                                {entry.score}/{entry.maxScore}
                                            </span>
                                        </div>
                                    </div>
                                </div>

                                {/* Right side - Metrics */}
                                <div className="flex items-center gap-6">
                                    {/* Score */}
                                    <div className="text-center">
                                        <div className="text-3xl font-bold text-yellow-400">{entry.score}</div>
                                        <div className="text-slate-400 text-sm">Score</div>
                                    </div>

                                    {/* Accuracy */}
                                    <div className="text-center min-w-[80px]">
                                        <div className="flex items-center justify-center gap-1">
                                            <span className="text-slate-400">🎯</span>
                                            <span className="text-white font-semibold">{entry.accuracy}%</span>
                                        </div>
                                        <div className="text-slate-400 text-sm">Accuracy</div>
                                    </div>

                                    {/* Avg Time */}
                                    <div className="text-center min-w-[80px]">
                                        <div className="flex items-center justify-center gap-1">
                                            <span className="text-slate-400">⏱</span>
                                            <span className="text-white font-semibold">{entry.avgTime}s</span>
                                        </div>
                                        <div className="text-slate-400 text-sm">Avg Time</div>
                                    </div>

                                    {/* Cost */}
                                    <div className="text-center min-w-[80px]">
                                        <div className="flex items-center justify-center gap-1">
                                            <span className="text-slate-400">💲</span>
                                            <span className="text-white font-semibold">${entry.cost}</span>
                                        </div>
                                        <div className="text-slate-400 text-sm">Cost</div>
                                    </div>

                                    {/* Efficiency */}
                                    <div className="text-center min-w-[80px]">
                                        <div className="flex items-center justify-center gap-1">
                                            <span className="text-slate-400">⚡</span>
                                            <span className="text-white font-semibold">{entry.efficiency}</span>
                                        </div>
                                        <div className="text-slate-400 text-sm">Efficiency</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}