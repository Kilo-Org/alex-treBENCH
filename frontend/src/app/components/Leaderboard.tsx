// Mock data - will be replaced with real data later
const leaderboardData = [
    {
        id: 1,
        name: "GPT-4",
        rank: 1,
        score: 785,
        maxScore: 1000,
        accuracy: 78.5,
        avgTime: 1.24,
        cost: 4.67,
        efficiency: 168,
        trend: "up"
    },
    {
        id: 2,
        name: "Claude-3.5",
        rank: 2,
        score: 718,
        maxScore: 1000,
        accuracy: 71.8,
        avgTime: 0.98,
        cost: 3.21,
        efficiency: 224,
        trend: "up"
    },
    {
        id: 3,
        name: "GPT-3.5",
        rank: 3,
        score: 645,
        maxScore: 1000,
        accuracy: 64.5,
        avgTime: 0.65,
        cost: 1.85,
        efficiency: 349,
        trend: "up"
    },
    {
        id: 4,
        name: "Gemini Pro",
        rank: 4,
        score: 612,
        maxScore: 1000,
        accuracy: 61.2,
        avgTime: 1.12,
        cost: 2.94,
        efficiency: 208,
        trend: "down"
    },
    {
        id: 5,
        name: "LLaMA-2",
        rank: 5,
        score: 578,
        maxScore: 1000,
        accuracy: 57.8,
        avgTime: 0.89,
        cost: 1.23,
        efficiency: 470,
        trend: "up"
    }
];

const sortOptions = [
    { name: "Rank", key: "rank" },
    { name: "Score", key: "score" },
    { name: "Accuracy", key: "accuracy" },
    { name: "Speed", key: "avgTime" },
    { name: "Cost", key: "cost" },
    { name: "Efficiency", key: "efficiency" }
];

export default function Leaderboard() {
    const getMedalIcon = (rank: number) => {
        if (rank === 1) return "🥇";
        if (rank === 2) return "🥈";
        if (rank === 3) return "🥉";
        return rank.toString();
    };

    const getTrendIcon = (trend: string) => {
        return trend === "up" ? "📈" : "📉";
    };

    return (
        <div className="bg-slate-800 rounded-2xl p-6 border border-slate-700 my-12 max-w-5xl mx-auto">
            <div className="flex items-center justify-between mb-6">
                <div className="flex items-center gap-3">
                    <span className="text-2xl">🏆</span>
                    <h2 className="text-xl font-semibold text-white">Championship Standings</h2>
                </div>
                <button className="bg-yellow-600 hover:bg-yellow-700 text-black font-medium px-4 py-2 rounded-lg transition-colors">
                    Live Results
                </button>
            </div>

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
        </div>
    );
}