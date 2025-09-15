'use client';

import { useEffect, useState } from 'react';

interface StatsData {
  totalQuestions: number;
  modelsTested: number;
  questionsAnswered: number;
  bestAccuracy: number;
}

interface StatItem {
  name: string;
  stat: string;
  loading?: boolean;
  error?: boolean;
}

export default function HomeStats() {
    const [stats, setStats] = useState<StatItem[]>([
        { name: 'Total Questions', stat: '0', loading: true },
        { name: 'Models Tested', stat: '0', loading: true },
        { name: 'Questions Answered', stat: '0', loading: true },
        { name: 'Best Accuracy', stat: '0%', loading: true },
    ]);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const response = await fetch('/api/stats/summary');
                
                if (!response.ok) {
                    throw new Error(`Failed to fetch stats: ${response.status}`);
                }
                
                const data: StatsData = await response.json();
                
                // Format the data for display
                const formattedStats: StatItem[] = [
                    {
                        name: 'Total Questions',
                        stat: data.totalQuestions.toLocaleString(),
                        loading: false
                    },
                    {
                        name: 'Models Tested',
                        stat: data.modelsTested.toString(),
                        loading: false
                    },
                    {
                        name: 'Questions Answered',
                        stat: data.questionsAnswered.toLocaleString(),
                        loading: false
                    },
                    {
                        name: 'Best Accuracy',
                        stat: `${data.bestAccuracy.toFixed(2)}%`,
                        loading: false
                    },
                ];
                
                setStats(formattedStats);
                setError(null);
                
            } catch (err) {
                console.error('Error fetching stats:', err);
                setError(err instanceof Error ? err.message : 'Failed to load statistics');
                
                // Set error state for all stats
                setStats(prev => prev.map(stat => ({
                    ...stat,
                    loading: false,
                    error: true
                })));
            }
        };

        fetchStats();
    }, []);

    return (
        <div className="px-8 my-4">
            {error && (
                <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded-lg dark:bg-red-900/20 dark:border-red-800 dark:text-red-400">
                    <p className="text-sm">{error}</p>
                </div>
            )}
            
            <dl className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-4">
                {stats.map((item) => (
                    <div
                        key={item.name}
                        className="overflow-hidden rounded-lg bg-white px-4 py-5 shadow sm:p-6 dark:bg-gray-800/75 dark:ring-1 dark:ring-inset dark:ring-white/10"
                    >
                        <dt className="truncate text-sm font-medium text-gray-500 dark:text-gray-400">
                            {item.name}
                        </dt>
                        <dd className="mt-1 text-3xl font-semibold tracking-tight text-gray-900 dark:text-white">
                            {item.loading ? (
                                <div className="flex items-center">
                                    <div className="animate-pulse bg-gray-300 dark:bg-gray-600 rounded h-8 w-20"></div>
                                </div>
                            ) : item.error ? (
                                <span className="text-red-500 dark:text-red-400 text-base">Error</span>
                            ) : (
                                item.stat
                            )}
                        </dd>
                    </div>
                ))}
            </dl>
        </div>
    )
}
