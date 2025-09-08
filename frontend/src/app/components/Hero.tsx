
import { TrophyIcon } from "@heroicons/react/20/solid"
import { ClockIcon, CodeBracketIcon } from '@heroicons/react/24/outline'

export default function Hero() {
    return (
        <div className="relative">
            <div className="relative z-10 flex flex-col items-center justify-center px-6 md:py-6">
                <div className="mb-12 flex items-center space-x-4">
                    <span className="inline-flex items-center gap-x-1.5 rounded-md px-2 py-1 text-sm font-medium text-gray-900 ring-1 ring-inset ring-gray-200 dark:text-white dark:ring-white/10">
                        <svg viewBox="0 0 6 6" aria-hidden="true" className="size-1.5 fill-green-500 dark:fill-green-400">
                            <circle r={3} cx={3} cy={3} />
                        </svg>
                        LLM performance tracking across 200k+ Jeopardy questions
                    </span>
                </div>

                <h1 className="flex items-center gap-2 text-xl md:text-5xl font-bold text-white text-center mb-8">
                    {/* <TrophyIcon className="size-8 shrink-0 text-brand-primary border border-amber-50 bg-amber-50" /> */}
                    LLM Jeopardy Championship Leaderboard
                </h1>

                <p className="text-xl text-gray-400 text-center max-w-4xl mb-2">
                    "What is... the smartest language model?"
                </p>
            </div>
        </div>
    )
}