
import { ClockIcon, CodeBracketIcon } from '@heroicons/react/24/outline'

export default function Hero() {
    return (
        <div className="relative">
            <div className="relative z-10 flex flex-col items-center justify-center px-6 py-12">
                {/* Top notification banner */}
                <div className="mb-12 flex items-center space-x-4">
                    <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="text-gray-300 text-sm">GPT-5 and GPT OSS now available in LLM Stats!</span>
                    </div>
                    <div className="flex space-x-2">
                        <button className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white text-sm rounded-md flex items-center space-x-1 transition-colors">
                            <span>🎮</span>
                            <span>Try in playground</span>
                        </button>
                        <button className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-md flex items-center space-x-1 transition-colors">
                            <CodeBracketIcon className="w-4 h-4" />
                            <span>Use in API</span>
                        </button>
                    </div>
                </div>

                {/* Main heading */}
                <h1 className="text-6xl md:text-7xl font-bold text-white text-center mb-8">
                    LLM Leaderboardx
                </h1>

                {/* Subtitle */}
                <p className="text-xl text-gray-400 text-center max-w-4xl mb-16">
                    Analyze and compare <span className="text-blue-400">API models</span> across benchmarks, pricing, and capabilities.
                </p>

                {/* Bottom section */}
                <div className="flex items-center space-x-6">
                    <div className="flex items-center space-x-2 text-gray-400">
                        <ClockIcon className="w-4 h-4" />
                        <span className="text-sm">Updated daily</span>
                    </div>
                    <button className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg flex items-center space-x-2 transition-colors">
                        <span>💬</span>
                        <span>Join our Discord</span>
                    </button>
                </div>
            </div>
        </div>
    )
}