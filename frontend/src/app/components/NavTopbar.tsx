import { Bars3Icon } from '@heroicons/react/24/outline'

interface NavTopbarProps {
    setSidebarOpen: (open: boolean) => void;
}

export default function NavTopbar({ setSidebarOpen }: NavTopbarProps) {
    return (
        <div className="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-x-4 border-b border-gray-200 bg-white px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:px-8 dark:border-white/10 dark:bg-gray-900 dark:shadow-none">
            <button
                type="button"
                onClick={() => setSidebarOpen(true)}
                className="-m-2.5 p-2.5 text-gray-700 hover:text-gray-900 lg:hidden dark:text-gray-400 dark:hover:text-white"
            >
                <span className="sr-only">Open sidebar</span>
                <Bars3Icon aria-hidden="true" className="size-6" />
            </button>

            {/* Separator */}
            <div aria-hidden="true" className="h-6 w-px bg-gray-900/10 lg:hidden dark:bg-white/10" />

            <div className="flex justify-between w-full gap-x-4 items-center lg:gap-x-6">
                <div className="hidden md:block flex-1 text-amber-100">
                    LLM Jeopardy Championship
                </div>
                <h1 className="flex-1 text-xl md:text-4xl font-bold text-brand-primary dark:text-white" style={{ fontFamily: 'Gyparody, sans-serif' }}>
                    alex tre-BENCH
                </h1>
                <div className="flex items-center gap-x-4 lg:gap-x-6">
                    {/* Placeholder to keep title centered */}
                    <div className="w-20 lg:w-32"></div>
                </div>
            </div>
        </div>
    )
}