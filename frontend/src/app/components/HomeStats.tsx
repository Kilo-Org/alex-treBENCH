const stats = [
    { name: 'Total Questions', stat: '216,078' },
    { name: 'Models Tested', stat: '5' },
    { name: 'Questions Answered', stat: '5,452' },
    { name: 'Best Accuracy', stat: '24.57%' },
]

export default function HomeStats() {
    return (
        <div className="px-8 my-4">
            {/* <h3 className="text-base font-semibold text-gray-900 dark:text-white">Last 30 days</h3> */}
            <dl className="mt-5 grid grid-cols-1 gap-5 sm:grid-cols-4">
                {stats.map((item) => (
                    <div
                        key={item.name}
                        className="overflow-hidden rounded-lg bg-white px-4 py-5 shadow sm:p-6 dark:bg-gray-800/75 dark:ring-1 dark:ring-inset dark:ring-white/10"
                    >
                        <dt className="truncate text-sm font-medium text-gray-500 dark:text-gray-400">{item.name}</dt>
                        <dd className="mt-1 text-3xl font-semibold tracking-tight text-gray-900 dark:text-white">{item.stat}</dd>
                    </div>
                ))}
            </dl>
        </div>
    )
}
