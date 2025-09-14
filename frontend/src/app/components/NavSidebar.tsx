'use client'

import {
    Dialog,
    DialogBackdrop,
    DialogPanel,
    TransitionChild,
} from '@headlessui/react'
import {
    ChartBarIcon,
    ChartPieIcon,
    HomeIcon,
    QuestionMarkCircleIcon,
    RectangleGroupIcon,
    TrophyIcon,
    XMarkIcon,
} from '@heroicons/react/24/outline'

const navigation = [
    { name: 'Leaderboard', href: '/', icon: TrophyIcon, current: true },
    { name: 'Jeopardy Board', href: '/board', icon: RectangleGroupIcon, current: false },
    { name: 'Results', href: '/results', icon: ChartBarIcon, current: false },
    { name: 'Analytics', href: '/analytics', icon: ChartPieIcon, current: false },
    { name: 'Questions', href: '/questions', icon: QuestionMarkCircleIcon, current: false },
]

function classNames(...classes: string[]) {
    return classes.filter(Boolean).join(' ')
}

interface NavSidebarProps {
    sidebarOpen: boolean;
    setSidebarOpen: (open: boolean) => void;
}

export default function NavSidebar({ sidebarOpen, setSidebarOpen }: NavSidebarProps) {
    return (
        <>
            <div>
                {/* Mobile Sidebar */}
                <Dialog open={sidebarOpen} onClose={setSidebarOpen} className="relative z-50 lg:hidden">
                    <DialogBackdrop
                        transition
                        className="fixed inset-0 bg-gray-900/80 transition-opacity duration-300 ease-linear data-[closed]:opacity-0"
                    />

                    <div className="fixed inset-0 flex">
                        <DialogPanel
                            transition
                            className="relative mr-16 flex w-full max-w-xs flex-1 transform transition duration-300 ease-in-out data-[closed]:-translate-x-full"
                        >
                            <TransitionChild>
                                <div className="absolute left-full top-0 flex w-16 justify-center pt-5 duration-300 ease-in-out data-[closed]:opacity-0">
                                    <button type="button" onClick={() => setSidebarOpen(false)} className="-m-2.5 p-2.5">
                                        <span className="sr-only">Close sidebar</span>
                                        <XMarkIcon aria-hidden="true" className="size-6 text-white" />
                                    </button>
                                </div>
                            </TransitionChild>

                            {/* Sidebar component, swap this element with another sidebar if you like */}
                            <div className="relative flex grow flex-col gap-y-5 overflow-y-auto bg-brand-primary px-6 pb-4 dark:bg-brand-primary-dark dark:ring-1 dark:ring-white/10">
                                <div className="flex h-16 shrink-0 items-center">
                                    <img
                                        alt="Your Company"
                                        src="https://tailwindcss.com/plus-assets/img/logos/mark.svg?color=white"
                                        className="h-8 w-auto"
                                    />
                                </div>
                                <nav className="flex flex-1 flex-col">
                                    <ul role="list" className="flex flex-1 flex-col gap-y-7">
                                        <li>
                                            <ul role="list" className="-mx-2 space-y-1">
                                                {navigation.map((item) => (
                                                    <li key={item.name}>
                                                        <a
                                                            href={item.href}
                                                            className={classNames(
                                                                item.current
                                                                    ? 'bg-brand-primary-dark text-white dark:bg-brand-primary/40'
                                                                    : 'text-blue-200 hover:bg-brand-primary-dark hover:text-white dark:text-blue-100 dark:hover:bg-brand-primary/40',
                                                                'group flex gap-x-3 rounded-md p-2 text-sm/6 font-semibold',
                                                            )}
                                                        >
                                                            <item.icon
                                                                aria-hidden="true"
                                                                className={classNames(
                                                                    item.current
                                                                        ? 'text-white'
                                                                        : 'text-blue-200 group-hover:text-white dark:text-blue-100',
                                                                    'size-6 shrink-0',
                                                                )}
                                                            />
                                                            {item.name}
                                                        </a>
                                                    </li>
                                                ))}
                                            </ul>
                                        </li>
                                    </ul>
                                </nav>
                            </div>
                        </DialogPanel>
                    </div>
                </Dialog>

                {/* Desktop Sidebar */}
                <div className="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-72 lg:flex-col">
                    {/* Sidebar component, swap this element with another sidebar if you like */}
                    <div className="relative flex grow flex-col gap-y-5 overflow-y-auto bg-brand-primary px-6 pb-4 dark:bg-brand-primary-dark dark:after:pointer-events-none dark:after:absolute dark:after:inset-y-0 dark:after:right-0 dark:after:w-px dark:after:bg-white/10">
                        <div className="flex h-16 shrink-0 items-center">
                            <img
                                alt="Your Company"
                                src="https://tailwindcss.com/plus-assets/img/logos/mark.svg?color=white"
                                className="h-8 w-auto"
                            />
                        </div>
                        <nav className="flex flex-1 flex-col">
                            <ul role="list" className="flex flex-1 flex-col gap-y-7">
                                <li>
                                    <ul role="list" className="-mx-2 space-y-1">
                                        {navigation.map((item) => (
                                            <li key={item.name}>
                                                <a
                                                    href={item.href}
                                                    className={classNames(
                                                        item.current
                                                            ? 'bg-brand-primary-dark text-white dark:bg-brand-primary/40'
                                                            : 'text-blue-200 hover:bg-brand-primary-dark hover:text-white dark:text-blue-100 dark:hover:bg-brand-primary/40',
                                                        'group flex gap-x-3 rounded-md p-2 text-sm/6 font-semibold',
                                                    )}
                                                >
                                                    <item.icon
                                                        aria-hidden="true"
                                                        className={classNames(
                                                            item.current
                                                                ? 'text-white'
                                                                : 'text-blue-200 group-hover:text-white dark:text-blue-100',
                                                            'size-6 shrink-0',
                                                        )}
                                                    />
                                                    {item.name}
                                                </a>
                                            </li>
                                        ))}
                                    </ul>
                                </li>
                            </ul>
                        </nav>
                    </div>
                </div>
            </div>
        </>
    )
}