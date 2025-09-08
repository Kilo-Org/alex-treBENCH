import {
    Menu,
    MenuButton,
    MenuItem,
    MenuItems,
} from '@headlessui/react'
import {
    Bars3Icon,
    BellIcon,
} from '@heroicons/react/24/outline'
import { ChevronDownIcon, MagnifyingGlassIcon } from '@heroicons/react/20/solid'

const userNavigation = [
    { name: 'Your profile', href: '#' },
    { name: 'Sign out', href: '#' },
]

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

            <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
                <form action="#" method="GET" className="grid flex-1 grid-cols-1">
                    <input
                        name="search"
                        placeholder="Search"
                        aria-label="Search"
                        className="col-start-1 row-start-1 block size-full bg-white pl-8 text-base text-gray-900 outline-none placeholder:text-gray-400 sm:text-sm/6 dark:bg-gray-900 dark:text-white dark:placeholder:text-gray-500"
                    />
                    <MagnifyingGlassIcon
                        aria-hidden="true"
                        className="pointer-events-none col-start-1 row-start-1 size-5 self-center text-gray-400"
                    />
                </form>
                <div className="flex items-center gap-x-4 lg:gap-x-6">
                    <button
                        type="button"
                        className="-m-2.5 p-2.5 text-gray-400 hover:text-gray-500 dark:text-gray-400 dark:hover:text-white"
                    >
                        <span className="sr-only">View notifications</span>
                        <BellIcon aria-hidden="true" className="size-6" />
                    </button>

                    {/* Separator */}
                    <div
                        aria-hidden="true"
                        className="hidden lg:block lg:h-6 lg:w-px lg:bg-gray-900/10 dark:lg:bg-gray-100/10"
                    />

                    {/* Profile dropdown */}
                    <Menu as="div" className="relative">
                        <MenuButton className="relative flex items-center">
                            <span className="absolute -inset-1.5" />
                            <span className="sr-only">Open user menu</span>
                            <img
                                alt=""
                                src="https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=facearea&facepad=2&w=256&h=256&q=80"
                                className="size-8 rounded-full bg-gray-50 outline-1 -outline-offset-1 outline-black/5 dark:bg-gray-800 dark:outline-white/10"
                            />
                            <span className="hidden lg:flex lg:items-center">
                                <span aria-hidden="true" className="ml-4 text-sm/6 font-semibold text-gray-900 dark:text-white">
                                    Tom Cook
                                </span>
                                <ChevronDownIcon aria-hidden="true" className="ml-2 size-5 text-gray-400 dark:text-gray-500" />
                            </span>
                        </MenuButton>
                        <MenuItems
                            transition
                            className="absolute right-0 z-10 mt-2.5 w-32 origin-top-right rounded-md bg-white py-2 shadow-lg outline-1 outline-gray-900/5 transition data-[closed]:scale-95 data-[closed]:transform data-[closed]:opacity-0 data-[enter]:duration-100 data-[leave]:duration-75 data-[enter]:ease-out data-[leave]:ease-in dark:bg-gray-800 dark:shadow-none dark:-outline-offset-1 dark:outline-white/10"
                        >
                            {userNavigation.map((item) => (
                                <MenuItem key={item.name}>
                                    <a
                                        href={item.href}
                                        className="block px-3 py-1 text-sm/6 text-gray-900 data-[focus]:bg-gray-50 data-[focus]:outline-none dark:text-white dark:data-[focus]:bg-white/5"
                                    >
                                        {item.name}
                                    </a>
                                </MenuItem>
                            ))}
                        </MenuItems>
                    </Menu>
                </div>
            </div>
        </div>
    )
}