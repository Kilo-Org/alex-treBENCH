'use client'

import { useState } from 'react'
import NavSidebar from "./NavSidebar"
import NavTopbar from "./NavTopbar"

export default function LayoutClient({
  children,
}: {
  children: React.ReactNode;
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div>
      <NavSidebar sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />

      <div className="lg:pl-72">
        <NavTopbar setSidebarOpen={setSidebarOpen} />

        <main className="py-10">
          <div className="px-4 sm:px-6 lg:px-8">
            {children}
          </div>
        </main>
      </div>
      {children}
    </div>
  );
}