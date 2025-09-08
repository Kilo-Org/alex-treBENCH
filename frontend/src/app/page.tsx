'use client'

import Hero from "./components/Hero";
import HomeStats from "./components/HomeStats";
import Leaderboard from "./components/Leaderboard";

export default function Home() {
  return (
    <>
      <div>
        <Hero />

        <HomeStats />

        <Leaderboard />
      </div>
    </>
  )
}
