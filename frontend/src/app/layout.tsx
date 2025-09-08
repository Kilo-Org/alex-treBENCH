import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import LayoutClient from "./components/LayoutClient";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "alex treBENCH",
  description: "This project uses historical Jeopardy questions as a benchmark for AI Large Language Models, allowing for the download, sampling, and evaluation of models' question-answering capabilities via OpenRouter.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="h-full bg-white dark:bg-gray-900">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased h-full`}
      >
        <LayoutClient>{children}</LayoutClient>
      </body>
    </html>
  );
}
