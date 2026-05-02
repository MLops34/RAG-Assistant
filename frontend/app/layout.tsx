import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "RAG Assistant",
  description: "Next.js UI for syllabus parsing and study scheduling",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
