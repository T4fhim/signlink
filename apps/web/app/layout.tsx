import type { ReactNode } from "react";

export const metadata = {
  title: "SignLnk",
  description: "Free, open-source, on-device bridge between signing and non-signing users.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
