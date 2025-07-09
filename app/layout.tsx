import Head from "./head";
import Header from "./header";
import Footer from "./footer";
import "./globals.css";

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <Head />
      <body>
        <div className="main">
          <div className="gradient"></div>
        </div>
        <div className="mx-auto max-w-3xl px-6">
          <Header />
          {children}
          <Footer />
        </div>
      </body>
    </html>
  );
}
