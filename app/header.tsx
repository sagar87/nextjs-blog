import Link from "next/link";

const Header = () => {
  return (
    <header>
      <h1>
        <Link href="/">Welcome to my Blog</Link>
      </h1>
      <p>The latest from my mind.</p>
    </header>
  );
};

export default Header;
