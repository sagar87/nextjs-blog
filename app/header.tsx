import Link from "next/link";
import Image from "next/image";

const Header = () => {
  return (
    <header>
      <div className="text-center bg-slate-700 p-8 my-6 rounded-md">
        <Image
          src="/logo.png"
          width={40}
          height={40}
          alt="Logo"
          className="mx-auto"
        />
        <h1 className="text-3xl text-white py-2">
          <Link href="/">Harald Vöhringer</Link>
        </h1>
        <p className="text-slate-300">The latest from my mind.</p>
      </div>
    </header>
  );
};

export default Header;
